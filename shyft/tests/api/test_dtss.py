import numpy as np
from numpy.testing import assert_array_almost_equal
import unittest
import re
import tempfile
from shyft.api import deltahours
from shyft.api import Calendar
from shyft.api import TimeAxis
from shyft.api import TimeSeries
from shyft.api import TsVector
from shyft.api import point_interpretation_policy as point_fx
from shyft.api import DtsServer
from shyft.api import DtsClient
from shyft.api import IntVector
from shyft.api import UtcPeriod
from shyft.api import StringVector
from shyft.api import TsInfo
from shyft.api import TsInfoVector
from shyft.api import utctime_now


class DtssTestCase(unittest.TestCase):
    """Verify and illustrate dtts, distributed ts service

     """

    def __init__(self, *args, **kwargs):
        super(DtssTestCase, self).__init__(*args, **kwargs)
        self.callback_count = 0
        self.find_count = 0
        self.ts_infos = TsInfoVector()
        self.rd_throws = False
        utc = Calendar()
        t_now = utctime_now()
        self.stored_tsv = list()

        for i in range(30):
            self.ts_infos.append(
                TsInfo(
                    name='netcdf://dummy.nc/ts{0}'.format(i),
                    point_fx=point_fx.POINT_AVERAGE_VALUE,
                    delta_t=deltahours(1),
                    olson_tz_id='',
                    data_period=UtcPeriod(utc.time(2017, 1, 1), utc.time(2018, 1, 1)),
                    created=t_now,
                    modified=t_now
                )
            )

    def dtss_read_callback(self, ts_ids: StringVector, read_period: UtcPeriod) -> TsVector:
        self.callback_count += 1
        r = TsVector()
        ta = TimeAxis(read_period.start, deltahours(1), read_period.timespan() // deltahours(1))
        if self.rd_throws:
            self.rd_throws = False
            raise RuntimeError("read-ts-problem")
        for ts_id in ts_ids:
            r.append(TimeSeries(ta, fill_value=1.0, point_fx=point_fx.POINT_AVERAGE_VALUE))
        return r

    def dtss_find_callback(self, search_expression: str) -> TsInfoVector:
        self.find_count += 1
        r = TsInfoVector()
        if self.rd_throws:
            self.rd_throws = False  # we hope py do the right thing, release r here.
            raise ValueError("Any exception is translated to runtime-error")

        prog = re.compile(search_expression)
        for tsi in self.ts_infos:
            if prog.fullmatch(tsi.name):
                r.append(tsi)
        return r

    def dtss_store_callback(self, tsv: TsVector) -> None:
        self.stored_tsv.append(tsv)

    def test_functionality_hosting_localhost(self):

        # setup data to be calculated
        utc = Calendar()
        d = deltahours(1)
        d24 = deltahours(24)
        n = 240
        n24 = 10
        t = utc.time(2016, 1, 1)
        ta = TimeAxis(t, d, n)
        ta24 = TimeAxis(t, d24, n24)
        n_ts = 100
        percentile_list = IntVector([0, 35, 50, 65, 100])
        tsv = TsVector()
        store_tsv = TsVector()  # something we store at server side
        for i in range(n_ts):
            pts = TimeSeries(ta, np.linspace(start=0, stop=1.0, num=ta.size()),
                             point_fx.POINT_AVERAGE_VALUE)
            tsv.append(float(1 + i / 10) * pts)
            store_tsv.append(TimeSeries("cache://test/{0}".format(i), pts))  # generate a bound pts to store

        dummy_ts = TimeSeries('dummy://a')
        tsv.append(dummy_ts.integral(ta))

        # then start the server
        dtss = DtsServer()
        port_no = 20000
        host_port = 'localhost:{0}'.format(port_no)
        dtss.set_listening_port(port_no)
        dtss.cb = self.dtss_read_callback
        dtss.find_cb = self.dtss_find_callback
        dtss.store_ts_cb = self.dtss_store_callback

        dtss.start_async()

        dts = DtsClient(host_port)
        # then try something that should work
        dts.store_ts(store_tsv)
        r1 = dts.evaluate(tsv, ta.total_period())
        r2 = dts.percentiles(tsv, ta.total_period(), ta24, percentile_list)
        r3 = dts.find('netcdf://dummy\.nc/ts\d')
        self.rd_throws = True
        ex_count = 0
        try:
            rx = dts.evaluate(tsv, ta.total_period())
        except RuntimeError as e:
            ex_count = 1
            pass
        self.rd_throws = True
        try:
            fx = dts.find('should throw')
        except RuntimeError as e:
            ex_count += 1
            pass

        dts.close()  # close connection (will use context manager later)
        dtss.clear()  # close server
        self.assertEqual(ex_count, 2)
        self.assertEqual(len(r1), len(tsv))
        self.assertEqual(self.callback_count, 3)
        for i in range(n_ts - 1):
            self.assertEqual(r1[i].time_axis, tsv[i].time_axis)
            assert_array_almost_equal(r1[i].values.to_numpy(), tsv[i].values.to_numpy(), decimal=4)

        self.assertEqual(len(r2), len(percentile_list))
        dummy_ts.bind(TimeSeries(ta, fill_value=1.0))
        p2 = tsv.percentiles(ta24, percentile_list)
        # r2 = tsv.percentiles(ta24,percentile_list)

        for i in range(len(p2)):
            self.assertEqual(r2[i].time_axis, p2[i].time_axis)
            assert_array_almost_equal(r2[i].values.to_numpy(), p2[i].values.to_numpy(), decimal=1)

        self.assertEqual(self.find_count, 2)
        self.assertEqual(len(r3), 10)  # 0..9
        for i in range(len(r3)):
            self.assertEqual(r3[i], self.ts_infos[i])

        self.assertEqual(1, len(self.stored_tsv))
        self.assertEqual(len(store_tsv), len(self.stored_tsv[0]))
        for i in range(len(store_tsv)):
            self.assertEqual(self.stored_tsv[0][i].ts_id(), store_tsv[i].ts_id())

    def test_ts_store(self):
        """
        This test verifies the shyft internal time-series store,
        that allow identified time-series to be stored
        in the backend using a directory container specified for the
        location.

        All time-series of the form shyft://<container>/<ts-name>
        is mapped to the configured <container> (aka a directory on the server)

        This applies to expressions, as well as the new
        .store_ts(ts_vector) function that allows the user to
        stash away time-series into the configured back-end container.

        All find-operations of the form shyft://<container>/<regular-expression>
        is mapped to a search in the corresponding directory for the <container>

        :return:
        """
        with tempfile.TemporaryDirectory() as c_dir:
            # setup data to be calculated
            utc = Calendar()
            d = deltahours(1)
            n = 240
            t = utc.time(2016, 1, 1)
            ta = TimeAxis(t, d, n)
            n_ts = 10
            store_tsv = TsVector()  # something we store at server side
            tsv = TsVector()  # something we put an expression into, refering to stored ts-symbols

            def ts_url(name: str) -> str:
                return "shyft://test/{}".format(name)  # shyft:// maps to internal, test= container-name

            for i in range(n_ts):
                pts = TimeSeries(ta, np.linspace(start=0, stop=1.0 * i, num=ta.size()),
                                 point_fx.POINT_AVERAGE_VALUE)
                ts_id = ts_url("{0}".format(i))
                tsv.append(float(1.0) * TimeSeries(ts_id))  # make an expression that returns what we store
                store_tsv.append(TimeSeries(ts_id, pts))  # generate a bound pts to store

            # then start the server
            dtss = DtsServer()
            port_no = 20000
            host_port = 'localhost:{0}'.format(port_no)
            dtss.set_listening_port(port_no)
            dtss.set_container("test", c_dir)  # notice we set container 'test' to point to c_dir directory
            dtss.start_async()  # the internal shyft time-series will be stored to that container
            # also notice that we dont have to setup callbacks in this case (but we could, and they would work)
            #
            # finally start the action
            dts = DtsClient(host_port)
            # then try something that should work
            dts.store_ts(store_tsv)
            r1 = dts.evaluate(tsv, ta.total_period())
            f1 = dts.find(r"shyft://test/\d")  # find all ts with one digit, 0..9
            dts.close()  # close connection (will use context manager later)
            dtss.clear()  # close server

            # now the moment of truth:
            self.assertEqual(len(r1), len(tsv))
            for i in range(n_ts - 1):
                self.assertEqual(r1[i].time_axis, store_tsv[i].time_axis)
                assert_array_almost_equal(r1[i].values.to_numpy(), store_tsv[i].values.to_numpy(), decimal=4)

            self.assertEqual(len(f1), 10)
            # notice that the created, and delta_t is always no_utctime, and delta_t is 0 (since we don't waste
            # time digging to much into the file to figure it out.
            # the created time could possibly be computed as file-creation time (if avail via file-api).
            #
            # self.assertEqual(1,len(self.stored_tsv))
            # self.assertEqual(len(store_tsv),len(self.stored_tsv[0]))
            # for i in range(len(store_tsv)):
            #    self.assertEqual(self.stored_tsv[0][i].ts_id(),store_tsv[i].ts_id())
