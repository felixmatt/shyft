import re
import socket
import tempfile
import unittest
from contextlib import closing

import numpy as np
from numpy.testing import assert_array_almost_equal

from shyft.api import Calendar
from shyft.api import DtsClient
from shyft.api import DtsServer
from shyft.api import IntVector
from shyft.api import StringVector
from shyft.api import TimeAxis
from shyft.api import TimeSeries
from shyft.api import TsInfo
from shyft.api import TsInfoVector
from shyft.api import TsVector
from shyft.api import UtcPeriod
from shyft.api import deltahours
from shyft.api import point_interpretation_policy as point_fx
from shyft.api import utctime_now


def shyft_store_url(name: str) -> str:
    return "shyft://test/{}".format(name)  # shyft:// maps to internal, test= container-name


fake_store_container = "netcdf://dummy.nc"


def fake_store_url(name: str) -> str:
    return "{}/ts{}".format(fake_store_container, name)  #


def find_free_port() -> int:
    """
    from SO https://stackoverflow.com/questions/1365265/on-localhost-how-to-pick-a-free-port-number
    :return: available port number for use
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


class DtssTestCase(unittest.TestCase):
    """Verify and illustrate dtts, distributed ts service

     """

    def __init__(self, *args, **kwargs):
        super(DtssTestCase, self).__init__(*args, **kwargs)
        self.callback_count = 0
        self.find_count = 0
        self.ts_infos = TsInfoVector()
        self.rd_throws = False
        self.cache_reads = False
        self.cache_dtss = None
        utc = Calendar()
        t_now = utctime_now()
        self.stored_tsv = list()

        for i in range(30):
            self.ts_infos.append(
                TsInfo(
                    name=fake_store_url('{0}'.format(i)),
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
        ta = TimeAxis(read_period.start, deltahours(1), read_period.timespan()//deltahours(1))
        if self.rd_throws:
            self.rd_throws = False
            raise RuntimeError("read-ts-problem")
        for ts_id in ts_ids:
            r.append(TimeSeries(ta, fill_value=1.0, point_fx=point_fx.POINT_AVERAGE_VALUE))

        if self.cache_reads and self.cache_dtss:  # illustrate how the read-callback can ask the dtss to cache it's reads
            self.cache_dtss.cache(ts_ids, r)

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
            tsv.append(float(1 + i/10)*pts)
            store_tsv.append(TimeSeries("cache://test/{0}".format(i), pts))  # generate a bound pts to store

        dummy_ts = TimeSeries('dummy://a')
        tsv.append(dummy_ts.integral(ta))

        # then start the server
        dtss = DtsServer()
        port_no = find_free_port()
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
        dummy_ts.bind(TimeSeries(ta, fill_value=1.0, point_fx=point_fx.POINT_AVERAGE_VALUE))
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
            n = 365*24//3
            t = utc.time(2016, 1, 1)
            ta = TimeAxis(t, d, n)
            n_ts = 10
            store_tsv = TsVector()  # something we store at server side
            tsv = TsVector()  # something we put an expression into, refering to stored ts-symbols

            for i in range(n_ts):
                pts = TimeSeries(ta, np.sin(np.linspace(start=0, stop=1.0*i, num=ta.size())),
                                 point_fx.POINT_AVERAGE_VALUE)
                ts_id = shyft_store_url("{0}".format(i))
                tsv.append(float(1.0)*TimeSeries(ts_id))  # make an expression that returns what we store
                store_tsv.append(TimeSeries(ts_id, pts))  # generate a bound pts to store
            # krls with some extra challenges related to serialization
            tsv_krls = TsVector()
            krls_ts = TimeSeries(shyft_store_url("9")).krls_interpolation(dt=d, gamma=1e-3, tolerance=0.001, size=ta.size())
            tsv_krls.append(krls_ts)

            # then start the server
            dtss = DtsServer()
            port_no = find_free_port()
            host_port = 'localhost:{0}'.format(port_no)
            dtss.set_auto_cache(True)
            std_max_items = dtss.cache_max_items
            dtss.cache_max_items = 3000
            tst_max_items = dtss.cache_max_items
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
            r2 = dts.evaluate(tsv_krls, ta.total_period())
            dts.close()  # close connection (will use context manager later)
            dtss.clear()  # close server

            # now the moment of truth:
            self.assertEqual(len(r1), len(tsv))
            for i in range(n_ts - 1):
                self.assertEqual(r1[i].time_axis, store_tsv[i].time_axis)
                assert_array_almost_equal(r1[i].values.to_numpy(), store_tsv[i].values.to_numpy(), decimal=4)

            self.assertEqual(len(f1), 10)
            self.assertEqual(len(r2), len(tsv_krls))
            self.assertEqual(1000000,std_max_items)
            self.assertEqual(3000,tst_max_items)

    def test_ts_cache(self):
        """ Verify dtss ts-cache functions exposed to python """
        with tempfile.TemporaryDirectory() as c_dir:
            # setup data to be calculated
            utc = Calendar()
            d = deltahours(1)
            n = 100
            t = utc.time(2016, 1, 1)
            ta = TimeAxis(t, d, n)
            n_ts = 10
            store_tsv = TsVector()  # something we store at server side
            tsv = TsVector()  # something we put an expression into, refering to stored ts-symbols

            for i in range(n_ts):
                pts = TimeSeries(ta, np.sin(np.linspace(start=0, stop=1.0*i, num=ta.size())),
                                 point_fx.POINT_AVERAGE_VALUE)
                ts_id = shyft_store_url("{0}".format(i))
                tsv.append(float(1.0)*TimeSeries(ts_id))  # make an expression that returns what we store
                store_tsv.append(TimeSeries(ts_id, pts))  # generate a bound pts to store

            # add one external ts
            tsv.append(TimeSeries(fake_store_url("_any_ts_id_will_do")))
            # then start the server
            dtss = DtsServer()

            dtss.cb = self.dtss_read_callback  # rig external callbacks as well.
            self.callback_count = 0
            self.rd_throws = False
            cache_on_write = True
            port_no = find_free_port()
            host_port = 'localhost:{0}'.format(port_no)
            dtss.set_auto_cache(True)
            dtss.set_listening_port(port_no)
            dtss.set_container("test", c_dir)  # notice we set container 'test' to point to c_dir directory
            dtss.start_async()  # the internal shyft time-series will be stored to that container

            dts = DtsClient(host_port)
            cs0 = dtss.cache_stats
            dts.store_ts(store_tsv, cache_on_write)
            r1 = dts.evaluate(tsv, ta.total_period())
            cs1 = dtss.cache_stats
            dtss.flush_cache_all()  # force the cache empty
            dtss.clear_cache_stats()
            cs2 = dtss.cache_stats  # just to ensure clear did work
            r1 = dts.evaluate(tsv, ta.total_period())  # second evaluation, cache is empty, will force read(misses)
            cs3 = dtss.cache_stats
            r1 = dts.evaluate(tsv, ta.total_period())  # third evaluation, cache is now filled, all hits
            cs4 = dtss.cache_stats
            # now verify explicit caching performed by the python callback
            self.cache_dtss = dtss
            self.cache_reads = True
            dtss.flush_cache_all()
            dtss.clear_cache_stats()
            dtss.set_auto_cache(False)  # turn off auto caching, we want to test the explicit caching
            r1 = dts.evaluate(tsv, ta.total_period())  # evaluation, just misses, but we cache explict the external
            cs5 = dtss.cache_stats  # ok base line a lots of misses
            r1 = dts.evaluate(tsv, ta.total_period())
            cs6 = dtss.cache_stats  # should be one hit here

            dts.close()  # close connection (will use context manager later)
            dtss.clear()  # close server

            # now the moment of truth:
            self.assertEqual(len(r1), len(tsv))
            for i in range(n_ts - 1):
                self.assertEqual(r1[i].time_axis, store_tsv[i].time_axis)
                assert_array_almost_equal(r1[i].values.to_numpy(), store_tsv[i].values.to_numpy(), decimal=4)

            self.assertEqual(cs0.hits, 0)
            self.assertEqual(cs0.misses, 0)
            self.assertEqual(cs0.coverage_misses, 0)
            self.assertEqual(cs0.id_count, 0)
            self.assertEqual(cs0.point_count, 0)
            self.assertEqual(cs0.fragment_count, 0)

            self.assertEqual(cs1.hits, n_ts)
            self.assertEqual(cs1.misses, 1)  # because we cache on store, so 10 cached, 1 external with miss
            self.assertEqual(cs1.coverage_misses, 0)
            self.assertEqual(cs1.id_count, n_ts + 1)
            self.assertEqual(cs1.point_count, (n_ts + 1)*n)
            self.assertEqual(cs1.fragment_count, n_ts + 1)

            self.assertEqual(cs2.hits, 0)
            self.assertEqual(cs2.misses, 0)
            self.assertEqual(cs2.coverage_misses, 0)
            self.assertEqual(cs2.id_count, 0)
            self.assertEqual(cs2.point_count, 0)
            self.assertEqual(cs2.fragment_count, 0)

            self.assertEqual(cs3.hits, 0)
            self.assertEqual(cs3.misses, n_ts + 1)  # because we cache on store, we don't even miss one time
            self.assertEqual(cs3.coverage_misses, 0)
            self.assertEqual(cs3.id_count, n_ts + 1)
            self.assertEqual(cs3.point_count, (n_ts + 1)*n)
            self.assertEqual(cs3.fragment_count, n_ts + 1)

            self.assertEqual(cs4.hits, n_ts + 1)  # because previous read filled cache
            self.assertEqual(cs4.misses, n_ts + 1)  # remembers previous misses.
            self.assertEqual(cs4.coverage_misses, 0)
            self.assertEqual(cs4.id_count, n_ts + 1)
            self.assertEqual(cs4.point_count, (n_ts + 1)*n)
            self.assertEqual(cs4.fragment_count, n_ts + 1)

            self.assertEqual(cs6.hits, 1)  # because previous read filled cache
            self.assertEqual(cs6.misses, n_ts*2 + 1)  # remembers previous misses.
            self.assertEqual(cs6.coverage_misses, 0)
            self.assertEqual(cs6.id_count, 1)
            self.assertEqual(cs6.point_count, 1*n)
            self.assertEqual(cs6.fragment_count,  1)
