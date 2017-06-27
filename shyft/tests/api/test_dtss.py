import numpy as np
from numpy.testing import assert_array_almost_equal
import unittest
import re
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
    def __init__(self,*args,**kwargs):
        super(DtssTestCase,self).__init__(*args,**kwargs)
        self.callback_count = 0
        self.find_count = 0
        self.ts_infos = TsInfoVector()
        utc=Calendar()
        t_now = utctime_now()
        for i in range(30):
            self.ts_infos.append(
                TsInfo(
                    name='netcdf://dummy.nc/ts{0}'.format(i),
                    point_fx=point_fx.POINT_AVERAGE_VALUE,
                    delta_t=deltahours(1),
                    olson_tz_id='',
                    data_period=UtcPeriod(utc.time(2017,1,1), utc.time(2018, 1, 1)),
                    created=t_now,
                    modified=t_now
                )
            )

    def dtss_read_callback(self,ts_ids:StringVector,read_period:UtcPeriod)->TsVector:
        self.callback_count += 1
        r = TsVector()
        ta = TimeAxis(read_period.start,deltahours(1),read_period.timespan()//deltahours(1))
        for ts_id in ts_ids:
            r.append(TimeSeries(ta,fill_value=1.0,point_fx=point_fx.POINT_AVERAGE_VALUE))
        return r

    def dtss_find_callback(self,search_expression: str)-> TsInfoVector:
        self.find_count += 1
        r = TsInfoVector()
        prog = re.compile(search_expression)
        for tsi in self.ts_infos:
            if prog.fullmatch(tsi.name):
                r.append(tsi)
        return r

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
        for i in range(n_ts):
            tsv.append(float(1 + i / 10) * TimeSeries(ta, np.linspace(start=0, stop=1.0, num=ta.size()),
                                                      point_fx.POINT_AVERAGE_VALUE))

        dummy_ts= TimeSeries('dummy://a')
        tsv.append(dummy_ts.integral(ta))

        # then start the server
        dtss = DtsServer()
        port_no = 20000
        host_port = 'localhost:{0}'.format(port_no)
        dtss.set_listening_port(port_no)
        dtss.cb = self.dtss_read_callback
        dtss.find_cb = self.dtss_find_callback

        dtss.start_async()

        dts = DtsClient(host_port)
        # then try something that should work
        r1 = dts.evaluate(tsv, ta.total_period())
        r2 = dts.percentiles(tsv, ta.total_period(), ta24, percentile_list)
        r3 = dts.find('netcdf://dummy\.nc/ts\d')
        dts.close()  # close connection (will use context manager later)
        dtss.clear()  # close server

        self.assertEqual(len(r1), len(tsv))
        self.assertEqual(self.callback_count,2)
        for i in range(n_ts-1):
            self.assertEqual(r1[i].time_axis, tsv[i].time_axis)
            assert_array_almost_equal(r1[i].values.to_numpy(), tsv[i].values.to_numpy(), decimal=4)

        self.assertEqual(len(r2), len(percentile_list))
        dummy_ts.bind(TimeSeries(ta,fill_value=1.0))
        p2 = tsv.percentiles(ta24, percentile_list)
        # r2 = tsv.percentiles(ta24,percentile_list)

        for i in range(len(p2)):
            self.assertEqual(r2[i].time_axis, p2[i].time_axis)
            assert_array_almost_equal(r2[i].values.to_numpy(), p2[i].values.to_numpy(), decimal=1)

        self.assertEqual(self.find_count, 1)
        self.assertEqual(len(r3), 10)  # 0..9
        for i in range(len(r3)):
            self.assertEqual(r3[i],self.ts_infos[i])


