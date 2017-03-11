import numpy as np
from numpy.testing import assert_array_almost_equal
import unittest

from shyft.api import deltahours
from shyft.api import Calendar
from shyft.api import TimeAxis
from shyft.api import TimeSeries
from shyft.api import TsVector
from shyft.api import point_interpretation_policy as point_fx
from shyft.api import DtsServer
from shyft.api import DtsClient
from shyft.api import IntVector


class DtssTestCase(unittest.TestCase):
    """Verify and illustrate dtts, distributed ts service

     """

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

        # then start the server
        dtss = DtsServer()
        port_no = 20000
        host_port = 'localhost:{0}'.format(port_no)
        dtss.set_listening_port(port_no)
        dtss.start_async()
        dts = DtsClient(host_port)
        # then try something that should work
        r1 = dts.evaluate(tsv, ta.total_period())
        r2 = dts.percentiles(tsv, ta.total_period(), ta24, percentile_list)
        dts.close()  # close connection (will use context manager later)
        dtss.clear()  # close server

        self.assertEqual(len(r1), len(tsv))

        for i in range(n_ts):
            self.assertEqual(r1[i].time_axis, tsv[i].time_axis)
            assert_array_almost_equal(r1[i].values.to_numpy(), tsv[i].values.to_numpy(), decimal=4)

        self.assertEqual(len(r2), len(percentile_list))
        p2 = tsv.percentiles(ta24, percentile_list)
        # r2 = tsv.percentiles(ta24,percentile_list)

        for i in range(len(p2)):
            self.assertEqual(r2[i].time_axis, p2[i].time_axis)
            assert_array_almost_equal(r2[i].values.to_numpy(), p2[i].values.to_numpy(), decimal=1)
