import unittest
import numpy as np

from shyft.api import Calendar
from shyft.api import TsVector
from shyft.api import TimeSeries
from shyft.api import TimeAxis
from shyft.api import point_interpretation_policy as ts_point_fx
from shyft.api import deltahours
from shyft.api import DoubleVector as dv


class ForecastMerge(unittest.TestCase):
    def _create_forecasts(self, t0: int, dt: int, n: int, fc_dt: int, fc_n: int) -> TsVector:
        tsv = TsVector()
        stair_case = ts_point_fx.POINT_AVERAGE_VALUE
        for i in range(fc_n):
            ta = TimeAxis(t0 + i * fc_dt, dt, n)
            mrk = (i+1)/100.0
            v = dv.from_numpy(np.linspace(1 + mrk, 1 + n +mrk, n, endpoint=False))
            tsv.append(TimeSeries(ta, v, stair_case))
        return tsv

    def test_merge_arome_forecast(self):
        utc = Calendar()
        t0 = utc.time(2017, 1, 1)
        dt = deltahours(1)
        n = 66  # typical arome
        fc_dt_n_hours = 6
        fc_dt = deltahours(fc_dt_n_hours)
        fc_n = 4 * 10  # 4 each day 10 days
        fc_v = self._create_forecasts(t0, dt, n, fc_dt, fc_n)

        m0_6 = fc_v.forecast_merge(deltahours(0), fc_dt)
        m1_7 = fc_v.forecast_merge(deltahours(1), fc_dt)
        self.assertIsNotNone(m0_6)
        self.assertIsNotNone(m1_7)
        # check returned sizes
        self.assertEqual(m0_6.size(),fc_n *fc_dt_n_hours)
        self.assertEqual(m1_7.size(), fc_n * fc_dt_n_hours)
        # some few values (it's fully covered in c++ tests
        self.assertAlmostEqual(m0_6.value(0), 1.01)
        self.assertAlmostEqual(m0_6.value(6), 1.02)
        self.assertAlmostEqual(m1_7.value(0), 2.01)
        self.assertAlmostEqual(m1_7.value(6), 2.02)