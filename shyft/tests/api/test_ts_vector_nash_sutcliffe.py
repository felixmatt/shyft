import unittest
import math
import numpy as np
from shyft.api import Calendar
from shyft.api import TsVector
from shyft.api import TimeSeries
from shyft.api import TimeAxis
from shyft.api import point_interpretation_policy as ts_point_fx
from shyft.api import deltahours


class TsVectorNashSutcliffe(unittest.TestCase):
    def _create_forecasts(self, t0: int, dt: int, n: int, fc_dt: int, fc_n: int) -> TsVector:
        tsv = TsVector()
        stair_case = ts_point_fx.POINT_AVERAGE_VALUE
        for i in range(fc_n):
            ta = TimeAxis(t0 + i * fc_dt, dt, n)
            ts = TimeSeries(ta, fill_value=0.0, point_fx=stair_case)
            for t in range(len(ta)):
                tt = i * fc_dt / dt + t
                ts.set(t, math.sin(0.314 + 3.14 * tt / 240.0))  # make it a sin-wave at time tt
            tsv.append(ts)
        return tsv

    def _create_observation(self, t0: int, dt: int, fc_lead_steps: int, fc_n: int) -> TimeSeries:
        ta = TimeAxis(t0, dt, fc_n * fc_lead_steps)
        ts = TimeSeries(ta, fill_value=0.0, point_fx=ts_point_fx.POINT_AVERAGE_VALUE)
        for i in range(len(ta)):
            ts.set(i, math.sin(0.314 + 3.14 * i / 240.0))  # sin-wave at specified 'time' i
        return ts

    def test_forecasts_ns(self):
        utc = Calendar()
        t0 = utc.time(2017, 1, 1)
        dt = deltahours(1)
        n = 66  # typical arome
        fc_dt_n_hours = 6
        fc_dt = deltahours(fc_dt_n_hours)
        fc_n = 4 * 10  # 4 each day 10 days
        fc_v = self._create_forecasts(t0, dt, n, fc_dt, fc_n)
        obs = self._create_observation(t0, dt, fc_dt_n_hours, fc_n)
        for lead_time_hours in range(12):
            for slice_length_units in [1, 2, 3, 4, 6, 12]:
                for dt_hours in [1, 2, 3]:
                    ns = fc_v.nash_sutcliffe(
                        obs,
                        deltahours(lead_time_hours),
                        deltahours(dt_hours),
                        slice_length_units
                    )
                    self.assertAlmostEqual(ns, 1.0, 1,  # for some reason we get 0.99..
                                           'should match close to 1.0 for lead_hour= {},dt={},n={}'.format(
                                               lead_time_hours, dt_hours, slice_length_units)
                                           )

    def test_forecast_average_slice(self):
        """
        Demo and test TsVector.average_slice(lead_time,dt,n)
        """
        utc = Calendar()
        t0 = utc.time(2017, 1, 1)
        dt = deltahours(1)
        n = 66  # typical arome
        fc_dt_n_hours = 6
        fc_dt = deltahours(fc_dt_n_hours)
        fc_n = 4 * 10  # 4 each day 10 days
        fc_v = self._create_forecasts(t0, dt, n, fc_dt, fc_n)
        for lead_time_hours in range(12):
            for slice_length_units in [1, 2, 3, 4, 6, 12]:
                for dt_hours in [1, 2, 3]:
                    slice_v = fc_v.average_slice(
                        deltahours(lead_time_hours),
                        deltahours(dt_hours),
                        slice_length_units
                    )
                    self.assertEqual(len(slice_v),len(fc_v))
                    # then loop over the slice_v and prove it's equal
                    # to the average of the same portion on the originalj
                    for s,f in zip(slice_v,fc_v):
                        ta=TimeAxis(f.time_axis.time(0)+deltahours(lead_time_hours),deltahours(dt_hours),slice_length_units)
                        ts_expected = f.average(ta)
                        self.assertTrue(s.time_axis == ts_expected.time_axis)
                        self.assertTrue(np.allclose(s.values.to_numpy(),ts_expected.values.to_numpy()))
        pass