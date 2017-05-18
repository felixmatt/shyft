from shyft import api
import numpy as np
from numpy.testing import assert_array_almost_equal
import unittest


class KalmanAndBiasPrediction(unittest.TestCase):
    """
    These tests verifies and demonstrates the Kalman and BiasPrediction functions available
    in SHyFT api.

    """

    def test_parameter(self):
        p = api.KalmanParameter()
        self.assertEqual(p.n_daily_observations, 8)
        self.assertAlmostEqual(p.hourly_correlation, 0.93)
        self.assertAlmostEqual(p.covariance_init, 0.5)
        self.assertAlmostEqual(p.std_error_bias_measurements, 2.0)
        self.assertAlmostEqual(p.ratio_std_w_over_v, 0.15)
        p = api.KalmanParameter(n_daily_observations=6, hourly_correlation=0.9, covariance_init=0.4,
                                std_error_bias_measurements=1.0, ratio_std_w_over_v=0.05)
        self.assertEqual(p.n_daily_observations, 6)
        self.assertAlmostEqual(p.ratio_std_w_over_v, 0.05)
        self.assertAlmostEqual(p.hourly_correlation, 0.9)
        self.assertAlmostEqual(p.covariance_init, 0.4)
        self.assertAlmostEqual(p.std_error_bias_measurements, 1.0)
        self.assertAlmostEqual(p.ratio_std_w_over_v, 0.05)
        p.n_daily_observations = 8
        self.assertEqual(p.n_daily_observations, 8)
        q = api.KalmanParameter(p)
        self.assertEqual(p.n_daily_observations, q.n_daily_observations)
        self.assertAlmostEqual(p.hourly_correlation, q.hourly_correlation)
        self.assertAlmostEqual(p.covariance_init, q.covariance_init)
        self.assertAlmostEqual(p.std_error_bias_measurements, q.std_error_bias_measurements)
        self.assertAlmostEqual(p.ratio_std_w_over_v, q.ratio_std_w_over_v)

    def test_state(self):
        s = api.KalmanState()
        self.assertEqual(s.size(), 0)
        s = api.KalmanState(n_daily_observations=8, covariance_init=0.5, hourly_correlation=0.93,
                            process_noise_init=0.06)
        self.assertEqual(s.size(), 8)

    def test_filter(self):
        f = api.KalmanFilter()
        self.assertEqual(f.parameter.n_daily_observations, 8)
        s = f.create_initial_state()
        self.assertEqual(s.size(), 8)
        utc = api.Calendar()
        t0 = utc.time(2015, 1, 1)
        dt = api.deltahours(3)
        n = 8
        ta = api.TimeAxis(t0, dt, n)
        for i in range(ta.size()):
            f.update(2.0, ta.time(i), s)
        x = s.x
        self.assertEqual(len(x), 8)
        self.assertEqual(len(s.k), 8)
        self.assertEqual(s.P.shape[0], 8)
        self.assertEqual(s.P.shape[1], 8)
        self.assertEqual(s.W.shape[0], 8)
        self.assertEqual(s.W.shape[1], 8)

    def _create_geo_forecast_set(self, n_fc, t0, dt, n_steps, dt_fc, fx):
        """

        Parameters
        ----------
        n_fc : int number of forecasts, e.g. 8
        t0 : utctime start of first forecast
        dt : utctimespan delta t for forecast-ts
        n_steps : number of steps in one forecast-ts
        dt_fc : utctimespan delta t between each forecast, like deltahours(6)
        fx : lambda time_axis:  a function returning a DoubleVector with values for the supplied time-axis

        Returns
        -------
        api.TemperatureSourceVector()

        """
        fc_set = api.TemperatureSourceVector()
        geo_point = api.GeoPoint(0.0, 0.0, 0.0)  # any point will do, we just reuse the geo-ts
        for i in range(n_fc):
            ta = api.TimeAxis(t0 + i * dt_fc, dt, n_steps)
            ts = api.TimeSeries(ta=ta, values=fx(ta),
                                point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)
            geo_ts = api.TemperatureSource(geo_point, ts)
            fc_set.append(geo_ts)
        return fc_set

    def _create_forecast_set(self, n_fc, t0, dt, n_steps, dt_fc, fx):
        """

        Parameters
        ----------
        n_fc : int number of forecasts, e.g. 8
        t0 : utctime start of first forecast
        dt : utctimespan delta t for forecast-ts
        n_steps : number of steps in one forecast-ts
        dt_fc : utctimespan delta t between each forecast, like deltahours(6)
        fx : lambda time_axis:  a function returning a DoubleVector with values for the supplied time-axis

        Returns
        -------
        api.TsVector()

        """
        fc_set = api.TsVector()
        for i in range(n_fc):
            ta = api.TimeAxis(t0 + i * dt_fc, dt, n_steps)
            ts = api.TimeSeries(ta=ta, values=fx(ta),
                                point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)
            fc_set.append(ts)
        return fc_set

    def _create_fc_values(self, time_axis, x):
        v = np.arange(time_axis.size())
        v.fill(x)
        return api.DoubleVector.from_numpy(v)

    def test_bias_predictor(self):
        """
        Verify that if we feed forecast[n] and observation into the bias-predictor
        it will create the estimated bias offsets
        """
        f = api.KalmanFilter()
        bp = api.KalmanBiasPredictor(f)
        self.assertIsNotNone(bp)
        self.assertEqual(bp.filter.parameter.n_daily_observations, 8)

        n_fc = 8
        utc = api.Calendar()
        t0 = utc.time(2016, 1, 1)
        dt = api.deltahours(1)
        n_fc_steps = 36  # e.g. like arome 36 hours
        fc_dt = api.deltahours(6)
        fc_fx = lambda time_axis: self._create_fc_values(time_axis, 2.0)  # just return a constant 2.0 deg C for now
        fc_set = self._create_geo_forecast_set(n_fc, t0, dt, n_fc_steps, fc_dt, fc_fx)
        n_obs = 24
        obs_ta = api.TimeAxis(t0, dt, n_obs)
        obs_ts = api.TimeSeries(obs_ta, fill_value=0.0)
        kalman_dt = api.deltahours(3)  # suitable average for prediction temperature
        kalman_ta = api.TimeAxis(t0, kalman_dt, 8)
        bp.update_with_forecast(fc_set, obs_ts, kalman_ta)  # here we feed in forecast-set and observation into kalman
        fc_setv = self._create_forecast_set(n_fc, t0, dt, n_fc_steps, fc_dt, fc_fx)
        bp.update_with_forecast(fc_setv, obs_ts, kalman_ta)  # also verify we can feed in a pure TsVector
        bias_pattern = bp.state.x  # the bp.state.x is now the best estimates fo the bias between fc and observation
        self.assertEqual(len(bias_pattern), 8)
        for i in range(len(bias_pattern)):
            self.assertLess(abs(bias_pattern[i] - 2.0), 0.2)  # bias should iterate to approx 2.0 degC now.

    def test_compute_running_bias(self):
        """
        Verify that if we feed forecast[n] and observation into the bias-predictor
        it will create the estimated bias offsets
        """
        f = api.KalmanFilter()
        bp = api.KalmanBiasPredictor(f)
        self.assertIsNotNone(bp)
        self.assertEqual(bp.filter.parameter.n_daily_observations, 8)

        n_fc = 1
        utc = api.Calendar()
        t0 = utc.time(2016, 1, 1)
        dt = api.deltahours(1)
        n_fc_steps = 24 * 10  # 10 days history
        fc_dt = api.deltahours(6)
        fc_fx = lambda time_axis: self._create_fc_values(time_axis, 2.0)  # just return a constant 2.0 deg C for now

        n_obs = n_fc_steps
        obs_ta = api.TimeAxis(t0, dt, n_obs)
        obs_ts = api.TimeSeries(obs_ta, fill_value=0.0)
        kalman_dt = api.deltahours(3)  # suitable average for prediction temperature
        kalman_ta = api.TimeAxis(t0, kalman_dt, n_obs // 3)
        fc_ts = self._create_forecast_set(n_fc, t0, dt, n_fc_steps, fc_dt, fc_fx)[0]
        bias_ts = bp.compute_running_bias(fc_ts, obs_ts, kalman_ta)  # also verify we can feed in a pure TsVector
        bias_pattern = bp.state.x  # the bp.state.x is now the best estimates fo the bias between fc and observation
        self.assertEqual(len(bias_pattern), 8)
        for i in range(len(bias_pattern)):
            self.assertLess(abs(bias_pattern[i] - 2.0), 0.2)  # bias should iterate to approx 2.0 degC now.
        # and...:
        for i in range(8):
            self.assertAlmostEqual(bias_ts.value(i), 0.0)  # expect 0.0 for the first day

        for i in range(8):
            self.assertLess(abs(bias_ts.value(bias_ts.size() - i-1) - 2.0), 0.2)  # last part should be 2.0 deg.C


if __name__ == "__main__":
    unittest.main()
