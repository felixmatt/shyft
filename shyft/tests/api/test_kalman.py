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
        p=api.KalmanParameter()
        self.assertEqual(p.n_daily_observations, 8)
        self.assertAlmostEqual(p.hourly_correlation, 0.93)
        self.assertAlmostEqual(p.covariance_init, 0.5)
        self.assertAlmostEqual(p.std_error_bias_measurements, 2.0)
        self.assertAlmostEqual(p.ratio_std_w_over_v, 0.06)
        p=api.KalmanParameter(n_daily_observations=6, hourly_correlation=0.9,covariance_init=0.4,std_error_bias_measurements=1.0,ratio_std_w_over_v=0.05)
        self.assertEqual(p.n_daily_observations, 6)
        self.assertAlmostEqual(p.ratio_std_w_over_v, 0.05)
        self.assertAlmostEqual(p.hourly_correlation, 0.9)
        self.assertAlmostEqual(p.covariance_init, 0.4)
        self.assertAlmostEqual(p.std_error_bias_measurements, 1.0)
        self.assertAlmostEqual(p.ratio_std_w_over_v, 0.05)
        p.n_daily_observations = 8
        self.assertEqual(p.n_daily_observations, 8)
        q=api.KalmanParameter(p)
        self.assertEqual(p.n_daily_observations, q.n_daily_observations)
        self.assertAlmostEqual(p.hourly_correlation, q.hourly_correlation)
        self.assertAlmostEqual(p.covariance_init, q.covariance_init)
        self.assertAlmostEqual(p.std_error_bias_measurements, q.std_error_bias_measurements)
        self.assertAlmostEqual(p.ratio_std_w_over_v, q.ratio_std_w_over_v)

    def test_state(self):
        s=api.KalmanState()
        self.assertEqual(s.size(), 0)
        s=api.KalmanState(n_daily_observations=8, covariance_init=0.5, hourly_correlation=0.93, process_noise_init=0.06)
        self.assertEqual(s.size(), 8)

    def test_filter(self):
        f = api.KalmanFilter()
        self.assertEqual(f.parameter.n_daily_observations, 8)
        s = f.create_initial_state()
        self.assertEqual(s.size(),8)
        utc=api.Calendar()
        t0 = utc.time(2015,1,1)
        dt = api.deltahours(3)
        n = 8
        ta = api.Timeaxis2(t0,dt,n)
        for i in range(ta.size()):
            f.update(2.0,ta.time(i),s)
        x = s.x
        self.assertEqual(len(x),8)
        self.assertEqual(len(s.k),8)
        self.assertEqual(s.P.shape[0], 8)
        self.assertEqual(s.P.shape[1], 8)
        self.assertEqual(s.W.shape[0], 8)
        self.assertEqual(s.W.shape[1], 8)

if __name__ == "__main__":
    unittest.main()
