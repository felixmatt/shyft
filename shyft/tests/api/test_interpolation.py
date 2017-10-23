from shyft import api
import numpy as np
from numpy.testing import assert_array_almost_equal
import unittest


class BayesianKriging(unittest.TestCase):
    """
    The purpose of this testsuite is to verify and illustrate BayesianKriging Temperature interpolation

    The scenarios foreseen where this function is useful is
       in pre-processing data before passing it into SHyFT region-environment
    or
       for studying and testing the algorithm and parameters from python

     """

    def setUp(self):
        self.c = api.Calendar()
        self.d = api.deltahours(1)
        self.n = 24
        self.t = self.c.trim(self.c.time(2016, 9, 1), self.d)
        self.ta = api.TimeAxis(self.t, self.d, self.n)
        self.dx_arome = 2500
        self.dx_model = 1000
        self.nx = 2
        self.ny = 2
        self.mnx = 5
        self.mny = 5
        self.max_elevation = 1000

    def tearDown(self):
        pass

    def _create_geo_ts_grid(self, nx, ny, dx, fx):
        """Create a geo_ts_grid of TemperatureSources,
           We just create a terrain model starting at 0 and increasing to max_elevation at max x and y.
           parameters
           ----------
           nx : int
            number of grid-cells in x direction
           ny : int
            number of grid-cells in y direction
           dx : float
             distance in meters for one of the sides in the grid
           fx : lambda z : 1.0
             a function that takes elevation z as input and generates a np.array len(self.ta) of float64 as time-series
           returns
           -------
           a TemperatureSourceVector() filled with geo-ts representing the grid.

        """
        arome_grid = api.TemperatureSourceVector()
        for i in range(nx):
            for j in range(ny):
                z = self.max_elevation * (i + j) / (nx + ny)
                ts = api.TimeSeries(ta=self.ta, values=fx(z), point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)
                geo_ts = api.TemperatureSource(api.GeoPoint(i * dx, j * dx, z), ts)
                arome_grid.append(geo_ts)
        return arome_grid

    def _create_geo_precipitation_grid(self, nx, ny, dx, fx):
        arome_grid = api.PrecipitationSourceVector()
        for i in range(nx):
            for j in range(ny):
                z = self.max_elevation * (i + j) / (nx + ny)
                ts = api.TimeSeries(ta=self.ta, values=fx(z), point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)
                geo_ts = api.PrecipitationSource(api.GeoPoint(i * dx, j * dx, z), ts)
                arome_grid.append(geo_ts)
        return arome_grid

    def _create_geo_point_grid(self, nx, ny, dx):
        gpv = api.GeoPointVector()
        for i in range(nx):
            for j in range(ny):
                z = self.max_elevation * (i + j) / (nx + ny)
                gpv.append(api.GeoPoint(i*dx, j*dx, z))
        return gpv

    def test_can_run_bayesian_kriging_from_arome25_to_1km(self):
        """
        Verify that if we run btk interpolation, we do get updated time-series according to time-axis and range
        specified.

        """
        # arrange the test with a btk_parameter, a source grid and a destination grid
        btk_parameter = api.BTKParameter(temperature_gradient=-0.6, temperature_gradient_sd=0.25, sill=25.0, nugget=0.5, range=20000.0, zscale=20.0)
        fx = lambda z: api.DoubleVector.from_numpy((20.0 - 0.6 * z / 100) + 3.0 * np.sin(np.arange(start=0, stop=self.n, step=1) * 2 * np.pi / 24.0 - np.pi / 2.0))
        arome_grid = self._create_geo_ts_grid(self.nx, self.ny, self.dx_arome, fx)
        destination_grid = self._create_geo_point_grid(self.mnx, self.mny, self.dx_model)
        ta = api.TimeAxisFixedDeltaT(self.t, self.d * 3, int(self.n / 3))
        # act, - run the bayesian_kriging_temperature algoritm.
        r = api.bayesian_kriging_temperature(arome_grid, destination_grid, ta, btk_parameter)
        # assert
        self.assertIsNotNone(r)
        self.assertEqual(len(r), self.mnx * self.mny)
        for gts in r:  # do some sanity checks for the btk. Note that full-range checking is already done elsewhere
            self.assertEqual(gts.ts.size(), ta.size())
            self.assertLess(np.max(gts.ts.values.to_numpy()), 23.0)  # all values less than ~max
            self.assertGreater(np.min(gts.ts.values.to_numpy()), 7.0)  # all values larger than ~ min

    def test_can_run_bayesian_kriging_from_observation_sites_to_1km_grid(self):
        """
        Somewhat more complex test, first do kriging of 1 timeseries out to grid (expect same values flat)
        then do kriging of 3 time-series out to the grid (expect different values, no real verification here since this is done elsewhere

        """
        # arrange the test with a btk_parameter, a source grid and a destination grid
        btk_parameter = api.BTKParameter(temperature_gradient=-0.6, temperature_gradient_sd=0.25, sill=25.0, nugget=0.5, range=20000.0, zscale=20.0)
        fx = lambda z: api.DoubleVector.from_numpy(np.zeros(self.n))

        grid_1km_1 = self._create_geo_point_grid(self.mnx, self.mny, self.dx_model)
        grid_1km_3 = self._create_geo_point_grid(self.mnx, self.mny, self.dx_model)

        observation_sites = api.TemperatureSourceVector()
        ta_obs = api.TimeAxisFixedDeltaT(self.t, self.d * 3, int(self.n / 3))
        ta_grid = api.TimeAxisFixedDeltaT(self.t, self.d, self.n)
        linear = api.point_interpretation_policy.POINT_INSTANT_VALUE
        ts_site_1 = api.TimeSeries(ta_obs,
                                   values=api.DoubleVector.from_numpy(
            (20.0 - 0.6 * 5.0 / 100) + 3.0 * np.sin(np.arange(start=0, stop=ta_obs.size(), step=1) * 2 * np.pi / 8.0 - np.pi / 2.0)
                                   ),
                                   point_fx=linear)
        ts_site_2 = api.TimeSeries(ta_obs, values=api.DoubleVector.from_numpy(
            (20.0 - 0.6 * 500.0 / 100) + 3.0 * np.sin(np.arange(start=0, stop=ta_obs.size(), step=1) * 2 * np.pi / 8.0 - np.pi / 2.0)),
                                   point_fx=linear)
        ts_site_3 = api.TimeSeries(ta_obs, values=api.DoubleVector.from_numpy(
            (20.0 - 0.6 * 1050.0 / 100) + 3.0 * np.sin(np.arange(start=0, stop=ta_obs.size(), step=1) * 2 * np.pi / 8.0 - np.pi / 2.0)),
                                   point_fx=linear)

        observation_sites.append(api.TemperatureSource(api.GeoPoint(50.0, 50.0, 5.0), ts_site_1))

        # act 1: just one time-series put into the system, should give same ts (true-averaged) in all the grid-1km_ts (which can be improved using std.gradient..)
        grid_1km_1ts = api.bayesian_kriging_temperature(observation_sites, grid_1km_1, ta_grid, btk_parameter)

        # assert 1:
        self.assertEqual(len(grid_1km_1ts), self.mnx * self.mny)
        expected_grid_1ts_values = ts_site_1.average(api.TimeAxis(ta_grid)).values.to_numpy()

        for gts in grid_1km_1ts:
            self.assertEqual(gts.ts.size(), ta_grid.size())
            self.assertTrue(np.allclose(expected_grid_1ts_values, gts.ts.values.to_numpy()))

        observation_sites.append(api.TemperatureSource(api.GeoPoint(9000.0, 500.0, 500), ts_site_2))
        observation_sites.append(api.TemperatureSource(api.GeoPoint(9000.0, 12000.0, 1050.0), ts_site_3))

        grid_1km_3ts = api.bayesian_kriging_temperature(observation_sites, grid_1km_3, ta_grid, btk_parameter)

        self.assertEqual(len(grid_1km_3ts), self.mnx * self.mny)

        for gts in grid_1km_3ts:
            self.assertEqual(gts.ts.size(), ta_grid.size())
            self.assertFalse(np.allclose(expected_grid_1ts_values, gts.ts.values.to_numpy()))

    def test_can_run_ordinary_kriging_from_observation_sites_to_1km_grid(self):
        """
        Somewhat more complex test, first do kriging of 1 timeseries out to grid (expect same values flat)
        then do kriging of 3 time-series out to the grid (expect different values, no real verification here since this is done elsewhere

        """
        # arrange the test with a btk_parameter, a source grid and a destination grid
        ok_parameter = api.OKParameter(c=1.0,a=10.0*1000.0,cov_type=api.OKCovarianceType.EXPONENTIAL,z_scale=1.0)
        fx = lambda z: api.DoubleVector.from_numpy(np.zeros(self.n))

        grid_1km_1 = self._create_geo_point_grid(self.mnx, self.mny, self.dx_model)
        grid_1km_3 = self._create_geo_point_grid(self.mnx, self.mny, self.dx_model)

        observation_sites = api.GeoPointSourceVector()
        ta_obs = api.TimeAxisFixedDeltaT(self.t, self.d * 3, int(self.n / 3))
        ta_grid = api.TimeAxisFixedDeltaT(self.t, self.d, self.n)
        linear = api.point_interpretation_policy.POINT_INSTANT_VALUE
        ts_site_1 = api.TimeSeries(ta_obs, values=api.DoubleVector.from_numpy(
            (1.0) + 0.1 * np.sin(np.arange(start=0, stop=ta_obs.size(), step=1) * 2 * np.pi / 8.0 - np.pi / 2.0)), point_fx=linear)
        ts_site_2 = api.TimeSeries(ta_obs, values=api.DoubleVector.from_numpy(
            (0.8) + 0.2 * np.sin(np.arange(start=0, stop=ta_obs.size(), step=1) * 2 * np.pi / 8.0 - np.pi / 2.0)), point_fx=linear)
        ts_site_3 = api.TimeSeries(ta_obs, values=api.DoubleVector.from_numpy(
            (1.2) + 0.1 * np.sin(np.arange(start=0, stop=ta_obs.size(), step=1) * 2 * np.pi / 8.0 - np.pi / 2.0)), point_fx=linear)

        observation_sites.append(api.GeoPointSource(api.GeoPoint(50.0, 50.0, 5.0), ts_site_1))

        # act 1: just one time-series put into the system, should give same ts (true-averaged) in all the grid-1km_ts (which can be improved using std.gradient..)
        grid_1km_1ts = api.ordinary_kriging(observation_sites, grid_1km_1, ta_grid, ok_parameter)

        # assert 1:
        self.assertEqual(len(grid_1km_1ts), self.mnx * self.mny)
        expected_grid_1ts_values = ts_site_1.average(api.TimeAxis(ta_grid)).values.to_numpy()

        for gts in grid_1km_1ts:
            self.assertEqual(gts.ts.size(), ta_grid.size())
            self.assertTrue(np.allclose(expected_grid_1ts_values, gts.ts.values.to_numpy()))

        observation_sites.append(api.GeoPointSource(api.GeoPoint(9000.0, 500.0, 500), ts_site_2))
        observation_sites.append(api.GeoPointSource(api.GeoPoint(9000.0, 12000.0, 1050.0), ts_site_3))
        ok_parameter.cov_type= api.OKCovarianceType.GAUSSIAN  # just to switch covariance formula
        grid_1km_3ts = api.ordinary_kriging(observation_sites, grid_1km_3, ta_grid, ok_parameter)

        self.assertEqual(len(grid_1km_3ts), self.mnx * self.mny)

        for gts in grid_1km_3ts:
            self.assertEqual(gts.ts.size(), ta_grid.size())
            self.assertFalse(np.allclose(expected_grid_1ts_values, gts.ts.values.to_numpy()))


    def test_idw_temperature_transform_from_set_to_grid(self):
        """
        Test IDW interpolation transforms temperature time-series according to time-axis and range.

        """
        idw_p = api.IDWTemperatureParameter()
        self.assertEqual(idw_p.max_distance, 200000)
        self.assertEqual(idw_p.max_members, 20)
        fx = lambda z : [15 for x in range(self.n)]
        arome_grid = self._create_geo_ts_grid(self.nx, self.ny, self.dx_arome, fx)
        dest_grid_points = self._create_geo_point_grid(self.mnx, self.mny, self.dx_model)
        ta = api.TimeAxisFixedDeltaT(self.t, self.d * 3, int(self.n / 3))
        dest_grid = api.idw_temperature(arome_grid, dest_grid_points, ta, idw_p)
        self.assertIsNotNone(dest_grid)
        self.assertEqual(len(dest_grid), self.mnx * self.mny)

    def test_idw_precipitation_transform_from_set_to_grid(self):
        """
        Test IDW interpolation transforms precipitation time-series according to time-axis and range.

        """
        idw_p = api.IDWPrecipitationParameter()
        self.assertEqual(idw_p.max_distance, 200000)
        self.assertEqual(idw_p.max_members, 20)
        fx = lambda z : [15 for x in range(self.n)]
        arome_grid = self._create_geo_precipitation_grid(self.nx, self.ny, self.dx_arome, fx)
        dest_grid_points = self._create_geo_point_grid(self.mnx, self.mny, self.dx_model)
        ta = api.TimeAxisFixedDeltaT(self.t, self.d * 3, int(self.n / 3))
        dest_grid = api.idw_precipitation(arome_grid, dest_grid_points, ta, idw_p)
        self.assertIsNotNone(dest_grid)
        self.assertEqual(len(dest_grid), self.mnx * self.mny)

if __name__ == "__main__":
    unittest.main()
