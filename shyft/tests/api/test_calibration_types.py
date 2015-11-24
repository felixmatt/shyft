from shyft import api
from shyft.api import pt_gs_k
from shyft.api import pt_hs_k
import unittest
import numpy as np


class ShyftApi(unittest.TestCase):
    def test_pt_hs_k_param(self):
        pthsk = pt_hs_k.PTHSKParameter()
        self.assertIsNotNone(pthsk)
        self.assertEqual(pthsk.size(), 10)
        pthsk.snow.lw = 0.23
        self.assertAlmostEqual(pthsk.snow.lw, 0.23)
        snow = api.HbvSnowParameter(lw=0.2)  # keyword does work now!
        self.assertIsNotNone(snow)
        snow.lw = 0.2
        self.assertAlmostEqual(snow.lw, 0.2);
        # TODO: add more simple tests, especially since we for snow do have some extra features in the parameters

    def _create_std_ptgsk_param(self):
        ptp = api.PriestleyTaylorParameter(albedo=0.85, alpha=1.23)
        ptp.albedo = 0.9
        ptp.alpha = 1.26
        aep = api.ActualEvapotranspirationParameter(ae_scale_factor=1.5)
        aep.ae_scale_factor = 1.2
        gsp = api.GammaSnowParameter(winter_end_day_of_year=99, initial_bare_ground_fraction=0.04, snow_cv=0.44,
                                     tx=-0.3, wind_scale=1.9, wind_const=0.9, max_water=0.11, surface_magnitude=33.0,
                                     max_albedo=0.88, min_albedo=0.55, fast_albedo_decay_rate=6.0,
                                     slow_albedo_decay_rate=4.0, snowfall_reset_depth=6.1, glacier_albedo=0.44,
                                     calculate_iso_pot_energy=False)
        gsp.snow_cv = 0.5
        gsp.initial_bare_ground_fraction = 0.04
        kp = api.KirchnerParameter(c1=-2.55, c2=0.8, c3=-0.01)
        kp.c1 = 2.5
        kp.c2 = -0.9
        kp.c3 = 0.01
        spcp = api.PrecipitationCorrectionParameter(scale_factor=0.9)
        ptgsk_p = pt_gs_k.PTGSKParameter(ptp, gsp, aep, kp, spcp)
        return ptgsk_p

    def test_create_ptgsk_param(self):
        ptgsk_p = self._create_std_ptgsk_param()
        self.assertTrue(ptgsk_p != None, "should be possible to create a std param")

    def _create_std_geo_cell_data(self):
        geo_point = api.GeoPoint(1, 2, 3)
        ltf = api.LandTypeFractions()
        ltf.set_fractions(0.2, 0.2, 0.1, 0.3)
        geo_cell_data = api.GeoCellData(geo_point, 1000.0 ** 2, 0, 0.7, ltf)
        geo_cell_data.radiation_slope_factor = 0.7
        return geo_cell_data

    def test_create_ptgsk_grid_cells(self):
        geo_cell_data = self._create_std_geo_cell_data()
        param = self._create_std_ptgsk_param()
        cell_ts = [pt_gs_k.PTGSKCellAll, pt_gs_k.PTGSKCellOpt]
        for cell_t in cell_ts:
            c = cell_t()
            c.geo = geo_cell_data
            c.set_parameter(param)
            m = c.mid_point()
            self.assertTrue(m != None)
            c.set_state_collection(True)

    def test_create_region_environment(self):
        region_env = api.ARegionEnvironment()
        temp_vector = api.TemperatureSourceVector()
        region_env.temperature = temp_vector;
        self.assertTrue(region_env != None)

    def test_create_TargetSpecificationPts(self):
        t = api.TargetSpecificationPts();
        t.scale_factor = 1.0
        t.calc_mode = api.NASH_SUTCLIFFE
        t.calc_mode = api.KLING_GUPTA;
        t.s_r = 1.0  # KGEs scale-factors
        t.s_a = 2.0
        t.s_b = 3.0
        self.assertAlmostEqual(t.scale_factor, 1.0)
        # create a ts with some points
        cal = api.Calendar();
        start = cal.time(api.YMDhms(2015, 1, 1, 0, 0, 0))
        dt = api.deltahours(1)
        tsf = api.TsFactory();
        times = api.UtcTimeVector()
        times.push_back(start + 1 * dt);
        times.push_back(start + 3 * dt);
        times.push_back(start + 4 * dt)

        values = api.DoubleVector()
        values.push_back(1.0)
        values.push_back(3.0)
        values.push_back(np.nan)
        tsp = tsf.create_time_point_ts(api.UtcPeriod(start, start + 24 * dt), times, values)
        # convert it from a time-point ts( as returned from current smgrepository) to a fixed interval with timeaxis, needed by calibration
        tst = api.TsTransform()
        tsa = tst.to_average(start, dt, 24, tsp)
        # tsa2 = tst.to_average(start,dt,24,tsp,False)
        # tsa_staircase = tst.to_average_staircase(start,dt,24,tsp,False) # nans infects the complete interval to nan
        # tsa_staircase2 = tst.to_average_staircase(start,dt,24,tsp,True) # skip nans, nans are 0
        # stuff it into the target spec.
        t2 = api.TargetSpecificationPts(tsa, api.IntVector([0, 2, 3]), 0.7, api.KLING_GUPTA)
        t.ts = tsa
        tv = api.TargetSpecificationVector([t, t2])
        # now verify we got something ok
        self.assertEqual(2, tv.size())
        self.assertAlmostEqual(tv[0].ts.value(1), 1.5)  # average value 0..1 ->0.5
        self.assertAlmostEqual(tv[0].ts.value(2), 2.5)  # average value 0..1 ->0.5
        self.assertAlmostEqual(tv[0].ts.value(3), 3.0)  # average value 0..1 ->0.5
        # and that the target vector now have its own copy of ts
        tsa.set(1, 3.0)
        self.assertAlmostEqual(tv[0].ts.value(1), 1.5)  # make sure the ts passed onto target spec, is a copy
        self.assertAlmostEqual(tsa.value(1), 3.0)  # and that we really did change the source

    def test_IntVector(self):
        v1 = api.IntVector()  # empy
        v2 = api.IntVector([i for i in range(10)])  # by list
        v3 = api.IntVector([1, 2, 3])  # simple list
        self.assertEqual(v2.size(), 10)
        self.assertEqual(v1.size(), 0)
        self.assertEqual(len(v3), 3)

    def test_DoubleVector(self):
        v1 = api.DoubleVector([i for i in range(10)])  # empy
        v2 = api.DoubleVector.FromNdArray(np.arange(0, 10.0, 0.5))
        v3 = api.DoubleVector(np.arange(0, 10.0, 0.5))
        self.assertEqual(len(v1), 10)
        self.assertEqual(len(v2), 20)
        self.assertEqual(len(v3), 20)
        self.assertAlmostEqual(v2[3], 1.5)


if __name__ == "__main__":
    unittest.main()
