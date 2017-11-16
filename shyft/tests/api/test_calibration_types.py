import math
import numpy as np
import unittest

from shyft import api
from shyft.api import pt_gs_k
from shyft.api import pt_hs_k
from shyft.api import pt_ss_k
from shyft.api import hbv_stack


class ShyftApi(unittest.TestCase):
    """
    Verify basic SHyFT api calibration related functions and structures
    """

    def verify_parameter_for_calibration(self, param, expected_size, valid_names):
        min_p_value = -1e+10
        max_p_value = +1e+10
        self.assertEqual(expected_size, param.size(), "expected parameter size changed")
        pv = api.DoubleVector([param.get(i) for i in range(param.size())])
        for i in range(param.size()):
            v = param.get(i)
            self.assertTrue(v > min_p_value and v < max_p_value)
            pv[i] = v * 1.01
            param.set(pv)  # set the complete vector, only used during C++ calibration, but we verify it here
            x = param.get(i)
            self.assertAlmostEqual(v * 1.01, x, 3, "Expect new value when setting value")
            p_name = param.get_name(i)
            self.assertTrue(len(p_name) > 0, "parameter name should exist")
            self.assertEqual(valid_names[i], p_name)

    def test_pt_hs_k_param(self):
        pthsk_size = 16
        pthsk = pt_hs_k.PTHSKParameter()
        self.assertIsNotNone(pthsk)
        self.assertEqual(pthsk.size(), pthsk_size)
        pthsk.hs.lw = 0.23
        self.assertAlmostEqual(pthsk.hs.lw, 0.23)
        snow = api.HbvSnowParameter(
            tx=0.2)  # ordered .. keyword does work now! TODO: verify if we can have boost provide real kwargs
        self.assertIsNotNone(snow)
        snow.lw = 0.2
        self.assertAlmostEqual(snow.lw, 0.2)
        valid_names = [
            "kirchner.c1",
            "kirchner.c2",
            "kirchner.c3",
            "ae.ae_scale_factor",
            "hs.lw",
            "hs.tx",
            "hs.cx",
            "hs.ts",
            "hs.cfr",
            "gm.dtf",
            "p_corr.scale_factor",
            "pt.albedo",
            "pt.alpha",
            "routing.velocity",
            "routing.alpha",
            "routing.beta"
        ]
        self.verify_parameter_for_calibration(pthsk, pthsk_size, valid_names)

    def test_hbv_stack_param(self):
        hbv_size = 20
        hbv = hbv_stack.HbvParameter()
        self.assertIsNotNone(hbv)
        self.assertEqual(hbv.size(), hbv_size)
        valid_names = [
            "soil.fc",
            "soil.beta",
            "ae.lp",
            "tank.uz1",
            "tank.kuz2",
            "tank.kuz1",
            "tank.perc",
            "tank.klz",
            "hs.lw",
            "hs.tx",
            "hs.cx",
            "hs.ts",
            "hs.cfr",
            "p_corr.scale_factor",
            "pt.albedo",
            "pt.alpha",
            "gm.dtf",
            "routing.velocity",
            "routing.alpha",
            "routing.beta"
        ]
        self.verify_parameter_for_calibration(hbv, hbv_size, valid_names)

    def test_pt_gs_k_param(self):
        ptgsk_size = 28
        valid_names = [
            "kirchner.c1",
            "kirchner.c2",
            "kirchner.c3",
            "ae.ae_scale_factor",
            "gs.tx",
            "gs.wind_scale",
            "gs.max_water",
            "gs.wind_const",
            "gs.fast_albedo_decay_rate",
            "gs.slow_albedo_decay_rate",
            "gs.surface_magnitude",
            "gs.max_albedo",
            "gs.min_albedo",
            "gs.snowfall_reset_depth",
            "gs.snow_cv",
            "gs.glacier_albedo",
            "p_corr.scale_factor",
            "gs.snow_cv_forest_factor",
            "gs.snow_cv_altitude_factor",
            "pt.albedo",
            "pt.alpha",
            "gs.initial_bare_ground_fraction",
            "gs.winter_end_day_of_year",
            "gs.calculate_iso_pot_energy",
            "gm.dtf",
            "routing.velocity",
            "routing.alpha",
            "routing.beta"
        ]
        p = pt_gs_k.PTGSKParameter()
        self.verify_parameter_for_calibration(p, ptgsk_size, valid_names)
        # special verification of bool parameter
        p.gs.calculate_iso_pot_energy = True
        self.assertTrue(p.gs.calculate_iso_pot_energy)
        self.assertAlmostEqual(p.get(23), 1.0, 0.00001)
        p.gs.calculate_iso_pot_energy = False
        self.assertFalse(p.gs.calculate_iso_pot_energy)
        self.assertAlmostEqual(p.get(23), 0.0, 0.00001)
        pv = api.DoubleVector.from_numpy([p.get(i) for i in range(p.size())])
        pv[23] = 1.0
        p.set(pv)
        self.assertTrue(p.gs.calculate_iso_pot_energy)
        pv[23] = 0.0;
        p.set(pv)
        self.assertFalse(p.gs.calculate_iso_pot_energy)
        # checkout new parameters for routing
        p.routing.velocity = 1 / 3600.0
        p.routing.alpha = 1.1
        p.routing.beta = 0.8
        self.assertAlmostEqual(p.routing.velocity, 1 / 3600.0)
        self.assertAlmostEqual(p.routing.alpha, 1.1)
        self.assertAlmostEqual(p.routing.beta, 0.8)

    def test_pt_ss_k_param(self):
        ptssk_size = 19
        valid_names = [
            "kirchner.c1",
            "kirchner.c2",
            "kirchner.c3",
            "ae.ae_scale_factor",
            "ss.alpha_0",
            "ss.d_range",
            "ss.unit_size",
            "ss.max_water_fraction",
            "ss.tx",
            "ss.cx",
            "ss.ts",
            "ss.cfr",
            "p_corr.scale_factor",
            "pt.albedo",
            "pt.alpha",
            "gm.dtf",
            "routing.velocity",
            "routing.alpha",
            "routing.beta"
        ]
        self.verify_parameter_for_calibration(pt_ss_k.PTSSKParameter(), ptssk_size, valid_names)

    def _create_std_ptgsk_param(self):
        ptp = api.PriestleyTaylorParameter(albedo=0.85, alpha=1.23)
        ptp.albedo = 0.9
        ptp.alpha = 1.26
        aep = api.ActualEvapotranspirationParameter(ae_scale_factor=1.5)
        aep.ae_scale_factor = 1.1
        gsp = api.GammaSnowParameter(winter_end_day_of_year=99, initial_bare_ground_fraction=0.04, snow_cv=0.44,
                                     tx=-0.3, wind_scale=1.9, wind_const=0.9, max_water=0.11, surface_magnitude=33.0,
                                     max_albedo=0.88, min_albedo=0.55, fast_albedo_decay_rate=6.0,
                                     slow_albedo_decay_rate=4.0, snowfall_reset_depth=6.1, glacier_albedo=0.44
                                     )  # TODO: This does not work due to boost.python template arity of 15,  calculate_iso_pot_energy=False)
        gsp.calculate_iso_pot_energy = False
        gsp.snow_cv = 0.5
        gsp.initial_bare_ground_fraction = 0.04
        kp = api.KirchnerParameter(c1=-2.55, c2=0.8, c3=-0.01)
        kp.c1 = 2.5
        kp.c2 = -0.9
        kp.c3 = 0.01
        spcp = api.PrecipitationCorrectionParameter(scale_factor=0.9)
        gm = api.GlacierMeltParameter(dtf=5.9)  # verify we can construct glacier parameter
        ptgsk_p = pt_gs_k.PTGSKParameter(ptp, gsp, aep, kp, spcp, gm)  # passing optional gm parameter here
        ptgsk_p.ae.ae_scale_factor = 1.2  # sih: just to demo ae scale_factor can be set directly
        return ptgsk_p

    def test_precipitation_correction_constructor(self):
        spcp = api.PrecipitationCorrectionParameter(scale_factor=0.9)
        self.assertAlmostEqual(0.9, spcp.scale_factor)

    def test_create_ptgsk_param(self):
        ptgsk_p = self._create_std_ptgsk_param()
        copy_p = pt_gs_k.PTGSKParameter(ptgsk_p)
        self.assertTrue(ptgsk_p != None, "should be possible to create a std param")
        self.assertIsNotNone(copy_p)

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
            self.assertTrue(m is not None)
            c.set_state_collection(True)

    def test_create_region_environment(self):
        region_env = api.ARegionEnvironment()
        temp_vector = api.TemperatureSourceVector()
        region_env.temperature = temp_vector
        self.assertTrue(region_env is not None)

    def test_create_TargetSpecificationPts(self):
        t = api.TargetSpecificationPts()
        t.scale_factor = 1.0
        t.calc_mode = api.NASH_SUTCLIFFE
        t.calc_mode = api.KLING_GUPTA
        t.calc_mode = api.ABS_DIFF
        t.calc_mode = api.RMSE
        t.s_r = 1.0  # KGEs scale-factors
        t.s_a = 2.0
        t.s_b = 3.0
        self.assertIsNotNone(t.uid)
        t.uid = 'test'
        self.assertEqual(t.uid, 'test')
        self.assertAlmostEqual(t.scale_factor, 1.0)
        # create a ts with some points
        cal = api.Calendar()
        start = cal.time(api.YMDhms(2015, 1, 1, 0, 0, 0))
        dt = api.deltahours(1)
        tsf = api.TsFactory()
        times = api.UtcTimeVector()
        times.push_back(start + 1 * dt)
        times.push_back(start + 3 * dt)
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
        # also show how to specify snow-calibration
        cids = api.IntVector([0, 2, 3])
        t2 = api.TargetSpecificationPts(tsa, cids, 0.7, api.KLING_GUPTA, 1.0, 1.0, 1.0, api.SNOW_COVERED_AREA,
                                        'test_uid')
        self.assertEqual(t2.uid, 'test_uid')
        t2.catchment_property = api.SNOW_WATER_EQUIVALENT
        self.assertEqual(t2.catchment_property, api.SNOW_WATER_EQUIVALENT)
        t2.catchment_property = api.CELL_CHARGE
        self.assertEqual(t2.catchment_property, api.CELL_CHARGE)
        self.assertIsNotNone(t2.catchment_indexes)
        for i in range(len(cids)):
            self.assertEqual(cids[i], t2.catchment_indexes[i])
        t.ts = api.TimeSeries(tsa)  # target spec is now a regular TimeSeries
        tv = api.TargetSpecificationVector()
        tv[:] = [t, t2]
        # now verify we got something ok
        self.assertEqual(2, tv.size())
        self.assertAlmostEqual(tv[0].ts.value(1), 1.5)  # average value 0..1 ->0.5
        self.assertAlmostEqual(tv[0].ts.value(2), 2.5)  # average value 0..1 ->0.5
        # self.assertAlmostEqual(tv[0].ts.value(3), 3.0)  # original flat out at end, but now:
        self.assertTrue(math.isnan(tv[0].ts.value(3)))  # strictly linear between points.
        # and that the target vector now have its own copy of ts
        tsa.set(1, 3.0)
        self.assertAlmostEqual(tv[0].ts.value(1), 1.5)  # make sure the ts passed onto target spec, is a copy
        self.assertAlmostEqual(tsa.value(1), 3.0)  # and that we really did change the source
        # Create a clone of target specification vector
        tv2 = api.TargetSpecificationVector(tv)
        self.assertEqual(2, tv2.size())
        self.assertAlmostEqual(tv2[0].ts.value(1), 1.5)  # average value 0..1 ->0.5
        self.assertAlmostEqual(tv2[0].ts.value(2), 2.5)  # average value 0..1 ->0.5
        self.assertTrue(math.isnan(tv2[0].ts.value(3)))  # average value 0..1 ->0.5
        tv2[0].scale_factor = 10.0
        self.assertAlmostEqual(tv[0].scale_factor, 1.0)
        self.assertAlmostEqual(tv2[0].scale_factor, 10.0)
        # test we can create from breakpoint time-series
        ts_bp = api.TimeSeries(
            api.TimeAxis(api.UtcTimeVector([0, 25, 20]), 30),
            fill_value=2.0,
            point_fx=api.POINT_AVERAGE_VALUE
        )

        tspec_bp = api.TargetSpecificationPts(
            ts_bp,
            cids, 0.7, api.KLING_GUPTA, 1.0, 1.0, 1.0, api.CELL_CHARGE,
            'test_uid'
        )
        self.assertIsNotNone(tspec_bp)

    def test_create_target_spec_from_std_time_series(self):
        """
        Verify we can create target-spec giving ordinary ts,
        and that passing a non-fixed time-axis raises exception

        """
        cal = api.Calendar()
        ta = api.TimeAxis(cal.time(2017, 1, 1), api.deltahours(1), 24)
        ts = api.TimeSeries(ta, fill_value=3.0, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)
        cids = api.IntVector([0, 2, 3])
        t0 = api.TargetSpecificationPts(ts, cids, 0.7, api.KLING_GUPTA, 1.0, 1.0, 1.0, api.SNOW_COVERED_AREA,
                                        'test_uid')
        self.assertAlmostEqual(t0.ts.value(0), ts.value(0))
        rid = 0
        t1 = api.TargetSpecificationPts(ts, rid, 0.7, api.KLING_GUPTA, 1.0, 1.0, 1.0, 'test_uid')
        self.assertAlmostEqual(t1.ts.value(0), ts.value(0))
        tax = api.TimeAxis(api.UtcTimeVector.from_numpy(ta.time_points[:-1]), ta.total_period().end)
        tsx = api.TimeSeries(tax, fill_value=2.0, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)
        tx = api.TargetSpecificationPts(tsx, rid, 0.7, api.KLING_GUPTA, 1.0, 1.0, 1.0, 'test_uid')
        self.assertIsNotNone(tx)

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
