from shyft.api import HbvPhysicalSnowParameter, HbvPhysicalSnowState, HbvPhysicalSnowCalculator, HbvPhysicalSnowResponse
import unittest
from shyft.api import deltahours
from shyft.api import Calendar


class HbvPhysicalSnow(unittest.TestCase):

    def test_hbv_physical_snow_parameter_sig1(self):
        p = HbvPhysicalSnowParameter(tx=0.0, lw=0.1, cfr=0.5,
                                     wind_scale=3.0, wind_const=1.3,
                                     surface_magnitude=10.2, max_albedo=0.9,
                                     min_albedo=0.5, fast_albedo_decay_rate=4.5,
                                     slow_albedo_decay_rate=3.6, 
                                     snowfall_reset_depth=1.3, 
                                     calculate_iso_pot_energy=False)

        self.assertAlmostEqual(p.tx, 0.0)
        self.assertAlmostEqual(p.lw, 0.1)
        self.assertAlmostEqual(p.cfr, 0.5)
        self.assertAlmostEqual(p.wind_scale, 3.0)
        self.assertAlmostEqual(p.wind_const, 1.3)
        self.assertAlmostEqual(p.surface_magnitude, 10.2)
        self.assertAlmostEqual(p.max_albedo, 0.9)
        self.assertAlmostEqual(p.min_albedo, 0.5)
        self.assertAlmostEqual(p.fast_albedo_decay_rate, 4.5)
        self.assertAlmostEqual(p.slow_albedo_decay_rate, 3.6)
        self.assertAlmostEqual(p.snowfall_reset_depth, 1.3)
        self.assertEqual(p.calculate_iso_pot_energy, False)


    def test_hbv_physical_snow_parameter_sig2(self):
        p = HbvPhysicalSnowParameter(snow_redist_factors=[2.0, 2.0, 2.0],
                                     quantiles= [0.0, 0.5, 1.0])
        for (el, comp) in zip(p.s, [1.0, 1.0, 1.0]):  # automatic normalization to 1.0
            self.assertAlmostEqual(el, comp)

        for (el, comp) in zip(p.intervals, [0.0, 0.5, 1.0]):  # should equal quantiles
            self.assertAlmostEqual(el, comp)


    def test_hbv_physical_snow_state(self):
        s = HbvPhysicalSnowState([0.3, 0.4, 0.5],
                                 [1.3, 2.0, 3.1],
                                 3200.0,
                                 1000.0,
                                 0.7)
        self.assertAlmostEqual(s.surface_heat, 3200.0)
        self.assertAlmostEqual(s.swe, 1000.0)
        self.assertAlmostEqual(s.sca, 0.7)
        for (el, comp) in zip(s.albedo, [0.3, 0.4, 0.5]):
            self.assertAlmostEqual(el, comp)
        for (el, comp) in zip(s.iso_pot_energy, [1.3, 2.0, 3.1]):
            self.assertAlmostEqual(el, comp)


    def test_hbv_physical_snow_step(self):
        utc = Calendar()
        s = HbvPhysicalSnowState()
        p = HbvPhysicalSnowParameter()
        r = HbvPhysicalSnowResponse()
        s.distribute(p)
        calc = HbvPhysicalSnowCalculator(p)
        t = utc.time(2016,10,1)
        dt = deltahours(1)
        temp = 0.4
        rad = 12.0
        prec_mm_h = 0.3
        wind_speed=1.3
        rel_hum=0.4
        # Just check that we don't get an error when stepping
        calc.step(s, r, t, dt, temp, rad, prec_mm_h, wind_speed, rel_hum)
