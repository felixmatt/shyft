from shyft.api import GammaSnowParameter, GammaSnowState, GammaSnowCalculator, GammaSnowResponse
import unittest
from shyft.api import deltahours
from shyft.api import Calendar

class GammaSnow(unittest.TestCase):

    def test_gamma_snow_parameter_sig1(self):
        p = GammaSnowParameter(winter_end_day_of_year=100, 
                               initial_bare_ground_fraction=0.0, 
                               snow_cv=0.21, tx=-0.32, wind_scale=2.3, 
                               wind_const=1.4, max_water=0.23, 
                               surface_magnitude=30.0, max_albedo=0.73,
                               min_albedo=0.4, fast_albedo_decay_rate=5.1,
                               slow_albedo_decay_rate=5.3,
                               snowfall_reset_depth=5.12,
                               glacier_albedo=0.23)
        self.assertAlmostEqual(p.winter_end_day_of_year, 100)
        self.assertAlmostEqual(p.initial_bare_ground_fraction, 0.0)
        self.assertAlmostEqual(p.snow_cv, 0.21)
        self.assertAlmostEqual(p.tx, -0.32)
        self.assertAlmostEqual(p.wind_scale, 2.3)
        self.assertAlmostEqual(p.wind_const, 1.4)
        self.assertAlmostEqual(p.max_water, 0.23)
        self.assertAlmostEqual(p.surface_magnitude, 30.0)
        self.assertAlmostEqual(p.max_albedo, 0.73)
        self.assertAlmostEqual(p.min_albedo, 0.4)
        self.assertAlmostEqual(p.fast_albedo_decay_rate, 5.1)
        self.assertAlmostEqual(p.slow_albedo_decay_rate, 5.3)
        self.assertAlmostEqual(p.snowfall_reset_depth, 5.12)
        self.assertAlmostEqual(p.glacier_albedo, 0.23)


    def test_gamma_snow_state(self):
        s = GammaSnowState(albedo=0.32, lwc=0.01, surface_heat=30003.0,
                         alpha=1.23, sdc_melt_mean=1.2, acc_melt=0.45,
                         iso_pot_energy=0.1, temp_swe=1.2)
        self.assertAlmostEqual(s.albedo, 0.32)
        self.assertAlmostEqual(s.lwc, 0.01)
        self.assertAlmostEqual(s.surface_heat, 30003.0)
        self.assertAlmostEqual(s.alpha, 1.23)
        self.assertAlmostEqual(s.sdc_melt_mean, 1.2)
        self.assertAlmostEqual(s.acc_melt, 0.45)
        self.assertAlmostEqual(s.iso_pot_energy, 0.1)
        self.assertAlmostEqual(s.temp_swe, 1.2)


    def test_gamma_snow_step(self):
        utc = Calendar()
        s = GammaSnowState()
        p = GammaSnowParameter()
        r = GammaSnowResponse()
        calc = GammaSnowCalculator()
        t = utc.time(2016,10,1)
        dt = deltahours(1)
        temp = 0.4
        rad = 12.0
        prec_mm_h = 0.3
        wind_speed=1.3
        rel_hum=0.4
        forest_fraction = 0.2
        altitude = 100.0
        # Just check that we don't get an error when stepping
        calc.step(s, r, t, dt, p, temp, rad, prec_mm_h, wind_speed, rel_hum, forest_fraction, altitude)
