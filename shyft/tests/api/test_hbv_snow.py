from shyft.api import HbvSnowParameter, HbvSnowState, HbvSnowCalculator, HbvSnowResponse
import unittest
from shyft.api import deltahours
from shyft.api import Calendar


class HbvSnow(unittest.TestCase):

    def test_hbv_snow_parameter_sig1(self):
        p = HbvSnowParameter(tx=0.0, cx=1.0, ts=0.0, lw=0.1, cfr=0.5)
        self.assertAlmostEqual(p.tx, 0.0)
        self.assertAlmostEqual(p.cx, 1.0)
        self.assertAlmostEqual(p.ts, 0.0)
        self.assertAlmostEqual(p.lw, 0.1)
        self.assertAlmostEqual(p.cfr, 0.5)

    def test_hbv_snow_parameter_sig2(self):
        p = HbvSnowParameter([1.0, 2.0, 3.0, 4.0, 5.0],
                             [0, 0.5, 0.6, 0.8, 1.0],
                             4.3, 5.1, 2.4, 4.0, 0.1)
        for (el, comp) in zip(p.s, [1.0, 2.0, 3.0, 4.0, 5.0]):
            self.assertAlmostEqual(el, comp)

        for (el, comp) in zip(p.intervals, [0, 0.5, 0.6, 0.8, 1.0]):
            self.assertAlmostEqual(el, comp)

    def test_hbv_snow_state(self):
        s = HbvSnowState(1000.0, 0.7)
        self.assertAlmostEqual(s.swe, 1000.0)
        self.assertAlmostEqual(s.sca, 0.7)

    def test_hbv_snow_step(self):
        utc = Calendar()
        s = HbvSnowState()
        p = HbvSnowParameter()
        s.distribute(p)
        r = HbvSnowResponse()
        calc = HbvSnowCalculator(p)
        t0 = utc.time(2016, 10, 1)
        t1 = utc.time(2016, 10, 2)
        dt = deltahours(1)
        temp = 0.4
        prec_mm_h = 0.3
        # Just check that we don't get an error when stepping
        calc.step(s, r, t0, t1, prec_mm_h, temp)
