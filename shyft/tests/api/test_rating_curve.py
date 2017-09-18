from shyft import api as sa
import numpy as np
import math
from numpy.testing import assert_array_almost_equal
import unittest


class RatingCurveTest(unittest.TestCase):

    def test_rating_curve_segment(self):
        lower = 0.0
        a = 1.0
        b = 2.0
        c = 3.0
        rcs = sa.RatingCurveSegment(lower=lower, a=a, b=b, c=c)
        self.assertEqual(rcs.lower, lower)
        self.assertAlmostEqual(rcs.a, a)
        self.assertAlmostEqual(rcs.b, b)
        self.assertAlmostEqual(rcs.c, c)
        for level in range(10):
            self.assertAlmostEqual(rcs.flow(level), a*pow(level-b, c))
        flows = rcs.flow([i for i in range(10)])
        for i in range(10):
            self.assertAlmostEqual(flows[i], a*pow(float(i)-b, c))

    def test_rating_curve_function(self):
        rcf = sa.RatingCurveFunction()
        self.assertEqual(rcf.size(), 0)
        lower = 0.0
        a = 1.0
        b = 2.0
        c = 3.0
        rcs = sa.RatingCurveSegment(lower=lower, a=a, b=b, c=c)
        rcf.add_segment(rcs)
        rcf.add_segment(lower+10.0, a, b, c)
        self.assertEqual(rcf.size(), 2)
        self.assertAlmostEqual(rcf.flow(4.0), 8.0)
        self.assertAlmostEqual(rcf.flow([4.0])[0], 8.0)
        s = str(rcf)  # just to check that str works
        self.assertGreater(len(s), 10)
        sum_levels = 0.0
        for rcs in rcf:  # demo iterable
            sum_levels += rcs.lower
        self.assertAlmostEqual(sum_levels,lower+ 10.0)