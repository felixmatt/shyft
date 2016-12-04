import unittest

from shyft.api import Calendar
from shyft.api import DoubleVector
from shyft.api import Timeaxis
from shyft.api import Timeseries
from shyft.api import convolve_policy
from shyft.api import deltahours
from shyft.api import point_interpretation_policy as point_fx
from shyft.api import River
from shyft.api import RoutingInfo
from shyft.api import UHGParameter
from shyft.api import RiverNetwork


class Routing(unittest.TestCase):
    """Verify and illustrate the Routing classes


     """
    def test_routing_info(self):
        ri= RoutingInfo(2,1000.0)
        self.assertIsNotNone(ri)
        self.assertEqual(ri.id,2)
        self.assertAlmostEqual(ri.distance,1000.0)
        ri.distance=2000.0
        self.assertAlmostEqual(ri.distance, 2000.0)

    def test_unit_hydrograph_parameter(self):
        p = UHGParameter()
        self.assertIsNotNone(p)
        self.assertAlmostEqual(p.alpha,3.0)  # default values
        self.assertAlmostEqual(p.beta,0.7)
        self.assertAlmostEqual(p.velocity,1.0)
        p.alpha=2.7
        p.beta=0.77
        p.velocity = 1/3600.0
        self.assertAlmostEqual(p.alpha,2.7)
        self.assertAlmostEqual(p.beta,0.77)
        self.assertAlmostEqual(p.velocity,1.0/3600.0)

    def test_river(self):
        r1 = River(1)
        self.assertIsNotNone(r1)
        self.assertEqual(r1.id, 1)
        r2 = River(2,RoutingInfo(3,1000.0),UHGParameter(1/3600.0,1.0,0.7))
        self.assertEqual(r2.id, 2)
        r3 = River(3,RoutingInfo(id=1,distance=36000.00))
        self.assertEqual(r3.id,3)
        self.assertEqual(r3.downstream.id,1)
        self.assertEqual(r3.downstream.distance, 36000.0)

        r3_uhg = r3.uhg(deltahours(1))
        self.assertEqual(len(r3_uhg), 10)
        r2.parameter.alpha = 2.0
        r2.parameter.beta = 0.99
        r2.parameter.velocity = 1/3600.0
        self.assertAlmostEqual(r2.parameter.alpha, 2.0)
        self.assertAlmostEqual(r2.parameter.beta, 0.99)
        self.assertAlmostEqual(r2.parameter.velocity,1/3600.0)
        r2.downstream = RoutingInfo(2,2000.0)
        self.assertEqual(r2.downstream.id,2)
        self.assertAlmostEqual(r2.downstream.distance, 2000.0)
        # not possible, read only: r2.id = 3