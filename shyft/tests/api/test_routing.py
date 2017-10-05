import unittest

from shyft.api import deltahours
from shyft.api import River
from shyft.api import RoutingInfo
from shyft.api import UHGParameter
from shyft.api import RiverNetwork


class Routing(unittest.TestCase):
    """Verify and illustrate the building of Routing classes
    """

    def test_routing_info(self):
        ri = RoutingInfo(2, 1000.0)
        self.assertIsNotNone(ri)
        self.assertEqual(ri.id, 2)
        self.assertAlmostEqual(ri.distance, 1000.0)
        ri.distance = 2000.0
        self.assertAlmostEqual(ri.distance, 2000.0)

    def test_unit_hydrograph_parameter(self):
        p = UHGParameter()
        self.assertIsNotNone(p)
        self.assertAlmostEqual(p.alpha, 7.0)  # default values
        self.assertAlmostEqual(p.beta, 0.0)
        self.assertAlmostEqual(p.velocity, 1.0)
        p.alpha = 2.7
        p.beta = 0.07
        p.velocity = 1 / 3600.0
        self.assertAlmostEqual(p.alpha, 2.7)
        self.assertAlmostEqual(p.beta, 0.07)
        self.assertAlmostEqual(p.velocity, 1.0 / 3600.0)

    def test_river(self):
        r1 = River(1)
        self.assertIsNotNone(r1)
        self.assertEqual(r1.id, 1)
        r2 = River(2, RoutingInfo(3, 1000.0), UHGParameter(1 / 3600.0, 7.0, 0.0))
        self.assertEqual(r2.id, 2)
        r3 = River(3, RoutingInfo(id=1, distance=36000.00))
        self.assertEqual(r3.id, 3)
        self.assertEqual(r3.downstream.id, 1)
        self.assertEqual(r3.downstream.distance, 36000.0)

        r3_uhg = r3.uhg(deltahours(1))
        self.assertEqual(len(r3_uhg), 10)
        r2.parameter.alpha = 2.0
        r2.parameter.beta = 0.00
        r2.parameter.velocity = 1 / 3600.0
        self.assertAlmostEqual(r2.parameter.alpha, 2.0)
        self.assertAlmostEqual(r2.parameter.beta, 0.00)
        self.assertAlmostEqual(r2.parameter.velocity, 1 / 3600.0)
        r2.downstream = RoutingInfo(2, 2000.0)
        self.assertEqual(r2.downstream.id, 2)
        self.assertAlmostEqual(r2.downstream.distance, 2000.0)
        # not possible, read only: r2.id = 3

    def test_river_network(self):
        rn = RiverNetwork()
        self.assertIsNotNone(rn)
        rn.add(River(1))
        rn.add(River(2))  # important detail, #2 must be added before referred
        rn.add(River(3, RoutingInfo(2, 1000.0), UHGParameter(1 / 3600.0, 7.0, 0.0)))
        rn.set_downstream_by_id(1, 2)
        # already done as pr. constuction above: rn.set_downstream_by_id(3,2)
        rn.add(River(4))
        rn.set_downstream_by_id(2, 4)
        self.assertEqual(len(rn.upstreams_by_id(4)), 1)
        self.assertEqual(len(rn.upstreams_by_id(2)), 2)
        self.assertEqual(len(rn.upstreams_by_id(1)), 0)
        self.assertEqual(rn.downstream_by_id(1), 2)
        self.assertEqual(rn.downstream_by_id(3), 2)
        up2 = rn.upstreams_by_id(2)
        self.assertIn(1, up2)
        self.assertIn(3, up2)
        rn.add(River(6))
        rn.set_downstream_by_id(6, 1)
        rn.remove_by_id(1)
        self.assertEqual(rn.downstream_by_id(6), 0)  # auto fix references ok
        rn_clone = RiverNetwork(rn)
        self.assertIsNotNone(rn_clone)
        rn_clone.add(River(1, RoutingInfo(2, 1000.0)))
        self.assertEqual(len(rn_clone.upstreams_by_id(2)), 2)
        self.assertEqual(len(rn.upstreams_by_id(2)), 1)  # still just one in original netw.
        r3 = rn.river_by_id(3)
        r3.downstream.distance = 1234.0  # got a reference We can modify
        r3b = rn.river_by_id(3)  # pull out the reference once again
        self.assertAlmostEqual(r3b.downstream.distance, 1234.0)  # verify its modified
