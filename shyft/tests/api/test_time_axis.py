from builtins import range
from shyft import api
import numpy as np
import unittest


class TimeAxis(unittest.TestCase):
    """Verify and illustrate TimeAxis
       defined as n periods non-overlapping ascending
        
     """

    def setUp(self):
        self.c = api.Calendar()
        self.d = api.deltahours(1)
        self.n = 24
        # self.t= self.c.trim(api.utctime_now(),self.d)
        self.t = self.c.trim(self.c.time(api.YMDhms(1969, 12, 31, 0, 0, 0)), self.d)
        self.ta = api.TimeAxis(self.t, self.d, self.n)

    def tearDown(self):
        pass

    def test_index_of(self):
        self.assertEqual(self.ta.index_of(self.t),0)
        self.assertEqual(self.ta.index_of(self.t,0), 0)
        self.assertEqual(self.ta.index_of(self.t-3600), api.npos)
        self.assertEqual(self.ta.open_range_index_of(self.t),0)
        self.assertEqual(self.ta.open_range_index_of(self.t,0), 0)
        self.assertEqual(self.ta.open_range_index_of(self.t-3600), api.npos)


    def test_create_timeaxis(self):
        self.assertEqual(self.ta.size(), self.n)
        self.assertEqual(len(self.ta), self.n)
        self.assertEqual(self.ta(0).start, self.t)
        self.assertEqual(self.ta(0).end, self.t + self.d)
        self.assertEqual(self.ta(1).start, self.t + self.d)
        self.assertEqual(self.ta.total_period().start, self.t)
        va = np.array([86400, 3600, 3], dtype=np.int64)
        xta = api.TimeAxisFixedDeltaT(int(va[0]), int(va[1]), int(va[2]))
        self.assertEqual(xta.size(), 3)

    def test_iterate_timeaxis(self):
        tot_dt = 0
        for p in self.ta:
            tot_dt += p.timespan()
        self.assertEqual(tot_dt, self.n * self.d)

    def test_timeaxis_str(self):
        s = str(self.ta)
        self.assertTrue(len(s) > 10)

    def test_point_timeaxis_(self):
        """ 
        A point time axis takes n+1 points do describe n-periods, where
        each period is defined as [ point_i .. point_i+1 >
        """
        all_points = api.UtcTimeVector([t for t in range(self.t, self.t + (self.n + 1) * self.d, self.d)])
        tap = api.PointTimeaxis(all_points)
        self.assertEqual(tap.size(), self.ta.size())
        for i in range(self.ta.size()):
            self.assertEqual(tap(i), self.ta(i))
        self.assertEqual(tap.t_end, all_points[-1], "t_end should equal the n+1'th point if supplied")
        s = str(tap)
        self.assertTrue(len(s) > 0)

    def test_generic_timeaxis(self):
        c = api.Calendar('Europe/Oslo')
        dt = api.deltahours(1)
        n = 240
        t0 = c.time(2016, 4, 10)

        tag1 = api.TimeAxis(t0, dt, n)
        self.assertEqual(len(tag1), n)
        self.assertEqual(tag1.time(0), t0)

        tag2 = api.TimeAxis(c, t0, dt, n)
        self.assertEqual(len(tag2), n)
        self.assertEqual(tag2.time(0), t0)
        self.assertIsNotNone(tag2.calendar_dt.calendar)

    def test_timeaxis_time_points(self):
        c = api.Calendar('Europe/Oslo')
        dt = api.deltahours(1)
        n = 240
        t0 = c.time(2016, 4, 10)
        ta = api.TimeAxis(c, t0, dt, n)
        tp = ta.time_points
        self.assertIsNotNone(tp)
        self.assertEqual(len(tp), n + 1)
        self.assertEqual(len(api.TimeAxis(c, t0, dt, 0).time_points), 0)


if __name__ == "__main__":
    unittest.main()
