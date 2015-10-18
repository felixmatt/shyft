from shyft import api
import unittest


class TimeAxis(unittest.TestCase):
    """Verify and illustrate Timeaxis
       defined as n periods non-overlapping ascending
        
     """
    def setUp(self):
        self.c=api.Calendar()
        self.d=api.deltahours(1)
        self.n=24
        self.t= self.c.trim(api.utctime_now(),self.d)
        self.ta=api.Timeaxis(self.t,self.d,self.n)
        
    def tearDown(self):
        pass
    
    def test_create_timeaxis(self):
        self.assertEqual(self.ta.size(),self.n)
        self.assertEqual(len(self.ta),self.n)
        self.assertEqual(self.ta(0).start,self.t)
        self.assertEqual(self.ta(0).end,self.t+self.d)
        self.assertEqual(self.ta(1).start,self.t+self.d)
        self.assertEqual(self.ta.total_period().start,self.t)
    
    def test_iterate_timeaxis(self):
        tot_dt=0
        for p in self.ta:
            tot_dt += p.timespan()
        self.assertEqual(tot_dt,self.n*self.d)
    
    def test_timeaxis_str(self):
        s=str(self.ta)
        self.assertTrue(len(s)>10)
    
    def test_point_timeaxis_(self):
        """ 
        A point time axis takes n+1 points do describe n-periods, where
        each period is defined as [ point_i .. point_i+1 >
        """
        tap=api.PointTimeaxis(api.UtcTimeVector([t for t in xrange(self.t,self.t+(self.n+1)*self.d,self.d)]))
        self.assertEquals(tap.size(),self.ta.size())
        for i in xrange(self.ta.size()):
            self.assertEquals(tap(i),self.ta(i))
        s=str(tap)
        self.assertTrue(len(s)>0)
        
if __name__ == "__main__":
    unittest.main()
