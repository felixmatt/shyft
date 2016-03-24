from builtins import range
from shyft import api
import numpy as np
import unittest


class TimeAxis(unittest.TestCase):
    """Verify and illustrate Timeaxis
       defined as n periods non-overlapping ascending
        
     """
    def setUp(self):
        self.c=api.Calendar()
        self.d=api.deltahours(1)
        self.n=24
        #self.t= self.c.trim(api.utctime_now(),self.d)
        self.t= self.c.trim(self.c.time(api.YMDhms(1969,12,31,0,0,0)),self.d)
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
        va=np.array([86400,3600,3],dtype=np.int64)
        xta = api.Timeaxis(int(va[0]), int(va[1]), int(va[2]))
        #xta = api.Timeaxis(va[0], va[1], va[2])# TODO: boost.python require this to be int, needs overload for np.int64 types..
        #xta = api.Timeaxis(86400,3600,3)
        self.assertEqual(xta.size(),3)
    
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
        tap=api.PointTimeaxis(api.UtcTimeVector([t for t in range(self.t,self.t+(self.n+1)*self.d,self.d)])) #TODO: Should work
        #tap=api.PointTimeaxis(api.UtcTimeVector.from_numpy(np.array([t for t in range(self.t,self.t+(self.n+1)*self.d,self.d)]))) #TODO: Should work
        self.assertEqual(tap.size(),self.ta.size())
        for i in range(self.ta.size()):
            self.assertEqual(tap(i), self.ta(i))
        s=str(tap)
        self.assertTrue(len(s)>0)
        
if __name__ == "__main__":
    unittest.main()
