from shyft import api
import numpy as np

import unittest


class TimeSeriesTestCase(unittest.TestCase):
    """Verify and illustrate TimeSeries
     
     a) point time-series:
        defined by a set of points, 
        projection from point to f(t) (does the point represent state in time, or average of a period?)
        projection of f(t) to average/integral ts, like
        ts_avg_1=average_accessor(ts1,time_axis)
        
     """
    def setUp(self):
        self.c=api.Calendar()
        self.d=api.deltahours(1)
        self.n=24
        self.t= self.c.trim(api.utctime_now(),self.d)
        self.ta=api.Timeaxis(self.t,self.d,self.n)
        
    def tearDown(self):
        pass
    
    def test_operations_on_TsFixed(self):
        dv=np.arange(self.ta.size())
        v=api.DoubleVector.FromNdArray(dv)
        #test create
        tsa=api.TsFixed(self.ta,v)
        # assert its contains time and values as expected.
        self.assertEqual(self.ta.total_period(),tsa.total_period())
        [self.assertAlmostEqual(tsa.value(i),v[i]) for i in xrange(self.ta.size())]
        [self.assertEqual(tsa.time(i),self.ta(i).start) for i in xrange(self.ta.size())]
        [self.assertAlmostEqual(tsa.get(i).v,v[i]) for i in xrange(self.ta.size())]
        # set one value
        v[0]=122
        tsa.set(0, v[0])
        self.assertAlmostEqual(v[0],tsa.value(0))
        # test fill with values
        for i in xrange(len(v)):v[i]=123
        tsa.fill(v[0])
        [self.assertAlmostEqual(tsa.get(i).v,v[i]) for i in xrange(self.ta.size())]
        
    def test_vector_of_timeseries(self):
        dv=np.arange(self.ta.size())
        v=api.DoubleVector.FromNdArray(dv)
        tsf=api.TsFactory();
        tsa=tsf.create_point_ts(self.n, self.t, self.d, v)
        tsvector=api.TsVector()
        self.assertEqual(len(tsvector),0)
        tsvector.push_back(tsa)
        self.assertEqual(len(tsvector),1)
        
    def test_TsFixed(self):
        dv=np.arange(self.ta.size())
        v=api.DoubleVector.FromNdArray(dv)
        tsfixed=api.TsFixed(self.ta,v)
        self.assertEqual(tsfixed.size(),self.ta.size())
        self.assertAlmostEqual(tsfixed.get(0).v, v[0] )
        
    def test_TsPoint(self):
        dv=np.arange(self.ta.size())
        v=api.DoubleVector.FromNdArray(dv)
        t=api.UtcTimeVector();
        for i in xrange(self.ta.size()):
            t.push_back(self.ta(i).start)
        t.push_back(self.ta(self.ta.size()-1).end)
        ta=api.PointTimeaxis(t)
        tspoint=api.TsPoint(ta,v)
        self.assertEqual(tspoint.size(), ta.size())
        self.assertAlmostEqual(tspoint.get(0).v,v[0])
        self.assertEqual(tspoint.get(0).t, ta(0).start)

    def test_TsFactory(self):
        dv=np.arange(self.ta.size())
        v=api.DoubleVector.FromNdArray(dv)
        t=api.UtcTimeVector();
        for i in xrange(self.ta.size()):
            t.push_back(self.ta(i).start)
        t.push_back(self.ta(self.ta.size()-1).end)
        tsf=api.TsFactory()
        ts1=tsf.create_point_ts(self.ta.size(), self.t, self.d, v)
        ts2=tsf.create_time_point_ts(self.ta.total_period(),t,v)
        tslist=api.TsVector()
        tslist.push_back(ts1)
        tslist.push_back(ts2)
        self.assertEqual(tslist.size(),2)
        
    def test_AverageAccessor(self):
        dv=np.arange(self.ta.size())
        v=api.DoubleVector.FromNdArray(dv)
        t=api.UtcTimeVector();
        for i in xrange(self.ta.size()):
            t.push_back(self.ta(i).start)
        t.push_back(self.ta(self.ta.size()-1).end) #important! needs n+1 points to determine n periods in the timeaxis
        tsf=api.TsFactory()
        ts1=tsf.create_point_ts(self.ta.size(), self.t, self.d, v)
        ts2=tsf.create_time_point_ts(self.ta.total_period(),t,v)
        tax=api.Timeaxis(self.ta.start()+api.deltaminutes(30),api.deltahours(1),self.ta.size())
        avg1=api.AverageAccessorTs(ts1,tax)
        self.assertEquals(avg1.size(),tax.size())

    def test_ts_transform(self):
        dv=np.arange(self.ta.size())
        v=api.DoubleVector.FromNdArray(dv)
        t=api.UtcTimeVector();
        for i in xrange(self.ta.size()):
            t.push_back(self.ta(i).start)
        #t.push_back(self.ta(self.ta.size()-1).end) #important! needs n+1 points to determine n periods in the timeaxis
        tax=api.Timeaxis(self.ta.start()+api.deltaminutes(30),api.deltahours(1),self.ta.size())
        tsf=api.TsFactory()
        ts1=tsf.create_point_ts(self.ta.size(), self.t, self.d, v)
        ts2=tsf.create_time_point_ts(self.ta.total_period(),t,v)
        ts3=api.TsFixed(tax,v)
        
        tst=api.TsTransform()
        tt1=tst.to_average(tax.start(),tax.delta(),tax.size(),ts1)
        tt2=tst.to_average(tax.start(),tax.delta(),tax.size(),ts2)
        tt3=tst.to_average(tax.start(),tax.delta(),tax.size(),ts3)
        self.assertEqual(tt1.size(),tax.size())
        self.assertEqual(tt2.size(),tax.size())
        self.assertEqual(tt3.size(),tax.size())

        
if __name__ == "__main__":
    unittest.main()
