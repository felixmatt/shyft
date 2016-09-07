from shyft import api
import numpy as np
from numpy.testing import assert_array_almost_equal
import unittest


class TimeSeries(unittest.TestCase):
    """Verify and illustrate TimeSeries
     
     a) point time-series:
        defined by a set of points, 
        projection from point to f(t) (does the point represent state in time, or average of a period?)
        projection of f(t) to average/integral ts, like
        ts_avg_1=average_accessor(ts1,time_axis)
        
     """

    def setUp(self):
        self.c = api.Calendar()
        self.d = api.deltahours(1)
        self.n = 24
        self.t = self.c.trim(api.utctime_now(), self.d)
        self.ta = api.Timeaxis(self.t, self.d, self.n)

    def tearDown(self):
        pass

    def test_operations_on_TsFixed(self):
        dv = np.arange(self.ta.size())
        v = api.DoubleVector.from_numpy(dv)
        # test create
        tsa = api.TsFixed(self.ta, v)
        # assert its contains time and values as expected.
        self.assertEqual(self.ta.total_period(), tsa.total_period())
        [self.assertAlmostEqual(tsa.value(i), v[i]) for i in range(self.ta.size())]
        [self.assertEqual(tsa.time(i), self.ta(i).start) for i in range(self.ta.size())]
        [self.assertAlmostEqual(tsa.get(i).v, v[i]) for i in range(self.ta.size())]
        # set one value
        v[0] = 122
        tsa.set(0, v[0])
        self.assertAlmostEqual(v[0], tsa.value(0))
        # test fill with values
        for i in range(len(v)): v[i] = 123
        tsa.fill(v[0])
        [self.assertAlmostEqual(tsa.get(i).v, v[i]) for i in range(self.ta.size())]

    def test_vector_of_timeseries(self):
        dv = np.arange(self.ta.size())
        v = api.DoubleVector.from_numpy(dv)
        tsf = api.TsFactory();
        tsa = tsf.create_point_ts(self.n, self.t, self.d, v)
        tsvector = api.TsVector()
        self.assertEqual(len(tsvector), 0)
        tsvector.push_back(tsa)
        self.assertEqual(len(tsvector), 1)

    def test_ts_fixed(self):
        dv = np.arange(self.ta.size())
        v = api.DoubleVector.from_numpy(dv)
        xv = v.to_numpy()

        tsfixed = api.TsFixed(self.ta, v)
        self.assertEqual(tsfixed.size(), self.ta.size())
        self.assertAlmostEqual(tsfixed.get(0).v, v[0])
        vv = tsfixed.values.to_numpy()  # introduced .values for compatibility
        assert_array_almost_equal(dv, vv)
        tsfixed.values[0] = 10.0
        dv[0] = 10.0
        assert_array_almost_equal(dv, tsfixed.v.to_numpy())
        # self.assertAlmostEqual(v,vv)
        # some reference testing:
        ref_v = tsfixed.v
        del tsfixed
        assert_array_almost_equal(dv, ref_v.to_numpy())

    def test_ts_point(self):
        dv = np.arange(self.ta.size())
        v = api.DoubleVector.from_numpy(dv)
        t = api.UtcTimeVector()
        for i in range(self.ta.size()):
            t.push_back(self.ta(i).start)
        t.push_back(self.ta(self.ta.size() - 1).end)
        ta = api.PointTimeaxis(t)
        tspoint = api.TsPoint(ta, v)
        self.assertEqual(tspoint.size(), ta.size())
        self.assertAlmostEqual(tspoint.get(0).v, v[0])
        self.assertAlmostEqual(tspoint.values[0], v[0])  # just to verfy compat .values works
        self.assertEqual(tspoint.get(0).t, ta(0).start)

    def test_ts_factory(self):
        dv = np.arange(self.ta.size())
        v = api.DoubleVector.from_numpy(dv)
        t = api.UtcTimeVector();
        for i in range(self.ta.size()):
            t.push_back(self.ta(i).start)
        t.push_back(self.ta(self.ta.size() - 1).end)
        tsf = api.TsFactory()
        ts1 = tsf.create_point_ts(self.ta.size(), self.t, self.d, v)
        ts2 = tsf.create_time_point_ts(self.ta.total_period(), t, v)
        tslist = api.TsVector()
        tslist.push_back(ts1)
        tslist.push_back(ts2)
        self.assertEqual(tslist.size(), 2)

    def test_average_accessor(self):
        dv = np.arange(self.ta.size())
        v = api.DoubleVector.from_numpy(dv)
        t = api.UtcTimeVector();
        for i in range(self.ta.size()):
            t.push_back(self.ta(i).start)
        t.push_back(
            self.ta(self.ta.size() - 1).end)  # important! needs n+1 points to determine n periods in the timeaxis
        tsf = api.TsFactory()
        ts1 = tsf.create_point_ts(self.ta.size(), self.t, self.d, v)
        ts2 = tsf.create_time_point_ts(self.ta.total_period(), t, v)
        tax = api.Timeaxis(self.ta.total_period().start + api.deltaminutes(30), api.deltahours(1), self.ta.size())
        avg1 = api.AverageAccessorTs(ts1, tax)
        self.assertEqual(avg1.size(), tax.size())
        self.assertIsNotNone(ts2)

    def test_ts_transform(self):
        dv = np.arange(self.ta.size())
        v = api.DoubleVector.from_numpy(dv)
        t = api.UtcTimeVector();
        for i in range(self.ta.size()):
            t.push_back(self.ta(i).start)
        # t.push_back(self.ta(self.ta.size()-1).end) #important! needs n+1 points to determine n periods in the timeaxis
        t_start = self.ta.total_period().start
        dt = api.deltahours(1)
        tax = api.Timeaxis(t_start + api.deltaminutes(30), dt, self.ta.size())
        tsf = api.TsFactory()
        ts1 = tsf.create_point_ts(self.ta.size(), self.t, self.d, v)
        ts2 = tsf.create_time_point_ts(self.ta.total_period(), t, v)
        ts3 = api.TsFixed(tax, v)

        tst = api.TsTransform()
        tt1 = tst.to_average(t_start, dt, tax.size(), ts1)
        tt2 = tst.to_average(t_start, dt, tax.size(), ts2)
        tt3 = tst.to_average(t_start, dt, tax.size(), ts3)
        self.assertEqual(tt1.size(), tax.size())
        self.assertEqual(tt2.size(), tax.size())
        self.assertEqual(tt3.size(), tax.size())

    def test_basic_timeseries_math_operations(self):
        """
        Test that timeseries functionality is exposed, and briefly verify correctness
        of operators (the  shyft core do the rest of the test job, not repeated here).
        """
        c = api.Calendar()
        t0 = api.utctime_now()
        dt = api.deltahours(1)
        n = 240
        ta = api.Timeaxis2(t0, dt, n)

        a = api.Timeseries(ta=ta, fill_value=3.0, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)
        b = api.Timeseries(ta=ta, fill_value=1.0)
        b.fill(2.0)  # demo how to fill a point ts
        c = a + b * 3.0 - a / 2.0  # operator + * - /
        d = -a  # unary minus
        e = a.average(ta)  # average
        f = api.max(c, 300.0)
        g = api.min(c, -300.0)
        h = a.max(c, 300)
        k = a.min(c, -300)

        self.assertEqual(a.size(), n)
        self.assertEqual(b.size(), n)
        self.assertEqual(c.size(), n)
        self.assertAlmostEqual(c.value(0), 3.0 + 2.0 * 3.0 - 3.0 / 2.0)  # 7.5
        for i in range(n):
            self.assertAlmostEqual(c.value(i), a.value(i) + b.value(i) * 3.0 - a.value(i) / 2.0, delta=0.0001)
            self.assertAlmostEqual(d.value(i), - a.value(i), delta=0.0001)
            self.assertAlmostEqual(e.value(i), a.value(i), delta=0.00001)
            self.assertAlmostEqual(f.value(i), +300.0, delta=0.00001)
            self.assertAlmostEqual(h.value(i), +300.0, delta=0.00001)
            self.assertAlmostEqual(g.value(i), -300.0, delta=0.00001)
            self.assertAlmostEqual(k.value(i), -300.0, delta=0.00001)
        # now some more detailed tests for setting values
        b.set(0, 3.0)
        self.assertAlmostEqual(b.value(0), 3.0)
        #  3.0 + 3 * 3 - 3.0/2.0
        self.assertAlmostEqual(c.value(1), 7.5, delta=0.0001)  # 3 + 3*3  - 1.5 = 10.5
        self.assertAlmostEqual(c.value(0), 10.5, delta=0.0001)  # 3 + 3*3  - 1.5 = 10.5

    def test_timeseries_vector(self):
        c = api.Calendar()
        t0 = api.utctime_now()
        dt = api.deltahours(1)
        n = 240
        ta = api.Timeaxis(t0, dt, n)

        a = api.Timeseries(ta=ta, fill_value=3.0, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)
        b = api.Timeseries(ta=ta, fill_value=2.0, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)

        v = api.TsVector()
        v.append(a)
        v.append(b)

        self.assertEqual(len(v), 2)
        self.assertAlmostEqual(v[0].value(0), 3.0, "expect first ts to be 3.0")
        aa = api.Timeseries(ta=a.time_axis, values=a.values,
                            point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)  # copy construct (really copy the values!)
        a.fill(1.0)
        self.assertAlmostEqual(v[0].value(0), 1.0, "expect first ts to be 1.0, because the vector keeps a reference ")
        self.assertAlmostEqual(aa.value(0), 3.0)

    def test_percentiles(self):
        c = api.Calendar()
        t0 = c.time(2016, 1, 1)
        dt = api.deltahours(1)
        n = 240
        ta = api.Timeaxis(t0, dt, n)
        timeseries = api.TsVector()

        for i in range(10):
            timeseries.append(
                api.Timeseries(ta=ta, fill_value=i, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE))

        wanted_percentiles = api.IntVector([0, 10, 50, -1, 70, 100])
        ta_day = api.Timeaxis(t0, dt * 24, n // 24)
        ta_day2 = api.Timeaxis2(t0, dt * 24, n // 24)
        percentiles = api.percentiles(timeseries, ta_day, wanted_percentiles)
        percentiles2 = timeseries.percentiles(ta_day2, wanted_percentiles)  # just to verify it works with alt. syntax

        self.assertEqual(len(percentiles2), len(percentiles))

        for i in range(len(ta_day)):
            self.assertAlmostEqual(0.0, percentiles[0].value(i), 3, "  0-percentile")
            self.assertAlmostEqual(0.9, percentiles[1].value(i), 3, " 10-percentile")
            self.assertAlmostEqual(4.5, percentiles[2].value(i), 3, " 50-percentile")
            self.assertAlmostEqual(4.5, percentiles[3].value(i), 3, "   -average")
            self.assertAlmostEqual(6.3, percentiles[4].value(i), 3, " 70-percentile")
            self.assertAlmostEqual(9.0, percentiles[5].value(i), 3, "100-percentile")

    def test_time_shift(self):
        c = api.Calendar()
        t0 = c.time(2016, 1, 1)
        t1 = c.time(2017, 1, 1)
        dt = api.deltahours(1)
        n = 240
        ta = api.Timeaxis(t0, dt, n)
        ts0 = api.Timeseries(ta=ta, fill_value=3.0, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)
        ts1 = api.time_shift(ts0, t1 - t0)
        ts2 = 2.0 * ts1.time_shift(t0 - t1)  # just to verify it still can take part in an expression

        for i in range(ts0.size()):
            self.assertAlmostEqual(ts0.value(i), ts1.value(i), 3, "expect values to be equal")
            self.assertAlmostEqual(ts0.value(i) * 2.0, ts2.value(i), 3, "expect values to be double value")
            self.assertEqual(ts0.time(i) + (t1 - t0), ts1.time(i), "expect time to be offset delta_t different")
            self.assertEqual(ts0.time(i), ts2.time(i), "expect time to be equal")

    def test_accumulate(self):
        c = api.Calendar()
        t0 = c.time(2016, 1, 1)
        dt = api.deltahours(1)
        n = 240
        ta = api.Timeaxis2(t0, dt, n)
        ts0 = api.Timeseries(ta=ta, fill_value=1.0, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)
        ts1 = ts0.accumulate(ts0.get_time_axis())  # ok, maybe we should make method that does time-axis implicit ?
        ts1_values = ts1.values
        for i in range(n):
            expected_value = i * dt * 1.0
            self.assertAlmostEqual(expected_value, ts1.value(i), 3, "expect integral f(t)*dt")
            self.assertAlmostEqual(expected_value, ts1_values[i], 3, "expect value vector equal as well")


if __name__ == "__main__":
    unittest.main()
