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
        self.ta = api.TimeAxisFixedDeltaT(self.t, self.d, self.n)

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
        tsf = api.TsFactory()
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
        ts_ta = tsfixed.time_axis  # a TsFixed do have .time_axis and .values
        self.assertEqual(len(ts_ta), len(self.ta))  # should have same length etc.

        # verify some simple core-ts to TimeSeries interoperability
        full_ts = tsfixed.TimeSeries  # returns a new TimeSeries as clone from tsfixed
        self.assertEqual(full_ts.size(),tsfixed.size())
        for i in range(tsfixed.size()):
            self.assertEqual(full_ts.time(i),tsfixed.time(i))
            self.assertAlmostEqual(full_ts.value(i),tsfixed.value(i),5)
        ns = tsfixed.nash_sutcliffe(full_ts)
        self.assertAlmostEqual(ns,1.0,4)
        kg = tsfixed.kling_gupta(full_ts,1.0,1.0,1.0)
        self.assertAlmostEqual(kg,1.0,4)

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
        ta = api.TimeAxisByPoints(t)
        tspoint = api.TsPoint(ta, v)
        ts_ta = tspoint.time_axis  # a TsPoint do have .time_axis and .values
        self.assertEqual(len(ts_ta), len(self.ta))  # should have same length etc.

        self.assertEqual(tspoint.size(), ta.size())
        self.assertAlmostEqual(tspoint.get(0).v, v[0])
        self.assertAlmostEqual(tspoint.values[0], v[0])  # just to verfy compat .values works
        self.assertEqual(tspoint.get(0).t, ta(0).start)
        # verify some simple core-ts to TimeSeries interoperability
        full_ts = tspoint.TimeSeries  # returns a new TimeSeries as clone from tsfixed
        self.assertEqual(full_ts.size(),tspoint.size())
        for i in range(tspoint.size()):
            self.assertEqual(full_ts.time(i),tspoint.time(i))
            self.assertAlmostEqual(full_ts.value(i),tspoint.value(i),5)
        ns = tspoint.nash_sutcliffe(full_ts)
        self.assertAlmostEqual(ns,1.0,4)
        kg = tspoint.kling_gupta(full_ts,1.0,1.0,1.0)
        self.assertAlmostEqual(kg,1.0,4)



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
        t = api.UtcTimeVector()
        for i in range(self.ta.size()):
            t.push_back(self.ta(i).start)
        t.push_back(
            self.ta(self.ta.size() - 1).end)  # important! needs n+1 points to determine n periods in the timeaxis
        tsf = api.TsFactory()
        ts1 = tsf.create_point_ts(self.ta.size(), self.t, self.d, v)
        ts2 = tsf.create_time_point_ts(self.ta.total_period(), t, v)
        tax = api.TimeAxisFixedDeltaT(self.ta.total_period().start + api.deltaminutes(30), api.deltahours(1), self.ta.size())
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
        tax = api.TimeAxisFixedDeltaT(t_start + api.deltaminutes(30), dt, self.ta.size())
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
        ta = api.TimeAxis(t0, dt, n)

        a = api.TimeSeries(ta=ta, fill_value=3.0, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)
        b = api.TimeSeries(ta=ta, fill_value=1.0)
        b.fill(2.0)  # demo how to fill a point ts
        self.assertAlmostEquals((1.0-b).values.to_numpy().max(), -1.0)
        self.assertAlmostEquals((b -1.0).values.to_numpy().max(), 1.0)
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
        ta = api.TimeAxisFixedDeltaT(t0, dt, n)

        a = api.TimeSeries(ta=ta, fill_value=3.0, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)
        b = api.TimeSeries(ta=ta, fill_value=2.0, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)

        v = api.TsVector()
        v.append(a)
        v.append(b)

        self.assertEqual(len(v), 2)
        self.assertAlmostEqual(v[0].value(0), 3.0, "expect first ts to be 3.0")
        aa = api.TimeSeries(ta=a.time_axis, values=a.values,
                            point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)  # copy construct (really copy the values!)
        a.fill(1.0)
        self.assertAlmostEqual(v[0].value(0), 1.0, "expect first ts to be 1.0, because the vector keeps a reference ")
        self.assertAlmostEqual(aa.value(0), 3.0)

    def test_percentiles(self):
        c = api.Calendar()
        t0 = c.time(2016, 1, 1)
        dt = api.deltahours(1)
        n = 240
        ta = api.TimeAxisFixedDeltaT(t0, dt, n)
        timeseries = api.TsVector()

        for i in range(10):
            timeseries.append(
                api.TimeSeries(ta=ta, fill_value=i, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE))

        wanted_percentiles = api.IntVector([0, 10, 50, -1, 70, 100])
        ta_day = api.TimeAxisFixedDeltaT(t0, dt * 24, n // 24)
        ta_day2 = api.TimeAxis(t0, dt * 24, n // 24)
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
        ta = api.TimeAxisFixedDeltaT(t0, dt, n)
        ts0 = api.TimeSeries(ta=ta, fill_value=3.0, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)
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
        ta = api.TimeAxis(t0, dt, n)
        ts0 = api.TimeSeries(ta=ta, fill_value=1.0, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)
        ts1 = ts0.accumulate(ts0.get_time_axis())  # ok, maybe we should make method that does time-axis implicit ?
        ts1_values = ts1.values
        for i in range(n):
            expected_value = i * dt * 1.0
            self.assertAlmostEqual(expected_value, ts1.value(i), 3, "expect integral f(t)*dt")
            self.assertAlmostEqual(expected_value, ts1_values[i], 3, "expect value vector equal as well")

    def test_kling_gupta_and_nash_sutcliffe(self):
        """
        Test/verify exposure of the kling_gupta and nash_sutcliffe correlation functions

        """

        def np_nash_sutcliffe(o, p):
            return 1 - (np.sum((o - p) ** 2)) / (np.sum((o - np.mean(o)) ** 2))

        c = api.Calendar()
        t0 = c.time(2016, 1, 1)
        dt = api.deltahours(1)
        n = 240
        ta = api.TimeAxis(t0, dt, n)
        from math import sin, pi
        rad_max = 10 * 2 * pi
        obs_values = api.DoubleVector.from_numpy(np.array([sin(i * rad_max / n) for i in range(n)]))
        mod_values = api.DoubleVector.from_numpy(np.array([0.1 + sin(pi / 10.0 + i * rad_max / n) for i in range(n)]))
        obs_ts = api.TimeSeries(ta=ta, values=obs_values, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)
        mod_ts = api.TimeSeries(ta=ta, values=mod_values, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)

        self.assertAlmostEqual(api.kling_gupta(obs_ts, obs_ts, ta, 1.0, 1.0, 1.0), 1.0, None, "1.0 for perfect match")
        self.assertAlmostEqual(api.nash_sutcliffe(obs_ts, obs_ts, ta), 1.0, None, "1.0 for perfect match")
        # verify some non trivial cases, and compare to numpy version of ns
        mod_inv = obs_ts * -1.0
        kge_inv = obs_ts.kling_gupta(mod_inv)  # also show how to use time-series.method itself to ease use
        ns_inv = obs_ts.nash_sutcliffe(mod_inv)  # similar for nash_sutcliffe, you can reach it directly from a ts
        ns_inv2 = np_nash_sutcliffe(obs_ts.values.to_numpy(), mod_inv.values.to_numpy())
        self.assertLessEqual(kge_inv, 1.0, "should be less than 1")
        self.assertLessEqual(ns_inv, 1.0, "should be less than 1")
        self.assertAlmostEqual(ns_inv, ns_inv2, 4, "should equal numpy calculated value")
        kge_obs_mod = api.kling_gupta(obs_ts, mod_ts, ta, 1.0, 1.0, 1.0)
        self.assertLessEqual(kge_obs_mod, 1.0)
        self.assertAlmostEqual(obs_ts.nash_sutcliffe( mod_ts), np_nash_sutcliffe(obs_ts.values.to_numpy(), mod_ts.values.to_numpy()))

    def test_periodic_pattern_ts(self):
        c = api.Calendar()
        t0 = c.time(2016, 1, 1)
        dt = api.deltahours(1)
        n = 240
        ta = api.TimeAxis(t0, dt, n)
        pattern_values = api.DoubleVector.from_numpy(np.arange(8))
        pattern_dt = api.deltahours(3)
        pattern_t0 = c.time(2015,6,1)
        pattern_ts = api.create_periodic_pattern_ts(pattern_values, pattern_dt, pattern_t0, ta)  # this is how to create a periodic pattern ts (used in gridpp/kalman bias handling)
        self.assertAlmostEqual(pattern_ts.value(0), 0.0)
        self.assertAlmostEqual(pattern_ts.value(1), 0.0)
        self.assertAlmostEqual(pattern_ts.value(2), 0.0)
        self.assertAlmostEqual(pattern_ts.value(3), 1.0)  # next step in pattern starts here
        self.assertAlmostEqual(pattern_ts.value(24), 0.0)  # next day repeats the pattern

    def test_partition_by(self):
        """
        verify/demo exposure of the .partition_by function that can
        be used to produce yearly percentiles statistics for long historical
        time-series

        """
        c = api.Calendar()
        t0 = c.time(1930, 9, 1)
        dt = api.deltahours(1)
        n = c.diff_units(t0, c.time(2016, 9, 1), dt)

        ta = api.TimeAxis(t0, dt, n)
        pattern_values = api.DoubleVector.from_numpy(np.arange(len(ta))) # increasing values

        src_ts = api.TimeSeries(ta=ta, values=pattern_values, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)

        partition_t0 = c.time(2016, 9, 1)
        n_partitions = 80
        partition_interval = api.Calendar.YEAR
        # get back TsVector,
        # where all TsVector[i].index_of(partition_t0)
        # is equal to the index ix for which the TsVector[i].value(ix) correspond to start value of that particular partition.
        ts_partitions = src_ts.partition_by(c, t0, partition_interval, n_partitions, partition_t0)
        self.assertEqual(len(ts_partitions),n_partitions)
        ty = t0
        for ts in ts_partitions:
            ix = ts.index_of(partition_t0)
            vix = ts.value(ix)
            expected_value = c.diff_units(t0, ty, dt)
            self.assertEqual(vix, expected_value)
            ty = c.add(ty, partition_interval, 1)

        # Now finally, try percentiles on the partitions
        wanted_percentiles = [0, 10, 25, -1, 50, 75, 90, 100]
        ta_percentiles = api.TimeAxis(partition_t0, api.deltahours(24), 365)
        percentiles = api.percentiles(ts_partitions,ta_percentiles,wanted_percentiles)
        self.assertEqual(len(percentiles), len(wanted_percentiles))

    def test_ts_reference_and_bind(self):
        c = api.Calendar()
        t0 = c.time(2016, 9, 1)
        dt = api.deltahours(1)
        n = c.diff_units(t0, c.time(2017, 9, 1), dt)

        ta = api.TimeAxis(t0, dt, n)
        pattern_values = api.DoubleVector.from_numpy(np.arange(len(ta))) # increasing values

        a = api.TimeSeries(ta=ta, values=pattern_values, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)
        b_id = "netcdf://path_to_file/path_to_ts"
        b = api.TimeSeries(b_id)
        c = (a + b)*4.0  # make an expression, with a ts-reference, not yet bound
        c_blob = c.serialize()  # converts the entire stuff into a blob
        bind_info= c.find_ts_bind_info()

        self.assertEqual(len(bind_info), 1,"should find just one ts to bind")
        self.assertEqual(bind_info[0].id, b_id,"the id to bind should be equal to b_id")
        try:
            c.value(0)  # verify touching a unbound ts raises exception
            self.assertFalse(True, "should not reach here!")
        except RuntimeError:
            pass

        # verify we can bind a ts
        bind_info[0].ts.bind(a)  # it's ok to bind same series multiple times, it takes a copy of a values

        # and now we can use c expression as pr. usual, evaluate etc.
        self.assertAlmostEqual(c.value(10), a.value(10)*2*4.0, 3)

        c_resurrected = api.TimeSeries.deserialize(c_blob)

        bi = c_resurrected.find_ts_bind_info()
        bi[0].ts.bind(a)
        self.assertAlmostEqual(c_resurrected.value(10), a.value(10) * 2*4.0, 3)

if __name__ == "__main__":
    unittest.main()
