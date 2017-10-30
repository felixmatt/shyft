from shyft import api
import numpy as np
import math
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
        tsa = api.TsFixed(self.ta, v, api.POINT_INSTANT_VALUE)
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
        tsvector.push_back(tsa)
        vv = tsvector.values_at_time(self.ta.time(3))  # verify it's easy to get out vectorized results at time t
        self.assertEqual(len(vv), len(tsvector))
        self.assertAlmostEqual(vv[0], 3.0)
        self.assertAlmostEqual(vv[1], 3.0)
        ts_list = [tsa, tsa]
        vv = api.ts_vector_values_at_time(ts_list, self.ta.time(4))  # also check it work with list(TimeSeries)
        self.assertEqual(len(vv), len(tsvector))
        self.assertAlmostEqual(vv[0], 4.0)
        self.assertAlmostEqual(vv[1], 4.0)

    def test_ts_fixed(self):
        dv = np.arange(self.ta.size())
        v = api.DoubleVector.from_numpy(dv)
        xv = v.to_numpy()

        tsfixed = api.TsFixed(self.ta, v, api.POINT_AVERAGE_VALUE)
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
        self.assertEqual(full_ts.size(), tsfixed.size())
        for i in range(tsfixed.size()):
            self.assertEqual(full_ts.time(i), tsfixed.time(i))
            self.assertAlmostEqual(full_ts.value(i), tsfixed.value(i), 5)
        ns = tsfixed.nash_sutcliffe(full_ts)
        self.assertAlmostEqual(ns, 1.0, 4)
        kg = tsfixed.kling_gupta(full_ts, 1.0, 1.0, 1.0)
        self.assertAlmostEqual(kg, 1.0, 4)

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
        tspoint = api.TsPoint(ta, v, api.POINT_AVERAGE_VALUE)
        ts_ta = tspoint.time_axis  # a TsPoint do have .time_axis and .values
        self.assertEqual(len(ts_ta), len(self.ta))  # should have same length etc.

        self.assertEqual(tspoint.size(), ta.size())
        self.assertAlmostEqual(tspoint.get(0).v, v[0])
        self.assertAlmostEqual(tspoint.values[0], v[0])  # just to verfy compat .values works
        self.assertEqual(tspoint.get(0).t, ta(0).start)
        # verify some simple core-ts to TimeSeries interoperability
        full_ts = tspoint.TimeSeries  # returns a new TimeSeries as clone from tsfixed
        self.assertEqual(full_ts.size(), tspoint.size())
        for i in range(tspoint.size()):
            self.assertEqual(full_ts.time(i), tspoint.time(i))
            self.assertAlmostEqual(full_ts.value(i), tspoint.value(i), 5)
        ns = tspoint.nash_sutcliffe(full_ts)
        self.assertAlmostEqual(ns, 1.0, 4)
        kg = tspoint.kling_gupta(full_ts, 1.0, 1.0, 1.0)
        self.assertAlmostEqual(kg, 1.0, 4)

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
        tax = api.TimeAxisFixedDeltaT(self.ta.total_period().start + api.deltaminutes(30), api.deltahours(1),
                                      self.ta.size())
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
        ts3 = api.TsFixed(tax, v, api.POINT_INSTANT_VALUE)

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
        self.assertTrue(a)  # should evaluate to true
        b = api.TimeSeries(ta=ta, fill_value=1.0,point_fx=api.point_interpretation_policy.POINT_INSTANT_VALUE)
        b.fill(2.0)  # demo how to fill a point ts
        self.assertAlmostEqual((1.0 - b).values.to_numpy().max(), -1.0)
        self.assertAlmostEqual((b - 1.0).values.to_numpy().max(), 1.0)
        c = a + b*3.0 - a/2.0  # operator + * - /
        d = -a  # unary minus
        e = a.average(ta)  # average
        f = api.max(c, 300.0)
        g = api.min(c, -300.0)
        # h = a.max(c, 300) # class static method not supported
        h = c.max(300.0)
        k = c.min(-300)

        self.assertEqual(a.size(), n)
        self.assertEqual(b.size(), n)
        self.assertEqual(c.size(), n)
        self.assertAlmostEqual(c.value(0), 3.0 + 2.0*3.0 - 3.0/2.0)  # 7.5
        for i in range(n):
            self.assertAlmostEqual(c.value(i), a.value(i) + b.value(i)*3.0 - a.value(i)/2.0, delta=0.0001)
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

        wanted_percentiles = api.IntVector([api.statistics_property.MIN_EXTREME,
                                            0, 10, 50,
                                            api.statistics_property.AVERAGE,
                                            70, 100,
                                            api.statistics_property.MAX_EXTREME])
        ta_day = api.TimeAxisFixedDeltaT(t0, dt*24, n//24)
        ta_day2 = api.TimeAxis(t0, dt*24, n//24)
        percentiles = api.percentiles(timeseries, ta_day, wanted_percentiles)
        percentiles2 = timeseries.percentiles(ta_day2, wanted_percentiles)  # just to verify it works with alt. syntax

        self.assertEqual(len(percentiles2), len(percentiles))

        for i in range(len(ta_day)):
            self.assertAlmostEqual(0.0, percentiles[0].value(i), 3, "min-extreme ")
            self.assertAlmostEqual(0.0, percentiles[1].value(i), 3, "  0-percentile")
            self.assertAlmostEqual(0.9, percentiles[2].value(i), 3, " 10-percentile")
            self.assertAlmostEqual(4.5, percentiles[3].value(i), 3, " 50-percentile")
            self.assertAlmostEqual(4.5, percentiles[4].value(i), 3, "   -average")
            self.assertAlmostEqual(6.3, percentiles[5].value(i), 3, " 70-percentile")
            self.assertAlmostEqual(9.0, percentiles[6].value(i), 3, "100-percentile")
            self.assertAlmostEqual(9.0, percentiles[7].value(i), 3, "max-extreme")

    def test_percentiles_with_min_max_extremes(self):
        """ the percentiles function now also supports picking out the min-max peak value
            within each interval.
            Setup test-data so that we have a well known percentile result,
            but also have peak-values within the interval that we can
            verify.
            We let hour ts 0..9 have values 0..9 constant 24*10 days
               then modify ts[1], every day first  value to a peak min value equal to - day_no*1
                                  every day second value to a peak max value equal to + day_no*1
                                  every day 3rd    value to a nan value
            ts[1] should then have same average value for each day (so same percentile)
                                            but min-max extreme should be equal to +- day_no*1
        """
        c = api.Calendar()
        t0 = c.time(2016, 1, 1)
        dt = api.deltahours(1)
        n = 240
        ta = api.TimeAxis(t0, dt, n)
        timeseries = api.TsVector()
        p_fx = api.point_interpretation_policy.POINT_AVERAGE_VALUE
        for i in range(10):
            timeseries.append(api.TimeSeries(ta=ta, fill_value=i, point_fx=p_fx))

        ts = timeseries[1]  # pick this one to insert min/max extremes
        for i in range(0, 240, 24):
            ts.set(i + 0, 1.0 - 100*i/24.0)
            ts.set(i + 1, 1.0 + 100*i/24.0)  # notice that when i==0, this gives 1.0
            ts.set(i + 2, float('nan'))  # also put in a nan, just to verify it is ignored during average processing

        wanted_percentiles = api.IntVector([api.statistics_property.MIN_EXTREME,
                                            0, 10, 50,
                                            api.statistics_property.AVERAGE,
                                            70, 100,
                                            api.statistics_property.MAX_EXTREME])
        ta_day = api.TimeAxis(t0, dt*24, n//24)
        percentiles = api.percentiles(timeseries, ta_day, wanted_percentiles)
        for i in range(len(ta_day)):
            if i == 0:  # first timestep, the min/max extremes are picked from 0'th and 9'th ts.
                self.assertAlmostEqual(0.0, percentiles[0].value(i), 3, "min-extreme ")
                self.assertAlmostEqual(9.0, percentiles[7].value(i), 3, "min-extreme ")
            else:
                self.assertAlmostEqual(1.0 - 100.0*i*24.0/24.0, percentiles[0].value(i), 3, "min-extreme ")
                self.assertAlmostEqual(1.0 + 100.0*i*24.0/24.0, percentiles[7].value(i), 3, "max-extreme")
            self.assertAlmostEqual(0.0, percentiles[1].value(i), 3, "  0-percentile")
            self.assertAlmostEqual(0.9, percentiles[2].value(i), 3, " 10-percentile")
            self.assertAlmostEqual(4.5, percentiles[3].value(i), 3, " 50-percentile")
            self.assertAlmostEqual(4.5, percentiles[4].value(i), 3, "   -average")
            self.assertAlmostEqual(6.3, percentiles[5].value(i), 3, " 70-percentile")
            self.assertAlmostEqual(9.0, percentiles[6].value(i), 3, "100-percentile")

    def test_time_shift(self):
        c = api.Calendar()
        t0 = c.time(2016, 1, 1)
        t1 = c.time(2017, 1, 1)
        dt = api.deltahours(1)
        n = 240
        ta = api.TimeAxisFixedDeltaT(t0, dt, n)
        ts0 = api.TimeSeries(ta=ta, fill_value=3.0, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)
        tsa = api.TimeSeries('a')

        ts1 = api.time_shift(tsa, t1 - t0)
        self.assertTrue(ts1.needs_bind())
        ts1_blob = ts1.serialize()
        ts1 = api.TimeSeries.deserialize(ts1_blob)
        tsb = ts1.find_ts_bind_info()
        self.assertEqual(len(tsb), 1)
        tsb[0].ts.bind(ts0)
        ts1.bind_done()
        self.assertFalse(ts1.needs_bind())

        ts2 = 2.0*ts1.time_shift(t0 - t1)  # just to verify it still can take part in an expression

        for i in range(ts0.size()):
            self.assertAlmostEqual(ts0.value(i), ts1.value(i), 3, "expect values to be equal")
            self.assertAlmostEqual(ts0.value(i)*2.0, ts2.value(i), 3, "expect values to be double value")
            self.assertEqual(ts0.time(i) + (t1 - t0), ts1.time(i), "expect time to be offset delta_t different")
            self.assertEqual(ts0.time(i), ts2.time(i), "expect time to be equal")

    def test_accumulate(self):
        c = api.Calendar()
        t0 = c.time(2016, 1, 1)
        dt = api.deltahours(1)
        n = 240
        ta = api.TimeAxis(t0, dt, n)
        ts0 = api.TimeSeries(ta=ta, fill_value=1.0, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)
        tsa = 1.0*api.TimeSeries('a') + 0.0  # an expression, that need bind

        ts1 = tsa.accumulate(ta)  # ok, maybe we should make method that does time-axis implicit ?
        self.assertTrue(ts1.needs_bind())
        ts1_blob = ts1.serialize()
        ts1 = api.TimeSeries.deserialize(ts1_blob)
        tsb = ts1.find_ts_bind_info()
        self.assertEqual(len(tsb), 1)
        tsb[0].ts.bind(ts0)
        ts1.bind_done()
        self.assertFalse(ts1.needs_bind())

        ts1_values = ts1.values
        for i in range(n):
            expected_value = i*dt*1.0
            self.assertAlmostEqual(expected_value, ts1.value(i), 3, "expect integral f(t)*dt")
            self.assertAlmostEqual(expected_value, ts1_values[i], 3, "expect value vector equal as well")

    def test_integral(self):
        c = api.Calendar()
        t0 = c.time(2016, 1, 1)
        dt = api.deltahours(1)
        n = 240
        ta = api.TimeAxis(t0, dt, n)
        fill_value = 1.0
        ts = api.TimeSeries(ta=ta, fill_value=fill_value, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)
        tsa = api.TimeSeries('a')*1.0 + 0.0  # expression, needing bind
        tsb = api.TimeSeries('b')*1.0 + 0.0  # another expression, needing bind for different ts
        ts_i1 = tsa.integral(ta)
        ts_i2 = api.integral(tsb, ta)
        # circulate through serialization
        ts_i1_blob = ts_i1.serialize()
        ts_i2_blob = ts_i2.serialize()
        ts_i1 = api.TimeSeries.deserialize(ts_i1_blob)
        ts_i2 = api.TimeSeries.deserialize(ts_i2_blob)

        for ts_i in [ts_i1, ts_i2]:
            self.assertTrue(ts_i.needs_bind())
            tsb = ts_i.find_ts_bind_info()
            self.assertEqual(len(tsb), 1)
            tsb[0].ts.bind(ts)
            ts_i.bind_done()
            self.assertFalse(ts_i.needs_bind())

        ts_i1_values = ts_i1.values
        for i in range(n):
            expected_value = dt*fill_value
            self.assertAlmostEqual(expected_value, ts_i1.value(i), 4, "expect integral of each interval")
            self.assertAlmostEqual(expected_value, ts_i2.value(i), 4, "expect integral of each interval")
            self.assertAlmostEqual(expected_value, ts_i1_values[i], 4, "expect integral of each interval")

    def test_kling_gupta_and_nash_sutcliffe(self):
        """
        Test/verify exposure of the kling_gupta and nash_sutcliffe correlation functions

        """

        def np_nash_sutcliffe(o, p):
            return 1 - (np.sum((o - p) ** 2))/(np.sum((o - np.mean(o)) ** 2))

        c = api.Calendar()
        t0 = c.time(2016, 1, 1)
        dt = api.deltahours(1)
        n = 240
        ta = api.TimeAxis(t0, dt, n)
        from math import sin, pi
        rad_max = 10*2*pi
        obs_values = api.DoubleVector.from_numpy(np.array([sin(i*rad_max/n) for i in range(n)]))
        mod_values = api.DoubleVector.from_numpy(np.array([0.1 + sin(pi/10.0 + i*rad_max/n) for i in range(n)]))
        obs_ts = api.TimeSeries(ta=ta, values=obs_values, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)
        mod_ts = api.TimeSeries(ta=ta, values=mod_values, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)

        self.assertAlmostEqual(api.kling_gupta(obs_ts, obs_ts, ta, 1.0, 1.0, 1.0), 1.0, None, "1.0 for perfect match")
        self.assertAlmostEqual(api.nash_sutcliffe(obs_ts, obs_ts, ta), 1.0, None, "1.0 for perfect match")
        # verify some non trivial cases, and compare to numpy version of ns
        mod_inv = obs_ts*-1.0
        kge_inv = obs_ts.kling_gupta(mod_inv)  # also show how to use time-series.method itself to ease use
        ns_inv = obs_ts.nash_sutcliffe(mod_inv)  # similar for nash_sutcliffe, you can reach it directly from a ts
        ns_inv2 = np_nash_sutcliffe(obs_ts.values.to_numpy(), mod_inv.values.to_numpy())
        self.assertLessEqual(kge_inv, 1.0, "should be less than 1")
        self.assertLessEqual(ns_inv, 1.0, "should be less than 1")
        self.assertAlmostEqual(ns_inv, ns_inv2, 4, "should equal numpy calculated value")
        kge_obs_mod = api.kling_gupta(obs_ts, mod_ts, ta, 1.0, 1.0, 1.0)
        self.assertLessEqual(kge_obs_mod, 1.0)
        self.assertAlmostEqual(obs_ts.nash_sutcliffe(mod_ts),
                               np_nash_sutcliffe(obs_ts.values.to_numpy(), mod_ts.values.to_numpy()))

    def test_periodic_pattern_ts(self):
        c = api.Calendar()
        t0 = c.time(2016, 1, 1)
        dt = api.deltahours(1)
        n = 240
        ta = api.TimeAxis(t0, dt, n)
        pattern_values = api.DoubleVector.from_numpy(np.arange(8))
        pattern_dt = api.deltahours(3)
        pattern_t0 = c.time(2015, 6, 1)
        pattern_ts = api.create_periodic_pattern_ts(pattern_values, pattern_dt, pattern_t0,
                                                    ta)  # this is how to create a periodic pattern ts (used in gridpp/kalman bias handling)
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
        pattern_values = api.DoubleVector.from_numpy(np.arange(len(ta)))  # increasing values

        src_ts = api.TimeSeries(ta=ta, values=pattern_values,
                                point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)

        partition_t0 = c.time(2016, 9, 1)
        n_partitions = 80
        partition_interval = api.Calendar.YEAR
        # get back TsVector,
        # where all TsVector[i].index_of(partition_t0)
        # is equal to the index ix for which the TsVector[i].value(ix) correspond to start value of that particular partition.
        ts_partitions = src_ts.partition_by(c, t0, partition_interval, n_partitions, partition_t0)
        self.assertEqual(len(ts_partitions), n_partitions)
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
        percentiles = api.percentiles(ts_partitions, ta_percentiles, wanted_percentiles)
        self.assertEqual(len(percentiles), len(wanted_percentiles))

    def test_empty_ts(self):
        a = api.TimeSeries()
        self.assertEqual(a.size(), 0)
        self.assertEqual(a.values.size(), 0)
        self.assertEqual(len(a.values.to_numpy()), 0)
        self.assertFalse(a.total_period().valid())
        self.assertFalse(a)  # evaluate to false
        try:
            a.time_axis
            self.assertFail("Expected exception")
        except RuntimeError as re:
            pass

    def test_abs(self):
        c = api.Calendar()
        t0 = c.time(2016, 1, 1)
        dt = api.deltahours(1)
        n = 4
        v = api.DoubleVector([1.0, -1.5, float("nan"), 3.0])
        ta = api.TimeAxisFixedDeltaT(t0, dt, n)
        ts0 = api.TimeSeries(ta=ta, values=v, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)
        tsa = api.TimeSeries('a')
        ts1 = tsa.abs()
        ts1_blob = ts1.serialize()
        ts1 = api.TimeSeries.deserialize(ts1_blob)
        self.assertTrue(ts1.needs_bind())
        bts = ts1.find_ts_bind_info()
        self.assertEqual(len(bts), 1)
        bts[0].ts.bind(ts0)
        ts1.bind_done()
        self.assertFalse(ts1.needs_bind())
        self.assertAlmostEqual(ts0.value(0), ts1.value(0), 6)
        self.assertAlmostEqual(abs(ts0.value(1)), ts1.value(1), 6)
        self.assertTrue(math.isnan(ts1.value(2)))
        self.assertAlmostEqual(ts0.value(3), ts1.value(3), 6)
        tsv0 = api.TsVector()
        tsv0.append(ts0)
        tsv1 = tsv0.abs()
        self.assertAlmostEqual(tsv0[0].value(0), tsv1[0].value(0), 6)
        self.assertAlmostEqual(abs(tsv0[0].value(1)), tsv1[0].value(1), 6)
        self.assertTrue(math.isnan(tsv1[0].value(2)))
        self.assertAlmostEqual(tsv0[0].value(3), tsv1[0].value(3), 6)

    def test_ts_reference_and_bind(self):
        c = api.Calendar()
        t0 = c.time(2016, 9, 1)
        dt = api.deltahours(1)
        n = c.diff_units(t0, c.time(2017, 9, 1), dt)

        ta = api.TimeAxis(t0, dt, n)
        pattern_values = api.DoubleVector.from_numpy(np.arange(len(ta)))  # increasing values

        a = api.TimeSeries(ta=ta, values=pattern_values, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)
        b_id = "netcdf://path_to_file/path_to_ts"
        b = api.TimeSeries(b_id)
        c = (a + b)*4.0  # make an expression, with a ts-reference, not yet bound
        c_blob = c.serialize()  # converts the entire stuff into a blob
        bind_info = c.find_ts_bind_info()

        self.assertEqual(len(bind_info), 1, "should find just one ts to bind")
        self.assertEqual(bind_info[0].id, b_id, "the id to bind should be equal to b_id")
        try:
            c.value(0)  # verify touching a unbound ts raises exception
            self.assertFalse(True, "should not reach here!")
        except RuntimeError:
            pass
        self.assertEqual(c.needs_bind(), True)  # verify this expression needs binding
        # verify we can bind a ts
        bind_info[0].ts.bind(a)  # it's ok to bind same series multiple times, it takes a copy of a values
        c.bind_done()
        self.assertEqual(c.needs_bind(), False)  # verify this expression do not need binding anymore
        # and now we can use c expression as pr. usual, evaluate etc.
        self.assertAlmostEqual(c.value(10), a.value(10)*2*4.0, 3)

        c_resurrected = api.TimeSeries.deserialize(c_blob)

        bi = c_resurrected.find_ts_bind_info()
        bi[0].ts.bind(a)
        c_resurrected.bind_done()
        self.assertAlmostEqual(c_resurrected.value(10), a.value(10)*2*4.0, 3)
        # verify we can create a ref.ts with something that resolves to a point ts.
        bind_expr_ts = api.TimeSeries("some_sym", 3.0*a)  # notice that we can bind with something that is an expression
        self.assertIsNotNone(bind_expr_ts)
        self.assertAlmostEqual(bind_expr_ts.value(0), 3.0*a.value(0))  # just to check, its for real


    def test_a_time_series_vector(self):
        c = api.Calendar()
        t0 = api.utctime_now()
        dt = api.deltahours(1)
        n = 240
        ta = api.TimeAxisFixedDeltaT(t0, dt, n)

        a = api.TimeSeries(ta=ta, fill_value=3.0, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)
        b = api.TimeSeries(ta=ta, fill_value=2.0, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)
        c = api.TimeSeries(ta=ta, fill_value=10.0, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)
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

        vt = v.values_at(t0).to_numpy()
        self.assertEqual(len(vt), len(v))
        v1 = v[0:1]
        self.assertEqual(len(v1), 1)
        self.assertAlmostEqual(v1[0].value(0), 1.0)
        v_clone = api.TsVector(v)
        self.assertEqual(len(v_clone), len(v))
        del v_clone[-1]
        self.assertEqual(len(v_clone), 1)
        self.assertEqual(len(v), 2)
        v_slice_all = v.slice(api.IntVector())
        v_slice_1 = v.slice(api.IntVector([1]))
        v_slice_12 = v.slice(api.IntVector([0, 1]))
        self.assertEqual(len(v_slice_all), 2)
        self.assertEqual(len(v_slice_1), 1)
        self.assertAlmostEqual(v_slice_1[0].value(0), 2.0)
        self.assertEqual(len(v_slice_12), 2)
        self.assertAlmostEqual(v_slice_12[0].value(0), 1.0)

        # multiplication by scalar
        v_x_2a = v*2.0
        v_x_2b = 2.0*v
        for i in range(len(v)):
            self.assertAlmostEqual(v_x_2a[i].value(0), 2*v[i].value(0))
            self.assertAlmostEqual(v_x_2b[i].value(0), 2*v[i].value(0))

        # division by scalar
        v_d_a = v/3.0
        v_d_b = 3.0/v
        for i in range(len(v)):
            self.assertAlmostEqual(v_d_a[i].value(0), v[i].value(0)/3.0)
            self.assertAlmostEqual(v_d_b[i].value(0), 3.0/v[i].value(0))

        # addition by scalar
        v_a_a = v + 3.0
        v_a_b = 3.0 + v
        for i in range(len(v)):
            self.assertAlmostEqual(v_a_a[i].value(0), v[i].value(0) + 3.0)
            self.assertAlmostEqual(v_a_b[i].value(0), 3.0 + v[i].value(0))

        # sub by scalar
        v_s_a = v - 3.0
        v_s_b = 3.0 - v
        for i in range(len(v)):
            self.assertAlmostEqual(v_s_a[i].value(0), v[i].value(0) - 3.0)
            self.assertAlmostEqual(v_s_b[i].value(0), 3.0 - v[i].value(0))

        # multiplication vector by ts
        v_x_ts = v*c
        ts_x_v = c*v
        for i in range(len(v)):
            self.assertAlmostEqual(v_x_ts[i].value(0), v[i].value(0)*c.value(0))
            self.assertAlmostEqual(ts_x_v[i].value(0), c.value(0)*v[i].value(0))

        # division vector by ts
        v_d_ts = v/c
        ts_d_v = c/v
        for i in range(len(v)):
            self.assertAlmostEqual(v_d_ts[i].value(0), v[i].value(0)/c.value(0))
            self.assertAlmostEqual(ts_d_v[i].value(0), c.value(0)/v[i].value(0))

        # add vector by ts
        v_a_ts = v + c
        ts_a_v = c + v
        for i in range(len(v)):
            self.assertAlmostEqual(v_a_ts[i].value(0), v[i].value(0) + c.value(0))
            self.assertAlmostEqual(ts_a_v[i].value(0), c.value(0) + v[i].value(0))

        # sub vector by ts
        v_s_ts = v - c
        ts_s_v = c - v
        for i in range(len(v)):
            self.assertAlmostEqual(v_s_ts[i].value(0), v[i].value(0) - c.value(0))
            self.assertAlmostEqual(ts_s_v[i].value(0), c.value(0) - v[i].value(0))

        # vector mult vector
        va = v
        vb = 2.0*v

        v_m_v = va*vb
        self.assertEqual(len(v_m_v), len(va))
        for i in range(len(va)):
            self.assertAlmostEqual(v_m_v[i].value(0), va[i].value(0)*vb[i].value(0))

        # vector div vector
        v_d_v = va/vb
        self.assertEqual(len(v_d_v), len(va))
        for i in range(len(va)):
            self.assertAlmostEqual(v_d_v[i].value(0), va[i].value(0)/vb[i].value(0))

        # vector add vector
        v_a_v = va + vb
        self.assertEqual(len(v_a_v), len(va))
        for i in range(len(va)):
            self.assertAlmostEqual(v_a_v[i].value(0), va[i].value(0) + vb[i].value(0))

        # vector sub vector
        v_s_v = va - vb
        self.assertEqual(len(v_s_v), len(va))
        for i in range(len(va)):
            self.assertAlmostEqual(v_s_v[i].value(0), va[i].value(0) - vb[i].value(0))

        # vector unary minus
        v_u = - va
        self.assertEqual(len(v_u), len(va))
        for i in range(len(va)):
            self.assertAlmostEqual(v_u[i].value(0), -va[i].value(0))

        # integral functions, just to verify exposure works, and one value is according to spec.
        ta2 = api.TimeAxis(t0, dt*24, n//24)
        v_avg = v.average(ta2)
        v_int = v.integral(ta2)
        v_acc = v.accumulate(ta2)
        v_sft = v.time_shift(dt*24)
        self.assertIsNotNone(v_avg)
        self.assertIsNotNone(v_int)
        self.assertIsNotNone(v_acc)
        self.assertIsNotNone(v_sft)
        self.assertAlmostEqual(v_avg[0].value(0), 1.0)
        self.assertAlmostEqual(v_int[0].value(0), 86400.0)
        self.assertAlmostEqual(v_acc[0].value(0), 0.0)
        self.assertAlmostEqual(v_sft[0].time(0), t0 + dt*24)

        # min/max functions
        min_v_double = va.min(-1000.0)
        max_v_double = va.max(1000.0)
        self.assertAlmostEqual(min_v_double[0].value(0), -1000.0)
        self.assertAlmostEqual(max_v_double[0].value(0), +1000.0)
        min_v_double = api.min(va, -1000.0)
        max_v_double = api.max(va, +1000.0)
        self.assertAlmostEqual(min_v_double[0].value(0), -1000.0)
        self.assertAlmostEqual(max_v_double[0].value(0), +1000.0)
        # c = 10.0
        c1000 = 100.0*c
        min_v_double = va.min(-c1000)
        max_v_double = va.max(c1000)
        self.assertAlmostEqual(min_v_double[0].value(0), -c1000.value(0))
        self.assertAlmostEqual(max_v_double[0].value(0), c1000.value(0))
        min_v_double = api.min(va, -c1000)
        max_v_double = api.max(va, c1000)
        self.assertAlmostEqual(min_v_double[0].value(0), -c1000.value(0))
        self.assertAlmostEqual(max_v_double[0].value(0), c1000.value(0))

        v1000 = va*1000.0
        min_v_double = va.min(-v1000)
        max_v_double = va.max(v1000)
        self.assertAlmostEqual(min_v_double[0].value(0), -v1000[0].value(0))
        self.assertAlmostEqual(max_v_double[0].value(0), v1000[0].value(0))
        min_v_double = api.min(va, -v1000)
        max_v_double = api.max(va, v1000)
        self.assertAlmostEqual(min_v_double[0].value(0), -v1000[0].value(0))
        self.assertAlmostEqual(max_v_double[0].value(0), v1000[0].value(0))

        # finally, test that exception is raised if we try to multiply two unequal sized vectors

        try:
            x = v_clone*va
            self.assertTrue(False, 'We expected exception for unequal sized ts-vector op')
        except RuntimeError as re:
            pass

    def test_ts_extend(self):
        t0 = api.utctime_now()
        dt = api.deltahours(1)
        n = 512
        ta_a = api.TimeAxisFixedDeltaT(t0, dt, 2*n)
        ta_b = api.TimeAxisFixedDeltaT(t0 + n*dt, dt, 2*n)
        ta_c = api.TimeAxisFixedDeltaT(t0 + 2*n*dt, dt, 2*n)
        ta_d = api.TimeAxisFixedDeltaT(t0 + 3*n*dt, dt, 2*n)

        a = api.TimeSeries(ta=ta_a, fill_value=1.0, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)
        b = api.TimeSeries(ta=ta_b, fill_value=2.0, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)
        c = api.TimeSeries(ta=ta_c, fill_value=4.0, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)
        d = api.TimeSeries(ta=ta_d, fill_value=8.0, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)

        # default behavior: extend from end of a
        ac = a.extend(c)

        for i in range(2*n):  # valus from first ts
            self.assertEqual(ac(t0 + i*dt), 1.0)
        for i in range(2*n):  # values from extension ts
            self.assertEqual(ac(t0 + (i + 2*n)*dt), 4.0)

        # default behavior: extend from end of a, fill gap with nan
        ad = a.extend(d)

        for i in range(2*n):  # values from first
            self.assertEqual(ad(t0 + i*dt), 1.0)
        for i in range(n):  # gap
            self.assertTrue(math.isnan(ad(t0 + (i + 2*n)*dt)))
        for i in range(2*n):  # extension
            self.assertEqual(ad(t0 + (i + 3*n)*dt), 8.0)

        # split at the first value of d instead of last of c
        cd = c.extend(d, split_policy=api.extend_split_policy.RHS_FIRST)

        for i in range(n):  # first, only until the extension start
            self.assertEqual(cd(t0 + (2*n + i)*dt), 4.0)
        for i in range(2*n):  # extension
            self.assertEqual(cd(t0 + (3*n + i)*dt), 8.0)

        # split at a given time step, and extend the last value through the gap
        ac = a.extend(c, split_policy=api.extend_split_policy.AT_VALUE, split_at=(t0 + dt*n//2),
                      fill_policy=api.extend_fill_policy.USE_LAST)

        for i in range(n//2):  # first, only until the given split value
            self.assertEqual(ac(t0 + i*dt), 1.0)
        for i in range(3*n//2):  # gap, uses last value before gap
            self.assertEqual(ac(t0 + (n//2 + i)*dt), 1.0)
        for i in range(2*n):  # extension
            self.assertEqual(ac(t0 + (2*n + i)*dt), 4.0)

        # split at the beginning of the ts to extend when the extension start before it
        cb = c.extend(b, split_policy=api.extend_split_policy.AT_VALUE, split_at=(t0 + 2*n*dt))

        for i in range(n):  # don't extend before
            self.assertTrue(math.isnan(cb(t0 + (n + i)*dt)))
        for i in range(n):  # we split at the beginning => only values from extension
            self.assertEqual(cb(t0 + (2*n + i)*dt), 2.0)
        for i in range(n):  # no values after extension
            self.assertTrue(math.isnan(cb(t0 + (3*n + i)*dt)))

        # extend with ts starting after the end, fill the gap with a given value
        ad = a.extend(d, fill_policy=api.extend_fill_policy.FILL_VALUE, fill_value=5.5)

        for i in range(2*n):  # first
            self.assertEqual(ad(t0 + i*dt), 1.0)
        for i in range(n):  # gap, filled with 5.5
            self.assertEqual(ad(t0 + (2*n + i)*dt), 5.5)
        for i in range(2*n):  # extension
            self.assertEqual(ad(t0 + (3*n + i)*dt), 8.0)

    def test_extend_vector_of_timeseries(self):
        t0 = api.utctime_now()
        dt = api.deltahours(1)
        n = 512

        tsvector = api.TsVector()

        ta = api.TimeAxisFixedDeltaT(t0 + 3*n*dt, dt, 2*n)

        tsvector.push_back(api.TimeSeries(
            ta=api.TimeAxisFixedDeltaT(t0, dt, 2*n),
            fill_value=1.0, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE))
        tsvector.push_back(api.TimeSeries(
            ta=api.TimeAxisFixedDeltaT(t0 + 2*n*dt, dt, 2*n),
            fill_value=2.0, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE))

        extension = api.TimeSeries(ta=ta, fill_value=8.0, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)

        # extend after all time-series in the vector
        extended_tsvector = tsvector.extend_ts(extension)

        # assert first element
        for i in range(2*n):
            self.assertEqual(extended_tsvector[0](t0 + i*dt), 1.0)
        for i in range(n):
            self.assertTrue(math.isnan(extended_tsvector[0](t0 + (2*n + i)*dt)))
        for i in range(2*n):
            self.assertEqual(extended_tsvector[0](t0 + (3*n + i)*dt), 8.0)

        # assert second element
        for i in range(2*n):
            self.assertEqual(extended_tsvector[1](t0 + (2*n + i)*dt), 2.0)
        for i in range(n):
            self.assertEqual(extended_tsvector[1](t0 + (4*n + i)*dt), 8.0)

        tsvector_2 = api.TsVector()
        tsvector_2.push_back(api.TimeSeries(
            ta=api.TimeAxisFixedDeltaT(t0 + 2*n*dt, dt, 4*n),
            fill_value=10.0, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE))
        tsvector_2.push_back(api.TimeSeries(
            ta=api.TimeAxisFixedDeltaT(t0 + 4*n*dt, dt, 4*n),
            fill_value=20.0, point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE))

        # extend each element in tsvector by the corresponding element in tsvector_2
        extended_tsvector = tsvector.extend_ts(tsvector_2)

        # assert first element
        for i in range(2*n):
            self.assertEqual(extended_tsvector[0](t0 + i*dt), 1.0)
        for i in range(4*n):
            self.assertEqual(extended_tsvector[0](t0 + (2*n + i)*dt), 10.0)

        # assert second element
        for i in range(2*n):
            self.assertEqual(extended_tsvector[1](t0 + (2*n + i)*dt), 2.0)
        for i in range(4*n):
            self.assertEqual(extended_tsvector[1](t0 + (4*n + i)*dt), 20.0)

    def test_rating_curve_ts(self):
        t0 = api.utctime_now()
        ta = api.TimeAxis(t0, api.deltaminutes(30), 48*2)
        data = np.linspace(0, 10, ta.size())
        ts = api.TimeSeries(ta, data, api.POINT_INSTANT_VALUE)

        rcf1 = api.RatingCurveFunction()
        rcf1.add_segment(0, 1, 0, 1)
        rcf1.add_segment(api.RatingCurveSegment(5, 2, 0, 1))

        rcf2 = api.RatingCurveFunction()
        rcf2.add_segment(0, 3, 0, 1)
        rcf2.add_segment(api.RatingCurveSegment(8, 4, 0, 1))

        rcp = api.RatingCurveParameters()
        rcp.add_curve(t0, rcf1)
        rcp.add_curve(t0 + api.deltahours(24), rcf2)

        sts = api.TimeSeries("a")
        rcsts = sts.rating_curve(rcp)

        rcsts_blob = rcsts.serialize()
        rcsts_2 = api.TimeSeries.deserialize(rcsts_blob)

        self.assertTrue(rcsts_2.needs_bind())
        fbi = rcsts_2.find_ts_bind_info()
        self.assertEqual(len(fbi), 1)
        fbi[0].ts.bind(ts)
        rcsts_2.bind_done()
        self.assertFalse(rcsts_2.needs_bind())

        self.assertEqual(len(rcsts_2), len(ts))
        for i in range(rcsts_2.size()):
            expected = (1*ts.get(i).v if ts.get(i).v < 5 else 2*ts.get(i).v) if ts.get(i).t < t0 + api.deltahours(24) else (3*ts.get(i).v if ts.get(i).v < 8 else 4*ts.get(i).v)
            self.assertEqual(rcsts_2.get(i).t, ts.get(i).t)
            self.assertEqual(rcsts_2.get(i).v, expected)

    def test_krls_ts(self):
        t0 = api.utctime_now()
        ta = api.TimeAxis(t0, api.deltahours(1), 30*24)
        data = np.sin(np.linspace(0, 2*np.pi, ta.size()))
        ts_data = api.TimeSeries(ta, data, api.POINT_INSTANT_VALUE)

        ts = api.TimeSeries("a")
        ts_krls = ts.krls_interpolation(api.deltahours(3))

        ts_krls_blob = ts_krls.serialize()
        ts2_krls = api.TimeSeries.deserialize(ts_krls_blob)

        self.assertTrue(ts2_krls.needs_bind())
        fbi = ts2_krls.find_ts_bind_info()
        self.assertEqual(len(fbi), 1)
        fbi[0].ts.bind(ts_data)
        ts2_krls.bind_done()
        self.assertFalse(ts2_krls.needs_bind())

        self.assertEqual(len(ts2_krls), len(ts_data))
        for i in range(len(ts2_krls)):
            self.assertAlmostEqual(ts2_krls.values[i], ts_data.values[i], places=1)

    def test_ts_get_krls_predictor(self):
        t0 = api.utctime_now()
        ta = api.TimeAxis(t0, api.deltahours(1), 30*24)
        data = np.sin(np.linspace(0, 2*np.pi, ta.size()))
        ts_data = api.TimeSeries(ta, data, api.POINT_INSTANT_VALUE)

        ts = api.TimeSeries("a")

        try:
            ts.get_krls_predictor()
            self.fail("should not be able to get predictor for unbound")
        except: pass

        fbi = ts.find_ts_bind_info()
        fbi[0].ts.bind(ts_data)
        ts.bind_done()

        pred = ts.get_krls_predictor(api.deltahours(3))

        ts_krls = pred.predict(ta)
        self.assertEqual(len(ts_krls), len(ts_data))
        ts_mse = pred.mse_ts(ts_data)
        self.assertEqual(len(ts_mse), len(ts_data))
        for i in range(len(ts_krls)):
            self.assertAlmostEqual(ts_krls.values[i], ts_data.values[i], places=1)
            self.assertAlmostEqual(ts_mse.values[i], 0, places=2)
        self.assertAlmostEqual(pred.predictor_mse(ts_data), 0, places=2)


if __name__ == "__main__":
    unittest.main()
