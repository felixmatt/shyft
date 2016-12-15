import unittest

try: # we fail with a message on the import, to reduce noise outside statkraft environments
    from statkraft.ssa.environment import SMG_PREPROD as PREPROD
    from statkraft.ssa.environment import FORECAST_PREPROD as FC_PREPROD
    from statkraft.ssa.environment import SMG_PROD as PROD # Not for testing in Continuous Integration
    from statkraft.ssa.environment import FORECAST_PROD as FC_PROD # Not for testing in Continuous Integration
    from shyft import api
    from shyft.repository.service.ssa_smg_db import SmGTsRepository
    import numpy as np
    from math import fabs

    
    class TestSmgRepository(unittest.TestCase):
        """
        NOTICE: This test is for testing and internal(but important) part
        of the statkraft script-api component that provides timeseries,
        forecast and ensembles to the shyft eco-system.
        It will only run within Statkraft environment
        
        """
        def setUp(self):
            pass
    

        def test_namelist_to_ListOf_TsIdentities(self):
            ds = SmGTsRepository(PREPROD)
            nl = [u'/ICC-test-v9.2', u'/test/b2', u'/test/c2']
            r = ds._namelist_to_ListOf_TsIdentities(nl)
            self.assertEqual(len(nl), r.Count)
            [(self.assertEqual(nl[i], r[i].Name) and self.assertEqual(0, r[i].Id)) for i in range(len(nl))]
    

        def _create_shyft_ts(self):
            b = 946684800 # 2000.01.01 00:00:00
            h = 3600 # One hour in seconds
            v = np.array([1.0, 2.0, 3.0])
            return api.TsFactory().create_point_ts(len(v), b, h, api.DoubleVector(v))
    

        def test_make_xts_from_shyft_ts(self):
            shyft_ts = self._create_shyft_ts()
            ts_name = u'/make_xts'
            xts = SmGTsRepository._make_xts_from_shyft_ts(ts_name, shyft_ts)
            self.assertEqual(xts.Count, shyft_ts.size())
            self.assertEqual(xts.Name, ts_name)
            [self.assertAlmostEqual(xts.Value(i).V, shyft_ts.value(i)) for i in range(shyft_ts.size())]
            [self.assertAlmostEqual(xts.Value(i).Q, 0) for i in range(shyft_ts.size())]
            [self.assertAlmostEqual(xts.Time(i).ToUnixTime(), shyft_ts.time(i)) for i in range(shyft_ts.size())]
       
 
        def test_make_shyft_ts_from_xts(self):
            shyft_ts = self._create_shyft_ts()
            ts_name = u'/make_xts'
            xts = SmGTsRepository._make_xts_from_shyft_ts(ts_name, shyft_ts)
            test_ts = SmGTsRepository._make_shyft_ts_from_xts(xts)
            [self.assertAlmostEqual(test_ts.value(i), xts.Value(i).V) for i in range(test_ts.size())]
            [self.assertAlmostEqual(test_ts.time(i), xts.Time(i).ToUnixTime()) for i in range(test_ts.size())]
            

        def test_store(self):
            ds = SmGTsRepository(PREPROD)
            nl = [u'/shyft/test/a', u'/shyft/test/b', u'/shyft/test/c']
            t0 = 946684800 # time_t/unixtime 2000.01.01 00:00:00
            dt = 3600 # One hour in seconds
            values = np.array([1.0, 2.0, 3.0])
            shyft_ts_factory = api.TsFactory()
            shyft_result_ts = shyft_ts_factory.create_point_ts(len(values), t0, dt, api.DoubleVector(values))
            shyft_catchment_result = dict()
            shyft_catchment_result[nl[0]] = shyft_result_ts
            shyft_catchment_result[nl[1]] = shyft_result_ts
            shyft_catchment_result[nl[2]] = shyft_result_ts
            r = ds.store(shyft_catchment_result) 
            self.assertEqual(r, True)
            # Read back the ts.. and verify it's there
            read_period = api.UtcPeriod(t0, t0 + 3*dt)
            rts_list = ds.read(nl, read_period)
            self.assertIsNotNone(rts_list)
            c2 = rts_list[nl[-1]]
            [self.assertAlmostEqual(c2.value(i), values[i]) for i in range(len(values))]
    

        def test_read_forecast(self):
            utc = api.Calendar()
            ds = SmGTsRepository(PREPROD, FC_PREPROD)
            nl = [u'/LTMS-Abisko........-T0000A5P_EC00_ENS']
            t0 = utc.time(api.YMDhms(2016, 8, 1, 00, 00, 00))
            t1 = utc.time(api.YMDhms(2016, 8, 10, 00, 00, 00))
            p = api.UtcPeriod(t0, t1)
            fclist = ds.read_forecast(nl, p)
            self.assertIsNotNone(fclist)
            fc1 = fclist[u'/LTMS-Abisko........-T0000A5P_EC00_ENS']
            fc1_v = [fc1.value(i) for i in range(fc1.size())]
            # Test times here, left for manual inspection fc1_t = [utc.to_string(fc1.time(i)) for i in range(fc1.size())]
            self.assertIsNotNone(fc1_v)
            # Values read from SMG PREPROD:
            fc1_repeat = [1.449, 1.001, 0.423, 0.249, 0.252, 0.126, 0.068, 0.067, 0.189, 0.309, 0.300, 0.086, 0.055, 0.121, 0.149, 0.020, 0.014, 0.055, 0.222, 0.070, 0.094, 0.196, 0.132, 0.085, 0.087, 0.159, 0.158, 0.150, 0.214, 0.254, 0.239, 0.099, 0.089, 0.140, 0.154]
            fc1_v_read = [0.0] + [x for x in fc1_repeat for i in range(6)] + [0.079]*5
            [self.assertLess(fabs(fc1_v_read[i] - fc1_v[i]), 0.01, "{}: {} != {}".format(i, fc1_v_read[i], fc1_v[i])) for i in range(len(fc1_v_read))]
  
  
        def test_period(self):
            utc = api.Calendar()
            t0 = utc.time(api.YMDhms(2014, 1, 1, 00, 00, 00))
            t1 = utc.time(api.YMDhms(2014, 3, 1, 00, 00, 00))
            p = api.UtcPeriod(t0, t1)
            self.assertEqual(p.start, t0)
            self.assertEqual(p.end, t1)
            self.assertTrue(isinstance(p, api.UtcPeriod))
            ssa_period = SmGTsRepository._make_ssa_Period_from_shyft_period(p)
            t0ssa = ssa_period.Start.ToUnixTime()
            t1ssa = ssa_period.End.ToUnixTime()
            self.assertEqual(t0ssa, t0)
            self.assertEqual(t1ssa, t1)


except ImportError as ie:
    if 'statkraft' in str(ie):
        print('Test requires statkraft-scriptapi to run: {}'.format(ie))
    else:
        print('ImportError: {}'.format(ie))

        
if __name__ == '__main__':
    unittest.main()
