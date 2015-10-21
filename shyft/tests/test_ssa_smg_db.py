from __future__ import absolute_import, print_function
import unittest

try: # we fail with a message on the import, to reduce noise outside statkraft environments
    from statkraft.ssa.environment import SMG_PREPROD as PREPROD
    from statkraft.ssa.environment import FORECAST_PREPROD as FC_PREPROD
    from statkraft.ssa.environment import SMG_PROD as PROD
    from statkraft.ssa.environment import FORECAST_PROD as FC_PROD
    
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
            ds=SmGTsRepository(PREPROD)
            nl=[u'/ICC-test-v9.2',u'/test/b2',u'/test/c2']
            r=ds._namelist_to_ListOf_TsIdentities(nl)
            self.assertEqual(len(nl),r.Count)
            [ (self.assertEqual(nl[i],r[i].Name) and self.assertEqual(0,r[i].Id)) for i in range(len(nl))]
    
        def _create_shyft_ts(self):
            b=946684800 # 2000.01.01 00:00:00
            h=3600 #one hour in seconds
            values=np.array([1.0,2.0,3.0])
            shyft_ts_factory=api.TsFactory()
            return shyft_ts_factory.create_point_ts(len(values),b,h,api.DoubleVector(values))
    
        def test_make_ssa_ts_from_shyft_ts(self):
            #ds=SmGTsRepository(PREPROD)
            ts_name=u'/abc'
            
            shyft_ts=self._create_shyft_ts()
            r=SmGTsRepository._make_ssa_ts_from_shyft_ts(ts_name,shyft_ts)
            self.assertEqual(r.Count,shyft_ts.size())
            self.assertEqual(r.Name,ts_name)
            [self.assertAlmostEqual(shyft_ts.value(i),r.Value(i).V) for i in range(shyft_ts.size())]
            [self.assertAlmostEqual(0,r.Value(i).Q) for i in range(shyft_ts.size())]
            [self.assertAlmostEqual(shyft_ts.time(i),r.Time(i).ToUnixTime()) for i in range(shyft_ts.size())]
        
        def test_make_shyft_ts_from_ssa_ts(self):
            shyft_ts1=self._create_shyft_ts()
            ssa_ts=SmGTsRepository._make_ssa_ts_from_shyft_ts(u'/just_a_test',shyft_ts1)
            shyft_ts=SmGTsRepository._make_shyft_ts_from_ssa_ts(ssa_ts)
            [self.assertAlmostEqual(shyft_ts.value(i),ssa_ts.Value(i).V) for i in range(shyft_ts.size())]
            [self.assertAlmostEqual(shyft_ts.time(i),ssa_ts.Time(i).ToUnixTime()) for i in range(shyft_ts.size())]
            
        
    
        def test_store(self):
            ds=SmGTsRepository(PREPROD)
            nl=[u'/shyft/test/a',u'/shyft/test/b',u'/shyft/test/c'] #[u'/ICC-test-v9.2']
            t0=946684800 # time_t/unixtime 2000.01.01 00:00:00
            dt=3600 #one hour in seconds
            values=np.array([1.0,2.0,3.0])
            shyft_ts_factory=api.TsFactory()
            shyft_result_ts=shyft_ts_factory.create_point_ts(len(values),t0,dt,api.DoubleVector(values))
            shyft_catchment_result=dict()
            shyft_catchment_result[nl[0]]=shyft_result_ts
            shyft_catchment_result[nl[1]]=shyft_result_ts
            shyft_catchment_result[nl[2]]=shyft_result_ts
            r=ds.store(shyft_catchment_result) 
            self.assertEqual(r,True)
            # now read back the ts.. and verify it's there..
            read_period=api.UtcPeriod(t0,t0+3*dt)
            rts_list=ds.read(nl,read_period)
            self.assertIsNotNone(rts_list)
            c2=rts_list[nl[-1]]
            [self.assertAlmostEqual(c2.value(i),values[i]) for i in range(len(values))]
    
        def test_read_forecast(self):
            utc=api.Calendar()
            ds=SmGTsRepository(PROD,FC_PROD)
            nl=[u'/LTMS-Abisko........-T0000A5P_EC00_ENS',u'/LTMS-Abisko........-T0000A5P_EC00_E04',u'/Vikf-Tistel........-T0017A3P_MAN',u'/Vikf-Tistel........-T0000A5P_MAN' ] #[u'/ICC-test-v9.2']
            t0=utc.time(api.YMDhms(2015,10, 1,00,00,00))
            t1=utc.time(api.YMDhms(2015,10,10,00,00,00))
            p=api.UtcPeriod(t0,t1)
            fclist=ds.read_forecast(nl,p)
            self.assertIsNotNone(fclist)
            fc1=fclist[u'/LTMS-Abisko........-T0000A5P_EC00_E04']
            fc1_v=[fc1.value(i) for i in range(fc1.size())]
            # test times here, left for manual inspection here fc1_t=[utc.to_string(fc1.time(i)) for i in range(fc1.size())]
            self.assertIsNotNone(fc1_v)
            #values as read from preprod smg:
            fc1_v_expected=[0.00,0.33,0.33,0.33,0.33,0.33,0.33,0.08,0.08,0.08,0.08,0.08,0.08,0.16,0.16,0.16,0.16,0.16,0.16,0.11,0.11,0.11,0.11,0.11,0.11,0.47,0.47,0.47,0.47,0.47,0.47,0.15,0.15,0.15,0.15,0.15,0.15,0.12,0.12,0.12,0.12,0.12,0.12,0.20,0.20,0.20,0.20,0.20,0.20,0.14,0.14,0.14,0.14,0.14,0.14,0.02,0.02,0.02,0.02,0.02,0.02,0.01,0.01,0.01,0.01,0.01,0.01,0.00,0.00,0.00,0.00,0.00,0.00,0.09,0.09,0.09,0.09,0.09,0.09,0.10,0.10,0.10,0.10,0.10,0.10,0.08,0.08,0.08,0.08,0.08,0.08,0.11,0.11,0.11,0.11,0.11,0.11,0.23,0.23,0.23,0.23,0.23,0.23,0.03,0.03,0.03,0.03,0.03,0.03,0.01,0.01,0.01,0.01,0.01,0.01,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.03,0.03,0.03,0.03,0.03,0.03,0.06,0.06,0.06,0.06,0.06,0.06,0.14,0.14,0.14,0.14,0.14,0.14,0.13,0.13,0.13,0.13,0.13,0.13,0.10,0.10,0.10,0.10]
            [ self.assertLess(fabs(fc1_v_expected[i]-fc1_v[i]),0.01 ,"{}:{} !={}".format(i,fc1_v_expected[i],fc1_v[i]) ) for i in range(len(fc1_v_expected)) ]
    
        def test_period(self):
            utc=api.Calendar()
            t0=utc.time(api.YMDhms(2014, 1,1,00,00,00))
            t1=utc.time(api.YMDhms(2014,3, 1,00,00,00))
            p=api.UtcPeriod(t0,t1)
            self.assertEqual(p.start,t0)
            self.assertEqual(p.end,t1)
            #self.assertTrue(isinstance(t0, api.utctime))
            self.assertTrue(isinstance(p,api.UtcPeriod))
            ssa_period=SmGTsRepository._make_ssa_Period_from_shyft_period(p)
            t0ssa=ssa_period.Start.ToUnixTime()
            t1ssa=ssa_period.End.ToUnixTime()
            self.assertEqual(t0ssa,t0)
            self.assertEqual(t1ssa,t1)

except ImportError as ie:
    if 'statkraft' in str(ie):
        print("(Test require statkraft.script environment to run: {})".format(ie))
    else:
        print("ImportError: {}".format(ie))
        
if __name__ == '__main__':
    unittest.main()
