# -*- coding: utf-8 -*-
# Please notice that this file depends on Statkraft specific modules for integration with the Powel SMG System
# as such, not usable in OpenSource, but serve as example how to create repositories that uses services provided
# by other systems.
# In statkraft we use pythonnet to add in .NET functionality that provide a safe and efficient bridge to the Powel SMG system
# sigbjorn.helset@statkraft.com

from ssa.timeseriesrepository import TimeSeriesRepositorySmg as repository
from ssa.environment import SMG_PREPROD as PREPROD
from ssa.environment import SMG_PROD as PROD
from ssa.environment import SmgEnvironment
from Statkraft import Time
from Statkraft.Time import UtcTime,Period,Calendar
from datetime import datetime
from datetime import timedelta

import clr
import System
import System.Collections.Generic
from System import DateTime, TimeSpan
from System.Collections.Generic import List,IList,Dictionary,IDictionary
from Statkraft.XTimeSeries import MetaInfo, PointTimeStepConstraint, TsIdentity,ITimeSeries
from Statkraft.ScriptApi import TsAsVector,TimeSystemReference,SsaTimeSeries
from shyft import api
import numpy as np

import abc
#from shyft.repository.base_repository import BaseRepository

#from shyft.repository.base_repository import BaseTimeSeriesRepository


class SmgDataError(Exception):
    pass


class SmgDataFetcher(object):

    def __init__(self, env, t_start, t_end, names=None, ids=None):
        if names == ids == None:
            raise SmgDataError("Please provide either names or ids")
        if names != None and ids != None:
            raise SmgDataError("Please provide either names or ids")
        if not isinstance(t_start, datetime) or not isinstance(t_end, datetime):
            raise SmgDataError("t_start and t_end must be datetime.datetime instances")

        self.env = env
        self.keys = names if names else ids
        self.keys_are_names = True if names else False
        self.ids = ids
        self.t_start = t_start - timedelta(2)
        self.t_end = t_end + timedelta(2)

    def fetch(self, **kwargs):
        """Open a connection to the SMG database and fetch all the time series given in self.keys.
        Return the result as a dictionary of dictionaries."""
        result = {}
        raw_data = []
        with repository(self.env) as tsr:
            raw_data = tsr.readRawPoints(self.t_start, self.t_end, self.keys, useNameAsId=self.keys_are_names)
        if len(self.keys) != raw_data.Count:
            print "WARNING: Could only find {} out of {} requested timeseries".format(raw_data.Count, len(self.keys))
        for d in raw_data:
            times = [d.Time(i).ToUnixTime() for i in xrange(d.Count)]
            values = [d.Value(i).V for i in xrange(d.Count)]
            print min(values), max(values)
            print min(times), max(times)
            print "Getting timeseries {} with {} datapoints and end date {}".format(d.Name, len(times), str(datetime.utcfromtimestamp(times[-1])))
            key = d.Name if self.keys_are_names else d.Info.Id
            result[key] = {"times": times, "values": values}
        return result



class SmGTsRepository(object):
    def __init__(self, env):
        self.env=env

    def read(self,list_of_ts_id,period):
        """Open a connection to the SMG database and fetch all the time series given in list_of_ts_id.
        ts_id is currently the full unique name of the smg-ts. We could/should also support using
        unique number/keys instead. -more efficient, and more robust to namechanges.
        Return the result as a dictionary of shyft_ts."""
        if(not isinstance(period,api.UtcPeriod)):
           raise SmgDataError("period should be of type api.UtcPeriod")

        result = {}
        raw_data = []
        ListOf_TsIdentities=self._namelist_to_ListOf_TsIdentities(list_of_ts_id)
        ssa_period=self._make_ssa_Period_from_shyft_period(period)
        with repository(self.env) as tsr:
            raw_data = tsr.repo.ReadRawPoints(ListOf_TsIdentities,ssa_period)
        if len(list_of_ts_id) != raw_data.Count:
            print "WARNING: Could only find {} out of {} requested timeseries".format(raw_data.Count, len(list_of_ts_id))
        for d in raw_data:
            key = d.Name #todo : if self.keys_are_names else d.Info.Id
            result[key] = self._make_shyft_ts_from_ssa_ts(d)
        return result

    def store(self,timeseries_dict):
        """ Input the list of Enki result timeseries_dict,
            where the keys are the wanted SmG ts-path names
            and the values are Enki result api.shyft_timeseries_double, time-series.
            If the named time-series does not exist, create it.
            Then store time-series data to the named entities.
            
        """
        # 0. First, get the list of ts identities that Tss uses
        list_of_names=timeseries_dict.keys()
        ListOf_TsIdentities=self._namelist_to_ListOf_TsIdentities(list_of_names)
        ok=False
        with repository(self.env) as tss:
            # 1. We check if any of the tsnames are missing..
            exists_kv_pairs=tss.repo.Exists(ListOf_TsIdentities)
            missing_list= List[MetaInfo]([])
            # 2. We create those missing..
            for e in exists_kv_pairs:
                if e.Value == False:
                    tsid=e.Key
                    mi= MetaInfo()
                    mi.Identity=tsid
                    mi.Description='Automatically created by shyft '
                    # Here we might fill in some properties to the created timeseries
                    # e.g. unit, if we could figure out that
                    missing_list.Add(mi)
            if missing_list.Count > 0 : # Yes, something was missing, create them
                created_list=tss.repo.Create(missing_list)
            # 3. We store the datapoints (identity period, then  time,value)
            ssa_timeseries_list= List[SsaTimeSeries]([]) # This is what Tss Xts eats
            for name,shyft_ts in timeseries_dict.iteritems():
                ssa_ts=self._make_ssa_ts_from_shyft_ts(name,shyft_ts)
                ssa_timeseries_list.Add(ssa_ts)
            ok=tss.repo.Write(ssa_timeseries_list) # Write into SmG!
        return ok

    @staticmethod
    def _namelist_to_ListOf_TsIdentities(list_of_names):
        ''' 
        returns a .NET List<TsIdentity> from a list of names
        '''
        result = List[TsIdentity]([])
        for name in list_of_names:
            result.Add(TsIdentity(0, name))
        return result

    @staticmethod
    def _make_ssa_ts_from_shyft_ts(name,shyft_ts):
        ''' Geturns a SSaTimeSeries from shyft_ts '''
        t=np.array([shyft_ts.time(i) for i in xrange(shyft_ts.size()) ])
        v=np.array([shyft_ts.value(i) for i in xrange(shyft_ts.size()) ])
        q = np.zeros_like(t, dtype=np.int)

        numPoints = shyft_ts.size();
        tsv = TsAsVector(numPoints, TimeSystemReference.Unix1970Utc)
        p = Period(UtcTime.CreateFromUnixTime(t[0]), UtcTime.CreateFromUnixTime(t[-1]+3600))
        tsv.SetVectors(p, t, v, q)
        tsv.Name = name
        return SsaTimeSeries(tsv)
    
    @staticmethod
    def _make_shyft_ts_from_ssa_ts(ssa_ts):
        if(not isinstance(ssa_ts,SsaTimeSeries)):
            raise  SmgDataError("supplied ssa_ts should be of type SsaTimeSeries")
        tsv=ssa_ts.GetTsAsVector(TimeSystemReference.Unix1970Utc)
        tsfactory=api.TsFactory()
        #todo: this can be done much faster using clr direct accesss, https://mail.python.org/pipermail/pythondotnet/2014-May/001526.html
        times= api.UtcTimeVector.FromNdArray(np.fromiter(tsv.Time,dtype=np.long))
        values=api.DoubleVector.FromNdArray(np.fromiter(tsv.Value,dtype=np.float))
        ts_period=api.UtcPeriod(tsv.TotalPeriod.Start.ToUnixTime(),tsv.TotalPeriod.End.ToUnixTime())
        shyft_ts= tsfactory.create_time_point_ts(ts_period,times,values)
        return shyft_ts

    @staticmethod
    def _make_ssa_Period_from_shyft_period(shyft_period):
        if(not isinstance(shyft_period,api.UtcPeriod)):
            raise SmgDataError("shyft_period must be of type api.UtcPeriod")
        return Period(UtcTime.CreateFromUnixTime(shyft_period.start),UtcTime.CreateFromUnixTime(shyft_period.end))

import unittest

class TestSmgRepository(unittest.TestCase):
    def setUp(self):
        pass

    def test_namelist_to_ListOf_TsIdentities(self):
        ds=SmGTsRepository(PREPROD)
        nl=[u'/ICC-test-v9.2',u'/test/b2',u'/test/c2']
        r=ds._namelist_to_ListOf_TsIdentities(nl)
        self.assertEqual(len(nl),r.Count)
        [ (self.assertEqual(nl[i],r[i].Name) and self.assertEqual(0,r[i].Id)) for i in xrange(len(nl))]

    def _create_shyft_ts(self):
        b=946684800 # 2000.01.01 00:00:00
        h=3600 #one hour in seconds
        values=np.array([1.0,2.0,3.0])
        shyft_ts_factory=api.TsFactory()
        return shyft_ts_factory.create_point_ts(len(values),b,h,api.DoubleVector(values))

    def test_make_ssa_ts_from_shyft_ts(self):
        ds=SmGTsRepository(PREPROD)
        ts_name=u'/abc'
        
        shyft_ts=self._create_shyft_ts()
        r=SmGTsRepository._make_ssa_ts_from_shyft_ts(ts_name,shyft_ts)
        self.assertEqual(r.Count,shyft_ts.size())
        self.assertEqual(r.Name,ts_name)
        [self.assertAlmostEquals(shyft_ts.value(i),r.Value(i).V) for i in xrange(shyft_ts.size())]
        [self.assertAlmostEquals(0,r.Value(i).Q) for i in xrange(shyft_ts.size())]
        [self.assertAlmostEquals(shyft_ts.time(i),r.Time(i).ToUnixTime()) for i in xrange(shyft_ts.size())]
    
    def test_make_shyft_ts_from_ssa_ts(self):
        shyft_ts1=self._create_shyft_ts()
        ssa_ts=SmGTsRepository._make_ssa_ts_from_shyft_ts(u'/just_a_test',shyft_ts1)
        shyft_ts=SmGTsRepository._make_shyft_ts_from_ssa_ts(ssa_ts)
        [self.assertAlmostEqual(shyft_ts.value(i),ssa_ts.Value(i).V) for i in xrange(shyft_ts.size())]
        [self.assertAlmostEqual(shyft_ts.time(i),ssa_ts.Time(i).ToUnixTime()) for i in xrange(shyft_ts.size())]
        
    
    #def test_is_type_of_BaseTimeSerieRepository(self):
    #    ds=SmGTsRepository(PREPROD)
    #    self.assertTrue(issubclass(SmGTsRepository,BaseTimeSeriesRepository))
    #    self.assertTrue(isinstance(ds,BaseTimeSeriesRepository))

    def test_store(self):
        ds=SmGTsRepository(PREPROD)
        nl=[u'/ICC-test-v9.2',u'/enki/test/b2',u'/enki/test/c2']
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
        self.assertEquals(r,True)
        # now read back the ts.. and verify it's there..
        read_period=api.UtcPeriod(t0,t0+3*dt)
        rts_list=ds.read(nl,read_period)
        self.assertIsNotNone(rts_list)
        c2=rts_list[nl[-1]]
        [self.assertAlmostEqual(c2.value(i),values[i]) for i in xrange(len(values))]

    def test_period(self):
        utc=api.Calendar()
        t0=utc.time(api.YMDhms(2014,01,01,00,00,00))
        t1=utc.time(api.YMDhms(2014,03,01,00,00,00))
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

if __name__ == "__main__":
    unittest.main()
    #t_start = datetime(2014, 3, 25)
    #t_end = datetime(2014, 3, 30)
    #df = SmgDataFetcher(PREPROD, t_start, t_end, names=[u"/ENKI/VTS/Tokke/Tokk-Bitdalen......-T1050S3BT0108"])
    #result=df.fetch()
    #print "Got result:",len(result)
