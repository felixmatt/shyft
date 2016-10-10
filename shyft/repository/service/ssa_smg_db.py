# -*- coding: utf-8 -*-
# Please notice that this file depends on Statkraft specific modules for integration with the Powel SMG System
# as such, not usable in OpenSource, but serve as example how to create repositories that uses services provided
# by other systems.
# In statkraft we use pythonnet to add in .NET functionality that provide a safe and efficient bridge to the Powel SMG system
# sigbjorn.helset@statkraft.com
from __future__ import print_function

from statkraft.ssa.timeseriesrepository import TimeSeriesRepositorySmg as repository
from statkraft.ssa.forecast import ForecastRepositorySmg
from statkraft.ssa.environment import SMG_PREPROD as PREPROD
from statkraft.ssa.environment import SMG_PROD as PROD
from statkraft.ssa.environment import FORECAST_PREPROD as FC_PROD
from statkraft.ssa.environment import FORECAST_PREPROD as FC_PREPROD
from statkraft.ssa.environment import SmgEnvironment,NetMetaInfoValidationSet
from Statkraft import Time
from Statkraft.Time import UtcTime,Period,Calendar
#from datetime import datetime
#from datetime import timedelta

import clr
import System
import System.Collections.Generic
from System import DateTime, TimeSpan
from System.Collections.Generic import List,IList,Dictionary,IDictionary
from Statkraft.XTimeSeries import MetaInfo, PointTimeStepConstraint, TsIdentity,ITimeSeries,TimeSeriesPointSegments,IPointSegment,PointSegment
from Statkraft.ScriptApi import TsAsVector,TimeSystemReference,TimeSeries
from shyft import api
from .ssa_geo_ts_repository import TsRepository
import numpy as np
from math import fabs
import abc

class SmgDataError(Exception):
    pass

class SmGTsRepository(TsRepository):
    def __init__(self, env,fc_env=None):
        self.env=env
        self.fc_env=fc_env

    def read(self,list_of_ts_id,period):
        """Open a connection to the SMG database and fetch all the time series given in list_of_ts_id.
        ts_id is currently the full unique name of the smg-ts. We could/should also support using
        unique number/keys instead. -more efficient, and more robust to namechanges.
        Return the result as a dictionary of shyft_ts."""
        if not period.valid():
           raise SmgDataError("period should be valid()  of type api.UtcPeriod")

        result = {}
        raw_data = []
        ListOf_TsIdentities=self._namelist_to_ListOf_TsIdentities(list_of_ts_id)
        ssa_period=self._make_ssa_Period_from_shyft_period(period)
        with repository(self.env) as tsr:
            raw_data = tsr.repo.ReadRawPoints(ListOf_TsIdentities,ssa_period)
        if len(list_of_ts_id) != raw_data.Count:
            print( "WARNING: Could only find {} out of {} requested timeseries".format(raw_data.Count, len(list_of_ts_id)))
        for d in raw_data:
            key = d.Name #todo : if self.keys_are_names else d.Info.Id
            result[key] = self._make_shyft_ts_from_ssa_ts(d)
        return result

    def read_forecast(self,list_of_fc_id,period):
        if not period.valid():
           raise SmgDataError("period should be valid()  of type api.UtcPeriod")
        result = {}
        ListOf_fc_identities=self._namelist_to_ListOf_TsIdentities(list_of_fc_id)
        ts_id_list=[]

        with repository(self.env) as tss:
            ts_id_list= tss.repo.GetIdentities (tss.repo.FindMetaInfo(ListOf_fc_identities))

        ssa_period=self._make_ssa_Period_from_shyft_period(period)
        fcr= ForecastRepositorySmg(self.fc_env)
        read_forecasts = fcr.repo.ReadForecast(ts_id_list,ssa_period)
        if ts_id_list.Count != read_forecasts.Count:
            print( "WARNING: Could only find {} out of {} requested timeseries".format(read_forecasts.Count, ts_id_list.Count))
        for fc_ts in read_forecasts:
            key=fc_ts.Name
            result[key]=self._make_shyft_ts_from_ssa_ts(fc_ts)
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
                    mi.Type=9000 # just a general time-series
                    # Here we might fill in some properties to the created timeseries
                    # e.g. unit, if we could figure out that
                    missing_list.Add(mi)
            if missing_list.Count > 0 : # Yes, something was missing, create them
                created_list=tss.repo.Create(missing_list,True)
                #TODO verify we got them created
            # fetch tsids from the names
            ts_id_list= tss.repo.GetIdentities (tss.repo.FindMetaInfo(ListOf_TsIdentities))
            name_to_ts_id={x.Name:x for x in ts_id_list}
            # 3. We store the datapoints (identity period, then  time,value)
            ssa_timeseries_list= List[TimeSeriesPointSegments]([]) # This is what Tss Xts eats
            for name,shyft_ts in iter(timeseries_dict.items()):
                ssa_ts=self._make_ssa_tsps_from_shyft_ts(name_to_ts_id[name],shyft_ts)
                ssa_timeseries_list.Add(ssa_ts)
            error_list=tss.repo.Write(ssa_timeseries_list,False) # Write into SmG!
            if error_list is None:ok=True
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
        ''' Returns a TimeSeries from shyft_ts '''
        t=np.array([shyft_ts.time(i) for i in range(shyft_ts.size()) ])
        v=np.array([shyft_ts.value(i) for i in range(shyft_ts.size()) ])
        q = np.zeros_like(t, dtype=np.int)

        numPoints = shyft_ts.size();
        tsv = TsAsVector(numPoints, TimeSystemReference.Unix1970Utc)
        p = Period(UtcTime.CreateFromUnixTime(t[0]), UtcTime.CreateFromUnixTime(t[-1]+3600))
        tsv.SetVectors(p, t, v, q)
        tsv.Name = name
        return TimeSeries(tsv)

    @staticmethod
    def _make_ssa_tsps_from_shyft_ts(ts_id,shyft_ts):
        ''' returns a TimeSeriesPointSegments from shyft_ts '''
        t=np.array([shyft_ts.time(i) for i in range(shyft_ts.size()) ])
        v=np.array([shyft_ts.value(i) for i in range(shyft_ts.size()) ])
        q = np.zeros_like(t, dtype=np.int)
        numPoints = shyft_ts.size();
        tsv = TsAsVector(numPoints, TimeSystemReference.Unix1970Utc)
        p = Period(UtcTime.CreateFromUnixTime(t[0]), UtcTime.CreateFromUnixTime(t[-1]+3600))
        tsv.SetVectors(p, t, v, q)
        tsps=TimeSeriesPointSegments()
        tsps.Identity=ts_id
        psl=List[IPointSegment]([])
        psl.Add(PointSegment(p,tsv.Points))
        tsps.PointSegments=psl
        return tsps

    @staticmethod
    def _make_shyft_ts_from_ssa_ts(ssa_ts):
        if not isinstance(ssa_ts, TimeSeries):
            raise SmgDataError("supplied ssa_ts should be of type TimeSeries")
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
        if not shyft_period.valid():
            raise SmgDataError("shyft_period must be of type api.UtcPeriod")
        return Period(UtcTime.CreateFromUnixTime(shyft_period.start),UtcTime.CreateFromUnixTime(shyft_period.end))
