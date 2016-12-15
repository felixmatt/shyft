# Please note that this file depends on Statkraft specific modules for integration with the Powel SMG System
# as such, not usable in OpenSource, but serve as example how to create repositories that uses services provided
# by other systems.
# In statkraft we use pythonnet to add in .NET functionality that provide a safe and efficient bridge to the Powel SMG system
# sigbjorn.helset@statkraft.com

from statkraft.ssa.timeseriesrepository import TimeSeriesRepositorySmg
from statkraft.ssa.forecast import ForecastRepositorySmg
from statkraft.ssa.environment import SMG_PREPROD as PREPROD
from statkraft.ssa.environment import SMG_PROD as PROD
from statkraft.ssa.environment import FORECAST_PREPROD as FC_PROD
from statkraft.ssa.environment import FORECAST_PREPROD as FC_PREPROD
from statkraft.ssa.environment import SmgEnvironment, NetMetaInfoValidationSet
from Statkraft import Time
from Statkraft.Time import UtcTime, Period, Calendar
import clr
import System
import System.Collections.Generic
from System import DateTime, TimeSpan
from System.Collections.Generic import List, IList, Dictionary, IDictionary
from Statkraft.XTimeSeries import MetaInfo, PointTimeStepConstraint, TsIdentity, ITimeSeries
from Statkraft.XTimeSeries import TimeSeriesPointSegments, IPointSegment, PointSegment
from Statkraft.ScriptApi import TsAsVector, TimeSystemReference
from Statkraft.ScriptApi import SsaTimeSeries # Only Available in statkraft-scriptapi=1.2.8
from shyft import api
from .ssa_geo_ts_repository import TsRepository
import numpy as np
from math import fabs
import abc


class SmgDataError(Exception):
    pass


class SmGTsRepository(TsRepository):
    def __init__(self, env, fc_env=None):
        self.env = env
        self.fc_env = fc_env


    def read(self, ts_names, period):
        """Open a connection to the SMG database and fetch all the time series given in ts_names.
        ts_id is currently the full unique name of the smg-ts. We could/should also support using
        unique number/keys instead. -more efficient, and more robust to name changes.
        Return the result as a dictionary of shyft_ts."""
        if not period.valid():
           raise SmgDataError("Period should be valid() of type api.UtcPeriod")
        shyft_series = {}
        ts_reads = []
        tsIdentities = self._namelist_to_ListOf_TsIdentities(ts_names)
        ssa_period = self._make_ssa_Period_from_shyft_period(period)
        with TimeSeriesRepositorySmg(self.env) as tsr:
            ts_reads = tsr.repo.ReadRawPoints(tsIdentities, ssa_period)
        if len(ts_names) != ts_reads.Count:
            print("WARNING: Could only find {} out of {} requested timeseries".format(ts_reads.Count, len(ts_names)))
        for ts in ts_reads:
            # TODO: self.keys_are_names else ts.Info.Id
            shyft_series[ts.Name] = self._make_shyft_ts_from_ssa_ts(ts)
        return shyft_series


    def read_forecast(self, fc_names, period):
        if not period.valid():
           raise SmgDataError("Period should be valid() of type api.UtcPeriod")
        fc_series = {}
        ts_ids = []
        fcIdentities = self._namelist_to_ListOf_TsIdentities(fc_names)
        with TimeSeriesRepositorySmg(self.env) as tsr:
            ts_ids = tsr.repo.GetIdentities(tsr.repo.FindMetaInfo(fcIdentities))
        ssa_period = self._make_ssa_Period_from_shyft_period(period)
        fcr = ForecastRepositorySmg(self.fc_env)  # TODO: implement __exit__ to use 'with as'
        fc_reads = fcr.repo.ReadForecast(ts_ids, ssa_period)
        if ts_ids.Count != fc_reads.Count:
            print("WARNING: Could only find {} out of {} requested timeseries".format(fc_reads.Count, ts_ids.Count))
        for fc_ts in fc_reads:
            fc_series[fc_ts.Name] = self._make_shyft_ts_from_ssa_ts(fc_ts)
        return fc_series


    def store(self, ts_dict):
        """ Input the list of Enki result ts_dict,
            where the keys are the wanted SmG ts-path names
            and the values are Enki result api.shyft_timeseries_double, time-series.
            If the named time-series does not exist, create it.
            Then store time-series data to the named entities.
            
        """
        # 0. Get the list of ts identities that tsr uses
        tsIdentities = self._namelist_to_ListOf_TsIdentities(ts_dict.keys())
        res = False
        with TimeSeriesRepositorySmg(self.env) as tsr:
            # 1. Check if any of the tsnames are missing
            exists_kv_pairs = tsr.repo.Exists(tsIdentities)
            missing_list= List[MetaInfo]([])
            # 2. Create those missing
            for e in exists_kv_pairs:
                if e.Value == False:
                    tsid = e.Key
                    mi = MetaInfo()
                    mi.Identity = tsid
                    mi.Description = 'Automatically created by shyft'
                    mi.Type = 9000 # General time-series
                    # Here we might fill in some properties to the created timeseries
                    # e.g. unit, if we could figure out that
                    missing_list.Add(mi)
            if missing_list.Count > 0:
                created_list = tsr.repo.Create(missing_list, True)
                # TODO: verify they have been created
            ts_ids = tsr.repo.GetIdentities(tsr.repo.FindMetaInfo(tsIdentities))
            ts_names = {x.Name: x for x in ts_ids}
            # 3. Store the datapoints (identity period, then time, value)
            ssaTimeSeries = List[TimeSeriesPointSegments]([]) # This is what tsr Xts eats
            for name, shyft_ts in iter(ts_dict.items()):
                ssa_ts = self._make_ssa_tsps_from_shyft_ts(ts_names[name], shyft_ts)
                ssaTimeSeries.Add(ssa_ts)
            errors = tsr.repo.Write(ssaTimeSeries, False) # Write into SmG
            if errors is None: 
                res = True
        return res


    @staticmethod
    def _namelist_to_ListOf_TsIdentities(names):
        ''' 
        returns a .NET List<TsIdentity> from a list of names
        '''
        tsIdentities = List[TsIdentity]([])
        for name in names:
            tsIdentities.Add(TsIdentity(0, name))
        return tsIdentities


    @staticmethod
    def _make_ssa_ts_from_shyft_ts(name, shyft_ts):
        ''' Returns a SsaTimeSeries from shyft_ts '''
        ts_size = shyft_ts.size();
        t = np.array([shyft_ts.time(i) for i in range(ts_size)])
        v = np.array([shyft_ts.value(i) for i in range(ts_size)])
        q = np.zeros_like(t, dtype=np.int)
        tsv = TsAsVector(ts_size, TimeSystemReference.Unix1970Utc)
        p = Period(UtcTime.CreateFromUnixTime(t[0]), UtcTime.CreateFromUnixTime(t[-1] + 3600))
        tsv.SetVectors(p, t, v, q)
        tsv.Name = name
        return SsaTimeSeries(tsv)


    @staticmethod
    def _make_ssa_tsps_from_shyft_ts(ts_id, shyft_ts):
        ''' returns a TimeSeriesPointSegments from shyft_ts '''
        ts_size = shyft_ts.size();
        t = np.array([shyft_ts.time(i) for i in range(ts_size)])
        v = np.array([shyft_ts.value(i) for i in range(ts_size)])
        q = np.zeros_like(t, dtype=np.int)
        tsv = TsAsVector(ts_size, TimeSystemReference.Unix1970Utc)
        p = Period(UtcTime.CreateFromUnixTime(t[0]), UtcTime.CreateFromUnixTime(t[-1] + 3600))
        tsv.SetVectors(p, t, v, q)
        ts_ps = TimeSeriesPointSegments()
        ts_ps.Identity = ts_id
        psList = List[IPointSegment]([])
        psList.Add(PointSegment(p, tsv.Points))
        ts_ps.PointSegments = psList
        return ts_ps


    @staticmethod
    def _make_shyft_ts_from_ssa_ts(ssa_ts):
        if not isinstance(ssa_ts, SsaTimeSeries):
            raise SmgDataError("Supplied ssa_ts should be of type SsaTimeSeries")
        tsv = ssa_ts.GetTsAsVector(TimeSystemReference.Unix1970Utc)
        ts_factory = api.TsFactory()
        # TODO: this can be done much faster using clr direct accesss, https://mail.python.org/pipermail/pythondotnet/2014-May/001526.html
        t = api.UtcTimeVector.FromNdArray(np.fromiter(tsv.Time, dtype=np.long))
        v = api.DoubleVector.FromNdArray(np.fromiter(tsv.Value, dtype=np.float))
        period = api.UtcPeriod(tsv.TotalPeriod.Start.ToUnixTime(), tsv.TotalPeriod.End.ToUnixTime())
        shyft_ts = ts_factory.create_time_point_ts(period, t, v)
        return shyft_ts


    @staticmethod
    def _make_ssa_Period_from_shyft_period(shyft_period):
        if not shyft_period.valid():
            raise SmgDataError("shyft_period must be of type api.UtcPeriod")
        return Period(UtcTime.CreateFromUnixTime(shyft_period.start), UtcTime.CreateFromUnixTime(shyft_period.end))
