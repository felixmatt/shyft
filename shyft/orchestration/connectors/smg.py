import sys
import os
from collections import namedtuple
import datetime as dt
import numpy as np
import pdb

#This is internal to Statkraft
#sys.path.append('D:/Users/jfb/projects/statkraft')
import pyStatkraftScriptAPI.pyStatkraftScriptAPI as Ssa
TrueAverage = Ssa.Statkraft.XTimeSeries.Infrastructure.TransformMethod.TrueAverage

from base import BaseConnector


class Connector(BaseConnector):

    TsConfig = namedtuple('TsConfig', ['source_id', 'enki_networkname',
                                       'ensemble_id', 'enki_id',
                                       'series_type', 'update_freq',
                                       'lead_time','data_source'])

    def __init__(self, ssa_config=Ssa.CONFIG):
        '''Initializes a connection to SMG'''
        self.ssa_config = ssa_config
        # no.. wait until needed:: self.pTSR = Ssa.pyTimeSeriesRepository(Ssa.CONFIG)

    def _cleanup(self):
        '''Called at end of with statement, or if any exceptions is raised'''
        self.tss.dispose()
        #print("tss.dispose called")

    def _open_resource(self):
        '''Called at start of with statement'''
        self.tss = Ssa.TSS(self.ssa_config)
        return self

    def read_data(self, *args):
        return self._load_ts(*args)

    def _load_ts(self, datasets, run_start, deltaT, nsteps):
        """
        loads timeseries data from SMG
        Input:
        datasets - the dataset dictionary
        run_start - datetime
        run_stop - datetime
        """
        if not hasattr(self, 'tss'):
            import warnings
            warnings.warn("Every time you don't use a contextmanager, a kitten dies. Please help save the kittens by using the context manager for SMG")
            return
        time_series = {}

        # mvn comment:
        # Opening and closing of resource should be done with the context manager.
        # If we remove opening of resource from here, this method will fail,
        # unless _open_resource is called from contextmanager/manually first.
        # Removing cleanup because we should trust users to not be idiots,
        # and it allows multiple reads with the same connection.
        # Only one 'public' method that won't work witouth the contextmanager should be enough.

        #self.tss = Ssa.TSS(self.ssa_config)
        #try:  # ensure to have a well defined resource usage for the tss
        for datasetName, dataset in datasets.iteritems():
            # get a list of all datasets to read
            ts_config_list = [Connector.TsConfig(*ts) for ts in dataset]
            ts_name_list = [ts.source_id for ts in ts_config_list]
            if len(ts_name_list) == 0:
                continue
            
            ts_identities = Ssa.namelist_to_ts_identities(ts_name_list)
        
            # read in all data
            
            run_stop = run_start + dt.timedelta(seconds = deltaT * nsteps)
            period = Ssa.period(run_start, run_stop)
            #read_result = self.tss.ReadRawPoints(ts_identities, period)
            ## NOTE: Assuming we always want hourly data
            delta_tspan = Ssa.TimeSpan(0,0,deltaT)
            if "Output" in datasetName:
                time_series[datasetName] = []
                for tsc in ts_config_list:
                    time_series[datasetName].append((tsc,None))
            else:
                read_result = self.tss.ReadTransformed(ts_identities, period.Start, delta_tspan, nsteps, TrueAverage)
            
                assert(len(ts_config_list) == read_result.Count)
                time_series[datasetName] = []
                for tsc, ts in zip(ts_config_list, read_result):
                    time_series[datasetName].append((tsc, Ssa.ts_to_numpy(ts)))
        #finally:
        #    self.tss.dispose()
        
        return time_series


    
    def get_networks(self, result, enkirc):
        """ returns a dictionary of network timeseries, keyed by source_id 
        
        searches through the datasets and result from ENKI to populate """
        networks = {}

        datasets = enkirc.ConfigOps['data_sets']
        for datasetName, dataset in datasets.iteritems():
            # get a list of all datasets to read
            ts_config_list = [Connector.TsConfig(*ts) for ts in dataset]
            #ts_name_list = [ts.source_id for ts in ts_config_list]
            
            # read in all data
            # need to iterate over results to find those that match
            for ts in ts_config_list:
                if ts.series_type == 'OutputTs':
                    
                    for i in range(result.tsCount()):
                        ts_i=result.ts(i)
                        name = ts_i.getEnkiName()
                        enki_id = ts_i.id()
                        enki_ens_id = ts_i.getEnsembleMemberId()
                        if (ts.enki_networkname == name and ts.ensemble_id == enki_ens_id and ts.enki_id == enki_id):
                            values = np.array([float(ts_i.getValue(j)) for j in range(ts_i.nPoints())])
                            time = np.array([float(ts_i.getTime(k)) for k in range(ts_i.nPoints())])
                            networks[ts.source_id] = (time, values)          

        return networks
        


    def write_results(self, *args):
        return self._write_ts(*args)

    def _write_ts(self, networks):
        """
        writes the network timeseries to SMG
        Input:
        networks - dictionary of output timeseries from RunConfiguration.get_network_dict, keys are timeseries names in SMG
        
        """
        if not hasattr(self, 'tss'):
            import warnings
            warnings.warn("Every time you don't use a contextmanager, a kitten dies. Please help save the kittens by using the context manager for SMG")
            return
        networks = networks

        # write all data
        npts_list = []
        for network_sourceid, network_timeseries in networks.iteritems():
            
            time = network_timeseries[0]
            values = network_timeseries[1]

            npts_list.append(Ssa.NpTs(network_sourceid, values, time, q=None))
                    
        success = self.tss.npts_write(npts_list,unit='m3/sek', owner=0, type_=0, step='TIME')
        return success