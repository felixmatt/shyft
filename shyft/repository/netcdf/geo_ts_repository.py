"""
Module for reading netcdf dataset with a specific layout needed for an SHyFT run.
"""

from __future__ import absolute_import
from __future__ import print_function

import os

import numpy as np
from netCDF4 import Dataset

from ..interfaces import GeoTsRepository
from shyft import api, shyftdata_dir
#from ..utils import abs_datafilepath



def abs_datafilepath(filepath):
    """Get the absolute path for a data `filepath`.
    """
    if os.path.isabs(filepath):
        return filepath
    else:
        return os.path.join(shyftdata_dir, filepath)

class NetCDFGeoTsRepository(GeoTsRepository):
    
    def __init__(self,params, metstation_filepath,discharge_filepath):
        """
        Parameters:
        params is a structure with the following layout
            types list of
              type (precipitation|temperature|relative_humidity|wind_speed ..) 
              list of stations
                values: netcdf (group)/variablename of the values, e.g. /station1/precipitation
                time: netcdf variable for the time corresponding to values e.g. /station1/time
                location: netcdf variables tuple(x,y,z) /station.x, /station.y,/station1.z
            Yaml config files can be used to provide this information
            and we have the shyft.Repository.BaseYamlConfig that provides the structure from 
            yaml file (given the structure in the supplied yaml-files is ok
            
        """
        self._params=params
        self._metstation_filepath=metstation_filepath
        self._discharge_filepath=discharge_filepath
        
    @property
    def _stations_met(self):
        return abs_datafilepath(self._metstation_filepath)

    @property
    def _stations_discharge(self):
        return abs_datafilepath(self._discharge_filepath)

    def __repr__(self):
        return "%s(metstation_filepath=%r, discharge_filepath=%r)" % (
            self.__class__.__name__, self._metstation_filepath,  self._discharge_filepath)

    def _fetch_station_tseries(self, input_source, types, period):
        stations_ts = []
        with Dataset(self._stations_met) as dset:
            for type_ in types:
                if type_['type'] != input_source:
                    continue
                for station in type_['stations']:
                    tseries = {}
                    tpath=station['time'].split('/')[1:] # v1 time, the first one is empty
                    vpath=station['values'].split('/')[1:]
                    times = dset.groups[tpath[0]].variables[tpath[1]][:]#variables[station['time']][:]
                    imin = times.searchsorted(period.start, side='left')
                    imax = times.searchsorted(period.end, side='right')
                    # Get the indices of the valid period
                    tseries['values'] = dset.groups[vpath[0]].variables[vpath[1]][imin:imax]
                    tseries['time'] = times[imin:imax].astype(np.long).tolist()
                    coords = []
                    for loc in station['location'].split(","):
                        grname, axis = loc.split(".")
                        coords.append(getattr(dset.groups[grname.split('/')[1]], axis))
                    tseries['location'] = tuple(coords)
                    stations_ts.append(tseries)
        print(len(stations_ts), input_source, 'series found.')
        return stations_ts

    def get_timeseries(self, input_source_types, geo_location_criteria, utc_period):
        """Method for fetching the sources in NetCDF files.

        Parameters
        ----------
        input_source_types : dict
            A map between the data to be extracted and the data containers in shyft.api.
        geo_location_criteria: bbox + proj.ref ?
        utc_period : of type UtcPeriod

        Returns
        -------
        data: dict
            Shyft.api container for geo-located time series. Types are found from the 
            input_source_type.vector_t attribute.

        """
        #self.__dict__.update(params)
        data = dict()
        # Fill the data with actual values
        for input_source, source_api in input_source_types.iteritems():
            ts = self._fetch_station_tseries(input_source, self._params['types'], utc_period)
            assert type(ts) is list
            tsf = api.TsFactory()
            acc_data = []
            for station in ts:
                times = station['time']
                assert type(times) is list
                dt = times[1] - times[0] if len(times) > 1 else api.deltahours(1)
                total_period = api.UtcPeriod(times[0], times[-1] + dt)
                time_points = api.UtcTimeVector(times)
                time_points.push_back(total_period.end)
                values = station['values']
                value_points = api.DoubleVector.FromNdArray(values)
                api_ts = tsf.create_time_point_ts(total_period, time_points, value_points)
                data_source = source_api(api.GeoPoint(*station['location']), api_ts)
                acc_data.append(data_source)
            data[input_source] = source_api.vector_t(acc_data)
        return data
    
    def get_forecast(self, input_source_types,geo_location_criteria, utc_period):
        """
        Parameters:
        see get_timeseries
        semantics for utc_period: Get the forecast closest up to utc_period.start
        """
        raise NotImplementedError("get_forecast")
    
    def get_forecast_ensemble(self, input_source_types, geo_location_criteria,utc_period):
        raise NotImplementedError("get_forecast_ensemble")
