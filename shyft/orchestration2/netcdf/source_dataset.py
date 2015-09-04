"""
Module for reading dataset files needed for an SHyFT run.
"""

from __future__ import absolute_import
from __future__ import print_function

import os

import numpy as np
from netCDF4 import Dataset

from ..base_config import BaseSourceDataset
from shyft import api, shyftdata_dir
from ..utils import abs_datafilepath


class SourceDataset(BaseSourceDataset):

    @property
    def _stations_met(self):
        print("stations_met:", self.stations_met)
        return abs_datafilepath(self.stations_met)

    @property
    def _stations_discharge(self):
        return abs_datafilepath(self.stations_discharge)

    def __repr__(self):
        return "%s(stations_met=%r, stations_discharge=%r)" % (
            self.__class__.__name__, self.stations_met,  self.stations_discharge)

    def _fetch_station_tseries(self, input_source, types, period):
        stations_ts = []
        with Dataset(self._stations_met) as dset:
            for type_ in types:
                if type_['type'] != input_source:
                    continue
                for station in type_['stations']:
                    tseries = {}
                    times = dset[station['time']][:]
                    imin = times.searchsorted(period[0], side='left')
                    imax = times.searchsorted(period[1], side='right')
                    # Get the indices of the valid period
                    tseries['values'] = dset[station['values']][imin:imax]
                    tseries['time'] = times[imin:imax].astype(np.long).tolist()
                    coords = []
                    for loc in station['location'].split(","):
                        grname, axis = loc.split(".")
                        coords.append(getattr(dset[grname[1:]], axis))
                    tseries['location'] = tuple(coords)
                    stations_ts.append(tseries)
        print(len(stations_ts), input_source, 'series found.')
        return stations_ts

    def fetch_sources(self, input_source_types, data, params, period):
        """Method for fetching the sources in NetCDF files.

        Parameters
        ----------
        input_source_types : dict
            A map between the data to be extracted and the data containers in shyft.api.
        data : dict
            An geo-located time series shyft.api container.
        params : dict
            Additional parameters for locating the datasets.
        period : tuple
            A (start_time, stop_time) tuple that species the simulation period.

        """
        self.__dict__.update(params)
        # Fill the data with actual values
        for input_source, source_api in input_source_types.iteritems():
            ts = self._fetch_station_tseries(input_source, params['types'], period)
            assert type(ts) is list
            tsf = api.TsFactory()
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
                data[input_source].append(data_source)
        return data
