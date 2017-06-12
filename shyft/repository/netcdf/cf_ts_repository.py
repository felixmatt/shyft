from __future__ import absolute_import
from __future__ import print_function
from six import iteritems
from builtins import range


from os import path
import numpy as np
from netCDF4 import Dataset
from pyproj import Proj
from pyproj import transform
from shyft import api
from shyft import shyftdata_dir
from .. import interfaces
from shyft.repository.service.ssa_geo_ts_repository import TsRepository
from .time_conversion import convert_netcdf_time

class CFTsRepositoryError(Exception):
    pass


class CFTsRepository(TsRepository):
    """
    Repository for geo located timeseries stored in netCDF files.

    """
                     
    def __init__(self, file, var_type):
        """
        Construct the netCDF4 dataset reader for data from Arome NWP model,
        and initialize data retrieval.
        """
        #directory = params['data_dir']
        filename = path.expandvars(file)
        self.var_name = var_type
        
        #if not path.isdir(directory):
        #    raise CFDataRepositoryError("No such directory '{}'".format(directory))
        if not path.isabs(filename):
            # Relative paths will be prepended the data_dir
            filename = path.join(shyftdata_dir, filename)
        if not path.isfile(filename):
            raise CFTsRepositoryError("No such file '{}'".format(filename))
            
        self._filename = filename # path.join(directory, filename)


    def read(self,list_of_ts_id,period):
        if not period.valid():
           raise CFTsRepositoryError("period should be valid()  of type api.UtcPeriod")

        filename = self._filename

        if not path.isfile(filename):
            raise CFTsRepositoryError("File '{}' not found".format(filename))
        with Dataset(filename) as dataset:
            return self._get_data_from_dataset(dataset, period, list_of_ts_id)

    def _convert_to_timeseries(self, data, t, ts_id):
        ta = api.TimeAxisFixedDeltaT(int(t[0]), int(t[1]) - int(t[0]),  len(t))
        tsc = api.TsFactory().create_point_ts
        def construct(d):
            return tsc(ta.size(), ta.start, ta.delta_t,
                        api.DoubleVector.FromNdArray(d))
        ts = [construct(data[:,j]) for j in range(data.shape[-1])]
        return {k:v for k, v in zip(ts_id,ts)}

    def _get_data_from_dataset(self, dataset, utc_period, ts_id_to_extract):
        ts_id_key = [k for (k,v) in dataset.variables.items() if getattr(v,'cf_role',None)=='timeseries_id'][0]
        ts_id_in_file = dataset.variables[ts_id_key][:]

        time = dataset.variables.get("time", None)
        data = dataset.variables.get(self.var_name, None)
        dim_nb_series = [dim.name for dim in dataset.dimensions.values() if dim.name != 'time'][0]
        if not all([data, time]):
            raise CFTsRepositoryError("Something is wrong with the dataset."
                                           " hydroclim variable or time not found.")
        time = convert_netcdf_time(time.units,time)
        idx_min = np.searchsorted(time, utc_period.start, side='left')
        if time[idx_min] > utc_period.start and idx_min > 0:  # important ! ensure data *cover* the requested period, Shyft ts do take care of resolution etc.
            idx_min -= 1  # extend range downward so we cover the entire requested period

        idx_max = np.searchsorted(time, utc_period.end, side='right')

        if time[idx_max] < utc_period.end and idx_max +1 < len(time):
            idx_max += 1  # extend range upward so that we cover the requested period

        issubset = True if idx_max < len(time) - 1 else False
        time_slice = slice(idx_min, idx_max)

        mask = np.array([id in ts_id_to_extract for id in ts_id_in_file])

        dims = data.dimensions
        data_slice = len(data.dimensions)*[slice(None)]
        data_slice[dims.index(dim_nb_series)] = mask
        data_slice[dims.index("time")] = time_slice
        extracted_data = data[data_slice]
        if isinstance(extracted_data, np.ma.core.MaskedArray):
            extracted_data = extracted_data.filled(np.nan)

        return self._convert_to_timeseries(extracted_data, time[time_slice], ts_id_in_file[mask])