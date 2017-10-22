# import unittest
# from os import path
# import os
import numpy as np
from netCDF4 import Dataset
# from netCDF4 import buffer
from pyproj import Proj
# from pyproj import transform

# from shyft import shyftdata_dir
# from shyft.repository.netcdf.geo_ts_repository import GeoTsRepository
# from shyft.repository.netcdf.yaml_config import YamlContent
#from shyft.api import Calendar
# from shyft.api import UtcPeriod
# from shyft.api import TemperatureSource
from shyft.api import TimeSeries
from shyft.api import TimeAxis
from shyft.api import point_interpretation_policy as point_fx
#from shyft.api import deltahours, deltaminutes
from shyft.api import DoubleVector as dv
#from shyft.api import GeoPoint
from shyft.api import UtcTimeVector
from shyft.api import UtcPeriod

#from datetime import datetime

# from shyft import api
# from shyft import shyftdata_dir
from shyft.repository.netcdf.time_conversion import convert_netcdf_time


# from shyft.repository.netcdf.cf_geo_ts_repository import CFDataRepository, CFDataRepositoryError

#  http://cfconventions.org/Data/cf-standard-names/41/build/cf-standard-name-table.html


class CFInfo:
    def __init__(self, standard_name: str, units: str):
        self.standard_name = standard_name
        self.units = units


class TimeSeriesMetaInfoError(Exception):
    pass


class TimeSeriesMetaInfo:
    """
    Contain enough information to create a netcdf4 cf-compliant file
    that can be ready by shyft cf_geo_ts_repository
    """
    shyft_name_to_cf_info = {
        'precipitation': CFInfo('convective_precipitation_rate', 'mm/h'),
        'temperature': CFInfo('air_temperature', 'degC'),  # or K ??
        'wind_speed': CFInfo('wind_speed', 'm/s^2'),
        'wind_direction': CFInfo('wind_from_direction', 'degrees'),  #
        'relative_humidity': CFInfo('relative_humidity', ''),  # no unit (1.0)
        'radiation': CFInfo('global_radiation', 'W/m^2'),  # atmosphere_net_rate_of_absorption_of_shortwave_energy ?
        'discharge': CFInfo('discharge', 'm^3/s')  # m³/s
    }

    def __init__(self, name: str, ts_id: str, long_name: str, pos_x: float, pos_y: float, pos_z: float, epsg_id: int):
        if name not in self.shyft_name_to_cf_info.keys():
            msg = "Name '{}' not supported! must be one of: {}".format(name, ', '.join(
                list(self.shyft_name_to_cf_info.keys())))
            raise TimeSeriesMetaInfoError(msg)

        self.variable_name = name  if name != 'radiation' else 'global_radiation'
        self.shyft_name = name
        # shyft  (precipitiation|temperature|radiation|wind_speed|relative_humidity)
        self.timeseries_id = ts_id  # like /observed/temperature/<location>/ or any
        self.long_name = long_name  # descriptive name like measured temperature by sensor xyz
        self.x = pos_x
        self.y = pos_y
        self.z = pos_z
        self.epsg_id = epsg_id
        # compression
        self.least_significant_digit = 3  # 0.001 { m/s | deg C | mm/h | W/m^2 | rh} is  all ok
        self.zlib = True

    @property
    def units(self):
        return TimeSeriesMetaInfo.shyft_name_to_cf_info[self.shyft_name].units

    @property
    def standard_name(self):
        return TimeSeriesMetaInfo.shyft_name_to_cf_info[self.shyft_name].standard_name


class TimeSeriesStoreError(Exception):
    pass


class TimeSeriesStore:
    """
    Provides a netcdf time-series store for metered variables used in shyft,
    like temperature, precipitation, relative humidity, radiation, wind-speed
    as collected from remote sensing stations, or scada-system.

    Notice that the current approach uses one .nc file for each variable.
    (an alternative would be to have one file with one group for each variable)
    The reasons for just one variable-one-file approach is that it gives
    a simple file, that have no inter-dependencies with other variables that
    might or might not be sampled from the same remote sensing station.
    Also the sampling frequency for each signal will vary over time, so
    having a common time-variable would not make sense.

    The primary goal of this class is to provide functionality to
    create file, append or wipe out data from the signal variable.
    The file should be CF-compliant, and should fit directly into
    the already existing netcdf reader repositories, like
    cf_geo_ts_repository.py

    http://cfconventions.org/Data/cf-standard-names/41/build/cf-standard-name-table.html


    """

    def __init__(self, file_path: str, ts_meta_info: TimeSeriesMetaInfo):
        """
        Constructs a TimeSeriesStore with specified path and meta-description.

        :param file_path: The path to the netcdf file (does not have to exist)
        :param ts_meta_info: Information describing the variable (so that a nc *could* be created if needed)
        """
        self.file_path = file_path
        self.ts_meta_info = ts_meta_info

    def create_new_file(self):
        """
        creates a new netcdf file, and initializes the file with
        positional and time variables
        plus the variable as described by self.ts_meta_info
        :return: None
        """

        with Dataset(self.file_path, 'w') as ds:
            ds.Conventions = 'CF-1.6'
            # dimensions
            ds.createDimension('station', 1)
            ds.createDimension('time', None)

            # standard variables
            # Coordinate Reference System
            crs = ds.createVariable('crs', 'i4')
            epsg_spec = 'EPSG:{0}'.format(self.ts_meta_info.epsg_id)
            crs.epsg_code = epsg_spec
            crs.proj4 = Proj(init=epsg_spec).srs  # shyft expect crs.proj4 to exist
            crs.grid_mapping_name = 'transverse_mercator'

            ts_id = ds.createVariable('series_name', 'str', ('station',))
            ts_id.long_name = 'timeseries_id'
            ts_id.cf_role = 'timeseries_id'
            # ts_id.units = ''

            time = ds.createVariable('time', 'i8', ('time',), least_significant_digit=1, zlib=True)
            time.long_name = 'time'
            time.units = 'seconds since 1970-01-01 00:00:00 +00:00'
            time.calendar = 'gregorian'

            x = ds.createVariable('x', 'f8', ('station',))
            x.axis = 'X'
            x.standard_name = 'projection_x_coordinate'
            x.units = 'm'

            y = ds.createVariable('y', 'f8', ('station',))
            y.axis = 'Y'
            y.standard_name = 'projection_y_coordinate'
            y.units = 'm'

            z = ds.createVariable('z', 'f8', ('station',))
            z.axis = 'Z'
            z.standard_name = 'height'
            z.long_name = 'height above mean sea level'
            z.units = 'm'

            v = ds.createVariable(self.ts_meta_info.variable_name, 'f8',
                                  dimensions=('time', 'station'),
                                  zlib=self.ts_meta_info.zlib)

            v.units = self.ts_meta_info.units
            v.standard_name = self.ts_meta_info.standard_name
            v.long_name = self.ts_meta_info.long_name
            v.coordinates = 'y x z'
            v.grid_mapping = 'crs'
            x[0] = self.ts_meta_info.x
            y[0] = self.ts_meta_info.y
            z[0] = self.ts_meta_info.z
            # ds.flush()

    def append_ts_data(self, time_series: TimeSeries):
        """
        ensure that the data-file content
        are equal to time_series for the time_series.time_axis.total_period().
        If needed, create and update the file meta-data.
        :param time_series:
        :return:
        """
        period = time_series.total_period()
        n_new_val = time_series.size()
        crop_data = False
        time_series_cropped = None

        with Dataset(self.file_path, 'a') as ds:
            # read time, from ts.time_axis.start()
            #  or last value of time
            # then consider if we should fill in complete time-axis ?
            #
            # figure out the start-index,
            # then
            # ds.time[startindex:] = ts.time_axis.numpy values
            # ds.temperature[startindex:] = ts.values.to_numpy()
            #
            # or if more advanced algorithm,
            #  first read
            #  diff
            #   result -> delete range, replace range, insert range..
            time_variable = 'time'
            time = ds.variables.get(time_variable, None)

            if time is None:
                raise TimeSeriesStoreError('Something is wrong with the dataset. time not found.')
            var = ds.variables.get(self.ts_meta_info.variable_name, None)

            if var is None:
                raise TimeSeriesStoreError('Something is wrong with the dataset. variable {0} not found.'.format(
                    self.ts_meta_info.variable_name))

            if len(time):
                time_utc = convert_netcdf_time(time.units, time)

                idx_min = np.searchsorted(time_utc, period.start, side='left')
                idx_max = np.searchsorted(time_utc, period.end,
                                          side='left')  # use 'left' since period.end = time_point(last_value)+dt
                idx_data_end = idx_min + n_new_val
                # print('indices ', idx_min, idx_max, idx_data_end, len(time))
                # move data if we are overlap or new data`s time before saved time:
                if idx_min < len(time_utc) and idx_max < len(time_utc) and idx_max - idx_min != n_new_val:
                    # print('In moving condition ', idx_max - idx_min, n_new_val)
                    idx_last = len(time_utc)
                    time[idx_data_end:] = time[idx_max:idx_last]
                    var[idx_data_end:, 0] = var[idx_max:idx_last, 0]
                # insert new data
                time[idx_min:idx_data_end] = time_series.time_axis.time_points[:-1]
                var[idx_min:idx_data_end, 0] = time_series.values.to_numpy()
                # crop all data which should not be there
                if idx_max - idx_min - n_new_val > 0:
                    idx_del_start = len(time) - idx_max + idx_min + n_new_val
                    # print("we need to delete something at the end ", idx_max - idx_min - n_new_val, idx_del_start)
                    crop_data = True
                    time_cropped = time[0:idx_del_start]
                    var_cropped = var[0:idx_del_start, 0]
                    last_time_point = 2 * time_cropped[-1] - time_cropped[-2]
                    # print(type(time_cropped[0]))
                    # print(UtcTimeVector.from_numpy(time_cropped.astype(np.int64)).to_numpy())
                    ta = TimeAxis(UtcTimeVector.from_numpy(time_cropped.astype(np.int64)), int(last_time_point))
                    # print(var_cropped)
                    # print(type(var_cropped))
                    time_series_cropped = TimeSeries(ta, dv.from_numpy(var_cropped), point_fx.POINT_INSTANT_VALUE)  # TODO: is this right policy?

            else:
                time[:] = time_series.time_axis.time_points[:-1]
                var[:, 0] = time_series.values.to_numpy()

            # for i, (t, val) in enumerate(zip(time[:], var[:])):
            #     print('{:<4} : {} - {} - {}'.format(i, datetime.fromtimestamp(t), val[0], type(val[0])))
            ds.sync()

        if crop_data and time_series_cropped:
            self.create_new_file()
            self.append_ts_data(time_series_cropped)

    def remove_tp_data(self, period: UtcPeriod):
        """
        delete data given within the time_period

        :param time_period:
        :return:
        """
        time_series_cropped = None

        with Dataset(self.file_path, 'a') as ds:
            # 1. load the data
            time_variable = 'time'
            time = ds.variables.get(time_variable, None)

            if time is None:
                raise TimeSeriesStoreError('Something is wrong with the dataset. time not found.')
            var = ds.variables.get(self.ts_meta_info.variable_name, None)

            if var is None:
                raise TimeSeriesStoreError('Something is wrong with the dataset. variable {0} not found.'.format(
                    self.ts_meta_info.variable_name))

            if len(time):
                # 2. get indices of the data to delete
                time_utc = convert_netcdf_time(time.units, time)

                idx_min = np.searchsorted(time_utc, period.start, side='left')
                idx_max = np.searchsorted(time_utc, period.end, side='right')

                # check if there is data outside the range
                if idx_max - idx_min != len(time):
                    # print('indices ', idx_min, idx_max, len(time))
                    # 3. crop the data array
                    time_cropped = np.append(time[0:idx_min], time[idx_max:])
                    var_cropped = np.append(var[0:idx_min], var[idx_max:])
                    last_time_point = 2 * time_cropped[-1] - time_cropped[-2]
                    # print(type(time_cropped[0]))
                    # print(UtcTimeVector.from_numpy(time_cropped.astype(np.int64)).to_numpy())
                    ta = TimeAxis(UtcTimeVector.from_numpy(time_cropped.astype(np.int64)), int(last_time_point))
                    # print(var_cropped)
                    # print(type(var_cropped))
                    time_series_cropped = TimeSeries(ta, dv.from_numpy(var_cropped), point_fx.POINT_INSTANT_VALUE)  # TODO: is this correct point policy?

        # 4. save the cropped data
        self.create_new_file()
        if time_series_cropped:
            self.append_ts_data(time_series_cropped)
