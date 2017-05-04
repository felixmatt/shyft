import unittest
from os import path
import os
import numpy as np
from netCDF4 import Dataset
#from netCDF4 import buffer
from pyproj import Proj
#from pyproj import transform

#from shyft import shyftdata_dir
#from shyft.repository.netcdf.geo_ts_repository import GeoTsRepository
#from shyft.repository.netcdf.yaml_config import YamlContent
from shyft.api import Calendar
#from shyft.api import UtcPeriod
#from shyft.api import TemperatureSource
from shyft.api import TimeSeries
from shyft.api import TimeAxis
from shyft.api import point_interpretation_policy as point_fx
from shyft.api import deltahours, deltaminutes
from shyft.api import DoubleVector as dv
from shyft.api import GeoPoint
from shyft.api import UtcTimeVector
from shyft.api import UtcPeriod

from datetime import datetime

#from shyft import api
#from shyft import shyftdata_dir
from shyft.repository.netcdf.time_conversion import convert_netcdf_time
from shyft.repository.netcdf.cf_geo_ts_repository import CFDataRepository

#  http://cfconventions.org/Data/cf-standard-names/41/build/cf-standard-name-table.html

class CFInfo:
    def __init__(self, standard_name: str, units: str):
        self.standard_name = standard_name
        self.units = units


class TimeSeriesMetaInfo:
    """
    Contain enough information to create a netcdf4 cf-compliant file
    that can be ready bey shyft cf_geo_ts_repository
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

    def __init__(self, name: str, ts_id: str, long_name: str, pos_x: float, pos_y: float, pos_z:float, epsg_id: int):
        self.variable_name = name  # shyft  (precipitiation|temperature|radiation|wind_speed|relative_humidity)
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
        return TimeSeriesMetaInfo.shyft_name_to_cf_info[self.variable_name].units

    @property
    def standard_name(self):
        return TimeSeriesMetaInfo.shyft_name_to_cf_info[self.variable_name].standard_name

    @staticmethod
    def temperature(ts_id: str, long_name: str, pos_x: float, pos_y: float, pos_z: float, epsg_id:int):
        return TimeSeriesMetaInfo('temperature', ts_id, long_name, pos_x, pos_y, pos_z, epsg_id)

class TimeSeriesStoreError(Exception):
    pass

class TimeSeriesStore:

    def __init__(self, file_path:str, ts_meta_info:TimeSeriesMetaInfo):
        self.file_path = file_path
        self.ts_meta_info = ts_meta_info

    def create_new_file(self):

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
            #ts_id.units = ''

            time = ds.createVariable('time', 'f8', ('time',), least_significant_digit=1, zlib=True)
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

            v = ds.createVariable(self.ts_meta_info.variable_name, 'i8',
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
            #ds.flush()



    def append_ts_data(self, time_series:TimeSeries):
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
                raise TimeSeriesStoreError('Something is wrong with the dataset. variable {0} not found.'.format(self.ts_meta_info.variable_name))

            if len(time):
                time_utc = convert_netcdf_time(time.units, time)

                idx_min = np.searchsorted(time_utc, period.start, side='left')
                idx_max = np.searchsorted(time_utc, period.end, side='left')  # use 'left' since period.end = time_point(last_value)+dt
                idx_data_end = idx_min+n_new_val
                print('indices ', idx_min, idx_max, idx_data_end, len(time))
                # move data if we are overlap or new data`s time before saved time:
                if idx_min < len(time_utc) and idx_max < len(time_utc) and idx_max-idx_min != n_new_val:
                    print('In moving condition ', idx_max-idx_min, n_new_val)
                    idx_last = len(time_utc)
                    time[idx_data_end:] = time[idx_max:idx_last]
                    var[idx_data_end:, 0] = var[idx_max:idx_last, 0]
                # insert new data
                time[idx_min:idx_data_end] = time_series.time_axis.time_points[:-1]
                var[idx_min:idx_data_end, 0] = time_series.values.to_numpy()
                # crop all data which should not be there
                if idx_max - idx_min - n_new_val > 0:
                    idx_del_start = len(time) - idx_max + idx_min + n_new_val
                    print("we need to delete something at the end ", idx_max - idx_min - n_new_val, idx_del_start)
                    crop_data = True
                    time_cropped = time[0:idx_del_start]
                    var_cropped = var[0:idx_del_start, 0]
                    last_time_point = 2*time_cropped[-1] - time_cropped[-2]
                    print(type(time_cropped[0]))
                    print(UtcTimeVector.from_numpy(time_cropped.astype(np.int64)).to_numpy())
                    ta = TimeAxis(UtcTimeVector.from_numpy(time_cropped.astype(np.int64)), int(last_time_point))
                    print(var_cropped)
                    print(type(var_cropped))
                    time_series_cropped = TimeSeries(ta, dv.from_numpy(var_cropped), point_fx=point_fx.POINT_AVERAGE_VALUE)

            else:
                time[:] = time_series.time_axis.time_points[:-1]
                var[:, 0] = time_series.values.to_numpy()

            for i, (t, val) in enumerate(zip(time[:], var[:])):
                print('{:<4} : {} - {} - {}'.format(i, datetime.fromtimestamp(t), val[0], type(val[0])))
            ds.sync()

        if crop_data and time_series_cropped:
            self.create_new_file()
            self.append_ts_data(time_series_cropped)

class NetCDFGeoTsRWTestCase(unittest.TestCase):
    """
    Verify that we correctly can read geo-located timeseries from a netCDF
    based file-store.
    """
    def _construct_from_test_data(self):
        #met = path.join(shyftdata_dir, "netcdf", "orchestration-testdata", "atnasjoen_met.nc")
        #dis = path.join(shyftdata_dir, "netcdf", "orchestration-testdata", "atnasjoen_discharge.nc")
        #map_cfg_file = path.join(path.dirname(__file__), "netcdf","datasets.yaml")
        #map_cfg = YamlContent(map_cfg_file)
        #params = map_cfg.sources[0]['params']  # yes, hmm.
        #return GeoTsRepository(params, met, dis)
        pass

    def test_can_create_cf_compliant_file(self):
        # create files
        test_file = path.join(path.abspath(os.curdir), 'shyft_test.nc')
        if path.exists(test_file):
            os.remove(test_file)
        # create meta info
        epsg_id = 32633
        x0 = 100000
        x1 = 200000
        y0 = 100000
        y1 = 200000
        x = 101000
        y = 101000
        z = 1200
        temperature = TimeSeriesMetaInfo.temperature('/observed/at_stn_abc/temperature', 'observed air temperature', x, y, z, epsg_id)

        # create time axis
        utc = Calendar()
        ta = TimeAxis(utc.time(2016, 1, 1), deltahours(1), 24)
        data = np.arange(0, ta.size(), dtype=np.float64)
        ts = TimeSeries(ta, dv.from_numpy(data), point_fx=point_fx.POINT_AVERAGE_VALUE)

        # save the first batch
        t_ds = TimeSeriesStore(test_file, temperature)
        t_ds.create_new_file()
        t_ds.append_ts_data(ts)

        # expected result
        ts_exp = ts

        # now read back the result using a *standard* shyft cf geo repository
        selection_criteria = {'bbox': [[x0, x1, x1, x0], [y0, y0, y1, y1]]}
        ts_dr = CFDataRepository(epsg_id, test_file, selection_criteria)
        # now read back 'temperature' that we know should be there
        rts_map = ts_dr.get_timeseries(['temperature'], ts_exp.total_period())

        # and verify that we get exactly back what we wanted.
        self.assertIsNotNone(rts_map)
        self.assertTrue('temperature' in rts_map)
        geo_temperature = rts_map['temperature']
        self.assertEqual(len(geo_temperature), 1)
        self.assertLessEqual(GeoPoint.distance2(geo_temperature[0].mid_point(), GeoPoint(x, y, z)), 1.0)
        # check if time axis is as expected
        self.assertEqual(geo_temperature[0].ts.time_axis, ts_exp.time_axis)
        self.assertTrue(np.allclose(geo_temperature[0].ts.time_axis.time_points, ts_exp.time_axis.time_points))
        self.assertEqual(geo_temperature[0].ts.point_interpretation(), point_fx.POINT_AVERAGE_VALUE)
        # check if variable data is as expected
        self.assertTrue(np.allclose(geo_temperature[0].ts.values.to_numpy(), ts_exp.values.to_numpy()))

        # --------------------------------------
        # Append data
        print("\n\n append at the end data")
        # create time axis
        ta = TimeAxis(utc.time(2016, 1, 2), deltahours(1), 48)
        ts = TimeSeries(ta, dv.from_numpy(np.arange(0, ta.size(), dtype=np.float64)), point_fx=point_fx.POINT_AVERAGE_VALUE)
        # save the data
        t_ds.append_ts_data(ts)

        # expected result
        ta = TimeAxis(utc.time(2016, 1, 1), deltahours(1), 72)
        data = np.append(data, np.arange(0, 48, dtype=np.float64))
        data = np.empty(72)
        data[:24] = np.arange(0, 24, dtype=np.float64)
        data[24:72] = np.arange(0, 48, dtype=np.float64)# <-- new data
        ts_exp = TimeSeries(ta, dv.from_numpy(data), point_fx=point_fx.POINT_AVERAGE_VALUE)

        # now read back the result using a *standard* shyft cf geo repository
        selection_criteria = {'bbox': [[x0, x1, x1, x0], [y0, y0, y1, y1]]}
        ts_dr = CFDataRepository(epsg_id, test_file, selection_criteria)
        # now read back 'temperature' that we know should be there
        rts_map = ts_dr.get_timeseries(['temperature'], ts_exp.total_period())

        # and verify that we get exactly back what we wanted.
        self.assertIsNotNone(rts_map)
        self.assertTrue('temperature' in rts_map)
        geo_temperature = rts_map['temperature']
        self.assertEqual(len(geo_temperature), 1)
        self.assertLessEqual(GeoPoint.distance2(geo_temperature[0].mid_point(), GeoPoint(x, y, z)), 1.0)
        # check if time axis is as expected
        self.assertEqual(geo_temperature[0].ts.time_axis, ts_exp.time_axis)
        self.assertTrue(np.allclose(geo_temperature[0].ts.time_axis.time_points, ts_exp.time_axis.time_points))
        self.assertEqual(geo_temperature[0].ts.point_interpretation(), point_fx.POINT_AVERAGE_VALUE)
        # check if variable data is as expected
        self.assertTrue(np.allclose(geo_temperature[0].ts.values.to_numpy(), ts_exp.values.to_numpy()))

        # --------------------------------------
        # Append with overlap
        print("\n\n append with overlap")
        # create time axis
        ta = TimeAxis(utc.time(2016, 1, 3), deltahours(1), 48)
        ts = TimeSeries(ta, dv.from_numpy(np.arange(0, ta.size(), dtype=np.float64)),
                        point_fx=point_fx.POINT_AVERAGE_VALUE)
        # save the data
        t_ds.append_ts_data(ts)

        # expected result
        ta = TimeAxis(utc.time(2016, 1, 1), deltahours(1), 96)
        data = np.empty(96)
        data[:24] = np.arange(0, 24, dtype=np.float64)
        data[24:48] = np.arange(0, 24, dtype=np.float64)  # <-- new data
        data[48:96] = np.arange(0, 48, dtype=np.float64)  # <-- new data
        ts_exp = TimeSeries(ta, dv.from_numpy(data), point_fx=point_fx.POINT_AVERAGE_VALUE)

        # now read back the result using a *standard* shyft cf geo repository
        selection_criteria = {'bbox': [[x0, x1, x1, x0], [y0, y0, y1, y1]]}
        ts_dr = CFDataRepository(epsg_id, test_file, selection_criteria)
        # now read back 'temperature' that we know should be there
        rts_map = ts_dr.get_timeseries(['temperature'], ts_exp.total_period())

        # and verify that we get exactly back what we wanted.
        self.assertIsNotNone(rts_map)
        self.assertTrue('temperature' in rts_map)
        geo_temperature = rts_map['temperature']
        self.assertEqual(len(geo_temperature), 1)
        self.assertLessEqual(GeoPoint.distance2(geo_temperature[0].mid_point(), GeoPoint(x, y, z)), 1.0)
        # check if time axis is as expected
        self.assertEqual(geo_temperature[0].ts.time_axis, ts_exp.time_axis)
        self.assertTrue(np.allclose(geo_temperature[0].ts.time_axis.time_points, ts_exp.time_axis.time_points))
        self.assertEqual(geo_temperature[0].ts.point_interpretation(), point_fx.POINT_AVERAGE_VALUE)
        # check if variable data is as expected
        self.assertTrue(np.allclose(geo_temperature[0].ts.values.to_numpy(), ts_exp.values.to_numpy()))

        # --------------------------------------
        # Append with gap in time axis
        print("\n\n Append with gap in time axis")
        # create time axis
        ta = TimeAxis(utc.time(2016, 1, 6), deltahours(1), 24)
        ts = TimeSeries(ta, dv.from_numpy(np.arange(0, ta.size(), dtype=np.float64)),
                        point_fx=point_fx.POINT_AVERAGE_VALUE)
        # save the data
        t_ds.append_ts_data(ts)

        # expected result
        time_vals = np.append(TimeAxis(utc.time(2016, 1, 1), deltahours(1), 96).time_points[:-1], ta.time_points)
        print(time_vals)
        ta = TimeAxis(UtcTimeVector.from_numpy(time_vals.astype(np.int64)))
        data = np.empty(120)
        data[:24] = np.arange(0, 24, dtype=np.float64)
        data[24:48] = np.arange(0, 24, dtype=np.float64)
        data[48:96] = np.arange(0, 48, dtype=np.float64)
        data[96:120] = np.arange(0, 24, dtype=np.float64)  # <-- new data
        ts_exp = TimeSeries(ta, dv.from_numpy(data), point_fx=point_fx.POINT_AVERAGE_VALUE)

        # now read back the result using a *standard* shyft cf geo repository
        selection_criteria = {'bbox': [[x0, x1, x1, x0], [y0, y0, y1, y1]]}
        ts_dr = CFDataRepository(epsg_id, test_file, selection_criteria)
        # now read back 'temperature' that we know should be there
        print(ts_exp.total_period())
        rts_map = ts_dr.get_timeseries(['temperature'], ts_exp.total_period())

        # # and verify that we get exactly back what we wanted.
        # self.assertIsNotNone(rts_map)
        # self.assertTrue('temperature' in rts_map)
        # geo_temperature = rts_map['temperature']
        # self.assertEqual(len(geo_temperature), 1)
        # self.assertLessEqual(GeoPoint.distance2(geo_temperature[0].mid_point(), GeoPoint(x, y, z)), 1.0)
        # # check if time axis is as expected
        # print(geo_temperature[0].ts.time_axis.time_points - ts_exp.time_axis.time_points)
        # print(geo_temperature[0].ts.time_axis.time_points - time_vals)
        # print(ts_exp.time_axis.time_points - time_vals)
        # self.assertEqual(geo_temperature[0].ts.time_axis, ts_exp.time_axis)
        # self.assertTrue(np.allclose(geo_temperature[0].ts.time_axis.time_points, ts_exp.time_axis.time_points))
        # self.assertEqual(geo_temperature[0].ts.point_interpretation(), point_fx.POINT_AVERAGE_VALUE)
        # # check if variable data is as expected
        # self.assertTrue(np.allclose(geo_temperature[0].ts.values.to_numpy(), ts_exp.values.to_numpy()))

        # --------------------------------------
        # Add new data in the middle where nothing was defined (no moving)
        print("\n\n Add new data in the middle where nothing was defined (no moving)")
        # create time axis
        ta = TimeAxis(utc.time(2016, 1, 2), deltahours(1), 24)
        ts = TimeSeries(ta, dv.from_numpy(np.arange(100, 100+ta.size(), dtype=np.float64)),
                        point_fx=point_fx.POINT_AVERAGE_VALUE)
        # save the data
        t_ds.append_ts_data(ts)

        # expected result
        time_vals = np.append(TimeAxis(utc.time(2016, 1, 1), deltahours(1), 96).time_points[:-1], ta.time_points)
        ta = TimeAxis(UtcTimeVector.from_numpy(time_vals.astype(np.int64)))
        data = np.empty(120)
        data[:24] = np.arange(0, 24, dtype=np.float64)
        data[24:48] = np.arange(100, 124, dtype=np.float64)  # <-- new data
        data[48:96] = np.arange(0, 48, dtype=np.float64)
        data[96:120] = np.arange(0, 24, dtype=np.float64)
        ts_exp = TimeSeries(ta, dv.from_numpy(data), point_fx=point_fx.POINT_AVERAGE_VALUE)

        # now read back the result using a *standard* shyft cf geo repository
        selection_criteria = {'bbox': [[x0, x1, x1, x0], [y0, y0, y1, y1]]}
        ts_dr = CFDataRepository(epsg_id, test_file, selection_criteria)
        # now read back 'temperature' that we know should be there
        rts_map = ts_dr.get_timeseries(['temperature'], ts_exp.total_period())

        # # and verify that we get exactly back what we wanted.
        # self.assertIsNotNone(rts_map)
        # self.assertTrue('temperature' in rts_map)
        # geo_temperature = rts_map['temperature']
        # self.assertEqual(len(geo_temperature), 1)
        # self.assertLessEqual(GeoPoint.distance2(geo_temperature[0].mid_point(), GeoPoint(x, y, z)), 1.0)
        # # check if time axis is as expected
        # self.assertEqual(geo_temperature[0].ts.time_axis, ts_exp.time_axis)
        # self.assertTrue(np.allclose(geo_temperature[0].ts.time_axis.time_points, ts_exp.time_axis.time_points))
        # self.assertEqual(geo_temperature[0].ts.point_interpretation(), point_fx.POINT_AVERAGE_VALUE)
        # # check if variable data is as expected
        # self.assertTrue(np.allclose(geo_temperature[0].ts.values.to_numpy(), ts_exp.values.to_numpy()))

        # --------------------------------------
        # Insert new data in the middle and move rest
        print("\n\n insert new data and move rest")
        # create time axis
        ta = TimeAxis(utc.time(2016, 1, 5), deltahours(1), 36)
        ts = TimeSeries(ta, dv.from_numpy(np.arange(200, 200 + ta.size(), dtype=np.float64)),
                        point_fx=point_fx.POINT_AVERAGE_VALUE)
        # save the data
        t_ds.append_ts_data(ts)

        # expected result
        ta = TimeAxis(utc.time(2016, 1, 1), deltahours(1), 144)
        data = np.empty(144)
        data[:24] = np.arange(0, 24, dtype=np.float64)
        data[24:48] = np.arange(100, 124, dtype=np.float64)
        data[48:96] = np.arange(0, 48, dtype=np.float64)
        data[96:132] = np.arange(200, 236, dtype=np.float64)  # <-- new data
        data[132:144] = np.arange(12, 24, dtype=np.float64)
        ts_exp = TimeSeries(ta, dv.from_numpy(data), point_fx=point_fx.POINT_AVERAGE_VALUE)

        # now read back the result using a *standard* shyft cf geo repository
        selection_criteria = {'bbox': [[x0, x1, x1, x0], [y0, y0, y1, y1]]}
        ts_dr = CFDataRepository(epsg_id, test_file, selection_criteria)
        # now read back 'temperature' that we know should be there
        rts_map = ts_dr.get_timeseries(['temperature'], ts_exp.total_period())

        # and verify that we get exactly back what we wanted.
        self.assertIsNotNone(rts_map)
        self.assertTrue('temperature' in rts_map)
        geo_temperature = rts_map['temperature']
        self.assertEqual(len(geo_temperature), 1)
        self.assertLessEqual(GeoPoint.distance2(geo_temperature[0].mid_point(), GeoPoint(x, y, z)), 1.0)
        # check if time axis is as expected
        self.assertEqual(geo_temperature[0].ts.time_axis, ts_exp.time_axis)
        self.assertTrue(np.allclose(geo_temperature[0].ts.time_axis.time_points, ts_exp.time_axis.time_points))
        self.assertEqual(geo_temperature[0].ts.point_interpretation(), point_fx.POINT_AVERAGE_VALUE)
        # check if variable data is as expected
        self.assertTrue(np.allclose(geo_temperature[0].ts.values.to_numpy(), ts_exp.values.to_numpy()))

        # --------------------------------------
        # Add new data before existing data without overlap
        print("\n\n add new data before existing data without overlap")
        # create time axis
        ta = TimeAxis(utc.time(2015, 12, 31), deltahours(1), 24)
        ts = TimeSeries(ta, dv.from_numpy(np.arange(300, 300 + ta.size(), dtype=np.float64)),
                        point_fx=point_fx.POINT_AVERAGE_VALUE)
        # save the first batch
        t_ds.append_ts_data(ts)

        # expected result
        ta = TimeAxis(utc.time(2015, 12, 31), deltahours(1), 168)
        data = np.empty(168)
        data[:24] = np.arange(300, 324, dtype=np.float64) # <-- new data
        data[24:48] = np.arange(0, 24, dtype=np.float64)
        data[48:72] = np.arange(100, 124, dtype=np.float64)
        data[72:120] = np.arange(0, 48, dtype=np.float64)
        data[120:156] = np.arange(200, 236, dtype=np.float64)
        data[156:168] = np.arange(12, 24, dtype=np.float64)
        ts_exp = TimeSeries(ta, dv.from_numpy(data), point_fx=point_fx.POINT_AVERAGE_VALUE)

        # now read back the result using a *standard* shyft cf geo repository
        selection_criteria = {'bbox': [[x0, x1, x1, x0], [y0, y0, y1, y1]]}
        ts_dr = CFDataRepository(epsg_id, test_file, selection_criteria)
        # now read back 'temperature' that we know should be there
        rts_map = ts_dr.get_timeseries(['temperature'], ts_exp.total_period())

        # and verify that we get exactly back what we wanted.
        self.assertIsNotNone(rts_map)
        self.assertTrue('temperature' in rts_map)
        geo_temperature = rts_map['temperature']
        self.assertEqual(len(geo_temperature), 1)
        self.assertLessEqual(GeoPoint.distance2(geo_temperature[0].mid_point(), GeoPoint(x, y, z)), 1.0)
        # check if time axis is as expected
        self.assertEqual(geo_temperature[0].ts.time_axis, ts_exp.time_axis)
        self.assertTrue(np.allclose(geo_temperature[0].ts.time_axis.time_points, ts_exp.time_axis.time_points))
        self.assertEqual(geo_temperature[0].ts.point_interpretation(), point_fx.POINT_AVERAGE_VALUE)
        # check if variable data is as expected
        self.assertTrue(np.allclose(geo_temperature[0].ts.values.to_numpy(), ts_exp.values.to_numpy()))

        # --------------------------------------
        # add new data before existing data with overlap
        print("\n\n add new data before existing data with overlap")
        # create time axis
        ta = TimeAxis(utc.time(2015, 12, 30), deltahours(1), 36)
        ts = TimeSeries(ta, dv.from_numpy(np.arange(400, 400 + ta.size(), dtype=np.float64)), point_fx=point_fx.POINT_AVERAGE_VALUE)
        # save the first batch
        #t_ds = TimeSeriesStore(test_file, temperature)
        t_ds.append_ts_data(ts)

        # expected result
        ta = TimeAxis(utc.time(2015, 12, 30), deltahours(1), 192)
        data = np.empty(192)
        data[:36] = np.arange(400, 436, dtype=np.float64)  # <-- new data
        data[36:48] = np.arange(312, 324, dtype=np.float64)
        data[48:72] = np.arange(0, 24, dtype=np.float64)
        data[72:96] = np.arange(100, 124, dtype=np.float64)
        data[96:144] = np.arange(0, 48, dtype=np.float64)
        data[144:180] = np.arange(200, 236, dtype=np.float64)
        data[180:192] = np.arange(12, 24, dtype=np.float64)
        ts_exp = TimeSeries(ta, dv.from_numpy(data), point_fx=point_fx.POINT_AVERAGE_VALUE)

        # now read back the result using a *standard* shyft cf geo repository
        selection_criteria = {'bbox': [[x0, x1, x1, x0], [y0, y0, y1, y1]]}
        ts_dr = CFDataRepository(epsg_id, test_file, selection_criteria)
        # now read back 'temperature' that we know should be there
        rts_map = ts_dr.get_timeseries(['temperature'], ts_exp.total_period())

        # and verify that we get exactly back what we wanted.
        self.assertIsNotNone(rts_map)
        self.assertTrue('temperature' in rts_map)
        geo_temperature = rts_map['temperature']
        self.assertEqual(len(geo_temperature), 1)
        self.assertLessEqual(GeoPoint.distance2(geo_temperature[0].mid_point(), GeoPoint(x, y, z)), 1.0)
        # check if time axis is as expected
        self.assertEqual(geo_temperature[0].ts.time_axis, ts_exp.time_axis)
        self.assertTrue(np.allclose(geo_temperature[0].ts.time_axis.time_points, ts_exp.time_axis.time_points))
        self.assertEqual(geo_temperature[0].ts.point_interpretation(), point_fx.POINT_AVERAGE_VALUE)
        # check if variable data is as expected
        self.assertTrue(np.allclose(geo_temperature[0].ts.values.to_numpy(), ts_exp.values.to_numpy()))

        # --------------------------------------
        # Overwrite everything with less data points
        # create time axis
        print('\n\n Overwrite everything with less data points')
        ta = TimeAxis(utc.time(2015, 12, 30), deltahours(24), 9)
        ts = TimeSeries(ta, dv.from_numpy(np.arange(1000, 1000 + ta.size(), dtype=np.float64)),
                        point_fx=point_fx.POINT_AVERAGE_VALUE)
        # write the time series
        t_ds.append_ts_data(ts)

        # expected result
        ts_exp = ts

        # now read back the result using a *standard* shyft cf geo repository
        selection_criteria = {'bbox': [[x0, x1, x1, x0], [y0, y0, y1, y1]]}
        ts_dr = CFDataRepository(epsg_id, test_file, selection_criteria)
        # now read back 'temperature' that we know should be there
        rts_map = ts_dr.get_timeseries(['temperature'], ts_exp.total_period())

        # and verify that we get exactly back what we wanted.
        self.assertIsNotNone(rts_map)
        self.assertTrue('temperature' in rts_map)
        geo_temperature = rts_map['temperature']
        self.assertEqual(len(geo_temperature), 1)
        self.assertLessEqual(GeoPoint.distance2(geo_temperature[0].mid_point(), GeoPoint(x, y, z)), 1.0)
        # check if time axis is as expected
        self.assertEqual(geo_temperature[0].ts.time_axis, ts_exp.time_axis)
        self.assertTrue(np.allclose(geo_temperature[0].ts.time_axis.time_points, ts_exp.time_axis.time_points))
        self.assertEqual(geo_temperature[0].ts.point_interpretation(), point_fx.POINT_AVERAGE_VALUE)
        # check if variable data is as expected
        self.assertTrue(np.allclose(geo_temperature[0].ts.values.to_numpy(), ts_exp.values.to_numpy()))

        # --------------------------------------
        # Insert data with different dt
        # create time axis
        print('\n\n Insert data with different dt')
        ta = TimeAxis(utc.time(2016, 1, 1), deltahours(1), 24)
        ts = TimeSeries(ta, dv.from_numpy(np.arange(0, 24, dtype=np.float64)),
                        point_fx=point_fx.POINT_AVERAGE_VALUE)
        # write the time series
        t_ds.append_ts_data(ts)

        # expected result
        time_points = np.empty(33, dtype=np.int)
        time_points[0:2] = TimeAxis(utc.time(2015, 12, 30), deltahours(24), 1).time_points
        time_points[2:26] = TimeAxis(utc.time(2016, 1, 1), deltahours(1), 23).time_points
        time_points[26:] = TimeAxis(utc.time(2016, 1, 2), deltahours(24), 6).time_points
        ta = TimeAxis(UtcTimeVector.from_numpy(time_points))
        data = np.empty(32)
        data[0:2] = np.array([1000, 1001])
        data[2:26] = np.arange(0, 24)  # <-- new data
        data[26:] = np.arange(1003, 1009)
        ts_exp = TimeSeries(ta, dv.from_numpy(data), point_fx=point_fx.POINT_AVERAGE_VALUE)

        # now read back the result using a *standard* shyft cf geo repository
        selection_criteria = {'bbox': [[x0, x1, x1, x0], [y0, y0, y1, y1]]}
        ts_dr = CFDataRepository(epsg_id, test_file, selection_criteria)
        # now read back 'temperature' that we know should be there
        rts_map = ts_dr.get_timeseries(['temperature'], ts_exp.total_period())

        # # and verify that we get exactly back what we wanted.
        # self.assertIsNotNone(rts_map)
        # self.assertTrue('temperature' in rts_map)
        # geo_temperature = rts_map['temperature']
        # self.assertEqual(len(geo_temperature), 1)
        # self.assertLessEqual(GeoPoint.distance2(geo_temperature[0].mid_point(), GeoPoint(x, y, z)), 1.0)
        # # check if time axis is as expected
        # print(geo_temperature[0].ts.time_axis.time_points, ts_exp.time_axis.time_points)
        # print(geo_temperature[0].ts.time_axis.time_points - ts_exp.time_axis.time_points)
        # self.assertEqual(geo_temperature[0].ts.time_axis, ts_exp.time_axis)
        # self.assertTrue(np.allclose(geo_temperature[0].ts.time_axis.time_points, ts_exp.time_axis.time_points))
        # self.assertEqual(geo_temperature[0].ts.point_interpretation(), point_fx.POINT_AVERAGE_VALUE)
        # # check if variable data is as expected
        # self.assertTrue(np.allclose(geo_temperature[0].ts.values.to_numpy(), ts_exp.values.to_numpy()))
        # --------------------------------------

        # Write empty data i.e. UtcPeriod
        # print('\n\n Insert data with different dt')
        #
        # tp = UtcPeriod()
        # #ta = TimeAxis(utc.time(2016, 1, 1), deltahours(1), 24)
        # #ts = TimeSeries(ta, dv.from_numpy(np.arange(0, 24, dtype=np.float64)),
        #                 point_fx=point_fx.POINT_AVERAGE_VALUE)
        # # write the time series
        # t_ds.append_ts_data(ts)
        #
        # # expected result
        # time_points = np.empty(33, dtype=np.int)
        # time_points[0:2] = TimeAxis(utc.time(2015, 12, 30), deltahours(24), 1).time_points
        # time_points[2:26] = TimeAxis(utc.time(2016, 1, 1), deltahours(1), 23).time_points
        # time_points[26:] = TimeAxis(utc.time(2016, 1, 2), deltahours(24), 6).time_points
        # ta = TimeAxis(UtcTimeVector.from_numpy(time_points))
        # data = np.empty(32)
        # data[0:2] = np.array([1000, 1001])
        # data[2:26] = np.arange(0, 24)  # <-- new data
        # data[26:] = np.arange(1003, 1009)
        # ts_exp = TimeSeries(ta, dv.from_numpy(data), point_fx=point_fx.POINT_AVERAGE_VALUE)
        #
        # # now read back the result using a *standard* shyft cf geo repository
        # selection_criteria = {'bbox': [[x0, x1, x1, x0], [y0, y0, y1, y1]]}
        # ts_dr = CFDataRepository(epsg_id, test_file, selection_criteria)
        # # now read back 'temperature' that we know should be there
        # rts_map = ts_dr.get_timeseries(['temperature'], ts_exp.total_period())
        #
        # # and verify that we get exactly back what we wanted.
        # self.assertIsNotNone(rts_map)
        # self.assertTrue('temperature' in rts_map)
        # geo_temperature = rts_map['temperature']
        # self.assertEqual(len(geo_temperature), 1)
        # self.assertLessEqual(GeoPoint.distance2(geo_temperature[0].mid_point(), GeoPoint(x, y, z)), 1.0)
        # # check if time axis is as expected
        # print(geo_temperature[0].ts.time_axis.time_points, ts_exp.time_axis.time_points)
        # print(geo_temperature[0].ts.time_axis.time_points - ts_exp.time_axis.time_points)
        # self.assertEqual(geo_temperature[0].ts.time_axis, ts_exp.time_axis)
        # self.assertTrue(np.allclose(geo_temperature[0].ts.time_axis.time_points, ts_exp.time_axis.time_points))
        # self.assertEqual(geo_temperature[0].ts.point_interpretation(), point_fx.POINT_AVERAGE_VALUE)
        # # check if variable data is as expected
        # self.assertTrue(np.allclose(geo_temperature[0].ts.values.to_numpy(), ts_exp.values.to_numpy()))
        # # --------------------------------------


if __name__ == '__main__':
    unittest.main()