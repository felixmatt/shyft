import unittest
from os import path
import os
import numpy as np
from netCDF4 import Dataset
from pyproj import Proj

from shyft.api import Calendar
# from shyft.api import UtcPeriod
# from shyft.api import TemperatureSource
from shyft.api import TimeSeries
from shyft.api import TimeAxis
from shyft.api import point_interpretation_policy as point_fx
from shyft.api import deltahours, deltaminutes
from shyft.api import DoubleVector as dv
from shyft.api import GeoPoint
from shyft.api import UtcTimeVector
from shyft.api import UtcPeriod

from shyft.repository.netcdf.cf_geo_ts_repository import CFDataRepository, CFDataRepositoryError
from shyft.repository.netcdf.cf_ts_store import CFInfo, TimeSeriesMetaInfo, TimeSeriesStore, TimeSeriesStoreError


class NetCDFGeoTsRWTestCase(unittest.TestCase):
    """
    Verify that we correctly can read geo-located timeseries from a netCDF
    based file-store.
    """

    def _construct_from_test_data(self):
        # met = path.join(shyftdata_dir, "netcdf", "orchestration-testdata", "atnasjoen_met.nc")
        # dis = path.join(shyftdata_dir, "netcdf", "orchestration-testdata", "atnasjoen_discharge.nc")
        # map_cfg_file = path.join(path.dirname(__file__), "netcdf","datasets.yaml")
        # map_cfg = YamlContent(map_cfg_file)
        # params = map_cfg.sources[0]['params']  # yes, hmm.
        # return GeoTsRepository(params, met, dis)
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
        temperature = TimeSeriesMetaInfo('temperature', '/observed/at_stn_abc/temperature', 'observed air temperature',
                                         x, y, z, epsg_id)

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
        ts = TimeSeries(ta, dv.from_numpy(np.arange(0, ta.size(), dtype=np.float64)),
                        point_fx=point_fx.POINT_AVERAGE_VALUE)
        # save the data
        t_ds.append_ts_data(ts)

        # expected result
        ta = TimeAxis(utc.time(2016, 1, 1), deltahours(1), 72)
        data = np.empty(72)
        data[:24] = np.arange(0, 24, dtype=np.float64)
        data[24:72] = np.arange(0, 48, dtype=np.float64)  # <-- new data
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
        # print(time_vals)
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
        # print(ts_exp.total_period())
        rts_map = ts_dr.get_timeseries(['temperature'], ts_exp.total_period())

        # and verify that we get exactly back what we wanted.
        self.assertIsNotNone(rts_map)
        self.assertTrue('temperature' in rts_map)
        geo_temperature = rts_map['temperature']
        self.assertEqual(len(geo_temperature), 1)
        self.assertLessEqual(GeoPoint.distance2(geo_temperature[0].mid_point(), GeoPoint(x, y, z)), 1.0)
        # check if time axis is as expected
        # print(geo_temperature[0].ts.time_axis.time_points - ts_exp.time_axis.time_points)
        # print(geo_temperature[0].ts.time_axis.time_points - time_vals)
        # print(ts_exp.time_axis.time_points - time_vals)
        self.assertEqual(geo_temperature[0].ts.time_axis, ts_exp.time_axis)
        self.assertTrue(np.allclose(geo_temperature[0].ts.time_axis.time_points, ts_exp.time_axis.time_points))
        self.assertEqual(geo_temperature[0].ts.point_interpretation(), point_fx.POINT_AVERAGE_VALUE)
        # check if variable data is as expected
        self.assertTrue(np.allclose(geo_temperature[0].ts.values.to_numpy(), ts_exp.values.to_numpy()))

        # --------------------------------------
        # Add new data in the middle where nothing was defined (no moving)
        print("\n\n Add new data in the middle where nothing was defined (no moving)")
        # create time axis
        ta = TimeAxis(utc.time(2016, 1, 2), deltahours(1), 24)
        ts = TimeSeries(ta, dv.from_numpy(np.arange(100, 100 + ta.size(), dtype=np.float64)),
                        point_fx=point_fx.POINT_AVERAGE_VALUE)
        # save the data
        t_ds.append_ts_data(ts)

        # expected result
        time_vals = np.append(TimeAxis(utc.time(2016, 1, 1), deltahours(1), 96).time_points[:-1],
                              TimeAxis(utc.time(2016, 1, 6), deltahours(1), 24).time_points)
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
        # print(ts_exp.total_period())
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
        data[:24] = np.arange(300, 324, dtype=np.float64)  # <-- new data
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
        ts = TimeSeries(ta, dv.from_numpy(np.arange(400, 400 + ta.size(), dtype=np.float64)),
                        point_fx=point_fx.POINT_AVERAGE_VALUE)
        # save the first batch
        # t_ds = TimeSeriesStore(test_file, temperature)
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
        # delete data with range UtcPeriod in the middle
        print('\n\n delete data with range UtcPeriod')
        tp = UtcPeriod(utc.time(2015, 12, 31), utc.time(2016, 1, 1, 12))
        # ta = TimeAxis(utc.time(2016, 1, 1), deltahours(1), 24)
        # ts = TimeSeries(ta, dv.from_numpy(np.arange(0, 24, dtype=np.float64)), point_fx=point_fx.POINT_AVERAGE_VALUE)
        # write the time series
        t_ds.remove_tp_data(tp)

        # expected result
        time_points = np.array([1451433600, 1451653200, 1451656800, 1451660400, 1451664000, 1451667600,
                                1451671200, 1451674800, 1451678400, 1451682000, 1451685600, 1451689200,
                                1451692800, 1451779200, 1451865600, 1451952000, 1452038400, 1452124800,
                                1452211200])
        ta = TimeAxis(UtcTimeVector.from_numpy(time_points))
        data = np.array([1000, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 1003, 1004, 1005, 1006, 1007, 1008])
        ts_exp = TimeSeries(ta, dv.from_numpy(data),point_fx.POINT_INSTANT_VALUE)  # TODO: is this correct policy to use

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
        # delete data with range UtcPeriod at the start
        print('\n\n delete data with range UtcPeriod at the start')
        tp = UtcPeriod(1451433600, 1451667600)
        # ta = TimeAxis(utc.time(2016, 1, 1), deltahours(1), 24)
        # ts = TimeSeries(ta, dv.from_numpy(np.arange(0, 24, dtype=np.float64)), point_fx=point_fx.POINT_AVERAGE_VALUE)
        # write the time series
        t_ds.remove_tp_data(tp)

        # expected result
        time_points = np.array([1451671200, 1451674800, 1451678400, 1451682000, 1451685600, 1451689200,
                                1451692800, 1451779200, 1451865600, 1451952000, 1452038400, 1452124800,
                                1452211200])
        ta = TimeAxis(UtcTimeVector.from_numpy(time_points))
        data = np.array([18, 19, 20, 21, 22, 23, 1003, 1004, 1005, 1006, 1007, 1008])
        ts_exp = TimeSeries(ta, dv.from_numpy(data), point_fx.POINT_INSTANT_VALUE)  # TODO: is this correct policy to use for this test

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
        # delete data with range UtcPeriod at the end
        print('\n\n delete data with range UtcPeriod at the end')
        tp = UtcPeriod(1451952000, utc.time(2016, 1, 10))
        # ta = TimeAxis(utc.time(2016, 1, 1), deltahours(1), 24)
        # ts = TimeSeries(ta, dv.from_numpy(np.arange(0, 24, dtype=np.float64)), point_fx=point_fx.POINT_AVERAGE_VALUE)
        # write the time series
        t_ds.remove_tp_data(tp)

        # expected result
        time_points = np.array([1451671200, 1451674800, 1451678400, 1451682000, 1451685600, 1451689200,
                                1451692800, 1451779200, 1451865600, 1451952000])
        ta = TimeAxis(UtcTimeVector.from_numpy(time_points))
        data = np.array([18, 19, 20, 21, 22, 23, 1003, 1004, 1005])
        ts_exp = TimeSeries(ta, dv.from_numpy(data), point_fx.POINT_INSTANT_VALUE)

        # now read back the result using a *standard* shyft cf geo repository
        selection_criteria = {'bbox': [[x0, x1, x1, x0], [y0, y0, y1, y1]]}
        ts_dr = CFDataRepository(epsg_id, test_file, selection_criteria)
        # now read back 'temperature' that we know should be there
        try:
            rts_map = ts_dr.get_timeseries(['temperature'], ts_exp.total_period())
        except CFDataRepositoryError:
            pass

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
        # delete data with range UtcPeriod everything
        print('\n\n delete data with range UtcPeriod everything')
        tp = UtcPeriod(utc.time(2016, 1, 1), utc.time(2016, 1, 10))
        # write the time series
        t_ds.remove_tp_data(tp)

        # now read back the result using a *standard* shyft cf geo repository
        selection_criteria = {'bbox': [[x0, x1, x1, x0], [y0, y0, y1, y1]]}
        ts_dr = CFDataRepository(epsg_id, test_file, selection_criteria)
        # now read back 'temperature' that we know should be there
        self.assertRaises(CFDataRepositoryError, ts_dr.get_timeseries, ['temperature'], tp)

        # --------------------------------------
        # insert data in between time saved data points
        print('\n\n insert data in between time saved data points')
        # insert first data in which we want to insert the second batch
        utc = Calendar()
        ta = TimeAxis(utc.time(2016, 1, 1), deltahours(24), 2)
        data = np.arange(0, ta.size(), dtype=np.float64)
        ts = TimeSeries(ta, dv.from_numpy(data), point_fx=point_fx.POINT_AVERAGE_VALUE)
        # save the first batch
        t_ds.append_ts_data(ts)

        # insert first data for every hour in between
        utc = Calendar()
        ta = TimeAxis(utc.time(2016, 1, 1) + deltahours(1), deltahours(1), 23)
        data = np.arange(10, 10 + ta.size(), dtype=np.float64)
        ts = TimeSeries(ta, dv.from_numpy(data), point_fx=point_fx.POINT_AVERAGE_VALUE)
        # save the first batch
        t_ds.append_ts_data(ts)

        # expected result
        time_points = np.array([1451606400, 1451610000, 1451613600, 1451617200, 1451620800, 1451624400, 1451628000,
                                1451631600, 1451635200, 1451638800, 1451642400, 1451646000, 1451649600, 1451653200,
                                1451656800, 1451660400, 1451664000, 1451667600, 1451671200, 1451674800, 1451678400,
                                1451682000, 1451685600, 1451689200, 1451692800, 0])
        time_points[-1] = 2 * time_points[-2] - time_points[-3]  # last time point calc
        data = np.array([0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                         27, 28, 29, 30, 31, 32, 1])
        ta = TimeAxis(UtcTimeVector.from_numpy(time_points))
        ts_exp = TimeSeries(ta, dv.from_numpy(data), point_fx.POINT_INSTANT_VALUE)  # TODO: is this correct policy value for this case

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
        # insert data including nan
        print('\n\n insert data including nan')
        utc = Calendar()
        ta = TimeAxis(utc.time(2016, 1, 1) + deltahours(1), deltahours(1), 23)
        data = np.arange(10, 10 + ta.size(), dtype=np.float64)
        data[4] = np.nan
        data[6] = np.nan  # np.inf, but trouble getting inf trough all version of numpy/netcdf
        data[8] = np.nan  # -np.inf, --"--
        ts = TimeSeries(ta, dv.from_numpy(data), point_fx=point_fx.POINT_AVERAGE_VALUE)
        # save the first batch
        t_ds.append_ts_data(ts)

        # expected result
        time_points = np.array([1451606400, 1451610000, 1451613600, 1451617200, 1451620800, 1451624400, 1451628000,
                                1451631600, 1451635200, 1451638800, 1451642400, 1451646000, 1451649600, 1451653200,
                                1451656800, 1451660400, 1451664000, 1451667600, 1451671200, 1451674800, 1451678400,
                                1451682000, 1451685600, 1451689200, 1451692800, 0])
        time_points[-1] = 2 * time_points[-2] - time_points[-3]  # last time point calc

        data = np.array([0, 10, 11, 12, 13, np.nan, 15,
                         # np.inf,
                         np.nan,  # TODO: figure out how to unmask restoring 'used' mask-values
                         17,
                         #-np.inf,
                         np.nan,
                         19, 20, 21, 22, 23, 24, 25, 26,
                         27, 28, 29, 30, 31, 32, 1])
        ta = TimeAxis(UtcTimeVector.from_numpy(time_points))
        ts_exp = TimeSeries(ta, dv.from_numpy(data), point_fx.POINT_INSTANT_VALUE)  # TODO: policy right ?

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
        self.assertTrue(np.allclose(geo_temperature[0].ts.values.to_numpy(), ts_exp.values.to_numpy(), equal_nan=True))
        if path.exists(test_file):
            os.remove(test_file)



if __name__ == '__main__':
    unittest.main()
