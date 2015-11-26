import unittest
from os import path
from shyft import shyftdata_dir
from shyft.repository.netcdf.geo_ts_repository import GeoTsRepository
from shyft.repository.netcdf.yaml_config import YamlContent
from shyft.api import Calendar
from shyft.api import YMDhms
from shyft.api import UtcPeriod
from shyft.api import TemperatureSource


class NetCDFGeoTsRepositoryTestCase(unittest.TestCase):
    """
    Verify that we correctly can read geo-located timeseries from a netCDF
    based file-store.
    """
    def _construct_from_test_data(self):
        met = path.join(shyftdata_dir, "netcdf", "orchestration-testdata","atnasjoen_met.nc")
        dis = path.join(shyftdata_dir, "netcdf", "orchestration-testdata","atnasjoen_discharge.nc")
        map_cfg_file = path.join(path.dirname(__file__), "netcdf", "datasets.yaml")
        map_cfg = YamlContent(map_cfg_file)
        params = map_cfg.sources[0]['params']  # yes, hmm.
        return GeoTsRepository(params, met, dis)

    def test_construct_repository(self):
        utc_calendar = Calendar()
        netcdf_repository = self._construct_from_test_data()
        self.assertIsNotNone(netcdf_repository)
        utc_period = UtcPeriod(utc_calendar.time(YMDhms(2005, 1, 1, 0, 0, 0)),
                               utc_calendar.time(YMDhms(2014, 12, 31, 0, 0, 0)))
        type_source_map = dict()
        type_source_map['temperature'] = TemperatureSource
        geo_ts_dict =  netcdf_repository.get_timeseries(
                                                type_source_map,
                                                geo_location_criteria=None,
                                                utc_period=utc_period)
        self.assertIsNotNone(geo_ts_dict)

    def test_returns_empty_ts_when_no_data_in_request_period(self):
        utc_calendar = Calendar()
        netcdf_repository = self._construct_from_test_data()
        self.assertIsNotNone(netcdf_repository)
        utc_period = UtcPeriod(utc_calendar.time(YMDhms(2017, 1, 1, 0, 0, 0)),# a period where there is no data in
                               utc_calendar.time(YMDhms(2020, 12, 31, 0, 0, 0)))# the file supplied
        type_source_map = dict()
        type_source_map['temperature'] = TemperatureSource
        geo_ts_dict = netcdf_repository.get_timeseries(
                                                type_source_map,
                                                geo_location_criteria=None,
                                                utc_period=utc_period)
        self.assertIsNotNone(geo_ts_dict)

    def test_raise_exception_when_no_data_in_request_period(self):
        utc_calendar = Calendar()
        netcdf_repository = self._construct_from_test_data()
        netcdf_repository.raise_if_no_data=True # yes, for now, just imagine this could work.
        self.assertIsNotNone(netcdf_repository)
        utc_period = UtcPeriod(utc_calendar.time(YMDhms(2017, 1, 1, 0, 0, 0)),# a period where there is no data in
                               utc_calendar.time(YMDhms(2020, 12, 31, 0, 0, 0)))# the file supplied
        type_source_map = dict()
        type_source_map['temperature'] = TemperatureSource

        #def test_function():
        #
        #    return netcdf_repository.get_timeseries(
        #                                        type_source_map,
        #                                        geo_location_criteria=None,
        #                                        utc_period=utc_period)

        #self.assertRaises(RuntimeError, test_function)
        self.assertRaises(RuntimeError, netcdf_repository.get_timeseries, type_source_map, **{'geo_location_criteria':None, 'utc_period':utc_period})
        #,
        #                  type_source_map,
        #                  **{geo_location_criteria:None,utc_period:utc_period}
        #       )

if __name__ == '__main__':
    unittest.main()
