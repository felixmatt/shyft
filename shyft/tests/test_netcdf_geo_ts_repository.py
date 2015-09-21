import unittest
from os import path
from shyft import shyftdata_dir
from shyft.repository.netcdf.geo_ts_repository import GeoTsRepository
from shyft.repository.yaml_config import YamlContent
from shyft.api import Calendar
from shyft.api import YMDhms
from shyft.api import UtcPeriod
from shyft.api import TemperatureSource


class NetCDFSourceRepositoryTestCase(unittest.TestCase):
    """
    Verify that we correctly can read geo-located timeseries from a netCDF
    based file-store.
    """

    def test_construct_repository(self):
        utc_calendar = Calendar()
        met = path.join(shyftdata_dir, "netcdf", "orchestration-testdata",
                        "stations_met.nc")
        dis = path.join(shyftdata_dir, "netcdf", "orchestration-testdata",
                        "stations_discharge.nc")
        map_cfg_file = path.join(path.dirname(__file__), "netcdf",
                                 "datasets.yaml")
        map_cfg = YamlContent(map_cfg_file)
        params = map_cfg.sources[0]['params']  # yes, hmm.
        netcdf_repository = GeoTsRepository(params, met, dis)
        self.assertIsNotNone(netcdf_repository)
        utc_period = UtcPeriod(utc_calendar.time(YMDhms(1990, 1, 1, 0, 0, 0)),
                               utc_calendar.time(YMDhms(2000, 1, 1, 0, 0, 0)))
        type_source_map = dict()
        type_source_map['temperature'] = TemperatureSource
        geo_ts_dict =  \
            netcdf_repository.get_timeseries(type_source_map,
                                             geo_location_criteria=None,
                                             utc_period=utc_period)
        self.assertIsNotNone(geo_ts_dict)
