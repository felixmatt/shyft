import unittest
from os import path
from shyft import shyftdata_dir
from shyft.repository.service.ssa_geo_ts_repository import GeoTsRepository
from shyft.repository.yaml_config import YamlContent
from shyft.api import Calendar
from shyft.api import YMDhms
from shyft.api import UtcPeriod
from shyft.api import TemperatureSource


class SSAGeoTsRepositoryTestCase(unittest.TestCase):
    """
    Verify that we correctly can read geo-located timeseries from a the SSA service based
    location and ts-store.
    """

    def test_construct_repository(self):
        utc = Calendar()
        #TODO: figure out config-classes, mappings
        #      that exist, at least in statkraft,
        #      to the call, expect some data back..
        #      i.e. we can only do a system-type test here, relying on certain data in the GIS and SmG system
        #
        #met = path.join(shyftdata_dir, "netcdf", "orchestration-testdata","stations_met.nc")
        #dis = path.join(shyftdata_dir, "netcdf", "orchestration-testdata","stations_discharge.nc")
        #map_cfg_file = path.join(path.dirname(__file__), "netcdf","datasets.yaml")
        #map_cfg = YamlContent(map_cfg_file)
        #params = map_cfg.sources[0]['params']  # yes, hmm.
        repository = GeoTsRepository()
        self.assertIsNotNone(repository)
        utc_period = UtcPeriod(utc.time(YMDhms(1990, 1, 1, 0, 0, 0)),utc.time(YMDhms(2000, 1, 1, 0, 0, 0)))
        type_source_map = dict()
        type_source_map['temperature'] = TemperatureSource
        geo_ts_dict = repository.get_timeseries(type_source_map,geo_location_criteria=None,utc_period=utc_period)
        self.assertIsNotNone(geo_ts_dict)
