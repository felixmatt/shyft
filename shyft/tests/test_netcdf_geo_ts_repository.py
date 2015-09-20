import unittest
from os import path
from shyft import shyftdata_dir
from shyft.repository.netcdf.geo_ts_repository import NetCDFGeoTsRepository 
from shyft.repository.yaml_config import BaseYamlConfig
from shyft.api import Calendar
from shyft.api import YMDhms
from shyft.api import UtcPeriod
from shyft.api import RelHumSource
from shyft.api import TemperatureSource
from shyft.api import PrecipitationSource
from shyft.api import WindSpeedSource
from shyft.api import RadiationSource

class NetCDFSourceRepositoryTestCase(unittest.TestCase):
    """ Verify that we correctly can read 
        geo-located timeseries from a netCDF based file-store,
    """

    def test_construct_repository(self):
        utc_calendar=Calendar()
        met = path.join(shyftdata_dir, "netcdf", "orchestration-testdata","stations_met.nc")
        dis = path.join(shyftdata_dir, "netcdf","orchestration-testdata", "stations_discharge.nc")
        map_cfg_file=path.join(path.dirname(__file__), "netcdf","datasets.yaml")
        map_cfg=BaseYamlConfig(map_cfg_file)
        params=map_cfg.sources[0]['params']# yes, hmm. 
        netcdf_repository = NetCDFGeoTsRepository(params,met,dis)
        self.assertIsNotNone(netcdf_repository)
        utc_period=UtcPeriod(utc_calendar.time(YMDhms(1990,1,1,0,0,0)),utc_calendar.time(YMDhms(2000,1,1,0,0,0)))
        type_source_map=dict()
        type_source_map['temperature']=TemperatureSource
        geo_ts_dict=netcdf_repository.get_timeseries(type_source_map, geo_location_criteria=None, utc_period=utc_period)
        self.assertIsNotNone(geo_ts_dict)
        