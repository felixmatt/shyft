from __future__ import absolute_import
import unittest

from shyft.repository.service.ssa_geo_ts_repository import GeoTsRepository
from shyft.repository.service.ssa_geo_ts_repository import MetStationConfig
from shyft.api import Calendar
from shyft.api import YMDhms
from shyft.api import UtcPeriod


class SSAGeoTsRepositoryTestCase(unittest.TestCase):
    """
    Verify that we correctly can read geo-located timeseries from a the SSA service based
    location and ts-store.
    """

    def test_using_known_service_and_db_content(self):
        utc = Calendar() # always use Calendar() stuff
        met_stations=[ # this is the list of MetStations, the gis_id tells the position, the remaining tells us what properties we observe/forecast/calculate at the metstation (smg-ts)
            MetStationConfig(gis_id=598,temperature=u'/NeNi-Sylsjøen......-T0017V3KI0114',precipitation=u'/NeNi-Sylsjøen-2....-T0000D9BI0124'),
            MetStationConfig(gis_id=574,temperature=u'/NeNi-Stuggusjøen...-T0017V3KI0114',precipitation=u'/NeNi-Stuggusjøen...-T0000D9BI0124',radiation=u'/ENKI/STS/Radiation/Sim.-Stuggusjøen...-T0006A0B-0119')
        ]
        #note: the MetStationConfig can be constructed from yaml-config
        geo_ts_repository = GeoTsRepository(gis_service="",smg_service='prod',met_station_list=met_stations) #pass service info and met_stations

        self.assertIsNotNone(geo_ts_repository)
        utc_period = UtcPeriod(utc.time(YMDhms(2010, 1, 1, 0, 0, 0)),utc.time(YMDhms(2010, 1, 2, 0, 0, 0)))
        ts_types= ['temperature','precipitation','radiation']
        geo_ts_dict = geo_ts_repository.get_timeseries(ts_types,utc_period=utc_period,geo_location_criteria=None)
        self.assertIsNotNone(geo_ts_dict)
        for ts_type in ts_types:
            self.assertTrue(geo_ts_dict.has_key(ts_type),"we ecpect to find an entry for each requested type (it could be empty list though")
            self.assertTrue(geo_ts_dict[ts_type].size()>0,"we expect to find the series that we pass in, given they have not changed the name in SmG PROD")

