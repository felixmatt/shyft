﻿import unittest

try:
    from shyft.repository.service.ssa_geo_ts_repository import GeoTsRepository
    from shyft.repository.service.ssa_geo_ts_repository import MetStationConfig
    from shyft.repository.service.ssa_geo_ts_repository import EnsembleStation
    from shyft.repository.service.ssa_geo_ts_repository import EnsembleConfig
    from shyft.repository.service.gis_location_service import GisLocationService
    from shyft.repository.service.ssa_smg_db import SmGTsRepository
    from shyft.repository.service.ssa_smg_db import PREPROD, FC_PREPROD
    from shyft.api import Calendar
    from shyft.api import YMDhms
    from shyft.api import UtcPeriod
    from math import fabs
    
    
    class SSAGeoTsRepositoryTestCase(unittest.TestCase):
        """
        NOTE: These tests are Statkraft specific in the sense that they require
              statkraft script api (SSA) components (relying on Powel Tss services) 
              and statkraft gis (esri type) of published geo-services.
              
        Verify that we correctly can read geo-located timeseries from a the SSA service based
        location and ts-store.
        """

        def test_gis_location_service(self):
            glr=GisLocationService()
            nea_nid=[402, 460]
            # we test that for the same point, different projections, we get same heights and approx. different positions
            utm32_loc=glr.get_locations(nea_nid, 32632)
            utm33_loc=glr.get_locations(nea_nid, 32633)
            self.assertIsNotNone(utm32_loc)
            self.assertIsNotNone(utm33_loc)
            for p in nea_nid:
                self.assertAlmostEqual(utm32_loc[p][2],utm33_loc[p][2])
                self.assertLess(fabs(utm32_loc[p][1]-utm33_loc[p][1]),10*1000,"expect y same")
                self.assertGreater(fabs(utm32_loc[p][0]-utm33_loc[p][0]),30*1000,"expect x diff same")

        def test_get_timeseries_using_known_service_and_db_content(self):
            utc = Calendar()
            met_stations=[ # this is the list of MetStations, the gis_id tells the position, the remaining tells us what properties we observe/forecast/calculate at the metstation (smg-ts)
                MetStationConfig(gis_id=402, temperature=u'/NeNi-Sylsjøen......-T0017V3KI0114', precipitation=u'/NeNi-Sylsjøen-2....-T0000D9BI0124'),
                MetStationConfig(gis_id=460, temperature=u'/NeNi-Stuggusjøen...-T0017V3KI0114', precipitation=u'/NeNi-Stuggusjøen...-T0000D9BI0124')
            ]
            gis_location_repository = GisLocationService()
            smg_ts_repository = SmGTsRepository(PREPROD, FC_PREPROD)
            geo_ts_repository = GeoTsRepository(epsg_id = 32633,
                geo_location_repository = gis_location_repository,
                ts_repository = smg_ts_repository,
                met_station_list = met_stations,
                ens_config = None)
    
            self.assertIsNotNone(geo_ts_repository)
            utc_period = UtcPeriod(utc.time(2016, 1, 1, 0, 0, 0), utc.time(2016, 1, 2, 0, 0, 0))
            ts_types= ['temperature', 'precipitation']
            geo_ts_dict = geo_ts_repository.get_timeseries(ts_types, utc_period=utc_period, geo_location_criteria=None)
            self.assertIsNotNone(geo_ts_dict)
            for ts_type in ts_types:
                self.assertTrue(ts_type in geo_ts_dict.keys(), "we expect to find an entry for each requested type (it could be empty list though)")
                self.assertTrue(len(geo_ts_dict[ts_type])>0, "we expect to find the series that we pass in, given by name in SmG PREPROD")
    
        def test_get_forecast_using_known_service_and_db_content(self):
            utc = Calendar()
            met_stations=[ # this is the list of MetStations, the gis_id tells the position, the remaining tells us what properties we observe/forecast/calculate at the metstation (smg-ts)
                MetStationConfig(gis_id=402, temperature=u'/LTM5-Nea...........-T0017A3P_EC00_ENS', precipitation=u'/LTM5-Nea...........-T0000A5P_EC00_ENS')
            ]
            gis_location_repository = GisLocationService()
            smg_ts_repository = SmGTsRepository(PREPROD, FC_PREPROD)
            geo_ts_repository = GeoTsRepository(
                epsg_id = 32633,
                geo_location_repository = gis_location_repository,
                ts_repository = smg_ts_repository,
                met_station_list = met_stations,
                ens_config = None)
    
            self.assertIsNotNone(geo_ts_repository)
            utc_period = UtcPeriod(utc.time(2015, 10, 1, 0, 0, 0), utc.time(2015, 10, 10, 0, 0, 0))
            ts_types= ['temperature', 'precipitation']
            geo_ts_dict = geo_ts_repository.get_forecast(ts_types, utc_period=utc_period, t_c=None, geo_location_criteria=None)
            self.assertIsNotNone(geo_ts_dict)
            for ts_type in ts_types:
                self.assertTrue(ts_type in geo_ts_dict.keys(), "we expect to find an entry for each requested type (it could be empty list though)")
                self.assertTrue(len(geo_ts_dict[ts_type])>0, "we expect to find the series that we pass in by name in SmG PREPROD")
    
        def test_get_ensemble_forecast_using_known_service_and_db_content(self):
            utc = Calendar()
            met_stations = [MetStationConfig(gis_id=402, temperature=u'/LTM5-Nea...........-T0017A3P_EC00_ENS', precipitation=u'/LTM5-Nea...........-T0000A5P_EC00_ENS')]
            gis_location_repository = GisLocationService()
            smg_ts_repository = SmGTsRepository(PREPROD, FC_PREPROD)
            n_ensembles = 51
            ens_station_list = [
                EnsembleStation(402, n_ensembles,
                    temperature_ens = lambda i:u'/LTM5-Nea...........-T0017A3P_EC00_E{0:02}'.format(i),
                    precipitation_ens = lambda i:u'/LTM5-Nea...........-T0000A5P_EC00_E{0:02}'.format(i),
                    wind_speed_ens = None,
                    radiation_ens = None,
                    relative_humidity_ens = None
                ),
                EnsembleStation(460, n_ensembles,
                    temperature_ens = lambda i:u'/LTM5-Tya...........-T0017A3P_EC00_E{0:02}'.format(i),
                    precipitation_ens = lambda i:u'/LTM5-Tya...........-T0000A5P_EC00_E{0:02}'.format(i),
                    wind_speed_ens = None,
                    radiation_ens = None,
                    relative_humidity_ens = None
                )
            ]
            ens_config = EnsembleConfig(n_ensembles, ens_station_list)
            geo_ts_repository = GeoTsRepository(
                epsg_id = 32633,
                geo_location_repository = gis_location_repository,
                ts_repository = smg_ts_repository,
                met_station_list = met_stations,
                ens_config = ens_config)
    
            self.assertIsNotNone(geo_ts_repository)
            utc_period = UtcPeriod(utc.time(2015, 10, 1, 0, 0, 0),utc.time(2015, 10, 10, 0, 0, 0))
            ts_types= ['temperature', 'precipitation']
            ens_geo_ts_dict = geo_ts_repository.get_forecast_ensemble(ts_types, utc_period=utc_period, t_c=None, geo_location_criteria=None)
            self.assertIsNotNone(ens_geo_ts_dict)
            self.assertEqual(ens_config.n_ensembles, len(ens_geo_ts_dict))
            for i in range(ens_config.n_ensembles):
                for ts_type in ts_types:
                    self.assertTrue(ts_type in ens_geo_ts_dict[i].keys(), "we expect to find an entry for each requested type (it could be empty list though)")
                    self.assertTrue(len(ens_geo_ts_dict[i][ts_type])>0, "we expect to find the series that we pass in by name in SmG PREPROD")
    

except ImportError as ie:
    if 'statkraft' in str(ie):
        print("(Test require statkraft.script environment to run: {})".format(ie))
    else:
        print("ImportError: {}".format(ie))
        

if __name__ == '__main__':
    unittest.main()
