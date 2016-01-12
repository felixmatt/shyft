# -*- coding: utf-8 -*-
import unittest

try:
    from os import path
    from shyft import api
    from shyft.api import Calendar, YMDhms, Timeaxis, deltahours
    from shyft.api import pt_gs_k
    from shyft.api import pt_ss_k
    from shyft.repository.default_state_repository import DefaultStateRepository
    from shyft.repository.service.gis_region_model_repository import GridSpecification
    from shyft.repository.service.gis_region_model_repository import RegionModelConfig
    from shyft.repository.service.gis_region_model_repository import GisRegionModelRepository
    from shyft.repository.interpolation_parameter_repository import InterpolationParameterRepository
    from shyft.repository.service.ssa_geo_ts_repository import GeoTsRepository
    from shyft.repository.service.ssa_geo_ts_repository import MetStationConfig
    from shyft.repository.service.gis_location_service import GisLocationService
    from shyft.repository.service.ssa_smg_db import SmGTsRepository, PROD, FC_PROD, PREPROD, FC_PREPROD
    from shyft.orchestration.simulator import DefaultSimulator
    from shyft.repository.interfaces import TsStoreItem
    from shyft.repository.interfaces import TimeseriesStore
    from shyft.repository.netcdf.arome_data_repository import AromeDataRepository
    from shyft.repository.geo_ts_repository_collection import GeoTsRepositoryCollection


    class InterpolationConfig(object):
        """ A bit clumsy, but to reuse dictionary based InterpolationRepository:"""

        def interpolation_parameters(self):
            return {
                'btk': {
                    'gradient': -0.6,
                    'gradient_sd': 0.25,
                    'nugget': 0.5,
                    'range': 2000000.0,
                    'sill': 25.0,
                    'zscale': 20.0,
                },

                'idw': {
                    'max_distance': 2000000.0,
                    'max_members': 10,
                    'precipitation_gradient': 2.0
                }
            }


    class StatkraftTistelTest(unittest.TestCase):
        """
        This is a test/demo class, showing how we could run catchment Tistel (vik in sogn)
        using statkraft repositories (incl gis and db-services),
        and the orchestator DefaultSimulator to do the work.
        """

        def test_run_observed_then_arome_and_store(self):
            """
              Start Tistel 2015.09.01, dummy state with some kirchner water
               use observations around Tistel (geo_ts_repository)
               and simulate forwared to 2015.10.01 (store discharge and catchment level precip/temp)
               then use arome forecast for 65 hours (needs arome for this period in arome-directory)
               finally store the arome results.

            """
            utc = Calendar()  # No offset gives Utc
            time_axis = Timeaxis(utc.time(YMDhms(2015, 9, 1, 0)), deltahours(1), 30 * 24)
            fc_time_axis = Timeaxis(utc.time(YMDhms(2015, 10, 1, 0)), deltahours(1), 65)

            interpolation_id = 0
            ptgsk = DefaultSimulator("Tistel-ptgsk",
                                     interpolation_id,
                                     self.region_model_repository,
                                     self.geo_ts_repository,
                                     self.interpolation_repository, None)
            n_cells = ptgsk.region_model.size()
            ptgsk_state = DefaultStateRepository(ptgsk.region_model.__class__, n_cells)

            ptgsk.region_model.set_state_collection(-1, True)  # collect state so we can inspect it
            s0 = ptgsk_state.get_state(0)
            for i in range(s0.size()):  # add some juice to get started
                s0[i].kirchner.q = 0.5

            ptgsk.run(time_axis, s0)

            print("Done simulation, testing that we can extract data from model")

            cids = api.IntVector()  # we pull out for all the catchments-id if it's empty
            model = ptgsk.region_model  # fetch out  the model
            sum_discharge = model.statistics.discharge(cids)
            self.assertIsNotNone(sum_discharge)
            avg_temperature = model.statistics.temperature(cids)
            avg_precipitation = model.statistics.precipitation(cids)
            self.assertIsNotNone(avg_precipitation)
            self.assertIsNotNone(avg_temperature)
            for time_step in range(time_axis.size()):
                precip_raster = model.statistics.precipitation(cids, time_step)  # example raster output
                self.assertEqual(precip_raster.size(), n_cells)
            avg_gs_lwc = model.gamma_snow_state.lwc(cids)  # sca skaugen|gamma
            self.assertIsNotNone(avg_gs_lwc)
            # lwc surface_heat alpha melt_mean melt iso_pot_energy temp_sw
            avg_gs_output = model.gamma_snow_response.outflow(cids)
            self.assertIsNotNone(avg_gs_output)
            print("done. now save to db")
            # SmGTsRepository(PROD,FC_PROD)
            save_list = [
                TsStoreItem(u'/test/sih/shyft/tistel/discharge_m3s', lambda m: m.statistics.discharge(cids)),
                TsStoreItem(u'/test/sih/shyft/tistel/temperature', lambda m: m.statistics.temperature(cids)),
                TsStoreItem(u'/test/sih/shyft/tistel/precipitation', lambda m: m.statistics.precipitation(cids)),
            ]

            tss = TimeseriesStore(SmGTsRepository(PREPROD, FC_PREPROD), save_list)

            self.assertTrue(tss.store_ts(ptgsk.region_model))

            print("Run forecast arome")
            endstate = ptgsk.region_model.state_t.vector_t()
            ptgsk.region_model.get_states(endstate)  # get the state at end of obs
            ptgsk.geo_ts_repository = self.arome_repository  # switch to arome here
            ptgsk.run_forecast(fc_time_axis, fc_time_axis.start(), endstate)  # now forecast
            print("Done forecast")
            fc_save_list = [
                TsStoreItem(u'/test/sih/shyft/tistel/fc_discharge_m3s', lambda m: m.statistics.discharge(cids)),
                TsStoreItem(u'/test/sih/shyft/tistel/fc_temperature', lambda m: m.statistics.temperature(cids)),
                TsStoreItem(u'/test/sih/shyft/tistel/fc_precipitation', lambda m: m.statistics.precipitation(cids)),
                TsStoreItem(u'/test/sih/shyft/tistel/fc_radiation', lambda m: m.statistics.radiation(cids)),
                TsStoreItem(u'/test/sih/shyft/tistel/fc_rel_hum', lambda m: m.statistics.rel_hum(cids)),
                TsStoreItem(u'/test/sih/shyft/tistel/fc_wind_speed', lambda m: m.statistics.wind_speed(cids)),

            ]
            TimeseriesStore(SmGTsRepository(PREPROD, FC_PREPROD), fc_save_list).store_ts(ptgsk.region_model)
            print("Done save to db")

        @property
        def statkraft_data_dir(self):
            return path.abspath(path.join("D:", path.join(path.sep, "statkraft_data")))

        @property
        def region_model_repository(self):
            """
            Returns
            -------
             - RegionModelRepository - configured with 'Tistel-ptgsk' etc.
            """
            id_list = [1225]
            # parameters can be loaded from yaml_config Model parameters..
            pt_params = api.PriestleyTaylorParameter()  # *params["priestley_taylor"])
            gs_params = api.GammaSnowParameter()  # *params["gamma_snow"])
            ss_params = api.SkaugenParameter()
            ae_params = api.ActualEvapotranspirationParameter()  # *params["act_evap"])
            k_params = api.KirchnerParameter()  # *params["kirchner"])
            p_params = api.PrecipitationCorrectionParameter()  # TODO; default 1.0, is it used ??
            ptgsk_rm_params = pt_gs_k.PTGSKParameter(pt_params, gs_params, ae_params, k_params, p_params)
            ptssk_rm_params = pt_ss_k.PTSSKParameter(pt_params, ss_params, ae_params, k_params, p_params)
            # create the description for 2 models of tistel,ptgsk, ptssk
            tistel_grid_spec = self.grid_spec  #
            cfg_list = [
                RegionModelConfig("Tistel-ptgsk", pt_gs_k.PTGSKModel, ptgsk_rm_params, tistel_grid_spec, "unregulated", "FELTNR", id_list),
                RegionModelConfig("Tistel-ptssk", pt_ss_k.PTSSKModel, ptssk_rm_params, tistel_grid_spec, "unregulated", "FELTNR", id_list)
            ]
            rm_cfg_dict = {x.name: x for x in cfg_list}
            return GisRegionModelRepository(rm_cfg_dict)

        @property
        def fc_geo_ts_repository(self):
            """
            Returns
            -------
             - geo_ts_repository that have met-station-config relevant for tistel
            """

            met_stations = [
                # this is the list of MetStations, the gis_id tells the position, the remaining tells us what properties we observe/forecast/calculate at the metstation (smg-ts)
                MetStationConfig(gis_id=218,  # 0 midtpunkt
                                 temperature=u'/Vikf-Tistel........-T0017A3P_MAN',
                                 precipitation=u'/Vikf-Tistel........-T0000A5P_MAN',
                                 radiation=u'/ENKI/STS/Radiation/Sim.-Hestvollan....-T0006V0B-0119-0.8',
                                 wind_speed=u'/Dnmi-Fjærland.Bremu-T0016V3K-A55820-332',
                                 relative_humidity=u'/SHFT-rel-hum-dummy.-T0002A3R-0103')
            ]

            gis_location_repository = GisLocationService()  # yaml... geo_location service..this provides the gis locations for my stations
            smg_ts_repository = SmGTsRepository(PROD, FC_PROD)  # this provide the read function for my time-series

            return GeoTsRepository(  # together, the location provider, ts-provider, and the station, we have
                epsg_id=self.epsg_id,
                geo_location_repository=gis_location_repository,  # a complete geo_ts-repository
                ts_repository=smg_ts_repository,
                met_station_list=met_stations,
                ens_config=None)  # pass service info and met_stations

        @property
        def epsg_id(self):
            return self.grid_spec.epsg()

        @property
        def grid_spec(self):
            return GridSpecification(epsg_id=32633, x0=35000.0, y0=6788000.0, dx=1000, dy=1000, nx=16, ny=17)

        @property
        def arome_repository(self):
            """ 
            This shows how we
            join together two arome repositories to a repository that gives us all the variabes that we need.
             arome_4: all except radiation
             arome_rad: ratiation
            Return
            ------
            Arome repository (inside a GeoTsRepositoryCollection), with all datafiles that matches
            the filepattrns we currently use for arome downloads.

            """
            base_dir = path.join(self.statkraft_data_dir, "repository", "arome_data_repository")

            epsg = self.grid_spec.epsg()
            bbox = self.grid_spec.bounding_box(epsg)
            f1 = "arome_metcoop_default2_5km_20151001_00.nc"
            f2 = "arome_metcoop_test2_5km_20151001_00.nc"
            arome_4 = AromeDataRepository(epsg, base_dir, filename=f1,
                                          bounding_box=bbox, allow_subset=True)
            arome_rad = AromeDataRepository(epsg, base_dir, filename=f2,
                                            elevation_file=f1,
                                            bounding_box=bbox, allow_subset=True)
            return GeoTsRepositoryCollection([arome_4, arome_rad])

        @property
        def geo_ts_repository(self):
            """
            Returns
            -------
             - geo_ts_repository with observed values, that have met-station-config relevant for tistel
            """

            met_stations = [
                # this is the list of MetStations, the gis_id tells the position, the remaining tells us what properties we observe/forecast/calculate at the metstation (smg-ts)
                MetStationConfig(gis_id=129,  # 0 Fjærland Bremu
                                 temperature=None,
                                 precipitation=None,
                                 radiation=None,
                                 wind_speed=u'/Dnmi-Fjærland.Bremu-T0016V3K-A55820-332'),

                MetStationConfig(gis_id=619,  # 1 Tistel
                                 temperature=u'/Vikf-Tistel........-T0017A3KI0114',
                                 precipitation=None,
                                 radiation=None,
                                 wind_speed=None,
                                 relative_humidity=u'/SHFT-rel-hum-dummy.-T0002A3R-0103'),

                MetStationConfig(gis_id=684,  # 2 Vossevangen
                                 temperature=None,  # u'/Dnmi-Vossevangen...-T0017V3K-A51530-1337'
                                 precipitation=u'/Dnmi-Vossevangen...-T0000D9B-A51530-1337',
                                 radiation=None,
                                 wind_speed=u'/Dnmi-Vossevangen...-T0016V3K-A51530-337'),

                MetStationConfig(gis_id=654,  # 2 Vossevangen
                                 temperature=None,
                                 precipitation=u'/Dnmi-Vangsnes......-T0000D9B-A53101-338',
                                 radiation=None,
                                 wind_speed=u'/Dnmi-Vangsnes......-T0016V3K-A53101-338'),

                MetStationConfig(gis_id=218,  # 4 Hestvollan
                                 temperature=u'/Vikf-Hestvollan....-T0017A3KI0114',
                                 precipitation=u'/Vikf-Hestvollan....-T0000D9BI0124',
                                 radiation=u'/ENKI/STS/Radiation/Sim.-Hestvollan....-T0006V0B-0119-0.8',  # clear sky,reduced to 0.8
                                 wind_speed=u'/Vikf-Hestvollan....-T0015V3KI0120'),

                MetStationConfig(gis_id=542,  # 5 Sopandefjell
                                 temperature=None,  # u'/Vikf-Sopandefjell..-T0017V3KI0114'
                                 precipitation=None,
                                 radiation=None,
                                 wind_speed=None),

                MetStationConfig(gis_id=650,  # 6 Ulldalsvatnet
                                 temperature=None,
                                 precipitation=None,
                                 radiation=None,
                                 wind_speed=u'/Hoey-Ulldalsvatnet.-T0015V3KI0120'),

            ]

            gis_location_repository = GisLocationService()  # this provides the gis locations for my stations
            smg_ts_repository = SmGTsRepository(PROD, FC_PROD)  # this provide the read function for my time-series

            return GeoTsRepository(  # together, the location provider, ts-provider, and the station, we have
                epsg_id=self.epsg_id,
                geo_location_repository=gis_location_repository,  # a complete geo_ts-repository
                ts_repository=smg_ts_repository,
                met_station_list=met_stations,
                ens_config=None)  # pass service info and met_stations

        @property
        def interpolation_repository(self):
            return InterpolationParameterRepository(InterpolationConfig())





except ImportError as ie:
    if 'statkraft' in str(ie):
        print("(Test require statkraft.script environment to run: {})".format(ie))
    else:
        print("ImportError: {}".format(ie))

if __name__ == '__main__':
    unittest.main()
