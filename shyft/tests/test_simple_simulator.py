"""
Tests for the simple simulator.
"""

from __future__ import print_function
from __future__ import absolute_import
from os import path

import unittest
from shyft.repository.netcdf import RegionModelRepository
from shyft.repository.netcdf import AromeDataRepository
from shyft.repository.netcdf import GeoTsRepository
from shyft.repository.interpolation_parameter_repository import InterpolationParameterRepository
from shyft.repository.yaml_config import YamlContent
from shyft.repository.yaml_config import RegionConfig
from shyft.repository.yaml_config import ModelConfig
from shyft.repository.default_state_repository import DefaultStateRepository
from shyft.api import pt_gs_k
from shyft import api
from shyft.orchestration.simulator import SimpleSimulator
from shyft import shyftdata_dir


class SimulationTestCase(unittest.TestCase):

    def setUp(self):

        self.region_config_file = path.join(path.dirname(__file__), "netcdf", "atnasjoen_region.yaml")
        self.model_config_file = path.join(path.dirname(__file__), "netcdf", "model.yaml")

    def test_construct_arome_data_simulator(self):
        return

        # Simulation time axis
        year, month, day, hour = 2015, 8, 23, 6
        n_hours = 30
        dt = api.deltahours(1)
        utc = api.Calendar()  # No offset gives Utc
        t0 = utc.time(api.YMDhms(year, month, day, hour))
        time_axis = api.Timeaxis(t0, dt, n_hours)

        # Some fake ids
        region_id = 0
        interpolation_id = 0
        
        # Simulation coordinate system
        epsg = "32633"

        # Configs and repositories
        region_config = RegionConfig(self.region_config_file)
        model_config = ModelConfig(self.model_config_file)
        region_model_repository = RegionModelRepository(region_config, model_config, epsg)
        interp_repos = InterpolationParameterRepository(model_config)
        date_str = "{}{:02}{:02}_{:02}".format(year, month, day, hour)
        #base_dir = path.join(shyftdata_dir, "repository", "arome_data_repository")
        base_dir = path.join(shyftdata_dir, "netcdf", "arome")
        f1 = "arome_metcoop_default2_5km_{}.nc".format(date_str)
        f2 = "arome_metcoop_test2_5km_{}.nc".format(date_str)

        ar1 = AromeDataRepository(epsg, base_dir, filename=f1, allow_subset=True)
        ar2 = AromeDataRepository(epsg, base_dir, filename=f2, elevation_file=f1, allow_subset=True)

        simulator = SimpleSimulator(pt_gs_k.PTGSKModel, 
                                    region_id, 
                                    interpolation_id, 
                                    region_model_repository, 
                                    [ar1, ar2], 
                                    interp_repos, 
                                    None)
        n_cells = simulator.region_model.size()
        state_repos = DefaultStateRepository(pt_gs_k.PTGSKState, pt_gs_k.PTGSKStateVector, n_cells)
        simulator.run(time_axis, state_repos.get_state(0))

    def test_construct_geo_ts_data_simulator(self):
        # Simulation time axis
        year, month, day, hour = 2010, 1, 1, 0
        dt = 24*api.deltahours(1)
        n_steps = 30
        utc = api.Calendar()  # No offset gives Utc
        t0 = utc.time(api.YMDhms(year, month, day, hour))
        time_axis = api.Timeaxis(t0, dt, n_steps)

        # Some fake ids
        region_id = 0
        interpolation_id = 0
        
        # Simulation coordinate system
        epsg = "32633"

        # Configs and repositories
        dataset_config_file = path.join(path.dirname(__file__), "netcdf", "atnasjoen_datasets.yaml")
        region_config = RegionConfig(self.region_config_file)
        model_config = ModelConfig(self.model_config_file)
        dataset_config = YamlContent(dataset_config_file)
        region_model_repository = RegionModelRepository(region_config, model_config, epsg)
        interp_repos = InterpolationParameterRepository(model_config)
        geo_ts_repos = []
        for source in dataset_config.sources:
            station_file = source["params"]["stations_met"]
            geo_ts_repos.append(GeoTsRepository(source["params"], station_file, ""))

        simulator = SimpleSimulator(pt_gs_k.PTGSKModel, 
                                    region_id, 
                                    interpolation_id, 
                                    region_model_repository, 
                                    geo_ts_repos, 
                                    interp_repos, 
                                    None)
        n_cells = simulator.region_model.size()
        state_repos = DefaultStateRepository(pt_gs_k.PTGSKState, pt_gs_k.PTGSKStateVector, n_cells)
        simulator.run(time_axis, state_repos.get_state(0))

