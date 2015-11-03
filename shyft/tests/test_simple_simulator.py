"""
Tests for the simple simulator.
"""
from __future__ import print_function
from __future__ import absolute_import

from os import path
import random
import unittest
import numpy as np
from functools import reduce
import operator

from shyft.repository.netcdf import RegionModelRepository
from shyft.repository.geo_ts_repository_collection import GeoTsRepositoryCollection
from shyft.repository.netcdf import AromeDataRepository
from shyft.repository.netcdf import GeoTsRepository
from shyft.repository.interpolation_parameter_repository import InterpolationParameterRepository
from shyft.repository.netcdf.yaml_config import YamlContent
from shyft.repository.netcdf.yaml_config import RegionConfig
from shyft.repository.netcdf.yaml_config import ModelConfig
from shyft.repository.default_state_repository import DefaultStateRepository
from shyft.api import pt_gs_k
from shyft import api
from shyft.orchestration.simulator import SimpleSimulator
from shyft import shyftdata_dir


class SimulationTestCase(unittest.TestCase):

    def setUp(self):

        self.region_config_file = path.join(path.dirname(__file__), "netcdf", "atnasjoen_region.yaml")
        self.model_config_file = path.join(path.dirname(__file__), "netcdf", "model.yaml")

    def test_run_arome_data_simulator(self):
        # Simulation time axis
        year, month, day, hour = 2015, 8, 23, 6
        n_hours = 30
        dt = api.deltahours(1)
        utc = api.Calendar()  # No offset gives Utc
        t0 = utc.time(api.YMDhms(year, month, day, hour))
        time_axis = api.Timeaxis(t0, dt, n_hours)

        # Some dummy ids not needed for the netcdf based repositories
        region_id = 0
        interpolation_id = 0

        # Simulation coordinate system
        epsg = "32633"

        # Model
        model_t = pt_gs_k.PTGSKModel

        # Configs and repositories
        region_config = RegionConfig(self.region_config_file)
        model_config = ModelConfig(self.model_config_file)
        region_model_repository = RegionModelRepository(region_config, model_config,model_t, epsg)
        interp_repos = InterpolationParameterRepository(model_config)
        date_str = "{}{:02}{:02}_{:02}".format(year, month, day, hour)
        base_dir = path.join(shyftdata_dir, "repository", "arome_data_repository")
        f1 = "arome_metcoop_red_default2_5km_{}.nc".format(date_str)
        f2 = "arome_metcoop_red_test2_5km_{}.nc".format(date_str)

        ar1 = AromeDataRepository(epsg, base_dir, filename=f1, allow_subset=True)
        ar2 = AromeDataRepository(epsg, base_dir, filename=f2, elevation_file=f1, allow_subset=True)

        geo_ts_repository = GeoTsRepositoryCollection([ar1, ar2])

        simulator = SimpleSimulator(region_id,
                                    interpolation_id,
                                    region_model_repository,
                                    geo_ts_repository,
                                    interp_repos,
                                    None)
        n_cells = simulator.region_model.size()
        state_repos = DefaultStateRepository(model_t, n_cells)
        simulator.run(time_axis, state_repos.get_state(0))

    def test_set_observed_state(self):
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

        # Model
        model_t = pt_gs_k.PTGSKModel

        # Configs and repositories
        dataset_config_file = path.join(path.dirname(__file__), "netcdf", "atnasjoen_datasets.yaml")
        region_config = RegionConfig(self.region_config_file)
        model_config = ModelConfig(self.model_config_file)
        dataset_config = YamlContent(dataset_config_file)
        region_model_repository = RegionModelRepository(region_config, model_config,model_t, epsg)
        interp_repos = InterpolationParameterRepository(model_config)
        netcdf_geo_ts_repos = []
        for source in dataset_config.sources:
            station_file = source["params"]["stations_met"]
            netcdf_geo_ts_repos.append(GeoTsRepository(source["params"], station_file, ""))
        geo_ts_repository = GeoTsRepositoryCollection(netcdf_geo_ts_repos)
        simulator = SimpleSimulator(region_id, interpolation_id, region_model_repository,
                                    geo_ts_repository, interp_repos, None)
        n_cells = simulator.region_model.size()
        state_repos = DefaultStateRepository(model_t, n_cells)
        state = state_repos.get_state(0) 
        simulator.run(time_axis, state)
        simulator.region_model.get_states(state)
        obs_discharge = 0.0
        state = simulator.discharge_adjusted_state(obs_discharge, state)
        tot_cell_areas = reduce(operator.add, (cell.geo.area()
                                for cell in simulator.region_model.get_cells()))
        
        self.assertAlmostEqual(0.0, reduce(operator.add, (state[i].kirchner.q for i 
                                                          in range(state.size()))))
        simulator.region_model.get_states(state)

        obs_discharge = 10.0 # m3/s
        state = simulator.discharge_adjusted_state(obs_discharge, state)

        # Convert from l/h to m3/s by dividing by 3.6e6
        adj_discharge = reduce(operator.add, (state[i].kirchner.q*cell.geo.area() for (i, cell) 
                               in enumerate(simulator.region_model.get_cells())))/(3.6e6)
        self.assertAlmostEqual(obs_discharge, adj_discharge)

    def test_run_geo_ts_data_simulator(self):
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

        # Model
        model_t = pt_gs_k.PTGSKModel

        # Configs and repositories
        dataset_config_file = path.join(path.dirname(__file__), "netcdf", "atnasjoen_datasets.yaml")
        region_config = RegionConfig(self.region_config_file)
        model_config = ModelConfig(self.model_config_file)
        dataset_config = YamlContent(dataset_config_file)
        region_model_repository = RegionModelRepository(region_config, model_config,model_t, epsg)
        interp_repos = InterpolationParameterRepository(model_config)
        netcdf_geo_ts_repos = []
        for source in dataset_config.sources:
            station_file = source["params"]["stations_met"]
            netcdf_geo_ts_repos.append(GeoTsRepository(source["params"], station_file, ""))
        geo_ts_repository = GeoTsRepositoryCollection(netcdf_geo_ts_repos)
        simulator = SimpleSimulator(region_id, interpolation_id, region_model_repository,
                                    geo_ts_repository, interp_repos, None)
        n_cells = simulator.region_model.size()
        state_repos = DefaultStateRepository(model_t, n_cells)
        simulator.run(time_axis, state_repos.get_state(0))

    def test_calibration(self):
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

        # Model
        model_t = pt_gs_k.PTGSKOptModel

        # Configs and repositories
        dataset_config_file = path.join(path.dirname(__file__), "netcdf", "atnasjoen_datasets.yaml")
        region_config_file = path.join(path.dirname(__file__), "netcdf", "atnsjoen_calibration_region.yaml")
        region_config = RegionConfig(region_config_file)
        model_config = ModelConfig(self.model_config_file)
        dataset_config = YamlContent(dataset_config_file)
        region_model_repository = RegionModelRepository(region_config, model_config,model_t, epsg)
        interp_repos = InterpolationParameterRepository(model_config)
        netcdf_geo_ts_repos = []
        for source in dataset_config.sources:
            station_file = source["params"]["stations_met"]
            netcdf_geo_ts_repos.append(GeoTsRepository(source["params"], station_file, ""))
        geo_ts_repository = GeoTsRepositoryCollection(netcdf_geo_ts_repos)

        # Construct target discharge series
        simulator = SimpleSimulator(region_id, interpolation_id, region_model_repository,
                                    geo_ts_repository, interp_repos, None)
        n_cells = simulator.region_model.size()
        state_repos = DefaultStateRepository(model_t, n_cells)
        simulator.run(time_axis, state_repos.get_state(0))
        cid = 1
        target_discharge = simulator.region_model.statistics.discharge([cid])

        # Perturb parameters
        param = simulator.region_model.get_region_parameter()
        p_vec_orig = [param.get(i) for i in range(param.size())]
        p_vec_min = p_vec_orig[:]
        p_vec_max = p_vec_orig[:]
        p_vec_guess = p_vec_orig[:]
        random.seed(0)
        p_names = []
        for i in range(4):
            p_names.append(param.get_name(i))
            p_vec_min[i] *= 0.5
            p_vec_max[i] *= 1.5
            p_vec_guess[i] = random.uniform(p_vec_min[i], p_vec_max[i])
            if p_vec_min[i] > p_vec_max[i]:
                p_vec_min[i], p_vec_max[i] = p_vec_max[i], p_vec_min[i] 
        p_min = simulator.region_model.parameter_t()
        p_max = simulator.region_model.parameter_t()
        p_guess = simulator.region_model.parameter_t()
        p_min.set(p_vec_min)
        p_max.set(p_vec_max)
        p_guess.set(p_vec_guess)

        # Find parameters
        target_spec = api.TargetSpecificationPts(target_discharge, api.IntVector([cid]), 1.0, api.KLING_GUPTA)
        target_spec_vec = api.TargetSpecificationVector([target_spec])
        p_opt = simulator.optimize(time_axis, state_repos.get_state(0), target_spec_vec, p_guess, p_min, p_max)

        simulator.region_model.set_catchment_parameter(cid, p_opt)
        simulator.run(time_axis, state_repos.get_state(0))
        found_discharge = simulator.region_model.statistics.discharge([cid])
                        
        t_vs = np.array([target_discharge.value(i) for i in range(target_discharge.size())])
        t_ts = np.array([target_discharge.time(i) for i in range(target_discharge.size())])
        f_vs = np.array([found_discharge.value(i) for i in range(found_discharge.size())])
        f_ts = np.array([found_discharge.time(i) for i in range(found_discharge.size())])
        self.assertTrue(np.linalg.norm(t_ts - f_ts) < 1.0e-10)
        self.assertTrue(np.linalg.norm(t_vs - f_vs) < 1.0e-4)

    def test_run_arome_ensemble(self):
        # Simulation time axis
        year, month, day, hour = 2015, 7, 26, 0
        n_hours = 30
        dt = api.deltahours(1)
        utc = api.Calendar()  # No offset gives Utc
        t0 = utc.time(api.YMDhms(year, month, day, hour))
        time_axis = api.Timeaxis(t0, dt, n_hours)

        # Some dummy ids not needed for the netcdf based repositories
        region_id = 0
        interpolation_id = 0

        # Simulation coordinate system
        epsg = "32633"

        # Model
        model_t = pt_gs_k.PTGSKOptModel

        # Configs and repositories
        region_config = RegionConfig(self.region_config_file)
        model_config = ModelConfig(self.model_config_file)
        region_model_repository = RegionModelRepository(region_config, model_config,model_t, epsg)
        interp_repos = InterpolationParameterRepository(model_config)
        base_dir = path.join(shyftdata_dir, "netcdf", "arome")
        pattern = "fc*.nc"
        try:
            geo_ts_repository = AromeDataRepository(epsg, base_dir, filename=pattern, allow_subset=True)
        except Exception as e:
            print("**** test_run_arome_ensemble: Arome data missing or wrong, test inconclusive ****")
            print("****{}****".format(e))
            self.skipTest("**** test_run_arome_ensemble: Arome data missing or wrong, test inconclusive ****\n\t exception:{}".format(e))
        simulator = SimpleSimulator(region_id,
                                    interpolation_id,
                                    region_model_repository,
                                    geo_ts_repository,
                                    interp_repos,
                                    None)
        n_cells = simulator.region_model.size()
        state_repos = DefaultStateRepository(model_t, n_cells)
        simulators = simulator.create_ensembles(time_axis, t0, state_repos.get_state(0))
        for s in simulators:
            s.simulate()

if __name__ == '__main__':
    unittest.main()
