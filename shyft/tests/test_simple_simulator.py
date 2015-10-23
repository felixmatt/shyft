"""
Tests for the simple simulator.
"""

from __future__ import print_function
from __future__ import absolute_import
import random
from os import path

import unittest

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
        cid = 1
        target_discharge = simulator.region_model.statistics.discharge([cid])

        simulator = SimpleSimulator(region_id, interpolation_id, region_model_repository,
                                    geo_ts_repository, interp_repos, None)
        n_cells = simulator.region_model.size()
        state_repos = DefaultStateRepository(model_t, n_cells)
        param = simulator.region_model.get_catchment_parameter(cid)
        p_orig = [param.get(i) for i in range(param.size())]
        p = p_orig[:]
        p_min = p[:]
        p_max = p[:]
        for i in range(4):
            p_min[i] *= 0.5
            p_max[i] *= 1.5
            p[i] = random.uniform(p_min[i], p_max[i])
            if p_min[i] > p_max[i]:
                p_min[i], p_max[i] = p_max[i], p_min[i] 
        print("Min,", p_min[:4])
        print("Max,", p_max[:4])
        print("Guess,", p[:4])
        target_spec = api.TargetSpecificationPts(target_discharge, api.IntVector([cid]), 1.0, api.KLING_GUPTA)
        tsv = api.TargetSpecificationVector([target_spec])
        p_opt = simulator.optimize(time_axis, state_repos.get_state(0), tsv, p, p_min, p_max)
        print("True,", p_orig[:4])
        print("Computed,", [p for p in p_opt][:4])
                        
        #vs = [discharge.value(i) for i in range(discharge.size())]
        #ts = [discharge.time(i) for i in range(discharge.size())]
        #from matplotlib import pylab as plt
        #plt.plot(ts, vs)
        #plt.show()



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
