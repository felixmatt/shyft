"""
Tests for the simple simulator.
"""

from os import path

import random
import unittest
import numpy as np
from functools import reduce
import operator

from shyft import shyftdata_dir
from shyft import api
from shyft.api import pt_gs_k
from shyft.api import pt_ss_k
from shyft.api import pt_hs_k
from shyft.repository.netcdf import RegionModelRepository
from shyft.repository.geo_ts_repository_collection import GeoTsRepositoryCollection
from shyft.repository.netcdf import AromeDataRepository
from shyft.repository.netcdf import GeoTsRepository
from shyft.repository.interpolation_parameter_repository import InterpolationParameterRepository
from shyft.repository.netcdf.yaml_config import YamlContent
from shyft.repository.netcdf.yaml_config import RegionConfig
from shyft.repository.netcdf.yaml_config import ModelConfig
from shyft.repository.default_state_repository import DefaultStateRepository
from shyft.orchestration.simulator import DefaultSimulator
from shyft import orchestration


def print_param(header_text, param):
    print(header_text)
    print("Kirchner:   {:8.2f}{:8.2f}{:8.2f}".format(param.kirchner.c1, param.kirchner.c2, param.kirchner.c3))
    print("Gamma snow: {:8.2f}{:8.2f}".format(param.gs.tx, param.gs.max_water))


class SimulationTestCase(unittest.TestCase):

    def setUp(self):

        self.region_config_file = path.join(path.dirname(__file__), "netcdf",
                                            "atnsjoen_region.yaml")
        self.model_config_file = path.join(path.dirname(__file__), "netcdf", "model.yaml")

    def run_simulator(self, model_t):
        # Simulation time axis
        dt0 = api.YMDhms(2015, 8, 24, 6)
        n_hours = 30
        dt = api.deltahours(1)
        utc = api.Calendar()  # No offset gives Utc
        t0 = utc.time(dt0)
        time_axis = api.TimeAxisFixedDeltaT(t0, dt, n_hours)

        # Some dummy ids not needed for the netcdf based repositories
        region_id = 0
        interpolation_id = 0

        # Simulation coordinate system
        epsg = "32633"

        # Configs and repositories
        region_config = RegionConfig(self.region_config_file)
        model_config = ModelConfig(self.model_config_file)
        region_model_repository = RegionModelRepository(region_config, model_config, model_t, epsg)
        interp_repos = InterpolationParameterRepository(model_config)
        date_str = "{}{:02}{:02}_{:02}".format(dt0.year, dt0.month, dt0.day, dt0.hour)
        base_dir = path.join(shyftdata_dir, "repository", "arome_data_repository")
        f1 = "arome_metcoop_red_default2_5km_{}.nc".format(date_str)
        f2 = "arome_metcoop_red_test2_5km_{}.nc".format(date_str)

        ar1 = AromeDataRepository(epsg, base_dir, filename=f1, allow_subset=True)
        ar2 = AromeDataRepository(epsg, base_dir, filename=f2, elevation_file=f1, allow_subset=True)

        geo_ts_repository = GeoTsRepositoryCollection([ar1, ar2])

        simulator = DefaultSimulator(region_id, interpolation_id, region_model_repository,
                                    geo_ts_repository, interp_repos, None)
        n_cells = simulator.region_model.size()
        state_repos = DefaultStateRepository(model_t, n_cells)
        simulator.run(time_axis, state_repos.get_state(0))

    def test_run_arome_data_pt_gs_k_simulator(self):
        self.run_simulator(pt_gs_k.PTGSKModel)

    def test_run_arome_data_pt_hs_k_simulator(self):
        self.run_simulator(pt_hs_k.PTHSKModel)

    def test_run_arome_data_pt_ss_k_simulator(self):
        self.run_simulator(pt_ss_k.PTSSKModel)

    def test_set_observed_state(self):
        # set up configuration
        config_dir = path.join(path.dirname(__file__), "netcdf")
        cfg = orchestration.YAMLConfig(
            "atnsjoen_simulation.yaml", "atnsjoen",
            config_dir=config_dir, data_dir=shyftdata_dir)

        # get a simulator
        simulator = cfg.get_simulator()

        n_cells = simulator.region_model.size()
        state_repos = DefaultStateRepository(cfg.model_t, n_cells)
        state = state_repos.get_state(0)
        simulator.run(cfg.time_axis, state)
        simulator.region_model.get_states(state)
        obs_discharge = 0.0
        state = simulator.discharge_adjusted_state(obs_discharge, state)

        self.assertAlmostEqual(0.0, reduce(operator.add, (state[i].kirchner.q for i
                                                          in range(state.size()))))
        simulator.region_model.get_states(state)

        obs_discharge = 10.0  # m3/s
        state = simulator.discharge_adjusted_state(obs_discharge, state)

        # Convert from l/h to m3/s by dividing by 3.6e6
        adj_discharge = reduce(operator.add, (state[i].kirchner.q*cell.geo.area() for (i, cell)
                               in enumerate(simulator.region_model.get_cells())))/(3.6e6)
        self.assertAlmostEqual(obs_discharge, adj_discharge)

    def test_run_geo_ts_data_simulator(self):
        # set up configuration
        config_dir = path.join(path.dirname(__file__), "netcdf")
        cfg = orchestration.YAMLConfig(
            "atnsjoen_simulation.yaml", "atnsjoen",
            config_dir=config_dir, data_dir=shyftdata_dir)

        # get a simulator
        simulator = cfg.get_simulator()

        n_cells = simulator.region_model.size()
        state_repos = DefaultStateRepository(cfg.model_t, n_cells)
        simulator.run(cfg.time_axis, state_repos.get_state(0))
        sim_copy = simulator.copy()
        sim_copy.run(cfg.time_axis,state_repos.get_state(0))


    def run_calibration(self, model_t):
        # set up configuration
        config_dir = path.join(path.dirname(__file__), "netcdf")
        cfg = orchestration.YAMLConfig(
            "atnsjoen_calibration.yaml", "atnsjoen",
            config_dir=config_dir, data_dir=shyftdata_dir,
            model_t=model_t)
        time_axis = cfg.time_axis

        # get a simulator
        simulator = cfg.get_simulator()

        n_cells = simulator.region_model.size()
        state_repos = DefaultStateRepository(cfg.model_t, n_cells)
        simulator.run(time_axis, state_repos.get_state(0))
        cid = 1

        target_discharge_ts = simulator.region_model.statistics.discharge([cid])
        target_discharge = api.TsTransform().to_average(time_axis.time(0), time_axis.time(1)-time_axis.time(0), time_axis.size(), target_discharge_ts)
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
        target_spec = api.TargetSpecificationPts(target_discharge, api.IntVector([cid]),
                                                 1.0, api.KLING_GUPTA)
        target_spec_vec = api.TargetSpecificationVector() #([target_spec]) does not yet work
        target_spec_vec.append(target_spec)
        self.assertEqual(simulator.optimizer.trace_size,0)  # before optmize, trace_size should be 0
        p_opt = simulator.optimize(time_axis, state_repos.get_state(0),
                                   target_spec_vec, p_guess, p_min, p_max)
        self.assertGreater(simulator.optimizer.trace_size,0)  # after opt, some trace values should be there
        # the trace values are in the order of appearance 0...trace_size-1
        #
        goal_fn_values = simulator.optimizer.trace_goal_function_values.to_numpy()  # all of them, as np array
        self.assertEqual(len(goal_fn_values),simulator.optimizer.trace_size)
        p_last = simulator.optimizer.trace_parameter(simulator.optimizer.trace_size-1)  # get out the last (not neccessary the best)
        self.assertIsNotNone(p_last)
        simulator.region_model.set_catchment_parameter(cid, p_opt)
        simulator.run(time_axis, state_repos.get_state(0))
        found_discharge = simulator.region_model.statistics.discharge([cid])

        t_vs = np.array([target_discharge.value(i) for i in range(target_discharge.size())])
        t_ts = np.array([target_discharge.time(i) for i in range(target_discharge.size())])
        f_vs = np.array([found_discharge.value(i) for i in range(found_discharge.size())])
        f_ts = np.array([found_discharge.time(i) for i in range(found_discharge.size())])
        self.assertTrue(np.linalg.norm(t_ts - f_ts) < 1.0e-10)
        self.assertTrue(np.linalg.norm(t_vs - f_vs) < 1.0e-3)

    def test_pt_gs_k_calibration(self):
        self.run_calibration(pt_gs_k.PTGSKOptModel)

    def test_pt_ss_k_calibration(self):
        self.run_calibration(pt_ss_k.PTSSKOptModel)

    def test_pt_hs_k_calibration(self):
        self.run_calibration(pt_hs_k.PTHSKOptModel)

    def test_compute_lwc_percentiles(self):
        # Simulation time axis
        year, month, day, hour = 2010, 9, 1, 0
        dt = api.deltahours(24)
        n_steps = 400
        utc = api.Calendar()  # No offset gives Utc
        t0 = utc.time(api.YMDhms(year, month, day, hour))
        time_axis = api.TimeAxisFixedDeltaT(t0, dt, n_steps)

        # Some fake ids
        region_id = 0
        interpolation_id = 0

        # Simulation coordinate system
        epsg = "32633"

        # Model
        model_t = pt_gs_k.PTGSKModel

        # Configs and repositories
        dataset_config_file = path.join(path.dirname(__file__), "netcdf",
                                        "atnsjoen_datasets.yaml")
        region_config_file = path.join(path.dirname(__file__), "netcdf",
                                       "atnsjoen_calibration_region.yaml")
        region_config = RegionConfig(region_config_file)
        model_config = ModelConfig(self.model_config_file)
        dataset_config = YamlContent(dataset_config_file)
        region_model_repository = RegionModelRepository(region_config, model_config, model_t, epsg)
        interp_repos = InterpolationParameterRepository(model_config)
        netcdf_geo_ts_repos = []
        for source in dataset_config.sources:
            station_file = source["params"]["stations_met"]
            netcdf_geo_ts_repos.append(GeoTsRepository(source["params"], station_file, ""))
        geo_ts_repository = GeoTsRepositoryCollection(netcdf_geo_ts_repos)

        # Construct target discharge series
        simulator = DefaultSimulator(region_id, interpolation_id, region_model_repository,
                                    geo_ts_repository, interp_repos, None)
        n_cells = simulator.region_model.size()
        state_repos = DefaultStateRepository(model_t, n_cells)
        cid = 1
        simulator.region_model.set_state_collection(cid, True)
        simulator.run(time_axis, state_repos.get_state(0))
        self.assertAlmostEqual(simulator.region_model.cells[0].rc.pe_output.values[0], 0.039768354, 5) # just to verify pot.evap by regression, mm/h

        percentile_list = [10, 25, 50, 75, 90]
        # From here, things could be calculated without copies (except for 't')
        # TODO: Graham optimize with numba :-)
        #cells = simulator.region_model.get_cells()
        #lwcs = [np.array(cell.sc.gs_lwc.v) for cell in cells]  # Contiguous
        #t = np.array([cells[0].sc.gs_lwc.time(i) for i in range(cells[0].sc.gs_lwc.size())])
        #percentiles = np.percentile(np.array(lwcs), percentile_list, 0)
        # The next should be moved into an example directory
        # if 'DISPLAY' in environ.keys():
        #     from matplotlib import pylab as plt
        #     plot_np_percentiles(utc_to_greg(t), percentiles, base_color=(51/256, 102/256, 193/256))
        #     set_calendar_formatter(api.Calendar())
        #     plt.show()
        # else:
        #     print("DISPLAY not set, not showing plot")

    def test_snow_and_ground_water_response_calibration(self):
        """
        Test dual calibration strategy:
            * First fit the three Kirchner parameters for
              ground water response during July, August, and
              September.
            * Then fit two snow routine parameters (tx and max_water)
              from November to April.
        """
        # Simulation time axis
        dt = api.deltahours(24)
        n_steps = 400
        utc = api.Calendar()  # No offset gives Utc
        t0 = utc.time(2010, 9, 1, 0)
        time_axis = api.TimeAxisFixedDeltaT(t0, dt, n_steps)

        # Some fake ids
        region_id = 0
        interpolation_id = 0

        # Simulation coordinate system
        epsg = "32633"

        # Model
        model_t = pt_gs_k.PTGSKOptModel

        # Configs and repositories
        dataset_config_file = path.join(path.dirname(__file__), "netcdf",
                                        "atnsjoen_datasets.yaml")
        region_config_file = path.join(path.dirname(__file__), "netcdf",
                                       "atnsjoen_calibration_region.yaml")
        region_config = RegionConfig(region_config_file)
        model_config = ModelConfig(self.model_config_file)
        dataset_config = YamlContent(dataset_config_file)
        region_model_repository = RegionModelRepository(region_config, model_config, model_t, epsg)
        interp_repos = InterpolationParameterRepository(model_config)
        netcdf_geo_ts_repos = []
        for source in dataset_config.sources:
            station_file = source["params"]["stations_met"]
            netcdf_geo_ts_repos.append(GeoTsRepository(source["params"], station_file, ""))
        geo_ts_repository = GeoTsRepositoryCollection(netcdf_geo_ts_repos)

        # Construct target discharge series
        simulator = DefaultSimulator(region_id, interpolation_id, region_model_repository,
                                    geo_ts_repository, interp_repos, None)
        n_cells = simulator.region_model.size()
        state_repos = DefaultStateRepository(model_t, n_cells)
        simulator.run(time_axis, state_repos.get_state(0))
        cid = 1
        target_discharge = api.TsTransform().to_average(t0,dt,n_steps,simulator.region_model.statistics.discharge([cid]))

        # Construct kirchner parameters
        param = simulator.region_model.parameter_t(simulator.region_model.get_region_parameter())
        print_param("True solution", param)

        kirchner_param_min = simulator.region_model.parameter_t(param)
        kirchner_param_max = simulator.region_model.parameter_t(param)
        # Kichner parameters are quite abstract (no physical meaning), so simply scale them
        kirchner_param_min.kirchner.c1 *= 0.8
        kirchner_param_min.kirchner.c2 *= 0.8
        kirchner_param_min.kirchner.c3 *= 0.8
        kirchner_param_max.kirchner.c1 *= 1.2
        kirchner_param_max.kirchner.c2 *= 1.2
        kirchner_param_max.kirchner.c3 *= 1.2
        # kirchner_t_start = utc.time(api.YMDhms(2011, 4, 1, 0))
        # kirchner_time_axis = api.TimeAxisFixedDeltaT(kirchner_t_start, dt, 150)
        kirchner_time_axis = time_axis

        # Construct gamma snow parameters (realistic tx and max_lwc)
        gamma_snow_param_min = simulator.region_model.parameter_t(param)
        gamma_snow_param_max = simulator.region_model.parameter_t(param)
        gamma_snow_param_min.gs.tx = -1.0  # Min snow/rain temperature threshold
        gamma_snow_param_min.gs.max_water = 0.05  # Min 8% max water in snow in costal regions
        gamma_snow_param_max.gs.tx = 1.0
        gamma_snow_param_max.gs.max_water = 0.25  # Max 35% max water content, or we get too little melt
        gs_t_start = utc.time(2010, 11, 1, 0)
        gs_time_axis = api.TimeAxisFixedDeltaT(gs_t_start, dt, 250)
        # gs_time_axis = time_axis

        # Find parameters
        target_spec = api.TargetSpecificationPts(target_discharge, api.IntVector([cid]),
                                                 1.0, api.KLING_GUPTA)
        target_spec_vec = api.TargetSpecificationVector() # TODO: We currently dont fix list initializer for vectors
        target_spec_vec.append(target_spec)
        # Construct a fake, perturbed starting point for calibration
        p_vec = [param.get(i) for i in range(param.size())]
        for i, name in enumerate([param.get_name(i) for i in range(len(p_vec))]):
            if name not in ("c1" "c2", "c3", "TX", "max_water"):
                next
            if name in ("c1", "c2", "c3"):
                p_vec[i] = random.uniform(0.8*p_vec[i], 1.2*p_vec[i])
            elif name == "TX":
                p_vec[i] = random.uniform(gamma_snow_param_min.gs.tx, gamma_snow_param_max.gs.tx)
            elif name == "max_water":
                p_vec[i] = random.uniform(gamma_snow_param_min.gs.max_water, gamma_snow_param_max.gs.max_water)
        param.set(p_vec)
        print_param("Initial guess", param)
        # Two pass optimization, once for the ground water response, and second time for
        kirchner_p_opt = simulator.optimize(kirchner_time_axis, state_repos.get_state(0), target_spec_vec, param,
                                            kirchner_param_min, kirchner_param_max)
        gamma_snow_p_opt = simulator.optimize(gs_time_axis, state_repos.get_state(0), target_spec_vec, kirchner_p_opt,
                                              gamma_snow_param_min, gamma_snow_param_max)
        print_param("Half way result", kirchner_p_opt)
        print_param("Result", gamma_snow_p_opt)

        simulator.region_model.set_catchment_parameter(cid, gamma_snow_p_opt)
        simulator.run(time_axis, state_repos.get_state(0))
        found_discharge = simulator.region_model.statistics.discharge([cid])

        #t_vs = np.array(target_discharge.values)
        #t_ts = np.array([target_discharge.time(i) for i in range(target_discharge.size())])
        #f_vs = np.array(found_discharge.v)
        #f_ts = np.array([found_discharge.time(i) for i in range(found_discharge.size())])
        # The next should be moved into an example directory
        # Simple demo plotting that should be turned off during unit testing:
        # if 'DISPLAY' in environ.keys():
        #     from matplotlib import pylab as plt
        #     plt.plot(utc_to_greg(t_ts), t_vs)
        #     plt.hold(1)
        #     plt.plot(utc_to_greg(f_ts), f_vs)
        #     plt.ylabel("Discharge in $m^3s^{-1}$")
        #     plt.xlabel("Time in utc time axis dimension")
        #     set_calendar_formatter(api.Calendar())
        #     plt.legend(["Synthetic discharge", "Calibrated discharge"])
        #     plt.show()

    def test_run_arome_ensemble(self):
        # Simulation time axis
        utc = api.Calendar()  # No offset gives Utc
        t0 = utc.time( 2015, 7, 26, 0)
        n_hours = 30
        dt = api.deltahours(1)
        time_axis = api.TimeAxisFixedDeltaT(t0, dt, n_hours)

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
        region_model_repository = RegionModelRepository(region_config, model_config, model_t, epsg)
        interp_repos = InterpolationParameterRepository(model_config)
        base_dir = path.join(shyftdata_dir, "netcdf", "arome")
        pattern = "fc*.nc"
        try:
            geo_ts_repository = AromeDataRepository(epsg, base_dir, filename=pattern,
                                                    allow_subset=True)
        except Exception as e:
            print("**** test_run_arome_ensemble: Arome data missing or"
                  " wrong, test inconclusive ****")
            print("****{}****".format(e))
            self.skipTest("**** test_run_arome_ensemble: Arome data missing or wrong, test "
                          "inconclusive ****\n\t exception:{}".format(e))
        simulator = DefaultSimulator(region_id,
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
