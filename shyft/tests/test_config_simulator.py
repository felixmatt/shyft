from os import path
import unittest

from shyft.repository.default_state_repository import DefaultStateRepository
from shyft.orchestration.configuration.yaml_configs import YAMLSimConfig
from shyft.orchestration.simulators.config_simulator import ConfigSimulator

class ConfigSimulationTestCase(unittest.TestCase):

    def test_run_geo_ts_data_config_simulator(self):
        # set up configuration
        config_dir = path.join(path.dirname(__file__), "netcdf")
        config_file = path.join(config_dir,"neanidelva_simulation.yaml")
        config_section = "neanidelva"
        cfg = YAMLSimConfig(config_file, config_section)

        # get a simulator
        simulator = ConfigSimulator(cfg)

        n_cells = simulator.region_model.size()
        state_repos = DefaultStateRepository(simulator.region_model.__class__, n_cells)
        simulator.run(cfg.time_axis, state_repos.get_state(0))

if __name__ == '__main__':
    unittest.main()