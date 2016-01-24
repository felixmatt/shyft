from os import path
import unittest

from shyft import shyftdata_dir
from shyft.repository.default_state_repository import DefaultStateRepository
from shyft.orchestration.configuration import yaml_configs
from shyft.orchestration.simulators.config_simulator import ConfigSimulator

class ConfigSimulationTestCase(unittest.TestCase):

    def test_run_geo_ts_data_config_simulator(self):
        # set up configuration
        config_dir = path.join(path.dirname(__file__), "netcdf")
        cfg = yaml_configs.YAMLConfig(
            "neanidelva_simulation.yaml", "neanidelva",
            config_dir=config_dir, data_dir=shyftdata_dir)

        # get a simulator
        simulator = ConfigSimulator(cfg)

        n_cells = simulator.region_model.size()
        state_repos = DefaultStateRepository(cfg.model_t, n_cells)
        simulator.run(cfg.time_axis, state_repos.get_state(0))

if __name__ == '__main__':
    unittest.main()