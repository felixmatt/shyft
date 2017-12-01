from os import path
import unittest

from shyft.repository.default_state_repository import DefaultStateRepository
from shyft.orchestration.configuration.yaml_configs import YAMLSimConfig
from shyft.orchestration.simulators.config_simulator import ConfigSimulator
from shyft.api import IntVector, Calendar
from shyft.orchestration.config import utctime_from_datetime
import datetime as dt


class ConfigSimulationTestCase(unittest.TestCase):
    def test_utctime_from_datetime(self):
        utc = Calendar()
        dt1 = dt.datetime(2015, 6, 1, 2, 3, 4)
        t1 = utctime_from_datetime(dt1)
        self.assertEqual(t1, utc.time(2015, 6, 1, 2, 3, 4))

    def test_run_geo_ts_data_config_simulator(self):
        # These config files are versioned in shyft git
        config_dir = path.join(path.dirname(__file__), "netcdf")
        config_file = path.join(config_dir, "neanidelva_simulation.yaml")
        config_section = "neanidelva"
        cfg = YAMLSimConfig(config_file, config_section, overrides={'config': {'number_of_steps': 168}})

        # These config files are versioned in shyft-data git. Read from ${SHYFTDATA}/netcdf/orchestration-testdata/
        # TODO: Put all config files needed to run this test under the same versioning system (shyft git)
        simulator = ConfigSimulator(cfg)
        n_cells = simulator.region_model.size()
        state_repos = DefaultStateRepository(simulator.region_model.__class__, n_cells)
        simulator.run(cfg.time_axis, state_repos.get_state(0))
        cids = IntVector()
        discharge = simulator.region_model.statistics.discharge(cids)

        # Regression tests on discharge values
        self.assertAlmostEqual(discharge.values[0],  0.0957723, 3)
        self.assertAlmostEqual(discharge.values[3], 3.9098, 3)  #
        # x self.assertAlmostEqual(discharge.values[6400], 58.8385, 3) # was 58.9381,3 before glacier&fractions adjustments
        # x self.assertAlmostEqual(discharge.values[3578],5.5069,3)
        # glacier_melt, not much, but enough to test
        # x self.assertAlmostEqual(simulator.region_model.gamma_snow_response.glacier_melt(cids).values.to_numpy().max(),0.201625547258,4)
        self.assertAlmostEqual(simulator.region_model.gamma_snow_response.glacier_melt(cids).values.to_numpy().max(), 0.12393672891230645, 4)
        # Regression tests on geo fractions
        self.assertAlmostEqual(simulator.region_model.cells[0].geo.land_type_fractions_info().unspecified(), 1.0, 3)
        self.assertAlmostEqual(simulator.region_model.cells[2].geo.land_type_fractions_info().unspecified(), 0.1433, 3)
        self.assertAlmostEqual(simulator.region_model.cells[2].geo.land_type_fractions_info().forest(), 0.0, 3)
        self.assertAlmostEqual(simulator.region_model.cells[2].geo.land_type_fractions_info().reservoir(), 0.8566, 3)
        # x self.assertAlmostEqual(simulator.region_model.cells[3383].geo.land_type_fractions_info().lake(),0.7432,3)
        # x self.assertAlmostEqual(simulator.region_model.cells[652].geo.land_type_fractions_info().glacier(),0.1351,3)


if __name__ == '__main__':
    unittest.main()
