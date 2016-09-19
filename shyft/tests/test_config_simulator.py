from os import path
import unittest

from shyft.repository.default_state_repository import DefaultStateRepository
from shyft.orchestration.configuration.yaml_configs import YAMLSimConfig
from shyft.orchestration.simulators.config_simulator import ConfigSimulator
from shyft.api import IntVector
cell_nc_file= path.join(path.dirname(__file__),"..","..","..","shyft-data","netcdf","orchestration-testdata","cell_data.nc")

class ConfigSimulationTestCase(unittest.TestCase):

    # @unittest.skipIf(not path.exists(cell_nc_file),"missing file {0} \n, test ignored ".format(cell_nc_file))
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
        cids = IntVector()
        discharge = simulator.region_model.statistics.discharge(cids)
        #  regression test on discharge values
        self.assertAlmostEqual(discharge.values[0],0.1961,3)
        self.assertAlmostEqual(discharge.values[3],2.7582,3)
        self.assertAlmostEqual(discharge.values[6400],58.9381,3)
        self.assertAlmostEqual(discharge.values[3578],5.5069,3)
        # regression test on geo fractions
        self.assertAlmostEqual(simulator.region_model.cells[0].geo.land_type_fractions_info().unspecified(),1.0,3)
        self.assertAlmostEqual(simulator.region_model.cells[2].geo.land_type_fractions_info().unspecified(),0.1433,3)
        self.assertAlmostEqual(simulator.region_model.cells[2].geo.land_type_fractions_info().forest(),0.0,3)
        self.assertAlmostEqual(simulator.region_model.cells[2].geo.land_type_fractions_info().reservoir(),0.8566,3)
        self.assertAlmostEqual(simulator.region_model.cells[3383].geo.land_type_fractions_info().lake(),0.7432,3)
        self.assertAlmostEqual(simulator.region_model.cells[652].geo.land_type_fractions_info().glacier(),0.1351,3)


if __name__ == '__main__':
    unittest.main()