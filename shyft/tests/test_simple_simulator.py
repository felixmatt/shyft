"""
Tests for the simple simulator.
"""

from __future__ import print_function
from __future__ import absolute_import
from os import path

import unittest
from shyft.repository.netcdf import RegionModelRepository
from shyft.repository.netcdf import AromeDataRepository
from shyft.repository.yaml_config import RegionConfig
from shyft.repository.yaml_config import ModelConfig
from shyft.api import pt_gs_k
from shyft import api
from shyft.orchestration.simulator import SimpleSimulator
from shyft import shyftdata_dir


class SimulationTestCase(unittest.TestCase):

    def setUp(self):

        self.region_config_file = path.join(path.dirname(__file__), "netcdf", "region.yaml")
        self.model_config_file = path.join(path.dirname(__file__), "netcdf", "model.yaml")

    def test_construct_simulator(self):
        region_config = RegionConfig(self.region_config_file)
        model_config = ModelConfig(self.model_config_file)
        print(region_config.__dict__)
        print(model_config.__dict__)
        region_model_repository = RegionModelRepository(region_config, model_config)
        simulator = SimpleSimulator(pt_gs_k.PTGSKModel, 0, region_model_repository, None, None)
        self.assertIsNotNone(simulator.region_model)
        EPSGID = region_config.domain()["EPSG"]

        year, month, day, hour = 2015, 8, 23, 6
        n_hours = 30
        date_str = "{}{:02}{:02}_{:02}".format(year, month, day, hour)
        utc = api.Calendar()  # No offset gives Utc
        t0 = api.YMDhms(year, month, day, hour)
        period = api.UtcPeriod(utc.time(t0), utc.time(t0) + api.deltahours(n_hours))

        base_dir = path.join(shyftdata_dir, "repository", "arome_data_repository")
        f1 = "arome_metcoop_red_default2_5km_{}.nc".format(date_str)
        f2 = "arome_metcoop_red_test2_5km_{}.nc".format(date_str)

        #bbox = ([upper_left_x, upper_left_x + nx*dx, upper_left_x + nx*dx, upper_left_x],
        #        [upper_left_y, upper_left_y, upper_left_y - ny*dy, upper_left_y - ny*dy])
        #ar1 = AromeDataRepository(EPSG, period, base_dir, filename=f1, bounding_box=bbox)
        #ar2 = AromeDataRepository(EPSG, period, base_dir, filename=f2, elevation_file=f1)
        #ar1_data_names = ("temperature", "wind_speed", "precipitation", "relative_humidity")
        #ar2_data_names = ("radiation",)
        #sources = ar1.get_timeseries(ar1_data_names, period, None)
        #self.assertTrue(len(sources) > 0)
        #sources2 = ar2.get_timeseries(ar2_data_names, period, geo_location_criteria=bbox)

        t_start = 0
        #time_axis = api.Timeaxis(t_start, delta_t, n_steps)
