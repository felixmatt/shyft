"""
Tests for the netcdf datasets.
"""

from __future__ import print_function
from __future__ import absolute_import

import os

import unittest
import numpy as np

from shyft import api
from shyft.orchestration2 import config_constructor, cell_extractor, CalibrationConfig
from shyft.orchestration2.shyft_runner import Simulator, Calibrator


class Simulation():
    def setUp(self):
        # Get the configuration section
        config_file, section = self.config_file, self.section
        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "netcdf/%s" % config_file)
        config = config_constructor(config_file, section)

        # Build the simulator
        self.simulator = simulator = Simulator(config)
        time_axis = api.Timeaxis(config.start_time, config.run_time_step, config.number_of_steps)
        simulator.build_model(time_axis.start(), time_axis.delta(), time_axis.size())
        simulator.run_model()

    def test_simulation(self):
        cells = self.simulator.model.get_cells()
        assert len(cells) == 4
        expected_results = {
            "total_discharge": [0.00697, 0.00706, 0.00705, 0.00705],
            "discharge": [0.00364, 0.00362, 0.00353, 0.00353],
            "snow_storage": [167.8, 151.4, 147.9, 148.3],
            "temperature": [-1.66, -1.22, -0.702, -0.702],
            "precipitation": [0.0753, 0.0741, 0.0726, 0.0726],
        }

        # for the fun of it, demonstrate how to use cell_statistics
        cids = api.IntVector()
        temperature = self.simulator.model.statistics.temperature(cids)
        precipitation = self.simulator.model.statistics.precipitation(cids)
        discharge = self.simulator.model.statistics.discharge(cids)
        assert discharge.size() > 0
        assert temperature.size() > 0
        assert precipitation.size() > 0
        assert discharge.size() > 0

        for i, cell in enumerate(cells):
            for param in expected_results:
                value = cell_extractor[param](cell)
                if type(value) is np.ndarray:
                    assert len(np.where(value != value)[0]) == 0
                    # Take the mean value as a gross estimator
                    value = value.sum() / len(value)
                #print("param, cell, value:", param, i, value)
                np.testing.assert_allclose(np.float64(value), np.float64(expected_results[param][i]), rtol=1e-2)


# Some examples of simulation.  Feel free to add more.
class Simulation1(Simulation, unittest.TestCase):
    config_file = "configuration.yaml"
    section = "Atnsjoen"


class Calibration():
    def setUp(self):
        # Get the configuration section
        config_file, section = self.config_file, self.section
        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "netcdf/%s" % config_file)
        config = CalibrationConfig(config_file, section)

        # Build the calibrator
        self.time_axis = api.Timeaxis(config.model_config.start_time, config.model_config.run_time_step,
                                      config.model_config.number_of_steps)
        self.calibrator = Calibrator(config)

    def test_calibration(self):
        self.calibrator.init(self.time_axis)
        calibr_results = self.calibrator.calibrate(tol=1.0e-5)
        # print("calibrated results:", calibr_results)
        expected_results = {
            'wind_const': 1.0, 'max_albedo': 0.9, 'p_corr_scale_factor': 1.0, 'fast_albedo_decay_rate': 10.0,
            'TX': -0.5, 'glacier_albedo': 0.4, 'surface_magnitude': 30.0, 'snowfall_reset_depth': 5.0,
            'wind_scale': 3.5, 'slow_albedo_decay_rate': 30.0, 'ae_scale_factor': 1.5, 'c3': -0.1,
            'c2': 1.0, 'c1': -0.5, 'snow_cv': 0.4,
            'min_albedo': 0.6, 'max_water': 0.1}
        # Check that the results are as expected
        for key, val in calibr_results.iteritems():
            np.testing.assert_allclose(np.float64(val), np.float64(expected_results[key]), rtol=1e-2)

    def test_calibration_dream(self):
        # Mock the calibration configuration to use the dream optimizer
        self.calibrator._config.calibration_type = "PTGSKOptimizer.optimize_dream"
        self.calibrator.init(self.time_axis)
        calibr_results = self.calibrator.calibrate(tol=1.0e-5)
        # print("calibrated results:", calibr_results)
        expected_results = {'wind_const': 1.0, 'max_albedo': 0.9, 'p_corr_scale_factor': 1.0,
                            'fast_albedo_decay_rate': 10.194, 'TX': 0.394, 'glacier_albedo': 0.4,
                            'surface_magnitude': 30.0, 'snowfall_reset_depth': 5.0, 'wind_scale': 5.67,
                            'slow_albedo_decay_rate': 20.69, 'ae_scale_factor': 1.5, 'c3': -0.128, 'c2': 0.983,
                            'c1': -2.34, 'snow_cv': 0.4, 'min_albedo': 0.6, 'max_water': 0.1}
        # Check that the results are as expected
        for key, val in calibr_results.iteritems():
            np.testing.assert_allclose(np.float64(val), np.float64(expected_results[key]), rtol=1e-2)


# Some examples of calibration.  Feel free to add more.
class Calibration1(Calibration, unittest.TestCase):
    config_file = "calibration.yaml"
    section = "Atnsjoen"
