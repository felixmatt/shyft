"""
Utilities for reading configurations for SHyFT simulations and calibrations.
"""

import os
from datetime import datetime

import yaml
import numpy as np

from shyft import api
from shyft.api import pt_gs_k
import shyft.orchestration


utc_calendar = api.Calendar()
"""Invariant global calendar in UTC."""


def utctime_from_datetime(dt):
    """Returns utctime of datetime dt (calendar interpreted as UTC)."""
    # Number of seconds since epoch
    nsec = np.array([dt], dtype="datetime64[s]").astype(np.long)[0]
    dt = datetime.fromtimestamp(nsec)
    return utc_calendar.time(api.YMDhms(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second))


class ConfigError(Exception):
    pass


class OrchestrationConfig(object):
    """
    Concrete class for yaml content.
    """

    def __init__(self, config_file, config_section, **kwargs):
        # The config_file needs to be an absolute path or have 'config_dir'
        if os.path.isabs(config_file):
            self._config_file = config_file
        elif "config_dir" in kwargs:
            self._config_file = os.path.join(kwargs["config_dir"], config_file)
        else:
            raise ConfigError(
                "'config_file' must be an absolute path "
                "or 'config_dir' passed as an argument")

        self._config_section = config_section

        # Load main configuration file
        with open(self._config_file) as cfg:
            config = yaml.load(cfg)[config_section]
        # Expose all keys in yaml file as attributes
        self.__dict__.update(config)
        # Override the parameters with kwargs
        self.__dict__.update(kwargs)

        # Check validity of some attributes
        if not hasattr(self, "config_dir"):
            raise ConfigError(
                "'config_dir' must be present in config section "
                "or passed as an argument")
        if not (os.path.isdir(self.config_dir) and
                os.path.isabs(self.config_dir)):
            raise ConfigError(
                "'config_dir' must exist and be an absolute path")
        if not hasattr(self, "data_dir"):
            raise ConfigError(
                "'data_dir' must be present in config section "
                "or passed as an argument")
        if not (os.path.isdir(self.data_dir) and
                os.path.isabs(self.data_dir)):
            raise ConfigError(
                "'data_dir' must exist and be an absolute path")

        # Create a time axis
        self.start_time = utctime_from_datetime(self.start_datetime)
        self.time_axis = api.Timeaxis(
            self.start_time, self.run_time_step, self.number_of_steps)
        # Get the region model in API
        self.model_t = getattr(pt_gs_k, self.model_t)

    def get_simulator(self):
        if not hasattr(self, "simulator"):
            raise ConfigError("Asking for a missing 'simulator' section.")
        # Get the flavor and the params for the simulator
        flavor = getattr(shyft.orchestration, self.simulator['flavor'])
        params = self.simulator['params']
        simulator = flavor.get_simulator(self, params)
        return simulator


    def __repr__(self):
        srepr = "%s::%s(" % (self.__class__.__name__, self._config_section)
        for key in self.__dict__:
            srepr += "%s=%r, " % (key, self.__dict__[key])
        srepr = srepr[:-2]
        return srepr + ")"
