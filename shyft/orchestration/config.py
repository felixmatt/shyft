"""
Utilities for reading YAML configurations for SHyFT simulations.
"""

import os
from datetime import datetime

import yaml
import numpy as np

from shyft import api
from shyft.api import pt_gs_k, pt_ss_k, pt_hs_k, hbv_stack
from shyft.repository.netcdf import (
    RegionModelRepository, GeoTsRepository, get_geo_ts_collection, yaml_config)
from shyft.repository.interpolation_parameter_repository import (
    InterpolationParameterRepository)
from .simulator import DefaultSimulator


utc_calendar = api.Calendar()
"""Invariant global calendar in UTC."""


def utctime_from_datetime(dt):
    """Returns utctime of datetime dt (calendar interpreted as UTC)."""
    # Number of seconds since epoch
    return utc_calendar.time(dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second)


class ConfigError(Exception):
    pass


class YAMLConfig(object):

    def __init__(self, config_file, config_section, **kwargs):
        """
        Setup a config instance for a netcdf orchestration from a YAML file.

        Parameters
        ----------
        config_file : string
          Path to the YAML configuration file
        config_section : string
          Section in YAML file for simulation parameters.

        Returns
        -------
        YAMLConfig instance
        """
        # The config_file needs to be an absolute path or have 'config_dir'
        if os.path.isabs(config_file):
            self._config_file = config_file
            self.config_dir = os.path.dirname(config_file)
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
        # Get the region model in API (already an object if in kwargs)
        if 'model_t' not in kwargs:
            module, model_t = self.model_t.split(".")
            self.model_t = getattr(globals()[module], model_t)

    def get_simulator(self):
        """
        Return a DefaultSimulator based on `cfg`.

        Returns
        -------
        DefaultSimulator instance
        """
        # Read region, model and datasets config files
        region_config_file = os.path.join(
            self.config_dir, self.region_config_file)
        region_config = yaml_config.RegionConfig(region_config_file)
        model_config_file = os.path.join(
            self.config_dir, self.model_config_file)
        model_config = yaml_config.ModelConfig(model_config_file)
        datasets_config_file = os.path.join(
            self.config_dir, self.datasets_config_file)
        datasets_config = yaml_config.YamlContent(datasets_config_file)

        # Build some interesting constructs
        region_model = RegionModelRepository(
            region_config, model_config, self.model_t, self.epsg)
        interp_repos = InterpolationParameterRepository(model_config)
        geo_ts = get_geo_ts_collection(datasets_config, self.data_dir)

        # If region and interpolation ids are not present, just use fake ones
        region_id = 0 if not hasattr(self, "region_id") else int(self.region_id)
        interpolation_id = 0 if not hasattr(self, "interpolation_id") \
                           else int(self.interpolation_id)
        # set up the simulator
        simulator = DefaultSimulator(region_id, interpolation_id, region_model,
                                     geo_ts, interp_repos, None)
        return simulator

    def __repr__(self):
        srepr = "%s::%s(" % (self.__class__.__name__, self._config_section)
        for key in self.__dict__:
            srepr += "%s=%r, " % (key, self.__dict__[key])
        srepr = srepr[:-2]
        return srepr + ")"
