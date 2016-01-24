import os
from datetime import datetime

import yaml
import numpy as np

from shyft import api
from shyft.api import pt_gs_k, pt_ss_k, pt_hs_k
from shyft.repository.interpolation_parameter_repository import (
    InterpolationParameterRepository)
from shyft.repository import geo_ts_repository_collection
from .yaml_constructors import (r_m_repo_constructors, geo_ts_repo_constructors)
from . import config_interfaces


class YamlContent(object):
    """
    Concrete class for yaml content.
    """

    def __init__(self, config_file):
        self._config_file = config_file
        with open(config_file) as cfg_file:
            config = yaml.load(cfg_file)
        # Expose all keys in yaml file as attributes
        self.__dict__.update(config)

    def __repr__(self):
        srepr = "%s(" % self.__class__.__name__
        for key in self.__dict__:
            srepr += "%s=%r, " % (key, self.__dict__[key])
        srepr = srepr[:-2]
        return srepr + ")"


class RegionConfig(config_interfaces.RegionConfig):
    """
    Yaml based region configuration, using a YamlContent instance
    for holding the content.
    """

    def __init__(self, config_file):
        self._config = YamlContent(config_file)

    def parameter_overrides(self):
        return getattr(self._config, "parameter_overrides", {})

    def domain(self):
        return self._config.domain

    def repository(self):
        return self._config.repository
        
    def catchments(self):
        return getattr(self._config, "catchment_indices", None)


class ModelConfig(config_interfaces.ModelConfig):
    """
    Yaml based model configuration, using a YamlContent instance
    for holding the content.
    """

    def __init__(self, config_file):
        self._config = YamlContent(config_file)

    def interpolation_parameters(self):
        return self._config.parameters["interpolation"]

    def model_parameters(self):
        return self._config.parameters["model"]

    def model_type(self):
        module, model_t = self._config.model_t.split(".")
        return getattr(globals()[module], model_t)

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
        #self.start_time = utctime_from_datetime(self.start_datetime)
        # It is assumed that the time specified in the config file is in UTC
        dt = self.start_datetime
        self.start_time = utc_calendar.time(api.YMDhms(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second))
        self.time_axis = api.Timeaxis(
            self.start_time, self.run_time_step, self.number_of_steps)
        # Get the region model in API (already an object if in kwargs)
        if 'model_t' not in kwargs:
            module, model_t = self.model_t.split(".")
            self.model_t = getattr(globals()[module], model_t)

        self.construct_repos()

    def construct_repos(self):
        """
        Construct repositories
        """
        # Read region, model and datasets config files
        region_config_file = os.path.join(
            self.config_dir, self.region_config_file)
        region_config = RegionConfig(region_config_file)

        model_config_file = os.path.join(
            self.config_dir, self.model_config_file)
        model_config = ModelConfig(model_config_file)

        datasets_config_file = os.path.join(
            self.config_dir, self.datasets_config_file)
        datasets_config = YamlContent(datasets_config_file)

        # Construct RegionModelRepository
        self.region_model = r_m_repo_constructors[region_config.repository()['name']](
            region_config, model_config)
        # Construct InterpolationParameterRepository
        self.interp_repos = InterpolationParameterRepository(model_config)
        # Construct GeoTsRepository
        geo_ts_repos = []
        for source in datasets_config.sources:
            geo_ts_repos.append(geo_ts_repo_constructors[source['repository']](source['params'],region_config))
        self.geo_ts = geo_ts_repository_collection.GeoTsRepositoryCollection(geo_ts_repos)

        # If region and interpolation ids are not present, just use fake ones
        self.region_id = 0 if not hasattr(self, "region_id") else int(self.region_id)
        self.interpolation_id = 0 if not hasattr(self, "interpolation_id") \
                           else int(self.interpolation_id)

    def __repr__(self):
        srepr = "%s::%s(" % (self.__class__.__name__, self._config_section)
        for key in self.__dict__:
            srepr += "%s=%r, " % (key, self.__dict__[key])
        srepr = srepr[:-2]
        return srepr + ")"