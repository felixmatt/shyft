# -*- coding: utf-8 -*-
import os
import yaml
#import numpy as np
from datetime import datetime

from shyft import api
#from shyft.api import pt_gs_k, pt_ss_k, pt_hs_k
from shyft.repository.interpolation_parameter_repository import (
    InterpolationParameterRepository)
from shyft.repository import geo_ts_repository_collection
from .yaml_constructors import (region_model_repo_constructor, geo_ts_repo_constructor, target_repo_constructor)
from . import config_interfaces

def utctime_from_datetime(dt):
    utc_calendar = api.Calendar()
    return utc_calendar.time(api.YMDhms(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second))


class YamlContent(object):
    """
    Concrete class for yaml content.
    """

    def __init__(self, config_file):
        self._config_file = config_file
        with open(config_file,encoding='utf8') as cfg_file:
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

    def __init__(self, config_file, overrides=None):
        self._config = YamlContent(config_file)
        if overrides is not None:
            self._config.__dict__.update(overrides)

    def model_parameters(self):
        return self._config.model_parameters

    def model_type(self):
        #module, model_t = self._config.model_t.split(".")
        #return getattr(globals()[module], model_t)
        return self._config.model_t


class InterpolationConfig(object):
    """
    Yaml based model configuration, using a YamlContent instance
    for holding the content.
    """

    def __init__(self, config_file):
        self._config = YamlContent(config_file)

    def interpolation_parameters(self):
        return self._config.interpolation_parameters


class ConfigError(Exception):
    pass


class YAMLSimConfig(object):

    def __init__(self, config_file, config_section, overrides=None):
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
        if overrides is None:
            overrides = {}
        # The config_file needs to be an absolute path
        if os.path.isabs(config_file):
            self._config_file = config_file
            self.config_dir = os.path.dirname(config_file)
        else:
            raise ConfigError(
                "'config_file' must be an absolute path ")

        self._config_section = config_section

        # Load main configuration file
        with open(self._config_file,encoding='utf8') as cfg:
            config = yaml.load(cfg)[config_section]
        # Expose all keys in yaml file as attributes
        self.__dict__.update(config)
        # Override the parameters with kwargs
        #self.__dict__.update(kwargs)
        self.__dict__.update(overrides.get("config", {}))


        self.validate()

        # Create a time axis
        # It is assumed that the time specified in the config file is in UTC
        self.start_time = utctime_from_datetime(self.start_datetime)
        self.time_axis = api.Timeaxis(
            self.start_time, self.run_time_step, self.number_of_steps)
        # Get the region model in API (already an object if in kwargs)
        #if 'model_t' not in kwargs:
        #    module, model_t = self.model_t.split(".")
        #    self.model_t = getattr(globals()[module], model_t)

        # If region and interpolation ids are not present, just use fake ones
        # self.region_id = 0 if not hasattr(self, "region_id") else int(self.region_id)
        self.region_model_id = str(self.region_model_id)
        self.interpolation_id = 0 if not hasattr(self, "interpolation_id") \
                           else int(self.interpolation_id)
        self.initial_state_repo = None
        self.end_state_repo = None

        self.construct_repos(overrides)

    def validate(self):
        """Check for the existence of mandatory fields."""
        assert hasattr(self, "region_config_file")
        assert hasattr(self, "model_config_file")
        assert hasattr(self, "datasets_config_file")
        assert hasattr(self, "interpolation_config_file")
        assert hasattr(self, "start_datetime")
        assert hasattr(self, "run_time_step")
        assert hasattr(self, "number_of_steps")
        assert hasattr(self, "region_model_id")

    def construct_repos(self, overrides):
        """
        Construct repositories
        """
        # Read region, model and datasets config files
        region_config_file = os.path.join(
            self.config_dir, self.region_config_file)
        self.region_config = RegionConfig(region_config_file)

        self.model_config_file = os.path.join(
            self.config_dir, self.model_config_file)
        model_config = ModelConfig(self.model_config_file, overrides=overrides.get("model", {}))

        datasets_config_file = os.path.join(
            self.config_dir, self.datasets_config_file)
        datasets_config = YamlContent(datasets_config_file)

        interpolation_config_file = os.path.join(
            self.config_dir, self.interpolation_config_file)
        interpolation_config = InterpolationConfig(interpolation_config_file)

        # Construct RegionModelRepository
        self.region_model = region_model_repo_constructor(self.region_config.repository()['class'],
            self.region_config, model_config, self.region_model_id)
        # Construct InterpolationParameterRepository
        self.interp_repos = InterpolationParameterRepository(interpolation_config)
        # Construct GeoTsRepository
        geo_ts_repos = []
        src_types_to_extract = []
        for source in datasets_config.sources:
            geo_ts_repos.append(geo_ts_repo_constructor(source['repository'], source['params'], self.region_config))
            src_types_to_extract.append(source['types'])
        self.geo_ts = geo_ts_repository_collection.GeoTsRepositoryCollection(geo_ts_repos,
                                                                             src_types_per_repo = src_types_to_extract)
        # Construct destination repository
        self.dst_repo = []
        if hasattr(datasets_config, 'destinations'):
            for repo in datasets_config.destinations:
                repo['repository'] = target_repo_constructor(repo['repository'],repo['params'])
                #[dst['time_axis'].update({'start_datetime': utctime_from_datetime(dst['time_axis']['start_datetime'])})
                # for dst in repo['1D_timeseries'] if dst['time_axis'] is not None]
                [dst.update({'time_axis':self.time_axis}) if dst['time_axis'] is None
                 else dst.update({'time_axis':api.Timeaxis(utctime_from_datetime(dst['time_axis']['start_datetime']),
                                                           dst['time_axis']['time_step_length'],
                                                           dst['time_axis']['number_of_steps'])}) for dst in repo['1D_timeseries']]
                self.dst_repo.append(repo)

        # Construct StateRepository
        if hasattr(self, 'initial_state'):
            self.initial_state_repo = self.initial_state['repository']['class'](
                **self.initial_state['repository']['params'])
        if hasattr(self, 'end_state'):
            self.end_state_repo = self.end_state['repository']['class'](
                **self.end_state['repository']['params'])

    def __repr__(self):
        srepr = "%s::%s(" % (self.__class__.__name__, self._config_section)
        for key in self.__dict__:
            srepr += "%s=%r, " % (key, self.__dict__[key])
        srepr = srepr[:-2]
        return srepr + ")"

class YAMLCalibConfig(object):

    def __init__(self, config_file, config_section):
        self._config_file = config_file
        config = yaml.load(open(config_file,encoding='utf8'))[config_section]
        self.__dict__.update(config)

        self.validate()


        # Get the location of the model_config_file relative to the calibration config file
        if not os.path.isabs(self.model_config_file):
            model_config_file = os.path.join(
                os.path.dirname(os.path.abspath(config_file)), self.model_config_file)
        # Create a new sim_config attribute
        self.sim_config = YAMLSimConfig(
            model_config_file, config_section, overrides=getattr(self, "overrides", None))
        # Get the location of the calibrated_model_file relative to the calibration config file
        if hasattr(self, 'calibrated_model_file'):
            if not os.path.isabs(self.calibrated_model_file):
                self.calibrated_model_file = os.path.join(
                    os.path.dirname(os.path.abspath(config_file)), self.calibrated_model_file)

        self.target_ts = []

        self._fetch_target_timeseries()

    def validate(self):
        """Check for the existence of mandatory fields."""
        assert hasattr(self, "model_config_file")
        assert hasattr(self, "optimization_method")
        assert hasattr(self, "calibration_parameters")
        assert hasattr(self, "target")

    def _fetch_target_timeseries(self):
        for repository in self.target:
            ts_repository = target_repo_constructor(repository['repository'],repository['params'])
            for target in repository['1D_timeseries']:
                target['start_datetime'] = utctime_from_datetime(target['start_datetime'])
                period = api.UtcPeriod(target['start_datetime'],
                                       target['start_datetime'] + target['number_of_steps'] * target['run_time_step'])
                target.update({'ts':ts_repository.read([target['uid']],period)[target['uid']]})
                self.target_ts.append(target)


class YAMLForecastConfig(object):

    def __init__(self, config_file, config_section, forecast_names, forecast_time=None, overrides=None):
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
        if overrides is None:
            overrides = {}
        # The config_file needs to be an absolute path
        if os.path.isabs(config_file):
            self._config_file = config_file
            self.config_dir = os.path.dirname(config_file)
        else:
            raise ConfigError(
                "'config_file' must be an absolute path ")

        self._config_section = config_section

        self.sim_config = YAMLSimConfig(self._config_file, self._config_section)
        if forecast_time is None:
            self.forecast_time = self.sim_config.time_axis.total_period().end
        else:
            self.forecast_time = forecast_time
            self.sim_config.start_time = self.forecast_time - self.sim_config.number_of_steps * self.sim_config.run_time_step
            self.sim_config.time_axis = api.Timeaxis(self.sim_config.start_time,
                                                     self.sim_config.run_time_step, self.sim_config.number_of_steps)

        #self.region_config = self.sim_config.region_config

        self.forecast_names = forecast_names

        # Load main configuration file
        with open(self._config_file,encoding='utf8') as cfg:
            configs = yaml.load(cfg)[self._config_section]['forecast_runs']
        for name in self.forecast_names:
            assert name in configs

        fc0 = self.forecast_names[0]
        configs[fc0].update({'start_datetime': datetime.utcfromtimestamp(self.forecast_time)})
        fc_time = self.forecast_time
        for i in range(1,len(self.forecast_names)):
            fc_1 = self.forecast_names[i-1]
            fc_time += configs[fc_1]['number_of_steps']*configs[fc_1]['run_time_step']
            configs[self.forecast_names[i]].update({'start_datetime': datetime.utcfromtimestamp(fc_time)})

        self.forecast_config = {name: YAMLSimConfig(self._config_file, self._config_section, overrides={'config':configs[name]})
                                for name in self.forecast_names}
