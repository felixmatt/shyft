# -*- coding: utf-8 -*-
import os
import yaml
#import numpy as np
from datetime import datetime
import copy

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

def geots_repo_from_config(datasets_config_file, epsg=None):
    datasets_config = YamlContent(datasets_config_file)
    # Construct GeoTsRepository
    geo_ts_repos = []
    src_types_to_extract = []
    for source in datasets_config.sources:
        if epsg is not None:
            source['params'].update({'epsg': epsg})
        # geo_ts_repos.append(geo_ts_repo_constructor(source['repository'], source['params'], self.region_config))
        geo_ts_repos.append(geo_ts_repo_constructor(source['repository'], source['params']))
        src_types_to_extract.append(source['types'])
    return geo_ts_repository_collection.GeoTsRepositoryCollection(geo_ts_repos,
                                                                         src_types_per_repo=src_types_to_extract)


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
        #self.start_time = utctime_from_datetime(self.start_datetime)
        #self.time_axis = api.Timeaxis(
        #    self.start_time, self.run_time_step, self.number_of_steps)

        # If region and interpolation ids are not present, just use fake ones
        self.region_model_id = str(self.region_model_id)
        self.interpolation_id = 0 if not hasattr(self, "interpolation_id") \
                           else int(self.interpolation_id)
        #self.initial_state_repo = None
        #self.end_state_repo = None

        self.construct_configs(overrides)

    @staticmethod
    def construct_geots_repo(datasets_config, epsg=None):
        #datasets_config = YamlContent(datasets_config_file)
        # Construct GeoTsRepository
        geo_ts_repos = []
        src_types_to_extract = []
        for source in datasets_config.sources:
            if epsg is not None:
                source['params'].update({'epsg': epsg})
            # geo_ts_repos.append(geo_ts_repo_constructor(source['repository'], source['params'], self.region_config))
            geo_ts_repos.append(geo_ts_repo_constructor(source['repository'], source['params']))
            src_types_to_extract.append(source['types'])
        return geo_ts_repository_collection.GeoTsRepositoryCollection(geo_ts_repos,
                                                                      src_types_per_repo=src_types_to_extract)

    @staticmethod
    def construct_region_model_repo(region_config, model_config, region_model_id):
        #region_config = RegionConfig(self.region_config_file)
        return region_model_repo_constructor(region_config.repository()['class'],
            region_config, model_config, region_model_id)

    @staticmethod
    def construct_interp_repo(interp_config):
        return InterpolationParameterRepository(interp_config)

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

    @property
    def time_axis(self):
        return api.Timeaxis(utctime_from_datetime(self.start_datetime), self.run_time_step, self.number_of_steps)

    def get_geots_repo(self):
        return self.construct_geots_repo(self.datasets_config, self.region_config.domain()["EPSG"])

    def get_region_model_repo(self):
        return self.construct_region_model_repo(self.region_config, self.model_config, self.region_model_id)

    def get_interp_repo(self):
        return self.construct_interp_repo(self.interpolation_config)

    def get_destination_repo(self):
        if not hasattr(self.datasets_config, 'destinations'):
            return []
        dst_repo = [{'repository': target_repo_constructor(repo['repository'], repo['params']), '1D_timeseries': [dst for dst in repo['1D_timeseries']]} for repo in self.datasets_config.destinations]
        [dst.update({'time_axis': self.time_axis}) if dst['time_axis'] is None
         else dst.update({'time_axis': api.Timeaxis(utctime_from_datetime(dst['time_axis']['start_datetime']),
                                                    dst['time_axis']['time_step_length'],
                                                    dst['time_axis']['number_of_steps'])}) for repo in dst_repo for dst in repo['1D_timeseries']]
        return dst_repo

    def get_reference_repo(self):
        if not hasattr(self, 'references'):
            return []
        return [{'repository': target_repo_constructor(repo['repository'], repo['params']), '1D_timeseries': [ref for ref in repo['1D_timeseries']]} for repo in self.references]

    def get_initial_state_repo(self):
        if hasattr(self, 'initial_state'):
            return self.initial_state['repository']['class'](**self.initial_state['repository']['params'])

    def get_end_state_repo(self):
        if hasattr(self, 'end_state'):
            return self.end_state['repository']['class'](**self.end_state['repository']['params'])

    def construct_configs(self, overrides):
        # Read region, model and datasets config files
        self.region_config_file = os.path.join(
            self.config_dir, self.region_config_file)
        self.region_config = RegionConfig(self.region_config_file)

        self.model_config_file = os.path.join(
            self.config_dir, self.model_config_file)
        self.model_config = ModelConfig(self.model_config_file, overrides=overrides.get("model", {}))

        self.datasets_config_file = os.path.join(
            self.config_dir, self.datasets_config_file)
        self.datasets_config = YamlContent(self.datasets_config_file)

        self.interpolation_config_file = os.path.join(
            self.config_dir, self.interpolation_config_file)
        self.interpolation_config = InterpolationConfig(interpolation_config_file)

        # Construct RegionModelRepository
        #self.region_model_repo = region_model_repo_constructor(self.region_config.repository()['class'],
        #    self.region_config, self.model_config, self.region_model_id)

        # Construct InterpolationParameterRepository
        #self.interp_repos = InterpolationParameterRepository(self.interpolation_config)

        # Construct GeoTsRepository
        #self.geo_ts = geots_repo_from_config(self.datasets_config_file, self.region_config.domain()["EPSG"])

        # Construct destination repository
        # self.dst_repo = []
        # if hasattr(self.datasets_config, 'destinations'):
        #     for repo in self.datasets_config.destinations:
        #         repo['repository'] = target_repo_constructor(repo['repository'],repo['params'])
        #         #[dst['time_axis'].update({'start_datetime': utctime_from_datetime(dst['time_axis']['start_datetime'])})
        #         # for dst in repo['1D_timeseries'] if dst['time_axis'] is not None]
        #         [dst.update({'time_axis':self.time_axis}) if dst['time_axis'] is None
        #          else dst.update({'time_axis':api.Timeaxis(utctime_from_datetime(dst['time_axis']['start_datetime']),
        #                                                    dst['time_axis']['time_step_length'],
        #                                                    dst['time_axis']['number_of_steps'])}) for dst in repo['1D_timeseries']]
        #         self.dst_repo.append(repo)

        # Construct reference data repository
        # self.ref_repo = []
        # if hasattr(self, 'references'):
        #     for repo in self.references:
        #         repo_ = target_repo_constructor(repo['repository'],repo['params'])
        #         [dst.update({'repo': repo_}) for dst in repo['1D_timeseries']]
        #         self.ref_repo.extend(repo['1D_timeseries'])

        # Construct StateRepository
        # if hasattr(self, 'initial_state'):
        #     self.initial_state_repo = self.initial_state['repository']['class'](
        #         **self.initial_state['repository']['params'])
        # if hasattr(self, 'end_state'):
        #     self.end_state_repo = self.end_state['repository']['class'](
        #         **self.end_state['repository']['params'])

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
        #self.target_repo = []
        #self.construct_target_repo()

    def validate(self):
        """Check for the existence of mandatory fields."""
        assert hasattr(self, "model_config_file")
        assert hasattr(self, "optimization_method")
        assert hasattr(self, "calibration_parameters")
        assert hasattr(self, "target")

    def get_target_repo(self):
        target_repo = [{'repository': target_repo_constructor(repo['repository'], repo['params']), '1D_timeseries': [target_ts for target_ts in repo['1D_timeseries']]} for repo in self.target]
        [target_ts.update({'start_datetime': utctime_from_datetime(target_ts['start_datetime'])}) for repo in target_repo for target_ts in repo['1D_timeseries']]
        return target_repo
        # for repo in self.target:
        #     repo['repository'] = target_repo_constructor(repo['repository'], repo['params'])
        #     [target_ts.update({'start_datetime': utctime_from_datetime(target_ts['start_datetime'])}) for target_ts in repo['1D_timeseries']]
        #     self.target_repo.append(repo)


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

        self.forecast_names = forecast_names

        # Load main configuration file
        with open(self._config_file,encoding='utf8') as cfg:
            cfg_m = yaml.load(cfg)[self._config_section]
        configs = cfg_m['forecast_runs']
        for name in self.forecast_names:
            assert name in configs


        start, n, dt = utctime_from_datetime(cfg_m['start_datetime']), cfg_m['number_of_steps'], cfg_m['run_time_step']
        if forecast_time is None:
            self.forecast_time = start + n * dt
        else:
            self.forecast_time = forecast_time
            t_override = {'start_datetime': datetime.utcfromtimestamp(self.forecast_time - n *dt)}

        sim_config_overrides = {'config': t_override}
        self.sim_config = YAMLSimConfig(self._config_file, self._config_section, overrides=sim_config_overrides)

        fc0 = self.forecast_names[0]
        configs[fc0].update({'start_datetime': datetime.utcfromtimestamp(self.forecast_time)})
        fc_time = self.forecast_time
        for i in range(1,len(self.forecast_names)):
            fc_1 = self.forecast_names[i-1]
            fc_time += configs[fc_1]['number_of_steps']*configs[fc_1]['run_time_step']
            configs[self.forecast_names[i]].update({'start_datetime': datetime.utcfromtimestamp(fc_time)})

        self.forecast_config = {name: YAMLSimConfig(self._config_file, self._config_section, overrides={'config':configs[name]})
                                for name in self.forecast_names}
