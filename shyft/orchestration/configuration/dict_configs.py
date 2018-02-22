# -*- coding: utf-8 -*-


from shyft import api

from . import config_interfaces

def utctime_from_datetime(dt):
    utc_calendar = api.Calendar()
    return utc_calendar.time(api.YMDhms(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second))


class DictRegionConfig(config_interfaces.RegionConfig):
    """
    Dict based region configuration, using a Dictionary instance
    for holding the content.
    """

    def __init__(self, config_dict):
        self._config = config_dict

    def parameter_overrides(self):
        return self._config.get("parameter_overrides", {})

    def domain(self):
        return self._config['domain']

    def repository(self):
        return self._config['repository']

    def catchments(self):
        return self._config.get("catchment_indices", None)


class DictModelConfig(config_interfaces.ModelConfig):
    """
    Dict based model configuration, using a Dictionary instance
    for holding the content.
    """

    def __init__(self, config_dict, overrides=None):
        self._config = config_dict
        if overrides is not None:
            self._config.update(overrides)

    def model_parameters(self):
        return self._config['model_parameters']

    def model_type(self):
        return self._config['model_t']


class InterpolationConfig(object):
    """
    Dict based model configuration, using a YamlContent instance
    for holding the content.
    """

    def __init__(self, config_file):
        pass

    def interpolation_parameters(self):
        pass


class ConfigError(Exception):
    pass

