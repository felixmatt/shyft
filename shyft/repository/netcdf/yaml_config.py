from __future__ import absolute_import
from __future__ import print_function
from . import region_model
import yaml


class YamlContent(object):
    """
    Concrete class for yaml content.
    """

    def __init__(self, config_file):
        self._config_file = config_file
        config = yaml.load(open(config_file))
        # Expose all keys in yaml file as attributes
        self.__dict__.update(config)

    def __repr__(self):
        srepr = "%s(" % self.__class__.__name__
        for key in self.__dict__:
            srepr += "%s=%r, " % (key, self.__dict__[key])
        srepr = srepr[:-2]
        return srepr + ")"


class RegionConfig(region_model.RegionConfig):
    """
    Yaml based region configuration, using a YamlContent instance
    for holding the content.
    """

    def __init__(self, config_file):
        self._config = YamlContent(config_file)

    def parameter_overrides(self):
        return self._config.parameter_overrides

    def domain(self):
        return self._config.domain

    def repository(self):
        return self._config.repository


class ModelConfig(region_model.ModelConfig):
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
