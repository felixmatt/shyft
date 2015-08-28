"""
Interfaces for all configuration objects (region, model, datasets...).
"""

from __future__ import absolute_import

import os
from abc import ABCMeta, abstractproperty, abstractmethod
import urlparse

import yaml

from . import state
from .utils import utctime_from_datetime, get_class


def config_constructor(config_file, config_section, **overrides):
    config = yaml.load(open(config_file))[config_section]
    # Return the appropriate class depending on the scheme
    config_cls = get_class(config['repository']['class'])
    config_instance = config_cls(config_file, config_section, **overrides)
    if not isinstance(config_instance, BaseConfig):
        raise ValueError("The repository class is not an instance of 'BaseConfig'")
    return config_instance


class BaseConfig(object):
    """
    Abstract class hosting a complete configuration section for an Shyft run.
    """
    __metaclass__ = ABCMeta

    @abstractproperty
    def region_config(self):
        pass

    @abstractproperty
    def model_config(self):
        pass

    @abstractproperty
    def datasets_config(self):
        pass

    @abstractmethod
    def process_params(self, params):
        pass

    # Shared properties and methods follow...

    @property
    def start_time(self):
        return utctime_from_datetime(self.start_datetime)

    @property
    def stop_time(self):
        return self.start_time + self.run_time_step * self.number_of_steps

    @property
    def model_api(self):
        from shyft import api
        return getattr(api, self.model_config.parameters['model']['model_api'])

    def __init__(self, config_file, config_section, **overrides):
        section = yaml.load(open(config_file))[config_section]
        self.__dict__.update(section)
        if overrides is not None:
            self.__dict__.update(overrides)
        if not os.path.isabs(self.config_dir):
            self.config_absdir = os.path.join(os.path.dirname(config_file), self.config_dir)
        else:
            self.config_absdir = self.config_dir
        self.process_params(section['repository']['params'])

    def abspath(self, filepath):
        """Return the absolute path to the directory of netcdf data files."""
        if os.path.isabs(filepath):
            return filepath
        else:
            return os.path.join(self.config_absdir, filepath)

    def state_saver(self, state_):
        """Save the state in a YAML file."""
        if not hasattr(self, "state_output"):
            # The "state_output section is not present, so nothing to do here"
            return
        url_split = urlparse.urlsplit(self.state_output['file'])
        extension = url_split.path.split('.')[-1]
        if url_split.scheme == "yaml" or extension in ("yaml", "yml"):
            state_file = self.abspath(url_split.path)
            print("Storing state in:", state_file)
            state.save_state_as_yaml_file(state_, state_file, **self.state_output['params'])
        else:
            raise NotImplementedError(
                "Output scheme '%s' or extension '%s' is not implemented yet" % (url_split.scheme, extension))

    def state_loader(self):
        """Load the state from YAML file."""
        if not hasattr(self, "state_output"):
            # The "state_output section is not present, so nothing to do here"
            return
        url_split = urlparse.urlsplit(self.state_output['file'])
        extension = url_split.path.split('.')[-1]
        if url_split.scheme == "yaml" or extension in ("yaml", "yml"):
            state_file = self.abspath(url_split.path)
            print("Loading state from:", state_file)
            return state.load_state_from_yaml_file(state_file)
        else:
            raise NotImplementedError(
                "Output scheme '%s' or extension '%s' is not implemented yet" % (url_split.scheme, extension))


# *** Calibration ***

class CalibrationConfig(object):

    def __init__(self, config_file, config_section):
        self._config_file = config_file
        config = yaml.load(open(config_file))[config_section]
        self.__dict__.update(config)

        self.validate()

        # Get the location of the model_config_file relative to the calibration config file
        if not os.path.isabs(self.model_config_file):
            model_config_file = os.path.join(
                os.path.dirname(os.path.abspath(config_file)), self.model_config_file)
        # Create a new model_config attribute
        self.model_config = config_constructor(
            model_config_file, config_section, overrides=getattr(self, "overrides", None))

    def validate(self):
        """Check for the existence of mandatory fields."""
        assert hasattr(self, "model_config_file")
        assert hasattr(self, "calibration_type")
        assert hasattr(self, "calibration_parameters")
        assert hasattr(self, "target")
        assert hasattr(self, "catchment_index")


class BaseTarget(object):
    """Interface for target time series objects."""
    __metaclass__ = ABCMeta

    def absdir(self, data_dir):
        """Return the absolute path to the directory of data files."""
        if os.path.isabs(data_dir):
            return data_dir
        else:
            return os.path.join(os.path.dirname(self._config_file), data_dir)

    def __init__(self, data_file, config):
        self._config_file = config._config_file
        self.data_file = os.path.join(self.absdir(config.data_dir), data_file)

    def __repr__(self):
        repr = "%s(" % self.__class__.__name__
        for key in self.__dict__:
            repr += "%s=%r, " % (key, self.__dict__[key])
        repr = repr[:-2]
        return repr + ")"

    @abstractmethod
    def fetch_id(self, internal_id, uids, period):
        pass


def target_constructor(repo, config_section):
    # Return the appropriate class depending on the scheme
    config_cls = get_class(repo['class'])
    config_instance = config_cls(repo['file'], config_section)
    if not isinstance(config_instance, BaseTarget):
        raise ValueError("The repository class is not an instance of 'BaseTarget'")
    return config_instance

# *** Ancillary config files ***

class BaseAncillaryConfig(object):
    """Base class for all other configuration ABC classes."""

    def __init__(self, config_file):
        self._config_file = config_file
        config = yaml.load(open(config_file))
        # Expose all keys in yaml file as attributes
        self.__dict__.update(config)

    def __repr__(self):
        repr = "%s(" % self.__class__.__name__
        for key in self.__dict__:
            repr += "%s=%r, " % (key, self.__dict__[key])
        repr = repr[:-2]
        return repr + ")"


# *** Region ***

class BaseRegion(BaseAncillaryConfig):
    """Interface for Region objects."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def fetch_cell_properties(self, varname):
        pass

    @abstractmethod
    def fetch_cell_centers(self):
        pass

    @abstractmethod
    def fetch_cell_areas(self):
        pass

    @abstractmethod
    def fetch_catchments(self, what):
        pass

    def absdir(self, filepath):
        """Return the absolute path to the directory of data files."""
        if os.path.isabs(filepath):
            return filepath
        else:
            return os.path.join(os.path.dirname(self._config_file), filepath)


# *** Model ***

class BaseModel(BaseAncillaryConfig):
    """Interface for Model objects."""
    __metaclass__ = ABCMeta
    # Add methods as needed


# *** Datasets ***

class BaseDatasets(BaseAncillaryConfig):
    """Interface for Dataset objects."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def fetch_sources(self, input_source_types=None, period=None):
        pass

    @abstractmethod
    def store_destinations(self):
        pass


class BaseSourceDataset(BaseAncillaryConfig):
    """Interface for SourceDataset objects."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def fetch_sources(self, input_source_types, data, params, period):
        """`data` is a container for geo-located time-series that should be filled (appended)."""
        pass


# *** SimulationOutput ***

class BaseSimulationOutput(object):
    """Interface for SimulationOutput objects."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def save_output(self, cells, outfile):
        pass
