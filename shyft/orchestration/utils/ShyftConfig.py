import yaml
import os
from datetime import datetime
import collections
from shyft.orchestration.repository.testsupport.mocks import MockStateRepository
from shyft.orchestration.repository.testsupport.mocks import MockInputSourceRepository
from shyft.orchestration.repository.testsupport.mocks import mock_state_data
from shyft.orchestration.repository.testsupport.mocks import mock_station_data
from shyft.orchestration.repository.testsupport.mocks import mock_cell_data


def update(d, u):
    """Simple recursive update of dictionary d with u"""
    for k, v in u.iteritems():
        if isinstance(v, collections.Mapping):
            r = update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d


class ConfigurationError(Exception):

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


class CalibrationConfig(object):

    def __init__(self, filename, name=None):
        with open(filename, "r") as ff:
            config_set = yaml.load(ff.read())

        config = config_set[name]
        self.validate(config)

        self._model_config = config["model_config_type"](
            config["model_config_file"], overrides=config.get("overrides", None), name=name)
        self._calibration_parameters = config["calibration_parameters"]
        self._calibration_type = config["calibration_type"]
        self._catchment_index = config["catchment_index"]
        self._target = config["target"]

    @property
    def model_config(self):
        return self._model_config

    @property
    def calibration_type(self):
        return self._calibration_type

    @property
    def catchment_index(self):
        return self._catchment_index

    @property
    def calibration_parameters(self):
        return self._calibration_parameters

    @property
    def target(self):
        return self._target

    @staticmethod
    def validate(config):
        if "catchment_index" not in config:
            raise ConfigurationError("You must specify a catchment to calibrate")
        if "model_config_type" not in config:
            raise ConfigurationError("Calibration configuration must specify a model config type")
        if "model_config_file" not in config:
            raise ConfigurationError("Calibration configuration must specify a model config file")
        if not os.path.isfile(config["model_config_file"]):
            raise ConfigurationError("Couldn't find file {}".format(config["model_config_file"]))
        if "calibration_parameters" not in config:
            raise ConfigurationError("Calibration configuration must a calibration_parameters section.")
        if "target" not in config:
            raise ConfigurationError("Calibration configuration needs a target time series.")
        parameters = config["calibration_parameters"]
        for name,limit in parameters.items():
            if "min" not in limit.keys():
                raise ConfigurationError("Parameter {} needs a minimum value".format(name))
            if "max" not in limit.keys():
                raise ConfigurationError("Parameter {} needs a maximum value".format(name))


class BaseConfig(object):

    def __init__(self, filename, config=None):
        if not os.path.isfile(filename) and config is None:
            raise ConfigurationError("No file named {} found, and configuration data not given.".format(filename))
        ff = open(filename, 'r')
        self.config = yaml.load(ff.read())
        ff.close()
        self.validate()

    def validate(self):
        pass


class ShyftConfig(BaseConfig):

    def __init__(self, filename, name=None, config=None, overrides=None):
        super(ShyftConfig, self).__init__(filename, config=config)
        if overrides is None:
            overrides = {}
        base_dir = os.path.dirname(filename)
        print base_dir
        if name is None and len(self.config) > 1:
            raise ConfigurationError("Please choose one of the case names: {}".format(", ".join(self.config.keys())))
        elif name is None:
            name = self.config.keys()[0]
            print "Automatically choosing case {}".format(name)
        case = self.config[name]
        self._region_config = RegionConfig(os.path.join(base_dir, case["region_config"]))
        self._model_config = ModelConfig(
            os.path.join(base_dir, case["model_config"]), overrides=overrides.get("model", {}))
        self.datasets_config = DataConfig(os.path.join(base_dir, case["datasets_config"]))

        self.t_start = int(round((case["start_datetime"] - datetime.utcfromtimestamp(0)).total_seconds()))
        self.dt = case["run_time_step"]
        self.n_steps = case["number_of_steps"]
        state_repository_constructor = case["state_repository"]["constructor"][0]
        state_repository_constructor_args = case["state_repository"]["constructor"][1:]

        region_constructor = self.region_config.model_constructor
        region_constructor_args = self.region_config.model_constructor_args
        self._cell_read_only_repository = region_constructor(self.region_config, *region_constructor_args)
        info_dict = {"t_start": self.t_start,
                     "num_cells": len(self.cell_read_only_repository.get("catchment_id"))}
        self._state_repository = state_repository_constructor(info_dict, *state_repository_constructor_args)
        self._state_saver = case["state_repository"]["serializer"][0]
        self._state_saver_args = case["state_repository"]["serializer"][1:]
        self._input_source_repository = self.datasets_config.dataset_repository_constructor(
            self, self.t_start, self.t_start + self.dt*self.n_steps)

        self.destinations = self.datasets_config.destinations

    def validate(self):
        config = self.config
        fields = ["region_config", "model_config", "datasets_config", "start_datetime", "run_time_step",
                  "number_of_steps", "state_repository"]
        messages = []
        if not config.keys():
            messages.append("No entries found in configuration file.")
        for key, value in config.iteritems():
            for field in fields:
                if field not in value.keys():
                    messages.append("Configuration file misses {}".format(field))
            for k in value.keys():
                if k not in fields:
                    messages.append("Unknown option {}".format(k))
        if messages:
            raise ConfigurationError("\n".join(messages))

    @property
    def state_repository(self):
        return self._state_repository

    @property
    def state_saver(self):
        return self._state_saver

    @property
    def state_saver_args(self):
        return self._state_saver_args[:]

    @property
    def cell_read_only_repository(self):
        return self._cell_read_only_repository

    @property
    def input_source_repository(self):
        return self._input_source_repository

    @property
    def model_config(self):
        return self._model_config

    @property
    def region_config(self):
        return self._region_config

    @property
    def model(self):
        return self.model_config.model

    @property
    def parameters(self):
        return self.model_config.parameters

    @property
    def n_x(self):
        return self.region_config.n_x

    @property
    def n_y(self):
        return self.region_config.n_y

    @property
    def bounding_box(self):
        rc = self.region_config
        return [rc.x_min, rc.x_min + rc.dx*rc.n_x, rc.y_min, rc.y_min + rc.dy*rc.n_y]


class RegionConfig(BaseConfig):

    def __init__(self, filename, config=None):
        super(RegionConfig, self).__init__(filename, config=config)
        domain = self.config["domain"]
        self.n_x = domain["nx"]
        self.n_y = domain["ny"]
        self.dx = domain["step_x"]
        self.dy = domain["step_y"]
        self.epsg_id = domain["EPSG"]
        self.x_min = domain["upper_left_x"]
        self.y_min = domain["upper_left_y"] - self.n_y*self.dy
        self.x_max = domain["upper_left_x"] + self.n_x*self.dx
        self.y_max = domain["upper_left_y"]
        self.catchment_indices = self.config["repository"]["catchment_indices"]
        self.model_constructor = self.config["repository"]["constructor"][0]
        self.model_constructor_args = self.config["repository"]["constructor"][1:]
        self.stations = self.config["stations"]
        self.repository_data = self.config["repository"]
        if "parameter_overrides" in self.config.keys():
            self.parameter_overrides = self.config["parameter_overrides"]
        else:
            self.parameter_overrides = None

    def validate(self):
        config = self.config
        messages = []
        expected = {"domain": {"upper_left_x": float,
                               "upper_left_y": float,
                               "nx": int,
                               "ny": int,
                               "step_x": float,
                               "step_y": float,
                               "EPSG": int},
                    "repository": {"constructor":  list,
                                   "catchment_indices": list},
                    "stations": {"database": type,
                                 "indices": list}}
        optional = {"parameter_overrides": list}
        for k, v in expected.iteritems():
            if k not in config:
                messages.append("Mandatory entry '{}' not found in configuration.".format(k))
                continue
            if isinstance(v, dict):
                for kk, vv in v.iteritems():
                    if kk not in config[k]:
                        messages.append("Mandatory entry '{}' needed by '{}' not found.".format(kk, k))
                        continue
                    try:
                        vv(config[k][kk])
                    except:
                        messages.append("Could not convert entry value to {} for {} with type {}".format(
                            vv, kk, type(vv)))
        for k, v in config.iteritems():
            if k not in expected and k not in optional:
                messages.append("Unknown option: {}".format(k))
        if messages:
            raise ConfigurationError("\n".join(messages))


class ModelConfig(BaseConfig):

    def __init__(self, filename, config=None, overrides=None):
        super(ModelConfig, self).__init__(filename, config=config)
        if overrides is not None:
            self.config.update(overrides)
        self.parameters = self.config["parameters"]
        self.model = self.config["model"]

    def validate(self):
        pass


class DataConfig(BaseConfig):

    def __init__(self, filename, config=None):
        super(DataConfig, self).__init__(filename, config=config)
        self.dataset_repository_constructor = self.config["dataset_repository"]["constructor"]
        #self.output_repository_constructor = self.config["output_repository"]["constructor"]
        self.destinations = self.config["destinations"]
        self.sources = self.config["sources"]

    def validate(self):
        pass


class MockConfig(object):

    def __init__(self, filename):

        ff = open(filename, "r")
        config = yaml.load(ff.read())
        ff.close()

        MockConfig.validate(config)

        bounding_box = config["bounding_box"]
        cell_dx = config["cell_dx"]
        cell_dy = config["cell_dy"]

        # Adjust box in case resolutions don't match
        n_x = int(round((bounding_box[1] - bounding_box[0])/cell_dx))
        n_y = int(round((bounding_box[3] - bounding_box[2])/cell_dy))
        bounding_box[1] = bounding_box[0] + n_x*cell_dx
        bounding_box[3] = bounding_box[2] + n_y*cell_dy

        self._input_source_repository = config["input_source_repository"]["type"].factory(
            *mock_station_data(bounding_box, 2, 2))
        self._cell_read_only_repository = config["cell_repository"]["type"].factory(
            **mock_cell_data(bounding_box, n_x, n_y))
        self._state_repository = config["state_repository"]["type"].factory(
            **mock_state_data(n_x, n_y))

        self._model = config["model"]
        self._parameters = config["parameters"]

        self._bounding_box = bounding_box
        self._n_x = n_x
        self._n_y = n_y

    @property
    def state_repository(self):
        return self._state_repository

    @property
    def cell_read_only_repository(self):
        return self._cell_read_only_repository

    @property
    def input_source_repository(self):
        return self._input_source_repository

    @property
    def model(self):
        return self._model

    @property
    def parameters(self):
        return self._parameters

    @property
    def n_x(self):
        return self._n_x

    @property
    def n_y(self):
        return self._n_y

    @property
    def bounding_box(self):
        return self._bounding_box

    @staticmethod
    def validate(config):
        for key in ["input_source_repository", "state_repository", "cell_repository"]:
            if key not in config:
                raise ConfigurationError("Configuration must specify {}.".format(key))
            if "type" not in config[key]:
                raise ConfigurationError("Configuration must for {} needs a type.".format(key))

        required_types = [MockInputSourceRepository, MockStateRepository, MockCellReadOnlyRepository]
        provided_types = [config[x]["type"] for x in ["input_source_repository", "state_repository", "cell_repository"]]
        for a, b in zip(required_types, provided_types):
            if a is not b:
                raise ConfigurationError("MockConfig requires that {} is {}".format(a, b))


if __name__ == "__main__":
    filename = "config/NeaNidelva_configuration.yaml"
    config = ShyftConfig(filename)
