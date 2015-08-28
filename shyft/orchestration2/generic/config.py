from __future__ import print_function
from __future__ import absolute_import

from shyft.orchestration2.base_config import BaseConfig
from .model import Model
from .region import Region
from .datasets import Datasets


class Config(BaseConfig):
    """
    Main class hosting a complete configuration section for an Shyft run.
    """

    @property
    def region_config(self):
        if '_region_config' not in self.__dict__:
            self._region_config = Region(self.abspath(self.region_config_file))
        return self._region_config

    @property
    def model_config(self):
        if '_model_config' not in self.__dict__:
            self._model_config = Model(self.abspath(self.model_config_file))
        return self._model_config

    @property
    def datasets_config(self):
        if '_datasets_config' not in self.__dict__:
            self._datasets_config = Datasets(self.abspath(self.datasets_config_file))
        return self._datasets_config

    def process_params(self, params):
        # No additional params yet for the reference
        pass
