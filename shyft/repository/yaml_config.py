from __future__ import absolute_import
from __future__ import print_function

#import os
#from abc import ABCMeta, abstractproperty, abstractmethod
#import urlparse

import yaml

# *** Ancillary config files ***

class BaseYamlConfig(object):
    """Base class for all other configuration ABC classes."""

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
