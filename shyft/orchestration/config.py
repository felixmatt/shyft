"""
Utilities for reading configurations for SHyFT simulations and calibrations.
"""

from datetime import datetime
import yaml
from shyft import api

import numpy as np

utc_calendar = api.Calendar()
"""Invariant global calendar in UTC."""


def utctime_from_datetime(dt):
    """Returns utctime of datetime dt (calendar interpreted as UTC)."""
    # Number of seconds since epoch
    nsec = np.array([dt], dtype="datetime64[s]").astype(np.long)[0]
    dt = datetime.fromtimestamp(nsec)
    return utc_calendar.time(api.YMDhms(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second))


class OrchestrationConfig(object):
    """
    Concrete class for yaml content.
    """

    def __init__(self, config_file, config_section):
        self._config_file = config_file
        self._config_section = config_section
        with open(config_file) as cfg_file:
            config = yaml.load(cfg_file)[config_section]
        # Expose all keys in yaml file as attributes
        self.__dict__.update(config)
        # Create a time axis
        self.start_time = utctime_from_datetime(self.start_datetime)
        self.time_axis = api.Timeaxis(
            self.start_time, self.run_time_step, self.number_of_steps)

    def __repr__(self):
        srepr = "%s::%s(" % (self.__class__.__name__, self._config_section)
        for key in self.__dict__:
            srepr += "%s=%r, " % (key, self.__dict__[key])
        srepr = srepr[:-2]
        return srepr + ")"
