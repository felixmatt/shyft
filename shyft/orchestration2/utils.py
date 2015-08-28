# Some utilities

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from shyft import api


utc_calendar = api.Calendar()
""" invariant, global calendar, we use utc pr. default, but are still explicit about it"""

# Define the accessors for the cell data
cell_extractor = {
    'total_discharge': lambda x: x.rc.end_response.total_discharge,
    'snow_storage': lambda x: x.rc.end_response.gs.storage * (
        1.0 - (x.geo.land_type_fractions_info().lake() + x.geo.land_type_fractions_info().reservoir())),
    'discharge': lambda x: np.array([x.rc.avg_discharge.get(i).v for i in xrange(
        x.rc.avg_discharge.size())]),
    'temperature': lambda x: np.array([x.env_ts.temperature.get(i).v for i in xrange(
        x.env_ts.temperature.size())]),
    'precipitation': lambda x: np.array([x.env_ts.precipitation.get(i).v for i in xrange(
        x.env_ts.precipitation.size())])
    }
"""Accessors for the cell data"""


def utctime_from_datetime(dt):
    """returns utctime of the supplied datetime dt calendar coordinates interpreted as utc"""
    return utc_calendar.time(api.YMDhms(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second))


def utctime_from_datetime2(dt):
    """returns seconds since epoch to datetime dt interpreted as utc"""
    return np.array([dt], dtype="datetime64[s]").astype(np.long)[0]

def get_class(repo):
    """Get a Python class out of the `repo` class path in string form.
    """
    import_path = repo[:repo.rfind(".")]
    # The actual class is at the end of the import path
    repo_cls = repo[repo.rfind(".") + 1:]
    try:
        exec "from %s import %s" % (import_path, repo_cls)
    except ImportError:
        print("Repository '%s' cannot be imported.  Please check the path.")
    return eval(repo_cls)
