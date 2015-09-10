"""
Module for reading target time series needed for an SHyFT calibration run.
"""

from __future__ import absolute_import

import os

import numpy as np
from netCDF4 import Dataset

from shyft import api
from ..base_config import BaseTarget


class Target(BaseTarget):
    """Concrete implementation for target time series in reference."""

    def absdir(self, data_dir):
        """Return the absolute path to the directory of data files."""
        if os.path.isabs(data_dir):
            return data_dir
        else:
            return os.path.join(os.path.dirname(self._config_file), data_dir)

    def __init__(self, data_file, config):
        super(Target, self).__init__(data_file, config)

    def fetch_id(self, internal_id, uids, period):
        """Fetch all the time series given in `uids` list within date `period`.

        Return the result as a dictionary of shyft_ts.
        """
        result = {}
        ts_period = api.UtcPeriod(period[0], period[1])
        np_period = np.array((ts_period.start,ts_period.end), dtype="datetime64[s]")
        with Dataset(self.data_file) as dset:
            time = dset.groups[internal_id].variables['time'][:]
            nptime = np.array(time, dtype='datetime64[s]')
            # Index of time in the interesting period
            idx = np.where((nptime >= np_period[0]) & (nptime < np_period[1]))[0]
            times = api.UtcTimeVector.FromNdArray(nptime[idx].astype(np.long))
            for uid in uids:
                data = dset.groups[internal_id].variables[uid][:]
                values = api.DoubleVector.FromNdArray(data[idx])
                tsfactory = api.TsFactory()
                shyft_ts = tsfactory.create_time_point_ts(ts_period, times, values)
                result[uid] = shyft_ts
        return result
