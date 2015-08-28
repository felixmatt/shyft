"""
Module for reading target time series needed for an SHyFT calibration run.
"""

from __future__ import absolute_import

from ..base_config import BaseTarget


class Target(BaseTarget):
    """Concrete implementation for target time series in reference."""

    def __init__(self, data_file, config):
        raise NotImplementedError("The ReferenceTarget is currently unimplemented!")


    def fetch_id(self, internal_id, uids, period):
        """Fetch all the time series given in `uids` list within date `period`.

        Return the result as a dictionary of shyft_ts.
        """
        return None
