"""
Simulator classes for running SHyFT forward simulations.
"""
from __future__ import print_function
from __future__ import absolute_import


class SimulationError(Exception):
    pass


class SimpleSimulator(object):
    """
    This simlator orchestrates a simple shyft run based on repositories
    given as input.
    """

    def __init__(self, model, region_id, region_model_repository, geo_ts_repository, catchments=None):
        self.region_model_repository = region_model_repository
        self.geo_ts_repository = geo_ts_repository
        self.region_model = region_model_repository.get_region_model(region_id, model, catchments=catchments)

    def run(self, time_axis):
        return True

