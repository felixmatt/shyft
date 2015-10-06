"""
Simulator classes for running SHyFT forward simulations.
"""
from __future__ import print_function
from __future__ import absolute_import

from shyft import api

class SimulationError(Exception):
    pass


class SimpleSimulator(object):
    """
    This simlator orchestrates a simple shyft run based on repositories
    given as input.
    """

    def __init__(self, model, region_id, interpolation_id, region_model_repository, 
                 geo_ts_repositories, interpolation_parameter_repository, catchments=None):
        self.region_model_repository = region_model_repository
        self.interpolation_id = interpolation_id
        self.ip_repos = interpolation_parameter_repository
        self._geo_ts_names = ("temperature", "wind_speed", "precipitation", "relative_humidity", "radiation")
        if not isinstance(geo_ts_repositories, (list, tuple)):
            geo_ts_repositories = [geo_ts_repositories]
        self._geo_ts_repos = geo_ts_repositories
        self.region_model = region_model_repository.get_region_model(region_id, model, catchments=catchments)
        self.epsg = self.region_model.bounding_region.epsg()

    def set_geo_ts_repositories(self, geo_ts_repositories):
        if not isinstance(geo_ts_repositories, (list, tuple)):
                geo_ts_repositories = [geo_ts_repositories]
        self._geo_ts_repos = geo_ts_repositories

    def run(self, time_axis):
        bbox = self.region_model.bounding_region.bounding_box(self.epsg)
        period = time_axis.total_period()
        sources = self._geo_ts_repos[0].get_timeseries(self._geo_ts_names, period, geo_location_criteria=bbox)
        for gt in self._geo_ts_repos[1:]:
            sources.update(gt.get_timeseries(self._geo_ts_names, period, geo_location_criteria=bbox))
        region_env = api.ARegionEnvironment()
        region_env.temperature = sources["temperature"]
        region_env.precipitation = sources["precipitation"]
        region_env.radiation = sources["radiation"]
        region_env.wind_speed = sources["wind_speed"]
        region_env.rel_hum = sources["relative_humidity"]
        interp_params = self.ip_repos.get_parameters(self.interpolation_id)
        self.region_model.run_interpolation(interp_params, time_axis, region_env)
        return True

