"""
Simulator classes for running SHyFT forward simulations.
"""
from __future__ import print_function
from __future__ import absolute_import
from shyft import api


class SimulatorError(Exception):
    pass


class SimpleSimulator(object):
    """
    This simlator orchestrates a simple shyft run based on repositories
    given as input.
    """

    def __init__(self, region_id, interpolation_id, region_model_repository,
                 geo_ts_repository, interpolation_parameter_repository, catchments=None):
        """
        Create a simple simulator.

        Repositories below as referred to as relative to the shyft.repository package

        Parameters
        ----------
        region_id: string
            Region identifyer to be used with the region model repository
            to qualify what region to use.
        interpolation_id: string
            Identifier to use with the interpolation parameter
            repository.
        region_model_repostiory: interfaces.RegionModelRepository subclass
            Repository that can deliver a model with initialized cells
        geo_ts_repository: interfaces.GeoTsRepository subclass
            Repository that can deliver time series data to drive simulator.
        interpolation_parameter_repository: interfaces.InterpolationParameterRepository subclass
            Repository that can deliver interpolation parameters
        catchments: list of identifies, optional
            List of catchment identifiers to extract from region through
            the region_model_repository.
        """
        self.region_model_repository = region_model_repository
        self.interpolation_id = interpolation_id
        self.ip_repos = interpolation_parameter_repository
        self._geo_ts_names = ("temperature", "wind_speed", "precipitation",
                              "relative_humidity", "radiation")
        self.geo_ts_repository = geo_ts_repository
        self.region_model = region_model_repository.get_region_model(region_id, catchments=catchments)
        self.epsg = self.region_model.bounding_region.epsg()

    def copy(self):
        return self.region_model.__class__(self)

    def _get_region_environment(self, sources):
        region_env = api.ARegionEnvironment()
        region_env.temperature = sources["temperature"]
        region_env.precipitation = sources["precipitation"]
        region_env.radiation = sources["radiation"]
        region_env.wind_speed = sources["wind_speed"]
        region_env.rel_hum = sources["relative_humidity"]
        return region_env

    def simulate(self):
        runnable = all((self.state is not None, self.time_axis is not None,
                        self.region_env is not None))
        if runnable:
            interp_params = self.ip_repos.get_parameters(self.interpolation_id)
            self.region_model.run_interpolation(interp_params, self.time_axis, self.region_env)
            self.region_model.set_states(self.state)
            self.region_model.run_cells()
            self.state = None
            self.time_axis = None
            self.region_env = None
        else:
            raise SimulatorError("Model not runnable.")

    def run(self, time_axis, state):
        """
        Forward simulation over time axis

        Parameters
        ----------
        time_axis: shyft.api.TimeAxis
            Time axis defining the simulation period, and step sizes.
        state: shyft.api state
        """
        bbox = self.region_model.bounding_region.bounding_box(self.epsg)
        period = time_axis.total_period()
        sources = self.geo_ts_repository.get_timeseries(self._geo_ts_names, period,
                                                        geo_location_criteria=bbox)
        self.region_env = self._get_region_environment(sources)
        self.state = state
        self.time_axis = time_axis
        self.simulate()

    def run_forecast(self, time_axis, t_c, state):
        bbox = self.region_model.bounding_region.bounding_box(self.epsg)
        period = time_axis.total_period()
        sources = self.geo_ts_repository.get_forecast(self._geo_ts_names, period, t_c,
                                                      geo_location_criteria=bbox)
        self.region_env = self._get_region_environment(sources)
        self.state = state
        self.time_axis = time_axis
        self.simulate()

    def create_ensembles(self, time_axis, t_c, state=None):
        bbox = self.region_model.bounding_region.bounding_box(self.epsg)
        period = time_axis.total_period()
        sources = self.geo_ts_repository.get_forecast_ensemble(self._geo_ts_names, period, t_c,
                                                               geo_location_criteria=bbox)
        runnables = []
        for source in sources:
            simulator = self.copy()
            simulator.state = state
            simulator.time_axis = time_axis
            simulator.region_env = self._get_region_environment(source)
            runnables.append(simulator)
        return runnables
