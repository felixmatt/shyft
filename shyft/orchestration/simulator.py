"""
Simulator classes for running SHyFT forward simulations.
"""
from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import math
from shyft import api


class SimulatorError(Exception):
    pass


class DefaultSimulator(object):
    """
    This simulator orchestrates a simple shyft run based on repositories
    given as input.
    """

    def __init__(self, *args, **kwargs):
        """
        Create a simple simulator.

        Repositories below as referred to as relative to the shyft.repository package

        Parameters
        ----------
        region_id: string
            Region identifier to be used with the region model repository
            to qualify what region to use.
        interpolation_id: string
            Identifier to use with the interpolation parameter
            repository.
        region_model_repository: interfaces.RegionModelRepository subclass
            Repository that can deliver a model with initialized cells
        geo_ts_repository: interfaces.GeoTsRepository subclass
            Repository that can deliver time series data to drive simulator.
        interpolation_parameter_repository: interfaces.InterpolationParameterRepository subclass
            Repository that can deliver interpolation parameters
        catchments: list of identifies, optional
            List of catchment identifiers to extract from region through
            the region_model_repository.
        """
        if len(args) == 1:
            self._copy_construct(*args)
        else:
            self._construct_from_repositories(*args, **kwargs)

    def _construct_from_repositories(self, region_id, interpolation_id, region_model_repository,
                                     geo_ts_repository, interpolation_parameter_repository, initial_state_repository=None,
                                     catchments=None):
        """
        Create a simple simulator.

        Repositories below as referred to as relative to the shyft.repository package

        Parameters
        ----------
        region_id: string
            Region identifier to be used with the region model repository
            to qualify what region to use.
        interpolation_id: string
            Identifier to use with the interpolation parameter
            repository.
        region_model_repository: interfaces.RegionModelRepository subclass
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
        self.region_model = region_model_repository.get_region_model(region_id,
                                                                     catchments=catchments)
        self.epsg = self.region_model.bounding_region.epsg()
        self.initial_state_repo = initial_state_repository
        self.state = None
        self.time_axis = None
        self.region_env = None
        self.optimizer = None

    def _copy_construct(self, other):
        self.region_model_repository = other.region_model_repository
        self.interpolation_id = other.interpolation_id
        self.ip_repos = other.ip_repos
        self._geo_ts_names = other._geo_ts_names
        self.geo_ts_repository = other.geo_ts_repository
        clone_op = getattr(other.region_model, "clone", None)
        if callable(clone_op):
            self.region_model = clone_op(other.region_model)
        else:
            self.region_model = other.region_model.__class__(other.region_model)
        #  self.region_model = other.region_model.clone() # __class__(other.region_model)
        self.epsg = other.epsg
        self.initial_state_repo = other.initial_state_repo
        self.state = other.state
        self.time_axis = other.time_axis
        self.region_env = other.region_env

    def copy(self):
        return self.__class__(self)

    def _get_region_environment(self, sources):
        region_env = api.ARegionEnvironment()
        region_env.temperature = sources["temperature"]
        region_env.precipitation = sources["precipitation"]
        region_env.radiation = sources["radiation"]
        region_env.wind_speed = sources["wind_speed"]
        region_env.rel_hum = sources["relative_humidity"]
        return region_env

    def _fetch_source_run_interp(self, time_axis):
        print("Fetching sources and running interpolation...")
        bbox = self.region_model.bounding_region.bounding_box(self.epsg)
        period = time_axis.total_period()
        sources = self.geo_ts_repository.get_timeseries(self._geo_ts_names, period,
                                                        geo_location_criteria=bbox)
        self.region_env = self._get_region_environment(sources)

        interp_params = self.ip_repos.get_parameters(self.interpolation_id)
        self.region_model.run_interpolation(interp_params, time_axis, self.region_env)

    def simulate(self):
        runnable = all((self.state is not None, self.time_axis is not None,
                        self.region_env is not None))
        if runnable:
            self.region_model.set_states(self.state)
            print("Running simulation...")
            self.region_model.run_cells()
        else:
            raise SimulatorError("Model not runnable.")

    def run(self, time_axis=None, state=None):
        """
        Forward simulation over time axis

        Parameters
        ----------
        time_axis: shyft.api.TimeAxis
            Time axis defining the simulation period, and step sizes.
        state: shyft.api state
        """
        if time_axis is not None:
            self.time_axis = time_axis
        self.state = self.get_initial_state() if state is None else state
        self._fetch_source_run_interp(self.time_axis)
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
            simulator.region_env = simulator._get_region_environment(source)
            runnables.append(simulator)
        return runnables

    def optimize(self, time_axis, state, target_specification, p, p_min, p_max, optim_method = 'min_bobyqa',
                 optim_method_params = {'max_n_evaluations':1500, 'tr_start':0.1, 'tr_stop':1.0e-5}, verbose_level=0, run_interp=True):
        if not hasattr(self.region_model, "optimizer_t"):
            raise SimulatorError("Simulator's region model {} cannot be optimized, please choose "
                                 "another!".format(self.region_model.__class__.__name__))
        is_correct_p_type = [isinstance(_, self.region_model.parameter_t) for _ in [p, p_min, p_max]]
        if not all(is_correct_p_type):
            raise SimulatorError("{} must be of type {}".format(
                ','.join([name for i, name in enumerate(['min', 'max', 'init'])
                          if not is_correct_p_type[i]]), self.region_model.parameter_t.__name__))
        self.state = state
        self.time_axis = time_axis
        if run_interp:
            self._fetch_source_run_interp(self.time_axis)
        self.region_model.set_states(self.state)
        self.optimizer = self.region_model.optimizer_t(self.region_model,
                                                       target_specification,
                                                       [p_min.get(i) for i in range(p_min.size())],
                                                       [p_max.get(i) for i in range(p_max.size())])
        self.optimizer.set_verbose_level(verbose_level)
        p_vec = [p.get(i) for i in range(p.size())]
        print("Calibrating...")
        if optim_method == "min_bobyqa":
            p_vec_opt = self.optimizer.optimize(p_vec, **optim_method_params)
        elif optim_method == "dream":
            p_vec_opt = self.optimizer.optimize_dream(p_vec, **optim_method_params)
        elif optim_method == "sceua":
            p_vec_opt = self.optimizer.optimize_sceua(p_vec, **optim_method_params)
        else:
            raise ValueError("Unknown optimization method: %s" % self.optim_method)
        p_res = self.region_model.parameter_t()
        p_res.set(p_vec_opt)
        return p_res

    @property
    def reg_model_state(self):
        state = self.region_model.__class__.state_t.vector_t()
        self.region_model.get_states(state)
        return state

    def get_initial_state(self):
        state_id = 0
        if hasattr(self.initial_state_repo, 'n'): # No stored state, generated on-the-fly
            self.initial_state_repo.n = self.region_model.size()
        else:
            states = self.initial_state_repo.find_state(
                region_model_id_criteria=self.region_model_id,
                utc_timestamp_criteria=self.time_axis.start, tag_criteria=None)
            if len(states) > 0:
                state_id = states[0].state_id #most_recent_state i.e. <= start time
            else:
                raise SimulatorError('No initial state matching criteria.')
        return self.initial_state_repo.get_state(state_id)

    def discharge_adjusted_state(self, obs_discharge, state=None):
        """
        Parameters
        ----------
        obs_discharge: float
            Observed discharge in units m3/s
        state: shyft.api state vector type, optional
            Vector of state having ground water response kirchner with variable q.
        """
        if state is None:
            state = self.reg_model_state
        reg_mod = self.region_model
        areas = np.array([cell.geo.area() for cell in reg_mod.get_cells()])
        area_tot = areas.sum()
        avg_obs_discharge = obs_discharge*3600.*1000./area_tot  # Convert to l/h per m2
        state_discharge = np.array([state[i].kirchner.q for i in range(state.size())])
        avg_state_discharge = (state_discharge*areas).sum()/area_tot
        discharge_ratios = state_discharge/avg_state_discharge
        updated_state_discharge = avg_obs_discharge*discharge_ratios
        for i in range(state.size()):
            if not math.isnan(updated_state_discharge[i]):
                state[i].kirchner.q = updated_state_discharge[i]
            else:
                state[i].kirchner.q = 0.5
        return state
