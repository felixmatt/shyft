"""
Simulator class for running an SHyFT simulation.
"""

from __future__ import print_function
from __future__ import absolute_import

import copy
import collections

import numpy as np

from shyft import api
from shyft.api import pt_gs_k
from .state import build_ptgsk_model_state_from_string, extract_ptgsk_model_state
from shyft.orchestration.utils.CellBuilder import cell_argument_factory
#from shyft.orchestration.repository.state_repository import TimeCondition
from . import utctime_from_datetime


def update_dict(d, u):
    """Simple recursive update of dictionary d with u"""
    for k, v in u.iteritems():
        if isinstance(v, collections.Mapping):
            r = update_dict(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d


class SimulatorError(Exception):
    pass


class Simulator(object):
    def __init__(self, config):
        self._config = config
        self._model = None
        self.end_utc_timestamp = None
        self.get_catchment_avg_result = {
            'SimDischarge': self._get_sum_catchment_discharge,
            'precipitation': self._get_sum_catchment_precipitation,
            'temperature': self._get_sum_catchment_temperature
            }
        # Check for the optional max_cells parameter in the configuration
        self._max_cells = config.max_cells if hasattr(config, "max_cells") else None

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("No model created, please call build_model first")
        return self._model

    @property
    def model_api(self):
        return self._config.model_api

    def sources(self):
        return [self._config.input_source_repository.get(k) for k in self._config.input_source_repository.find()]

    def raster(self, name):
        return self._config.cell_read_only_repository[name]

    def parameter(self, major, minor, name):
        parameters = self._config.model_config.parameters
        if major in parameters and minor in parameters[major] and name in parameters[major][minor]:
            return parameters[major][minor][name]
        raise ValueError("Unknown parameter: {}.{}.{}".format(major, minor, name))

    def _model_values(self, model_name, *names):
        return [self.parameter("model", model_name, name) for name in names]

    def _interpolation_values(self, interpolation_name, *names):
        return [self.parameter("interpolation", interpolation_name, name) for name in names]

    def cell_data(self, name):
        region_config = self._config.region_config
        if name == "catchments":
            data = region_config.fetch_catchments("values")
        elif name == "glacier_fraction":
            data = region_config.fetch_cell_properties("glacier-fraction")
        elif name == "lake_fraction":
            data = np.double(region_config.fetch_cell_properties("lake-fraction"))
        elif name == "forest_fraction":
            data = region_config.fetch_cell_properties("forest-fraction")
        elif name == "reservoir_fraction":
            data = region_config.fetch_cell_properties("reservoir-fraction")
        elif name == "geo_position":
            data = region_config.fetch_cell_centers()
        elif name == "area":
            data = region_config.fetch_cell_areas()
        else:
            raise ValueError("Cell property '%s' not supported" % name)
        return data[:self._max_cells]

    def cell_data_types(self):
        return self._config.cell_read_only_repository.find()

    def model_parameters_dict(self):
        # TODO: Replace with polymorphism
        if self.model_api in [pt_gs_k.PTGSKModel, pt_gs_k.PTGSKOptModel]:
            priestley_taylor = self._model_values("priestley_taylor", "albedo", "alpha")
            gamma_snow = self._model_values("gamma_snow", "winter_end_day_of_year", "initial_bare_ground_fraction",
                                            "snow_cv", "snow_tx",
                                            "wind_scale", "wind_const", "max_water", "surface_magnitude", "max_albedo",
                                            "min_albedo",
                                            "fast_albedo_decay_rate", "slow_albedo_decay_rate", "snowfall_reset_depth",
                                            "glacier_albedo")
            act_evap = self._model_values("actual_evapotranspiration", "scale_factor")
            kirchner = self._model_values("kirchner", "c1", "c2", "c3")
            cell = []

            return {"priestley_taylor": priestley_taylor, "gamma_snow": gamma_snow, "act_evap": act_evap,
                    "kirchner": kirchner, "cell": cell}
        raise ValueError("Unknown model: {}".format(self._config.model))

    def api_model_parameters(self):
        params = self.model_parameters_dict()
        pt_params = api.PriestleyTaylorParameter(*params["priestley_taylor"])
        gs_params = api.GammaSnowParameter(*params["gamma_snow"])
        ae_params = api.ActualEvapotranspirationParameter(*params["act_evap"])
        k_params = api.KirchnerParameter(*params["kirchner"])
        #c_params = api.CellParameter(*params["cell"])
        p_params = api.PrecipitationCorrectionParameter()
        return pt_gs_k.PTGSKParameter(pt_params, gs_params, ae_params, k_params, p_params)

    def interpolation_parameters(self):
        btk_param = api.BTKParameter(
            *self._interpolation_values("btk", "gradient", "gradient_sd", "sill", "nugget", "range", "zscale"))
        idw_arguments = self._interpolation_values("idw", "precipitation_gradient", "max_members", "max_distance")
        prec_param = api.IDWPrecipitationParameter(*idw_arguments)
        idw_arguments.pop(0)  # To remove parameter 'precipitation_gradient'
        ws_param = api.IDWParameter(*idw_arguments)
        rad_param = api.IDWParameter(*idw_arguments)
        rel_hum_param = api.IDWParameter(*idw_arguments)
        return api.InterpolationParameter(btk_param, prec_param, ws_param, rad_param, rel_hum_param)

    def state_data(self, condition=None, tags=None):
        """
        Return a valid state from the state repository.

        TODO: Implement more advanced filtering logic directly in repository?
        """
        keys = self._config.state_repository.find(condition, tags)
        if len(keys) == 0:
            raise RuntimeError("Could not get a valid model state from the state repository.")
        if len(keys) == 1:
            newest = self._config.state_repository.get(keys[0])
        else:
            newest = self._config.state_repository.get(keys[0])
            for key in keys[1:]:
                tmp = self._config.state_repository.get(key)
                if tmp.utc_timestamp > newest.utc_timestamp:
                    newest = tmp
        return copy.deepcopy(newest)

    def model_state(self, condition=None, tags=None):
        """Extract a model state and return it together with the timestamp. TODO: make it unaware of ptgsk!"""
        data = self.state_data(condition, tags)
        # print(data.state_list)
        return build_ptgsk_model_state_from_string(data.state_list), data.utc_timestamp

    def build_cells(self):
        # Fetch cell data
        catchment_id = self.cell_data("catchments")
        geo_position = self.cell_data("geo_position")
        lake_fraction = self.cell_data("lake_fraction")
        glacier_fraction = self.cell_data("glacier_fraction")
        reservoir_fraction = self.cell_data("reservoir_fraction")
        forest_fraction = self.cell_data("forest_fraction")
        area = self.cell_data("area")

        # condition = TimeCondition() <= self._config.start_datetime
        # state, _ = self.model_state(condition=condition)

        cells = self.model_api.cell_t.vector_t()
        num_cells = len(catchment_id)
        cells.reserve(num_cells)

        arg_builder = cell_argument_factory(self.model_api, geo_position, glacier_fraction, lake_fraction,
                                            reservoir_fraction, forest_fraction, area)
        alpha = self.parameter("model", "priestley_taylor", "alpha")
        c1 = self.parameter("model", "kirchner", "c1")
        c2 = self.parameter("model", "kirchner", "c2")
        c3 = self.parameter("model", "kirchner", "c3")
        overrides = self._config.region_config.parameter_overrides
        snow_cv = self.parameter("model", "gamma_snow", "snow_cv")
        ibf = self.parameter("model", "gamma_snow", "initial_bare_ground_fraction")
        arg_dict = {
            'gamma_snow': {
                "initial_bare_ground_fraction": ibf,
                "snow_cv": snow_cv
            },
            'priestley_taylor': {
                "alpha": alpha
            },
            'kirchner': {
                "c1": c1,
                "c2": c2,
                "c3": c3
            }
        }

       # Cell assembly
        catchment_map = []
        catchment_parameters=[]
        # print("Creating cells")
        for i in xrange(num_cells):
            c_id = catchment_id[i]
            if c_id not in catchment_map:
                catchment_map.append(c_id)
                mapped_catchment_id = len(catchment_map) - 1
                args_dict = copy.deepcopy(arg_dict)
                args_dict = update_dict(args_dict, overrides.get(c_id, {}))
                catchment_parameters.append(args_dict)
                #arg_builder.update(mapped_catchment_id, args_dict)
            else:
                mapped_catchment_id = catchment_map.index(c_id)

            ltf = api.LandTypeFractions()
            ltf.set_fractions(glacier_fraction[i],lake_fraction[i],reservoir_fraction[i],forest_fraction[i])
            mid_point = api.GeoPoint(geo_position[i][0],geo_position[i][1],geo_position[i][2])
            radiation_slope_factor = 0.9 # TODO: read from config
            geo = api.GeoCellData(mid_point,area[i],mapped_catchment_id,radiation_slope_factor,ltf)
            cell = self.model_api.cell_t() #(*arg_builder[i])
            cell.geo = geo
            cell.state.kirchner.q = 0.0001
            # cell.set_state(state[i])  # TODO: state needs to be created
            cells.append(cell)
        self.catchment_map = catchment_map
        self.catchment_parameters = catchment_parameters
        return cells

    def build_model(self, t_start, delta_t, n_steps):
        cells = self.build_cells()  # Need to do this to get the number of catchments
        model_parameter = self.api_model_parameters() #api.PTGSKParameter()
        # Ready to create the model
        self._model = self.model_api(cells, model_parameter)
        # Next is reading model
        sources = self._config.datasets_config.fetch_sources(
            period=(self._config.start_time, self._config.stop_time))
        time_axis = api.Timeaxis(t_start, delta_t, n_steps)
        self.end_utc_timestamp = t_start + delta_t * n_steps
        region_env = api.ARegionEnvironment()
        region_env.temperature = sources["temperature"]
        region_env.precipitation = sources["precipitation"]
        region_env.radiation = sources["radiation"]
        region_env.wind_speed = sources["wind_speed"]
        region_env.rel_hum = sources["relative_humidity"]
        self._model.run_interpolation(self.interpolation_parameters(), time_axis, region_env)

    def run_model(self, *args, **kwargs):
        self.model.run_cells()
        self.save_state()
        # self.save_result_timeseries()  # TODO

    def save_result_timeseries(self):
        shyft_ts_factory = api.TsFactory()
        shyft_catchment_result = dict()
        destinations = self._config.destinations
        for repository in destinations:
            for catch_res in repository['targets']:
                print(catch_res)
                # result = self.get_catchment_avg_result[catch_res['type']](catch_res['catchment_id'])
                if catch_res['time_axis'] is not None:
                    ts_start = utctime_from_datetime(catch_res['time_axis']['start_datetime'])
                    ts_dt = catch_res['time_axis']['time_step_length']
                    ts_nsteps = catch_res['time_axis']['number_of_steps']
                else:
                    ts_start = self._config.start_time
                    ts_dt = self._config.run_time_step
                    ts_nsteps = self._config.number_of_steps
                result = self.get_sum_catchment_result(catch_res['type'], catch_res['catchment_id'],
                                                       ts_start=ts_start, ts_dt=ts_dt, ts_nsteps=ts_nsteps)
                shyft_result_ts = shyft_ts_factory.create_point_ts(len(result), ts_start, ts_dt,
                                                                 api.DoubleVector([val for val in result]),
                                                                 api.IntVector([0, 0, 0]))
                shyft_catchment_result[catch_res['uid']] = shyft_result_ts
            constructor = repository["repository"][0]
            arg = repository["repository"][1]
            ts_repository = constructor(arg)
            ts_repository.store(shyft_catchment_result)

    def save_state(self):
        state = extract_ptgsk_model_state(self.model)
        state.utc_timestamp = self.end_utc_timestamp
        self._config.state_saver(state)

    def load_state(self):
        return self._config.state_loader()

    def get_calculated_discharge(self, i_list):
        found_idx = np.in1d(self.catchment_map, i_list)
        if np.count_nonzero(found_idx) == len(i_list):
            return self.model.get_sum_catchment_discharge(api.IntVector([i for i, j in enumerate(found_idx) if j]))
        else:
            raise SimulatorError("Global catchment index {} not found.".format(
                ','.join([str(val) for val in [i for i in i_list if i not in self.catchment_map]])))

    def _get_sum_catchment_discharge(self, indx_list):
        return self.model.get_sum_catchment_discharge(api.IntVector(indx_list))

    def _get_sum_catchment_precipitation(self, indx_list):
        return self.model.get_sum_catchment_precipitation(api.IntVector(indx_list))

    def _get_sum_catchment_temperature(self, indx_list):
        return self.model.get_sum_catchment_temperature(api.IntVector(indx_list))

    def get_sum_catchment_result(self, var_type, i_list, ts_start=None, ts_dt=None, ts_nsteps=None):
        if None in [ts_start, ts_dt, ts_nsteps]:
            ts_start = self._config.start_time
            ts_dt = self._config.run_time_step
            ts_nsteps = self._config.number_of_steps
        shyft_ts_factory = api.TsFactory()
        tst = api.TsTransform()
        found_indx = np.in1d(self.catchment_map, i_list)
        if np.count_nonzero(found_indx) == len(i_list):
            result_ts = self.get_catchment_avg_result[var_type]([i for i, j in enumerate(found_indx) if j])
            shyft_ts = shyft_ts_factory.create_point_ts(
                len(result_ts), self._config.start_time, self._config.run_time_step,
                api.DoubleVector([val for val in result_ts]), api.IntVector([0, 0, 0]))
            return tst.to_average_staircase(ts_start, ts_dt, ts_nsteps, shyft_ts)
        else:
            raise SimulatorError(
                "Global catchment index {} not found.".format(
                    ','.join([str(val) for val in [i for i in i_list if i not in self.catchment_map]])))
