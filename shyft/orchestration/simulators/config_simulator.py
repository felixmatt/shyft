import numpy as np
import yaml
import os

from .. import simulator
from shyft import api
from shyft.repository.interfaces import TsStoreItem
from shyft.repository.interfaces import TimeseriesStore

class ConfigSimulatorError(Exception):
    pass


class ConfigSimulator(simulator.DefaultSimulator):
    def __init__(self, arg):
        if isinstance(arg, self.__class__):
            super().__init__(arg)
            self.config = arg.config
        else:
            super().__init__(arg.region_model_id,arg.interpolation_id,arg.region_model,
                             arg.geo_ts, arg.interp_repos)
            self.config = arg

        self.time_axis = self.config.time_axis
        self.state = self.get_initial_state()

    def _extraction_method_1d(self,ts_info):
        c_id = ts_info['catchment_id']
        t_st, t_dt, t_n = ts_info['time_axis'].start(), ts_info['time_axis'].delta(), ts_info['time_axis'].size()
        tst = api.TsTransform()
        found_indx = np.in1d(self.region_model.catchment_id_map,c_id)
        if np.count_nonzero(found_indx) != len(c_id):
            raise ConfigSimulatorError(
                "Global catchment index {} not found.".format(
                    ','.join([str(val) for val in [i for i in c_id if i not in self.region_model.catchment_id_map]])))
        c_indx = [i for i,j in enumerate(found_indx) if j]
        methods = {'discharge': lambda m: tst.to_average(t_st, t_dt, t_n, m.statistics.discharge(c_indx))}
        return methods[ts_info['type']]

    def save_result_timeseries(self):
        for repo in self.config.dst_repo:
            save_list = [TsStoreItem(ts_info['uid'],self._extraction_method_1d(ts_info)) for ts_info in repo['1D_timeseries']]
            TimeseriesStore(repo['repository'], save_list).store_ts(self.region_model)

    def get_initial_state(self):
        state_id = 0
        if hasattr(self.config.initial_state_repo, 'n'): # No stored state, generated on-the-fly
            self.config.initial_state_repo.n = self.region_model.size()
        else:
            states = self.config.initial_state_repo.find_state(
                region_model_id_criteria = self.config.region_model_id,
                utc_timestamp_criteria = self.config.time_axis.start,tag_criteria=None)
            if len(states) > 0:
                state_id = states[0].state_id #most_recent_state i.e. <= start time
            else:
                raise ConfigSimulatorError('No initial state matching criteria.')
        return self.config.initial_state_repo.get_state(state_id)

    def save_end_state(self):
        endstate = self.region_model.state_t.vector_t()
        self.region_model.get_states(endstate)  # get the state at end of simulation
        self.config.end_state_repo.put_state(self.config.region_model_id, self.region_model.time_axis.total_period().end,
                                             endstate, tags=None)

    def run(self, time_axis=None, state=None):
        if time_axis is not None:
            self.time_axis = time_axis
        if state is not None:
            self.state = state
        super().run(self.time_axis, self.state)



class ConfigCalibrator(simulator.DefaultSimulator):
    @property
    def param_accessor(self):
        return self.region_model.get_region_parameter()

    @property
    def p_min(self):
        return api.DoubleVector([self._config.calibration_parameters[name]['min']
                                 for name in self.calib_param_names])

    @property
    def p_max(self):
        return api.DoubleVector([self._config.calibration_parameters[name]['max']
                                 for name in self.calib_param_names])

    def __init__(self, config):
        sim_config = config.sim_config
        super().__init__(sim_config.region_model_id,sim_config.interpolation_id,sim_config.region_model,
                         sim_config.geo_ts, sim_config.interp_repos)
        self._config = config
        self.tv = None
        self.obj_funcs = {'NSE': api.NASH_SUTCLIFFE, 'KGE': api.KLING_GUPTA}
        self.verbose_level = 1

    def init(self):
        self.calib_param_names = [self.param_accessor.get_name(i) for i in range(self.param_accessor.size())]
        if self.tv is None:
            self._create_target_specvect()

    def _create_target_specvect(self):
        self.tv = api.TargetSpecificationVector()
        tst = api.TsTransform()
        for ts_info in self._config.target_ts:
            mapped_indx = [i for i, j in enumerate(self.region_model.catchment_id_map) if j in ts_info['catch_id']]
            catch_indx = api.IntVector(mapped_indx)
            tsp = ts_info['ts']
            t = api.TargetSpecificationPts()
            t.catchment_indexes = catch_indx
            t.scale_factor = ts_info['weight']
            t.calc_mode = self.obj_funcs[ts_info['obj_func']['name']]
            t.s_r = ts_info['obj_func']['scaling_factors']['s_corr']
            t.s_a = ts_info['obj_func']['scaling_factors']['s_var']
            t.s_b = ts_info['obj_func']['scaling_factors']['s_bias']
            tsa = tst.to_average(ts_info['start_datetime'], ts_info['run_time_step'], ts_info['number_of_steps'], tsp)
            t.ts = tsa
            self.tv.append(t)
            #print(ts_info['uid'], mapped_indx)

    def calibrate(self, time_axis, state, optim_method, optim_method_params, p_vec=None):
        if p_vec is None:
            # p_vec = [(a + b) * 0.5 for a, b in zip(self.p_min, self.p_max)]
            p_vec = [a + (b - a) * 0.5 for a, b in zip(self.p_min, self.p_max)]
        if not hasattr(self.region_model, "optimizer_t"):
            raise ConfigSimulatorError("Simulator's region model {} cannot be optimized, please choose "
                                 "another!".format(self.region_model.__class__.__name__))
        # if not all([isinstance(_, self.region_model.parameter_t) for _ in [p, p_min, p_max]]):
        #     raise ConfigSimulatorError("p, p_min, and p_max must be of type {}"
        #                          "".format(self.region_model.parameter_t.__name__))

        #p_vec = [p.get(i) for i in range(p.size())]
        p_vec_min = self.p_min # [p_min.get(i) for i in range(p_min.size())]
        p_vec_max = self.p_max # [p_max.get(i) for i in range(p_max.size())]

        bbox = self.region_model.bounding_region.bounding_box(self.epsg)
        period = time_axis.total_period()
        sources = self.geo_ts_repository.get_timeseries(self._geo_ts_names, period,
                                                        geo_location_criteria=bbox)
        self.region_env = self._get_region_environment(sources)
        self.state = state
        self.time_axis = time_axis
        interp_params = self.ip_repos.get_parameters(self.interpolation_id)
        self.region_model.run_interpolation(interp_params, self.time_axis, self.region_env)
        self.region_model.set_states(self.state)
        self.optimizer = self.region_model.optimizer_t(self.region_model,
                                                       self.tv,
                                                       p_vec_min,
                                                       p_vec_max)
        self.optimizer.set_verbose_level(self.verbose_level)
        #p_vec_opt = self.optimizer.optimize(p_vec, max_n_evaluations=max_n_evaluations,
        #                                    tr_start=tr_start, tr_stop=tr_stop)
        print("Calibrating...")
        if optim_method == "min_bobyqa":
            p_vec_opt = self.optimizer.optimize(p_vec, **optim_method_params)
        elif optim_method == "dream":
            p_vec_opt = self.optimizer.optimize_dream(p_vec, **optim_method_params)
        elif optim_method == "sceua":
            p_vec_opt = self.optimizer.optimize_sceua(p_vec, **optim_method_params)
        else:
            raise ValueError("Unknown optimization method: %s" % optim_method)
        p_res = self.region_model.parameter_t()
        p_res.set(p_vec_opt)
        return p_res

    def save_calibrated_model(self, optim_param, outfile=None):
        """Save calibrated params in a model-like YAML file."""
        name_map = {"pt": "priestley_taylor", "kirchner": "kirchner",
                    "p_corr": "precipitation_correction", "ae": "actual_evapotranspiration",
                    "gs": "gamma_snow", "ss": "skaugen_snow", "hs": "hbv_snow"}
        mapped_results = {optim_param.get_name(i): optim_param.get(i) for i in range(optim_param.size())}

        # Existing model parameters structure
        model_file = self._config.sim_config.model_config_file
        model_dict = yaml.load(open(model_file))
        model = model_dict['model_parameters']
        # Overwrite overlapping params
        for k, v in mapped_results.items():
            routine_name, param_name = k.split('.')
            model[name_map[routine_name]][param_name] = v

        # Finally, save the update parameters on disk
        if outfile is None:
            outfile = self._config.calibrated_model_file
        else:
            if not os.path.isabs(outfile):
                outfile = os.path.join(os.path.dirname(model_file), outfile)
        print("Storing calibrated params in:", outfile)
        cls_rep_str = '!!python/name:'+model_dict['model_t'].__module__+'.'+model_dict['model_t'].__name__
        model_dict['model_t'] = cls_rep_str
        with open(outfile, "w") as out:
            out.write("# This file has been automatically generated after a calibration run\n")
            #yaml.dump(model_dict, out, default_flow_style=False)
            out.write(yaml.dump(model_dict, default_flow_style=False).replace("'"+cls_rep_str+"'",cls_rep_str))
