import numpy as np
import yaml
import os
import copy

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
            config = arg.config
        else:
            super().__init__(arg.region_model_id,arg.interpolation_id,arg.get_region_model_repo(),
                             arg.get_geots_repo(), arg.interp_repo_repo())
            config = arg

        self.region_model_id = config.region_model_id
        self.time_axis = config.time_axis
        self.dst_repo = config.get_destination_repo()
        self.initial_state_repo = config.get_initial_state_repo()
        self.end_state_repo = config.get_end_state_repo()

        self.state = self.get_initial_state()

    def _extraction_method_1d(self,ts_info):
        c_id = ts_info['catchment_id']
        t_st, t_dt, t_n = ts_info['time_axis'].start, ts_info['time_axis'].delta_t, ts_info['time_axis'].size()
        tst = api.TsTransform()
        found_indx = np.in1d(self.region_model.catchment_id_map,c_id)
        if np.count_nonzero(found_indx) != len(c_id):
            raise ConfigSimulatorError(
                "Global catchment index {} not found.".format(
                    ','.join([str(val) for val in [i for i in c_id if i not in self.region_model.catchment_id_map]])))
        #c_indx = [i for i,j in enumerate(found_indx) if j] # since ID to Index conversion not necessary
        methods = {'discharge': lambda m: tst.to_average(t_st, t_dt, t_n, m.statistics.discharge(c_id))}
        return methods[ts_info['type']]

    def save_result_timeseries(self):
        for repo in self.dst_repo:
            save_list = [TsStoreItem(ts_info['uid'],self._extraction_method_1d(ts_info)) for ts_info in repo['1D_timeseries']]
            TimeseriesStore(repo['repository'], save_list).store_ts(self.region_model)

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
                raise ConfigSimulatorError('No initial state matching criteria.')
        return self.initial_state_repo.get_state(state_id)

    def save_end_state(self):
        endstate = self.region_model.state_t.vector_t()
        self.region_model.get_states(endstate)  # get the state at end of simulation
        self.end_state_repo.put_state(self.region_model_id, self.region_model.time_axis.total_period().end,
                                             endstate, tags=None)

    def update_state(self, var='discharge', catch_id=None):
        pass

    def run(self, time_axis=None, state=None):
        if time_axis is not None:
            self.time_axis = time_axis
        if state is not None:
            self.state = state
        super().run(self.time_axis, self.state)



class ConfigCalibrator(simulator.DefaultSimulator):
    @property
    def params_as_vct(self):
        return {k: [v.get(i) for i in range(v.size())] for k, v in self.p_spec.items()}

    @property
    def calib_param_names(self):
        p_reg = self.region_model.get_region_parameter()
        return [p_reg.get_name(i) for i in range(p_reg.size())]

    def __init__(self, config):
        sim_config = config.sim_config
        super().__init__(sim_config.region_model_id,sim_config.interpolation_id,sim_config.get_region_model_repo(),
                         sim_config.get_geots_repo(), sim_config.get_interp_repo())
        self.obj_funcs = {'NSE': api.NASH_SUTCLIFFE, 'KGE': api.KLING_GUPTA}
        self.verbose_level = 1
        self.time_axis = sim_config.time_axis
        self.initial_state_repo = sim_config.get_initial_state_repo()
        self.region_model_id = sim_config.region_model_id
        self.model_config_file = sim_config.model_config_file
        self.optim_method = config.optimization_method['name']
        self.optim_method_params = config.optimization_method['params']
        self.target_repo = config.get_target_repo() # copy.deepcopy(config.target_repo)
        self.p_spec = self.get_params_from_dict(copy.deepcopy(config.calibration_parameters))
        self.calibrated_model_file = None
        if hasattr(config, 'calibrated_model_file'):
            self.calibrated_model_file = config.calibrated_model_file
        self.state = self.get_initial_state()
        self.optimum_parameters = None
        self.optimizer = None
        self.tv = None

    def init(self, time_axis=None):
        if time_axis is not None:
            self.time_axis = time_axis
        if self.tv is None:
            self._create_target_specvect()
        self._fetch_source_run_interp()

    def get_params_from_dict(self, params_as_dict):
        p_spec = {k: self.region_model.parameter_t() for k in ['min','max','init']}
        [p.update({'init': (p['min'] + p['max']) * 0.5}) for p in params_as_dict.values()]
        self._validate_params_spec(params_as_dict)
        [p_spec[k].set([params_as_dict[name][k] for name in self.calib_param_names]) for k in ['min','max','init']]
        return p_spec

    def _validate_params_spec(self, params_as_dict):
        valid_param_name = [params_as_dict.get(name, False) for name in self.calib_param_names]
        if not all(valid_param_name):
            raise ConfigSimulatorError("The following parameters were not found: {}".format(
                ','.join([name for i, name in enumerate(self.calib_param_names) if not valid_param_name[i]])))
        valid_param_spec = [all([k in ['min','max','init'] for k in params_as_dict[name]])
                            for name in self.calib_param_names]
        if not all(valid_param_spec):
            raise ConfigSimulatorError("Min, max or init spec for the following parameters are invalid: {}".format(
                ','.join([name for i, name in enumerate(self.calib_param_names) if not valid_param_spec[i]])))

    def _fetch_source_run_interp(self):
        print("Fetching sources and running interpolation...")
        bbox = self.region_model.bounding_region.bounding_box(self.epsg)
        period = self.time_axis.total_period()
        sources = self.geo_ts_repository.get_timeseries(self._geo_ts_names, period,
                                                        geo_location_criteria=bbox)
        self.region_env = self._get_region_environment(sources)

        interp_params = self.ip_repos.get_parameters(self.interpolation_id)
        self.region_model.run_interpolation(interp_params, self.time_axis, self.region_env)

    def _create_target_specvect(self):
        print("Creating TargetSpecificationVector...")
        self.tv = api.TargetSpecificationVector()
        tst = api.TsTransform()
        cid_map = self.region_model.catchment_id_map
        for repo in self.target_repo:
            for ts_info in repo['1D_timeseries']:
                if np.count_nonzero(np.in1d(cid_map, ts_info['catch_id'])) != len(ts_info['catch_id']):
                    raise ConfigSimulatorError("Catchment ID {} for target series {} not found.".format(
                            ','.join([str(val) for val in [i for i in ts_info['catch_id'] if i not in cid_map]]), ts_info['uid']))
                period = api.UtcPeriod(ts_info['start_datetime'],
                                       ts_info['start_datetime'] + ts_info['number_of_steps'] * ts_info['run_time_step'])
                if not self.time_axis.total_period().contains(period):
                    raise ConfigSimulatorError(
                        "Period {} for target series {} is not within the calibration period {}.".format(
                            period().to_string(), ts_info['uid'], self.time_axis.total_period().to_string()))
                print('Start fetching...')
                tsp = repo['repository'].read([ts_info['uid']], period)[ts_info['uid']]
                print('Finished fetching...')
                t = api.TargetSpecificationPts()
                t.uid = ts_info['uid']
                t.catchment_indexes = api.IntVector(ts_info['catch_id'])
                t.scale_factor = ts_info['weight']
                t.calc_mode = self.obj_funcs[ts_info['obj_func']['name']]
                [setattr(t, nm, ts_info['obj_func']['scaling_factors'][k]) for nm, k in zip(['s_r','s_a','s_b'], ['s_corr','s_var','s_bias'])]
                t.ts = tst.to_average(ts_info['start_datetime'], ts_info['run_time_step'], ts_info['number_of_steps'], tsp)
                self.tv.append(t)

    def calibrate(self, state=None, optim_method=None, optim_method_params=None, p_min=None, p_max=None, p_init=None):
        if state is not None:
            self.state = state
        if optim_method is not None:
            self.optim_method = optim_method
        if optim_method_params is not None:
            self.optim_method_params = optim_method_params
        [self.p_spec.update({k: p}) for k, p in zip(['min', 'max', 'init'], [p_min, p_max, p_init]) if p is not None]
        if not hasattr(self.region_model, "optimizer_t"):
            raise ConfigSimulatorError("Simulator's region model {} cannot be optimized, please choose another!".format(
                self.region_model.__class__.__name__))
        is_correct_p_type = [isinstance(self.p_spec[k], self.region_model.parameter_t) for k in ['min', 'max', 'init']]
        if not all(is_correct_p_type):
            raise ConfigSimulatorError("{} must be of type {}".format(
                ','.join([name for i, name in enumerate(['min', 'max', 'init'])
                          if not is_correct_p_type[i]]), self.region_model.parameter_t.__name__))
        self.region_model.set_states(self.state)
        self.optimizer = self.region_model.optimizer_t(self.region_model,
                                                       self.tv,
                                                       self.params_as_vct['min'],
                                                       self.params_as_vct['max'])
        self.optimizer.set_verbose_level(self.verbose_level)
        print("Calibrating...")
        if self.optim_method == "min_bobyqa":
            p_vec_opt = self.optimizer.optimize(self.params_as_vct['init'], **self.optim_method_params)
        elif self.optim_method == "dream":
            p_vec_opt = self.optimizer.optimize_dream(self.params_as_vct['init'], **self.optim_method_params)
        elif self.optim_method == "sceua":
            p_vec_opt = self.optimizer.optimize_sceua(self.params_as_vct['init'], **self.optim_method_params)
        else:
            raise ValueError("Unknown optimization method: %s" % self.optim_method)
        p_res = self.region_model.parameter_t()
        p_res.set(p_vec_opt)
        if self.calibrated_model_file is not None:
            self.save_calibrated_model(p_res)
        self.optimum_parameters = self.region_model.parameter_t() # To keep a copy
        self.optimum_parameters.set(p_vec_opt)
        return p_res

    def save_calibrated_model(self, optim_param, outfile=None):
        """Save calibrated params in a model-like YAML file."""
        name_map = {"pt": "priestley_taylor", "kirchner": "kirchner",
                    "p_corr": "precipitation_correction", "ae": "actual_evapotranspiration",
                    "gs": "gamma_snow", "ss": "skaugen_snow", "hs": "hbv_snow"}

        # Existing model parameters structure
        #model_file = self.model_config_file
        #model_dict = yaml.load(open(model_file))
        #model = model_dict['model_parameters']
        model_dict = {'model_parameters': {}}
        # Overwrite overlapping params
        #[params.update({param_name: getattr(getattr(optim_param, name_map[routine_name]), param_name)}) for routine_name, params in model.items() for param_name in params]
        [model_dict['model_parameters'][name_map[r]].update({p: getattr(getattr(optim_param, r), p)})
         if name_map[r] in model_dict['model_parameters'] else model_dict['model_parameters'].update(
            {name_map[r]: {p: getattr(getattr(optim_param, r), p)}}) for r, p in [nm.split('.') for nm in self.calib_param_names]]


        # Finally, save the update parameters on disk
        if outfile is not None:
            self.calibrated_model_file = outfile
        if not os.path.isabs(self.calibrated_model_file):
            self.calibrated_model_file = os.path.join(os.path.dirname(model_file), self.calibrated_model_file)
        print("Storing calibrated params in:", self.calibrated_model_file)
        #cls_rep_str = '!!python/name:'+model_dict['model_t'].__module__+'.'+model_dict['model_t'].__name__
        #model_dict['model_t'] = cls_rep_str
        cls_rep_str = '!!python/name:'+self.region_model.__class__.__module__+'.'+self.region_model.__class__.__name__
        model_dict['model_t'] = cls_rep_str
        with open(self.calibrated_model_file, "w") as out:
            out.write("# This file has been automatically generated after a calibration run\n")
            #yaml.dump(model_dict, out, default_flow_style=False)
            out.write(yaml.dump(model_dict, default_flow_style=False).replace("'"+cls_rep_str+"'",cls_rep_str))

    def get_initial_state(self):
        state_id = 0
        if hasattr(self.initial_state_repo, 'n'): # No stored state, generated on-the-fly
            self.initial_state_repo.n = self.region_model.size()
        else:
            states = self.initial_state_repo.find_state(
                region_model_id_criteria=self.region_model_id,
                utc_timestamp_criteria=self.time_axis.start,tag_criteria=None)
            if len(states) > 0:
                state_id = states[0].state_id #most_recent_state i.e. <= start time
            else:
                raise ConfigSimulatorError('No initial state matching criteria.')
        return self.initial_state_repo.get_state(state_id)

    def run_calibrated_model(self):
        if self.optimum_parameters is None:
            raise ConfigSimulatorError('The model has noe been calibrated.')
        optim_params_vct = [self.optimum_parameters.get(i) for i in range(self.optimum_parameters.size())]
        return 1-self.optimizer.calculate_goal_function(optim_params_vct)


class ConfigForecaster(object):
    def __init__(self, config):
        self.historical_cfg = config.sim_config
        self.forecast_cfg = config.forecast_config
        self.historical_sim = ConfigSimulator(self.historical_cfg)
        # self.forecast_sim = {k: ConfigSimulator(v) for k, v in self.forecast_cfg}  # creating ConfigSimulator from config
        self.forecast_sim = {name: self.historical_sim.copy() for name in self.forecast_cfg}  # making copy rather than creating ConfigSimulator from config to avoid get_region_model being called multiple times
        for k, v in self.forecast_sim.items():
            v.geo_ts_repository = self.forecast_cfg[k].geo_ts
            v.ip_repos = self.forecast_cfg[k].interp_repos
            v.time_axis = self.forecast_cfg[k].time_axis
            v.config = self.forecast_cfg[k]

    def run(self):
        self.historical_sim.run()
        state = self.historical_sim.reg_model_state
        for k, v in self.forecast_sim.items():
            v.run_forecast(v.time_axis, v.time_axis.start, state)
            state = v.reg_model_state

    def save_end_state(self):
        self.historical_sim.save_end_state()

    def save_result_timeseries(self):
        self.historical_sim.save_result_timeseries()
        for k, v in self.forecast_sim.items():
            v.save_result_timeseries()
