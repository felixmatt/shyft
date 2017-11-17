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
            self.region_model_id = arg.region_model_id
            self.dst_repo = arg.dst_repo
            self.end_state_repo = arg.end_state_repo
        else:
            super().__init__(arg.region_model_id, arg.interpolation_id, arg.get_region_model_repo(),
                             arg.get_geots_repo(), arg.get_interp_repo(), initial_state_repository=arg.get_initial_state_repo())
            self.region_model_id = arg.region_model_id
            self.region_model.initialize_cell_environment(arg.time_axis)
            self.dst_repo = arg.get_destination_repo()
            self.end_state_repo = arg.get_end_state_repo()

    def _extraction_method_1d(self,ts_info):
        c_id = ts_info['catchment_id']
        t_st, t_dt, t_n = ts_info['time_axis'].start, ts_info['time_axis'].delta_t, ts_info['time_axis'].size()
        tst = api.TsTransform()
        found_indx = np.in1d(self.region_model.catchment_id_map,c_id)
        if np.count_nonzero(found_indx) != len(c_id):
            raise ConfigSimulatorError(
                "Global catchment index {} not found.".format(
                    ','.join([str(val) for val in [i for i in c_id if i not in self.region_model.catchment_id_map]])))
        methods = {'discharge': lambda m: tst.to_average(t_st, t_dt, t_n, m.statistics.discharge(c_id))}
        return methods[ts_info['type']]

    def save_result_timeseries(self, is_forecast=False):
        for repo in self.dst_repo:
            save_list = [TsStoreItem(ts_info['uid'],self._extraction_method_1d(ts_info)) for ts_info in repo['1D_timeseries']]
            TimeseriesStore(repo['repository'], save_list).store_ts(self.region_model, is_forecast)

    def save_end_state(self):
        endstate = self.region_model.state_t.vector_t()
        self.region_model.get_states(endstate)  # get the state at end of simulation
        self.end_state_repo.put_state(self.region_model_id, self.region_model.time_axis.total_period().end,
                                             endstate, tags=None)

    def update_state(self, var='discharge', catch_id=None):
        pass


class ConfigCalibrator(simulator.DefaultSimulator):
    @property
    def calib_param_names(self):
        p_reg = self.region_model.get_region_parameter()
        return [p_reg.get_name(i) for i in range(p_reg.size())]

    def __init__(self, config):
        sim_config = config.sim_config
        super().__init__(sim_config.region_model_id,sim_config.interpolation_id,sim_config.get_region_model_repo(),
                         sim_config.get_geots_repo(), sim_config.get_interp_repo(), initial_state_repository=sim_config.get_initial_state_repo())
        if self.optimizer is None:
            raise ConfigSimulatorError("Simulator's region model {} cannot be optimized, please choose "
                                 "another!".format(self.region_model.__class__.__name__))
        self.obj_funcs = {'NSE': api.NASH_SUTCLIFFE, 'KGE': api.KLING_GUPTA}
        self.region_model.initialize_cell_environment(sim_config.time_axis)
        self.region_model_id = sim_config.region_model_id
        self.model_config_file = sim_config.model_config_file
        self.optim_method = config.optimization_method['name']
        self.optim_method_params = config.optimization_method['params']
        self.target_repo = config.get_target_repo() # copy.deepcopy(config.target_repo)
        p_min, p_max, self.p_init = self._get_params_from_dict(copy.deepcopy(config.calibration_parameters))
        self.calibrated_model_file = None
        if hasattr(config, 'calibrated_model_file'):
            self.calibrated_model_file = config.calibrated_model_file
        self.optimum_parameters = None
        self.optimizer.target_specification = self._create_target_specvect()
        self.optimizer.parameter_lower_bound = p_min
        self.optimizer.parameter_upper_bound = p_max

    def copy(self):
        raise NotImplementedError("Copying of ConfigCalibrator not supported yet.")

    @property
    def tv(self):
        return self.optimizer.target_specification

    def _get_params_from_dict(self, params_as_dict):
        p_spec = {k: self.region_model.parameter_t() for k in ['min','max','init']}
        [p.update({'init': (p['min'] + p['max']) * 0.5}) for p in params_as_dict.values() if 'init' not in p]
        self._validate_params_spec(params_as_dict)
        [p_spec[k].set([params_as_dict[name][k] for name in self.calib_param_names]) for k in ['min','max','init']]
        return [p_spec[k] for k in ['min','max','init']]

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

    def _create_target_specvect(self):
        print("Creating TargetSpecificationVector...")
        tv = api.TargetSpecificationVector()
        tst = api.TsTransform()
        cid_map = self.region_model.catchment_id_map
        for repo in self.target_repo:
            tsp = repo['repository'].read([ts_info['uid'] for ts_info in repo['1D_timeseries']], self.time_axis.total_period())
            for ts_info in repo['1D_timeseries']:
                if np.count_nonzero(np.in1d(cid_map, ts_info['catch_id'])) != len(ts_info['catch_id']):
                    raise ConfigSimulatorError("Catchment ID {} for target series {} not found.".format(
                            ','.join([str(val) for val in [i for i in ts_info['catch_id'] if i not in cid_map]]), ts_info['uid']))
                period = api.UtcPeriod(ts_info['start_datetime'],
                                       ts_info['start_datetime'] + ts_info['number_of_steps'] * ts_info['run_time_step'])
                if not self.time_axis.total_period().contains(period):
                    raise ConfigSimulatorError(
                        "Period {} for target series {} is not within the calibration period {}.".format(
                            period.to_string(), ts_info['uid'], self.time_axis.total_period().to_string()))
                #tsp = repo['repository'].read([ts_info['uid']], period)[ts_info['uid']]
                t = api.TargetSpecificationPts()
                t.uid = ts_info['uid']
                t.catchment_indexes = api.IntVector(ts_info['catch_id'])
                t.scale_factor = ts_info['weight']
                t.calc_mode = self.obj_funcs[ts_info['obj_func']['name']]
                [setattr(t, nm, ts_info['obj_func']['scaling_factors'][k]) for nm, k in zip(['s_r','s_a','s_b'], ['s_corr','s_var','s_bias'])]
                t.ts = api.TimeSeries(tst.to_average(ts_info['start_datetime'], ts_info['run_time_step'], ts_info['number_of_steps'], tsp[ts_info['uid']]))
                tv.append(t)
        return tv

    def calibrate(self, time_axis=None, state=None, optim_method=None, optim_method_params=None, p_min=None, p_max=None,
                  p_init=None, tv=None, run_interp=True, verbose_level=1):
        if time_axis is not None:
            self.region_model.initialize_cell_environment(time_axis)
            run_interp = True
        self.region_model.initial_state = self.get_initial_state_from_repo() if state is None else state
        self.optim_method = optim_method if optim_method is not None else self.optim_method
        self.optim_method_params = optim_method_params if optim_method_params is not None else self.optim_method_params
        if tv is not None:
            self.optimizer.target_specification = tv
        if p_min is not None:
            self.optimizer.parameter_lower_bound = p_min
        if p_max is not None:
            self.optimizer.parameter_upper_bound = p_max
        if p_init is not None:
            self.p_init = p_init
        self.optimizer.set_verbose_level(verbose_level)
        p_res = self._optimize(self.p_init, self.optim_method, self.optim_method_params, run_interp=run_interp)

        if self.calibrated_model_file is not None:
            self.save_calibrated_model(p_res)
        self.optimum_parameters = self.region_model.parameter_t(p_res) # To keep a copy
        return p_res

    def save_calibrated_model(self, optim_param, outfile=None):
        """Save calibrated params in a model-like YAML file."""
        name_map = {"pt": "priestley_taylor", "kirchner": "kirchner", "p_corr": "precipitation_correction",
                    "ae": "actual_evapotranspiration", "gs": "gamma_snow", "ss": "skaugen_snow", "hs": "hbv_snow","gm":"glacier_melt",
                    "ae": "hbv_actual_evapotranspiration", "soil": "hbv_soil", "tank": "hbv_tank","routing":"routing"
                    }
        model_file = self.model_config_file
        model_dict = yaml.load(open(model_file))
        model_params = {}
        [model_params[name_map[r]].update({p: getattr(getattr(optim_param, r), p)})
         if name_map[r] in model_params else model_params.update(
            {name_map[r]: {p: getattr(getattr(optim_param, r), p)}}) for r, p in [nm.split('.') for nm in self.calib_param_names]]
        model_dict.update({'model_parameters': model_params})
        # Save the update parameters on disk
        if outfile is not None:
            self.calibrated_model_file = outfile
        if not os.path.isabs(self.calibrated_model_file):
            self.calibrated_model_file = os.path.join(os.path.dirname(model_file), self.calibrated_model_file)
        print("Storing calibrated params in:", self.calibrated_model_file)
        cls_rep_str = '!!python/name:'+model_dict['model_t'].__module__+'.'+model_dict['model_t'].__name__
        model_dict['model_t'] = cls_rep_str
        with open(self.calibrated_model_file, "w") as out:
            out.write("# This file has been automatically generated after a calibration run\n")
            out.write(yaml.dump(model_dict, default_flow_style=False).replace("'"+cls_rep_str+"'",cls_rep_str))

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
        # Make copy rather than create ConfigSimulator from config to avoid get_region_model being called multiple times
        self.forecast_sim = {name: self.historical_sim.copy() for name in self.forecast_cfg}  
        for k, v in self.forecast_sim.items():
            v.geo_ts_repository = self.forecast_cfg[k].get_geots_repo()
            v.ip_repos = self.forecast_cfg[k].get_interp_repo()

    def run(self, save_end_state=True, save_result_timeseries=True):
        self.historical_sim.run()
        state = self.historical_sim.reg_model_state
        if save_result_timeseries:
            self.historical_sim.save_result_timeseries()
        if save_end_state:
            self.historical_sim.save_end_state()
        for k, v in self.forecast_sim.items():
            v.run_forecast(self.forecast_cfg[k].time_axis, self.forecast_cfg[k].time_axis.start, state)
            state = v.reg_model_state
            if save_result_timeseries:
                v.save_result_timeseries(is_forecast=True)

    def save_end_state(self):
        self.historical_sim.save_end_state()

    def save_result_timeseries(self):
        self.historical_sim.save_result_timeseries()
        for k, v in self.forecast_sim.items():
            v.save_result_timeseries(is_forecast=True)
