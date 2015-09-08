"""
Calibrator class for running an SHyFT calibration.
"""

from __future__ import print_function
from __future__ import absolute_import

import os
import yaml

from shyft import api
from shyft  import pt_gs_k
from .utils import utctime_from_datetime2
from .base_config import target_constructor


class Calibrator(object):

    @property
    def param_accessor(self):
        return self.runner.model.get_region_parameter()

    @property
    def p_min(self):
        return api.DoubleVector([self._config.calibration_parameters[name]['min']
                                 for name in self.calib_param_names])

    @property
    def p_max(self):
        return api.DoubleVector([self._config.calibration_parameters[name]['max']
                                 for name in self.calib_param_names])

    def __init__(self, config):
        from .simulator import Simulator
        self._config = config
        self._model_config = config.model_config
        self.runner = Simulator(self._model_config)
        self.calibrator = None
        self.target_ts = {}
        self.ts_minfo = {}
        self.tv = None
        self.obj_funcs = {'NSE': api.NASH_SUTCLIFFE, 'KGE': api.KLING_GUPTA}

    def init(self, time_axis):
        self._load_target_spec_input()
        self._fetch_target_timeseries()
        self.runner.build_model(time_axis.start(), time_axis.delta(), time_axis.size())
        self.calib_param_names = [self.param_accessor.get_name(i) for i in range(self.param_accessor.size())]
        if self.tv is None:
            self._create_target_specvect()
        calibration_type = getattr(pt_gs_k, self._config.calibration_type)
        self.calibrator = calibration_type(self.runner.model, self.tv, self.p_min, self.p_max)
        self.calibrator.set_verbose_level(1)  # To control console print out during calibration
        print("Calibrator catchment index = {}".format(self._config.catchment_index))

    def calibrate(self, p_init=None, tol=1.0e-8):
        print("Calibrating...")
        if p_init is None:
            # p_init = [(a + b) * 0.5 for a, b in zip(self.p_min, self.p_max)]
            p_init = [a + (b - a) * 0.5 for a, b in zip(self.p_min, self.p_max)]
        n_iterations = 1500
        results = [p for p in self.calibrator.optimize(api.DoubleVector(p_init), n_iterations, 0.1, tol)]
        mapped_results = dict(zip(self.calib_param_names, results))
        return mapped_results

    def save_calibrated_model(self, outfile, mapped_results):
        """Save calibrated params in a model-like YAML file."""
        param_map = {
            'c1': ('kirchner', 'c1'),
            'c2': ('kirchner', 'c2'),
            'c3': ('kirchner', 'c3'),
            'ae_scale_factor': ('actual_evapotranspiration', 'scale_factor'),
            'TX': ('gamma_snow', 'snow_tx'),
            'wind_scale': ('gamma_snow', 'wind_scale'),
            'max_water': ('gamma_snow', 'max_water'),
            'wind_const': ('gamma_snow', 'wind_const'),
            'fast_albedo_decay_rate': ('gamma_snow', 'fast_albedo_decay_rate'),
            'slow_albedo_decay_rate': ('gamma_snow', 'slow_albedo_decay_rate'),
            'surface_magnitude': ('gamma_snow', 'surface_magnitude'),
            'max_albedo': ('gamma_snow', 'max_albedo'),
            'min_albedo': ('gamma_snow', 'min_albedo'),
            'snowfall_reset_depth': ('gamma_snow', 'snowfall_reset_depth'),
            'snow_cv': ('gamma_snow', 'snow_cv'),
            'glacier_albedo': ('gamma_snow', 'glacier_albedo'),
            'p_corr_scale_factor': ('p_corr_scale_factor',),
        }

        # Existing model parameters structure
        model_file = self._model_config._config_file
        model_dict = yaml.load(open(model_file))
        model = model_dict['parameters']['model']
        # Overwrite overlapping params
        for opt_param, model_param in param_map.iteritems():
            if len(model_param) == 2:
                model[model_param[0]][model_param[1]] = mapped_results[opt_param]
            elif len(model_param) == 1:
                model[model_param[0]] = mapped_results[opt_param]
            else:
                raise ValueError("Unrecognized model_param format")

        # Finally, save the update parameters on disk
        outfile = os.path.join(os.path.dirname(model_file), outfile)
        print("Storing calibrated params in:", outfile)
        with open(outfile, "w") as out:
            out.write("# This file has been automatically generated after a calibration run\n")
            yaml.dump(model_dict, out, default_flow_style=False)

    def calculate_goal_function(self, optim_param_list):
        """ calls calibrator with parameter vector"""
        self.calibrator.set_verbose_level(0)
        return self.calibrator.calculate_goal_function(api.DoubleVector(optim_param_list))

    def _load_target_spec_input(self):
        catch_indices = {catch['internal_id']: catch['catch_id'] for catch in self._config.catchment_index}
        for repository in self._config.target:
            for target in repository['1D_timeseries']:
                ID = target['internal_id']
                if ID in catch_indices:
                    spec = {}
                    # self.target_ts_minfo[target['internal_id']]={k: target[k] for k in (target.keys()) if k != 'internal_id'}
                    self.ts_minfo[ID] = {k: target[k] for k in (target.keys()) if k in ['uid', 'weight', 'obj_func']}
                    # Do some checks here on whether each target-period is within the run-period
                    spec['start_datetime'] = utctime_from_datetime2(target['start_datetime'])
                    spec['run_time_step'] = target['run_time_step']
                    spec['number_of_steps'] = target['number_of_steps']
                    spec['catch_indx'] = catch_indices[ID]
                    self.ts_minfo[ID].update(spec)

    def _fetch_target_timeseries(self):
        targets = self._config.target
        for repository in targets:
            ts_repository = target_constructor(repository, self._config)
            for target in repository['1D_timeseries']:
                ID = target['internal_id']
                if ID in self.ts_minfo:
                    ts_info = self.ts_minfo[ID]
                    period = (
                        ts_info['start_datetime'],
                        ts_info['start_datetime'] + ts_info['number_of_steps'] * ts_info['run_time_step'])
                    self.target_ts.update(ts_repository.fetch_id(ID, [target['uid']], period))

    def _create_target_specvect(self):
        self.tv = api.TargetSpecificationVector()
        tst = api.TsTransform()
        for ID, ts_info in self.ts_minfo.items():
            mapped_indx = [i for i, j in enumerate(self.runner.catchment_map) if j in ts_info['catch_indx']]
            catch_indx = api.IntVector(mapped_indx)
            tsp = self.target_ts[ts_info['uid']]
            t = api.TargetSpecificationPts()
            t.catchment_indexes = catch_indx
            t.scale_factor = ts_info['weight']
            # t.calc_mode=api.NASH_SUTCLIFFE
            t.calc_mode = self.obj_funcs[ts_info['obj_func']['name']]
            t.s_r = ts_info['obj_func']['scaling_factors']['s_corr']
            t.s_a = ts_info['obj_func']['scaling_factors']['s_var']
            tsa = tst.to_average(ts_info['start_datetime'], ts_info['run_time_step'], ts_info['number_of_steps'], tsp)
            t.ts = tsa
            self.tv.push_back(t)
            print(ID, ts_info['uid'], mapped_indx)
