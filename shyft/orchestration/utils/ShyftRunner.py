#!/usr/bin/env python
#Preconditions
# export LD_LIBRARY_PATH= path to enki.Os/bin/Debug
# PYTHONPATH contains enki.OS/bin/Debug (ro release)

import copy
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from shyft import api
from shyft.orchestration.state import build_ptgsk_model_state_from_data
from shyft.orchestration.state import extract_ptgsk_model_state
from shyft.orchestration.state import save_state_as_yaml_file
from shyft.repository.state_repository import TimeCondition
from itertools import imap
import collections
from matplotlib import pylab as plt
from descartes import PolygonPatch
from shapely.geometry import MultiPolygon
from matplotlib.collections import PatchCollection
from shyft.orchestration.utils.CellBuilder import cell_argument_factory


def update(d, u):
    """Simple recursive update of dictionary d with u"""
    for k, v in u.iteritems():
        if isinstance(v, collections.Mapping):
            r = update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d

class ShyftRunnerError(Exception): pass

class source_info(object):
    """ 
    The source_info is a helper class to generate type specific list of sources
    like Temperature,Precipitation,Radiation that is ready for feeding directly into
    the enki::core execution engine
    """

    def __init__(self, network_name=None, source_type=None, vector_type=None, ts_mapping=None):
        self.name = network_name
        self.vector_type = vector_type
        self.source_type = source_type
        self.ts_mapping = ts_mapping
        assert network_name != None and vector_type != None and source_type != None and ts_mapping != None
        
    def timeseries_map(self, network_index):
        return self.ts_mapping[network_index]


class ShyftRunner(object):

    def __init__(self, config):
        self._config = config
        self.cell_map = [] # Used for plotting
        self._model = None
        self.end_utc_timestamp = None
        self.init_state = None
        self.get_catchment_avg_result={'SimDischarge':self._get_sum_catchment_discharge,
                                       'precipitation':self._get_sum_catchment_precipitation,
                                       'temperature':self._get_sum_catchment_temperature
                                       }

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("No model created, please call build_model first")
        return self._model
    
    def sources(self): 
        return [self._config.input_source_repository.get(k) for k in self._config.input_source_repository.find()]

    def raster(self, name):
        return self._config.cell_read_only_repository[name]

    def parameter(self, major, minor, name):
        if major in self._config.parameters and minor in self._config.parameters[major] and name in self._config.parameters[major][minor]:
            return self._config.parameters[major][minor][name]
        raise ValueError("Unknown parameter: {}.{}.{}".format(major, minor, name))
    
    def _model_values(self, model_name, *names):
        return [self.parameter("model", model_name, name) for name in names]

    def _interpolation_values(self, interpolation_name, *names):
        return [self.parameter("interpolation", interpolation_name, name) for name in names]

    def cell_data(self, name):
        return self._config.cell_read_only_repository.get(name)

    def cell_data_types(self):
        return self._config.cell_read_only_repository.find()

    def model_parameters_dict(self):
        #TODO: Replace with polymorphism
        if self._config.model in [api.PTGSKModel, api.PTGSKOptModel]:
            priestley_taylor = self._model_values("priestley_taylor", "albedo", "alpha")
            gamma_snow = self._model_values("gamma_snow", "winter_end_day_of_year", "initial_bare_ground_fraction", "snow_cv", "snow_tx",
                                                "wind_scale", "wind_const", "max_water", "surface_magnitude", "max_albedo", "min_albedo",
                                                "fast_albedo_decay_rate", "slow_albedo_decay_rate", "snowfall_reset_depth", "glacier_albedo")
            act_evap = self._model_values("actual_evapotranspiration", "scale_factor")
            kirchner = self._model_values("kirchner", "c1", "c2", "c3")
            cell = []

            return {"priestley_taylor": priestley_taylor, "gamma_snow": gamma_snow, "act_evap": act_evap, "kirchner": kirchner, "cell": cell}
        raise ValueError("Unknown model: {}".format(self._config.model))

    def api_model_parameters(self):
        params = self.model_parameters_dict()
        pt_params = api.PriestleyTaylorParameter(*params["priestley_taylor"])
        gs_params = api.GammaSnowParameter(*params["gamma_snow"])
        ae_params = api.ActualEvapotranspirationParameter(*params["act_evap"])
        k_params = api.KirchnerParameter(*params["kirchner"])
        p_params = api.PrecipitationCorrectionParameter() #TODO; default 1.0, is it used ??
        return api.PTGSKParameter(pt_params, gs_params, ae_params, k_params, p_params)

    def interpolation_parameters(self):
        btk_param = api.BTKParameter(*self._interpolation_values("btk", "gradient", "gradient_sd", "sill", "nugget", "range", "zscale"))
        idw_arguments = self._interpolation_values("idw", "precipitation_gradient", "max_members", "max_distance")
        prec_param = api.IDWPrecipitationParameter(*idw_arguments)
        idw_arguments.pop(0) # To remove parameter 'precipitation_gradient'
        ws_param = api.IDWParameter(*idw_arguments)
        rad_param = api.IDWParameter(*idw_arguments)
        rel_hum_param = api.IDWParameter(*idw_arguments)
        return api.InterpolationParameter(btk_param, prec_param, ws_param, rad_param, rel_hum_param)

    def state_data(self, condition=None, tags=None):
        """Return a valid state from the state repository. TODO: Implement more advanced filtering logic directly in repository?"""
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
        """Exract a model state and return it together with the timestamp. TODO: make it unaware of ptgsk!"""
        data = self.state_data(condition, tags)
        #print data.state_list
        return build_ptgsk_model_state_from_data(data.state_list), data.utc_timestamp

    def sources(self):
        data_types = ["temperature", "precipitation", "radiation", "wind_speed", "relative_humidity"]
        result = {d_type: self._config.input_source_repository.get(d_type) for d_type in data_types}
        for d_type in result:
            if len(result[d_type]) == 0:
                result[d_type] = None # TODO: if wind_speed/relative_humidity, then create a ts and use data constant_windspeed,etc. 
        return result

    def add_to_cell_map(self, cell):
        x, y = cell.geo.mid_point().x, cell.geo.mid_point().y
        idx_x = int(x*self._config.n_x/(self._config.bounding_box[1] - self._config.bounding_box[0]))
        idx_y = int(y*self._config.n_y/(self._config.bounding_box[3] - self._config.bounding_box[2]))
        self.cell_map.append((idx_x, idx_y))

    def build_cells(self):
        if not self._config.model in [api.PTGSKModel, api.PTGSKOptModel]:
            raise RuntimeError("Unknown model: {}".format(self._config.model))
        catchment_id = self.cell_data("catchment_id")
        geo_position = self.cell_data("geo_position")
        lake_fraction = self.cell_data("lake_fraction")
        reservoir_fraction = self.cell_data("reservoir_fraction")
        forest_fraction = self.cell_data("forest_fraction")
        glacier_fraction= self.cell_data("glacier_fraction")
        area = self.cell_data("area")
        condition = TimeCondition() <= self._config.t_start
        state, _ = self.model_state(condition=condition)
        self.init_state = state
        alpha = self.parameter("model", "priestley_taylor", "alpha")
        c1 = self.parameter("model", "kirchner", "c1")
        c2 = self.parameter("model", "kirchner", "c2")
        c3 = self.parameter("model", "kirchner", "c3")
        snow_cv = self.parameter("model", "gamma_snow", "snow_cv")
        ibf = self.parameter("model", "gamma_snow", "initial_bare_ground_fraction")
        arg_dict = {
            "gamma_snow": {
                "initial_bare_ground_fraction": ibf, 
                "snow_cv": snow_cv
            },
            "priestley_taylor": {
                "alpha": alpha
            },
            "kirchner": {
                "c1": c1,
                "c2": c2,
                "c3": c3
            }
        }
        overrides = self._config.region_config.parameter_overrides
        num_cells = len(catchment_id)
        cells = self._config.model.cell_t.vector_t()
        cells.reserve(num_cells)
        # Cell assembly
        catchment_map = []
        catchment_parameters=[]
        for i in xrange(num_cells):
            c_id = catchment_id[i]
            if not c_id == 0:
                if not c_id in catchment_map:
                    catchment_map.append(c_id)
                    mapped_catchment_id = len(catchment_map) - 1
                    args_dict = copy.deepcopy(arg_dict)
                    args_dict = update(args_dict, overrides.get(c_id, {}))
                    catchment_parameters.append(args_dict)
                else:
                    mapped_catchment_id = catchment_map.index(c_id)
                ltf=api.LandTypeFractions()
                ltf.set_fractions(glacier_fraction[i],lake_fraction[i],reservoir_fraction[i],forest_fraction[i])
                mid_point=api.GeoPoint(geo_position[i][0],geo_position[i][1],geo_position[i][2])
                radiation_slope_factor=0.9 # TODO: read from config
                geo=api.GeoCellData(mid_point,area[i],mapped_catchment_id,radiation_slope_factor,ltf)
                cell = self._config.model.cell_t()
                cell.geo=geo;
                cell.state=state[i]
                cells.append(cell)
                self.add_to_cell_map(cell)
        self.catchment_map = catchment_map
        self.catchment_parameters=catchment_parameters
        return cells

    def _create_constant_geo_ts(self,geoTsType,geo_point,utc_period,value):
        """ creates a time point ts, with one value at the start of the supplied utc_period """
        tv=api.UtcTimeVector()
        tv.push_back(utc_period.start)
        vv=api.DoubleVector()
        vv.push_back(value)
        cts=api.TsFactory().create_time_point_ts(utc_period,tv,vv)
        return geoTsType(geo_point,cts)

    def _get_region_environment(self,time_axis,a_default_geo_point):
        """ fetches sources, and fills in and returns api.ARegionEnvironment that contains all the input sources,
            temp, precip etc, 
            rel_hum/wind_speed gets defaults to one constant source at a_default_geo_point if not supplied in sources()
        """
        sources = self.sources()
        region_env= api.ARegionEnvironment()
        region_env.temperature=sources["temperature"]
        region_env.precipitation= sources["precipitation"]
        region_env.radiation=sources["radiation"]

        #Special: if no wind_speed etc. replace with a 'fake ts' with one point *self._model_values("data", "constant_wind_speed", "constant_relative_humidity")
        wind_speed=sources["wind_speed"]
        if wind_speed == None:
            constant_wind_speed= self.parameter("model","data","constant_wind_speed")
            if constant_wind_speed == None:
                constant_wind_speed= 2.0
            wind_speed=api.WindSpeedSourceVector()
            wind_speed.push_back(self._create_constant_geo_ts(api.WindSpeedSource,a_default_geo_point,time_axis.total_period(),constant_wind_speed))
            
        region_env.wind_speed=wind_speed
        rel_hum=sources["relative_humidity"]

        if rel_hum == None:
            constant_rel_hum= self.parameter("model","data","constant_relative_humidity")
            if constant_rel_hum==None:
                constant_rel_hum=0.7
            rel_hum=api.RelHumSourceVector()
            rel_hum.push_back(self._create_constant_geo_ts(api.RelHumSource,a_default_geo_point,time_axis.total_period(),constant_rel_hum))

        region_env.rel_hum=rel_hum
        return region_env

    def build_model(self, t_start, delta_t, n_steps):
        """ """
        cells = self.build_cells() # first build the cells
        model_parameter = self.api_model_parameters() # get the region model parameters 
        self._model = self._config.model(model_parameter,cells)
        #TODO: loop through self.catchment_parameters[] and call
        #      self._model.set_catchment_parameter(zero_based_id, catchment_parameters[i])
        #   .. but only if there are 'real' overrides..

        time_axis = api.Timeaxis(t_start, delta_t, n_steps)
        self.end_utc_timestamp = t_start + delta_t*n_steps
        region_env=self._get_region_environment(time_axis,cells[0].geo.mid_point())
        
        self._model.run_interpolation(self.interpolation_parameters(), time_axis,region_env)

    def run_model(self, *args, **kwargs):
        self.model.run_cells() 
        self.save_state()
        self.save_result_timeseries()
    
    def save_result_timeseries(self):
        #enki_ts_factory=api.TsFactory()
        enki_catchment_result=dict()
        destinations = self._config.destinations
        for repository in destinations:
            for catch_res in repository['targets']:
                print catch_res
                #result = self.get_catchment_avg_result[catch_res['type']](catch_res['catchment_id'])
                #if ('time_axis' in catch_res.keys()):
                if (catch_res['time_axis'] != None):
                    ts_start = int(round((catch_res['time_axis']['start_datetime'] - datetime.utcfromtimestamp(0)).total_seconds()))
                    ts_dt = catch_res['time_axis']['time_step_length']
                    ts_nsteps = catch_res['time_axis']['number_of_steps']
                else:
                    ts_start = self._config.t_start
                    ts_dt = self._config.dt
                    ts_nsteps = self._config.n_steps                   
                result = self.get_sum_catchment_result(catch_res['type'],catch_res['catchment_id'],
                                                       ts_start=ts_start, ts_dt=ts_dt, ts_nsteps=ts_nsteps)
                #enki_result_ts = enki_ts_factory.create_point_ts(result),ts_start,ts_dt,api.DoubleVector([val for val in result]))
                enki_catchment_result[catch_res['uid']] = result
            #print "saving disabled for now, due to testing"
            constructor = repository["repository"][0]
            arg = repository["repository"][1]
            ts_repository = constructor(arg)
            ts_repository.store(enki_catchment_result) 
        
    def save_state(self):
        state = extract_ptgsk_model_state(self.model)
        state.utc_timestamp = self.end_utc_timestamp
        self._config.state_saver(state, *self._config.state_saver_args)

    def _get_sum_catchment_discharge(self, indx_list):
        return self.model.statistics.discharge(api.IntVector(indx_list))
        
    def _get_sum_catchment_precipitation(self, indx_list):
        return self.model.statistics.precipitation(api.IntVector(indx_list))
        
    def _get_sum_catchment_temperature(self, indx_list):
        return self.model.statistics.temperature(api.IntVector(indx_list))

    def get_sum_catchment_result(self, var_type, i_list, ts_start=None, ts_dt=None, ts_nsteps=None):
        if None in [ts_start,ts_dt,ts_nsteps]:
            ts_start=self._config.t_start
            ts_dt=self._config.dt
            ts_nsteps=self._config.n_steps
        #enki_ts_factory=api.TsFactory()
        tst=api.TsTransform()
        found_indx = np.in1d(self.catchment_map,i_list)
        if np.count_nonzero(found_indx)==len(i_list):
            result_ts = self.get_catchment_avg_result[var_type]([i for i,j in enumerate(found_indx) if j])
            #enki_ts = enki_ts_factory.create_point_ts(result_ts.size(),self._config.t_start,self._config.dt,result_ts.v)            
            return tst.to_average(ts_start,ts_dt,ts_nsteps,result_ts) 
        else:
            raise ShyftRunnerError(
            "Global catchment index {} not found.".format(','.join([str(val) for val in [i for i in i_list if i not in self.catchment_map]])))

    def plot_distributed_data(self, shapes, cells, extractor):
        n = len(extractor)
        fig, axes = plt.subplots(n)
        if n == 1:
            axes = [axes]
        for idx, name in enumerate(extractor.keys()):
            cmap = plt.get_cmap("jet")
            values = np.array([extractor[name](cell) for cell in cells])
            print "Plotting {} with min {} and max {}".format(name, min(values), max(values))
            values -= min(values)
            if max(values) != 0.0:
                values /= max(values)
            patches = []
            for shape, value in zip(shapes, values):
                pp_kwargs = {"fc": cmap(int(round(value*cmap.N))), "linewidth": 0.1}
                if isinstance(shape, MultiPolygon):
                    patches.extend([PolygonPatch(p, **pp_kwargs) for p in shape])
                else:
                    patches.append(PolygonPatch(shape, **pp_kwargs))
            geom = self._config.bounding_box
            axes[idx].set_aspect('equal')
            axes[idx].add_collection(PatchCollection(patches, match_original=True))
            axes[idx].set_xlim(geom[0], geom[1])
            axes[idx].set_ylim(geom[2], geom[3])
            axes[idx].set_title(name)
        plt.show()
        
 
class ShyftCalibrator(object):

    def __init__(self, config):
        self._config = config
        self._model_config = config.model_config
        self.runner = ShyftRunner(self._model_config)
        self.calibrator = None
        self.target_ts = {}
        self.ts_minfo = {}
        self.tv=None
        self.obj_funcs = {'NSE':api.NASH_SUTCLIFFE,'KGE':api.KLING_GUPTA}

    def init(self, time_axis):
        self._load_target_spec_input()
        self._fetch_target_timeseries()
        print self.target_ts
        self.runner.build_model(time_axis.start(), time_axis.delta(), time_axis.size())
        self.calib_param_names = [self.param_accessor.get_name(i) for i in range(self.param_accessor.size())]
        if self.tv is None:
            self._create_target_SpecVect()
        self.calibrator = self._config.calibration_type(self.runner.model, self.tv, api.DoubleVector(self.p_min), api.DoubleVector(self.p_max))
        self.calibrator.set_verbose_level(1) # To control console print out during calibration
        print "Calibrator catchment index = {}".format(self._config.catchment_index)
    @property
    def param_accessor(self):
        return self.runner.model.get_region_parameter()
    @property
    def p_min(self):
        return [self._config.calibration_parameters[name]['min'] for name in self.calib_param_names]

    @property
    def p_max(self):
        return [self._config.calibration_parameters[name]['max'] for name in self.calib_param_names]

    def calibrate(self, p_init=None, tol=1.0e-5):
        print "Calibrating"
        if p_init is None:
            #p_init=api.DoubleVector([(a + b)*0.5 for a,b in zip(self.p_min, self.p_max)])
            p_init=api.DoubleVector([a+(b-a)*0.5 for a,b in zip(self.p_min, self.p_max)])
        n_iterations = 1500
        return [p for p in self.calibrator.optimize(p_init,n_iterations, 0.1, tol)]
    
    def calculate_goal_function(self,optim_param_list):
        """ calls calibrator with parameter vector"""
        self.calibrator.set_verbose_level(0)
        return self.calibrator.calculate_goal_function(api.DoubleVector(optim_param_list))
        
    def _load_target_spec_input(self):
        catch_indices = {catch['internal_id']:catch['catch_id'] for catch in self._config.catchment_index}
        for repository in self._config.target:
            for target in repository['1D_timeseries']:
                ID = target['internal_id']
                if ID in catch_indices.keys():
                    spec={}
                    #self.target_ts_minfo[target['internal_id']]={k: target[k] for k in (target.keys()) if k != 'internal_id'}
                    self.ts_minfo[ID]={k: target[k] for k in (target.keys()) if k in ['uid','weight','obj_func']}
                    # Do some checks here on whether each target-period is within the run-period
                    spec['start_t'] = int(round((target['start_datetime'] - datetime.utcfromtimestamp(0)).total_seconds()))
                    spec['dt'] = target['time_step_length']
                    spec['nsteps'] = target['number_of_steps']
                    spec['catch_indx'] = catch_indices[ID]
                    self.ts_minfo[target['internal_id']].update(spec)
        print self.ts_minfo
        
    def _fetch_target_timeseries(self):
        targets = self._config.target
        for repository in targets:
            constructor = repository["repository"][0]
            arg = repository["repository"][1]
            ts_repository = constructor(arg)
            for target in repository['1D_timeseries']:
                ID = target['internal_id']
                if ID in self.ts_minfo.keys():
                    ts_info=self.ts_minfo[ID]
                    period = api.UtcPeriod(ts_info['start_t'],
                                           ts_info['start_t']+ts_info['nsteps']*ts_info['dt'])
                    self.target_ts.update(ts_repository.read([target['uid']],period))
            
    def _create_target_SpecVect(self):
        self.tv=api.TargetSpecificationVector()
        tst=api.TsTransform()            
        for ID,ts_info in self.ts_minfo.items():
            mapped_indx = [i for i,j in enumerate(self.runner.catchment_map) if j in ts_info['catch_indx']]
            catch_indx = api.IntVector(mapped_indx)
            tsp = self.target_ts[ts_info['uid']]
            #t=api.TargetOptSpecification()
            t=api.TargetSpecificationPts()
            t.catchment_indexes=catch_indx
            t.scale_factor=ts_info['weight']
            #t.calc_mode=api.NASH_SUTCLIFFE
            t.calc_mode=self.obj_funcs[ts_info['obj_func']['name']]
            t.s_r=ts_info['obj_func']['scaling_factors']['s_corr']
            t.s_a=ts_info['obj_func']['scaling_factors']['s_var']
            t.s_b=ts_info['obj_func']['scaling_factors']['s_bias']
            tsa= tst.to_average(ts_info['start_t'],ts_info['dt'],ts_info['nsteps'],tsp)
            #tsa= tst.to_average_staircase(ts_info['start_t'],ts_info['dt'],ts_info['nsteps'],tsp)
            t.ts=tsa
            #-To avoid any kind of averaging-----
            #for i in range(len(t.ts)):
            #    t.ts.set(i,tsp.value(i))
            #------------------------------------
            self.tv.push_back(t)
            print ID,ts_info['uid'],mapped_indx

        

def _main_runner(config_file):
    print 'Starting runner'
    from shyft.orchestration.utils.ShyftConfig import ShyftConfig
    
    config = ShyftConfig(config_file)
    simulator = ShyftRunner(config)
    simulator.build_model(config.t_start, config.dt, config.n_steps)
    simulator.run_model()
    
    if "shapes" in simulator.cell_data_types():
        extractor = {'Total discharge': lambda x: x.response.total_discharge,
                     'Snow storage': lambda x: x.response.gs.storage*(1.0 - (x.lake_frac + x.reservoir_frac)),
                     'Temperature': lambda x: x.temperature[len(x.temperature)-1],
                     'Precipitation': lambda x: x.precipitation[len(x.precipitation)-1]}
        simulator.plot_distributed_data(simulator.cell_data("shapes"), simulator.model.get_cells(), extractor)

    print "Exit.."

def make_fake_target(config, time_axis, catchment_index):
    print "Fake target Catchment index = {}".format(catchment_index)
    tv=api.TargetSpecificationVector()
    t=api.TargetSpecificationPts()
    simulator = ShyftRunner(config)
    simulator.build_model(time_axis.start(), time_axis.delta(), time_axis.size())
    simulator.run_model()
    ts = simulator.get_sum_catchment_result('SimDischarge',catchment_index)
    mapped_indx = [i for i,j in enumerate(simulator.catchment_map) if j in catchment_index]
    catch_indx = api.IntVector(mapped_indx)
    t.catchment_indexes=catch_indx
    t.scale_factor=1.0
    t.calc_mode=api.NASH_SUTCLIFFE
    t.ts=ts
    tv.push_back(t)
    return tv

def _main_calibration_runner(config_file):
    print 'Starting calibration runner'
    from shyft.orchestration.utils.ShyftConfig import CalibrationConfig
    config = CalibrationConfig(config_file)
    t_start = config.model_config.t_start
    delta_t = config.model_config.dt
    n_steps = config.model_config.n_steps
   # time_axis = api.FixedIntervalTimeAxis(t_start, delta_t, n_steps)
    time_axis = api.TimeAxis(t_start, delta_t, n_steps)
    config._target = make_fake_target(config.model_config, time_axis, config.catchment_index[0]['catch_id'])
    calibrator = ShyftCalibrator(config)
    calibrator.init(time_axis)
    print calibrator.calibrate(tol=1.0e-5)
    print "Exit.."

if __name__ == "__main__":
    import sys
    import os
    from shyft.orchestration.utils.ShyftRunner import ShyftRunner
    from shyft.orchestration.utils.ShyftConfig import ShyftConfig
    enki_root=os.path.join("D:\\","Users","sih","enki_config_for_test")
    config_file = os.path.join(enki_root, "runner_configurations.yaml")
    config= ShyftConfig(config_file,'NeaNidelva')
    simulator = ShyftRunner(config)
    simulator.build_model(config.t_start, config.dt, config.n_steps)
    simulator.run_model()
    discharge_0=simulator.get_calculated_discharge(38)
    if "shapes" in simulator.cell_data_types():
        extractor = {'Total discharge': lambda x: x.response.total_discharge,
                     'Snow storage': lambda x: x.response.gs.storage*(1.0 - (x.lake_frac + x.reservoir_frac)),
                     'Temperature': lambda x: x.temperature[len(x.temperature)-1],
                     'Precipitation': lambda x: x.precipitation[len(x.precipitation)-1]}
        simulator.plot_distributed_data(simulator.cell_data("shapes"), simulator.model.get_cells(), extractor)
    print "Exit.."
    #default_config_file = os.path.join(os.path.dirname(__file__), "config\NeaNidelva_calibration.yaml")
    #filename = sys.argv[1] if len(sys.argv) == 2 else default_config_file
    #_main_calibration_runner(filename)
