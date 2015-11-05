# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 11:30:50 2015

@author: u37855
"""
import numpy as np
from matplotlib import pylab as plt

from shyft.api import Calendar
from shyft.api import deltahours
from shyft.api import YMDhms
from shyft.api import Timeaxis
from shyft.api.pt_gs_k import PTGSKModel
from shyft_config import tistel
from shyft.repository.default_state_repository import DefaultStateRepository
from shyft.repository.service.gis_region_model_repository import RegionModelConfig
from shyft.repository.service.gis_region_model_repository import GisRegionModelRepository
from shyft.orchestration.simulator import SimpleSimulator as Simulator
from shyft.repository.service.ssa_smg_db import SmGTsRepository, PROD, FC_PROD


def create_tistel_simulator(geo_ts_repository):
    region_id = "Tistel-ptgsk"
    interpolation_id = 0
    cfg_list = [RegionModelConfig(region_id, PTGSKModel, tistel.pt_gs_k_parameters(),
                                  tistel.grid_spec, tistel.gis_table_def[0],
                                  tistel.gis_table_def[1], tistel.catchment_id_list)]
    reg_model_repository = GisRegionModelRepository({x.name: x for x in cfg_list})
    ptgsk = Simulator(region_id, interpolation_id, reg_model_repository,
                      geo_ts_repository,
                      tistel.interpolation_repository, None)
    return ptgsk


def observed_tistel_discharge(period):
    smg_ts_repository = SmGTsRepository(PROD, FC_PROD)
    result = smg_ts_repository.read([u"/Vikf-Tistel........-T1054A3KI0108"], period)
    return next(iter(result.values()))


def burn_in_state(t_start, t_stop, q_obs_m3s_ts):
    ptgsk = create_tistel_simulator(tistel.geo_ts_repository(tistel.grid_spec.epsg()))
    dt = deltahours(1)
    n = int(round((t_stop - t_start)/dt))
    time_axis = Timeaxis(t_start, dt, n)
    n_cells = ptgsk.region_model.size()
    state_repos = DefaultStateRepository(PTGSKModel, n_cells)
    ptgsk.run(time_axis, state_repos.get_state(0))
    # Go back in time (to t_start) and adjust q with observed discharge at that time.
    # This will give us a good initial state at t_start
    return ptgsk, tistel_tukle_state(ptgsk, ptgsk.reg_model_state(), t_start, q_obs_m3s_ts)


def tistel_tukle_state(sim, state, t, q_obs):
    q_obs_m3s = q_obs.value(q_obs.index_of(t))
    return sim.discharge_adjusted_state(q_obs_m3s, state)


def plot_results(ptgsk, q_obs_m3s_ts):
    discharge = ptgsk.region_model.statistics.discharge([0])
    temp = ptgsk.region_model.statistics.temperature([0])
    precip = ptgsk.region_model.statistics.precipitation([0])

    # Results on same time axis, so we only need one
    ts = [discharge.time(i) for i in range(discharge.size())]
    
    plt.subplot(3,1,1)
    plt.plot(ts, np.array(discharge.v))
    plt.hold(1)
    ots = [q_obs_m3s_ts.time(i) for i in range(q_obs_m3s_ts.size())]
    ovs = [q_obs_m3s_ts.value(i) for i in range(q_obs_m3s_ts.size())]
    plt.plot(ots, ovs)
    plt.legend(["Simulated Discharge", "Observed Discharge"])
    plt.subplot(3, 1, 2)
    plt.plot(ts, np.array(temp.v))
    plt.subplot(3, 1, 3)
    plt.plot(ts, np.array(precip.v))
    plt.interactive(1)
    plt.show()
    return ptgsk


def demo_tistel_forecast():
    """Now, with Burn In (... and tukle). Stay tuned"""
    utc = Calendar()
    t_start = utc.time(YMDhms(2011, 9, 1))
    t_fc_start = utc.time(YMDhms(2015, 10, 1))
    dt = deltahours(1)
    n_obs = int(round((t_fc_start - t_start)/dt))
    n_fc = 65
    obs_time_axis = Timeaxis(t_start, dt, n_obs)
    fc_time_axis = Timeaxis(t_fc_start, dt, n_fc)
    total_time_axis = Timeaxis(t_start, dt, n_obs + n_fc)      
    q_obs_m3s_ts = observed_tistel_discharge(total_time_axis.total_period())
    ptgsk, initial_state = burn_in_state(t_start, utc.time(YMDhms(2012, 9, 1)), q_obs_m3s_ts)
    ptgsk.run(obs_time_axis, initial_state)
    plot_results(ptgsk, q_obs_m3s_ts)  
    current_state = tistel_tukle_state(ptgsk, ptgsk.reg_model_state(), t_fc_start, q_obs_m3s_ts)
    ptgsk_fc = create_tistel_simulator(tistel.arome_repository(tistel.grid_spec, t_fc_start))
    ptgsk_fc.run(fc_time_axis, current_state)
    plt.figure()
    q_obs_m3s_ts = observed_tistel_discharge(fc_time_axis.total_period())
    plot_results(ptgsk_fc, q_obs_m3s_ts)
    
if __name__ == "__main__":
    demo_tistel_forecast()