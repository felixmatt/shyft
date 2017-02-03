# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 11:30:50 2015

@author: u37855
"""
import pytz
import numpy as np
from matplotlib import pylab as plt
from matplotlib.dates import AutoDateLocator

from shyft.api import UtcPeriod
from shyft.api import TsTransform
from shyft.api import IntVector
from shyft.api import Calendar
from shyft.api import deltahours
from shyft.api import YMDhms
from shyft.api import TimeAxisFixedDeltaT
from shyft.api import TargetSpecificationPts
from shyft.api import TargetSpecificationVector
from shyft.api import KLING_GUPTA
from shyft.api.pt_gs_k import PTGSKModel
from shyft.api.pt_gs_k import PTGSKOptModel
from shyft_config import tistel
from shyft.repository.default_state_repository import DefaultStateRepository
from shyft.repository.service.gis_region_model_repository import RegionModelConfig
from shyft.repository.service.gis_region_model_repository import GisRegionModelRepository
from shyft.repository.service.ssa_smg_db import SmGTsRepository, PROD, FC_PROD
from shyft.orchestration.plotting import plot_np_percentiles
from shyft.orchestration.plotting import set_calendar_formatter
from shyft.orchestration.plotting import utc_to_greg
from shyft.orchestration.simulator import DefaultSimulator as Simulator


def create_tistel_simulator(model, geo_ts_repository):
    region_id = "Tistel-ptgsk"
    interpolation_id = 0
    cfg_list = [RegionModelConfig(region_id, model, tistel.pt_gs_k_parameters(),
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


def burn_in_state(simulator, t_start, t_stop, q_obs_m3s_ts):
    dt = deltahours(1)
    n = int(round((t_stop - t_start)/dt))
    time_axis = TimeAxisFixedDeltaT(t_start, dt, n)
    n_cells = simulator.region_model.size()
    state_repos = DefaultStateRepository(simulator.region_model.__class__, n_cells)
    simulator.run(time_axis, state_repos.get_state(0))
    # Go back in time (to t_start) and adjust q with observed discharge at that time.
    # This will give us a good initial state at t_start
    return adjust_simulator_state(simulator, t_start, q_obs_m3s_ts)


def adjust_simulator_state(sim, t, q_obs):
    return sim.discharge_adjusted_state(q_obs.value(q_obs.index_of(t)))


def construct_calibration_parameters(simulator):
    p = simulator.region_model.get_region_parameter()
    p_min = simulator.region_model.parameter_t(p)
    p_max = simulator.region_model.parameter_t(p)
    p_min.kirchner.c1 *= 0.8
    p_max.kirchner.c1 *= 1.2
    p_min.kirchner.c2 *= 0.8
    p_max.kirchner.c2 *= 1.2
    p_min.kirchner.c3 *= 0.8
    p_max.kirchner.c3 *= 1.2
    return p, p_min, p_max


def plot_results(ptgsk, q_obs=None):
    h_obs = None
    if ptgsk is not None:
        plt.subplot(3, 1, 1)
        discharge = ptgsk.region_model.statistics.discharge([0])
        temp = ptgsk.region_model.statistics.temperature([0])
        precip = ptgsk.region_model.statistics.precipitation([0])
        # Results on same time axis, so we only need one
        times = utc_to_greg([discharge.time(i) for i in range(discharge.size())])
        plt.plot(times, np.array(discharge.v))
        plt.gca().set_xlim(times[0], times[-1])
        plt.ylabel(r"Discharge in $\mathbf{m^3s^{-1}}$")
        set_calendar_formatter(Calendar())
    if q_obs is not None:
        obs_times = utc_to_greg([q_obs.time(i) for i in range(q_obs.size())])
        ovs = [q_obs.value(i) for i in range(q_obs.size())]
        h_obs, = plt.plot(obs_times, ovs, linewidth=2, color='k')
        ax = plt.gca()
        ax.set_xlim(obs_times[0], obs_times[-1])
    if ptgsk is not None:
        plt.subplot(3, 1, 2)
        plt.plot(times, np.array(temp.v))
        set_calendar_formatter(Calendar())
        plt.gca().set_xlim(times[0], times[-1])
        plt.ylabel(r"Temperature in C")
        plt.subplot(3, 1, 3)
        plt.plot(times, np.array(precip.v))
        set_calendar_formatter(Calendar())
        plt.gca().set_xlim(times[0], times[-1])
        plt.ylabel(r"Precipitation in mm")
    return h_obs


def plot_percentiles(sim, percentiles, obs=None):
    discharges = [s.region_model.statistics.discharge([0]) for s in sim]
    times = utc_to_greg(np.array([discharges[0].time(i) for i in range(discharges[0].size())], dtype='d'))
    all_discharges = np.array([d.v for d in discharges])
    perc_arrs = [a for a in np.percentile(all_discharges, percentiles, 0)]
    h, fill_handles = plot_np_percentiles(times, perc_arrs, base_color=(51/256, 102/256, 193/256))
    percentile_texts = ["{} - {}".format(percentiles[i], percentiles[-(i + 1)]) for i in range(len(percentiles)//2)]
    ax = plt.gca()
    maj_loc = AutoDateLocator(tz=pytz.UTC, interval_multiples=True)
    ax.xaxis.set_major_locator(maj_loc)
    set_calendar_formatter(Calendar())
    if len(percentiles) % 2:
        fill_handles.append(h[0])
        percentile_texts.append("{}".format(percentiles[len(percentiles)//2]))
    if obs is not None:
        h_obs = plot_results(None, obs)
        fill_handles.append(h_obs)
        percentile_texts.append("Observed")

    ax.legend(fill_handles, percentile_texts)
    ax.grid(b=True, color=(51/256, 102/256, 193/256), linewidth=0.1, linestyle='-', axis='y')
    plt.xlabel("Time in UTC")
    plt.ylabel(r"Discharge in $\mathbf{m^3s^{-1}}$", verticalalignment="top", rotation="horizontal")
    ax.yaxis.set_label_coords(0, 1.1)
    return h, ax


def forecast_demo():
    """Simple forecast demo using arome data from met.no. Initial state
    is bootstrapped by simulating one hydrological year (starting
    Sept 1. 2011), and then calculating the state August 31. 2012. This
    state is then used as initial state for simulating Sept 1, 2011,
    after scaling with observed discharge. The validity of this approach
    is limited by the temporal variation of the spatial distribution of
    the discharge state, q, in the Kirchner method. The model is then
    stepped forward until Oct 1, 2015, and then used to compute the
    discharge for 65 hours using Arome data. At last, the results
    are plotted as simple timeseries.

    """
    utc = Calendar()
    t_start = utc.time(YMDhms(2011, 9, 1))
    t_fc_start = utc.time(YMDhms(2015, 10, 1))
    dt = deltahours(1)
    n_obs = int(round((t_fc_start - t_start)/dt))
    n_fc = 65
    obs_time_axis = TimeAxisFixedDeltaT(t_start, dt, n_obs)
    fc_time_axis = TimeAxisFixedDeltaT(t_fc_start, dt, n_fc)
    total_time_axis = TimeAxisFixedDeltaT(t_start, dt, n_obs + n_fc)
    q_obs_m3s_ts = observed_tistel_discharge(total_time_axis.total_period())
    ptgsk = create_tistel_simulator(PTGSKOptModel, tistel.geo_ts_repository(tistel.grid_spec.epsg()))
    initial_state = burn_in_state(ptgsk, t_start, utc.time(YMDhms(2012, 9, 1)), q_obs_m3s_ts)
    ptgsk.run(obs_time_axis, initial_state)
    plot_results(ptgsk, q_obs_m3s_ts)

    current_state = adjust_simulator_state(ptgsk, t_fc_start, q_obs_m3s_ts)

    ptgsk_fc = create_tistel_simulator(PTGSKModel, tistel.arome_repository(tistel.grid_spec, t_fc_start))
    ptgsk_fc.run(fc_time_axis, current_state)
    plt.figure()
    q_obs_m3s_ts = observed_tistel_discharge(fc_time_axis.total_period())
    plot_results(ptgsk_fc, q_obs_m3s_ts)
    #plt.interactive(1)
    plt.show()


def ensemble_demo():
    utc = Calendar()
    t_start = utc.time(YMDhms(2011, 9, 1))
    t_fc_ens_start = utc.time(YMDhms(2015, 7, 26))
    disp_start = utc.time(YMDhms(2015, 7, 20))
    dt = deltahours(1)
    n_obs = int(round((t_fc_ens_start - t_start)/dt))
    n_fc_ens = 30
    n_disp = int(round(t_fc_ens_start - disp_start)/dt) + n_fc_ens + 24*7

    obs_time_axis = TimeAxisFixedDeltaT(t_start, dt, n_obs + 1)
    fc_ens_time_axis = TimeAxisFixedDeltaT(t_fc_ens_start, dt, n_fc_ens)
    display_time_axis = TimeAxisFixedDeltaT(disp_start, dt, n_disp)

    q_obs_m3s_ts = observed_tistel_discharge(obs_time_axis.total_period())
    ptgsk = create_tistel_simulator(PTGSKOptModel, tistel.geo_ts_repository(tistel.grid_spec.epsg()))
    initial_state = burn_in_state(ptgsk, t_start, utc.time(YMDhms(2012, 9, 1)), q_obs_m3s_ts)

    ptgsk.run(obs_time_axis, initial_state)
    current_state = adjust_simulator_state(ptgsk, t_fc_ens_start, q_obs_m3s_ts)
    q_obs_m3s_ts = observed_tistel_discharge(display_time_axis.total_period())
    ens_repos = tistel.arome_ensemble_repository(tistel.grid_spec)
    ptgsk_fc_ens = create_tistel_simulator(PTGSKModel, ens_repos)
    sims = ptgsk_fc_ens.create_ensembles(fc_ens_time_axis, t_fc_ens_start, current_state)
    for sim in sims:
        sim.simulate()
    plt.hold(1)
    percentiles = [10, 25, 50, 75, 90]
    plot_percentiles(sims, percentiles, obs=q_obs_m3s_ts)
    #plt.interactive(1)
    plt.show()


def continuous_calibration():
    utc = Calendar()
    t_start = utc.time(YMDhms(2011, 9, 1))
    t_fc_start = utc.time(YMDhms(2015, 10, 1))
    dt = deltahours(1)
    n_obs = int(round((t_fc_start - t_start)/dt))
    obs_time_axis = TimeAxisFixedDeltaT(t_start, dt, n_obs + 1)
    q_obs_m3s_ts = observed_tistel_discharge(obs_time_axis.total_period())

    ptgsk = create_tistel_simulator(PTGSKOptModel, tistel.geo_ts_repository(tistel.grid_spec.epsg()))
    initial_state = burn_in_state(ptgsk, t_start, utc.time(YMDhms(2012, 9, 1)), q_obs_m3s_ts)

    num_opt_days = 30
    # Step forward num_opt_days days and store the state for each day:
    recal_start = t_start + deltahours(num_opt_days*24)
    t = t_start
    state = initial_state
    opt_states = {t: state}
    while t < recal_start:
        ptgsk.run(TimeAxisFixedDeltaT(t, dt, 24), state)
        t += deltahours(24)
        state = ptgsk.reg_model_state
        opt_states[t] = state

    recal_stop = utc.time(YMDhms(2011, 10, 30))
    recal_stop = utc.time(YMDhms(2012, 5, 30))
    curr_time = recal_start
    q_obs_avg = TsTransform().to_average(t_start, dt, n_obs + 1, q_obs_m3s_ts)
    target_spec = TargetSpecificationPts(q_obs_avg, IntVector([0]), 1.0, KLING_GUPTA)
    target_spec_vec = TargetSpecificationVector([target_spec])
    i = 0
    times = []
    values = []
    p, p_min, p_max = construct_calibration_parameters(ptgsk)
    while curr_time < recal_stop:
        print(i)
        i += 1
        opt_start = curr_time - deltahours(24*num_opt_days)
        opt_state = opt_states.pop(opt_start)
        p = ptgsk.region_model.get_region_parameter()
        p_opt = ptgsk.optimize(TimeAxisFixedDeltaT(opt_start, dt, 24*num_opt_days), opt_state, target_spec_vec,
                               p, p_min, p_max, tr_stop=1.0e-5)
        ptgsk.region_model.set_region_parameter(p_opt)
        corr_state = adjust_simulator_state(ptgsk, curr_time, q_obs_m3s_ts)
        ptgsk.run(TimeAxisFixedDeltaT(curr_time, dt, 24), corr_state)
        curr_time += deltahours(24)
        opt_states[curr_time] = ptgsk.reg_model_state
        discharge = ptgsk.region_model.statistics.discharge([0])
        times.extend(discharge.time(i) for i in range(discharge.size()))
        values.extend(list(np.array(discharge.v)))
    plt.plot(utc_to_greg(times), values)
    plot_results(None, q_obs=observed_tistel_discharge(UtcPeriod(recal_start, recal_stop)))
    set_calendar_formatter(Calendar())
    #plt.interactive(1)
    plt.title("Continuously recalibrated discharge vs observed")
    plt.xlabel("Time in UTC")
    plt.ylabel(r"Discharge in $\mathbf{m^3s^{-1}}$", verticalalignment="top", rotation="horizontal")
    plt.gca().yaxis.set_label_coords(0, 1.1)


if __name__ == "__main__":
    import sys
    demos = [forecast_demo, ensemble_demo, continuous_calibration]
    demo = demos[int(sys.argv[1]) if len(sys.argv) == 2 else 0]
    result = demo()
