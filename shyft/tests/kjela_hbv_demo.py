# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 11:30:50 2015

@author: Sigbjørn Helset
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
from shyft.api import Timeaxis
from shyft.api import TargetSpecificationPts
from shyft.api import TargetSpecificationVector
from shyft.api import KLING_GUPTA

from shyft.api.pt_gs_k import PTGSKModel
from shyft.api.pt_gs_k import PTGSKOptModel

from shyft.api.pt_hs_k import PTHSKModel
from shyft.api.pt_hs_k import PTHSKOptModel

# from shyft_config import tistel
from shyft.repository.default_state_repository import DefaultStateRepository
from shyft.repository.service.gis_region_model_repository import RegionModelConfig
from shyft.repository.service.gis_region_model_repository import GisRegionModelRepository
from shyft.repository.service.ssa_smg_db import SmGTsRepository, PROD, FC_PROD
from shyft.orchestration.plotting import plot_np_percentiles
from shyft.orchestration.plotting import set_calendar_formatter
from shyft.orchestration.plotting import utc_to_greg
from shyft.orchestration.simulator import DefaultSimulator as Simulator

# extra for the HbvRepository
from shyft.repository.interfaces import RegionModelRepository, BoundingRegion
from pyproj import Proj
from pyproj import transform
from shapely.geometry import Polygon, MultiPolygon, box, Point
from shyft.api import LandTypeFractions, GeoPoint, GeoCellData
from shyft.api import KirchnerParameter
from shyft.api import PriestleyTaylorParameter
from shyft.api import HbvSnowParameter
from shyft.api import ActualEvapotranspirationParameter
from shyft.api import PrecipitationCorrectionParameter
from shyft.api.pt_hs_k import PTHSKParameter


class GridSpecification(BoundingRegion):
    """
    Defines a grid, as lower left x0, y0, dx, dy, nx, ny
    in the specified epsg_id coordinate system
    given a coordindate system with y-axis positive upwards.

    """

    def __init__(self, epsg_id, x0, y0, dx, dy, nx, ny):
        self._epsg_id = epsg_id
        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy
        self.nx = nx
        self.ny = ny

    @property
    def geometry(self):
        """
        Return
        ------
        returns a list describing the bounding box of the gridspec
        lower left x0, y0 and upper right x1, y1
         [x0, y0, x1, y1]
        """
        return [self.x0, self.y0, self.x0 + self.dx * self.nx, self.y0 + self.dy * self.ny]

    def bounding_box(self, epsg):
        """Implementation of interface.BoundingRegion"""
        epsg = str(epsg)
        x0, dx, nx = self.x0, self.dx, self.nx
        y0, dy, ny = self.y0, self.dy, self.ny

        x = np.array([x0, x0 + dx * nx, x0 + dx * nx, x0], dtype="d")
        y = np.array([y0, y0, y0 + dy * ny, y0 + dy * ny], dtype="d")
        if epsg == self.epsg():
            return np.array(x), np.array(y)
        else:
            source_cs = "+init=EPSG:{}".format(self.epsg())
            target_cs = "+init=EPSG:{}".format(epsg)
            source_proj = Proj(source_cs)
            target_proj = Proj(target_cs)
            return [np.array(a) for a in transform(source_proj, target_proj, x, y)]

    def bounding_polygon(self, epsg):
        """Implementation of interface.BoundingRegion"""
        return self.bounding_box(epsg)

    def epsg(self):
        """Implementation of interface.BoundingRegion"""
        return str(self._epsg_id)


class HbvZone(object):
    def __init__(self, zone_id, x, y, from_masl, to_masl, area_km2, glacier_km2, lake_km2, reservoir_km2):
        self.id = zone_id
        self.x = x
        self.y = y
        self.from_masl = from_masl
        self.to_masl = to_masl
        self.area_km2 = area_km2
        self.glacier_km2 = glacier_km2
        self.lake_km2 = lake_km2
        self.reservoir_km2 = reservoir_km2

    @property
    def elevation(self):
        return 0.5 * (self.from_masl + self.to_masl)

    @property
    def mid_point(self):
        return GeoPoint(self.x, self.y, self.elevation)

    @property
    def land_type_fractions(self):
        ltf = LandTypeFractions()
        ltf.set_fractions(glacier=self.glacier_km2 / self.area_km2, lake=self.lake_km2 / self.area_km2, reservoir=self.reservoir_km2 / self.area_km2, forest=0.0)
        return ltf

    @property
    def total_area_m2(self):
        return self.area_km2 * 1000 * 1000


class HbvRegionModelConfig(object):
    """
    Describes the needed mapping between a symbolic region-model name and
    the fields/properties needed to provide a lump HBVModel.

    Requirements:

    and the DTM etc..:
       * bounding box (with epsk_id)

    This class is used to configure the GisRegionModelRepository,
    so that it have a list of known RegionModels, identified by names

    """

    def __init__(self, name, region_model_type, region_parameters, grid_specification,
                 hbv_zones, catchment_parameters={}):
        """
        Parameters
        ----------
        name:string
         - name of the Region-model, like Tistel-ptgsk, Tistel-ptgsk-opt etc.
        region_model_type: shyft api type
         - like pt_gs_k.PTGSKModel
        region_parameters: region_model_type.parameter_t()
         - specifies the concrete parameters at region-level that should be used
        grid_specification: GridSpecification
         - specifies the grid, provides bounding-box and means of creating the cells
        hbv_zones : HbvZone list
         - specifies the 10 HbvZone, elevation, area etc.
        catchment_parameters: dictionary with catchment id as key and region_model_type.parameter_t() as value
         - specifies catchment level parameters
        """
        self.name = name
        self.region_model_type = region_model_type
        self.region_parameters = region_parameters
        self.grid_specification = grid_specification
        self.catchment_parameters = catchment_parameters
        self.hbv_zones = hbv_zones

    @property
    def epsg_id(self):
        return self.grid_specification.epsg_id


def ltm2_kjela_hbv_zone_config(x0=65000, y0=6655000, dx=1000):
    hbv_zones = [  # as fetched from LTM2-Kjela HBV
        HbvZone(1, x0 + 0 * dx, y0, 889, 982, 44.334, 0.000, 6.194, 0.0),
        HbvZone(2, x0 + 1 * dx, y0, 982, 1090, 41.545, 0.000, 5.804, 0.0),
        HbvZone(3, x0 + 2 * dx, y0, 1090, 1106, 14.877, 0.000, 2.078, 0.0),
        HbvZone(4, x0 + 3 * dx, y0, 1106, 1203, 68.043, 0.000, 9.506, 0.0),
        HbvZone(5, x0 + 4 * dx, y0, 1203, 1252, 32.093, 0.000, 4.484, 0.0),
        HbvZone(6, x0 + 5 * dx, y0, 1252, 1313, 32.336, 0.068, 4.517, 0.0),
        HbvZone(7, x0 + 6 * dx, y0, 1313, 1355, 15.017, 0.034, 2.098, 0.0),
        HbvZone(8, x0 + 7 * dx, y0, 1355, 1463, 23.420, 0.346, 3.272, 0.0),
        HbvZone(9, x0 + 8 * dx, y0, 1463, 1542, 11.863, 0.868, 1.657, 0.0),
        HbvZone(10, x0 + 9 * dx, y0, 1542, 1690, 11.673, 2.620, 1.631, 0.0)
    ]
    return hbv_zones


ltm2_kjela_grid_spec=GridSpecification(epsg_id=32633, x0=65000.0, y0=6655000.0, dx=295.201 * 1000.0 / 10.0 / 10.0, dy=10 * 1000.0, nx=10, ny=1)

ltm2_kjela_region_parameters=PTHSKParameter(
        PriestleyTaylorParameter(0.2, 1.26), 
        HbvSnowParameter(tx=-0.1, cx=0.5 * (3.1 + 3.8), ts=0.5 * (-0.9 - 1.0), lw=0.05, cfr=0.001) , 
        ActualEvapotranspirationParameter(1.5),
        KirchnerParameter(c1=-2.810,c2=0.377,c3=-0.050),
        PrecipitationCorrectionParameter(1.0940)
    )



class HbvRegionModelRepository(RegionModelRepository):
    """
    Statkraft GIS service based version of repository for RegionModel objects.
    """

    def __init__(self, region_id_config):
        """
        Parameters
        ----------
        region_id_config: dictionary(region_id:HbvRegionModelConfig)
        """
        self._region_id_config = region_id_config

    def _get_cell_data_info(self, region_id, catchments):
        return self._region_id_config[region_id]  # return HbvZone[] list

    def get_region_model(self, region_id, catchments=None):
        """
        Return a fully specified shyft api region_model for region_id.

        Parameters
        -----------
        region_id: string
            unique identifier of region in data

        catchments: list of unique integers
            catchment id_list when extracting a region consisting of a subset
            of the catchments
        has attribs to construct  params and cells etc.

        Returns
        -------
        region_model: shyft.api type with
           .bounding_region = grid_specification - as fetched from the rm.config
           .catchment_id_map = array, where i'th item is the external catchment'id
           .gis_info = result from CellDataFetcher - used to fetch the grid spec (to help plot etc)
           {
             'cell_data': {catchment_id:[{'cell':shapely-shapes,'elevation':moh,'glacier':area,'lake':area,'reservoir':area,'forest':area}]}
             'catchment_land_types':{catchment_id:{'glacier':[shapelys..],'forest':[shapelys..],'lake':[shapelys..],'reservoir':[shapelys..]}}
             'elevation_raster': np.array(dtype.float64)
           }

        """

        rm = self._get_cell_data_info(region_id, catchments)  # fetch region model info needed to fetch efficiently
        cell_vector = rm.region_model_type.cell_t.vector_t()
        radiation_slope_factor = 0.9  # todo: get it from service layer
        catchment_id_map = []  # needed to build up the external c-id to shyft core internal 0-based c-ids
        for hbv_zone in rm.hbv_zones:
            c_id = hbv_zone.id
            if not c_id == 0:  # only cells with c_id different from 0
                if not c_id in catchment_id_map:
                    catchment_id_map.append(c_id)
                    c_id_0 = len(catchment_id_map) - 1
                else:
                    c_id_0 = catchment_id_map.index(c_id)
            cell = rm.region_model_type.cell_t()
            cell.geo = GeoCellData(hbv_zone.mid_point, hbv_zone.total_area_m2, c_id_0, radiation_slope_factor, hbv_zone.land_type_fractions)
            cell_vector.append(cell)
        catchment_parameter_map = rm.region_model_type.parameter_t.map_t()
        for cid, param in rm.catchment_parameters.items():
            if cid in catchment_id_map:
                catchment_parameter_map[catchment_id_map.index(cid)] = param
        # todo add catchment level parameters to map
        region_model = rm.region_model_type(cell_vector, rm.region_parameters, catchment_parameter_map)
        region_model.bounding_region = rm.grid_specification  # mandatory for orchestration
        region_model.catchment_id_map = catchment_id_map  # needed to map from externa c_id to 0-based c_id used internally in
        region_model.gis_info = {'no-data-for-gis-info': 'empty'}  # result  # opt:needed for internal statkraft use/presentation

        def do_clone(x):
            clone = x.__class__(x)
            clone.bounding_region = x.bounding_region
            clone.catchment_id_map = catchment_id_map
            clone.gis_info = {'no-data-for-gis-info': 'you are out of luck'}
            return clone

        region_model.clone = do_clone

        return region_model


def create_kjela_simulator(model, geo_ts_repository):
    region_id = "LTM2-Kjela-pt_hs_k"
    interpolation_id = 0
    cfg_list = [
        HbvRegionModelConfig(region_id, PTHSKModel.cell_t, ltm2_kjela_region_parameters, ltm2_kjela_grid_spec, ltm2_kjela_hbv_zone_config())
    ]
    reg_model_repository = HbvRegionModelRepository({x.name: x for x in cfg_list})
    pthsk = Simulator(region_id, interpolation_id, reg_model_repository,
                      geo_ts_repository,
                      tistel.interpolation_repository, None)
    return pthsk


def observed_kjela_discharge(period):
    smg_ts_repository = SmGTsRepository(PROD, FC_PROD)
    result = smg_ts_repository.read([u"/Tokk-Kjela.........-D9100A3B5132R016.206"], period)
    return next(iter(result.values()))


def burn_in_state(simulator, t_start, t_stop, q_obs_m3s_ts):
    dt = deltahours(1)
    n = int(round((t_stop - t_start) / dt))
    time_axis = Timeaxis(t_start, dt, n)
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
    h, fill_handles = plot_np_percentiles(times, perc_arrs, base_color=(51 / 256, 102 / 256, 193 / 256))
    percentile_texts = ["{} - {}".format(percentiles[i], percentiles[-(i + 1)]) for i in range(len(percentiles) // 2)]
    ax = plt.gca()
    maj_loc = AutoDateLocator(tz=pytz.UTC, interval_multiples=True)
    ax.xaxis.set_major_locator(maj_loc)
    set_calendar_formatter(Calendar())
    if len(percentiles) % 2:
        fill_handles.append(h[0])
        percentile_texts.append("{}".format(percentiles[len(percentiles) // 2]))
    if obs is not None:
        h_obs = plot_results(None, obs)
        fill_handles.append(h_obs)
        percentile_texts.append("Observed")

    ax.legend(fill_handles, percentile_texts)
    ax.grid(b=True, color=(51 / 256, 102 / 256, 193 / 256), linewidth=0.1, linestyle='-', axis='y')
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
    n_obs = int(round((t_fc_start - t_start) / dt))
    n_fc = 65
    obs_time_axis = Timeaxis(t_start, dt, n_obs)
    fc_time_axis = Timeaxis(t_fc_start, dt, n_fc)
    total_time_axis = Timeaxis(t_start, dt, n_obs + n_fc)
    q_obs_m3s_ts = observed_kjela_discharge(total_time_axis.total_period())
    ptgsk = create_kjela_simulator(PTGSKOptModel, tistel.geo_ts_repository(tistel.grid_spec.epsg()))
    initial_state = burn_in_state(ptgsk, t_start, utc.time(YMDhms(2012, 9, 1)), q_obs_m3s_ts)
    ptgsk.run(obs_time_axis, initial_state)
    plot_results(ptgsk, q_obs_m3s_ts)

    current_state = adjust_simulator_state(ptgsk, t_fc_start, q_obs_m3s_ts)

    ptgsk_fc = create_kjela_simulator(PTGSKModel, tistel.arome_repository(tistel.grid_spec, t_fc_start))
    ptgsk_fc.run(fc_time_axis, current_state)
    plt.figure()
    q_obs_m3s_ts = observed_kjela_discharge(fc_time_axis.total_period())
    plot_results(ptgsk_fc, q_obs_m3s_ts)
    # plt.interactive(1)
    plt.show()


def ensemble_demo():
    utc = Calendar()
    t_start = utc.time(YMDhms(2011, 9, 1))
    t_fc_ens_start = utc.time(YMDhms(2015, 7, 26))
    disp_start = utc.time(YMDhms(2015, 7, 20))
    dt = deltahours(1)
    n_obs = int(round((t_fc_ens_start - t_start) / dt))
    n_fc_ens = 30
    n_disp = int(round(t_fc_ens_start - disp_start) / dt) + n_fc_ens + 24 * 7

    obs_time_axis = Timeaxis(t_start, dt, n_obs + 1)
    fc_ens_time_axis = Timeaxis(t_fc_ens_start, dt, n_fc_ens)
    display_time_axis = Timeaxis(disp_start, dt, n_disp)

    q_obs_m3s_ts = observed_kjela_discharge(obs_time_axis.total_period())
    ptgsk = create_kjela_simulator(PTGSKOptModel, tistel.geo_ts_repository(tistel.grid_spec.epsg()))
    initial_state = burn_in_state(ptgsk, t_start, utc.time(YMDhms(2012, 9, 1)), q_obs_m3s_ts)

    ptgsk.run(obs_time_axis, initial_state)
    current_state = adjust_simulator_state(ptgsk, t_fc_ens_start, q_obs_m3s_ts)
    q_obs_m3s_ts = observed_kjela_discharge(display_time_axis.total_period())
    ens_repos = tistel.arome_ensemble_repository(tistel.grid_spec)
    ptgsk_fc_ens = create_kjela_simulator(PTGSKModel, ens_repos)
    sims = ptgsk_fc_ens.create_ensembles(fc_ens_time_axis, t_fc_ens_start, current_state)
    for sim in sims:
        sim.simulate()
    plt.hold(1)
    percentiles = [10, 25, 50, 75, 90]
    plot_percentiles(sims, percentiles, obs=q_obs_m3s_ts)
    # plt.interactive(1)
    plt.show()


def continuous_calibration():
    utc = Calendar()
    t_start = utc.time(YMDhms(2011, 9, 1))
    t_fc_start = utc.time(YMDhms(2015, 10, 1))
    dt = deltahours(1)
    n_obs = int(round((t_fc_start - t_start) / dt))
    obs_time_axis = Timeaxis(t_start, dt, n_obs + 1)
    q_obs_m3s_ts = observed_kjela_discharge(obs_time_axis.total_period())

    ptgsk = create_kjela_simulator(PTGSKOptModel, tistel.geo_ts_repository(tistel.grid_spec.epsg()))
    initial_state = burn_in_state(ptgsk, t_start, utc.time(YMDhms(2012, 9, 1)), q_obs_m3s_ts)

    num_opt_days = 30
    # Step forward num_opt_days days and store the state for each day:
    recal_start = t_start + deltahours(num_opt_days * 24)
    t = t_start
    state = initial_state
    opt_states = {t: state}
    while t < recal_start:
        ptgsk.run(Timeaxis(t, dt, 24), state)
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
        opt_start = curr_time - deltahours(24 * num_opt_days)
        opt_state = opt_states.pop(opt_start)
        p = ptgsk.region_model.get_region_parameter()
        p_opt = ptgsk.optimize(Timeaxis(opt_start, dt, 24 * num_opt_days), opt_state, target_spec_vec,
                               p, p_min, p_max, tr_stop=1.0e-5)
        ptgsk.region_model.set_region_parameter(p_opt)
        corr_state = adjust_simulator_state(ptgsk, curr_time, q_obs_m3s_ts)
        ptgsk.run(Timeaxis(curr_time, dt, 24), corr_state)
        curr_time += deltahours(24)
        opt_states[curr_time] = ptgsk.reg_model_state
        discharge = ptgsk.region_model.statistics.discharge([0])
        times.extend(discharge.time(i) for i in range(discharge.size()))
        values.extend(list(np.array(discharge.v)))
    plt.plot(utc_to_greg(times), values)
    plot_results(None, q_obs=observed_kjela_discharge(UtcPeriod(recal_start, recal_stop)))
    set_calendar_formatter(Calendar())
    # plt.interactive(1)
    plt.title("Continuously recalibrated discharge vs observed")
    plt.xlabel("Time in UTC")
    plt.ylabel(r"Discharge in $\mathbf{m^3s^{-1}}$", verticalalignment="top", rotation="horizontal")
    plt.gca().yaxis.set_label_coords(0, 1.1)


if __name__ == "__main__":
    import sys

    demos = [forecast_demo, ensemble_demo, continuous_calibration]
    demo = demos[int(sys.argv[1]) if len(sys.argv) == 2 else 0]
    result = demo()
