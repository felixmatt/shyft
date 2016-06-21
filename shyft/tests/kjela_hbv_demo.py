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
from shyft.api.pt_gs_k import PTGSKParameter

from shyft.api.pt_hs_k import PTHSKModel
from shyft.api.pt_hs_k import PTHSKOptModel
from shyft.api.pt_hs_k import PTHSKParameter

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
from shyft.api import GammaSnowParameter
from shyft.api import ActualEvapotranspirationParameter
from shyft.api import PrecipitationCorrectionParameter

from shyft.repository.interpolation_parameter_repository import InterpolationParameterRepository
from shyft.repository.service.ssa_geo_ts_repository import GeoTsRepository
from shyft.repository.service.ssa_geo_ts_repository import MetStationConfig
from shyft.repository.service.ssa_geo_ts_repository import GeoLocationRepository


#--------------------------------
#  This section contains building blocks, classes that are
#  candidates for promotion into SHyFT standard library
#  The main purpose here is to provide an easy path from a
#  HBV Lump model, to a similar SHyFT lump model
#  utilizing and extendingstandard SHyFT building blocks/repositories
#

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

    @property
    def epsg_id(self):
        return self._epsg_id


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


class HbvGeoLocationError(Exception):
    pass


class HbvGeoLocationRepository(GeoLocationRepository):
    """
    Provide a yaml-based key-location map for gis-identites not available(yet)

    """

    def __init__(self, geo_location_dict, epsg_id):
        """
        Parameters
        ----------
        geo_location_dict:dict(int,[x,y,z])
            map of location id and corresponding x,y,z
        epsg: int
            the epsg valid for coordinates
        """
        if not isinstance(geo_location_dict,dict):
            raise HbvGeoLocationError("geo_location_dict must be a dict(),  of type dict(int,[x,y,z])")
        if epsg_id is None:
            raise HbvGeoLocationError("epsg_id must be specified as a valid epsg_id, like 32633")
        self._geo_location_dict = geo_location_dict
        self._epsg_id = epsg_id

    def get_locations(self, location_id_list, epsg_id=32633):
        if epsg_id != self._epsg_id:
            raise HbvGeoLocationError("HbvGeoLocationRepository does not yet implement epsg-coordindate transforms, wanted epsg_id={0}, available = {1}".format(epsg_id,self._epsg_id))
        locations = {}
        for location_id in location_id_list:
            if self._geo_location_dict.get(location_id) is not None:
                locations[location_id] = tuple(self._geo_location_dict[location_id])
            else:
                raise HbvGeoLocationError("Could not get location of geo point-id!")
        return locations

class HbvModelConfigError(Exception):
    pass

class HbvModelConfig:
    """
    represents the 'things' that we need in order to
    map a type HBV model information into SHyFT models
    """
    def __init__(self,grid_spec,hbv_zones,met_stations,met_station_heights,model_parameters):

        if not isinstance(grid_spec,GridSpecification):
            raise HbvModelConfigError("argument grid_spec must be of type GridSpecification")
        if not isinstance(met_stations,list):
            raise HbvModelConfigError("argument met_stations must be of type list[MetStationConfig]")
        if not isinstance(met_station_heights, list):
            raise HbvModelConfigError("argument met_station_heights must be of type list[double]")
        if len(met_stations) != 2:
            raise HbvModelConfigError("met_stations list must contain exactly 2 metstations")
        if len(met_station_heights) != 2:
            raise HbvModelConfigError("met_station_heights list must contain exactly 2 metstation heights")

        self._grid_spec = grid_spec
        self._hbv_zones = hbv_zones
        self._met_stations = met_stations
        self._met_station_heights = met_station_heights
        self._model_parameters = model_parameters


    @property
    def hbv_zone_config(self,x0=65000, y0=6655000, dx=1000):
        return self._hbv_zones

    @property
    def grid_specification(self):
        return self._grid_spec

    @property
    def region_parameters(self):
        return self._model_parameters


    @property
    def geo_ts_repository(self):
        bbox=self.grid_specification.geometry # gives us [x0,y0,x1,y1], the corners of the bounding box
        fake_station_1_pos = [bbox[0],bbox[1], self._met_station_heights[0]]
        fake_station_2_pos = [bbox[2],bbox[3], self._met_station_heights[1]]
        gis_location_repository = HbvGeoLocationRepository(
            geo_location_dict={self._met_stations[0].gis_id:fake_station_1_pos,self._met_stations[1].gis_id:fake_station_2_pos},
            epsg_id=self.grid_specification._epsg_id)  # this provides the gis locations for my stations
        smg_ts_repository = SmGTsRepository(PROD, FC_PROD)  # this provide the read function for my time-series

        return GeoTsRepository(epsg_id=self.grid_specification._epsg_id, geo_location_repository=gis_location_repository,
                               ts_repository=smg_ts_repository, met_station_list=self._met_stations,
                               ens_config=None)

class InterpolationConfig(object):
    """ A bit clumsy, but to reuse dictionary based InterpolationRepository:"""

    def interpolation_parameters(self):
        return {
            'temperature': {
                'method': 'idw',
                'params': {
                    'max_distance': 600000.0,
                    'max_members': 10,
                    'distance_measure_factor': 1,
                    'default_temp_gradient': -0.006
                }
            },
            'precipitation': {
                'method': 'idw',
                'params': {
                    'max_distance': 600000.0,
                    'max_members': 10,
                    'distance_measure_factor': 1,
                    'scale_factor': 1.02
                }
            },
            'radiation': {
                'method': 'idw',
                'params': {
                    'max_distance': 600000.0,
                    'max_members': 10,
                    'distance_measure_factor': 1
                }
            },
            'wind_speed': {
                'method': 'idw',
                'params': {
                    'max_distance': 600000.0,
                    'max_members': 10,
                    'distance_measure_factor': 1
                }
            },
            'relative_humidity': {
                'method': 'idw',
                'params': {
                    'max_distance': 600000.0,
                    'max_members': 10,
                    'distance_measure_factor': 1
                }
            }
        }

interpolation_repository = InterpolationParameterRepository(InterpolationConfig())

del InterpolationConfig

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

#-------------------------------------------------------------
# This section is for experimenting with one concrete HBV model
# and utilize the classes above to work with lump HBV models in SHyFT
#

kjela_x0 = 65000
kjela_y0 = 6655000
dx = 1000

kjela = HbvModelConfig(
    grid_spec=GridSpecification(epsg_id=32633, x0=kjela_x0, y0=kjela_y0, dx=295.201 * 1000.0 / 10.0 / 10.0, dy=10 * 1000.0, nx=10, ny=1),
    hbv_zones=[  # as fetched from LTM2-Kjela HBV
            HbvZone(1, kjela_x0 + 0 * dx, kjela_y0, 889, 982, 44.334, 0.000, 6.194, 0.0),
            HbvZone(2, kjela_x0 + 1 * dx, kjela_y0, 982, 1090, 41.545, 0.000, 5.804, 0.0),
            HbvZone(3, kjela_x0 + 2 * dx, kjela_y0, 1090, 1106, 14.877, 0.000, 2.078, 0.0),
            HbvZone(4, kjela_x0 + 3 * dx, kjela_y0, 1106, 1203, 68.043, 0.000, 9.506, 0.0),
            HbvZone(5, kjela_x0 + 4 * dx, kjela_y0, 1203, 1252, 32.093, 0.000, 4.484, 0.0),
            HbvZone(6, kjela_x0 + 5 * dx, kjela_y0, 1252, 1313, 32.336, 0.068, 4.517, 0.0),
            HbvZone(7, kjela_x0 + 6 * dx, kjela_y0, 1313, 1355, 15.017, 0.034, 2.098, 0.0),
            HbvZone(8, kjela_x0 + 7 * dx, kjela_y0, 1355, 1463, 23.420, 0.346, 3.272, 0.0),
            HbvZone(9, kjela_x0 + 8 * dx, kjela_y0, 1463, 1542, 11.863, 0.868, 1.657, 0.0),
            HbvZone(10, kjela_x0 + 9 * dx, kjela_y0, 1542, 1690, 11.673, 2.620, 1.631, 0.0)
        ],
    met_stations=[
            # this is the list of MetStations, the gis_id tells the position, the remaining tells us what properties we observe/forecast/calculate at the metstation (smg-ts)
            MetStationConfig(gis_id=1,  # 1 _LO 889.0 masl
                             temperature=u'/LTM2-Kjela.........-D0017F3A-HBVSHYFT_LO',
                             precipitation=u'/LTM2-Kjela.........-D0000F9A-HBVSHYFT_LO',
                             radiation=u'/ENKI/STS/Radiation/Sim.-Hestvollan....-T0006V0B-0119-0.8',# clear sky,reduced to 0.8
                             wind_speed=u'/Tokk-Vågsli........-D0015A9KI0120'),

            MetStationConfig(gis_id=2,  # 2 _HI 1690.0 masl
                             temperature=u'/LTM2-Kjela.........-D0017F3A-HBVSHYFT_HI',
                             precipitation=u'/LTM2-Kjela.........-D0000F9A-HBVSHYFT_HI',
                             radiation=None,
                             wind_speed=None,
                             relative_humidity=u'/SHFT-rel-hum-dummy.-T0002A3R-0103'),
        ],
    met_station_heights=[889.0,1690.0],
    model_parameters=PTHSKParameter(
            PriestleyTaylorParameter(0.2, 1.26),
            HbvSnowParameter(tx=-0.2, cx=0.5 * (3.1 + 3.8), ts=0.5 * (-0.9 - 1.0), lw=0.05, cfr=0.001),
            ActualEvapotranspirationParameter(1.5),
            KirchnerParameter(c1=-2.810,c2=0.377,c3=-0.050),
            PrecipitationCorrectionParameter(1.0940)
        )
)
#note: this is the model_dt for the hbv
#  seems that this greatly influences the snow response, so we suspect there is
#  a parameter dimension somewhere that is not well formed
model_dt= deltahours(6) # could be 1,2,3,6,8,12,24

def create_kjela_model(shyft_model_type, geo_ts_repository):
    region_id = "LTM2-Kjela"
    interpolation_id = 0
    cfg_list = [
        HbvRegionModelConfig(region_id, shyft_model_type, kjela.region_parameters, kjela.grid_specification, kjela.hbv_zone_config)
    ]
    reg_model_repository = HbvRegionModelRepository({x.name: x for x in cfg_list})
    model_simulator = Simulator(region_id, interpolation_id, reg_model_repository,
                      kjela.geo_ts_repository,
                      interpolation_repository, None)
    model_simulator.region_model.set_snow_sca_swe_collection(-1,True) # so that we could calibrate on sca/swe
    model_simulator.region_model.set_state_collection(-1,True) # so that we can get out state information swe/sca
    return model_simulator


def observed_kjela_discharge(period):
    smg_ts_repository = SmGTsRepository(PROD, FC_PROD)
    discharge_ts_name=u'/Tokk-Kjela.........-D9100A3B5132R016.206'
    result = smg_ts_repository.read([discharge_ts_name], period)
    return result[discharge_ts_name]



def burn_in_state(shyft_model, time_axis, q_obs_m3s_at_start):
    n_cells = shyft_model.region_model.size()
    state_repos = DefaultStateRepository(shyft_model.region_model.__class__, n_cells)
    s0 = state_repos.get_state(0) # get out a state to start with
    for s in s0:
        s.kirchner.q = 0.5 # insert some more water than the default 0.001 mm
    shyft_model.run(time_axis, s0)
    # Go back in time (to t_start) and adjust q with observed discharge at that time.
    # This will give us a good initial state at t_start
    return shyft_model.discharge_adjusted_state(q_obs_m3s_at_start)



def plot_results(ptxsk, q_obs=None):
    h_obs = None
    n_temp = 1
    temp=[]
    precip = None
    discharge = None
    plt.figure() # to start a new plot window
    if ptxsk is not None:
        plt.subplot(8, 1, 1) # dimension 8 rows of plots
        discharge = ptxsk.region_model.statistics.discharge([]) # get the sum discharge all areas
        n_temp=ptxsk.region_model.size()
        temp = [ptxsk.region_model.statistics.temperature([i]) for i in range(n_temp)]
        precip = [ptxsk.region_model.statistics.precipitation([i]) for i in range(n_temp)]
        radiation = [ptxsk.region_model.statistics.radiation([i]) for i in range(n_temp)]
        rel_hum = [ptxsk.region_model.statistics.rel_hum([i]) for i in range(n_temp)]
        wind_speed = [ptxsk.region_model.statistics.wind_speed([i]) for i in range(n_temp)]
        snow_sca = [ptxsk.region_model.hbv_snow_state.sca([i]) for i in range(n_temp)]
        snow_swe = [ptxsk.region_model.hbv_snow_state.swe([i]) for i in range(n_temp)]
        # Results on same time axis, so we only need one
        times = utc_to_greg([discharge.time(i) for i in range(discharge.size())])
        plt.plot(times, np.array(discharge.v),color='blue')
        ax=plt.gca()
        ax.set_xlim(times[0], times[-1])
        plt.ylabel(r"Discharge in $\mathbf{m^3/s}$")
        set_calendar_formatter(Calendar())
    if q_obs is not None:
        obs_times = utc_to_greg([q_obs.time(i) for i in range(q_obs.size())])
        ovs = [q_obs.value(i) for i in range(q_obs.size())]
        h_obs, = plt.plot(obs_times, ovs, linewidth=2,color='green')


    if ptxsk is not None:
        plt.subplot(8, 1, 2,sharex=ax)
        for i in range(n_temp):
            plt.plot(times, np.array(temp[i].v))
        #set_calendar_formatter(Calendar())
        plt.ylabel(r"Temperature in C")

        plt.subplot(8, 1, 3,sharex=ax)
        for i in range(n_temp):
            plt.plot(times, np.array(precip[i].v))
        plt.ylabel(r"Precipitation in mm")

        plt.subplot(8,1,4,sharex=ax)
        for i in range(n_temp):
            plt.plot(times,np.array(radiation[i].v))
        plt.ylabel(r"Radiation w/m2")

        plt.subplot(8, 1, 5,sharex=ax)
        for i in range(n_temp):
            plt.plot(times, np.array(rel_hum[i].v))
        plt.ylabel(r"Rel.hum  %")

        plt.subplot(8, 1, 6,sharex=ax)
        for i in range(n_temp):
            plt.plot(times, np.array(wind_speed[i].v))
        plt.ylabel(r"Wind speed  m/s")

        plt.subplot(8, 1, 7,sharex=ax)
        for i in range(n_temp):
            plt.plot(times, np.asarray(snow_swe[i].v)[:-1])
        plt.ylabel(r"SWE  mm")

        plt.subplot(8, 1, 8,sharex=ax)
        for i in range(n_temp):
            plt.plot(times, np.asarray(snow_sca[i].v)[:-1])
        plt.ylabel(r"SCA  %")

    return h_obs

def simple_run_demo():
    """Simple demo using HBV time-series and similar model-values

    """
    # 1. Setup the time-axis for our simulation
    normal_calendar = Calendar(3600) # we need UTC+1, since day-boundaries in day series in SmG is at UTC+1
    t_start = normal_calendar.time(2010, 9, 1) # we start at
    t_end   = normal_calendar.add(t_start,Calendar.YEAR,5) # 5 years period for simulation
    time_axis = Timeaxis(t_start, model_dt, normal_calendar.diff_units(t_start,t_end,model_dt))

    # 2. Create the shyft model from the HBV model-repository
    shyft_model = create_kjela_model(PTHSKModel, kjela.geo_ts_repository)

    # 3. establish the initial state
    # using the *pattern* of distribution after one year (so hbv-zone 1..10 get approximate distribution of discharge)
    #      *and* the observed discharge at the start time t_start
    #
    t_burnin = normal_calendar.add(t_start,Calendar.YEAR,1) # use one year to get distribution between hbvzones
    burnin_time_axis = Timeaxis(t_start, model_dt, normal_calendar.diff_units(t_start, t_burnin, model_dt))
    q_obs_m3s_ts = observed_kjela_discharge(time_axis.total_period()) # get out the observation ts
    q_obs_m3s_at_t_start= q_obs_m3s_ts(t_start) # get the m3/s at t_start
    initial_state = burn_in_state(shyft_model,burnin_time_axis, q_obs_m3s_at_t_start)

    # 4. now run the model with the established state
    #    that will start out with the burn in state
    shyft_model.run(time_axis, initial_state)

    # 5. display results etc. goes here
    plot_results(shyft_model, q_obs_m3s_ts)
    plt.show()




if __name__ == "__main__":
    import sys

    demos = [simple_run_demo]
    demo = demos[int(sys.argv[1]) if len(sys.argv) == 2 else 0]
    result = demo()
