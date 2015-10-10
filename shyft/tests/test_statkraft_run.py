# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import unittest

#basic stuff from api
from shyft import api
from shyft.api import Calendar
from shyft.api import YMDhms
from shyft.api import deltahours
from shyft.api import Timeaxis

#from shyft.api import UtcPeriod
# we need state
#from shyft.repository.interfaces import StateInfo
#from shyft.repository.yaml_state_repository import YamlStateRepository
#from shyft.api.pt_gs_k import PTGSKState,PTGSKStateVector
# but we can do it easy with the defaults for now
from shyft.repository.default_state_repository import DefaultStateRepository

# some yaml-based location repositories
#from shyft.repository.service.yaml_geo_location_repository import YamlGeoLocationRepository
# GIS service based region-model
from shyft.repository.service.gis_region_model_repository import GridSpecification
from shyft.repository.service.gis_region_model_repository import RegionModelConfig
from shyft.repository.service.gis_region_model_repository import GisRegionModelRepository

from shyft.api.pt_gs_k import PTGSKModel #,PTGSKOptModel

from shyft.repository.interpolation_parameter_repository import InterpolationParameterRepository

from shyft.repository.service.ssa_geo_ts_repository import GeoTsRepository
from shyft.repository.service.ssa_geo_ts_repository import MetStationConfig
#from shyft.repository.service.ssa_geo_ts_repository import EnsembleStation
#from shyft.repository.service.ssa_geo_ts_repository import EnsembleConfig
from shyft.repository.service.gis_location_service import GisLocationService
from shyft.repository.service.ssa_smg_db import SmGTsRepository, PROD,FC_PROD
from shyft.orchestration.simulator import SimpleSimulator

class InterpolationConfig(object):
    def interpolation_parameters(self): 
        return {    
            'btk': {
                'gradient': -0.6,
                'gradient_sd': 0.25,
                'nugget': 0.5,
                'range': 200000.0,
                'sill': 25.0,
                'zscale': 20.0,
            },
      
            'idw':{
                'max_distance': 200000.0,
                'max_members': 10,
                'precipitation_gradient': 2.0
            }
        }
        
class StatkraftSimpleRunTestCase(unittest.TestCase):

    @property
    def region_model_repository(self):
        """
        Returns
        -------
         - RegionModelRepository - configured with 'Tistel-ptgsk' etc.
        """
        id_list=[1225]
        epsg_id=32632
        #parameters can be loaded from yaml_config Model parameters..
        pt_params = api.PriestleyTaylorParameter()#*params["priestley_taylor"])
        gs_params = api.GammaSnowParameter()#*params["gamma_snow"])
        ss_params= api.SkaugenParameter()
        ae_params = api.ActualEvapotranspirationParameter()#*params["act_evap"])
        k_params = api.KirchnerParameter()#*params["kirchner"])
        p_params = api.PrecipitationCorrectionParameter() #TODO; default 1.0, is it used ??
        ptgsk_rm_params= api.pt_gs_k.PTGSKParameter(pt_params, gs_params, ae_params, k_params, p_params)
        ptssk_rm_params= api.pt_ss_k.PTSSKParameter(pt_params,ss_params,ae_params,k_params,p_params)
        # create the description for 2 models of tistel,ptgsk, ptssk
        tistel_grid_spec=GridSpecification(x0=362000.0,y0=6765000.0,dx=1000,dy=1000,nx=8,ny=8)
        cfg_list=[
            RegionModelConfig("Tistel-ptgsk",epsg_id,tistel_grid_spec,"unregulated","FELTNR",id_list,ptgsk_rm_params),
            RegionModelConfig("Tistel-ptssk",epsg_id,tistel_grid_spec,"unregulated","FELTNR",id_list,ptssk_rm_params)
        ]
        rm_cfg_dict={ x.name:x for x in cfg_list}
        return GisRegionModelRepository(rm_cfg_dict)
        
    @property
    def geo_ts_repository(self):
        """
        Returns
        -------
         - geo_ts_repository that have met-station-config relevant for tistel
        """
        met_stations=[ # this is the list of MetStations, the gis_id tells the position, the remaining tells us what properties we observe/forecast/calculate at the metstation (smg-ts)
            MetStationConfig(gis_id=129, #0 Fjærland Bremu  
                             temperature   =None,
                             precipitation =None,
                             radiation     =None,
                             wind_speed    =u'/Dnmi-Fjærland.Bremu-T0016V3K-A55820-332'),

            MetStationConfig(gis_id=619, #1 Tistel
                             temperature   =u'/Vikf-Tistel........-T0017A3KI0114',
                             precipitation =None,
                             radiation     =None,
                             wind_speed    =None),

            MetStationConfig(gis_id=684, #2 Vossevangen
                             temperature   =None,#u'/Dnmi-Vossevangen...-T0017V3K-A51530-1337'
                             precipitation =u'/Dnmi-Vossevangen...-T0000D9B-A51530-1337',
                             radiation     =None,
                             wind_speed    =u'/Dnmi-Vossevangen...-T0016V3K-A51530-337'),

            MetStationConfig(gis_id=654, #2 Vossevangen
                             temperature   =None,
                             precipitation =u'/Dnmi-Vangsnes......-T0000D9B-A53101-338',
                             radiation     =None,
                             wind_speed    =u'/Dnmi-Vangsnes......-T0016V3K-A53101-338'),

            MetStationConfig(gis_id=218, #4 Hestvollan
                             temperature   =u'/Vikf-Hestvollan....-T0017A3KI0114',
                             precipitation =u'/Vikf-Hestvollan....-T0000D9BI0124',
                             radiation     =u'/ENKI/STS/Radiation/Sim.-Hestvollan....-T0006V0B-0119-0.8',# clear sky,reduced to 0.8
                             wind_speed    =u'/Vikf-Hestvollan....-T0015V3KI0120'),

            MetStationConfig(gis_id=542, #5 Sopandefjell
                             temperature   =None,#u'/Vikf-Sopandefjell..-T0017V3KI0114'
                             precipitation =None,
                             radiation     =None,
                             wind_speed    =None),

            MetStationConfig(gis_id=650,#6 Ulldalsvatnet
                             temperature   =None,
                             precipitation =None,
                             radiation     =None,
                             wind_speed    =u'/Hoey-Ulldalsvatnet.-T0015V3KI0120'),

        ]
        
        gis_location_repository=GisLocationService() # this provides the gis locations for my stations
        smg_ts_repository = SmGTsRepository(PROD,FC_PROD) # this provide the read function for my time-series

        return GeoTsRepository(
            geo_location_repository=gis_location_repository,
            ts_repository=smg_ts_repository,
            met_station_list=met_stations,
            ens_config=None) #pass service info and met_stations       
    @property
    def interpolation_repository(self):
        return InterpolationParameterRepository(InterpolationConfig())
        
    def test_Tistel_run(self):
        # create needed repositories for the Tistel  catchment
        # geo-ts-repository providing observed geo-located time-series for Tistel
        # gis-region-model-repository with suitable parameters for Tistel, pt_gs_k
        # state-repository, that provide suitable state
        # The Simulator needs some repositories to work, and we construct those
        
        # RegionModelRepository: We get those from the GIS-system, given correct config
        #region_model_repository=GisRegionModelRepository(self.gis_region_model_config)
                # Simulation time axis
        
        utc = Calendar()  # No offset gives Utc
        time_axis = Timeaxis(utc.time(YMDhms(2015,1, 1, 0)), deltahours(1), 240)
        region_id = "Tistel-ptgsk"
        interpolation_id = 0

        # Simulation coordinate system
        #epsg = "32633"

        # Model
        model_t = PTGSKModel

        # Configs and repositories
        #dataset_config_file = path.join(path.dirname(__file__), "netcdf", "atnasjoen_datasets.yaml")
        #region_config = RegionConfig(self.region_config_file)
        #model_config = ModelConfig(self.model_config_file)
        #dataset_config = YamlContent(dataset_config_file)
        
        
        simulator = SimpleSimulator(model_t, region_id, interpolation_id, self.region_model_repository,
                                    self.geo_ts_repository, self.interpolation_repository, None)
        n_cells = simulator.region_model.size()
        state_repos = DefaultStateRepository(model_t, n_cells)
        simulator.run(time_axis, state_repos.get_state(0))
        
        
        pass