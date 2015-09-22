# -*- coding: utf-8 -*-
"""
PLEASE NOTE: This module relies on services specific to Statkraft, for other parties, it could serve as a template for how to glue together a gis-system and a ts-db system
"""
from __future__ import print_function
from __future__ import absolute_import
from abc import ABCMeta,abstractmethod,abstractproperty
import os
from shyft import api
from .. import interfaces
from .gis_station_data import StationDataFetcher # StationDataFetcher is Statkraft dependent service, we could substitute with an interface later
from .ssa_smg_db import SmGTsRepository,PREPROD,PROD # ssa_smg_db is a Statkraft dependent service, we could substitute with an interface later

class MetStationConfig(object):
    """
    Contains keys needed to correctly map time-series and geo-location-id in the underlying services.
    Later when this information-mapping is available directly in the underlying servies, this configuration
    will be obsolete
    """
    def __init__(self,gis_id,temperature=None,precipitation=None,wind_speed=None,radiation=None,relative_humidity=None):
        """ 
        Constructs a MetStationConfig objects with needed keys
        NOTICE that the attribute names are choosen with care, they matches exactly the shyft attribute names for observations/forecast
               we will figure out better aproaches for this, but it allows us to keep it simple and clear at the interface
        Parameters
        ----------
        gis_id: int
            mandatory immutable unique identifier that can be passed to the gis-service to retrieve xyz
        temperature: string
            identifier for temperature[degC] time-series in SSA ts service (SMG)
        precipitation: string
            identifier for temperature[mm/h] time-series in SSA ts service (SMG)
        wind_speed: string
            identifier for wind_speed[m/s] time-series in SSA ts service (SMG)
        radiation: string
            identifier for radiation[W/m2] time-series in SSA ts service (SMG)
        relative_humidity: string
            identifier for relative humidity [%] time-series in SSA ts service (SMG)

        """
        self.gis_id=gis_id
        self.temperature=temperature
        self.precipitation=precipitation
        self.wind_speed=wind_speed
        self.radiation=radiation
        self.relative_humidity=relative_humidity
    
    




class GeoTsRepository(interfaces.GeoTsRepository):
    """
    Statkraft Script Api version of  for GeoTsRepository (Geo Located Timeseries) objects.

    Provide a GeoTsRepository by means of services for 
       * timeseries/forecasts/ensembles: Powel SmG through Statkraft Script Api
       * geo-locations : ESRI arc GIS, and custom published services 

    These are tied together using configuration classes ( interfaces, but with yaml-implementations in real life)
    that provide data-sets with mapping of time-series and consistent geo-locations.


    Usage
    -----
    when constructing the GeoTsRepository,
     pass the following parameters: 
       
       1. GIS-service-endpoints , smg db-service (prod/preprod, role/username)
          we need those, so we can select prod/preprod etc.

       2. List of "met-station" items for observations/forecasts..
         gis-id : <unique id> we use to get x,y,z from GIS-location-service
         then a list of features(all optional, could be empty, each feature just one series)
           temperature  : <smg ts-id>
           precipitation: <smg ts-id>
           wind_speed   : <smg ts-id>
           radiation    : <smg ts-id>
           rel_humidity : <smg ts-id>
            :

       3.? List of catchments with observed discharge (calibration, and common sense/presentation etc)
         gis-id: <unique-id> that could be used to get the shape etc. in GIS-catchment service (but we don't need the location here)
                              we are more interested in the correlation between catchment-discharge time-series.
           discharge    : <smg ts-id>  

    """

    def __init__(self, gis_service,smg_service,met_station_list):
        """
        Parameters
        ----------
            gis_service: string
                server like oslwvagi001p, to be passed to the gis_station_data fetcher service

            smg_service: ssa.environment.SmgEnvironment
                the SMG_PROD or SMG_PREPROD environment configuration (user,db-service)

            met_station_list: list of type MetStationConfig
                list that identifies met_stations in the scope of this repository
                e.g. 
                [ 
                MetStation(gis_id='123',temperature='/Vossevangen/temperature',precipitation='/Vossevangen/precipitation')
                MetStation(gis_id='244', wind_speed='/Hestvollan/wind_speed')
                ]

        """
        self.gis_service=gis_service # later we will pass this to the gis-service api
        self.smg_service= PROD if smg_service=='prod' else PREPROD # we pass this to the ssa smg db interface
        self.met_station_list=met_station_list # this defines the scope of the service, and glue together consistent positions and time-series
        self.source_type_map = {"relative_humidity": api.RelHumSource, #we need this map to provide shyft.api types back to the orchestrator
                                "temperature": api.TemperatureSource,
                                "precipitation": api.PrecipitationSource,
                                "radiation": api.RadiationSource,
                                "wind_speed": api.WindSpeedSource}


    def get_timeseries(self, input_source_types, utc_period,geo_location_criteria=None):
        """
        Parameters
        ----------
        input_source_types: list
            List of source types to retrieve (precipitation,temperature..)
        geo_location_criteria: object
            Some type (to be decided), extent (bbox + coord.ref)
        utc_period: api.UtcPeriod
            The utc time period that should (as a minimum) be covered.

        Returns
        -------
        geo_loc_ts: dictionary
            dictionary keyed by ts type, where values are api vectors of geo
            located timeseries.
        """
        # 1 get the station-> location map
        station_ids=[ x.gis_id for x in self.met_station_list ]
        gis_station_service= StationDataFetcher(epsg_id=32632)
        geo_loc= gis_station_service.fetch(station_ids=station_ids)
        # 2 read all the timeseries
        # create map ts-id to tuple (attr_name,GeoPoint()), the xxxxSource.vector_t 
        ts_to_geo_ts_info= dict()
        result={}
        geo_match= lambda location: geo_location_criteria is None # TODO figure out the form of geo_location_criteria, a bounding box, lambda?
        for attr_name in input_source_types: # we have selected the attr_name and the MetStationConfig with care, so attr_name corresponds to each member in MetStationConfig
            result[attr_name]=self.source_type_map[attr_name].vector_t() # create an empty vector of requested type, we fill in the read-result geo-ts stuff when we are done reading
            ts_to_geo_ts_info.update(
                { k:v for k,v in ([ getattr(x,attr_name) , (attr_name , api.GeoPoint(*geo_loc[x.gis_id][0] ) )] #this constructs a key,value list from the result below
                         for x  in self.met_station_list if getattr(x,attr_name) is not None and geo_match(geo_loc[x.gis_id]) ) #this get out the matching station.attribute
                 })
        ts_list=ts_to_geo_ts_info.keys() # these we are going to read
        ssa_ts_service=SmGTsRepository(self.smg_service)
        read_ts_map=ssa_ts_service.read(ts_list,utc_period)
        # 3 map all returned series with geo-location, and fill in the
        #   vectors of geo-located  TemperatureSource,PrecipitationSource etc.
        for tsn,ts in read_ts_map.iteritems():
            geo_ts_info=ts_to_geo_ts_info[tsn]# this is a tuple( attr_name, api.GeoPoint() )
            attr_name=geo_ts_info[0] # this should be like temperature,precipitaton
            result[attr_name].push_back( self.source_type_map[attr_name](geo_ts_info[1],ts) ) #pick up the vector, push back new geo-located ts
        return result

    def get_forecast(self, input_source_types,
                     geo_location_criteria, utc_period):
        """
        Parameters
        ----------
        See get_timeseries
            Semantics for utc_period: Get the forecast closest up to
            utc_period.start
        Returns
        -------
        forecast: same layout/type as for get_timeseries
        """
        raise NotImplementedError("get_forecast will be implemented later")

    def get_forecast_ensemble(self, input_source_types,
                              geo_location_criteria, utc_period):
        """
        Returns
        -------
        ensemble: list of same type as get_timeseries
        """
        raise NotImplementedError("get_forecast will be implemented later")





