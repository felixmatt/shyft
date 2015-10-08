# -*- coding: utf-8 -*-
"""
PLEASE NOTE: This module relies on services specific to Statkraft, for other parties, it could serve as a template for how to glue together a gis-system and a ts-db system
"""
from __future__ import print_function
from __future__ import absolute_import
from abc import ABCMeta,abstractmethod,abstractproperty
import os
from shyft import api
from shyft.repository import interfaces

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

def create_forecast_ensemble_config(gis_id,temperature_name_format,precipitation_name_format,n_ensembles):
    """
    Create MetStationConfig for ensembles.
    In SmG they are aranged by naming convention
    EC00 forecast ensemble members is like this:
      LTM1-Nore.1........-T0000A5P_EC00_Exx (precipitation member xx)
      LTM1-Nore.1........-T0017A3P_EC00_Exx (temperature member xx)
    then use
      temperature_name_format  = LTM1-Nore.1........-T0017A3P_EC00_E{0:02}
      precipitation_name_format= LTM1-Nore.1........-T0000A5P_EC00_E{0:02}
    Parameters
    ----------
    gis_id:string|int
     - identifier passed to the gis-service to get the real position (x,y,z) of the ensemble 
    temperature_name_format:string
     - a string format for the temperature members that we apply .format(i) for i in 0..n-ensembles
    precipitiation_name_format:string
     - a string format for the precipitation members that we apply .format(i) for i in 0..n-ensembles
    n_ensembles:int
     - number of ensembles, in Statrkraft this is 52 for the ECxx forecasts

    Returns
    -------
    list of MetStationConfig, with gis_id and temperature, precipitation pairs.

    """
    return [ MetStationConfig(gis_id,temperature=temperature.format(i),precipitation=precipitation.format(i)) for i in xrange(n_ensembles)]
    
    
class GeoLocationRepository(object):
    """
    Responsible for providing locations for specified gis-identifiers
    A candidate for interfaces
    """
    __metaclass__ = ABCMeta
    @abstractmethod
    def get_locations(self,location_id_list,epsg_id=32632):
        """
        Given that we know the location-id (typically an integer, could be string)
        provide the locations (x,y,z) in a specified coordinate system
        Parameters
        ----------
        location_id_list:list of type integer
            identifies the gis-locationns, uniquely
        epsg_id: integer
            identifies the coordinate system

        Returns
        -------
        dictionary of identifier:tuple(x,y,z)
        """
        pass

class TsRepository(object):
    """
    Defines the contract of a time-series repository for this specific use
    The responsibility is to read and store time-series,forecasts, ensembles
    A candidate for interfaces
    """
    @abstractmethod
    def read(self,list_of_ts_id,period):
        """
        """
        pass
    
    @abstractmethod
    def read_forecast(self,list_of_fc_id,period):
        """ 
        read and return the newest forecast that have the biggest overlap with specified period
        note that we should check that the semantic of this is reasonable

        """
        pass

    @abstractmethod
    def store(self,timeseries_dict):
        """ Store the supplied time-series to the underlying db-system.
            Parameters
            ----------
            timeseries_dict: dict string:timeseries
                the keys are the wanted ts(-path) names
                and the values are shyft api.time-series.
                If the named time-series does not exist, create it.
        """
        pass

class GeoTsRepositoryError(Exception):
    pass

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

    def __init__(self, geo_location_repository,ts_repository,met_station_list):
        """
        Parameters
        ----------
        geo_location_repository: GeoLocationRepository
            -server like oslwvagi001p, to be passed to the gis_station_data fetcher service

        ts_repository: TsRepository
            -the SMG_PROD or SMG_PREPROD environment configuration (user,db-service)

        met_station_list: list of type MetStationConfig
            -list that identifies met_stations in the scope of this repository
            e.g. 
            [ 
            MetStation(gis_id='123',temperature='/Vossevangen/temperature',precipitation='/Vossevangen/precipitation')
            MetStation(gis_id='244', wind_speed='/Hestvollan/wind_speed')
            ]

        """
        if not isinstance(geo_location_repository,GeoLocationRepository): raise GeoTsRepositoryError("geo_location_repository should be an implementation of GeoLocationRepository")
        if not isinstance(ts_repository,TsRepository):raise GeoTsRepositoryError("ts_repository should be an implementation of TsRepository")
        self.geo_location_repository=geo_location_repository
        self.ts_repository= ts_repository # we pass this to the ssa smg db interface
        self.met_station_list=met_station_list # this defines the scope of the service, and glue together consistent positions and time-series
        self.source_type_map = {"relative_humidity": api.RelHumSource, #we need this map to provide shyft.api types back to the orchestrator
                                "temperature": api.TemperatureSource,
                                "precipitation": api.PrecipitationSource,
                                "radiation": api.RadiationSource,
                                "wind_speed": api.WindSpeedSource}

    def _get_ts_to_geo_ts_result(self,input_source_types,geo_match):
        """ given the input-sources (temp,precip etc), and a geo-match, return back tsname->geopos, plus a result structure dict ready to add on the read ts"""
        # 1 get the station-> location map
        station_ids=[ x.gis_id for x in self.met_station_list ]
        geo_loc= self.geo_location_repository.get_locations(location_id_list=station_ids,epsg_id=32632) #TODO use epsg for self
        # 2 create map ts-id to tuple (attr_name,GeoPoint()), the xxxxSource.vector_t 
        #   fill in a startingpoint for result{'tempeature':xxxxSourve.vector_t} etc
        ts_to_geo_ts_info= dict()
        result={}
        for attr_name in input_source_types: # we have selected the attr_name and the MetStationConfig with care, so attr_name corresponds to each member in MetStationConfig
            result[attr_name]=self.source_type_map[attr_name].vector_t() # create an empty vector of requested type, we fill in the read-result geo-ts stuff when we are done reading
            ts_to_geo_ts_info.update(
                { k:v for k,v in ([ getattr(x,attr_name) , (attr_name , api.GeoPoint(*geo_loc[x.gis_id] ) )] #this constructs a key,value list from the result below
                         for x  in self.met_station_list if getattr(x,attr_name) is not None and geo_match(geo_loc[x.gis_id]) ) #this get out the matching station.attribute
                 })
        return ts_to_geo_ts_info,result

    def _remap_to_result(self,read_ts_map,result,ts_to_geo_ts_info):
        """ given read_ts_map, as a result from read,read_forecast
            map it into correct vector in result,
            using geo_ts_info
        """
        for tsn,ts in read_ts_map.iteritems():
            geo_ts_info=ts_to_geo_ts_info[tsn]# this is a tuple( attr_name, api.GeoPoint() )
            attr_name=geo_ts_info[0] # this should be like temperature,precipitaton
            result[attr_name].push_back( self.source_type_map[attr_name](geo_ts_info[1],ts) ) #pick up the vector, push back new geo-located ts
        return result


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
        if geo_location_criteria is not None:raise("geo_location_criteria is not yet implemented")
        geo_match= lambda location: geo_location_criteria is None # TODO figure out the form of geo_location_criteria, a bounding box, lambda?
        ts_to_geo_ts_info,result=self._get_ts_to_geo_ts_result(input_source_types,geo_match)
        ts_list=ts_to_geo_ts_info.keys() # these we are going to read
        read_ts_map=self.ts_repository.read(ts_list,utc_period)
        return self._remap_to_result(read_ts_map,result,ts_to_geo_ts_info) # map back to result


    def get_forecast(self, input_source_types,utc_period,t_c,geo_location_criteria):
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
        if t_c is not None: raise("t_c, time created spec is not yet implemented")
        if geo_location_criteria is not None:raise("geo_location_criteria is not yet implemented")
        geo_match= lambda location: geo_location_criteria is None # TODO figure out the form of geo_location_criteria, a bounding box, lambda?
        ts_to_geo_ts_info,result=self._get_ts_to_geo_ts_result(input_source_types,geo_match)
        ts_list=ts_to_geo_ts_info.keys() # these we are going to read
        read_ts_map=self.ts_repository.read_forecast(ts_list,utc_period) #TODO: maybe pass tc (t-created)
        return self._remap_to_result(read_ts_map,result,ts_to_geo_ts_info) # map back to result

    def get_forecast_ensemble(self, input_source_types,utc_period,t_c,geo_location_criteria):
        """
        alg:
        Using ensemble-generators? based on name rules passed to constructor.
        result= [], type is list of  dictionary of input_source_type:input_source_type_vector_of_geo_ts
        this will have n_ensemble dictionaries,
        for each of them
           create the ts_to_geo_ts_info,result
        
        for each ensemble that matches geo_location_criteria
           add ts to read-list
        read all the ts
        remap tsid -> to the  _remap_to_result (kept within the result[] list)

        Parameters
        ----------
        Returns
        -------
        ensemble: list of same type as get_timeseries
        """
        raise NotImplementedError("get_forecast will be implemented later")





