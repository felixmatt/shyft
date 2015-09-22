# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from abc import ABCMeta,abstractmethod,abstractproperty
import os
from shyft import api
from .. import interfaces
from gis_station_data import StationDataFetcher
from ssa_smg_db import SmGTsRepository

class MetStationConfig(object):
    """
    Contains keys needed to correctly map time-series and geo-location-id in the underlying services.
    Later when this information-mapping is available directly in the underlying servies, this configuration
    will be obsolete
    """
    def __init__(self,gis_id,temperature=None,precipitation=None,wind_speed=None,radiation=None,rel_humidity=None):
        """ 
        Constructs a MetStationConfig objects with needed keys
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
        rel_humidity: string
            identifier for relative humidity [%] time-series in SSA ts service (SMG)

        """
        self.gis_id=gis_id
        self.temperature=temperature
        self.precipitation=precipitation
        self.wind_speed=wind_speed
        self.radiation=radiation
        self.rel_humidity=rel_humidity
    
    




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
        self.gis_service=gis_service
        self.smg_service=smg_service
        self.met_station_list=met_station_list


    def get_timeseries(self, input_source_types,
                       geo_location_criteria, utc_period):
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
        station_locaton_map= gis_station_service.fetch(station_ids=station_ids)
        # 2 read all the timeseries
        # create map ts-id to station (could be position..)
        ts_to_station={ k:v for k,v in ([x.temperature,x] for x  in self.met_station_list if x.temperature is not None) }
        ts_to_station.update({ k:v for k,v in ([x.temperature,x] for x  in self.met_station_list) if x.precipitation is not None })
        ts_to_station.update({ k:v for k,v in ([x.wind_speed,x] for x  in self.met_station_list) if x.wind_speed is not None })
        ts_to_station.update({ k:v for k,v in ([x.radiation,x] for x  in self.met_station_list) if x.radiation is not None })
        ts_to_station.update({ k:v for k,v in ([x.rel_humidity,x] for x  in self.met_station_list) if x.rel_humidity is not None })
        ts_list=ts_to_station.keys();
        ssa_ts_service=SmGTsRepository(self.smg_service)
        read_ts_map=ssa_ts_service.read(ts_ids,utc_period)
        # 3 map all returned series with geo-location, and create the
        #   vector of geo-located  TemperatureSource,PrecipitationSource
        #TODO!!! fix smart map ts --> geopos and type, then just lookup..
        return read_ts_map

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
        pass

    def get_forecast_ensemble(self, input_source_types,
                              geo_location_criteria, utc_period):
        """
        Returns
        -------
        ensemble: list of same type as get_timeseries
        """
        pass

class InputSourceRepository(object):

    def __init__(self, *args, **kwargs):
        #super(InputSourceRepository, self).__init__()
        self.input_source_types = {"temperature": api.TemperatureSource,
                                   "precipitation": api.PrecipitationSource,
                                   "radiation": api.RadiationSource,
                                   "wind_speed": api.WindSpeedSource,
                                   "relative_humidity": api.RelHumSource}
        self.data.update({k: v.vector_t() for (k, v) in
                          self.input_source_types.iteritems()})

    def add_input_source(self, station):
        for input_source in self.input_source_types:
            ts = getattr(station, input_source)
            if ts is not None:
                api_source = self.input_source_types[input_source](
                    api.GeoPoint(*station.geo_location), ts)
                self.data[input_source].append(api_source)

    def add_input_source_vct(self, points):
        input_type = points.keys()[0]
        if(input_type in self.input_source_types.keys()):
            self.data[input_type].reserve(self.data[input_type].size()
                                          + points[input_type].size())
            [self.data[input_type].push_back(i) for i in
             points[input_type].iterator()]


def dataset_repository_factory(config, t_start, t_end):
    # Construct data and repository
    isr = InputSourceRepository()
    cal = api.Calendar()  # UTC


    if('point_sources' in config.datasets_config.sources.keys()):
        sources = config.datasets_config.sources['point_sources']
        for repository in sources:
            constructor = repository["repository"][0]
            args_ = repository["repository"][1:]
            args = []
            t_str = cal.to_string(t_start).replace('.', '')
            args.append(os.path.join(args_[0], args_[1] + t_str[0:8] +
                        '_' + t_str[9:11] + '.nc'))  # filepath
            args.append(config.region_config.epsg_id)  # EPSG
            bounding_box = ([config.region_config.x_min,
                             config.region_config.x_max,
                             config.region_config.x_max,
                             config.region_config.x_min],
                            [config.region_config.y_max,
                             config.region_config.y_max,
                             config.region_config.y_min,
                             config.region_config.y_min])
            args.append(bounding_box)
            ts_fetcher = constructor(*args)
            sources = ts_fetcher.get_sources()
            for source_type in repository['types']:
                isr.add_input_source_vct({source_type: sources[source_type]})
    if('station_sources' in config.datasets_config.sources.keys()):
        uid2id = {}
        station_config = config.region_config.stations
        for item in station_config["indices"]:
            uid2id[item["uid"]] = item["id"]

        # Get station locations and elevations:
        constructor = station_config["database"]  # TODO: Fix name in yaml file
        data_fetcher = constructor(indices=uid2id.keys(),
                                   epsg_id=config.region_config.epsg_id)
        geo_pos_by_uid = data_fetcher.fetch()

        # Get time series data
        sources = config.datasets_config.sources['station_sources']
        time_series = {}
        for repository in sources:
            data_uids = []
            for source_type in repository["types"]:
                data_uids.extend([source["uid"] for source in
                                  source_type["stations"]])
            constructor = repository["repository"][0]
            arg = repository["repository"][1]
            ts_fetcher = constructor(arg)
            period = api.UtcPeriod(t_start, t_end)
            time_series = ts_fetcher.read(data_uids, period)

        for station in station_config["indices"]:
            input_source = construct_input_source(station, geo_pos_by_uid,
                                                  sources, time_series)
            if input_source:
                isr.add_input_source(input_source)
    return isr


def construct_input_source(station, geo_pos_by_uid, sources, time_series):
    station_uid = station["uid"]
    station_id = station["id"]
    source_by_type = {}
    for repository in sources:
        for source_type in repository["types"]:
            if station_id in [source["station_id"] for source in
                              source_type["stations"]]:
                source_uid = [source["uid"] for source in
                              source_type["stations"]
                              if source["station_id"] == station_id][0]
                if source_uid in time_series:
                    source_by_type[source_type["type"]] = \
                        time_series[source_uid]
                else:
                    print("WARNING: time series {} not found, \
 skipping".format(source_uid))
    if source_by_type:
        return InputSource(geo_pos_by_uid[station_uid][0], source_by_type)
    return None
