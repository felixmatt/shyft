from base_repository import DictRepository
from shyft.orchestration.input_source import InputSource
from shyft import api
import os


class InputSourceRepository(DictRepository):

    def __init__(self, *args, **kwargs):
        super(InputSourceRepository, self).__init__()
        self.input_source_types = {"temperature": api.TemperatureSource,
                                   "precipitation": api.PrecipitationSource,
                                   "radiation": api.RadiationSource,
                                   "wind_speed": api.WindSpeedSource,
                                   "relative_humidity": api.RelHumSource}
        self.data.update({k: v.vector_t() for (k, v) in self.input_source_types.iteritems()})

    def add_input_source(self, station):
        for input_source in self.input_source_types:
            ts = getattr(station, input_source)
            if ts is not None:
                api_source = self.input_source_types[input_source](api.GeoPoint(*station.geo_location), ts)
                self.data[input_source].append(api_source)
                
    def add_input_source_vct(self, points):
        input_type = points.keys()[0]
        if(input_type in self.input_source_types.keys()):
            #self.data[input_type] = points[input_type]
            for i in range(10):#len(points[input_type])):
                self.data[input_type].append(points[input_type][i])


def dataset_repository_factory(config, t_start, t_end):
    # Construct data and repository
    isr = InputSourceRepository()
    cal = api.Calendar() # UTC
    
    if('point_sources' in config.datasets_config.sources.keys()):
        sources = config.datasets_config.sources['point_sources']
        for repository in sources:
            constructor = repository["repository"][0]
            args_ = repository["repository"][1:]
            args=[]
            t_str = cal.to_string(t_start).replace('.','')
            print os.path.join(args_[0], args_[1]+t_str[0:8]+'_'+t_str[9:11]+'.nc')
            args.append(os.path.join(args_[0], args_[1]+t_str[0:8]+'_'+t_str[9:11]+'.nc')) # filepath
            args.append(config.region_config.epsg_id) # EPSG
            bounding_box = ([config.region_config.x_min, config.region_config.x_max, config.region_config.x_max, config.region_config.x_min],
                            [config.region_config.y_max, config.region_config.y_max, config.region_config.y_min, config.region_config.y_min])
            args.append(bounding_box)
            ts_fetcher = constructor(*args)
            sources = ts_fetcher.get_sources()
            for source_type in repository['types']:
                isr.add_input_source_vct({source_type:sources[source_type]})   
    if('station_sources' in config.datasets_config.sources.keys()):
        uid2id = {}
        station_config = config.region_config.stations
        for item in station_config["indices"]:
            uid2id[item["uid"]] = item["id"]
    
        # Get station locations and elevations:
        constructor = station_config["database"]   # TODO: Fix name in yaml file!
        data_fetcher = constructor(indices=uid2id.keys(), epsg_id=config.region_config.epsg_id)
        geo_pos_by_uid = data_fetcher.fetch()
    
        # Get time series data
        sources = config.datasets_config.sources['station_sources']
        time_series = {}
        for repository in sources:
            data_uids = []
            for source_type in repository["types"]:
                data_uids.extend([source["uid"] for source in source_type["stations"]])
            constructor = repository["repository"][0]
            arg = repository["repository"][1]
            ts_fetcher = constructor(arg)
            period = api.UtcPeriod(t_start, t_end)
            time_series = ts_fetcher.read(data_uids, period)
    
    
        for station in station_config["indices"]:
            input_source = construct_input_source(station, geo_pos_by_uid, sources, time_series)
            if input_source:
                isr.add_input_source(input_source)
    return isr


def construct_input_source(station, geo_pos_by_uid, sources, time_series):
    station_uid = station["uid"]
    station_id = station["id"]
    source_by_type = {}
    for repository in sources:
        for source_type in repository["types"]:
            if station_id in [source["station_id"] for source in source_type["stations"]]:
                source_uid = [source["uid"] for source in source_type["stations"]
                              if source["station_id"] == station_id][0]
                if source_uid in time_series:
                    source_by_type[source_type["type"]] = time_series[source_uid]
                else:
                    print "WARNING: time series {} not found, skipping".format(source_uid)
    if source_by_type:
        return InputSource(geo_pos_by_uid[station_uid][0], source_by_type)
    return None
