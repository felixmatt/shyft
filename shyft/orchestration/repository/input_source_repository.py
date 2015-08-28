from base_repository import DictRepository
from shyft.orchestration.input_source import InputSource
from shyft import api


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


def dataset_repository_factory(config, t_start, t_end):
    uid2id = {}
    station_config = config.region_config.stations
    for item in station_config["indices"]:
        uid2id[item["uid"]] = item["id"]

    # Get station locations and elevations:
    constructor = station_config["database"]   # TODO: Fix name in yaml file!
    data_fetcher = constructor(indices=uid2id.keys(), epsg_id=config.region_config.epsg_id)
    geo_pos_by_uid = data_fetcher.fetch()

    # Get time series data
    sources = config.datasets_config.sources
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

    # Construct data and repository
    isr = InputSourceRepository()
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
