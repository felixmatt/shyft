from datetime import datetime
import copy
from math import ceil
import random
import math
from shyft import api
from shyft.repository.base_repository import BaseRepository
from shyft.orchestration.state import State
from shyft.orchestration.input_source import InputSource
from shyft.repository.testsupport.time_series import create_mock_station_data
from shyft.repository.cell_read_only_repository import CellReadOnlyRepository


class MockRepository(BaseRepository):
    """Simple mock repository that uses a dictionary to hold the data."""

    def __init__(self):
        self.data = {}

    def get(self, key):
        if key in self.data:
            return self.data[key]
        raise RuntimeError("No entry with key {} found.".format(key))

    def put(self, key, entry):
        if key not in self.data:
            self.data[key] = entry

    def delete(self, key):
        if key in self.data:
            return self.data.pop(key)
        raise RuntimeError("No entry with key {} found.".format(key))

    def find(self, *args, **kwargs):
        return self.data.keys()


class MockStateRepository(MockRepository):

    def __init__(self, *args, **kwargs):
        super(MockStateRepository, self).__init__()
        self.num_states = kwargs.get("num_states", 10)

    def find(self, *args, **kwargs):
        condition = kwargs.get("condition", None)
        tags = kwargs.get("tags", None)
        result = []
        for key, entry in self.data.iteritems():
            if condition is not None:
                if not condition(entry):
                    continue
            if tags is not None:
                if entry.tags is None:
                    continue
                if not set(tags).intersection(entry.tags):
                    continue
            result.append(key)
        return result

    def generate_mock_state(self, utc_timestamp=None, tags=None):
        gs = {"albedo": 0.4,
              "lwc": 0.1,
              "surface_heat": 30000,
              "alpha": 1.26,
              "sdc_melt_mean": 1.0,
              "acc_melt": 0.0,
              "iso_pot_energy": 0.0,
              "temp_swe": 0.0,
              }
        kirchner = {"q": random.uniform}
        # format -> ptgsk:albedo alpha sdc_melt_mean acc_melt iso_pot_energy temp_swe surface_heat lwc q
        state_list = '\n'.join(
            ['ptgsk:{:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f}'.format(
                gs["albedo"],
                gs["alpha"], gs["sdc_melt_mean"], gs["acc_melt"], gs["iso_pot_energy"], gs["temp_swe"],
                gs["surface_heat"], gs["lwc"], kirchner["q"](0.1, 0.3)) for i in range(self.num_states)])
        if utc_timestamp is None:
            utc_timestamp = datetime.strftime(datetime.utcnow(), "%Y-%m-%dT%H:%M:%S")
        if tags is None:
            tags = ["mock", "test"]
        return State(state_list, utc_timestamp, tags=tags)

    def generate_mock_entry(self, key, utc_timestamp=None, tags=None):
        entry = self.generate_mock_state(utc_timestamp, tags)
        self.data[key] = entry

    @classmethod
    def factory(cls, num_states=10, **kwargs):
        instance = cls(num_states=num_states)
        key = "".join([random.choice("abcdefghijklmno") for _ in xrange(10)])
        instance.generate_mock_entry(key, **kwargs)
        return instance


def mock_state_data(n_x, n_y):
    return {"num_states": n_x*n_y}


def state_repository_factory(info):
    timestamp = datetime.strftime(datetime.fromtimestamp(info["t_start"]), "%Y-%m-%dT%H:%M:%S")
    return MockStateRepository.factory(info["num_cells"], utc_timestamp=timestamp)


class MockInputSourceRepository(MockRepository):

    def __init__(self, *args, **kwargs):
        super(MockInputSourceRepository, self).__init__()
        self.period = kwargs.get("period", None)
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
                api_source = self.input_source_types[input_source]()
                # TODO: Fix with setter_method in api/shyft_api.h!!
                api_source._geo_point = api.GeoPoint(*station.geo_location)
                api_source.ts = ts
                self.data[input_source].append(api_source)

    def generate_mock_entry(self, key, geo_location, mock_sources, name=None, tags=None):
        self.add_input_source(InputSource(geo_location, mock_sources, name=name, tags=tags))

    @classmethod
    def factory(cls, *stations):
        instance = cls()
        for station in stations:
            instance.add_input_source(station)
        return instance


def mock_station_data(bounding_box, n_x, n_y):
    geo_positions = []
    x_min, x_max, y_min, y_max = bounding_box
    dx = float(x_max - x_min)/n_x
    dy = float(y_max - y_min)/n_y

    data = []

    for i in xrange(n_x):
        x = x_min + (i + 0.5)*dx
        for j in xrange(n_y):
            y = y_min + (j + 0.5)*dy
            z = abs(1000*math.sin(x/10000.0 + y/10000.0))   # Crop at 1000 mas
            geo_positions.append([x, y, z])
            data.append(InputSource([x, y, z], create_mock_station_data(0, 3600, 10)))
    return data


def mock_get_geo_locations(stations):
    for station in stations:
        x = random.uniform(0, 5000)
        y = random.uniform(0, 5000)
        z = random.uniform(50, 1000)
        stations[station]["geo_location"] = (x, y, z)
    return stations


def mock_get_input_datasets(data_sources, t0, dt, n_steps):
    for data in data_sources:
        connector = data.pop("connector")   # We don't have a connector, so ignore here.
        for station_id, sources in data.iteritems():
            for source, uid in sources.iteritems():
                if source == "radiation":
                    mock_data = create_mock_station_data(t0,  dt, n_steps)


def dataset_repository_factory(config, t_start, t_end, dt=3600):
    n_steps = int(ceil((t_end - t_start)*1.0/dt))
    input_datasets = []
    stations = {}
    for input_source in config.sources:
        connector = input_source["repository"]
        input_datasets.append({"connector": connector})
        for data_type in input_source["types"]:
            type = data_type["type"]
            for station in data_type["stations"]:
                station_id = station["station_id"]
                if station_id not in stations:
                    stations[station_id] = {}
                uid = station["uid"]
                if station_id not in input_datasets[-1]:
                    input_datasets[-1][station_id] = {}
                input_datasets[-1][station_id][type] = uid
    stations = mock_get_geo_locations(stations)
    data_sets = mock_get_input_datasets(input_datasets, t0, dt, n_steps)


def cell_repository_factory(config):
    data = mock_cell_data(config)
    return CellReadOnlyRepository.factory(**data)


def mock_cell_data(config):
    x_min = config.x_min
    y_min = config.y_min
    dx = config.dx
    dy = config.dy
    n_x = config.n_x
    n_y = config.n_y

    geo_positions = []
    for i in xrange(n_x):
        x = x_min + (i + 0.5)*dx
        for j in xrange(n_y):
            y = y_min + (j + 0.5)*dy
            z = min(1000.0, x*100.0 + y*100.0)    # Crop at 1000 mas
            geo_positions.append([x, y, z])

    n_p = n_x*n_y
    half = n_p/2

    return {"catchment_id": half*[1] + (n_p - half)*[2],
            "geo_position": geo_positions,
            "land_type": n_p*[1],
            "glacier_fraction": n_p*[0.0],
            "reservoir_fraction": n_p*[0.0],
            "lake_fraction": n_p*[0.0],
            "forest_fraction": n_p*[0.0],
            "initial_bare_ground_fraction": n_p*[0.04],
            "snow_cv": n_p*[0.4],
            "radiation_slope_factor": n_p*[1.26],
            "c1": n_p*[-3.5],
            "c2": n_p*[0.7],
            "c3": n_p*[0.0],
            "area": n_p*[dx*dy]}
