import requests
import unicodedata
from gis_data import BaseGisDataFetcher


class StationDataError(Exception):
    pass


class StationDataFetcher(BaseGisDataFetcher):


    def __init__(self, station_ids=None, epsg_id=32632):
        super(StationDataFetcher, self).__init__(geometry=None, server_name="oslwvagi001p", server_port="6080", service_index=4)
        self.where = "OBJECTID IN ({})"
        self.outFields = "MOH, OBJECTID, EIER, ST_NAVN"
        self.outSR = epsg_id
        

    def build_query(self,station_ids):
        q = self.get_query()
        if station_ids is None:
            q["where"] = "1 = 1"
        else:
            q["where"] = self.where.format(", ".join([str(i) for i in station_ids]))
        q["outFields"] = self.outFields
        q["outSR"] = self.outSR
        return q

    def fetch(self, station_ids):
        q = self.build_query(station_ids)
        response = requests.get(self.url, params=q)
        stations = {}
        if response.status_code == 200:
            for feature in response.json()['features']:
                index = feature["attributes"]["OBJECTID"]
                x = feature["geometry"]["x"]
                y = feature["geometry"]["y"]
                z = feature["attributes"]["MOH"]
                name = unicodedata.normalize('NFKC', feature["attributes"]["ST_NAVN"])
                name = str(unicode(name).encode("ascii", errors="replace"))

                stations[index] = ((x,y,z), {"owner": feature["attributes"]["EIER"],"name": name})
        else:
            raise StationDataError("Could not get data from GIS service!")
        return stations


def _main():

    station_ids = [7] #,678,506,217,503,421,489,574,598,610,121,423
    sf = StationDataFetcher(epsg_id=32632)
    stations = sf.fetch(station_ids=station_ids)
    assert len(stations) == len(station_ids)
    for i in station_ids:
        assert i in stations

if __name__ == "__main__":
    _main()
