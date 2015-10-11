from __future__ import absolute_import
import requests
import unicodedata
from .gis_region_model_repository import BaseGisDataFetcher
from .ssa_geo_ts_repository import GeoLocationRepository

class StationDataError(Exception):
    pass


class GisLocationService(GeoLocationRepository):
    """
    Statkraft specific Geo Location Repository based on internally 
    published arc-gis services

    """

    def __init__(self, server_name="oslwvagi001p", server_port="6080", service_index=4 ):
        super(GeoLocationRepository, self).__init__()
        self.base_fetcher= BaseGisDataFetcher(geometry=None, server_name=server_name, server_port=server_port, service_index=service_index)
        self.where = "OBJECTID IN ({})"
        self.outFields = "MOH, OBJECTID, EIER, ST_NAVN"

        

    def build_query(self,station_ids,epsg_id):
        q = self.base_fetcher.get_query()
        if station_ids is None:
            q["where"] = "1 = 1"
        else:
            q["where"] = self.where.format(", ".join([str(i) for i in station_ids]))
        q["outFields"] = self.outFields
        q["outSR"] = epsg_id
        return q
    
    def get_locations(self, location_id_list,epsg_id=32632):
        """ contract implementation """
        return self.get_locations_and_info(location_id_list,epsg_id)[0]

    def get_locations_and_info(self, location_id_list,epsg_id=32632):
        """ 
        might be useful for ui/debug etc. 
        Returns
        -------
        tuple(location-dict(station:position),info-dict(station:info-dict))

        """
        self.outSR = epsg_id
        q = self.build_query(location_id_list,epsg_id)
        response = requests.get(self.base_fetcher.url, params=q)
        locations = {}
        station_info={}
        if response.status_code == 200:
            for feature in response.json()['features']:
                index = feature["attributes"]["OBJECTID"]
                x = feature["geometry"]["x"]
                y = feature["geometry"]["y"]
                z = feature["attributes"]["MOH"]
                name = unicodedata.normalize('NFKC', feature["attributes"]["ST_NAVN"])
                name = str(unicode(name).encode("ascii", errors="replace"))

                locations[index] = (x,y,z)
                station_info[index]= {"owner": feature["attributes"]["EIER"],"name": name}
        else:
            raise StationDataError("Could not get data from GIS service!")
        return locations,station_info


def _main():

    station_ids = [7] #,678,506,217,503,421,489,574,598,610,121,423
    sf = GisLocationService(epsg_id=32632)
    stations = sf.get_locations(location_id_list=station_ids)
    assert len(stations) == len(station_ids)
    for i in station_ids:
        assert i in stations

if __name__ == "__main__":
    _main()
