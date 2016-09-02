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

    def __init__(self, server_name=None, server_name_preprod = None, server_port="6080", service_index=5 ):
        super(GeoLocationRepository, self).__init__()
        self.server_name="oslwvagi002p"
        self.server_name_preprod="oslwvagi001q"
        if server_name is not None:
            self.server_name = server_name
        if server_name_preprod is not None:
            self.server_name_preprod = server_name_preprod
        self.server_port=server_port
        self.service_index=service_index

    def _get_response(self, url, **kwargs):
        response = requests.get(url, **kwargs)
        if response.status_code != 200:
            raise StationDataError("Could not get data from GIS service!")
        data = response.json()
        if "features" not in data:
            raise StationDataError(
                "GeoJson data missing mandatory field, please check your gis service or your query.")
        return data

    def build_query(self,base_fetcher,station_ids,epsg_id):
        q = base_fetcher.get_query()
        if station_ids is None:
            q["where"] = "1 = 1"
        else:
            q["where"] = "OBJECTID IN ({})".format(", ".join([str(i) for i in station_ids]))
        q["outFields"] = "MOH, OBJECTID, EIER, ST_NAVN"
        q["outSR"] = epsg_id
        return q
    
    def get_locations(self, location_id_list,epsg_id):
        """ contract implementation """
        return self.get_locations_and_info(location_id_list,epsg_id)[0]

    def get_locations_and_info(self, location_id_list,epsg_id):
        """ 
        might be useful for ui/debug etc. 
        Returns
        -------
        tuple(location-dict(station:position),info-dict(station:info-dict))

        """
        base_fetcher= BaseGisDataFetcher(epsg_id=epsg_id,geometry=None, server_name=self.server_name,
                                         server_name_preprod=self.server_name_preprod,
                                         server_port=self.server_port, service_index=self.service_index)
        q = self.build_query(base_fetcher,location_id_list,epsg_id)
        # response = requests.get(base_fetcher.url, params=q)
        locations = {}
        station_info={}
        try:
            data = self._get_response(base_fetcher.url, params=q)
        except Exception as e:
            print('Error in fetching GIS data: {}'.format('Station locations'))
            print('Error description: {}'.format(str(e)))
            print('Switching from PROD server {} to PREPROD server {}'.format(self.server_name, self.server_name_preprod))
            data = self._get_response(base_fetcher.url.replace(base_fetcher.server_name, base_fetcher.server_name_preprod), params=q)
        # if response.status_code == 200:
        for feature in data['features']:
            index = feature["attributes"]["OBJECTID"]
            x = feature["geometry"]["x"]
            y = feature["geometry"]["y"]
            z = feature["attributes"]["MOH"]
            name = unicodedata.normalize('NFKC', feature["attributes"]["ST_NAVN"])
            name = str(str(name).encode("ascii", errors="replace"))

            locations[index] = (x,y,z)
            station_info[index]= {"owner": feature["attributes"]["EIER"],"name": name}
        # else:
        #     raise StationDataError("Could not get data from GIS service!")
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
