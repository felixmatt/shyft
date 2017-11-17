from __future__ import absolute_import
import requests
import unicodedata
from .gis_region_model_repository import BaseGisDataFetcher, primary_server, secondary_server, port_num
from .ssa_geo_ts_repository import GeoLocationRepository
from .gis_region_model_repository import nordic_service, peru_service


class StationDataError(Exception):
    pass


class GisLocationService(GeoLocationRepository):
    """
    Statkraft specific Geo Location Repository based on internally 
    published arc-gis services

    """

    def __init__(self, server_name=primary_server, server_name_preprod=secondary_server, server_port=port_num, service_index=5,
                 sub_service=nordic_service, out_fields=[], return_all_fields=False):
        super(GeoLocationRepository, self).__init__()
        self.server_name=server_name
        self.server_name_preprod=server_name_preprod
        if server_name is not None:
            self.server_name = server_name
        if server_name_preprod is not None:
            self.server_name_preprod = server_name_preprod
        self.server_port=server_port
        self.sub_service=sub_service
        self.service_index=service_index

        if return_all_fields:
            self.out_fields = '*'
        else:
            if self.sub_service == peru_service:
                out_fields_list = ["STATION_ID", "NAME", "ELEVATION"]
            else:
                out_fields_list = ["MOH", "GIS_ID", "EIER", "ST_NAVN"]
            out_fields_list.extend(out_fields)
            self.out_fields = ', '.join(out_fields_list)

    def _get_response(self, url, **kwargs):
        kwargs.update({'verify': False})  # to go around authentication error when using https -> ssl.SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:749)
        response = requests.get(url, **kwargs)
        if response.status_code != 200:
            raise StationDataError("Could not get data from GIS service!")
        data = response.json()
        if "features" not in data:
            raise StationDataError(
                "GeoJson data missing mandatory field, please check your gis service or your query.")
        return data

    def build_query(self, base_fetcher, station_ids, epsg_id):
        q = base_fetcher.get_query()
        id_field = "STATION_ID" if self.sub_service == peru_service else "GIS_ID"
        if station_ids is None:
            q["where"] = "1 = 1"
        else:
            q["where"] = "{} IN ({})".format(id_field, ", ".join([str(i) for i in station_ids]))
        # q["outFields"] = "MOH, GIS_ID, EIER, ST_NAVN"
        q["outFields"] = self.out_fields
        q["outSR"] = epsg_id
        return q
    
    def get_locations(self, location_id_list,epsg_id,geometry=None):
        """ contract implementation """
        return self.get_locations_and_info(location_id_list,epsg_id,geometry=geometry)[0]

    def get_locations_and_info(self, location_id_list,epsg_id,geometry=None):
        """ 
        might be useful for ui/debug etc. 
        Returns
        -------
        tuple(location-dict(station:position),info-dict(station:info-dict))

        """
        base_fetcher= BaseGisDataFetcher(epsg_id=epsg_id,geometry=geometry, server_name=self.server_name,
                                         server_name_preprod=self.server_name_preprod,
                                         server_port=self.server_port, service_index=self.service_index,
                                         sub_service=self.sub_service)
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

        id_field = "STATION_ID" if self.sub_service == peru_service else "GIS_ID"
        name_field = "NAME" if self.sub_service == peru_service else "ST_NAVN"
        elevation_field = "ELEVATION" if self.sub_service == peru_service else "MOH"

        for feature in data['features']:
            index = feature["attributes"][id_field]
            x = feature["geometry"]["x"]
            y = feature["geometry"]["y"]
            z = feature["attributes"][elevation_field]
            name = unicodedata.normalize('NFKC', feature["attributes"][name_field])
            name = str(str(name).encode("ascii", errors="replace"))

            locations[index] = (x,y,z)
            station_info[index] = {k: v for k,v in feature["attributes"].items() if k not in ["EIER", "ST_NAVN"]}
            if self.sub_service != peru_service:
                station_info[index].update({"owner": feature["attributes"]["EIER"],"name": name})
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
