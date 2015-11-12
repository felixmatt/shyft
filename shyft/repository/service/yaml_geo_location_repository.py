# -*- coding: utf-8 -*-

from __future__ import absolute_import
import yaml
from .ssa_geo_ts_repository import GeoLocationRepository
from os import path


class YamlGeoLocationError(Exception):
    pass


class YamlGeoLocationRepository(GeoLocationRepository):
    """
    Provide a yaml-based key-location map for gis-identites not available(yet)

    """

    def __init__(self, yaml_file_dir):
        """
        Parameters
        ----------
        yaml_file_dir:string
            path to directory where files 
            pt_locations-epsg_32632.yml (UTM32N) and
            pt_locations-epsg_32633.yml (UTM33N)
            pt_locations-<epsg_id>.yml
        """
        self._file_dir = yaml_file_dir

    def read_location_dict(self, epsg_id):
        full_name = path.join(self._file_dir, _filename_of(epsg_id))
        with open(full_name, 'r') as f:
            return yaml.load(f)

    def get_locations(self, location_id_list, epsg_id=32632):
        loc_dict = self.read_location_dict(epsg_id)
        locations = {}
        for index in location_id_list:
            if loc_dict.get(index) is not None:
                locations[index] = tuple(loc_dict[index])
            else:
                raise YamlGeoLocationError("Could not get location of geo point-id!")
        return locations


def _filename_of(epsg_id):
    return "pt_locations-epsg_{}.yml".format(epsg_id)
