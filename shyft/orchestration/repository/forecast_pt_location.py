# -*- coding: utf-8 -*-

import yaml

class ForecastLocationError(Exception): pass

class ForecastLocationFetcher(object):

    def __init__(self, indices=None, epsg_id=32632):
        self.indices = indices
        
    def read_forecast_series_loc(self):
        f_path="D:/Users/ysa/config_auto/forecast_pt_locations-UTM32N.yml"
        return yaml.load(open(f_path, 'r'))

    def fetch(self, **kwargs):
        loc_dict = self.read_forecast_series_loc()
        locations = {}
        for index in self.indices:
            if loc_dict.get(index) != None:
                locations[index] = (tuple(loc_dict[index]),{})
                print locations[index]
#            name = unicodedata.normalize('NFKC', feature["attributes"]["ST_NAVN"])
#            name = str(unicode(name).encode("ascii", errors="replace"))
#            locations[index] = ((x,y,z), {"owner": feature["attributes"]["EIER"],
#                                         "name": name})
            else:
                raise ForecastLocationError("Could not get location of forecast point!")
        return locations
        
if __name__ == "__main__":
    fetcher=ForecastLocationFetcher([619])
    print fetcher.fetch()