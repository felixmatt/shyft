from __future__ import print_function
from __future__ import absolute_import
from . import interfaces

class GeoTsRepositoryCollection(interfaces.GeoTsRepository):

    def __init__(self, geo_ts_repositories, reduce_type="replace"):
        if reduce_type not in ("replace", "add"):
            raise GeoTsRepositoryCollectionError("reduce_type must be either 'replace' or 'add'")
        self.reduce_type = reduce_type
        self.geo_ts_repositories = geo_ts_repositories

    def get_timeseries(self, input_source_types, utc_period, geo_location_criteria=None):
        tss = [r.get_timeseries(input_source_types, utc_period, geo_location_criteria)
               for r in self.geo_ts_repositories]
        if self.reduce_type == "replace":
            sources = tss[0]
            for ts in tss[1:]:
                sources.update(ts)
        else:
            raise GeoTsRepositoryCollectionError("Only replace is supported yet")
        return sources

    def get_forecast(self, input_source_types, utc_period, t_c, geo_location_criteria=None):
        tss = [r.get_forecast(input_source_types, utc_period, t_c, geo_location_criteria)
               for r in self.geo_ts_repositories]
        if self.reduce_type == "replace":
            sources = tss[0]
            for ts in tss[1:]:
                sources.update(ts)
        else:
            raise GeoTsRepositoryCollectionError("Only replace is supported yet")
        return sources

    def get_forecast_ensemble(self, input_source_types, utc_period,
                              t_c, geo_location_criteria=None):
        ensembles = [r.get_forecast_ensemble(input_source_types, utc_period, t_c, 
                                             geo_location_criteria)
                     for r in self.geo_ts_repositories]
        ensemble = ensembles[0]
        for ens in ensembles[1:]:
            for i, s2 in enumerate(ens):
                if self.reduce_type == "replace":
                    ensemble[i].update(s2)
                else:
                    raise GeoTsRepositoryCollectionError("Only replace is supported yet")
        return ensemble


