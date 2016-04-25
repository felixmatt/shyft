from __future__ import print_function
from __future__ import absolute_import
from . import interfaces

class GeoTsRepositoryCollectionError(Exception):
    pass

class GeoTsRepositoryCollection(interfaces.GeoTsRepository):
    """
    In many situations we need to to combine geo-located time-series from
    different sources, - usually represented as already existing
    GeoTsRepository implementations.
    
    This class does exactly that, it keeps a collection of GeoTsRepository
    - when asked for time_series/forecast/ensembles,
     - it returns the combination of time-series found in the respective
       repository.
    We have started out providing to simple hopefully useful ways of combining
    the results:
    add: - the result will be the union of the data provided by the geo-ts-repositories
    replace: the geo-ts from the last geo-ts-repository will replace
             the one preceeding in the list.
             
    """

    def __init__(self, geo_ts_repositories, reduce_type="replace", src_types_per_repo=None):
        if reduce_type not in ("replace", "add"):
            raise GeoTsRepositoryCollectionError("reduce_type must be either 'replace' or 'add'")
        self.reduce_type = reduce_type
        self.geo_ts_repositories = geo_ts_repositories
        self.src_types_per_repo = src_types_per_repo

    def get_timeseries(self, input_source_types, utc_period, geo_location_criteria=None):
        src_types = self._get_src_types_per_repo(input_source_types)
        tss = [r.get_timeseries(src_type, utc_period, geo_location_criteria)
               for r, src_type in zip(self.geo_ts_repositories, src_types)]
        if self.reduce_type == "replace":
            sources = tss[0]
            for ts in tss[1:]:
                sources.update(ts)
        else:
            raise GeoTsRepositoryCollectionError("Only replace is supported yet")
        return sources

    def get_forecast(self, input_source_types, utc_period, t_c, geo_location_criteria=None):
        src_types = self._get_src_types_per_repo(input_source_types)
        tss = [r.get_forecast(src_type, utc_period, t_c, geo_location_criteria)
               for r, src_type in zip(self.geo_ts_repositories, src_types)]
        if self.reduce_type == "replace":
            sources = tss[0]
            for ts in tss[1:]:
                sources.update(ts)
        else:
            raise GeoTsRepositoryCollectionError("Only replace is supported yet")
        return sources

    def get_forecast_ensemble(self, input_source_types, utc_period,
                              t_c, geo_location_criteria=None):
        src_types = self._get_src_types_per_repo(input_source_types)
        ensembles = [r.get_forecast_ensemble(src_type, utc_period, t_c,
                                             geo_location_criteria)
                     for r, src_type in zip(self.geo_ts_repositories, src_types)]
        ensemble = ensembles[0]
        for ens in ensembles[1:]:
            for i, s2 in enumerate(ens):
                if self.reduce_type == "replace":
                    ensemble[i].update(s2)
                else:
                    raise GeoTsRepositoryCollectionError("Only replace is supported yet")
        return ensemble

    def _get_src_types_per_repo(self, input_source_types):
        if self.src_types_per_repo is None:
            return [input_source_types for _ in self.geo_ts_repositories]
        else:
            return [[name for name in types if name in input_source_types] for types in self.src_types_per_repo]


