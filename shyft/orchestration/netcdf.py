"""
Module with specific logic for NetCDF files.
"""

from shyft.repository import geo_ts_repository_collection
from shyft.repository.netcdf import RegionModelRepository, GeoTsRepository
from shyft.repository.interpolation_parameter_repository import (
    InterpolationParameterRepository)
from shyft.orchestration import SimpleSimulator


def get_simulator(cfg, *params):
    # Build some interesting constructs
    region_model = RegionModelRepository(
        cfg.region_config, cfg.model_config, cfg.model_t, cfg.epsg)
    interp_repos = InterpolationParameterRepository(cfg.model_config)
    netcdf_geo_ts_repos = []
    for source in cfg.datasets_config.sources:
        station_file = source["params"]["stations_met"]
        netcdf_geo_ts_repos.append(
            GeoTsRepository(source["params"], station_file, ""))
    geo_ts = geo_ts_repository_collection.GeoTsRepositoryCollection(
        netcdf_geo_ts_repos)

    # some fake ids
    region_id = 0
    interpolation_id = 0
    # set up the simulator
    simulator = SimpleSimulator(region_id, interpolation_id, region_model,
                                geo_ts, interp_repos, None)
    return simulator
