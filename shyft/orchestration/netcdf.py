"""
Module with specific logic for NetCDF files.
"""

import os

from shyft.repository import geo_ts_repository_collection
from shyft.repository.netcdf import (
    RegionModelRepository, GeoTsRepository, yaml_config)
from shyft.repository.interpolation_parameter_repository import (
    InterpolationParameterRepository)
from shyft.orchestration import SimpleSimulator


def get_simulator(cfg):
    """
    Return a SimpleSimulator based on `cfg`.

    Parameters
    ----------
    cfg : YAMLConfig instance
      Instance with the information for the simulation.

    Returns
    -------
    SimpleSimulator instance
    """
    # Read region, model and datasets config files
    region_config_file = os.path.join(
        cfg.config_dir, cfg.region_config_file)
    region_config = yaml_config.RegionConfig(region_config_file)
    model_config_file = os.path.join(
        cfg.config_dir, cfg.model_config_file)
    model_config = yaml_config.ModelConfig(model_config_file)
    datasets_config_file = os.path.join(
        cfg.config_dir, cfg.datasets_config_file)
    datasets_config = yaml_config.YamlContent(datasets_config_file)

    # Build some interesting constructs
    region_model = RegionModelRepository(
        region_config, model_config, cfg.model_t, cfg.epsg)
    interp_repos = InterpolationParameterRepository(model_config)
    netcdf_geo_ts_repos = []
    for source in datasets_config.sources:
        station_file = source["params"]["stations_met"]
        if not os.path.isabs(station_file):
            # Relative paths will be prepended the cfg.data_dir
            station_file = os.path.join(cfg.data_dir, station_file)
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
