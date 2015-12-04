"""
Module with specific logic for NetCDF files.
"""

import os

from shyft.repository.netcdf import (
    RegionModelRepository, GeoTsRepository, get_geo_ts_collection, yaml_config)
from shyft.repository.interpolation_parameter_repository import (
    InterpolationParameterRepository)
from shyft.orchestration import DefaultSimulator


def get_simulator(cfg):
    """
    Return a DefaultSimulator based on `cfg`.

    Parameters
    ----------
    cfg : YAMLConfig instance
      Instance with the information for the simulation.

    Returns
    -------
    DefaultSimulator instance
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
    geo_ts = get_geo_ts_collection(datasets_config, cfg.data_dir)

    # some fake ids
    region_id = 0
    interpolation_id = 0
    # set up the simulator
    simulator = DefaultSimulator(region_id, interpolation_id, region_model,
                                 geo_ts, interp_repos, None)
    return simulator
