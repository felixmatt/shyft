import os
#from shyft.repository.netcdf import cf_region_model_repository, cf_geo_ts_repository
from shyft.repository.netcdf.cf_region_model_repository import CFRegionModelRepository
from shyft.repository.netcdf.cf_geo_ts_repository import CFDataRepository
from shyft.repository.netcdf.cf_ts_repository import CFTsRepository
#from shyft.repository import geo_ts_repository_collection

def cls_path(cls):
    return cls.__module__+'.'+cls.__name__

def nc_geo_ts_repo_constructor(params,region_config):
    return CFDataRepository(params,region_config)

def nc_region_model_repo_constructor(region_config, model_config):
    return CFRegionModelRepository(region_config, model_config)

def nc_target_repo_constructor(params):
    return CFTsRepository(params)

r_m_repo_constructors = {cls_path(CFRegionModelRepository): nc_region_model_repo_constructor}
geo_ts_repo_constructors = {cls_path(CFDataRepository): nc_geo_ts_repo_constructor}
target_repo_constructors = {cls_path(CFTsRepository): nc_target_repo_constructor}
