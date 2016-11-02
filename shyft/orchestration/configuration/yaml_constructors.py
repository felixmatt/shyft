# -*- coding: utf-8 -*-
import os

class ConfigError(Exception):
    pass

def cls_path(cls):
    return cls.__module__+'.'+cls.__name__

def target_repo_constructor(cls, params):
    return cls(**params)

def geo_ts_repo_constructor(cls, params): # ,region_config):
    if cls_path(cls) == 'shyft.repository.service.ssa_geo_ts_repository.GeoTsRepository':
        # from shyft.repository.service.ssa_geo_ts_repository import GeoTsRepository
        from shyft.repository.service.ssa_geo_ts_repository import MetStationConfig
        from shyft.repository.service.gis_location_service import GisLocationService
        from shyft.repository.service.ssa_smg_db import SmGTsRepository, PROD, FC_PROD

        #epsg = region_config.domain()["EPSG"]
        epsg = params['epsg']
        met_stations = [MetStationConfig(**s) for s in params['stations_met']]
        gis_location_repository=GisLocationService(server_name=params.get('server_name', None),
                              server_name_preprod=params.get('server_name_preprod', None)) # this provides the gis locations for my stations
        smg_ts_repository = SmGTsRepository(PROD,FC_PROD) # this provide the read function for my time-series
        # return GeoTsRepository(epsg_id=epsg, geo_location_repository=gis_location_repository,
        #                        ts_repository=smg_ts_repository, met_station_list=met_stations,
        #                        ens_config=None)
        return cls(epsg_id=epsg, geo_location_repository=gis_location_repository,
                               ts_repository=smg_ts_repository, met_station_list=met_stations,
                               ens_config=None)
    else:
        #params.update({'epsg': region_config.domain()["EPSG"]})
        return cls(**params)

def region_model_repo_constructor(cls,region_config, model_config, region_model_id):
    if cls_path(cls) == 'shyft.repository.service.gis_region_model_repository.GisRegionModelRepository':
        #from shyft.repository.service.gis_region_model_repository import GisRegionModelRepository
        from shyft.repository.service.gis_region_model_repository import get_grid_spec_from_catch_poly
        from shyft.repository.service.gis_region_model_repository import RegionModelConfig
        from shyft.repository.service.gis_region_model_repository import GridSpecification
        from six import iteritems # This replaces dictionary.iteritems() on Python 2 and dictionary.items() on Python 3

        repo_params = region_config.repository()['params']
        server_name = repo_params.get('server_name')
        server_name_preprod = repo_params.get('server_name_preprod')
        use_cache = repo_params.get('use_cache', False)
        cache_folder = repo_params.get('cache_folder', None)
        cache_folder = cache_folder.replace('${SHYFTDATA}', os.getenv('SHYFTDATA', '.'))
        cache_file_type = repo_params.get('cache_file_type', None)
        calc_forest_frac = repo_params.get('calc_forest_frac', False)

        c_ids = region_config.catchments()
        d = region_config.domain()
        get_bbox_from_catchment_boundary = d.get('get_bbox_from_catchment_boundary', False)
        pad = d.get('buffer', 5)
        epsg_id = d['EPSG']
        dx, dy = [d['step_x'], d['step_y']]
        if use_cache or get_bbox_from_catchment_boundary:
            if dx != dy:
                raise ConfigError("step_x({}) and step_y({}) should be the same "
                                  "if 'use_cache' or 'get_bbox_from_catchment_boundary' is enabled".format(dx, dy))
        if get_bbox_from_catchment_boundary:
            grid_specification = get_grid_spec_from_catch_poly(c_ids, repo_params['catchment_regulated_type'],
                                                               repo_params['service_id_field_name'], epsg_id, dx, pad,
                                                               server_name=server_name, server_name_preprod=server_name_preprod)
        else:
            grid_specification = GridSpecification(epsg_id, d['lower_left_x'], d['lower_left_y'],
                                                   dx, dy, d['nx'], d['ny'])
        region_model_type = model_config.model_type()
        # Construct region parameter:
        name_map = {"priestley_taylor": "pt", "kirchner": "kirchner",
                    "precipitation_correction": "p_corr", "actual_evapotranspiration": "ae",
                    "gamma_snow": "gs", "skaugen_snow": "ss", "hbv_snow": "hs", "glacier_melt": "gm" }
        region_parameter = region_model_type.parameter_t()
        for p_type_name, value_ in iteritems(model_config.model_parameters()):
            if p_type_name in name_map:
                if hasattr(region_parameter, name_map[p_type_name]):
                    sub_param = getattr(region_parameter, name_map[p_type_name])
                    for p, v in iteritems(value_):
                        if hasattr(sub_param, p):
                            setattr(sub_param, p, v)
                        else:
                            raise ConfigError("Invalid parameter '{}' for parameter set '{}'".format(p, p_type_name))
                else:
                    raise ConfigError("Invalid parameter set '{}' for selected model '{}'".format(p_type_name, region_model_type.__name__))
            else:
                raise ConfigError("Unknown parameter set '{}'".format(p_type_name))

        # Construct catchment overrides
        catchment_parameters = {}
        for c_id, catch_param in iteritems(region_config.parameter_overrides()):
            if c_id in c_ids:
                param = region_model_type.parameter_t(region_parameter)
                for p_type_name, value_ in iteritems(catch_param):
                    if p_type_name in name_map:
                        if hasattr(param, name_map[p_type_name]):
                            sub_param = getattr(param, name_map[p_type_name])
                            for p, v in iteritems(value_):
                                if hasattr(sub_param, p):
                                    setattr(sub_param, p, v)
                                else:
                                    raise ConfigError("Invalid parameter '{}' for catchment parameter set '{}'".format(p, p_type_name))
                        else:
                            raise ConfigError("Invalid catchment parameter set '{}' for selected model '{}'".format(p_type_name, region_model_type.__name__))
                    else:
                        raise ConfigError("Unknown catchment parameter set '{}'".format(p_type_name))

                catchment_parameters[c_id] = param

        cfg_list=[
            RegionModelConfig(region_model_id, region_model_type, region_parameter, grid_specification,
                              repo_params['catchment_regulated_type'], repo_params['service_id_field_name'],
                              region_config.catchments(), catchment_parameters=catchment_parameters,
                              calc_forest_frac=calc_forest_frac),
        ]
        rm_cfg_dict = {x.name: x for x in cfg_list}
        # return GisRegionModelRepository(rm_cfg_dict)
        if server_name is not None:
            cls.server_name = repo_params.get('server_name')
        if server_name_preprod is not None:
            cls.server_name_preprod = repo_params.get('server_name_preprod')
        return cls(rm_cfg_dict, use_cache=use_cache, cache_folder=cache_folder, cache_file_type=cache_file_type)
    else:
        return cls(region_config, model_config)
