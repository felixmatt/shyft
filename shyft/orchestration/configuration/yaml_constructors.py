# -*- coding: utf-8 -*-

class ConfigError(Exception):
    pass

def cls_path(cls):
    return cls.__module__+'.'+cls.__name__

def target_repo_constructor(cls, params):
    return cls(**params)

def geo_ts_repo_constructor(cls, params,region_config):
    if cls_path(cls) == 'shyft.repository.service.ssa_geo_ts_repository.GeoTsRepository':
        # from shyft.repository.service.ssa_geo_ts_repository import GeoTsRepository
        from shyft.repository.service.ssa_geo_ts_repository import MetStationConfig
        from shyft.repository.service.gis_location_service import GisLocationService
        from shyft.repository.service.ssa_smg_db import SmGTsRepository, PROD, FC_PROD

        epsg = region_config.domain()["EPSG"]
        met_stations = [MetStationConfig(**s) for s in params['stations_met']]
        gis_location_repository=GisLocationService() # this provides the gis locations for my stations
        smg_ts_repository = SmGTsRepository(PROD,FC_PROD) # this provide the read function for my time-series
        # return GeoTsRepository(epsg_id=epsg, geo_location_repository=gis_location_repository,
        #                        ts_repository=smg_ts_repository, met_station_list=met_stations,
        #                        ens_config=None)
        return cls(epsg_id=epsg, geo_location_repository=gis_location_repository,
                               ts_repository=smg_ts_repository, met_station_list=met_stations,
                               ens_config=None)
    else:
        params.update({'epsg': region_config.domain()["EPSG"]})
        return cls(**params)

def region_model_repo_constructor(cls,region_config, model_config, region_model_id):
    if cls_path(cls) == 'shyft.repository.service.gis_region_model_repository.GisRegionModelRepository':
        #from shyft.repository.service.gis_region_model_repository import GisRegionModelRepository
        from shyft.repository.service.gis_region_model_repository import RegionModelConfig
        from shyft.repository.service.gis_region_model_repository import GridSpecification
        from six import iteritems # This replaces dictionary.iteritems() on Python 2 and dictionary.items() on Python 3

        repo_params = region_config.repository()['params']
        c_ids = region_config.catchments()
        d = region_config.domain()
        grid_specification = GridSpecification(d['EPSG'],
                                               d['lower_left_x'], d['lower_left_y'],
                                               d['step_x'], d['step_y'], d['nx'], d['ny'])
        region_model_type = model_config.model_type()
        # Construct region parameter:
        name_map = {"priestley_taylor": "pt", "kirchner": "kirchner",
                    "precipitation_correction": "p_corr", "actual_evapotranspiration": "ae",
                    "gamma_snow": "gs", "skaugen_snow": "ss", "hbv_snow": "hs"}
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
                              region_config.catchments(), catchment_parameters=catchment_parameters),
        ]
        rm_cfg_dict = {x.name: x for x in cfg_list}
        # return GisRegionModelRepository(rm_cfg_dict)
        return cls(rm_cfg_dict)
    else:
        return cls(region_config, model_config)
