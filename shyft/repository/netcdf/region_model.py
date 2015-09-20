"""
    Module for reading region netCDF files needed for an SHyFT run.
    note: it do require a specific content/layout of the supplied netcdf files
          this should be clearly stated
"""

from __future__ import absolute_import

import numpy as np
from netCDF4 import Dataset
from ..interfaces import RegionModelRepository
from ..yaml_config import BaseYamlConfig
#from ..base_config import BaseRegion
#from ..base_config import RegionModelRepository
#from ..base_config import BaseAncillaryConfig
from shyft import api


class NetCDFRegionModelRepository(RegionModelRepository):

    def __init__(self, region_config_file, model_config_file, data_file):
        self._rconf = BaseYamlConfig(region_config_file)
        self._mconf = BaseYamlConfig(model_config_file)
        self._mask = None
        self._data_file = data_file

    @property
    def mask(self):
        """Get the mask for cells that have actual info."""
        if self._mask is not None:
            return self._mask
        with Dataset(self._data_file) as dset:
            mask = dset.groups['catchments'].variables[
                "catchments"][:].reshape(-1) != 0
        self._mask = mask
        return mask

    def get_region_model(self, region_id, region_model, catchments=None):
        with Dataset(self._data_file) as dset:
            grp = dset.groups['elevation']
            xcoord = grp.variables['xcoord'][:]
            ycoord = grp.variables['ycoord'][:]
            mesh2d = np.dstack(np.meshgrid(xcoord, ycoord)).reshape(-1, 2)
            elevation = grp.variables['elevation'][:]
            coordinates = np.hstack((mesh2d,
                                     elevation.reshape(-1, 1)))[self.mask]
            areas = np.ones(len(xcoord)*len(ycoord),
                            dtype=xcoord.dtype)[self.mask]*(
                                xcoord[1] - xcoord[0])*(
                                ycoord[1] - ycoord[0])
            catchments = dset.groups['catchments'].variables[
                "catchments"][:].reshape(-1)[self.mask]
            c_ids = dset.groups['catchments'].variables["catchment_indices"][:]
            ff = dset.groups["forest-fraction"].variables["forest-fraction"][:].reshape(-1)[self.mask]
            lf = dset.groups["lake-fraction"].variables["lake-fraction"][:].reshape(-1)[self.mask]
            rf = dset.groups["reservoir-fraction"].variables["reservoir-fraction"][:].reshape(-1)[self.mask]
            gf = dset.groups["glacier-fraction"].variables["glacier-fraction"][:].reshape(-1)[self.mask]
        # Construct region parameter:
        name_map = {"gamma_snow": "gs", "priestley_taylor": "pt",
                    "kirchner": "kirchner", "actual_evapotranspiration": "ae",
                    "skaugen": "skaugen"}
        region_parameter = region_model.parameter_t()
        for p_type_name, value_ in self._mconf.parameters["model"].iteritems():
            if p_type_name in name_map:
                sub_param = getattr(region_parameter, name_map[p_type_name])
                for p, v in value_.iteritems():
                    setattr(sub_param, p, v)
            elif p_type_name == "p_corr_scale_factor":
                region_parameter.p_corr.scale_factor = value_

        # TODO: Move into yaml file similar to p_corr_scale_factor
        radiation_slope_factor = 0.9

        # Construct cells
        cell_vector = region_model.cell_t.vector_t()
        for pt, a, c_id, ff, lf, rf, gf in zip(coordinates, areas, catchments, ff, lf, rf, gf):
            cell = region_model.cell_t()
            cell.geo = api.GeoCellData(api.GeoPoint(*pt),
                                       a, c_id, radiation_slope_factor,
                                       api.LandTypeFractions(gf, lf, rf, ff, 0.0))
            cell_vector.append(cell)

        # Construct catchment overrides
        catchment_parameters = region_model.parameter_t.map_t()
        for k, v in self._rconf.parameter_overrides.iteritems():
            if k in c_ids:
                param = region_model.parameter_t(region_parameter)
                for p_type_name, value_ in v.iteritems():
                    if p_type_name in name_map:
                        sub_param = getattr(param, name_map[p_type_name])
                        for p, v in value_.iteritems():
                            setattr(sub_param, p, v)
                    elif p_type_name == "p_corr_scale_factor":
                        param.p_corr.scale_factor = value_
                catchment_parameters[k] = param
        return region_model(cell_vector, region_parameter, catchment_parameters)

