"""
Read region netCDF files with cell data.
Note: It does require a specific content/layout of the supplied netcdf files
      this should be clearly stated.
"""

from __future__ import absolute_import

from os import path
import numpy as np
from netCDF4 import Dataset
from .. import interfaces
from shyft import api
from shyft import shyftdata_dir


class RegionModelRepository(interfaces.RegionModelRepository):
    """
    Repository that delivers fully specified shyft api region_models
    based on data found in netcdf files.

    Netcdf dataset assumptions:
        * Group "elevation" with variables:
            * xcoord: array of floats
            * ycoord: array of floats
            * elevation: float array of dim (xcoord, ycoord)
        * Group "catchments" with variables:
            * catchments: int array of dim (xcoord, ycoord)
            * catchment_indices: int array of possible indices
        * Group "forest-fraction" with variables:
            * forest-fraction: float array of dim (xcoord, ycoord)
        * Group "lake-fraction" with variables:
            * lake-fraction: float array of dim (xcoord, ycoord)
        * Group "reservoir-fraction" with variables:
            * reservoir-fraction: float array of dim (xcoord, ycoord)
        * Group "glacier-fraction" with variables:
            * glacier-fraction: float array of dim (xcoord, ycoord)
    """

    def __init__(self, region_config, model_config):
        """
        Parameters
        ----------
        region_config: subclass of interfaces.RegionConfig
            Object containing regional information, like
            catchment overrides, and which netcdf file to read
        model_config: subclass of interfaces.ModelConfig
            Object containing model information, i.e.
            information concerning interpolation and model
            parameters
        """
        if not isinstance(region_config, interfaces.RegionConfig) or \
           not isinstance(model_config, interfaces.ModelConfig):
            raise interfaces.InterfaceError()
        self._rconf = region_config
        self._mconf = model_config
        self._mask = None
        self._data_file = path.join(shyftdata_dir, self._rconf.repository()["data_file"])

    @property
    def mask(self):
        """
        Get the mask for cells that have actual info.
        Returns
        -------
            mask : np.array of type bool
        """
        if self._mask is not None:
            return self._mask
        with Dataset(self._data_file) as dset:
            mask = dset.groups['catchments'].variables[
                "catchments"][:].reshape(-1) != 0
        self._mask = mask
        return mask

    def get_region_model(self, region_id, region_model, catchments=None):
        """
        Return a fully specified shyft api region_model for region_id, based on data found
        in netcdf dataset.

        Parameters
        -----------
        region_id: string
            unique identifier of region in data
        region_model: shyft.api type
            model to construct. Has cell constructor and region/catchment
            parameter constructor.
        catchments: list of unique integers
            catchment indices when extracting a region consisting of a subset
            of the catchments
        has attribs to construct  params and cells etc.

        Returns
        -------
        region_model: shyft.api type
        """

        with Dataset(self._data_file) as dset:
            grp = dset.groups['elevation']
            xcoord = grp.variables['xcoord'][:]
            ycoord = grp.variables['ycoord'][:]
            mesh2d = np.dstack(np.meshgrid(xcoord, ycoord)).reshape(-1, 2)
            elevation = grp.variables['elevation'][:]
            coordinates = np.hstack((mesh2d, elevation.reshape(-1, 1)))[self.mask]
            areas = np.ones(len(xcoord)*len(ycoord),
                            dtype=xcoord.dtype)[self.mask]*(
                                xcoord[1] - xcoord[0])*(ycoord[1] - ycoord[0])
            catchments = dset.groups['catchments'].variables[
                "catchments"][:].reshape(-1)[self.mask]
            c_ids = dset.groups['catchments'].variables["catchment_indices"][:]

            def frac_extract(name):
                g = dset.groups  # Alias for readability
                return g[name].variables[name][:].reshape(-1)[self.mask]
            ff = frac_extract("forest-fraction")
            lf = frac_extract("lake-fraction")
            rf = frac_extract("reservoir-fraction")
            gf = frac_extract("glacier-fraction")
        # Construct region parameter:
        name_map = {"gamma_snow": "gs", "priestley_taylor": "pt",
                    "kirchner": "kirchner", "actual_evapotranspiration": "ae",
                    "skaugen": "skaugen"}
        region_parameter = region_model.parameter_t()
        for p_type_name, value_ in self._mconf.model_parameters().iteritems():
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
            cell.geo = api.GeoCellData(api.GeoPoint(*pt), a, c_id, radiation_slope_factor,
                                       api.LandTypeFractions(gf, lf, rf, ff, 0.0))
            cell_vector.append(cell)

        # Construct catchment overrides
        catchment_parameters = region_model.parameter_t.map_t()
        for k, v in self._rconf.parameter_overrides().iteritems():
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
