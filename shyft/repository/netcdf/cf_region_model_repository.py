"""
Read region netCDF files with cell data.

"""

from __future__ import absolute_import
from six import iteritems

#from abc import ABCMeta, abstractmethod

from os import path
import numpy as np
from netCDF4 import Dataset
from pyproj import Proj
from pyproj import transform
from .. import interfaces
from shyft import api
from shyft import shyftdata_dir
from shyft.orchestration.configuration.config_interfaces import RegionConfig, ModelConfig, RegionConfigError

class CFRegionModelRepositoryError(Exception):
    pass

class CFRegionModelRepository(interfaces.RegionModelRepository):
    """
    Repository that delivers fully specified shyft api region_models
    based on data found in netcdf files.
    """

    def __init__(self, region_config, model_config):
        """
        Parameters
        ----------
        region_config: subclass of interface RegionConfig
            Object containing regional information, like
            catchment overrides, and which netcdf file to read
        model_config: subclass of interface ModelConfig
            Object containing model information, i.e.
            information concerning interpolation and model
            parameters
        """
        if not isinstance(region_config, RegionConfig) or \
           not isinstance(model_config, ModelConfig):
            raise interfaces.InterfaceError()
        self._rconf = region_config
        self._mconf = model_config
        self._region_model = model_config.model_type() # region_model
        self._mask = None
        self._epsg = self._rconf.domain()["EPSG"] # epsg
        self._data_file = path.join(shyftdata_dir, self._rconf.repository()["data_file"])
        self._catch_ids = self._rconf.catchments()
        self.bounding_box = None

    def _limit(self, x, y, data_cs, target_cs):
        """
        Parameters
        ----------
        """
        # Get coordinate system for arome data
        data_proj = Proj(data_cs)
        target_proj = Proj(target_cs)

        # Find bounding box in arome projection
        bbox = self.bounding_box
        bb_proj = transform(target_proj, data_proj, bbox[0], bbox[1])
        x_min, x_max = min(bb_proj[0]), max(bb_proj[0])
        y_min, y_max = min(bb_proj[1]), max(bb_proj[1])

        # Limit data
        x_upper = x >= x_min
        x_lower = x <= x_max
        y_upper = y >= y_min
        y_lower = y <= y_max
        if sum(x_upper == x_lower) < 2:
            if sum(x_lower) == 0 and sum(x_upper) == len(x_upper):
                raise CFRegionModelRepositoryError("Bounding box longitudes don't intersect with dataset.")
            x_upper[np.argmax(x_upper) - 1] = True
            x_lower[np.argmin(x_lower)] = True
        if sum(y_upper == y_lower) < 2:
            if sum(y_lower) == 0 and sum(y_upper) == len(y_upper):
                raise CFRegionModelRepositoryError("Bounding box latitudes don't intersect with dataset.")
            y_upper[np.argmax(y_upper) - 1] = True
            y_lower[np.argmin(y_lower)] = True

        x_inds = np.nonzero(x_upper == x_lower)[0]
        y_inds = np.nonzero(y_upper == y_lower)[0]

        # Masks
        x_mask = x_upper == x_lower
        y_mask = y_upper == y_lower
        xy_mask = ((x_mask)&(y_mask))
        

        # Transform from source coordinates to target coordinates
        #xx, yy = transform(data_proj, target_proj, *np.meshgrid(x[x_mask], y[y_mask]))
        xx, yy = transform(data_proj, target_proj, x[xy_mask], y[xy_mask])

        #return xx, yy, (x_mask, y_mask), (x_inds, y_inds)
        return xx, yy, xy_mask, (x_inds, y_inds)

    def get_region_model(self, region_id, catchments=None):
        """
        Return a fully specified shyft api region_model for region_id, based on data found
        in netcdf dataset.

        Parameters
        -----------
        region_id: string
            unique identifier of region in data

        catchments: list of unique integers
            catchment indices when extracting a region consisting of a subset
            of the catchments has attribs to construct params and cells etc.

        Returns
        -------
        region_model: shyft.api type
        """

        with Dataset(self._data_file) as dset:
            Vars = dset.variables
            c_ids = Vars["catchment_id"]
            xcoord = Vars['x'][:]
            ycoord = Vars['y'][:]
            m_catch = np.ones(len(c_ids), dtype=bool)
            if self._catch_ids is not None:
                m_catch = np.in1d(c_ids, self._catch_ids)
                xcoord_m = xcoord[m_catch]
                ycoord_m = ycoord[m_catch]
            
            dataset_epsg = None
            if 'crs' in Vars.keys():
                dataset_epsg = Vars['crs'].epsg_code.split(':')[1]
            if not dataset_epsg:
                raise interfaces.InterfaceError("netcdf: epsg attr not found in group elevation")
            #if dataset_epsg != self._epsg:
            target_cs = "+proj=utm +zone={} +ellps={} +datum={} +units=m +no_defs".format(
            int(self._epsg) - 32600, "WGS84", "WGS84")
            source_cs = "+proj=utm +zone={} +ellps={} +datum={} +units=m +no_defs".format(
            int(dataset_epsg) - 32600, "WGS84", "WGS84")

            # Construct bounding region
            box_fields = set(("lower_left_x", "lower_left_y", "step_x", "step_y", "nx", "ny", "EPSG"))
            if box_fields.issubset(self._rconf.domain()):
                tmp = self._rconf.domain()
                epsg = tmp["EPSG"]
                x_min = tmp["lower_left_x"]
                x_max = x_min + tmp["nx"]*tmp["step_x"]
                y_min = tmp["lower_left_y"]
                y_max = y_min + tmp["ny"]*tmp["step_y"]
                bounding_region = BoundingBoxRegion(np.array([x_min, x_max]),
                                                    np.array([y_min, y_max]), epsg, self._epsg)
            else:
                bounding_region = BoundingBoxRegion(xcoord_m, ycoord_m, dataset_epsg, self._epsg)
            self.bounding_box = bounding_region.bounding_box(self._epsg)
            x, y, m_xy, _ = self._limit(xcoord, ycoord,source_cs, target_cs)
            
            mask = ((m_xy)&(m_catch))

            areas = Vars['area'][mask]
            elevation = Vars["z"][mask]
            coordinates = np.dstack((x[mask], y[mask], elevation)).reshape(-1,3)

            c_ids = Vars["catchment_id"][mask]
            c_ids_unique = list(np.unique(c_ids))
            c_indx = np.array([c_ids_unique.index(cid) for cid in c_ids])
            

            ff = Vars["forest-fraction"][mask]
            lf = Vars["lake-fraction"][mask]
            rf = Vars["reservoir-fraction"][mask]
            gf = Vars["glacier-fraction"][mask]


        # Construct region parameter:
        name_map = {"gamma_snow": "gs", "priestley_taylor": "pt",
                    "kirchner": "kirchner", "actual_evapotranspiration": "ae",
                    "skaugen": "skaugen", "hbv_snow": "snow"}
        region_parameter = self._region_model.parameter_t()
        for p_type_name, value_ in iteritems(self._mconf.model_parameters()):
            if p_type_name in name_map:
                if hasattr(region_parameter, name_map[p_type_name]):
                    sub_param = getattr(region_parameter, name_map[p_type_name])
                    for p, v in iteritems(value_):
                        setattr(sub_param, p, v)
            elif p_type_name == "p_corr_scale_factor":
                region_parameter.p_corr.scale_factor = value_

        # TODO: Move into yaml file similar to p_corr_scale_factor
        radiation_slope_factor = 0.9

        # Construct cells
        cell_vector = self._region_model.cell_t.vector_t()
        for pt, a, c_id, ff, lf, rf, gf in zip(coordinates, areas, c_indx, ff, lf, rf, gf):
            cell = self._region_model.cell_t()
            cell.geo = api.GeoCellData(api.GeoPoint(*pt), a, c_id, radiation_slope_factor,
                                       api.LandTypeFractions(gf, lf, rf, ff, 0.0))
            cell_vector.append(cell)

        # Construct catchment overrides
        catchment_parameters = self._region_model.parameter_t.map_t()
        for k, v in iteritems(self._rconf.parameter_overrides()):
            if k in c_ids_unique:
                param = self._region_model.parameter_t(region_parameter)
                for p_type_name, value_ in iteritems(v):
                    if p_type_name in name_map:
                        sub_param = getattr(param, name_map[p_type_name])
                        for p, pv in iteritems(value_):
                            setattr(sub_param, p, pv)
                    elif p_type_name == "p_corr_scale_factor":
                        param.p_corr.scale_factor = value_
                    else:
                        # Avoid unknown params to go unadvertised
                        raise RegionConfigError(
                            "parameter {} is not in the set of allowed ones".format(p_type_name))

                catchment_parameters[c_ids_unique.index(k)] = param
        region_model = self._region_model(cell_vector, region_parameter, catchment_parameters)
        region_model.bounding_region = bounding_region
        region_model.catchment_id_map = c_ids_unique
        return region_model


class BoundingBoxRegion(interfaces.BoundingRegion):

    def __init__(self, x, y, point_epsg, target_epsg):
        self._epsg = str(point_epsg)
        x_min = x.ravel().min()
        x_max = x.ravel().max()
        y_min = y.ravel().min()
        y_max = y.ravel().max()
        self.x = np.array([x_min, x_max, x_max, x_min], dtype="d")
        #self.y = np.array([y_max, y_max, y_min, y_min], dtype="d")
        self.y = np.array([y_min, y_min, y_max, y_max], dtype="d")
        self.x, self.y = self.bounding_box(target_epsg)
        self._epsg = str(target_epsg)

    def bounding_box(self, epsg):
        epsg = str(epsg)
        if epsg == self.epsg():
            return np.array(self.x), np.array(self.y)
        else:
            source_cs = "+proj=utm +zone={} +ellps={} +datum={} +units=m +no_defs".format(
                int(self.epsg()) - 32600, "WGS84", "WGS84")
            target_cs = "+proj=utm +zone={} +ellps={} +datum={} +units=m +no_defs".format(
                int(epsg) - 32600, "WGS84", "WGS84")
            source_proj = Proj(source_cs)
            target_proj = Proj(target_cs)
            return [np.array(a) for a in transform(source_proj, target_proj, self.x, self.y)]

    def bounding_polygon(self, epsg):
        return self.bounding_box(epsg)

    def epsg(self):
        return self._epsg