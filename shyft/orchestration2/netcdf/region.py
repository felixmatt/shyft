"""
Module for reading region netCDF files needed for an SHyFT run.
"""

from __future__ import absolute_import

import numpy as np
from netCDF4 import Dataset

from ..base_config import BaseRegion


# Concrete versions of config file classes for NetCDF data access
class Region(BaseRegion):

    def __init__(self, config_file, data_file):
        super(Region, self).__init__(config_file)
        self._mask = None
        self._data_file = data_file

    @property
    def mask(self):
        """Get the mask for cells that have actual info."""
        if self._mask is not None:
            return self._mask
        with Dataset(self._data_file) as dset:
            mask = dset.groups['catchments'].variables["catchments"][:].reshape(-1) != 0
        self._mask = mask
        return mask

    def __repr__(self):
        return "%s(data_file=%r)" % (
            self.__class__.__name__, self._data_file)

    def fetch_cell_properties(self, varname):
        with Dataset(self._data_file) as dset:
            values = dset.groups[varname].variables[varname][:].reshape(-1)
        return values[self.mask]

    def fetch_cell_centers(self):
        with Dataset(self._data_file) as dset:
            grp = dset.groups['elevation']
            xcoord = grp.variables['xcoord'][:]
            ycoord = grp.variables['ycoord'][:]
            mesh2d = np.dstack(np.meshgrid(xcoord, ycoord)).reshape(-1, 2)
            elevation = grp.variables['elevation'][:]
            mesh3d = np.hstack((mesh2d, elevation.reshape(-1, 1)))
        return mesh3d[self.mask]

    def fetch_cell_areas(self):
        # WARNING: the next is only valid for regular grids
        with Dataset(self._data_file) as dset:
            grp = dset.groups['elevation']
            xcoord = grp.variables['xcoord'][:]
            ycoord = grp.variables['ycoord'][:]
            area = (xcoord[1] - xcoord[0]) * (ycoord[1] - ycoord[0])
            areas = np.ones(len(xcoord) * len(ycoord), dtype=xcoord.dtype) * area
        return areas[self.mask]

    def fetch_catchments(self, what):
        with Dataset(self._data_file) as dset:
            grp = dset.groups['catchments']
            if what == "values":
                return grp.variables["catchments"][:].reshape(-1)[self.mask]
            elif what == "names":
                return grp.variables["catchment_names"][:]
            elif what == "indices":
                return grp.variables["catchment_indices"][:]
            else:
                raise ValueError("Attribute '%s' not supported for catchments" % what)
