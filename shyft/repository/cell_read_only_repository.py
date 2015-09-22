from copy import deepcopy
import os
import gdal
from gdalconst import * # TODO: Fixme
import struct
import numpy as np
from gis_data import CellDataFetcher

from abc import ABCMeta
from .base_repository import BaseCellRepository

class CellReadOnlyRepository(BaseCellRepository):

    def __init__(self, **kwargs):
        assert("catchment_id" in kwargs)
        assert("geo_position" in kwargs)
        assert("land_type" in kwargs)
        assert("glacier_fraction" in kwargs)
        assert("reservoir_fraction" in kwargs)
        assert("lake_fraction" in kwargs)
        assert("forest_fraction" in kwargs)
        assert("area" in kwargs)
        # Shapes are optional, so pop off and insert later
        shapes = kwargs.pop("shapes", None)
        lengths = [len(v) for v in kwargs.iteritems()]
        if not min(lengths) == max(lengths):
            raise RuntimeError("Unequal length of cell data lists")
        self.data = deepcopy(kwargs) # TODO: Determine if this is really needed, or if we can live with a reference.

        if shapes:
            self.data["shapes"] = deepcopy(shapes)

    def get(self, key):
        if key in self.data:
            return self.data[key]
        raise RuntimeError("No entry with key {} found.".format(key))

    def number_of_cells(self):
        return len(self.data["catchment_id"])

    def find(self, *args, **kwargs):
        return self.data.keys()


def cell_repository_factory(config, *args):
    cdf = CellDataFetcher(args[0], args[1], config.x_min, config.y_min, config.dx, config.dy, config.n_x, config.n_y,
                          config.repository_data["catchment_indices"], epsg_id=config.epsg_id)
    region_cell_data = cdf.fetch()
    #from IPython.core.debugger import Tracer; Tracer()()
    catchment_ids = []
    geo_positions = []
    land_types = []
    lake_fractions = []
    reservoir_fractions = []
    glacier_fractions = []
    forest_fractions = []
    areas = []
    shapes = []
    for catchment_id in region_cell_data["cell_data"]:
        for cell in region_cell_data["cell_data"][catchment_id]:
            catchment_ids.append(catchment_id)
            geo_positions.append([cell['cell'].centroid.x, cell['cell'].centroid.y, cell["elevation"]])
            land_types.append(1 if abs(1 - (cell.get("lake", 0.0) + cell.get("reservoir", 0.0))) < 1.0e-7 else 0)
            lake_fractions.append(cell.get("lake", 0.0))
            reservoir_fractions.append(cell.get("reservoir", 0.0))
            glacier_fractions.append(cell.get("glacier", 0.0))
            forest_fractions.append(cell.get("forest", 0.0))
            areas.append(cell['cell'].area)
            shapes.append(cell['cell']) 

    cell_data = {"catchment_id": catchment_ids,
                 "geo_position": geo_positions,
                 "land_type": land_types,
                 "glacier_fraction": glacier_fractions,
                 "lake_fraction": lake_fractions,
                 "reservoir_fraction": reservoir_fractions,
                 "forest_fraction": forest_fractions, 
                 "area": areas,
                 "shapes": shapes}
    return CellReadOnlyRepository(**cell_data)


#class FileCellRepository(BaseCellReadOnlyRepository): # TODO: Implement ABC 
class FileCellRepository(BaseCellRepository):

    def __init__(self, config, *file_store):
        self.file_store = file_store
        self.config = config
        self.data = {}

    def construct(self):
        if self.data is None:
            return
        self._load()
        rdf, dims = raster_cell_data.fetch()

        catchment_ids = []
        geo_positions = []
        land_types = []
        lake_fractions = []
        reservoir_fractions = []
        glacier_fractions = []
        forest_fractions = []
        areas = []
        shapes = []

        for catchment_id in rdf:
            for cell in rdf[catchment_id]:
                catchment_ids.append(catchment_id)
                geo_positions.append(list(cell["cell_center"]) + [cell["elevation"]])
                land_types.append(1 if abs(1.0 - cell["landuse"]) < 1.0e-7 else 0)
                lake_fractions.append(land_types[-1])
                reservoir_fractions.append(0)
                glacier_fractions.append(cell["glacier"])
                forest_fractions.append(cell["forest"])
                areas.append(dims["dx"]*dims["dy"])
                shapes.append(None)

        # The main structure that holds all information
        self.data = {"catchment_id": catchment_ids,
                     "geo_position": geo_positions,
                     "land_type": land_types,
                     "glacier_fraction": glacier_fractions,
                     "lake_fraction": lake_fractions,
                     "reservoir_fraction": reservoir_fractions,
                     "forest_fraction": forest_fractions, 
                     "area": areas,
                     "shapes": shapes}

        # Simple validation of input data (lengths and types)
        lengths = [len(v) for v in self.data.iteritems()]
        if not min(lengths) == max(lengths):
            raise RuntimeError("Unequal length of cell data lists")


    @staticmethod
    def _file_raster_extract(filename):
        dataset = gdal.Open(filename, GA_ReadOnly)
        band = dataset.GetRasterBand(1)
        nx = band.XSize
        ny = band.YSize
        scanline = band.ReadRaster(0, 0, nx, ny, nx, ny, gdal.GDT_Float32)
        arr = np.array(struct.unpack('f'*nx*ny, scanline))
        arr.shape = ny, nx
        return arr, dataset.GetGeoTransform()


    @staticmethod
    def _idx_to_loc(idx, raster_shape, geo_trans):

        L = idx/raster_shape[1]
        P = idx%raster_shape[1]
        xp = geo_trans[0] + P*geo_trans[1] + L*geo_trans[2]
        yp = geo_trans[3] + P*geo_trans[4] + L*geo_trans[5]
        return xp, yp


    def _load(self):
        extracted = {}
        geo_transform = {}
        for (tp, fn) in self.file_store.iteritems():
            if os.path.isfile(fn):
                extracted[tp], geo_transform[tp] = self._file_raster_extract(fn)
            else:
                raise RuntimeError("File not found")
        raster_shape = extracted["catchments"].shape

        cell_indices = np.argwhere(extracted["catchments"].flatten() > 0)
        cell_indices.shape = cell_indices.shape[0],
        for k,v in extracted.items():
            extracted[k] = v.flatten()[cell_indices]
            extracted[k].shape = extracted[k].shape[0],

        geo_positions = []
        for idx in cell_indices:
            geo_positions.append(self._idx_to_loc(idx, raster_shape, geo_transform["catchments"]))

        catchments = extracted.pop("catchments")
        elevations = extracted.pop("elevation")

        cell_data = {}
        for i in xrange(len(catchments)):
            catchment_id = int(round(catchments[i]))
            geo_pos = geo_positions[i]
            if catchment_id not in cell_data:
                cell_data[catchment_id] = []
            data = {"elevation": elevations[i], "cell_center": geo_pos}
            for land_type_name, land_type in extracted.iteritems():
                data[land_type_name] = land_type[i]
            cell_data[catchment_id].append(data)
        return cell_data, {"dx": geo_transform["catchments"][1], "dy": abs(geo_transform["catchments"][-1])}

    def get(self, key):
        self.construct()
        if key in self.data:
            return self.data[key]
        raise RuntimeError("No entry with key {} found.".format(key))

    def number_of_cells(self):
        self.construct()
        return len(self.data["catchment_id"])

    def find(self, *args, **kwargs):
        self.construct()
        return self.data.keys()

