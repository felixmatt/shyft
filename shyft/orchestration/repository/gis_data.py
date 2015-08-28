# -*- coding: utf-8 -*-

import requests
import copy
import matplotlib.pylab as plt
from descartes import PolygonPatch
from matplotlib.collections import PatchCollection
from matplotlib import cm
from itertools import imap
import re
import time
import shutil
import numpy as np
import tempfile

from shapely.geometry import Polygon, MultiPolygon, box, LineString, Point
from shapely.ops import cascaded_union
import gdal
import os

class GisDataFetchError(Exception): pass

class BaseGisDataFetcher(object):

    def __init__(self, geometry=None, server_name=None, server_port=None, service_index=None, epsg_id=32632):
        self.server_name = server_name
        self.server_port = server_port
        self.service_index = service_index
        self.geometry = geometry
        self.epsg_id = epsg_id
        if server_name.endswith('p'):
            self.url_template="http://{}:{}/arcgis/rest/services/Enki/gis_info/MapServer/{}/query"
        else:
            self.url_template = "http://{}:{}/arcgis/rest/services/EnkiLandTypes/EnkiLandTypes/MapServer/{}/query"
        if os.environ.get("NO_PROXY", False) and not self.server_name in os.environ["NO_PROXY"]: os.environ["NO_PROXY"] += ", {}".format(self.server_name)

        self.query = dict(text="",
                          objectIds="",
                          time="",
                          geometry="",
                          geometryType="esriGeometryEnvelope",
                          inSR=self.epsg_id,
                          spatialRel="esriSpatialRelIntersects",
                          relationParam="",
                          outFields="",
                          returnGeometry=True,
                          maxAllowableOffset="",
                          geometryPrecision="",
                          outSR=self.epsg_id,
                          returnIdsOnly=False,
                          returnCountOnly=False,
                          orderByFields="",
                          groupByFieldsForStatistics="",
                          outStatistics="",
                          returnZ=False,
                          returnM=False,
                          gdbVersion="",
                          returnDistinctValues=False,
                          f="pjson")

    @property
    def url(self):
        url = self.url_template.format(self.server_name, self.server_port, self.service_index)
        return url

    def get_query(self, geometry=None):
        q = copy.deepcopy(self.query)
        if geometry is None:
            geometry = self.geometry
        if not geometry:
            q["geometry"] = ""
        else:
            q["geometry"] = ",".join([str(c) for c in geometry])
        return q


class LandTypeFetcher(BaseGisDataFetcher):

    def __init__(self, geometry=None, epsg_id=32632):
        super(LandTypeFetcher, self).__init__(geometry=geometry, server_name="oslwvagi001p", server_port="6080", service_index=0, epsg_id=epsg_id)
        self.name_to_layer_map = {"glacier":0,"forest":1,"lake":2}
        self.query["outFields"]="OBJECTID"
        #self.query["where"]="OBJTYPE = '{}'"
    
    @property
    def en_field_names(self):
        return self.name_to_layer_map.keys()
        
    def build_query(self, name, **kwargs):
        q = self.get_query(kwargs.pop("geometry", None))
        #q["where"] = q["where"].format(self.en_no_field_map[name]).decode("latin1")
        self.service_index=self.name_to_layer_map[name]
        q.update(kwargs)
        return q

    def fetch(self, **kwargs):
        name = kwargs.pop("name", None)
        if not name:
            raise RuntimeError("Mandatory argument 'name' not given")

        q = self.build_query(name, **kwargs)
        response = requests.get(self.url, params=q)
        if response.status_code != 200:
            raise GisDataFetchError("Could not fetch land type data from gis server.")
        data = response.json()
        polygons = []
        print "Extracting {} {} features".format(len(data["features"]), name)
        error_count=0
        for feature in data["features"]:
            shell = feature["geometry"]["rings"][0]
            holes = feature["geometry"]["rings"][1:]
            holes = holes if holes else None
            polygon=Polygon(shell=shell, holes=holes)
            if polygon.is_valid:
                polygons.append(polygon)
            else:
                error_count +=1
        if error_count>0:
            print "gis polygon error count is",error_count
        return cascaded_union(polygons)

class ReservoirFetcher(BaseGisDataFetcher):

    def __init__(self, geometry=None, epsg_id=32632):
        super(ReservoirFetcher, self).__init__(geometry=geometry, server_name="oslwvagi001p", server_port="6080", service_index=5, epsg_id=epsg_id)
        self.query["where"]="1 = 1"
        self.query["outFields"]="OBJECTID"

    def fetch(self, **kwargs):
        q = self.get_query(kwargs.pop("geometry", None))
        response = requests.get(self.url, params=q)
        if response.status_code != 200:
            raise GisDataFetchError("Could not fetch reservoir data from gis server.")
        data = response.json()
        if not "features" in data:
            raise GisDataFetchError("GeoJson data missing mandatory field, please check your gis service or your query.") 
        points = []
        for feature in data["features"]:
            x = feature["geometry"]["x"]
            y = feature["geometry"]["y"]
            points.append(Point(x, y))
        return points


class CatchmentFetcher(BaseGisDataFetcher):

    def __init__(self, geometry=None, indices=None, extract_all=False, epsg_id=32632):
        super(CatchmentFetcher, self).__init__(geometry=geometry, server_name="oslwvagi001p", server_port="6080", service_index=3, epsg_id=epsg_id)
        self.indices = indices
        if self.indices:
            self.query["where"] = "VANNKVNR IN ({})".format(", ".join([str(i) for i in indices]))
        else:
            self.query["where"] = "1 = 1"
        self.query["outFields"] = "VANNKVNR, DELFELTNR, VANNKVNAVN"

    def build_query(self, **kwargs):
        q = self.get_query(kwargs.pop("geometry", None))
        indices = kwargs.pop("indices", None)
        if indices:
            q["where"] = "VANNKVNR IN ({})".format(", ".join([str(i) for i in indices]))
        return q

    def fetch(self, **kwargs):
        q = self.build_query(**kwargs)
        response = requests.get(self.url, params=q)
        if response.status_code != 200:
            raise GisDataFetchError("Could not fetch catchment index data from gis server.")
        data = response.json()
        #from IPython.core.debugger import Tracer; Tracer()()
        polygons = {}
        for feature in data['features']:
            c_id = feature['attributes']['VANNKVNR']
            shell = feature["geometry"]["rings"][0]
            holes = feature["geometry"]["rings"][1:]
            polygon = Polygon(shell=shell, holes=holes)
            if polygon.is_valid:
                if not c_id in polygons:
                    polygons[c_id] = []
                polygons[c_id].append(polygon) 
        return {key: cascaded_union(polygons[key]) for key in polygons if polygons[key]}

class CatchmentFetcher2(BaseGisDataFetcher):

    def __init__(self, catchment_type, identifier, geometry=None, indices=None, extract_all=False, epsg_id=32632):
        if (catchment_type=='regulated'):
            service_index = 6
            #self.identifier = 'POWER_PLANT_ID'
        elif (catchment_type=='unregulated'):
            service_index = 7
            #self.identifier = 'FELTNR'
        else:
            raise GisDataFetchError("Undefined catchment type {} - use either regulated or unregulated".format(catchment_type))
        super(CatchmentFetcher2, self).__init__(geometry=geometry, server_name="oslwvagi001p", server_port="6080", service_index=service_index, epsg_id=epsg_id)
        self.identifier = identifier       
        self.indices = indices
        if self.indices:
            self.query["where"] = "{} IN ({})".format(self.identifier,", ".join([str(i) for i in indices]))
        else:
            self.query["where"] = "1 = 1"
        #self.query["outFields"] = "{}, CATCH_ID, name_statkraft".format(self.identifier)
        self.query["outFields"] = "{}".format(self.identifier) #additional attributes to be extracted can be specified here

    def build_query(self, **kwargs):
        q = self.get_query(kwargs.pop("geometry", None))
        indices = kwargs.pop("indices", None)
        if indices:
            q["where"] = "{} IN ({})".format(self.identifier,", ".join([str(i) for i in indices]))
        return q

    def fetch(self, **kwargs):
        q = self.build_query(**kwargs)
        response = requests.get(self.url, params=q)
        if response.status_code != 200:
            raise GisDataFetchError("Could not fetch catchment index data from gis server.")
        data = response.json()
        #from IPython.core.debugger import Tracer; Tracer()()
        polygons = {}
        for feature in data['features']:
            c_id = feature['attributes'][self.identifier]
            shell = feature["geometry"]["rings"][0]
            holes = feature["geometry"]["rings"][1:]
            polygon = Polygon(shell=shell, holes=holes)
            if polygon.is_valid:
                if not c_id in polygons:
                    polygons[c_id] = []
                polygons[c_id].append(polygon) 
        return {key: cascaded_union(polygons[key]) for key in polygons if polygons[key]}

class CellDataFetcher(object):

    def __init__(self, x0, y0, dx, dy, nx, ny, indices, epsg_id=32632):
        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy
        self.nx = nx
        self.ny = ny
        self.indices = indices
        server_name="oslwvagi001q"
        server_port="6080"  
        self.cell_data = {}
        self.catchment_land_types = {}
        self.epsg_id = epsg_id

    def fetch(self):
        from shapely.prepared import prep
        catchment_fetcher = CatchmentFetcher(geometry=self.geometry, indices=self.indices, epsg_id=self.epsg_id)
        catchments = catchment_fetcher.fetch()

        # Construct cells and populate with elevations from tdm
        dtm_fetcher = DTMFetcher(self.x0, self.y0, self.dx, self.dy, self.nx, self.ny, epsg_id=self.epsg_id)
        elevations = dtm_fetcher.fetch()
        cells = []
        for i in xrange(self.nx):
            for j in xrange(self.ny):
                cells.append((box(self.x0 + i*self.dx, self.y0 + j*self.dy, self.x0 + (i + 1)*self.dx,  self.y0 + (j + 1)*self.dy), float(elevations[self.ny-j-1,i]))) 
        catchment_land_types = {}
        catchment_cells = {}

        # Filter all data with each catchment
        ltf = LandTypeFetcher()
        rf = ReservoirFetcher()
        all_reservoir_coords=rf.fetch(geometry=self.geometry);
        all_glaciers=ltf.fetch(name="glacier",geometry=self.geometry)
        prep_glaciers=prep(all_glaciers)
        all_lakes   =ltf.fetch(name="lake",geometry=self.geometry)
        prep_lakes= prep(all_lakes)
        all_forest  =ltf.fetch(name="forest",geometry=self.geometry)
        prep_forest=prep(all_forest)
        print "Doing catchment loop, n reservoirs", len(all_reservoir_coords)
        for catchment_id, catchment in catchments.iteritems():
            if not catchment_id in catchment_land_types: # SiH: default landtype, plus the special ones fetched below
                catchment_land_types[catchment_id] = {}
            if prep_lakes.intersects(catchment):
                lake_in_catchment = all_lakes.intersection(catchment)
                if isinstance(lake_in_catchment, (Polygon, MultiPolygon)):
                    reservoir_list=[]
                    for rsv_point in all_reservoir_coords:
                        for lake in lake_in_catchment:
                            if lake.contains(rsv_point):
                                reservoir_list.append(lake)
                    if reservoir_list:
                        reservoir= MultiPolygon(reservoir_list)
                        catchment_land_types[catchment_id]["reservoir"] = reservoir
                        catchment_land_types[catchment_id]["lake"] = lake_in_catchment.difference(reservoir)
                    else:
                        catchment_land_types[catchment_id]["lake"]=lake_in_catchment
            if prep_glaciers.intersects(catchment):
                glacier_in_catchment=all_glaciers.intersection(catchment)
                if isinstance(glacier_in_catchment, (Polygon, MultiPolygon)):
                    catchment_land_types[catchment_id]["glacier"]=glacier_in_catchment
            if prep_forest.intersects(catchment):
                forest_in_catchment= all_forest.intersection(catchment)
                if isinstance(forest_in_catchment, (Polygon, MultiPolygon)):
                    catchment_land_types[catchment_id]["forest"]=forest_in_catchment


            catchment_cells[catchment_id] = []
            for cell, elevation in cells: 
                if cell.intersects(catchment):
                    catchment_cells[catchment_id].append((cell.intersection(catchment), elevation))

        # Gather cells on a per catchment basis, and compute the area fraction for each landtype
        print "Done with catchment cell loop, calc fractions"
        cell_data = {}
        for catchment_id in catchments.iterkeys():
            cell_data[catchment_id] = []
            for cell, elevation in catchment_cells[catchment_id]:
                data = {"cell": cell, "elevation": elevation}
                for land_type_name, land_type_shape in catchment_land_types[catchment_id].iteritems():
                    data[land_type_name] = cell.intersection(land_type_shape).area/cell.area
                cell_data[catchment_id].append(data)
        self.cell_data = cell_data
        self.catchment_land_types = catchment_land_types
        self.elevation_raster = elevations
        print "Done constructing cell-data and landtypes"
        return {"cell_data": self.cell_data, "catchment_land_types": self.catchment_land_types, "elevation_raster": self.elevation_raster}

    @property
    def geometry(self):
        return [self.x0, self.y0, self.x0 + self.dx*self.nx , self.y0 + self.dy*self.ny]

class CellDataFetcher2(object):

    def __init__(self, catchment_type, identifier, x0, y0, dx, dy, nx, ny, indices, epsg_id=32632):
        self.catchment_type=catchment_type
        self.identifier=identifier
        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy
        self.nx = nx
        self.ny = ny
        self.indices = indices
        self.cell_data = {}
        self.catchment_land_types = {}
        self.epsg_id = epsg_id

    def fetch(self):
        from shapely.prepared import prep
        catchment_fetcher = CatchmentFetcher2(self.catchment_type, self.identifier, geometry=self.geometry, indices=self.indices, epsg_id=self.epsg_id)
        catchments = catchment_fetcher.fetch()

        # Construct cells and populate with elevations from tdm
        dtm_fetcher = DTMFetcher(self.x0, self.y0, self.dx, self.dy, self.nx, self.ny, epsg_id=self.epsg_id)
        elevations = dtm_fetcher.fetch()
        cells = []
        for i in xrange(self.nx):
            for j in xrange(self.ny):
                cells.append((box(self.x0 + i*self.dx, self.y0 + j*self.dy, self.x0 + (i + 1)*self.dx,  self.y0 + (j + 1)*self.dy), float(elevations[self.ny-j-1,i]))) 
        catchment_land_types = {}
        catchment_cells = {}

        # Filter all data with each catchment
        ltf = LandTypeFetcher()
        rf = ReservoirFetcher()
        all_reservoir_coords=rf.fetch(geometry=self.geometry);
        all_glaciers=ltf.fetch(name="glacier",geometry=self.geometry)
        prep_glaciers=prep(all_glaciers)
        all_lakes   =ltf.fetch(name="lake",geometry=self.geometry)
        prep_lakes= prep(all_lakes)
        all_forest  =ltf.fetch(name="forest",geometry=self.geometry)
        prep_forest=prep(all_forest)
        print "Doing catchment loop, n reservoirs", len(all_reservoir_coords)
        for catchment_id, catchment in catchments.iteritems():
            if not catchment_id in catchment_land_types: # SiH: default landtype, plus the special ones fetched below
                catchment_land_types[catchment_id] = {}
            if prep_lakes.intersects(catchment):
                lake_in_catchment = all_lakes.intersection(catchment)
                if  isinstance(lake_in_catchment, (Polygon, MultiPolygon)) and lake_in_catchment.area >1000.0:
                    reservoir_list=[]
                    for rsv_point in all_reservoir_coords:
                        if isinstance(lake_in_catchment, (Polygon)):
                            if lake_in_catchment.contains(rsv_point):
                                reservoir_list.append(lake_in_catchment)
                        else:
                            for lake in lake_in_catchment:
                                if lake.contains(rsv_point):
                                    reservoir_list.append(lake)
                    if reservoir_list:
                        reservoir= MultiPolygon(reservoir_list)
                        catchment_land_types[catchment_id]["reservoir"] = reservoir
                        diff=lake_in_catchment.difference(reservoir)
                        if diff.area >1000.0:
                            catchment_land_types[catchment_id]["lake"] =diff 
                    else:
                        catchment_land_types[catchment_id]["lake"]=lake_in_catchment
            if prep_glaciers.intersects(catchment):
                glacier_in_catchment=all_glaciers.intersection(catchment)
                if isinstance(glacier_in_catchment, (Polygon, MultiPolygon)):
                    catchment_land_types[catchment_id]["glacier"]=glacier_in_catchment
            #if prep_forest.intersects(catchment):
            #    forest_in_catchment= all_forest.intersection(catchment)
            #    if isinstance(forest_in_catchment, (Polygon, MultiPolygon)):
            #        catchment_land_types[catchment_id]["forest"]=forest_in_catchment


            catchment_cells[catchment_id] = []
            for cell, elevation in cells: 
                if cell.intersects(catchment):
                    catchment_cells[catchment_id].append((cell.intersection(catchment), elevation))

        # Gather cells on a per catchment basis, and compute the area fraction for each landtype
        print "Done with catchment cell loop, calc fractions"
        cell_data = {}
        for catchment_id in catchments.iterkeys():
            cell_data[catchment_id] = []
            for cell, elevation in catchment_cells[catchment_id]:
                data = {"cell": cell, "elevation": elevation}
                for land_type_name, land_type_shape in catchment_land_types[catchment_id].iteritems():
                    data[land_type_name] = cell.intersection(land_type_shape).area/cell.area
                cell_data[catchment_id].append(data)
        self.cell_data = cell_data
        self.catchment_land_types = catchment_land_types
        self.elevation_raster = elevations
        return {"cell_data": self.cell_data, "catchment_land_types": self.catchment_land_types, "elevation_raster": self.elevation_raster}

    @property
    def geometry(self):
        return [self.x0, self.y0, self.x0 + self.dx*self.nx , self.y0 + self.dy*self.ny]

class DTMFetcher(object):

    def __init__(self, x0, y0, dx, dy, nx, ny, epsg_id=32632):
        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy
        self.nx = nx
        self.ny = ny
        self.epsg_id = epsg_id
        #self.server_name = "oslwvagi001q" #PREPROD
        self.server_name = "oslwvagi001p" #PROD
        self.server_port = "6080"
        #self.url_template = "http://{}:{}/arcgis/rest/services/RuneTest/DEM1000/ImageServer/exportImage" #PREPROD
        self.url_template = "http://{}:{}/arcgis/rest/services/Enki/Norway_DTM_1000m/ImageServer/exportImage" #PROD
        self.query = dict(bbox="", 
                          bboxSR=self.epsg_id,
                          size= "{},{}".format(self.nx, self.ny),
                          imageSR=self.epsg_id,
                          time="",
                          format="tiff", 
                          pixelType="F32",
                          noData="",
                          noDataInterpretation="esriNoDataMatchAny",
                          interpolation="RSP_BilinearInterpolation",
                          compressionQuality="",
                          bandIds="",
                          mosaicRule="",
                          renderingRule="",
                          f="image")
        if os.environ.get("NO_PROXY", False) and not self.server_name in os.environ["NO_PROXY"]: os.environ["NO_PROXY"] += ", {}".format(self.server_name)

    @property
    def url(self):
        return self.url_template.format(self.server_name, self.server_port)

    def get_query(self, **kwargs):
        q = copy.deepcopy(self.query)
        q["bbox"] = "{},{},{},{}".format(self.x0, self.y0, self.x0 + self.dx*self.nx, self.y0 + self.dy*self.ny)
        return q

    def fetch(self):
        q = self.get_query()
        response = requests.get(self.url, params=q, stream=True)
        if response.status_code != 200:
            raise GisDataFetchError("Could not fetch DTM data from gis server.")
        img = response.raw.read()
        filename = tempfile.mktemp()
        with open(filename, "wb") as tf:
            tf.write(img)
        dataset = gdal.Open(filename)
        try:
            band = dataset.GetRasterBand(1)
            data = band.ReadAsArray()
        finally:
            del dataset
        os.remove(filename)
        return np.array(data, dtype=np.float64)


def add_plot_polygons(ax, polygons, color):
    ps = []
    if not isinstance(polygons, list):
        polygons = [polygons]
    for polygon in polygons:
        if isinstance(polygon, MultiPolygon):
            ps.extend([p for p in polygon])
        else:
            ps.append(polygon)
    patches = imap(PolygonPatch, [p for p in ps])
    ax.add_collection(PatchCollection(patches, facecolors=(color,), linewidths=0.1, alpha=0.3))

def run_cell_example(x0, y0, dx, dy, nx, ny, indices):
    cf = CellDataFetcher(x0, y0, dx, dy, nx, ny, indices=indices)
    print "Start fetching data"
    cf.fetch()
    print "Done, now preparing plot"
    # Plot the extracted data
    fig, ax = plt.subplots(1)
    color_map = {"forest": 'g', "lake": 'b', "glacier": 'r', "cell": "0.75", "reservoir": "purple"}

    extent = cf.geometry[0], cf.geometry[2], cf.geometry[1], cf.geometry[3]
    ax.imshow(cf.elevation_raster, origin='upper', extent=extent, cmap=cm.gray)

    for catchment_cells in cf.cell_data.itervalues():
        add_plot_polygons(ax, [cell["cell"] for cell in catchment_cells], color=color_map["cell"])
    for catchment_land_type in cf.catchment_land_types.itervalues():
        for k,v in catchment_land_type.iteritems():
            add_plot_polygons(ax, v, color=color_map[k])

    geometry = cf.geometry
    ax.set_xlim(geometry[0], geometry[2])
    ax.set_ylim(geometry[1], geometry[3])
    plt.show()

def run_cell_example2(catchment_type,identifier,x0, y0, dx, dy, nx, ny, catch_indicies):
    cf = CellDataFetcher2(catchment_type,identifier,x0, y0, dx, dy, nx, ny, indices=catch_indicies)
    print "Start fetching data"
    cf.fetch()
    print "Done, now preparing plot"
    # Plot the extracted data
    fig, ax = plt.subplots(1)
    color_map = {"forest": 'g', "lake": 'b', "glacier": 'r', "cell": "0.75", "reservoir": "purple"}

    extent = cf.geometry[0], cf.geometry[2], cf.geometry[1], cf.geometry[3]
    ax.imshow(cf.elevation_raster, origin='upper', extent=extent, cmap=cm.gray)

    for catchment_cells in cf.cell_data.itervalues():
        add_plot_polygons(ax, [cell["cell"] for cell in catchment_cells], color=color_map["cell"])
    for catchment_land_type in cf.catchment_land_types.itervalues():
        for k,v in catchment_land_type.iteritems():
            add_plot_polygons(ax, v, color=color_map[k])

    geometry = cf.geometry
    ax.set_xlim(geometry[0], geometry[2])
    ax.set_ylim(geometry[1], geometry[3])
    plt.show()

def nea_nidelv_example():
    x0 = 557600.0
    y0 = 6960000.0
    dx = 1000
    dy = 1000
    nx = 122
    ny = 75
    #catch_id=[1966,2728,1330,3178,2195,1228,100011,2465,2041,2718,2277,3002,1000010,3630,2129,1726,1443,2198,1394,1867,1308,2545,2640,1996,2402,1996,2446,3536]
    #run_cell_example2(x0, y0, dx, dy, nx, ny, catch_id)
    # test fetching for regulated catchments
    indices=[38, 87, 115, 188, 259, 291, 292, 295, 389, 465, 496, 516, 551, 780]
    run_cell_example2('regulated','POWER_PLANT_ID',x0, y0, dx, dy, nx, ny, indices)
    # test fetching for unregulated catchments
    #feltnr=[1691,1686]
    #run_cell_example2('unregulated','FELTNR',x0, y0, dx, dy, nx, ny, feltnr)
    
    
def test_example():
    x0 = 350000.0
    y0 = 6751000.0
    dx = 1000
    dy = 1000
    nx = 32
    ny = 24
    #indices = [163,287,332]
    #run_cell_example2('regulated','POWER_PLANT_ID',x0, y0, dx, dy, nx, ny, indices)
    indices = [1225,1226]
    run_cell_example2('unregulated','FELTNR',x0, y0, dx, dy, nx, ny, indices)


def central_region_example():
    x0 = 340000
    y0 = 6572000
    dx  = 1000
    dy  = 1000
    nx = 122
    ny = 95
    indices=[446, 203]
    run_cell_example(x0, y0, dx, dy, nx, ny, indices)

if __name__ == "__main__":
    test_example()
    #nea_nidelv_example()
    #central_region_example()
    #dtm_example()
