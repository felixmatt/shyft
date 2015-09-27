# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from abc import ABCMeta, abstractmethod

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
from shapely.prepared import prep
import gdal
import os
from ..interfaces import RegionModelRepository


class GisDataFetchError(Exception): pass

class GridSpecification(object):
    """
    Defines a grid, as upper left x0,y0, dx,dy,nx,ny

    """
    def __init__(self,x0,y0,dx,dy,nx,ny):
        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy
        self.nx = nx
        self.ny = ny

    @property
    def geometry(self):
        """
        Return
        ------
        returns a list describing the bounding box of the gridspec
        upperleft x0,y0 and lower right x1,y1
         [x0,y0,x1,y1]
        """
        return [self.x0, self.y0, self.x0 + self.dx*self.nx , self.y0 + self.dy*self.ny]
    
    def cells(self,elevations):
        """
        Parameters
        ----------
        elevations:array[y,x]
            - a array[y,x] with elevations, layout is lower-to upper (opposite direction of the grid!)
        Return
        ------
        a list of shapely box'es flat list for nx,ny
        """
        r = []
        for i in xrange(self.nx):
            for j in xrange(self.ny):
                r.append((box(self.x0 + i*self.dx, self.y0 + j*self.dy, self.x0 + (i + 1)*self.dx,  self.y0 + (j + 1)*self.dy), float(elevations[self.ny-j-1,i]))) 
        return r

class BaseGisDataFetcher(object):
    """
    Statkraft GIS services are published on a set of servers, there is also a staging enviroment(preprod)
    for testing new versions of the services before going into production environment (prod)

    This class handles basic query/response, using an url-template for the query
    arguments to the constructor helps building up this template by means of
    server_name - the host of the server where GIS -services are published
    server_port - the port-number for the service
    service_index - arc-gis specific, we can publish several services at one url, the service_index selects
                    which of those we adress
    in addition, the arc-gis service request do have a set of standard parameters that are passed along
    with the http-request.
    This is represented as a dictinary (key-value)-pairs of named arguments.
    The python request package is used to pass on the request, and get the response.
    In general we try using json format wherever reasonable.

    """

    def __init__(self, geometry=None, server_name=None, server_port=None, service_index=None, epsg_id=32633):
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
        """
        Returns
        -------
        A ready to use url based on
        url-template,server-name,-port and service index 
        """
        url = self.url_template.format(self.server_name, self.server_port, self.service_index)
        return url

    def get_query(self, geometry=None):
        """
        Parameters
        ----------
        geometry:list
            - [x0,y0,x1,y1] the epsg_id ref. bounding box limiting the search
        """
        q = copy.deepcopy(self.query)
        if geometry is None:
            geometry = self.geometry
        if not geometry:
            q["geometry"] = ""
        else:
            q["geometry"] = ",".join([str(c) for c in geometry])
        return q


class LandTypeFetcher(BaseGisDataFetcher):

    def __init__(self, geometry=None, epsg_id=32633):
        super(LandTypeFetcher, self).__init__(geometry=geometry, server_name="oslwvagi001p", server_port="6080", service_index=0, epsg_id=epsg_id)
        self.name_to_layer_map = {"glacier":0,"forest":1,"lake":2}
        self.query["outFields"]="OBJECTID"
    
    @property
    def en_field_names(self):
        return self.name_to_layer_map.keys()
        
    def build_query(self, name):
        self.service_index=self.name_to_layer_map[name]
        q = self.get_query()
        return q

    def fetch(self,name):
        """
        Parameters
        ----------
        name:string
            - one of en_field_names
        Return
        ------
        shapely multipolygon that is within the geometry boundingbox 
        """
        
        if not name or name not in self.en_field_names:
            raise RuntimeError("Invalid or missing land_type_name 'name' not given")

        q = self.build_query(name)
        response = requests.get(self.url, params=q)
        if response.status_code != 200:
            raise GisDataFetchError("Could not fetch land type data from gis server.")
        data = response.json()
        polygons = []
        print ("Extracting {} {} features".format(len(data["features"]), name))
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
            print ("gis polygon error count is",error_count)
        return cascaded_union(polygons)

class ReservoirFetcher(BaseGisDataFetcher):

    def __init__(self, geometry=None, epsg_id=32633):
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

    def __init__(self, catchment_type, identifier, epsg_id=32633):
        if (catchment_type=='regulated'):
            service_index = 6
            #self.identifier = 'POWER_PLANT_ID'
        elif (catchment_type=='unregulated'):
            service_index = 7
            #self.identifier = 'FELTNR'
        else:
            raise GisDataFetchError("Undefined catchment type {} - use either regulated or unregulated".format(catchment_type))
        super(CatchmentFetcher, self).__init__(geometry=None, server_name="oslwvagi001p", server_port="6080", service_index=service_index, epsg_id=epsg_id)
        self.identifier = identifier       
        self.query["outFields"] = "{}".format(self.identifier) #additional attributes to be extracted can be specified here

    def build_query(self, **kwargs):
        q = self.get_query(kwargs.pop("geometry", None))
        id_list = kwargs.pop("id_list", None)
        if id_list:
            q["where"] = "{} IN ({})".format(self.identifier,", ".join([str(i) for i in id_list]))
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

    def __init__(self, catchment_type, identifier, grid_specification, id_list, epsg_id=32633):
        self.catchment_type=catchment_type
        self.identifier=identifier
        self.grid_specification=grid_specification
        self.id_list = id_list
        self.cell_data = {}
        self.catchment_land_types = {}
        self.epsg_id = epsg_id
    
    def fetch(self):
        
        catchment_fetcher = CatchmentFetcher(self.catchment_type, self.identifier,self.epsg_id)
        #, geometry=self.grid_specification.geometry, id_list=self.id_list, )
        catchments = catchment_fetcher.fetch(id_list=self.id_list)

        # Construct cells and populate with elevations from tdm
        dtm_fetcher = DTMFetcher(self.grid_specification, self.epsg_id)
        elevations = dtm_fetcher.fetch()
        cells = self.grid_specification.cells(elevations)
        catchment_land_types = {}
        catchment_cells = {}

        # Filter all data with each catchment
        ltf = LandTypeFetcher(geometry=self.grid_specification.geometry,epsg_id=self.epsg_id)
        rf = ReservoirFetcher(epsg_id=self.epsg_id)
        all_reservoir_coords=rf.fetch(geometry=self.grid_specification.geometry);
        all_glaciers=ltf.fetch(name="glacier")
        prep_glaciers=prep(all_glaciers)
        all_lakes   =ltf.fetch(name="lake")
        prep_lakes= prep(all_lakes)
        all_forest  =ltf.fetch(name="forest")
        prep_forest=prep(all_forest)
        print ("Doing catchment loop, n reservoirs", len(all_reservoir_coords))
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
        print ("Done with catchment cell loop, calc fractions")
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

    def __init__(self,grid_specification, epsg_id=32633):
        self.grid_specification=grid_specification
        self.epsg_id = epsg_id
        #self.server_name = "oslwvagi001q" #PREPROD
        self.server_name = "oslwvagi001p" #PROD
        self.server_port = "6080"
        self.url_template = "http://{}:{}/arcgis/rest/services/Enki/Norway_DTM_1000m/ImageServer/exportImage" #PROD
        
        self.query = dict(
                          bboxSR=self.epsg_id,
                          size= "{},{}".format(self.grid_specification.nx, self.grid_specification.ny),
                          bbox=",".join([str(c) for c in self.grid_specification.geometry]),
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

    def fetch(self):
        response = requests.get(self.url, params=self.query, stream=True)
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



class RegionModelConfig(object):
    """
    Describes the needed mapping betwen a symbolic region-model name and
    the fields/properties needed to extract correct boundaries(shapes)
    from the published services.
    The published services needs:
       a where-clause help, to filter out the shapes we are searching for
       * catchment_regulated_type (regulated| unregulated) maps to what service index to use 
       * service_id_field_name

    and the DTM etc..: 
       * bounding box (with epsk_id)

    """
    def __init__(self, name,epsg_id,grid_specification,catchment_regulated_type,service_id_field_name,id_list):
        """
        """
        self.name=name
        self.epsg_id=epsg_id
        self.grid_specification=grid_specification
        self.catchment_regulated_type=catchment_regulated_type
        self.service_id_field_name=service_id_field_name
        self.id_list=id_list


class GisRegionModelRepository(RegionModelRepository):
    """
    Statkraft GIS service based version of repository for RegionModel objects.
    """
     
    def __init__(self, region_id_config):
        """
        Parameters
        ----------
        region_id_config: dictionary(region_id:RegionModelConfig)
        """
        self._region_id_config=region_id_config

    def _get_cell_data_info(self,region_id,catchments):
        # alternative parse out from region_id, like
        # neannidelv.regulated.plant_field, or neanidelv.unregulated.catchment ?
        return self._region_id_config[region_id] # return tuple (regulated|unregulated,'POWER_PLANT_ID'|CATCH_ID', 'FELTNR',[id1, id2])

    def get_region_model(self, region_id, region_model, catchments=None):
        """
        Return a fully specified shyft api region_model for region_id.

        Parameters
        -----------
        region_id: string
            unique identifier of region in data
        region_model: shyft.api type
            model to construct. Has cell constructor and region/catchment
            parameter constructor.
        catchments: list of unique integers
            catchment id_list when extracting a region consisting of a subset
            of the catchments
        has attribs to construct  params and cells etc.

        Returns
        -------
        region_model: shyft.api type

        ```
        # Pseudo code below
        # Concrete implementation must construct cells, region_parameter
        # and catchment_parameters and return a region_model

        # Use data to create cells
        cells = [region_model.cell_type(*args, **kwargs) for cell in region]

         # Use data to create regional parameters
        region_parameter = region_model.parameter_type(*args, **kwargs)

        # Use data to override catchment parameters
        catchment_parameters = {}
        for all catchments in region:
            catchment_parameters[c] = region_model.parameter_type(*args,
                                                                  **kwargs)
        return region_model(cells, region_parameter, catchment_parameters)
        ```
        """
        # map region_id into catchment_type, identifier and indicies
        # catchment_type = {regulated,unregulated} 
        #  regulated: identifier={POWER_PLANT_ID,CATCH_ID}
        #  unregulated: {FELTNR}
        #  identifier list... []
        #
        rm= self._get_cell_data_info(region_id,catchments)
        cell_info_service = CellDataFetcher(rm.catchment_regulated_type, rm.service_id_field_name,rm.grid_specification,rm.id_list,rm.epsg_id)

        result=cell_info_service.fetch()
        return result

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

def run_cell_example(catchment_type,identifier,x0, y0, dx, dy, nx, ny, catch_indicies,epsg_id):
    grid_spec=GridSpecification(x0, y0, dx, dy, nx, ny)
    cf = CellDataFetcher(catchment_type,identifier,grid_spec, id_list=catch_indicies,epsg_id=epsg_id)
    print( "Start fetching data")
    cf.fetch()
    print ("Done, now preparing plot")
    # Plot the extracted data
    fig, ax = plt.subplots(1)
    color_map = {"forest": 'g', "lake": 'b', "glacier": 'r', "cell": "0.75", "reservoir": "purple"}

    extent = grid_spec.geometry[0], grid_spec.geometry[2], grid_spec.geometry[1], grid_spec.geometry[3]
    ax.imshow(cf.elevation_raster, origin='upper', extent=extent, cmap=cm.gray)

    for catchment_cells in cf.cell_data.itervalues():
        add_plot_polygons(ax, [cell["cell"] for cell in catchment_cells], color=color_map["cell"])
    for catchment_land_type in cf.catchment_land_types.itervalues():
        for k,v in catchment_land_type.iteritems():
            add_plot_polygons(ax, v, color=color_map[k])

    geometry = grid_spec.geometry
    ax.set_xlim(geometry[0], geometry[2])
    ax.set_ylim(geometry[1], geometry[3])
    plt.show()

def nea_nidelv_example(epsg_id):
    x0 = 270000.0
    y0 = 6960000.0
    dx = 1000
    dy = 1000
    nx = 105
    ny = 75
    # test fetching for regulated catchments using CATCH_ID
    id_list=[1228,1308,1394,1443,1726,1867,1996,2041,2129,2195,2198,2277,2402,2446,2465,2545,2640,2718,3002,3536,3630,1000010,1000011]
    run_cell_example('regulated','CATCH_ID',x0, y0, dx, dy, nx, ny, id_list,epsg_id=epsg_id)
    # test fetching for regulated catchments using POWER_PLANT_ID
    ##id_list=[38, 87, 115, 188, 259, 291, 292, 295, 389, 465, 496, 516, 551, 780]
    #id_list=[38,115,137,188,291,292,389,465,496,551,780,1371,1436]   
    #run_cell_example('regulated','POWER_PLANT_ID',x0, y0, dx, dy, nx, ny, id_list,epsg_id=epsg_id)
    # test fetching for unregulated catchments using FELTNR
    #id_list=[1691,1686]
    #run_cell_example2('unregulated','FELTNR',x0, y0, dx, dy, nx, ny, id_list,epsg_id=epsg_id)
    
def vinjevatn_example(epsg_id):
    x0 = 73000.0
    y0 = 6613000.0
    dx = 1000
    dy = 1000
    nx = 40
    ny = 35
    # test fetching for regulated catchments using CATCH_ID
    #id_list=[2668,2936,2937,2938,2939,2940,2941,2942]
    #run_cell_example('regulated','CATCH_ID',x0, y0, dx, dy, nx, ny, id_list,epsg_id=epsg_id)
    # test fetching for regulated catchments using POWER_PLANT_ID
    id_list=[446]
    run_cell_example('regulated','POWER_PLANT_ID',x0, y0, dx, dy, nx, ny, id_list,epsg_id=epsg_id)
    # test fetching for unregulated catchments using FELTNR
    #id_list=[645,702]
    #run_cell_example2('unregulated','FELTNR',x0, y0, dx, dy, nx, ny, id_list,epsg_id=epsg_id)
    
    
def test_example():
    x0 = 350000.0
    y0 = 6751000.0
    dx = 1000
    dy = 1000
    nx = 32
    ny = 24
    #id_list = [163,287,332]
    #run_cell_example2('regulated','POWER_PLANT_ID',x0, y0, dx, dy, nx, ny, id_list)
    id_list = [1225,1226]
    run_cell_example('unregulated','FELTNR',x0, y0, dx, dy, nx, ny, id_list)

if __name__ == "__main__":
    #vinjevatn_example(32633)
    nea_nidelv_example(32633)
    #test_example()
    #dtm_example()
