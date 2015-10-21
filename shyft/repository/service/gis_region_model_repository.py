# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import requests
import copy
from pyproj import Proj
from pyproj import transform
import numpy as np
import tempfile

from shapely.geometry import Polygon, MultiPolygon, box, Point
from shapely.ops import cascaded_union
from shapely.prepared import prep
import gdal
import os
from ..interfaces import RegionModelRepository
from ..interfaces import BoundingRegion
from shyft import api

class GisDataFetchError(Exception): pass

class GridSpecification(BoundingRegion):
    """
    Defines a grid, as lower left x0,y0, dx,dy,nx,ny
    in the specified epsg_id coordinate system
    given a coordindate system with y-axis positive upwards.

    """
    def __init__(self,epsg_id,x0,y0,dx,dy,nx,ny):
        self.epsg_id=epsg_id
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
        lower left x0,y0 and upper right x1,y1
         [x0,y0,x1,y1]
        """
        return [self.x0, self.y0, self.x0 + self.dx*self.nx , self.y0 + self.dy*self.ny]
    
    def cells(self,elevations):
        """
        Parameters
        ----------
        elevations:array[y,x]
            - a array[y,x] with elevations, layout is lower-to upper
        Return
        ------
        a list of (shapely box'es, elevation) flat list for nx,ny
        """
        r = []
        for i in range(self.nx):
            for j in range(self.ny):
                r.append((box(self.x0 + i*self.dx, self.y0 + j*self.dy, self.x0 + (i + 1)*self.dx,  self.y0 + (j + 1)*self.dy), float(elevations[j,i]))) 
        return r

    def bounding_box(self, epsg):
        """ implementation of interface.BoundingRegion  """
        epsg = str(epsg)
        x=np.array([self.x0,self.x0+self.dx*self.nx,self.x0+self.dx*self.nx,self.x0],dtype="d")
        y=np.array([self.y0,self.y0,self.y0+self.dy*self.ny,self.y0+self.dy*self.ny],dtype="d")
        if epsg == self.epsg():
            return np.array(x), np.array(y)
        else:
            source_cs = "+proj=utm +zone={} +ellps={} +datum={} +units=m +no_defs".format(
                int(self.epsg()) - 32600, "WGS84", "WGS84")
            target_cs = "+proj=utm +zone={} +ellps={} +datum={} +units=m +no_defs".format(
                int(epsg) - 32600, "WGS84", "WGS84")
            source_proj = Proj(source_cs)
            target_proj = Proj(target_cs)
            return [np.array(a) for a in transform(source_proj, target_proj, x, y)]

    def bounding_polygon(self, epsg):
        """ implementation of interface.BoundingRegion  """
        return self.bounding_box(epsg)
    
    def epsg(self):
        """ implementation of interface.BoundingRegion  """        
        return str(self.epsg_id)    

      

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

    def __init__(self, epsg_id,geometry=None, server_name=None, server_port=None, service_index=None):
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

    def __init__(self,  epsg_id,geometry=None):
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

    def __init__(self, epsg_id, geometry=None):
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

    def __init__(self, catchment_type, identifier, epsg_id):
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

    def __init__(self, catchment_type, identifier, grid_specification, id_list):
        self.catchment_type=catchment_type
        self.identifier=identifier
        self.grid_specification=grid_specification
        self.id_list = id_list
        self.cell_data = {}
        self.catchment_land_types = {}

    @property
    def epsg_id(self):
        return self.grid_specification.epsg_id
        
    def fetch(self):
        
        catchment_fetcher = CatchmentFetcher(self.catchment_type, self.identifier,self.epsg_id)
        #, geometry=self.grid_specification.geometry, id_list=self.id_list, )
        catchments = catchment_fetcher.fetch(id_list=self.id_list)

        # Construct cells and populate with elevations from tdm
        dtm_fetcher = DTMFetcher(self.grid_specification)
        elevations = dtm_fetcher.fetch()
        cells = self.grid_specification.cells(elevations)
        catchment_land_types = {}
        catchment_cells = {}

        # Filter all data with each catchment
        ltf = LandTypeFetcher(geometry=self.grid_specification.geometry,epsg_id=self.grid_specification.epsg_id)
        rf = ReservoirFetcher(epsg_id=self.grid_specification.epsg_id)
        all_reservoir_coords=rf.fetch(geometry=self.grid_specification.geometry);
        all_glaciers=ltf.fetch(name="glacier")
        prep_glaciers=prep(all_glaciers)
        all_lakes   =ltf.fetch(name="lake")
        prep_lakes= prep(all_lakes)
        #all_forest  =ltf.fetch(name="forest")
        #prep_forest=prep(all_forest)
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
            #if prep_forest.intersects(catchment): # we are not using forest at the moment, and it takes time!!
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
        return self.grid_specification.geometry

class DTMFetcher(object):

    def __init__(self,grid_specification):
        self.grid_specification=grid_specification
        self.server_name = "oslwvagi001p" #PROD
        self.server_port = "6080"
        self.url_template = "http://{}:{}/arcgis/rest/services/Enki/Norway_DTM_1000m/ImageServer/exportImage" #PROD
        
        self.query = dict(
                          bboxSR=self.grid_specification.epsg_id,
                          size= "{},{}".format(self.grid_specification.nx, self.grid_specification.ny),
                          bbox=",".join([str(c) for c in self.grid_specification.geometry]),
                          imageSR=self.grid_specification.epsg_id,
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
       
    This class is used to configure the GisRegionModelRepository,
    so that it have a list of known RegionModels,identified by names

    """
    def __init__(self, name,region_model_type,region_parameters,grid_specification,catchment_regulated_type,service_id_field_name,id_list):
        """
        Parameters
        ----------
        name:string
         - name of the Region-model, like Tistel-ptgsk, Tistel-ptgsk-opt etc.
        region_model_type: shyft api type
         - like pt_gs_k.PTGSKModel
        region_parameters: region_model_type.parameter_t()
         - specifies the concrete parameters at region-level that should be used
        grid_specifiction: GridSpecification
         - specifies the grid, provides boundingbox and means of creating the cells
        catchment_regulated_type: string
         - 'REGULATED'|'UNREGULATED' - type of catchments used in statkraft
        service_id_field_name:string
         - specifies the service- where clause field, that is matched up against the id_list
        id_list:list of identifiers, int
         - specifies the identifiers that should be retreived (matched against the service_id_field_name) 
         
        TODO: consider also to add catchment level parameters
        
        """
        self.name=name
        self.region_model_type=region_model_type
        self.region_parameters=region_parameters
        self.grid_specification=grid_specification
        self.catchment_regulated_type=catchment_regulated_type
        self.service_id_field_name=service_id_field_name
        self.id_list=id_list
        
    @property
    def epsg_id(self):
        return self.grid_specification.epsg_id



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

    def get_region_model(self, region_id,  catchments=None):
        """
        Return a fully specified shyft api region_model for region_id.

        Parameters
        -----------
        region_id: string
            unique identifier of region in data
       
        catchments: list of unique integers
            catchment id_list when extracting a region consisting of a subset
            of the catchments
        has attribs to construct  params and cells etc.

        Returns
        -------
        region_model: shyft.api type with
           .bounding_region = grid_specification - as fetched from the rm.config
           .catchment_id_map = array, where i'th item is the external catchment'id 
           .gis_info = result from CellDataFetcher - used to fetch the grid spec (to help plot etc)
           {
             'cell_data': {catchment_id:[{'cell':shapely-shapes,'elevation':moh,'glacier':area,'lake':area,'reservoir':area,'forest':area}]}
             'catchment_land_types':{catchment_id:{'glacier':[shapelys..],'forest':[shapelys..],'lake':[shapelys..],'reservoir':[shapelys..]}}
             'elevation_raster': np.array(dtype.float64)
           }
      
        """

        rm= self._get_cell_data_info(region_id,catchments)# fetch region model info needed to fetch efficiently
        cell_info_service = CellDataFetcher(rm.catchment_regulated_type, rm.service_id_field_name,rm.grid_specification,rm.id_list)
        result=cell_info_service.fetch() # clumsy result, we can adjust this..
        cell_info=result['cell_data'] # this is the part we need here
        cell_vector = rm.region_model_type.cell_t.vector_t()
        radiation_slope_factor=0.9 # todo: get it from service layer 
        catchment_id_map = [] # needed to build up the external c-id to shyft core internal 0-based c-ids
        for c_id,c_info_list in cell_info.iteritems():
            if not c_id == 0: # only cells with c_id different from 0
                if not c_id in catchment_id_map:
                    catchment_id_map.append(c_id)
                    c_id_0=len(catchment_id_map)-1
                else:
                    c_id_0=catchment_id_map.index(c_id)
                for c_info in c_info_list:
                    shape=c_info['cell'] # todo fetcher should return geopoint,area, ltf..
                    z=c_info['elevation']
                    geopoint=api.GeoPoint(shape.centroid.x,shape.centroid.y,z)
                    area=shape.area
                    ltf=api.LandTypeFractions()
                    ltf.set_fractions(c_info.get('glacier',0.0),c_info.get('lake',0.0),c_info.get('reservoir',0.0),c_info.get('forest',0.0))
                    cell = rm.region_model_type.cell_t()
                    cell.geo = api.GeoCellData(geopoint, area, c_id_0, radiation_slope_factor,ltf)
                    cell_vector.append(cell)
        catchment_parameter_map=rm.region_model_type.parameter_t.map_t()
        #todo add catchment level parameters to map
        region_model= rm.region_model_type(cell_vector,rm.region_parameters,catchment_parameter_map)
        region_model.bounding_region=rm.grid_specification  # mandatory for orchestration
        region_model.catchment_id_map=catchment_id_map #needed to map from externa c_id to 0-based c_id used internally in
        region_model.gis_info=result # opt:needed for internal statkraft use/presentation
        return region_model



