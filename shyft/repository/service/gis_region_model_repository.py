# -*- coding: utf-8 -*-
import requests
import copy
from typing import List, Tuple, Dict
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
# from shyft import api
from shyft import shyftdata_dir
import pickle

from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)  # to suppress authentication warning when using https variant

#
# Statkraft default server and service constants
# these constants is used for forming the url-request
#
# note: We rely on str as immutable type when using as default-arguments to the methods
#

# https variant
#primary_server: str = r"oslwvagi002p"
#secondary_server: str = r"oslwvagi001q"
#port_num: str = "6080"
# https variant
primary_server: str  = r"gisserver.statkraft.com"
secondary_server: str  = r"pregisserver.statkraft.com"
port_num: str = "6443"

# nordic service and catchment_type mappings (todo:should be cfg.classes or at least enums)
nordic_service: str = "SHyFT"
nordic_dem: str = r"Norway_DTM_1000m"

nordic_catchment_type_regulated: str = r"regulated"
nordic_catchment_type_unregulated: str = r"unregulated"
nordic_catchment_type_ltm: str = r"LTM"

# peru service and catchment_type mappings (todo: as for nordic, cfg class, or enums)
peru_service: str = r"Peru_ShyFT"
peru_dem: str = r"Peru_DEM_250m"

peru_catchment_type: str = r"peru_catchment"
peru_catchment_id_name: str = r"SUBCATCH_ID"
peru_subcatch_id_name: str = r"CATCH_ID"


class GisDataFetchError(Exception):
    pass


def set_no_proxy(server: str) -> None:
    env = os.environ.get('NO_PROXY', '')
    if server not in env:
        if env:
            env += ', '
        os.environ['NO_PROXY'] = env + server


class GridSpecification(BoundingRegion):
    """
    Defines a grid, as lower left x0, y0, dx, dy, nx, ny
    in the specified epsg_id coordinate system
    given a coordindate system with y-axis positive upwards.

    """

    def __init__(self, epsg_id: int, x0: float, y0: float, dx: float, dy: float, nx: int, ny: int):
        self._epsg_id = epsg_id
        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy
        self.nx = nx
        self.ny = ny

    @property
    def geometry(self) -> List[float]:
        """
        Return
        ------
        returns a list describing the bounding box of the gridspec
        lower left x0, y0 and upper right x1, y1
         [x0, y0, x1, y1]
        """
        return [self.x0, self.y0, self.x0 + self.dx*self.nx, self.y0 + self.dy*self.ny]

    def cells(self, elevations):
        """
        Parameters
        ----------
        elevations:array[y, x]
            - a array[y, x] with elevations, layout is lower-to upper
        Return
        ------
        a list of (shapely box'es, elevation) flat list for nx, ny
        """
        x0, dx = self.x0, self.dx
        y0, dy = self.y0, self.dy
        return [(box(x0 + i*dx, y0 + j*dy, x0 + (i + 1)*dx, y0 + (j + 1)*dy), float(e))
                for (i, j), e in np.ndenumerate(np.flipud(elevations).T)]

    def bounding_box(self, epsg: int) -> Tuple[np.ndarray, np.ndarray]:
        """Implementation of interface.BoundingRegion"""
        epsg = str(epsg)
        x0, dx, nx = self.x0, self.dx, self.nx
        y0, dy, ny = self.y0, self.dy, self.ny

        x = np.array([x0, x0 + dx*nx, x0 + dx*nx, x0], dtype="d")
        y = np.array([y0, y0, y0 + dy*ny, y0 + dy*ny], dtype="d")
        if epsg == self.epsg():
            return np.array(x), np.array(y)
        else:
            source_cs = "+init=EPSG:{}".format(self.epsg())
            target_cs = "+init=EPSG:{}".format(epsg)
            source_proj = Proj(source_cs)
            target_proj = Proj(target_cs)
            r = [np.array(a) for a in transform(source_proj, target_proj, x, y)]
            return r[0], r[1]

    def bounding_polygon(self, epsg: int) -> Tuple[np.ndarray, np.ndarray]:
        """Implementation of interface.BoundingRegion"""
        return self.bounding_box(epsg)

    def epsg(self) -> str:
        """Implementation of interface.BoundingRegion"""
        return str(self._epsg_id)

    @property
    def epsg_id(self) -> int:
        return self._epsg_id


class BaseGisDataFetcher(object):
    """
    Statkraft GIS services are published on a set of servers, there is also a staging enviroment(preprod)
    for testing new versions of the services before going into production environment (prod)

    This class handles basic query/response, using an url-template for the query
    arguments to the constructor helps building up this template by means of
    server_name - the host of the server where GIS -services are published
    server_port - the port-number for the service
    service_index - arc-gis specific, we can publish several services at one url, 
    the service_index selects which of those we adress
    in addition, the arc-gis service request does have a set of standard parameters that are passed along
    with the http-request.
    This is represented as a dictionary (key-value)-pairs of named arguments.
    The python request package is used to pass on the request, and get the response.
    In general we try using json format wherever reasonable.

    """

    def __init__(self, epsg_id, geometry=None, server_name=primary_server, server_name_preprod=secondary_server,
                 server_port=port_num, service_index=None, sub_service=nordic_service):
        self.server_name = server_name
        self.server_name_preprod = server_name_preprod
        self.server_port = server_port
        self.service_index = service_index
        self.geometry = geometry
        self.epsg_id = epsg_id
        #self.url_template = "http://{}:{}/arcgis/rest/services/SHyFT/" + sub_service + "/MapServer/{}/query"  # http variant
        self.url_template = "https://{}:{}/arcgis/rest/services/SHyFT/" + sub_service + "/MapServer/{}/query"  # https variant
        set_no_proxy(self.server_name)
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
        url-template, server-name, -port and service index 
        """
        url = self.url_template.format(self.server_name, self.server_port, self.service_index)
        return url

    def _get_response(self, url_, msg, **kwargs):
        kwargs.update({'verify': False})  # to go around authentication error when using https -> ssl.SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:749)
        response = requests.get(url_, **kwargs)
        if response.status_code != 200:
            raise GisDataFetchError('Could not fetch {} from gis server {}'.format(msg, self.server_name))
        data = response.json()
        if "features" not in data:
            raise GisDataFetchError(
                "GeoJson data for {} data fetching from server {} missing mandatory field, please check your gis service or your query.".format(
                    msg, self.server_name)
            )
        return data

    def get_response(self, msg, **kwargs):
        try:
            data = self._get_response(self.url, msg, **kwargs)
        except Exception as e:
            print('Error in fetching GIS data: {}'.format(msg))
            print('Error description: {}'.format(str(e)))
            print('Switching from PROD server {} to PREPROD server {}'.format(self.server_name, self.server_name_preprod))
            url_ = self.url.replace(self.server_name, self.server_name_preprod)
            set_no_proxy(self.server_name_preprod)
            data = self._get_response(url_, msg, **kwargs)
        return data

    def get_query(self, geometry=None):
        """
        Parameters
        ----------
        geometry:list
            - [x0, y0, x1, y1] the epsg_id ref. bounding box limiting the search
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
    def __init__(self, epsg_id, geometry=None, server_name=primary_server, server_name_preprod=secondary_server, sub_service=nordic_service):
        super(LandTypeFetcher, self).__init__(geometry=geometry,
                                              # server_name=primary_server,
                                              # server_port="6080",
                                              service_index=0, epsg_id=epsg_id,
                                              server_name=server_name,
                                              server_name_preprod=server_name_preprod,
                                              sub_service=sub_service
                                              )
        self.name_to_layer_map = {"glacier": 0, "forest": 1, "lake": 2}  # if sub_service == "SHyFT" else {"glacier": 1, "forest": 2, "lake": 3}
        self.query["outFields"] = "OBJECTID"

    @property
    def en_field_names(self):
        return self.name_to_layer_map.keys()

    def build_query(self, name):
        self.service_index = self.name_to_layer_map[name]
        return self.get_query()

    def fetch(self, name):
        """
        Parameters
        ----------
        name:string
            - one of en_field_names
        Return
        ------
        shapely multipolygon that is within the geometry boundingbox 
        """
        print('Fetching {} polygon data...'.format(name))
        if not name or name not in self.en_field_names:
            raise RuntimeError("Invalid or missing land_type_name 'name' not given")

        q = self.build_query(name)
        # response = requests.get(self.url, params=q)
        # if response.status_code != 200:
        #     raise GisDataFetchError("Could not fetch land type data from gis server.")
        # data = response.json()
        data = self.get_response(name, params=q)
        polygons = []
        if 'error' in data.keys():
            raise GisDataFetchError("Failed in GIS service:" + data['error']['message'])

        # TODO replace this with callback to user/client: print("Extracting {} {} features".format(len(data["features"]), name))
        error_count = 0
        for feature in data["features"]:
            shell = feature["geometry"]["rings"][0]
            holes = feature["geometry"]["rings"][1:]
            holes = holes if holes else None
            polygon = Polygon(shell=shell, holes=holes)
            if polygon.is_valid:
                polygons.append(polygon)
            else:
                error_count += 1
        if error_count > 0:
            print("{} polygon error count is".format(name), error_count)
        return cascaded_union(polygons)


class ReservoirFetcher(BaseGisDataFetcher):
    def __init__(self, epsg_id, geometry=None, server_name=primary_server, server_name_preprod=secondary_server, sub_service=nordic_service):
        super(ReservoirFetcher, self).__init__(geometry=geometry,
                                               server_name=server_name,
                                               server_name_preprod=server_name_preprod,
                                               service_index=6 if sub_service == nordic_service else 4,
                                               epsg_id=epsg_id,
                                               sub_service=sub_service)
        self.query["where"] = "1 = 1"
        self.query["outFields"] = "OBJECTID"

    def fetch(self, **kwargs):
        print('Fetching reservoir polygon data...')
        q = self.get_query(kwargs.pop("geometry", None))
        # response = requests.get(self.url, params=q)
        # if response.status_code != 200:
        #     raise GisDataFetchError("Could not fetch reservoir data from gis server.")
        # data = response.json()
        # if "features" not in data:
        #     raise GisDataFetchError(
        #         "GeoJson data missing mandatory field, please check your gis service or your query.")
        data = self.get_response("Reservoir", params=q)
        points = []
        for feature in data["features"]:
            x = feature["geometry"]["x"]
            y = feature["geometry"]["y"]
            points.append(Point(x, y))
        return points


class CatchmentFetcher(BaseGisDataFetcher):
    def __init__(self, catchment_type: str, identifier: str, epsg_id: int, server_port: str ='6443',
                 server_name: str = primary_server, server_name_preprod: str = secondary_server):
        sub_service = nordic_service
        if catchment_type == nordic_catchment_type_regulated:
            if identifier == 'SUBCATCH_ID':
                service_index = 4
            elif identifier in ['CATCH_ID', 'POWER_PLANT_ID']:
                service_index = 7
            else:
                raise GisDataFetchError(
                    "Unknown identifier {} for catchment_type {}. Use one of 'SUBCATCH_ID', 'CATCH_ID', 'POWER_PLANT_ID'".format(identifier, catchment_type))
        elif catchment_type == nordic_catchment_type_unregulated:
            service_index = 8
        elif catchment_type == nordic_catchment_type_ltm:
            service_index = 3
        elif catchment_type == peru_catchment_type:
            if identifier == 'SUBCATCH_ID':
                service_index = 5
            else:
                service_index = 6  # SUBCATCH_ID is used, 1 is Yaupi
            sub_service = peru_service
            #identifier = peru_subcatch_id_name
        else:
            raise GisDataFetchError(
                "Undefined catchment type {}. Use one of 'regulated', 'unregulated' or 'LTM'".format(catchment_type))
        super(CatchmentFetcher, self).__init__(geometry=None,
                                               server_name=server_name, server_port=server_port,
                                               server_name_preprod=server_name_preprod,
                                               service_index=service_index,
                                               epsg_id=epsg_id, sub_service=sub_service)
        self.identifier = identifier
        self.query["outFields"] = "{}".format(
            self.identifier)  # additional attributes to be extracted can be specified here

    def build_query(self, **kwargs):
        q = self.get_query(kwargs.pop("geometry", None))
        id_list = kwargs.pop("id_list", None)
        if id_list:
            if isinstance(id_list[0], str):
                id_list = ["'%s'" % value for value in id_list]
            q["where"] = "{} IN ({})".format(self.identifier, ", ".join([str(i) for i in id_list]))
        return q

    def fetch(self, **kwargs):
        print('Fetching catchment polygon data...')
        q = self.build_query(**kwargs)
        # response = requests.get(self.url, params=q)
        # if response.status_code != 200:
        #     raise GisDataFetchError("Could not fetch catchment index data from gis server.")
        # data = response.json()
        data = self.get_response("Catchment index", params=q)
        # from IPython.core.debugger import Tracer; Tracer()()
        polygons = {}
        error_count = 0
        for feature in data['features']:
            c_id = feature['attributes'][self.identifier]
            shell = feature["geometry"]["rings"][0]
            holes = feature["geometry"]["rings"][1:]
            polygon = Polygon(shell=shell, holes=holes)
            if polygon.is_valid:
                if c_id not in polygons:
                    polygons[c_id] = []
                polygons[c_id].append(polygon)
            else:
                error_count += 1
        if error_count > 0:
            print("Catchment polygon error count is", error_count)
        return {key: cascaded_union(polygons[key]) for key in polygons if polygons[key]}


class CellDataFetcher(object):
    def __init__(self, catchment_type: str, identifier: str, grid_specification: GridSpecification, id_list: List[int],
                 server_name: str = primary_server, server_name_preprod: str = secondary_server, calc_forest_frac: bool = False):
        self.server_name: str = server_name
        self.server_name_preprod: str = server_name_preprod
        self.catchment_type: str = catchment_type
        self.identifier: str = identifier
        self.grid_specification: GridSpecification = grid_specification
        self.id_list: List[int] = id_list
        self.cell_data = {}
        self.catchment_land_types = {}
        self.calc_forest_frac: bool = calc_forest_frac
        self.elevation_raster = None

    @property
    def epsg_id(self) -> int:
        return int(self.grid_specification.epsg())

    @property
    def dem(self) -> str:
        return peru_dem if self.catchment_type == peru_catchment_type else nordic_dem

    @property
    def sub_service(self) -> str:
        return peru_service if self.catchment_type == peru_catchment_type else nordic_service

    def fetch(self):

        catchment_fetcher = CatchmentFetcher(self.catchment_type, self.identifier, self.epsg_id,
                                             server_name=self.server_name, server_name_preprod=self.server_name_preprod)
        catchments = catchment_fetcher.fetch(id_list=self.id_list)

        # Construct cells and populate with elevations from tdm
        dtm_fetcher = DTMFetcher(self.grid_specification, dem=self.dem,
                                 server_name=self.server_name, server_name_preprod=self.server_name_preprod
                                 )
        elevations = dtm_fetcher.fetch()
        cells = self.grid_specification.cells(elevations)
        catchment_land_types = {}
        catchment_cells = {}

        # Filter all data with each catchment
        epsg = self.grid_specification.epsg()
        ltf = LandTypeFetcher(geometry=self.grid_specification.geometry, epsg_id=epsg, sub_service=self.sub_service,
                              server_name=self.server_name, server_name_preprod=self.server_name_preprod)
        rf = ReservoirFetcher(epsg_id=epsg, sub_service=self.sub_service,
                              server_name=self.server_name, server_name_preprod=self.server_name_preprod)
        all_reservoir_coords = rf.fetch(geometry=self.grid_specification.geometry)
        all_glaciers = ltf.fetch(name="glacier")
        prep_glaciers = prep(all_glaciers)
        all_lakes = ltf.fetch(name="lake")
        prep_lakes = prep(all_lakes)
        all_forest = None
        prep_forest = None
        if self.calc_forest_frac:
            all_forest = ltf.fetch(name="forest")
            prep_forest = prep(all_forest)
        print("Doing catchment loop, n reservoirs", len(all_reservoir_coords))
        for catchment_id, catchment in catchments.items():
            if catchment_id not in catchment_land_types:  # SiH: default land-type, plus the special ones fetched below
                catchment_land_types[catchment_id] = {}
            if prep_lakes.intersects(catchment):
                lake_in_catchment = all_lakes.intersection(catchment)
                if isinstance(lake_in_catchment, (Polygon, MultiPolygon)) and lake_in_catchment.area > 1000.0:
                    reservoir_list = []
                    for rsv_point in all_reservoir_coords:
                        if isinstance(lake_in_catchment, Polygon):
                            if lake_in_catchment.contains(rsv_point):
                                reservoir_list.append(lake_in_catchment)
                        else:
                            for lake in lake_in_catchment:
                                if lake.contains(rsv_point):
                                    reservoir_list.append(lake)
                    if reservoir_list:
                        reservoir = MultiPolygon(reservoir_list)
                        catchment_land_types[catchment_id]["reservoir"] = reservoir
                        diff = lake_in_catchment.difference(reservoir)
                        if diff.area > 1000.0:
                            catchment_land_types[catchment_id]["lake"] = diff
                    else:
                        catchment_land_types[catchment_id]["lake"] = lake_in_catchment
            if prep_glaciers.intersects(catchment):
                glacier_in_catchment = all_glaciers.intersection(catchment)
                if isinstance(glacier_in_catchment, (Polygon, MultiPolygon)):
                    catchment_land_types[catchment_id]["glacier"] = glacier_in_catchment
            if self.calc_forest_frac:
                if prep_forest.intersects(catchment):  # we are not using forest at the moment, and it takes time!!
                    forest_in_catchment = all_forest.intersection(catchment)
                    if isinstance(forest_in_catchment, (Polygon, MultiPolygon)):
                        catchment_land_types[catchment_id]["forest"] = forest_in_catchment

            catchment_cells[catchment_id] = []
            for cell, elevation in cells:
                if cell.intersects(catchment):
                    catchment_cells[catchment_id].append((cell.intersection(catchment), elevation))

        # Gather cells on a per catchment basis, and compute the area fraction for each landtype
        print("Done with catchment cell loop, calc fractions")
        cell_data = {}
        #frac_error_count = {'lake': 0, 'forest': 0, 'reservoir': 0, 'glacier': 0}
        for catchment_id in catchments.keys():
            cell_data[catchment_id] = []
            for cell, elevation in catchment_cells[catchment_id]:
                data = {"cell": cell, "elevation": elevation}
                for land_type_name, land_type_shape in iter(catchment_land_types[catchment_id].items()):
                    if not land_type_shape.is_valid:
                        # Passed a distance of 0, buffer() can be used to “clean” self-touching or self-crossing polygons such as the classic “bowtie”.
                        # http://toblerity.org/shapely/manual.html
                        land_type_shape = land_type_shape.buffer(0)
                    data[land_type_name] = cell.intersection(land_type_shape).area / cell.area
                    # to debug problems with invalid polygons
                    # print(land_type_name)
                    # try:
                    #     print('cell area', cell.area, '{} area'.format(land_type_name), land_type_shape.area)
                    #     print('is_landtype_shape_valid:', land_type_shape.is_valid)
                    #     data[land_type_name] = cell.intersection(land_type_shape.buffer(0)).area/cell.area
                    # except Exception as e:
                    #     print(str(e))
                    #     print('cell area', cell.area, '{} area'.format(land_type_name), land_type_shape.area)
                    #     frac_error_count[land_type_name] += 1
                cell_data[catchment_id].append(data)
        #print(frac_error_count)
        self.cell_data = cell_data
        self.catchment_land_types = catchment_land_types
        self.elevation_raster = elevations
        return {"cell_data": self.cell_data, "catchment_land_types": self.catchment_land_types,
                "elevation_raster": self.elevation_raster}

    @property
    def geometry(self) -> List[float]:
        """ return grid_specification.geometry, bounding box x0,y0,x1,y1 """
        return self.grid_specification.geometry


class DTMFetcher:
    def __init__(self, grid_specification: GridSpecification, server_name: str = primary_server, server_name_preprod: str = secondary_server, dem: str = nordic_dem):
        self.grid_specification: GridSpecification = grid_specification
        self.server_name: str = server_name  # PROD
        self.server_name_preprod: str = server_name_preprod  # PREPROD
        self.server_port: str = port_num
        self.dem: str = dem
        #self.url_template: str = "http://{}:{}/arcgis/rest/services/SHyFT/{}/ImageServer/exportImage"  # PROD
        self.url_template = "https://{}:{}/arcgis/rest/services/SHyFT/{}/ImageServer/exportImage"  # https variant
        set_no_proxy(self.server_name)
        self.query = dict(
            bboxSR=self.grid_specification.epsg(),
            size="{},{}".format(self.grid_specification.nx, self.grid_specification.ny),
            bbox=",".join([str(c) for c in self.grid_specification.geometry]),
            imageSR=self.grid_specification.epsg(),
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

    @property
    def url(self):
        return self.url_template.format(self.server_name, self.server_port, self.dem)

    def _fetch(self, url_=None):
        if url_ is None:
            url_ = self.url
        response = requests.get(url_, params=self.query, stream=True, verify=False)
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

    def fetch(self):
        print('Fetching DEM data...')
        try:
            data = self._fetch()
        except Exception as e:
            print('Error in fetching GIS data: {}'.format('DTM'))
            print('Error description: {}'.format(str(e)))
            print('Switching from PROD server {} to PREPROD server {}'.format(self.server_name, self.server_name_preprod))
            url_ = self.url.replace(self.server_name, self.server_name_preprod)
            data = self._fetch(url_)
        return data


class RegionModelConfig:
    """
    Describes the needed mapping between a symbolic region-model name and
    the fields/properties needed to extract correct boundaries(shapes)
    from the published services.
    The published services needs:
       a where-clause help, to filter out the shapes we are searching for
       * catchment_regulated_type (regulated| unregulated) maps to what service index to use 
       * service_id_field_name

    and the DTM etc..: 
       * bounding box (with epsk_id)
       
    This class is used to configure the GisRegionModelRepository,
    so that it have a list of known RegionModels, identified by names

    """

    def __init__(self, name: str, region_model_type, region_parameters, grid_specification,
                 catchment_regulated_type: str, service_id_field_name: str, id_list: List[int], catchment_parameters=None,
                 calc_forest_frac: bool = False):
        """
        TODO: consider also to add catchment level parameters

        Parameters
        ----------
        name:string
         - name of the Region-model, like Tistel-ptgsk, Tistel-ptgsk-opt etc.
        region_model_type: shyft api type
         - like pt_gs_k.PTGSKModel
        region_parameters: region_model_type.parameter_t()
         - specifies the concrete parameters at region-level that should be used
        grid_specification: GridSpecification
         - specifies the grid, provides bounding-box and means of creating the cells
        catchment_regulated_type: string
         - 'REGULATED'|'UNREGULATED' - type of catchments used in statkraft
        service_id_field_name:string
         - specifies the service- where clause field, that is matched up against the id_list
        id_list:list of identifiers, int
         - specifies the identifiers that should be fetched (matched against the service_id_field_name)
        catchment_parameters: dictionary with catchment id as key and region_model_type.parameter_t() as value
         - specifies catchment level parameters
         

        """
        self.name: str = name
        self.region_model_type = region_model_type
        self.region_parameters = region_parameters
        self.grid_specification: GridSpecification = grid_specification
        self.catchment_regulated_type: str = catchment_regulated_type
        self.service_id_field_name: str = service_id_field_name
        self.id_list: List[int] = id_list
        self.catchment_parameters = catchment_parameters or {}
        self.calc_forest_frac: bool = calc_forest_frac

    @property
    def epsg_id(self) -> int:
        return self.grid_specification.epsg_id


class CellDataCache(object):
    def __init__(self, folder: str, file_type: str):
        self.file_type = file_type
        self.folder = folder
        self.reader = {'pickle': self._load_from_pkl,
                       # 'netcdf': self._load_from_nc,
                       # 'numpy': self._load_from_npy
                       }
        self.saver = {'pickle': self._dump_to_pkl,
                      # 'netcdf': self._load_from_nc,
                      # 'numpy': self._load_from_npy
                      }
        self.file_ext = {'pickle': 'pkl',
                         # 'netcdf': 'nc',
                         # 'numpy': 'npy'
                         }

    def _make_filename(self, service_id_field_name: str, grid_specification: GridSpecification):
        gs = grid_specification
        file_name = 'EPSG_{}_ID_{}_dX_{}_dY_{}.{}'.format(gs.epsg(), service_id_field_name, int(gs.dx), int(gs.dy),
                                                          self.file_ext[self.file_type])
        return os.path.join(self.folder, file_name)

    def is_file_available(self, service_id_field_name: str, grid_specification: GridSpecification):
        file_path = self._make_filename(service_id_field_name, grid_specification)
        # file_path = os.path.join(self.folder, file_name)
        return os.path.isfile(file_path), file_path

    def remove_cache(self, service_id_field_name: str, grid_specification: GridSpecification):
        file_exists, file_path = self.is_file_available(service_id_field_name, grid_specification)
        if file_exists:
            print('Deleting file {}'.format(file_path))
            os.remove(file_path)
        else:
            print('No cashe file to remove!')

    def get_cell_data(self, service_id_field_name: str, grid_specification: GridSpecification, id_list: List[int]):
        # file_path = 'All_regions_as_dict_poly_wkb.pkl'
        file_path = self._make_filename(service_id_field_name, grid_specification)
        # file_path = os.path.join(self.folder, file_name)
        return self.reader[self.file_type](file_path, id_list)

    def save_cell_data(self, service_id_field_name: str, grid_specification: GridSpecification, cell_data):
        file_path = self._make_filename(service_id_field_name, grid_specification)
        file_status = self.is_file_available(service_id_field_name, grid_specification)[0]
        # file_path = os.path.join(self.folder, file_name)
        self.saver[self.file_type](file_path, cell_data, file_status)

    def _load_from_pkl(self, file_path: str, cids: np.ndarray):
        print('Loading from cache_file {}...'.format(file_path))
        with open(file_path, 'rb') as pkl_file:
            cell_info = pickle.load(pkl_file)
        geo_data = cell_info['geo_data']
        polygons = cell_info['polygons']
        cids_ = np.sort(cids)
        cid_map = cids_
        cids_in_cache = geo_data[:, 4].astype(int)
        if np.in1d(cids, cids_in_cache).sum() != len(cids_):  # Disable fetching if not all cids are found
            geo_data_ext = polygons_ext = cid_map = []
        else:
            idx = np.in1d(cids_in_cache, cids).nonzero()[0]
            geo_data_ext = geo_data[idx]
            polygons_ext = polygons[idx]
            # geo_data_ext[:, 4] = np.searchsorted(cids_, geo_data_ext[:, 4]) # since ID to Index conversion not necessary
        return {'cid_map': cid_map, 'geo_data': geo_data_ext, 'polygons': polygons_ext}

    def _dump_to_pkl(self, file_path: str, cell_data, is_existing_file: bool):
        print('Saving to cache_file {}...'.format(file_path))
        # new_geo_data = cell_data['geo_data'].copy()
        cid_map = cell_data['cid_map']
        # new_geo_data[:, 4] = cid_map[new_geo_data[:, 4].astype(int)] # since ID to Index conversion not necessary
        cell_info = {'geo_data': cell_data['geo_data'], 'polygons': cell_data['polygons']}
        if is_existing_file:
            with open(file_path, 'rb') as pkl_file_in:
                old = pickle.load(pkl_file_in)
            # new_geo_data = cell_data['geo_data']
            # cid_map = cell_data['cid_map']
            # new_geo_data[:, 4] = cid_map[new_geo_data[:, 4].astype(int)]
            old['polygons'] = np.array(old['polygons'])
            old_geo_data = old['geo_data']
            old_cid = old_geo_data[:, 4].astype(int)
            idx_keep = np.invert(np.in1d(old_cid, cid_map)).nonzero()[0]
            if len(idx_keep) > 0:
                cell_info = {'geo_data': np.vstack((old_geo_data[idx_keep], cell_data['geo_data'])),
                             'polygons': np.concatenate((old['polygons'][idx_keep], cell_data['polygons']))
                             }
        with open(file_path, 'wb') as pkl_file_out:
            pickle.dump(cell_info, pkl_file_out, -1)


class GisRegionModelRepository(RegionModelRepository):
    """
    Statkraft GIS service based version of repository for RegionModel objects.
    """
    cell_data_cache = CellDataCache(shyftdata_dir, 'pickle')
    # cache_file_type = 'pickle'
    server_name = primary_server
    server_name_preprod = secondary_server

    def __init__(self, region_id_config: Dict[str, RegionModelConfig], use_cache: bool = False, cache_folder: str = None, cache_file_type: str = None):
        """
        Parameters
        ----------
        region_id_config: dictionary(region_id:RegionModelConfig)
        """
        self._region_id_config: Dict[str, RegionModelConfig] = region_id_config
        self.use_cache: bool = use_cache
        if cache_folder is not None:
            self.cell_data_cache.folder = cache_folder
        if cache_file_type is not None:
            self.cell_data_cache.file_type = cache_file_type

    def _get_cell_data_info(self, region_id: str, catchments=None) -> RegionModelConfig:
        # alternative parse out from region_id, like
        # neanidelv.regulated.plant_field, or neanidelv.unregulated.catchment ?
        return self._region_id_config[region_id]  # return tuple (regulated|unregulated,'POWER_PLANT_ID'|CATCH_ID', 'FELTNR',[id1, id2])

    @classmethod
    def get_cell_data_from_gis(cls, catchment_regulated_type: str, service_id_field_name: str,
                               grid_specification: GridSpecification,
                               id_list: List[int],
                               calc_forest_frac: bool = False):
        print('Fetching gis_data from online GIS database...')
        cell_info_service = CellDataFetcher(catchment_regulated_type, service_id_field_name,
                                            grid_specification, id_list, server_name=cls.server_name,
                                            server_name_preprod=cls.server_name_preprod,
                                            calc_forest_frac=calc_forest_frac)

        result = cell_info_service.fetch()  # clumsy result, we can adjust this.. (I tried to adjust it below)
        cell_data = result['cell_data']  # this is the part we need here
        radiation_slope_factor = 0.9  # todo: get it from service layer
        unknown_fraction = 0.0  # todo: should we just remove this
        print('Making cell_data from gis_data...')
        cids = np.sort(list(cell_data.keys()))
        # geo_data = np.vstack([[c['cell'].centroid.x, c['cell'].centroid.y, c['elevation'], c['cell'].area, idx,
        #                        radiation_slope_factor,c.get('glacier', 0.0), c.get('lake', 0.0),c.get('reservoir', 0.0),
        #                        c.get('forest', 0.0), unknown_fraction] for c in cell_data[cid]]
        #                      for idx, cid in enumerate(cids))
        geo_data = np.vstack([[c['cell'].centroid.x, c['cell'].centroid.y, c['elevation'], c['cell'].area, cid,
                               radiation_slope_factor, c.get('glacier', 0.0), c.get('lake', 0.0),
                               c.get('reservoir', 0.0),
                               c.get('forest', 0.0), unknown_fraction] for c in cell_data[cid]]
                             for cid in cids)
        geo_data[:, -1] = 1 - geo_data[:, -5:-1].sum(axis=1)  # calculating the unknown fraction
        polys = np.concatenate(tuple([c['cell'] for c in cell_data[cid]] for cid in cids))
        return {'cid_map': cids, 'geo_data': geo_data, 'polygons': polys}

    @classmethod
    def get_cell_data_from_cache(cls, service_id_field_name: str, grid_specification: GridSpecification, id_list: List[int]):
        if cls.cell_data_cache.is_file_available(service_id_field_name, grid_specification)[0]:
            cell_info = cls.cell_data_cache.get_cell_data(service_id_field_name,
                                                          grid_specification, id_list)
            if len(cell_info['cid_map']) != 0:
                return cell_info
            else:
                print('MESSAGE: not all catchment IDs requested were found in cache!')
                return None
        else:
            print('MESSAGE: no cache file found!')
            return None

    @classmethod
    def update_cache(cls, catchment_regulated_type: str, service_id_field_name: str, grid_specification: GridSpecification, id_list: List[int], calc_forest_frac: bool = False):
        cell_info = cls.get_cell_data_from_gis(catchment_regulated_type, service_id_field_name, grid_specification, id_list, calc_forest_frac=calc_forest_frac)
        cls.cell_data_cache.save_cell_data(service_id_field_name, grid_specification, cell_info)

    @classmethod
    def remove_cache(cls, service_id_field_name: str, grid_specification: GridSpecification):
        cls.cell_data_cache.remove_cache(service_id_field_name, grid_specification)

    @classmethod
    def save_cell_data_to_cache(cls, service_id_field_name: str, grid_specification: GridSpecification, cell_info):
        cls.cell_data_cache.save_cell_data(service_id_field_name, grid_specification, cell_info)

    @classmethod
    def build_cell_vector(cls, region_model_type, cell_geo_data):
        print('Building cell_vector from cell_data...')
        return region_model_type.cell_t.vector_t.create_from_geo_cell_data_vector(np.ravel(cell_geo_data))

    def get_region_model(self, region_id: str, catchments: List[int] = None):
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

        rm = self._get_cell_data_info(region_id, catchments)  # fetch region model info needed to fetch efficiently
        if self.use_cache:
            cell_info = self.get_cell_data_from_cache(rm.service_id_field_name, rm.grid_specification, rm.id_list)
            if cell_info is None:
                print('MESSAGE: Reverting to online GIS database and updating cache!')
                cell_info = self.get_cell_data_from_gis(rm.catchment_regulated_type, rm.service_id_field_name,
                                                        rm.grid_specification, rm.id_list,
                                                        calc_forest_frac=rm.calc_forest_frac)
                self.save_cell_data_to_cache(rm.service_id_field_name, rm.grid_specification, cell_info)
        else:
            cell_info = self.get_cell_data_from_gis(rm.catchment_regulated_type, rm.service_id_field_name,
                                                    rm.grid_specification, rm.id_list,
                                                    calc_forest_frac=rm.calc_forest_frac)
        catchment_id_map, cell_geo_data, polygons = [cell_info[k] for k in ['cid_map', 'geo_data', 'polygons']]
        cell_vector = self.build_cell_vector(rm.region_model_type, cell_geo_data)

        catchment_parameter_map = rm.region_model_type.parameter_t.map_t()
        for cid, param in rm.catchment_parameters.items():
            if cid in catchment_id_map:
                catchment_parameter_map[cid] = param
        region_model = rm.region_model_type(cell_vector, rm.region_parameters, catchment_parameter_map)
        region_model.bounding_region = rm.grid_specification  # mandatory for orchestration
        region_model.catchment_id_map = catchment_id_map.tolist()  # needed to map from externa c_id to 0-based c_id used internally in
        # region_model.gis_info = result  # opt:needed for internal statkraft use/presentation
        region_model.gis_info = polygons  # opt:needed for internal statkraft use/presentation

        def do_clone(x):
            clone = x.__class__(x)
            clone.bounding_region = x.bounding_region
            clone.catchment_id_map = catchment_id_map
            clone.gis_info = polygons
            return clone

        region_model.clone = do_clone

        return region_model


def get_grid_spec_from_catch_poly(catch_ids: List[int], catchment_type: str, identifier: str, epsg_id: int, dxy: int, pad: int,
                                  server_name: str = primary_server, server_name_preprod: str = secondary_server):
    catchment_fetcher = CatchmentFetcher(catchment_type, identifier, epsg_id,
                                         server_name=server_name, server_name_preprod=server_name_preprod)
    catch = catchment_fetcher.fetch(id_list=catch_ids)
    bbox = np.array(MultiPolygon(polygons=list(catch.values())).bounds)  # [xmin, ymin, xmax, ymax]
    box_ = bbox/dxy
    xll, yll, xur, yur = (np.array(
        [np.floor(box_[0]), np.floor(box_[1]), np.ceil(box_[2]), np.ceil(box_[3])]) + [-pad, -pad, pad, pad])*dxy
    return GridSpecification(epsg_id, int(xll), int(yll), dxy, dxy, int((xur - xll)/dxy), int((yur - yll)/dxy))
