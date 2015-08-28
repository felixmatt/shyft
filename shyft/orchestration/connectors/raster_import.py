from osgeo import gdal,osr
import numpy as np
import warnings
"""
Connector for loading rasters into shyft, returns numpy-array of raster values
"""

def get_raster_array(rasterpath,domain):
	"""
	Uses GDAL to open the given raster. Cross-checks it's properties with
	regions given domain, raises errors or warnings if they do not match.
	Supports all rasters supported by GDAL (http://www.gdal.org/formats_list.html)

	Arguments:
		:rasterpath: path the raster to be read
		:domain: the domain dictionary for the region
	Returns:
		raster values as numpy
	"""
	# Open raster
	ds = gdal.Open(rasterpath)
	prj = ds.GetProjection()
	srs = osr.SpatialReference(wkt=prj)
	coord = ds.GetGeoTransform()

	# print srs.GetAttrValue("AUTHORITY", 1)
	# if domain['EPSG'] != srs.GetAttrValue("AUTHORITY", 1):
	# 	raise IOError("Wrong coordinate system for file {0} compared to domain settings in shyft config".format(rasterpath))
	if domain['nx'] != ds.RasterXSize:
		raise ("Wrong raster X size for file {0} compared to domain settings in shyft config".format(rasterpath))
	if domain['ny'] != ds.RasterYSize:
		raise ("Wrong raster Y size for file {0} compared to domain settings in shyft config".format(rasterpath))
	if domain['upperleft_x'] != coord[0]:
		warnings.warn("Wrong upper left x coordinate for file {0} compared to domain settings in shyft config; {1} != {2}".format(rasterpath, domain['upperleft_x'], coord[0]))
	if domain['upperleft_y'] != coord[3]:
		warnings.warn("Wrong upper left y coordinate for file {0} compared to domain settings in shyft config; {1} != {2}".format(rasterpath, domain['upperleft_y'], coord[3]))
	if domain['step_x'] != abs(coord[1]):
		warnings.warn("Wrong x step for file {0} compared to domain settings in shyft config".format(rasterpath))
	if domain['step_y'] != abs(coord[5]):
		warnings.warn("Wrong y step for file {0} compared to domain settings in shyft config".format(rasterpath))
	return np.array(ds.GetRasterBand(1).ReadAsArray())

# domain = dict()
# domain['upperleft_x'] = 562520.1717619
# domain['upperleft_y'] = 7041405.9048513
# domain['nx'] = 122
# domain['ny'] = 91
# domain['step_x'] = 1000.0
# domain['step_y'] = 1000.0
# domain['EPSG'] = 32632

# raster = r"C:\Projects\shyft_rasterimport\NeaNidelva\Catchments.rst"
# get_raster_array(raster,domain)

