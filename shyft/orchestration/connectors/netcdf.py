import numpy, gdal, urllib2, os,osr
from gdalconst import GA_ReadOnly
from netCDF4 import Dataset
from subprocess import call
from osgeo import gdal_array, gdalconst

nci    = Dataset('C:/Python Code for NETCDF extraction/arome_norway_default2_5km_latest.nc')
input_raster = 'C:/tmp/catchments.tif'
variables = {'air_temperature_2m':'','precipitation_amount_middle_estimate':'','relative_humidity_2m':'','x_wind_10m':''}

projection = nci.variables['projection_6'].proj4
#projection.proj4
class raster_ts:
    def __init__(self,time,raster):
        self.utc_time = time
        self.raster= raster


 
    

    '''
n = numpy_dataset()
n.add_precipitation_amount_middle_estimate()
n.precipitation_amount_middle_estimate.append(raster_band('12-luglio',999))
'''
      
class nc:
    def __init__(self, dataset, region_raster,desired_variables):
        '''reads in a Dataset from netcdf4.dataset(input_netcdf)
        The corner's indexes and values need to be provided in order to further process the NetCDF file (f.ex clipping, reprojecting, aligning, resampling etc)
        If values are not available, run first the get_indexes() method which will calculate them automatically.
        '''
        self.dataset = dataset
        self.TopIndex       = None
        self.TopValue       = None
        self.BottomIndex    = None
        self.BottomValue    = None
        self.RightIndex     = None
        self.RightValue     = None
        self.LeftIndex      = None
        self.LeftValue      = None
        self.nc_projection     = 'PROJCS["Lambert_Conformal_Conic",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["standard_parallel_1",63],PARAMETER["standard_parallel_2",63],PARAMETER["latitude_of_origin",63],PARAMETER["central_meridian",15],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]]]'

        # Setting Region Properties
        self.RegionProperties = []
        region_raster = gdal.Open(region_raster)
        self.RegionProperties.append(region_raster.GetProjection())
        self.RegionProperties.append(region_raster.GetGeoTransform())
        self.RegionProperties.append(region_raster.RasterYSize)
        self.RegionProperties.append(region_raster.RasterXSize)

        self.variables = desired_variables        
      
          
    def get_indexes(self):
        '''
        Calculates NetCDF boundary values and indexes to match current Region domain.
        '''
        # getting MET.no's NetCDF spatial reference
        lambert                 = osr.SpatialReference()
        lambert.ImportFromWkt(self.nc_projection)

        # getting region's spatial reference and domain extent details
        src_prj                 = osr.SpatialReference()
        src_prj.ImportFromWkt(self.RegionProperties[0])
        geot                    = self.RegionProperties[1]
        x_size                  = self.RegionProperties[3]
        y_size                  = self.RegionProperties[2]

        # projecting region's corners into the lambert projection to get correspondent x,y values of the corners
        coordt                  = osr.CoordinateTransformation(src_prj,lambert)
        (ulx,uly,ulz)           = coordt.TransformPoint(geot[0],geot[3])
        (lrx,lry,lrz)           = coordt.TransformPoint(geot[0] + geot[1]*x_size , geot[3] + geot[5]*y_size)
        (urx,ury,urz)           = coordt.TransformPoint(geot[0] + geot[1]*x_size , geot[3])
        (llx,lfy,lfz)           = coordt.TransformPoint(geot[0], geot[3]+geot[5]*y_size)
        

        #importing x and y cartesian system arrays
        x = self.dataset.variables['x']
        y = self.dataset.variables['y']

        #getting indices of the corners of the subdomain for slicing
        for i in range(0,y.size):
            if y[i] < uly and y[i+1] > uly:
                self.TopIndex = i+1
                self.TopValue = y[i+1]
        for i in range(0,y.size):
            if y[i] < lry and y[i+1] > lry:
                self.BottomIndex = i
                self.BottomValue = y[i]
        for i in range(0,x.size):
            if x[i] < urx and x[i+1]> urx:
                self.RightIndex = i+1
                self.RightValue = x[i+1]
        for i in range(0,x.size):
            if x[i] < llx and x[i+1]>llx:
                self.LeftIndex = i
                self.LeftValue = x[i]


    def slice_netcdf_variables(self):
        '''
        slices the NetCDF dataset and updates it
        If corner's indexes are not known, firt run get_indexes method. If they are knows, manually insert them all (f.ex nc.BottomIndex = 1234) before running this method.
        '''
      
        for key in self.variables.keys():
            self.dataset.variables[key] = self.dataset.variables[key][:,nc.BottomIndex:nc.TopIndex,nc.LeftIndex:nc.RightIndex][:]


    def netcdf_to_arraydataset(self):

        time            = self.dataset.variables['time']

        #preparing input in Lambert geotransform and projection
        time_steps      = time.shape[0]   
        src_y_pixels    = self.dataset.variables[self.variables.keys()[0]].shape[1]  
        src_x_pixels    = self.dataset.variables[self.variables.keys()[0]].shape[2]  
        src_pixel_size  = (self.TopValue-self.BottomValue+1)/(src_y_pixels)
        src_x_min       = nc.LeftValue - src_pixel_size/2
        src_y_max       = nc.TopValue  - src_pixel_size/2

        # defining output geotransform and projection based on Region's ones
        dst_proj        = self.RegionProperties[0]
        dst_geotrans    = self.RegionProperties[1]
        dst_x_pixels    = self.RegionProperties[3]
        dst_y_pixels    = self.RegionProperties[2]



        # Initializing empty input
        mem_driver      = gdal.GetDriverByName('MEM')
        data            = mem_driver.Create('',src_x_pixels,src_y_pixels,1,gdal.GDT_Float32,)
        data.SetGeoTransform((src_x_min,src_pixel_size,0,src_y_max,0,-src_pixel_size))
        data.SetProjection(self.nc_projection)
        
        # Initializing empty output
        dst = mem_driver.Create('', dst_x_pixels, dst_y_pixels, 1, gdalconst.GDT_Float32)
        dst.SetGeoTransform( dst_geotrans )
        dst.SetProjection(dst_proj)




        #looping through variables (keys) and timesteps (i)
        raster_dataset = {}
        for key in variables.keys():
            raster_dataset[key] = []

            for i in range (0,time_steps):
                #Load in the in-memory input empty array the values of the i timestep of the current variable
                data.GetRasterBand(1).WriteArray(self.dataset.variables[key][i][::-1])
                data.FlushCache()
             
                # Reproject,resample,align
                gdal.ReprojectImage(data, dst, None, None, gdalconst.GRA_Bilinear)
                reprj_raster = dst.ReadAsArray()
                dst.FlushCache()

                # Store into raster_dataset dictionary
                raster_dataset[key].append(raster_ts(time[i],reprj_raster))
        
                
        return raster_dataset


nc = nc(nci,input_raster,variables)
nc.get_indexes()


nc.slice_netcdf_variables()
raster_dataset = nc.netcdf_to_arraydataset()













































'''
test = 1234

######################################## for my thesis #########################################

dst_proj        = nc.RegionProperties[0]
dst_geotrans    = nc.RegionProperties[1]
dst_x_pixels    = nc.RegionProperties[3]
dst_y_pixels    = nc.RegionProperties[2]
mem_driver      = gdal.GetDriverByName('GTiff')
dst = mem_driver.Create('C:/tmp/wind_test_hd.tif', dst_x_pixels, dst_y_pixels, 1, gdalconst.GDT_Float32)
dst.SetGeoTransform( dst_geotrans )
dst.SetProjection(dst_proj)


dst.GetRasterBand(1).WriteArray(raster_dataset['air_temperature_2m'][i].raster)
dst.FlushCache()

temperature2 = dst.ReadAsArray()
'''

for i in range(0,67):
    array           = raster_dataset['air_temperature_2m'][i].raster
    name            = str.format('C:/tmp/temperature/{:.0f}.txt',raster_dataset['air_temperature_2m'][i].utc_time)
    numpy.savetxt(name, array,fmt='%f',delimiter='\t',newline='\n',header='',footer='',comments='')   







