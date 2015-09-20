import unittest
from os import path
from shyft import shyftdata_dir
from shyft.repository.netcdf.geo_ts_repository import NetCDFGeoTsRepository 


class NetCDFSourceRepositoryTestCase(unittest.TestCase):
    """ Verify that we correctly can read 
        geo-located timeseries from a netCDF based file-store,
    """

    def test_construct_repository(self):
        met = path.join(shyftdata_dir, "netcdf", "orchestration-testdata","stations_met.nc")
        dis = path.join(shyftdata_dir, "netcdf","orchestration-testdata", "stations_discharge.nc")
        params={}
        netcdf_repository = NetCDFGeoTsRepository(params,met,dis)
        self.assertIsNotNone(netcdf_repository)