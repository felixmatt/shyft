import unittest
from os import path
from shyft import shyftdata_dir
from shyft.repository.netcdf.source_dataset import NetCDFSourceRepository 


class NetCDFSourceRepositoryTestCase(unittest.TestCase):
    """ Verify that we correctly can read 
        geo-located timeseries from a netCDF based file-store,
    """

    def test_construct_repository(self):
        rcf = path.join(path.dirname(__file__), "netcdf", "region.yaml")
        mcf = path.join(path.dirname(__file__), "netcdf", "model.yaml")
        df = path.join(shyftdata_dir, "netcdf", "orchestration-testdata", "cell_info.nc")
        netcdf_repository = NetCDFSourceRepository(rcf, mcf, df)
