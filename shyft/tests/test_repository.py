import unittest
from os import path
from shyft import shyftdata_dir
from shyft.orchestration2.netcdf.region import NetCDFRegionRepository
from shyft.api.pt_gs_k import PTGSKModel



class NetCDFRegionRepositoryTestCase(unittest.TestCase):

    def test_construct_repository(self):
        rcf = path.join(path.dirname(__file__), "netcdf", "region.yaml")
        mcf = path.join(path.dirname(__file__), "netcdf", "model.yaml")
        df = path.join(shyftdata_dir, "netcdf", "orchestration-testdata", "cell_info.nc")
        repos = NetCDFRegionRepository(rcf, mcf, df)
        self.assertTrue(path.isfile(rcf))
        self.assertTrue(path.isfile(mcf))
        self.assertTrue(path.isfile(df))
        self.assertIsNotNone(repos.mask)
        model = repos.get_region(0, PTGSKModel)
