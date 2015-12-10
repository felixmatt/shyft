import unittest
from os import path
import numpy as np
from netCDF4 import Dataset
from shyft import shyftdata_dir
from shyft.repository.netcdf.region_model_repository import RegionModelRepository, RegionConfigError
from shyft.repository.netcdf.region_model_repository import BoundingBoxRegion
from shyft.repository.netcdf import yaml_config
from shyft.api.pt_gs_k import PTGSKModel


class NetCDFRegionModelRepositoryTestCase(unittest.TestCase):
    """ Verify that yaml-based config, with netcdf data
        can provide ready made shyft models extracted from
        configuration at region-model level, (parameters)
        cell properties (geo_cell_data, mid_point, elevation etc..)
    """
    def test_construct_repository(self):
        reg_conf = yaml_config.RegionConfig(path.join(path.dirname(__file__),
                                            "netcdf", "atnsjoen_region.yaml"))
        mod_conf = yaml_config.ModelConfig(path.join(path.dirname(__file__),
                                           "netcdf", "model.yaml"))
        epsg = "32633"
        region_model_repository = RegionModelRepository(reg_conf, mod_conf,PTGSKModel, epsg)
        self.assertIsNotNone(region_model_repository.mask)
        region_model = \
            region_model_repository.get_region_model("NeaNidelv_PTGSK",
                                                     PTGSKModel)
        self.assertIsNotNone(region_model)
        self.assertEqual(1848, region_model.size())
        rp = region_model.get_region_parameter()
        self.assertAlmostEqual(-2.63036759414, rp.kirchner.c1)
        self.assertTrue(
            region_model.has_catchment_parameter(1),
            "There is a catchment override in the region.yaml file")
        c1p = region_model.get_catchment_parameter(1)
        self.assertAlmostEqual(-2.539, c1p.kirchner.c1)

    def test_bounding_box_region(self):
        reg_conf = yaml_config.RegionConfig(path.join(path.dirname(__file__),
                                            "netcdf", "atnsjoen_region.yaml"))
        dsf = path.join(shyftdata_dir, reg_conf.repository()["data_file"])
        tmp = 10
        with Dataset(dsf) as ds:
            xcoords = ds.groups["elevation"].variables["xcoord"][:]
            ycoords = ds.groups["elevation"].variables["ycoord"][:]
            epsg = ds.groups["elevation"].epsg
        bbr = BoundingBoxRegion(xcoords, ycoords, epsg, 32632)
        bbox = bbr.bounding_box(32632)
        self.assertTrue(np.linalg.norm(bbr.x - bbox[0]) < 1.0e-14)
        self.assertTrue(np.linalg.norm(bbr.y - bbox[1]) < 1.0e-14)
        bbox = bbr.bounding_box(32633)
        self.assertFalse(np.linalg.norm(bbr.x - bbox[0]) < 500000.0)
        self.assertFalse(np.linalg.norm(bbr.y - bbox[1]) < 5000.0)

    def test_bad_region_yaml(self):
        reg_conf = yaml_config.RegionConfig(path.join(path.dirname(__file__),
                                            "netcdf", "atnsjoen_region_bad.yaml"))
        mod_conf = yaml_config.ModelConfig(path.join(path.dirname(__file__),
                                           "netcdf", "model.yaml"))
        epsg = "32633"
        region_model_repository = RegionModelRepository(reg_conf, mod_conf, PTGSKModel, epsg)
        with self.assertRaises(RegionConfigError) as ctx:
            region_model_repository.get_region_model("NeaNidelv_PTGSK",
                                                     PTGSKModel)


if __name__ == '__main__':
    unittest.main()
