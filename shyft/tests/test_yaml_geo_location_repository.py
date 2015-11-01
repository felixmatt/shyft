from __future__ import absolute_import
import unittest
from os import path
from os import remove
import glob
from shyft.repository.service.yaml_geo_location_repository import YamlGeoLocationRepository,YamlGeoLocationError
import yaml


class TestYamlGeoLocationRepository(unittest.TestCase):

    @property
    def _test_state_directory(self):
        return path.join(path.dirname(__file__), "state_tmp")
            
    def _clean_test_directory(self):
        files=glob.glob(path.join(self._test_state_directory,"*.yml"))
        for f in files:
            remove(f)
            
    def setUp(self):
        self._clean_test_directory()

    def tearDown(self):
        pass

    def test_missing_file_throws_IOError(self):
        s=YamlGeoLocationRepository(self._test_state_directory)
        self.assertRaises(IOError,lambda : s.get_locations([123]) )

    def test_missing_and_existing_locations(self):
        s=YamlGeoLocationRepository(self._test_state_directory)
        geol={123:(1,2,3),456:(4,5,6)}
        with open(path.join(self._test_state_directory,"pt_locations-epsg_32632.yml"),"w") as f:
            yml=yaml.dump(geol)
            f.write(yml)

        self.assertRaises(YamlGeoLocationError,lambda : s.get_locations([0]))
        geop=s.get_locations([123,456])
        self.assertIsNotNone(geop)
        self.assertAlmostEqual(geop[123][0],1)
        self.assertAlmostEqual(geop[123][1],2)
        self.assertAlmostEqual(geop[123][2],3)

if __name__ == '__main__':
    unittest.main()