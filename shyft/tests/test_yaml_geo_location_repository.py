from __future__ import absolute_import
import unittest
import os
import glob
from shyft.repository.service.yaml_geo_location_repository import YamlGeoLocationRepository, YamlGeoLocationError
import yaml


class TestYamlGeoLocationRepository(unittest.TestCase):
    @property
    def _test_state_directory(self):
        return os.path.join(os.path.dirname(__file__), "yaml_state_tmp")

    def _clean_test_directory(self):
        files = glob.glob(os.path.join(self._test_state_directory, "*.yml"))
        for f in files:
            os.remove(f)
        if os.path.exists(self._test_state_directory):
            os.rmdir(self._test_state_directory)

    def setUp(self):
        if not os.path.exists(self._test_state_directory):
            os.makedirs(self._test_state_directory)
        else:
            self._clean_test_directory()

    def tearDown(self):
        self._clean_test_directory()

    def test_missing_file_throws_IOError(self):
        s = YamlGeoLocationRepository(self._test_state_directory+"123")
        self.assertRaises(IOError, lambda: s.get_locations([123]))

    def test_missing_and_existing_locations(self):
        s = YamlGeoLocationRepository(self._test_state_directory)
        geol = {123: (1, 2, 3), 456: (4, 5, 6)}
        with open(os.path.join(self._test_state_directory, 'pt_locations-epsg_32632.yml'), 'w') as f:
            yml = yaml.dump(geol)
            f.write(yml)

        self.assertRaises(YamlGeoLocationError, lambda: s.get_locations([0]))
        geop = s.get_locations([123, 456])
        self.assertIsNotNone(geop)
        self.assertAlmostEqual(geop[123][0], 1)
        self.assertAlmostEqual(geop[123][1], 2)
        self.assertAlmostEqual(geop[123][2], 3)


if __name__ == '__main__':
    unittest.main()
