import unittest
#from os import path

#from shyft import shyftdata_dir
#from shyft import api

from netcdftime import utime


class NetCdfTimeTestCase(unittest.TestCase):

    def test_extract_conversion_factors_from_string(self):
        u = utime('hours since 2000-01-01 00:00:00')
        self.assertIsNotNone(u)
