from __future__ import print_function
import unittest
from os import path

from shyft import shyftdata_dir
from shyft import api
from shyft.repository.netcdf.opendap_data_repository import GFSDataRepository

class GFSDataRepositoryTestCase(unittest.TestCase):

    def test_get_timeseries(self):
        """
        Simple regression test of OpenDAP data repository.
        """
        epsg, bbox = self.epsg_bbox

        dem_file = path.join(shyftdata_dir, "netcdf", "etopo180.nc")

        # Period start
        year = 2015
        month = 10
        day = 21
        hour = 7
        n_hours = 30
        utc = api.Calendar()  # No offset gives Utc
        t0 = api.YMDhms(year, month, day, hour)
        period = api.UtcPeriod(utc.time(t0), utc.time(t0) + api.deltahours(n_hours))

        repos = GFSDataRepository(epsg, dem_file, utc.time(t0), bounding_box=bbox)
        data_names = ("temperature", "wind_speed", "precipitation", "relative_humidity", "radiation")
        sources = repos.get_timeseries(data_names, period, None)
        self.assertEqual(set(data_names), set(sources.keys()))
        self.assertEqual(len(sources["temperature"]), 2)
        data1 = sources["temperature"][0]
        data2 = sources["temperature"][1]
        self.assertNotEqual(data1.mid_point().x, data2.mid_point().x)
        self.assertNotEqual(data1.mid_point().y, data2.mid_point().y)
        self.assertNotEqual(data1.mid_point().z, data2.mid_point().z)
        h_dt = (data1.ts.time(1) - data1.ts.time(0))/3600
        self.assertEqual(data1.ts.size(), 30//h_dt)

    @property
    def epsg_bbox(self):
        """ this should cut a slice out of test-data located in shyft-data repository/arome  """
        EPSG = 32632
        x0 = 436100.0 # lower left
        y0 = 6823000.0 #lower right
        nx = 74
        ny = 124
        dx = 1000.0
        dy = 1000.0
        return EPSG, ([x0, x0 + nx*dx, x0 + nx*dx, x0], [y0, y0, y0 + ny*dy, y0 + ny*dy])
