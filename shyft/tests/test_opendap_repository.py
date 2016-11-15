import unittest
from os import path

from shyft import shyftdata_dir
from shyft import api
from shyft.repository.netcdf.opendap_data_repository import GFSDataRepository


class GFSDataRepositoryTestCase(unittest.TestCase):
    @property
    def start_date(self):
        utc = api.Calendar()
        today = utc.trim(api.utctime_now(), api.Calendar.DAY)
        return today - api.Calendar.DAY  # yesterday

    def test_get_timeseries(self):
        """
        Simple regression test of OpenDAP data repository.
        """
        epsg, bbox = self.epsg_bbox
        dem_file = path.join(shyftdata_dir, "netcdf", "etopo180.nc")
        n_hours = 30
        t0 = self.start_date + api.deltahours(7)
        period = api.UtcPeriod(t0, t0 + api.deltahours(n_hours))

        repos = GFSDataRepository(epsg, dem_file, t0, bounding_box=bbox)
        data_names = ("temperature", "wind_speed", "precipitation", "relative_humidity", "radiation")
        sources = repos.get_timeseries(data_names, period, None)
        self.assertEqual(set(data_names), set(sources.keys()))
        self.assertEqual(len(sources["temperature"]), 6)
        data1 = sources["temperature"][0]
        data2 = sources["temperature"][1]
        self.assertNotEqual(data1.mid_point().x, data2.mid_point().x)
        self.assertNotEqual(data1.mid_point().y, data2.mid_point().y)
        self.assertNotEqual(data1.mid_point().z, data2.mid_point().z)
        self.assertLessEqual(data1.ts.time(0), period.start, 'expect returned fc ts to cover requested period')
        self.assertGreaterEqual(data1.ts.total_period().end, period.end, 'expect returned fc ts to cover requested period')

    def test_get_forecast(self):
        """
        Simple forecast regression test of OpenDAP data repository.
        """
        epsg, bbox = self.epsg_bbox

        dem_file = path.join(shyftdata_dir, "netcdf", "etopo180.nc")
        n_hours = 30
        t0 = self.start_date + api.deltahours(9)
        period = api.UtcPeriod(t0, t0 + api.deltahours(n_hours))
        t_c = self.start_date + api.deltahours(7)  # the beginning of the forecast criteria

        repos = GFSDataRepository(epsg, dem_file, bounding_box=bbox)
        data_names = ("temperature",)  # the full set: "wind_speed", "precipitation", "relative_humidity", "radiation")
        sources = repos.get_forecast(data_names, period, t_c, None)
        self.assertEqual(set(data_names), set(sources.keys()))
        self.assertEqual(len(sources["temperature"]), 6)
        data1 = sources["temperature"][0]
        data2 = sources["temperature"][1]
        self.assertNotEqual(data1.mid_point().x, data2.mid_point().x)
        self.assertNotEqual(data1.mid_point().y, data2.mid_point().y)
        self.assertNotEqual(data1.mid_point().z, data2.mid_point().z)
        self.assertLessEqual(data1.ts.time(0), period.start, 'expect returned fc ts to cover requested period')
        self.assertGreaterEqual(data1.ts.total_period().end, period.end, 'expect returned fc ts to cover requested period')

    def test_get_ensemble(self):
        """
        Simple ensemble regression test of OpenDAP data repository.
        """
        epsg, bbox = self.epsg_bbox
        dem_file = path.join(shyftdata_dir, "netcdf", "etopo180.nc")
        n_hours = 30
        t0 = self.start_date + api.deltahours(9)  # api.YMDhms(year, month, day, hour)
        period = api.UtcPeriod(t0, t0 + api.deltahours(n_hours))
        t_c = t0

        repos = GFSDataRepository(epsg, dem_file, bounding_box=bbox)
        data_names = ("temperature",)  # this is the full set: "wind_speed", "precipitation", "relative_humidity", "radiation")
        ensembles = repos.get_forecast_ensemble(data_names, period, t_c, None)
        for sources in ensembles:
            self.assertEqual(set(data_names), set(sources.keys()))
            self.assertEqual(len(sources["temperature"]), 6)
            data1 = sources["temperature"][0]
            data2 = sources["temperature"][1]
            self.assertNotEqual(data1.mid_point().x, data2.mid_point().x)
            self.assertNotEqual(data1.mid_point().y, data2.mid_point().y)
            self.assertNotEqual(data1.mid_point().z, data2.mid_point().z)
            self.assertLessEqual(data1.ts.time(0), period.start, 'expect returned fc ts to cover requested period')
            self.assertGreaterEqual(data1.ts.total_period().end, period.end, 'expect returned fc ts to cover requested period')

    @property
    def epsg_bbox(self):
        """ this should cut a slice out of test-data located in shyft-data repository/arome  """
        EPSG = 32632
        x0 = 436100.0  # Lower left
        y0 = 6823000.0  # Lower right
        nx = 74
        ny = 124
        dx = 1000.0
        dy = 1000.0
        return EPSG, ([x0, x0 + nx * dx, x0 + nx * dx, x0], [y0, y0, y0 + ny * dy, y0 + ny * dy])


if __name__ == '__main__':
    unittest.main()
