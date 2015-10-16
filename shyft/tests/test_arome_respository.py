from __future__ import print_function
import unittest
from os import path

from shyft import shyftdata_dir
from shyft import api
from shyft.repository.netcdf.arome_data_repository import AromeDataRepository
from shyft.repository.netcdf.arome_data_repository import AromeDataRepositoryError

class AromeDataRepositoryTestCase(unittest.TestCase):

    def test_get_timeseries(self):
        """
        Simple regression test of arome data respository.
        """
        return
        EPSG = 32633
        upper_left_x = 436100.0
        upper_left_y = 7417800.0
        nx = 74
        ny = 94
        dx = 1000.0
        dy = 1000.0

        # Period start
        year = 2015
        month = 8
        day = 23
        hour = 6
        n_hours = 30
        date_str = "{}{:02}{:02}_{:02}".format(year, month, day, hour)
        utc = api.Calendar()  # No offset gives Utc
        t0 = api.YMDhms(year, month, day, hour)
        period = api.UtcPeriod(utc.time(t0), utc.time(t0) + api.deltahours(n_hours))

        base_dir = path.join(shyftdata_dir, "repository", "arome_data_repository")
        f1 = "arome_metcoop_red_default2_5km_{}.nc".format(date_str)
        f2 = "arome_metcoop_red_test2_5km_{}.nc".format(date_str)

        bbox = ([upper_left_x, upper_left_x + nx*dx, upper_left_x + nx*dx, upper_left_x],
                [upper_left_y, upper_left_y, upper_left_y - ny*dy, upper_left_y - ny*dy])
        ar1 = AromeDataRepository(EPSG, base_dir, filename=f1, bounding_box=bbox)
        ar2 = AromeDataRepository(EPSG, base_dir, filename=f2, elevation_file=f1)
        ar1_data_names = ("temperature", "wind_speed", "precipitation", "relative_humidity")
        ar2_data_names = ("radiation",)
        sources = ar1.get_timeseries(ar1_data_names, period, None)
        self.assertTrue(len(sources) > 0)
        sources2 = ar2.get_timeseries(ar2_data_names, period, geo_location_criteria=bbox)

        self.assertTrue(set(sources) == set(ar1_data_names))
        self.assertTrue(set(sources2) == set(ar2_data_names))
        self.assertTrue(sources["temperature"][0].ts.size() == n_hours + 1)
        r0 = sources2["radiation"][0].ts
        p0 = sources["precipitation"][0].ts
        temp0 = sources["temperature"][0].ts
        self.assertTrue(r0.size() == n_hours + 1)
        self.assertTrue(p0.size() == n_hours + 1)
        self.assertTrue(r0.time(0) == temp0.time(0))
        self.assertTrue(p0.time(0) == temp0.time(0))
        self.assertTrue(r0.time(r0.size() - 1) == temp0.time(temp0.size() - 1))
        self.assertTrue(p0.time(r0.size() - 1) == temp0.time(temp0.size() - 1))

    def test_get_forecast(self):
        EPSG = 32633
        upper_left_x = 436100.0
        upper_left_y = 7417800.0
        nx = 74
        ny = 94
        dx = 1000.0
        dy = 1000.0

        # Period start
        year = 2015
        month = 10
        day = 1
        hour = 0
        n_hours = 65
        utc = api.Calendar()  # No offset gives Utc
        t0 = api.YMDhms(year, month, day, hour)
        period = api.UtcPeriod(utc.time(t0), utc.time(t0) + api.deltahours(n_hours))
        t_c1 = utc.time(t0) + api.deltahours(1)
        t_c2 = utc.time(t0) + api.deltahours(7)

        base_dir = path.join(shyftdata_dir, "repository", "arome_data_repository")
        pattern = "arome_metcoop_default2_5km_*.nc"
        bbox = ([upper_left_x, upper_left_x + nx*dx,
                 upper_left_x + nx*dx, upper_left_x],
                [upper_left_y, upper_left_y,
                 upper_left_y - ny*dy, upper_left_y - ny*dy])
        repos = AromeDataRepository(EPSG, base_dir, filename=pattern, bounding_box=bbox)
        data_names = ("temperature", "wind_speed", "precipitation", "relative_humidity")
        tc1_sources = repos.get_forecast(data_names, period, t_c1, None)
        tc2_sources = repos.get_forecast(data_names, period, t_c2, None)

        self.assertTrue(len(tc1_sources) == len(tc2_sources))
        self.assertTrue(set(tc1_sources) == set(data_names))
        self.assertTrue(tc1_sources["temperature"][0].ts.size() == n_hours + 1)

        tc1_precip = tc1_sources["precipitation"][0].ts
        tc2_precip = tc2_sources["precipitation"][0].ts

        self.assertTrue(tc1_precip.size() == n_hours + 1)
        self.assertTrue(tc1_precip.time(0) != tc2_precip.time(0))

    def test_get_ensemble(self):
        EPSG = 32633
        upper_left_x = 436100.0
        upper_left_y = 7417800.0
        nx = 74
        ny = 94
        dx = 1000.0
        dy = 1000.0
        # Period start
        year = 2015
        month = 7
        day = 26
        hour = 0
        n_hours = 30
        utc = api.Calendar()  # No offset gives Utc
        t0 = api.YMDhms(year, month, day, hour)
        period = api.UtcPeriod(utc.time(t0), utc.time(t0) + api.deltahours(n_hours))
        t_c = utc.time(t0) + api.deltahours(1)

        base_dir = path.join(shyftdata_dir, "netcdf", "arome")
        pattern = "fc*.nc"
        bbox = ([upper_left_x, upper_left_x + nx*dx,
                 upper_left_x + nx*dx, upper_left_x],
                [upper_left_y, upper_left_y,
                 upper_left_y - ny*dy, upper_left_y - ny*dy])
        try:
            repos = AromeDataRepository(EPSG, base_dir, filename=pattern, bounding_box=bbox)
            data_names = ("temperature", "wind_speed", "relative_humidity")
            ensemble = repos.get_forecast_ensemble(data_names, period, t_c, None)
            self.assertTrue(isinstance(ensemble, list))
            self.assertEqual(len(ensemble), 10)
        except AromeDataRepositoryError as adre:
            self.skipTest( "(test inconclusive- missing arome-data {0})".format(adre))


if __name__ == "__main__":
    unittest.main()
