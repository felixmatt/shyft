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
        EPSG, bbox = self.arome_epsg_bbox

        # Period start
        n_hours = 30
        t0 = api.YMDhms(2015, 8, 24, 0)
        date_str = "{}{:02}{:02}_{:02}".format(t0.year,t0.month, t0.day, t0.hour)
        utc = api.Calendar()  # No offset gives Utc
        period = api.UtcPeriod(utc.time(t0), utc.time(t0) + api.deltahours(n_hours))

        base_dir = path.join(shyftdata_dir, "repository", "arome_data_repository")
        f1 = "arome_metcoop_red_default2_5km_{}_diff_time_unit.nc".format(date_str)
        f2 = "arome_metcoop_red_test2_5km_{}.nc".format(date_str)

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
        self.assertTrue(p0.time(0),period.start)

    @property
    def arome_epsg_bbox(self):
        """A slice of test-data located in shyft-data repository/arome."""
        EPSG = 32632
        x0 = 436100.0   # lower left
        y0 = 6823000.0  # lower right
        nx = 74
        ny = 24
        dx = 1000.0
        dy = 1000.0
        return EPSG, ([x0, x0 + nx*dx, x0 + nx*dx, x0], [y0, y0, y0 + ny*dy, y0 + ny*dy])

    def test_get_forecast(self):
        # Period start
        year = 2015
        month = 8
        day = 24
        hour = 6
        n_hours = 65
        utc = api.Calendar()  # No offset gives Utc
        t0 = api.YMDhms(year, month, day, hour)
        period = api.UtcPeriod(utc.time(t0), utc.time(t0) + api.deltahours(n_hours))
        t_c1 = utc.time(t0) + api.deltahours(1)
        t_c2 = utc.time(t0) + api.deltahours(7)

        base_dir = path.join(shyftdata_dir, "repository", "arome_data_repository")
        pattern = "arome_metcoop*default2_5km_*.nc"
        EPSG, bbox = self.arome_epsg_bbox

        repos = AromeDataRepository(EPSG, base_dir, filename=pattern, bounding_box=bbox)
        data_names = ("temperature", "wind_speed", "precipitation", "relative_humidity")
        tc1_sources = repos.get_forecast(data_names, period, t_c1, None)
        tc2_sources = repos.get_forecast(data_names, period, t_c2, None)

        self.assertTrue(len(tc1_sources) == len(tc2_sources))
        self.assertTrue(set(tc1_sources) == set(data_names))
        self.assertTrue(tc1_sources["temperature"][0].ts.size() == n_hours + 1)

        tc1_precip = tc1_sources["precipitation"][0].ts
        tc2_precip = tc2_sources["precipitation"][0].ts

        self.assertEqual(tc1_precip.size(), n_hours)
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
            self.skipTest("(test inconclusive- missing arome-data {0})".format(adre))

    def test_wrong_directory(self):
        with self.assertRaises(AromeDataRepositoryError) as context:
            AromeDataRepository(32632, "Foobar", filename="")
        self.assertEqual("No such directory 'Foobar'", context.exception.args[0])

    def test_wrong_elevation_file(self):
        with self.assertRaises(AromeDataRepositoryError) as context:
            AromeDataRepository(32632, shyftdata_dir, filename="", elevation_file="plain_wrong.nc")
        self.assertTrue(all(x in context.exception.args[0] for x in ["Elevation file",
                                                                     "not found"]))

    def test_wrong_file(self):
        with self.assertRaises(AromeDataRepositoryError) as context:
            utc = api.Calendar()  # No offset gives Utc
            t0 = api.YMDhms(2015, 12, 25, 18)
            period = api.UtcPeriod(utc.time(t0), utc.time(t0) + api.deltahours(30))
            ar1 = AromeDataRepository(32632, shyftdata_dir, filename="plain_wrong.nc")
            ar1.get_timeseries(("temperature",), period, None)
        self.assertTrue(all(x in context.exception.args[0] for x in ["File", "not found"]))

    def test_wrong_forecast(self):
        with self.assertRaises(AromeDataRepositoryError) as context:
            utc = api.Calendar()  # No offset gives Utc
            t0 = api.YMDhms(2015, 12, 25, 18)
            period = api.UtcPeriod(utc.time(t0), utc.time(t0) + api.deltahours(30))
            ar1 = AromeDataRepository(32632, shyftdata_dir, filename="plain_wrong_*.nc")
            ar1.get_forecast(("temperature",), period, utc.time(t0), None)
        self.assertTrue(all(x in context.exception.args[0] for x in
                            ["No matches found for file_pattern = ", "and t_c = "]))

    def test_non_overlapping_x_bbox(self):
        EPSG, bbox = self.arome_epsg_bbox
        bbox = list(bbox)
        bbox[0] = [-100000.0, -90000.0, -90000.0, -100000]
        # Period start
        year = 2015
        month = 8
        day = 24
        hour = 6
        n_hours = 30
        date_str = "{}{:02}{:02}_{:02}".format(year, month, day, hour)
        utc = api.Calendar()  # No offset gives Utc
        t0 = api.YMDhms(year, month, day, hour)
        period = api.UtcPeriod(utc.time(t0), utc.time(t0) + api.deltahours(n_hours))

        base_dir = path.join(shyftdata_dir, "repository", "arome_data_repository")
        filename = "arome_metcoop_red_default2_5km_{}.nc".format(date_str)
        reader = AromeDataRepository(EPSG, base_dir, filename=filename, bounding_box=bbox)
        data_names = ("temperature", "wind_speed", "precipitation", "relative_humidity")
        with self.assertRaises(AromeDataRepositoryError) as context:
            reader.get_timeseries(data_names, period, None)
        self.assertEqual("Bounding box longitudes don't intersect with dataset.",
                         context.exception.args[0])

    def test_non_overlapping_y_bbox(self):
        EPSG, bbox = self.arome_epsg_bbox
        bbox = list(bbox)
        bbox[1] = [6010000.0, 6010000.0, 6035000.0, 6035000]
        # Period start
        year = 2015
        month = 8
        day = 24
        hour = 6
        n_hours = 30
        date_str = "{}{:02}{:02}_{:02}".format(year, month, day, hour)
        utc = api.Calendar()  # No offset gives Utc
        t0 = api.YMDhms(year, month, day, hour)
        period = api.UtcPeriod(utc.time(t0), utc.time(t0) + api.deltahours(n_hours))

        base_dir = path.join(shyftdata_dir, "repository", "arome_data_repository")
        filename = "arome_metcoop_red_default2_5km_{}.nc".format(date_str)
        reader = AromeDataRepository(EPSG, base_dir, filename=filename, bounding_box=bbox)
        data_names = ("temperature", "wind_speed", "precipitation", "relative_humidity")
        with self.assertRaises(AromeDataRepositoryError) as context:
            reader.get_timeseries(data_names, period, None)
        self.assertEqual("Bounding box latitudes don't intersect with dataset.",
                         context.exception.args[0])

    def test_missing_bbox(self):
        EPSG, _ = self.arome_epsg_bbox
        # Period start
        year = 2015
        month = 8
        day = 24
        hour = 6
        n_hours = 30
        date_str = "{}{:02}{:02}_{:02}".format(year, month, day, hour)
        utc = api.Calendar()  # No offset gives Utc
        t0 = api.YMDhms(year, month, day, hour)
        period = api.UtcPeriod(utc.time(t0), utc.time(t0) + api.deltahours(n_hours))

        base_dir = path.join(shyftdata_dir, "repository", "arome_data_repository")
        filename = "arome_metcoop_red_default2_5km_{}.nc".format(date_str)
        reader = AromeDataRepository(EPSG, base_dir, filename=filename)
        data_names = ("temperature", "wind_speed", "precipitation", "relative_humidity")
        with self.assertRaises(AromeDataRepositoryError) as context:
            reader.get_timeseries(data_names, period, None)
        self.assertEqual("A bounding box must be provided.", context.exception.args[0])

    def test_tiny_bbox(self):
        EPSG, _ = self.arome_epsg_bbox

        x0 = 436250.0   # lower left
        y0 = 6823250.0  # lower right
        nx = 1
        ny = 1
        dx = 5.0
        dy = 5.0
        bbox = ([x0, x0 + nx*dx, x0 + nx*dx, x0], [y0, y0, y0 + ny*dy, y0 + ny*dy])
        print(bbox)
        
        # Period start
        year = 2015
        month = 8
        day = 24
        hour = 6
        n_hours = 30
        date_str = "{}{:02}{:02}_{:02}".format(year, month, day, hour)
        utc = api.Calendar()  # No offset gives Utc
        t0 = api.YMDhms(year, month, day, hour)
        period = api.UtcPeriod(utc.time(t0), utc.time(t0) + api.deltahours(n_hours))

        base_dir = path.join(shyftdata_dir, "repository", "arome_data_repository")
        filename = "arome_metcoop_red_default2_5km_{}.nc".format(date_str)
        reader = AromeDataRepository(EPSG, base_dir, filename=filename, 
                                     bounding_box=bbox, x_padding=0, y_padding=0)
        data_names = ("temperature", "wind_speed", "precipitation", "relative_humidity")
        try:
            tss = reader.get_timeseries(data_names, period, None)
        except AromeDataRepository as err:
            self.fail("reader.get_timeseries raised AromeDataRepositoryError('{}') "
                      "unexpectedly.".format(err.args[0]))

    def test_subsets(self):
        EPSG, bbox = self.arome_epsg_bbox
        # Period start
        year = 2015
        month = 8
        day = 24
        hour = 6
        n_hours = 30
        date_str = "{}{:02}{:02}_{:02}".format(year, month, day, hour)
        utc = api.Calendar()  # No offset gives Utc
        t0 = api.YMDhms(year, month, day, hour)
        period = api.UtcPeriod(utc.time(t0), utc.time(t0) + api.deltahours(n_hours))

        base_dir = path.join(shyftdata_dir, "repository", "arome_data_repository")
        filename = "arome_metcoop_red_default2_5km_{}.nc".format(date_str)

        data_names = ("temperature", "wind_speed", "precipitation", "relative_humidity", "radiation")
        allow_subset = False
        reader = AromeDataRepository(EPSG, base_dir, filename=filename,
                                     bounding_box=bbox, allow_subset=allow_subset)
        with self.assertRaises(AromeDataRepositoryError) as context:
            reader.get_timeseries(data_names, period, None)
        self.assertEqual("Could not find all data fields", context.exception.args[0])
        allow_subset = True
        reader = AromeDataRepository(EPSG, base_dir, filename=filename,
                                     bounding_box=bbox, allow_subset=allow_subset)
        try:
            sources = reader.get_timeseries(data_names, period, None)
        except AromeDataRepositoryError as e:
            self.fail("AromeDataRepository.get_timeseries(data_names, period, None) "
                      "raised AromeDataRepositoryError unexpectedly.")
        self.assertEqual(len(sources), len(data_names) - 1)

if __name__ == "__main__":
    unittest.main()
