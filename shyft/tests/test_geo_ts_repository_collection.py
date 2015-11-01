"""
Tests GeoTsRepositoryCollection
"""
from __future__ import print_function
from __future__ import absolute_import

from os import path
#import random
import unittest
#import numpy as np

from shyft import api
from shyft import shyftdata_dir
from shyft.repository.geo_ts_repository_collection import GeoTsRepositoryCollection
from shyft.repository.geo_ts_repository_collection import GeoTsRepositoryCollectionError
from shyft.repository.netcdf import AromeDataRepository
from shyft.repository.netcdf import AromeDataRepositoryError

class GeoTsRepositoryCollectionTestCase(unittest.TestCase):

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

    def test_get_timeseries_collection(self):
        year, month, day, hour = 2015, 8, 23, 6
        n_hours = 30
        dt = api.deltahours(1)
        utc = api.Calendar()  # No offset gives Utc
        t0 = utc.time(api.YMDhms(year, month, day, hour))
        period = api.UtcPeriod(t0, t0 + api.deltahours(n_hours))
        date_str = "{}{:02}{:02}_{:02}".format(year, month, day, hour)

        epsg, bbox = self.arome_epsg_bbox

        base_dir = path.join(shyftdata_dir, "repository", "arome_data_repository")
        f1 = "arome_metcoop_red_default2_5km_{}.nc".format(date_str)
        f2 = "arome_metcoop_red_test2_5km_{}.nc".format(date_str)

        ar1 = AromeDataRepository(epsg, base_dir, filename=f1, allow_subset=True)
        ar2 = AromeDataRepository(epsg, base_dir, filename=f2, elevation_file=f1, allow_subset=True)

        geo_ts_repository = GeoTsRepositoryCollection([ar1, ar2])
        sources = geo_ts_repository.get_timeseries(("temperature", "radiation"),
                                                   period, geo_location_criteria=bbox)

        with self.assertRaises(GeoTsRepositoryCollectionError) as context:
            GeoTsRepositoryCollection([ar1, ar2], reduce_type="foo")

        geo_ts_repository = GeoTsRepositoryCollection([ar1, ar2], reduce_type="add")
        with self.assertRaises(GeoTsRepositoryCollectionError) as context:
            sources = geo_ts_repository.get_timeseries(("temperature", "radiation"),
                                                       period, geo_location_criteria=bbox)

    def test_get_forecast_collection(self):
        year, month, day, hour = 2015, 8, 23, 6
        n_hours = 30
        dt = api.deltahours(1)
        utc = api.Calendar()  # No offset gives Utc
        t0 = utc.time(api.YMDhms(year, month, day, hour))
        period = api.UtcPeriod(t0, t0 + api.deltahours(n_hours))
        date_str = "{}{:02}{:02}_{:02}".format(year, month, day, hour)

        epsg, bbox = self.arome_epsg_bbox

        base_dir = path.join(shyftdata_dir, "repository", "arome_data_repository")
        f1 = "arome_metcoop_red_default2_5km_{}.nc".format(date_str)
        f2 = "arome_metcoop_red_test2_5km_{}.nc".format(date_str)

        ar1 = AromeDataRepository(epsg, base_dir, filename=f1, allow_subset=True)
        ar2 = AromeDataRepository(epsg, base_dir, filename=f2, elevation_file=f1, allow_subset=True)

        geo_ts_repository = GeoTsRepositoryCollection([ar1, ar2])
        source_names = ("temperature", "radiation")
        sources = geo_ts_repository.get_forecast(source_names, period, t0,
                                                 geo_location_criteria=bbox)
        self.assertTrue(all([x in source_names for x in sources]))

        geo_ts_repository = GeoTsRepositoryCollection([ar1, ar2], reduce_type="add")
        with self.assertRaises(GeoTsRepositoryCollectionError) as context:
            sources = geo_ts_repository.get_forecast(("temperature", "radiation"),
                                                     period, t0, geo_location_criteria=bbox)

    def test_get_ensemble_forecast_collection(self):
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
        pattern = "fc2015072600.nc"
        bbox = ([upper_left_x, upper_left_x + nx*dx,
                 upper_left_x + nx*dx, upper_left_x],
                [upper_left_y, upper_left_y,
                 upper_left_y - ny*dy, upper_left_y - ny*dy])
        try:
            ar1 = AromeDataRepository(EPSG, base_dir, filename=pattern, bounding_box=bbox)
            ar2 = AromeDataRepository(EPSG, base_dir, filename=pattern, bounding_box=bbox)
            repos = GeoTsRepositoryCollection([ar1, ar2])
            data_names = ("temperature", "wind_speed", "relative_humidity")
            ensemble = repos.get_forecast_ensemble(data_names, period, t_c, None)
            self.assertTrue(isinstance(ensemble, list))
            self.assertEqual(len(ensemble), 10)
            with self.assertRaises(GeoTsRepositoryCollectionError) as context:
                repos = GeoTsRepositoryCollection([ar1, ar2], reduce_type="add")
                repos.get_forecast_ensemble(data_names, period, t_c, None)
            self.assertEqual("Only replace is supported yet", context.exception.args[0])
        except AromeDataRepositoryError as adre:
            self.skipTest("(test inconclusive- missing arome-data {0})".format(adre))

if __name__ == '__main__':
    unittest.main()
