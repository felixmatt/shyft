import unittest
from os import path

from shyft import shyftdata_dir
from shyft import api
from shyft.repository.netcdf.wrf_data_repository import WRFDataRepository
from shyft.repository.netcdf.wrf_data_repository import WRFDataRepositoryError


class WRFDataRepositoryTestCase(unittest.TestCase):
    def test_get_timeseries(self):
        """
        Simple regression test of WRF data repository.
        """
        EPSG, bbox = self.wrf_epsg_bbox

        # Period start
        utc = api.Calendar()  # No offset gives Utc
        n_hours = 10
        t0_dt = api.YMDhms(2009, 10)
        t0 = utc.time(2009,10)
        date_str = "{}_{:02}".format(t0_dt.year, t0_dt.month)

        period = api.UtcPeriod(t0, t0 + api.deltahours(n_hours))

        base_dir = path.join(shyftdata_dir, "repository", "wrf_data_repository")
        f1 = "out_d02_{}.nc".format(date_str)

        wrf1 = WRFDataRepository(EPSG, base_dir, filename=f1, bounding_box=bbox, allow_subset=True)
        wrf1_data_names = ("temperature", "wind_speed", "precipitation", "relative_humidity", "radiation")
        sources = wrf1.get_timeseries(wrf1_data_names, period, None)
        self.assertTrue(len(sources) > 0)

        self.assertTrue(set(sources) == set(wrf1_data_names))
        self.assertTrue(sources["temperature"][0].ts.size() == n_hours + 1)
        r0 = sources["radiation"][0].ts
        p0 = sources["precipitation"][0].ts
        temp0 = sources["temperature"][0].ts
        self.assertTrue(r0.size() == n_hours + 1)
        self.assertTrue(p0.size() == n_hours + 1)
        self.assertTrue(r0.time(0) == temp0.time(0))
        self.assertTrue(p0.time(0) == temp0.time(0))
        self.assertTrue(r0.time(r0.size() - 1) == temp0.time(temp0.size() - 1))
        self.assertTrue(p0.time(r0.size() - 1) == temp0.time(temp0.size() - 1))
        self.assertTrue(p0.time(0), period.start)

    @property
    def wrf_epsg_bbox(self):
        """A slice of test-data located in shyft-data repository/wrf."""
        EPSG = 32643
        x0 = 674085.0  # lower left
        y0 = 3476204.0  # lower right
        nx = 102
        ny = 121
        dx = 1000.0
        dy = 1000.0
        return EPSG, ([x0, x0 + nx * dx, x0 + nx * dx, x0], [y0, y0, y0 + ny * dy, y0 + ny * dy])


if __name__ == "__main__":
    unittest.main()
