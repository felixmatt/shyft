from __future__ import print_function
import unittest
from os import path

from shyft import shyftdata_dir
from shyft import api
from shyft.repository.camels.camels_repository import CamelsDataRepository

class AromeDataRepositoryTestCase(unittest.TestCase):

    def test_get_timeseries(self):
        """
        Simple test of camels data respository.
        """
        EPSG = 32632
        sgid = '01013500'

        # Period start
        n_days = 30
        t0 = api.YMDhms(2010, 1, 1, 12)
        utc = api.Calendar()  # No offset gives Utc
        period = api.UtcPeriod(utc.time(t0), utc.time(t0) + api.deltahours(n_days*24))

        date_str = "{}{:02}{:02}_{:02}".format(t0.year,t0.month, t0.day, t0.hour)


        base_dir = path.join(shyftdata_dir, "repository", "camels_repository")
        path_to_database = path.join(base_dir, "CAMELS")

        cdr = CamelsDataRepository(EPSG, path_to_database, sgid)

        cdr_input_ts_names1 = ("temperature", "wind_speed", "precipitation", "relative_humidity")
        cdr_input_ts_names2 = ("radiation",)

        sources1 = cdr.get_timeseries(cdr_input_ts_names1, period)
        sources2 = cdr.get_timeseries(cdr_input_ts_names2, period)

        self.assertTrue(len(sources1) > 0)
        self.assertTrue(len(sources2) > 0)
        self.assertTrue(set(sources1) == set(cdr_input_ts_names1))
        self.assertTrue(set(sources2) == set(cdr_input_ts_names2))
        self.assertTrue(sources1["temperature"][0].ts.size() == n_days + 1)

        temp0 = sources1["temperature"][0].ts
        p0 = sources1["precipitation"][0].ts
        r0 = sources2["radiation"][0].ts

        self.assertTrue(r0.size() == n_days + 1)
        self.assertTrue(p0.size() == n_days + 1)
        self.assertTrue(r0.time(0) == temp0.time(0))
        self.assertTrue(p0.time(0) == temp0.time(0))
        self.assertTrue(r0.time(r0.size() - 1) == temp0.time(temp0.size() - 1))
        self.assertTrue(p0.time(r0.size() - 1) == temp0.time(temp0.size() - 1))
        self.assertTrue(p0.time(0)==period.start)