import unittest
from shyft import api
from shyft.repository.netcdf.time_conversion import convert_netcdf_time
from netcdftime import utime
import numpy as np


class NetCdfTimeTestCase(unittest.TestCase):
    def test_extract_conversion_factors_from_string(self):
        u = utime('hours since 1970-01-01 00:00:00')
        t_origin = api.Calendar(u.tzoffset).time(
            api.YMDhms(u.origin.year, u.origin.month, u.origin.day, u.origin.hour, u.origin.minute, u.origin.second))
        delta_t_dic = {'days': api.deltahours(24), 'hours': api.deltahours(1), 'minutes': api.deltaminutes(1)}
        delta_t = delta_t_dic[u.units]
        self.assertIsNotNone(u)
        self.assertEqual(delta_t, api.deltahours(1))
        self.assertEqual(t_origin, 0)

    def test_unit_conversion(self):
        utc = api.Calendar()
        t_num = np.arange(-24, 24, 1, dtype=np.float64)  # we use both before and after epoch to ensure sign is ok
        t_converted = convert_netcdf_time('hours since 1970-01-01 00:00:00', t_num)
        t_axis = api.TimeAxisFixedDeltaT(utc.time(1969, 12, 31, 0, 0, 0), api.deltahours(1), 2 * 24)
        [self.assertEqual(t_converted[i], t_axis(i).start) for i in range(t_axis.size())]
