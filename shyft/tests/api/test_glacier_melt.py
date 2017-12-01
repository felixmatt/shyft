
from shyft.api import GlacierMeltParameter
from shyft.api import glacier_melt_step
from shyft.api import deltahours
from shyft.api import Calendar

from shyft.api import TimeAxis
from shyft.api import TimeSeries
from shyft.api import point_interpretation_policy as fx_policy
from shyft.api import create_glacier_melt_ts_m3s
from shyft.api import DoubleVector as dv
from numpy.testing import assert_array_almost_equal
import numpy as np
import unittest


class GlacierMelt(unittest.TestCase):
    """Verify and illustrate GlacierMelt routine and GlacierMeltTs exposure to python
    """

    def test_glacier_melt_parameter(self):
        p = GlacierMeltParameter(5.0)
        self.assertAlmostEqual(p.dtf, 5.0)
        self.assertAlmostEqual(p.direct_response, 0.0)
        p = GlacierMeltParameter(5.0, 1.0)
        self.assertAlmostEqual(p.dtf, 5.0)
        self.assertAlmostEqual(p.direct_response, 1.0)
        p.direct_response = 0.5
        self.assertAlmostEqual(p.direct_response, 0.5)

    def test_glacier_melt_step_function(self):
        dtf = 6.0
        temperature = 10.0
        area_m2 = 3600.0/0.001
        sca = 0.5 * area_m2
        gf = 1.0 * area_m2
        m = glacier_melt_step(dtf, temperature, sca, gf)
        self.assertAlmostEqual(1.25, m)  # mm/h
        self.assertAlmostEqual(0.0, glacier_melt_step(dtf, 0.0, sca, gf),5, 'no melt at 0.0 deg C')
        self.assertAlmostEqual(0.0, glacier_melt_step(dtf, 10.0, 0.7, 0.6),5, 'no melt when glacier is covered')

    def test_glacier_melt_ts_m3s(self):
        utc = Calendar()
        t0 = utc.time(2016,10,1)
        dt = deltahours(1)
        n = 240
        ta = TimeAxis(t0, dt, n)
        area_m2 = 487*1000*1000  # Jostedalsbreen, largest i Europe
        temperature = TimeSeries(ta=ta, fill_value=10.0, point_fx=fx_policy.POINT_AVERAGE_VALUE)
        sca_values = dv.from_numpy(np.linspace(area_m2*1.0,0.0,num=n))
        sca = TimeSeries(ta=ta, values=sca_values, point_fx=fx_policy.POINT_AVERAGE_VALUE)
        gf = 1.0 *area_m2
        dtf = 6.0
        melt_m3s = create_glacier_melt_ts_m3s(temperature, sca, gf, dtf) # Here we get back a melt_ts, that we can do ts-stuff with
        self.assertIsNotNone(melt_m3s)
        full_melt_m3s = glacier_melt_step(dtf, 10.0, 0.0, gf)
        expected_melt_m3s = np.linspace(0.0,full_melt_m3s,num=n)
        assert_array_almost_equal(expected_melt_m3s,melt_m3s.values.to_numpy(),4)
        # Just to check we can work with the result as a ts in all aspects
        mx2 = melt_m3s*2.0
        emx2 = expected_melt_m3s * 2.0;
        assert_array_almost_equal(emx2, mx2.values.to_numpy(), 4)
