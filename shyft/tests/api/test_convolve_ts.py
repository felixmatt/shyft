import unittest

from shyft.api import Calendar
from shyft.api import DoubleVector
from shyft.api import TimeAxisFixedDeltaT
from shyft.api import TimeSeries
from shyft.api import convolve_policy
from shyft.api import deltahours
from shyft.api import point_interpretation_policy as point_fx


class ConvolveTs(unittest.TestCase):
    """Verify and illustrate the ts.convolve_w(weights,policy)

     """

    def test_convolve_policy(self):
        utc = Calendar()
        ts = TimeSeries(ta=TimeAxisFixedDeltaT(utc.time(2001, 1, 1), deltahours(1), 24), fill_value=10.0, point_fx=point_fx.POINT_AVERAGE_VALUE)
        w = DoubleVector.from_numpy([0.05, 0.15, 0.6, 0.15, 0.05])
        cts = ts.convolve_w(w, convolve_policy.USE_FIRST)  # ensure mass-balance between source and cts
        self.assertIsNotNone(cts)
        self.assertEquals(len(cts), len(ts))
        self.assertEquals(cts.values.to_numpy().sum(), ts.values.to_numpy().sum())
