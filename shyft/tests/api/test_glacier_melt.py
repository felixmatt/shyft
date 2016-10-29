
from shyft.api import GlacierMeltParameter
from shyft.api import glacier_melt_step
from shyft.api import deltahours
import unittest


class GlacierMelt(unittest.TestCase):
    """Verify and illustrate GlacierMelt routine and GlacierMeltTs exposure to python
    """

    def test_glacier_melt_parameter(self):
        p = GlacierMeltParameter(5.0)
        self.assertAlmostEqual(p.dtf, 5.0)

    def test_glacier_melt_step_function(self):
        dt = deltahours(1)
        dtf = 6.0
        temperature = 10.0
        sca = 0.5
        gf = 1.0
        m = glacier_melt_step(dt, dtf, temperature, sca, gf)
        self.assertAlmostEqual(1.25, m)