from shyft import api
import unittest


class GridPP(unittest.TestCase):
    """Verify GridPP API to process forecasts from met.no before feeding into shyft.
       Test diverse methods for transforming data sets to grids and vice versa.
       Expose API for IDW and BK from shyft core.
       Calculate bias timeseries using a Kalman filter algorithm. 
     """
    
    def test_idw_temperature_should_transform_set(self):
        # TODO: expose idw from core api
        p1 = api.GeoPoint(1, 2, 3)
        self.assertAlmostEqual(p1.x, 1)
        self.assertAlmostEqual(p1.y, 2)
        self.assertAlmostEqual(p1.z, 3)


if __name__ == "__main__":
    unittest.main()
