from shyft import api
import unittest


class TimeGeoPoint(unittest.TestCase):
    """Verify and illustrate GeoPoint exposure to python
       
     """
    
    def test_create_GeoPoint(self):
        p1=api.GeoPoint(1,2,3)
       
        self.assertAlmostEqual(p1.x,1)
        self.assertAlmostEqual(p1.y,2)
        self.assertAlmostEqual(p1.z,3)

    def test_create_default_GeoPoint(self):
        p2=api.GeoPoint()
        self.assertAlmostEqual(p2.x,0)
        self.assertAlmostEqual(p2.y,0)
        self.assertAlmostEqual(p2.z,0)
    
    def test_GeoPoint_difference(self):
        a=api.GeoPoint(1,2,3)
        b=api.GeoPoint(3,4,8)
        d=api.GeoPoint_difference(b,a)
        self.assertAlmostEquals(d.x,b.x-a.x)
        
    def test_GeoPoint_xy_distance(self):
        a=api.GeoPoint(1,2,3)
        b=api.GeoPoint(3,4,8)
        d=api.GeoPoint_xy_distance(b,a)
        self.assertAlmostEqual(d,2.8284271247 )
        
if __name__ == "__main__":
    unittest.main()
