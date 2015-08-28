from shyft import api
import unittest


class TimeGeoCellData(unittest.TestCase):
    """Verify and illustrate GeoCellData exposure to python
       
     """

    def test_create_GeoCellData(self):
        p=api.GeoPoint(100,200,300)
        ltf=api.LandTypeFractions()
        ltf.set_fractions(glacier=0.1,lake=0.1,reservoir=0.1,forest=0.1)
        self.assertAlmostEqual(ltf.unspecified(),0.6)
        gcd=api.GeoCellData(p,1000000.0,1,0.9,ltf)
        
        self.assertAlmostEqual(gcd.area(),1000000)
        self.assertAlmostEqual(gcd.catchment_id(),1)

        
if __name__ == "__main__":
    unittest.main()
