from shyft import api
import unittest


class GeoCellData(unittest.TestCase):
    """Verify and illustrate GeoCellData exposure to python
       
     """

    def test_create(self):
        p=api.GeoPoint(100,200,300)
        ltf=api.LandTypeFractions()
        ltf.set_fractions(glacier=0.1,lake=0.1,reservoir=0.1,forest=0.1)
        self.assertAlmostEqual(ltf.unspecified(),0.6)
        gcd=api.GeoCellData(p,1000000.0,1,0.9,ltf)
        
        self.assertAlmostEqual(gcd.area(),1000000)
        self.assertAlmostEqual(gcd.catchment_id(),1)

     
    def test_land_type_fractions(self):
        """ 
         LandTypeFractions describes how large parts of a cell is 
         forest,glacier, lake ,reservoir, - the rest is unspecified
         The current cell algorithms like ptgsk uses this information
         to manipulate the response.
         e.g. precipitation that falls into the reservoir fraction goes directly to 
         the response (the difference of lake and reservoir is that reservoir is a lake where
         we store water to the power-plants.)
         
        """
        # constructor 1 :all in one: specify glacier_size,lake_size,reservoir_size,forest_size,unspecified_size
        a=api.LandTypeFractions(1000.0,2000.0,3000.0,4000.0,5000.0)# keyword arguments does not work ??
        # constructor 2: create, and set (with possible exceptions)
        b=api.LandTypeFractions()
        b.set_fractions(glacier=1/15.0,lake=2/15.0,reservoir=3/15.0,forest=4/15.0)
        self.assertAlmostEqual(a.glacier(),b.glacier())
        self.assertAlmostEqual(a.lake(),b.lake())
        self.assertAlmostEqual(a.reservoir(),b.reservoir())
        self.assertAlmostEqual(a.forest(),b.forest())
        self.assertAlmostEqual(a.unspecified(),b.unspecified())
        try:
            
            b.set_fractions(glacier=0.9,forest=0.2,lake=0.0,reservoir=0.0)
            self.fail("expected exception, nothing raised")
        except  :
            self.assertTrue(True,"If we reach here all is ok")
        
        
if __name__ == "__main__":
    unittest.main()
