import unittest
from shyft.repository.service.gis_region_model_repository import CatchmentFetcher
from shyft.repository.service.gis_region_model_repository import GridSpecification
from shyft.repository.service.gis_region_model_repository import DTMFetcher
from shyft.repository.service.gis_region_model_repository import LandTypeFetcher
from shyft.repository.service.gis_region_model_repository import ReservoirFetcher
from shyft.repository.service.gis_region_model_repository import CellDataFetcher



class GisRegionModelRepositoryUsingKnownServiceResults(unittest.TestCase):
    """
    Note that all testcases are in order of building up from basic
    low-level services, up to the complete assembly that
    implements the RegionModelRepository

    From the GIS system we have the following services:
    
    Catchments --> giving the shape of specified catchment ids
                   this service can fetch shapes based on power_plant_id,catch_id (sk-ids) or feltnr(nve-id)
    Digital Terrain Model --> giving the elevation for a specified grid (dx,dy)
    LandTypes --> giving the shapes of forrest, lake,glaciers etc.
    Reservoir --> giving the mid-point of all lakes that are regulated, (precipitation is immediate inflow)

    """

    def test_catchment_fetcher_using_regulated_power_plant_id(self):
        id_list=[236] #RanaLangvatn power_plant_id
        cf=CatchmentFetcher("regulated","POWER_PLANT_ID")
        r=cf.fetch(id_list=id_list);
        self.assertIsNotNone(r)
        self.assertIsNotNone(r[id_list[0]])

    def test_catchment_fetcher_using_regulated_catch_id(self):
        id_list=[2402] #stuggusjøen catch_id
        cf=CatchmentFetcher("regulated","CATCH_ID")
        r=cf.fetch(id_list=id_list);
        self.assertIsNotNone(r)
        self.assertIsNotNone(r[id_list[0]])

    def test_catchment_fetcher_using_unregulated_feld_nr(self):
        cf=CatchmentFetcher("unregulated","FELTNR")
        id_list=[1225]
        r=cf.fetch(id_list=id_list)
        self.assertIsNotNone(r)
        self.assertIsNotNone(r[id_list[0]])

    def test_dtm_fetcher(self):
        gs=GridSpecification(x0=557600,y0=7040000,dx=1000,dy=1000,nx=122,ny=90)
        dtmf=DTMFetcher(gs)
        r=dtmf.fetch()
        self.assertIsNotNone(r)
        shape=r.shape
        self.assertEquals(shape[0],gs.ny)
        self.assertEquals(shape[1],gs.nx)

    def test_land_type_fetcher(self):
        gs=GridSpecification(x0=557600,y0=7040000,dx=1000,dy=1000,nx=10,ny=10)
        ltf=LandTypeFetcher(geometry=gs.geometry,epsg_id=32632)
        for lt_name in ltf.en_field_names:
            lt=ltf.fetch(name=lt_name)
            self.assertIsNotNone(lt)

    def test_reservoir_fetcher(self):
        gs=GridSpecification(x0=557600,y0=7040000,dx=1000,dy=1000,nx=122,ny=90)
        rf=ReservoirFetcher(gs.geometry,epsg_id=32632)
        rpts=rf.fetch()
        self.assertIsNotNone(rpts)
        self.assertEquals(22,len(rpts)) 


    def test_cell_data_fetcher_ranalangvatn(self):
        gs=GridSpecification(x0=704000,y0=7431000,dx=1000,dy=1000,nx=98,ny=105)
        pwrplants=[236]
        cdf=CellDataFetcher(catchment_type="regulated",identifier="POWER_PLANT_ID",grid_specification=gs,id_list=pwrplants,epsg_id=32632)
        cd=cdf.fetch()
        self.assertIsNotNone(cd)
        self.assertIsNotNone(cd['cell_data'])
        self.assertIsNotNone(cd['cell_data'][pwrplants[0]])
        self.assertIsNotNone(cd['catchment_land_types'])
        self.assertIsNotNone(cd['elevation_raster'])


if __name__ == '__main__':
    unittest.main()
