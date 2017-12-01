import unittest
from shyft import api
from shyft.api import pt_gs_k
from shyft.api import pt_ss_k
from shyft.api.pt_gs_k import PTGSKModel, PTGSKOptModel
from shyft.api.pt_ss_k import PTSSKModel

try:
    # just to verify we are at statkraft, leave this one inside
    from statkraft.ssa.environment import SMG_PREPROD as PREPROD
    from shyft.repository.service.gis_region_model_repository import CatchmentFetcher
    from shyft.repository.service.gis_region_model_repository import GridSpecification
    from shyft.repository.service.gis_region_model_repository import DTMFetcher
    from shyft.repository.service.gis_region_model_repository import LandTypeFetcher
    from shyft.repository.service.gis_region_model_repository import ReservoirFetcher
    from shyft.repository.service.gis_region_model_repository import CellDataFetcher
    from shyft.repository.service.gis_region_model_repository import RegionModelConfig
    from shyft.repository.service.gis_region_model_repository import GisRegionModelRepository
    from shyft.repository.service.gis_region_model_repository import get_grid_spec_from_catch_poly
    from shyft.repository.service.gis_region_model_repository import peru_service, peru_dem, peru_catchment_id_name, peru_catchment_type, peru_subcatch_id_name, nordic_dem
    from shyft.repository.service.gis_region_model_repository import nordic_service, nordic_catchment_type_regulated, nordic_catchment_type_ltm, nordic_catchment_type_unregulated


    def import_check():
        return PREPROD  # just to silence the module unused


    peru_grid_spec = GridSpecification(epsg_id=32718, x0=375000, y0=8789000, dx=1000, dy=1000, nx=68, ny=61)


    class GisRegionModelRepositoryUsingKnownServiceResults(unittest.TestCase):
        """
        Note that all test-cases are in order of building up from basic
        low-level services, up to the complete assembly that
        implements the RegionModelRepository

        From the GIS system we have the following services:

        Catchments --> giving the shape of specified catchment ids
                       this service can fetch shapes based on power_plant_id,catch_id (sk-ids) or feltnr(nve-id)
        Digital Terrain Model --> giving the elevation for a specified grid (dx,dy)
        LandTypes --> giving the shapes of forrest, lake,glaciers etc.
        Reservoir --> giving the mid-point of all lakes that are regulated, (precipitation is immediate inflow)

        """

        @property
        def epsg_id(self):
            return 32633

        def test_catchment_fetcher_using_regulated_power_plant_id(self):
            id_list = [236]  # RanaLangvatn power_plant_id
            cf = CatchmentFetcher(nordic_catchment_type_regulated, "POWER_PLANT_ID", self.epsg_id)
            r = cf.fetch(id_list=id_list)
            self.assertIsNotNone(r)
            self.assertIsNotNone(r[id_list[0]])

        def test_catchment_fetcher_using_regulated_catch_id(self):
            id_list = [2402]  # stuggusjøen catch_id
            cf = CatchmentFetcher(nordic_catchment_type_regulated, "CATCH_ID", self.epsg_id)
            r = cf.fetch(id_list=id_list)
            self.assertIsNotNone(r)
            self.assertIsNotNone(r[id_list[0]])

        def test_catchment_fetcher_using_unregulated_feld_nr(self):
            cf = CatchmentFetcher(nordic_catchment_type_unregulated, "FELTNR", self.epsg_id)
            id_list = [1225]
            r = cf.fetch(id_list=id_list)
            self.assertIsNotNone(r)
            self.assertIsNotNone(r[id_list[0]])

        def test_catchment_fetcher_peru(self):
            global peru_grid_spec
            cf = CatchmentFetcher(peru_catchment_type, peru_catchment_id_name, peru_grid_spec.epsg_id)
            id_list = [2, 4, 5, 6, 9, 10]  # this is yuapi
            r = cf.fetch(id_list=id_list)
            self.assertIsNotNone(r)
            self.assertIsNotNone(r[id_list[0]])
            self.assertGreater(r[id_list[0]].area, 50.0)


        def test_dtm_fetcher(self):
            gs = GridSpecification(32632, x0=557600, y0=7040000, dx=1000, dy=1000, nx=122, ny=90)
            dtmf = DTMFetcher(gs, dem=nordic_dem)
            r = dtmf.fetch()
            self.assertIsNotNone(r)
            shape = r.shape
            self.assertEqual(shape[0], gs.ny)
            self.assertEqual(shape[1], gs.nx)

        def test_dtm_fetcher_peru(self):
            gs = peru_grid_spec
            dtmf = DTMFetcher(gs, dem=peru_dem)
            r = dtmf.fetch()
            self.assertIsNotNone(r)
            shape = r.shape
            self.assertEqual(shape[0], gs.ny)
            self.assertEqual(shape[1], gs.nx)

        def test_land_type_fetcher(self):
            gs = GridSpecification(32632, x0=557600, y0=7040000, dx=1000, dy=1000, nx=10, ny=10)
            ltf = LandTypeFetcher(geometry=gs.geometry, epsg_id=32632)
            for lt_name in ltf.en_field_names:
                lt = ltf.fetch(name=lt_name)
                self.assertIsNotNone(lt)

        def test_land_type_fetcher_peru(self):
            # gs = GridSpecification(32718, x0=0, y0=8000000, dx=1000, dy=1000, nx=500, ny=500)
            gs = peru_grid_spec  # GridSpecification(32718, x0=370000, y0=8700000, dx=1000, dy=1000, nx=60, ny=55)
            ltf = LandTypeFetcher(geometry=gs.geometry, epsg_id=gs.epsg(), sub_service=peru_service)
            for lt_name in ltf.en_field_names:
                lt = ltf.fetch(name=lt_name)
                self.assertIsNotNone(lt)

        def test_reservoir_fetcher(self):
            gs = GridSpecification(32632, x0=557600, y0=7040000, dx=1000, dy=1000, nx=122, ny=90)
            rf = ReservoirFetcher(epsg_id=gs.epsg(), geometry=gs.geometry)
            rpts = rf.fetch()
            self.assertIsNotNone(rpts)
            self.assertEqual(24, len(rpts))  # well, seems that this can change when doing maintenance in db ?

        def test_reservoir_fetcher_peru(self):
            gs = peru_grid_spec
            rf = ReservoirFetcher(epsg_id=gs.epsg(), geometry=gs.geometry, sub_service=peru_service)
            rpts = rf.fetch()
            self.assertIsNotNone(rpts)
            self.assertEqual(9, len(rpts))  # well, seems that this can change when doing maintenance in db ?

        def test_cell_data_fetcher_ranalangvatn(self):
            gs = GridSpecification(32632, x0=704000, y0=7431000, dx=1000, dy=1000, nx=98, ny=105)
            pwrplants = [236]
            cdf = CellDataFetcher(catchment_type=nordic_catchment_type_regulated, identifier="POWER_PLANT_ID", grid_specification=gs,
                                  id_list=pwrplants)
            cd = cdf.fetch()
            self.assertIsNotNone(cd)
            self.assertIsNotNone(cd['cell_data'])
            self.assertIsNotNone(cd['cell_data'][pwrplants[0]])
            self.assertIsNotNone(cd['catchment_land_types'])
            self.assertIsNotNone(cd['elevation_raster'])

        def test_region_model_repository(self):
            id_list = [1225]
            epsg_id = 32632
            # parameters can be loaded from yaml_config Model parameters..
            pt_params = api.PriestleyTaylorParameter()  # *params["priestley_taylor"])
            gs_params = api.GammaSnowParameter()  # *params["gamma_snow"])
            ss_params = api.SkaugenParameter()
            ae_params = api.ActualEvapotranspirationParameter()  # *params["act_evap"])
            k_params = api.KirchnerParameter()  # *params["kirchner"])
            p_params = api.PrecipitationCorrectionParameter()  # TODO; default 1.0, is it used ??
            ptgsk_rm_params = pt_gs_k.PTGSKParameter(pt_params, gs_params, ae_params, k_params, p_params)
            ptssk_rm_params = pt_ss_k.PTSSKParameter(pt_params, ss_params, ae_params, k_params, p_params)
            # create the description for 4 models of tistel,ptgsk, ptssk, full and optimized
            tistel_grid_spec = GridSpecification(epsg_id=epsg_id, x0=362000.0, y0=6765000.0, dx=1000, dy=1000, nx=8, ny=8)
            cfg_list = [
                RegionModelConfig("tistel-ptgsk", PTGSKModel, ptgsk_rm_params, tistel_grid_spec, nordic_catchment_type_unregulated, "FELTNR", id_list),
                RegionModelConfig("tistel-ptgsk-opt", PTGSKOptModel, ptgsk_rm_params, tistel_grid_spec, nordic_catchment_type_unregulated, "FELTNR", id_list),
                RegionModelConfig("tistel-ptssk", PTSSKModel, ptssk_rm_params, tistel_grid_spec, nordic_catchment_type_unregulated, "FELTNR", id_list)
            ]
            rm_cfg_dict = {x.name: x for x in cfg_list}
            rmr = GisRegionModelRepository(rm_cfg_dict)  # ok, now we have a Gis RegionModelRepository that can handle all named entities we pass.
            cm1 = rmr.get_region_model("tistel-ptgsk")  # pull out a PTGSKModel for tistel
            cm2 = rmr.get_region_model("tistel-ptgsk-opt")
            # Does not work, fail on ct. model:
            cm3 = rmr.get_region_model("tistel-ptssk")  # pull out a PTGSKModel for tistel
            # cm4= rmr.get_region_model("tistel-ptssk",PTSSKOptModel)
            self.assertIsNotNone(cm3)
            self.assertIsNotNone(cm1)
            self.assertIsNotNone(cm2)
            self.assertIsNotNone(cm2.catchment_id_map)  # This one is needed in order to properly map catchment-id to internal id
            self.assertEqual(cm2.catchment_id_map[0], id_list[0])
            # self.assertIsNotNone(cm4)

        @property
        def std_ptgsk_parameters(self):
            pt_params = api.PriestleyTaylorParameter()  # *params["priestley_taylor"])
            gs_params = api.GammaSnowParameter()  # *params["gamma_snow"]
            ae_params = api.ActualEvapotranspirationParameter()  # *params["act_evap"])
            k_params = api.KirchnerParameter()  # *params["kirchner"])
            p_params = api.PrecipitationCorrectionParameter()  # TODO; default 1.0, is it used ??
            return pt_gs_k.PTGSKParameter(pt_params, gs_params, ae_params, k_params, p_params)

        def test_region_model_nea_nidelv(self):
            nea_nidelv_grid_spec = GridSpecification(epsg_id=32633, x0=266000.0, y0=6960000.0, dx=1000, dy=1000, nx=109, ny=80)
            catch_ids = [1228, 1308, 1394, 1443, 1726, 1867, 1996, 2041, 2129, 2195, 2198, 2277, 2402, 2446, 2465, 2545,
                         2640, 2718, 3002, 3536, 3630]  # , 1000010, 1000011]
            ptgsk_params = self.std_ptgsk_parameters
            cfg_list = [
                RegionModelConfig("nea-nidelv-ptgsk", PTGSKModel, ptgsk_params, nea_nidelv_grid_spec, nordic_catchment_type_regulated, "CATCH_ID", catch_ids)
            ]
            rm_cfg_dict = {x.name: x for x in cfg_list}
            rmr = GisRegionModelRepository(rm_cfg_dict)
            nea_nidelv = rmr.get_region_model("nea-nidelv-ptgsk")
            self.assertIsNotNone(nea_nidelv)
            self.assertEqual(len(nea_nidelv.catchment_id_map), len(catch_ids))
            print("nea-nidelv:{0}".format(nea_nidelv.catchment_id_map))

        def test_region_model_peru_yuapi(self):
            global peru_grid_spec
            catch_ids = [2, 4, 5, 6, 9, 10]
            ptgsk_params = self.std_ptgsk_parameters
            cfg_list = [
                RegionModelConfig("peru-yuapi-ptgsk", PTGSKModel, ptgsk_params, peru_grid_spec, peru_catchment_type, peru_subcatch_id_name, catch_ids)
            ]
            rm_cfg_dict = {x.name: x for x in cfg_list}
            rmr = GisRegionModelRepository(rm_cfg_dict)
            peru_yuapi = rmr.get_region_model("peru-yuapi-ptgsk")
            self.assertIsNotNone(peru_yuapi)
            self.assertEqual(len(peru_yuapi.catchment_id_map), len(catch_ids))

        def test_bounding_box_for_peru_catchment(self):
            gs = get_grid_spec_from_catch_poly(
                catch_ids=[2, 4, 5, 6, 9, 10],
                catchment_type=peru_catchment_type,
                identifier=peru_subcatch_id_name,
                epsg_id=32718,
                dxy=1000,
                pad=5)
            self.assertIsNotNone(gs)
            global peru_grid_spec
            self.assertEqual(gs.epsg_id, peru_grid_spec.epsg_id)
            self.assertEqual(gs.nx, peru_grid_spec.nx)
            self.assertEqual(gs.ny, peru_grid_spec.ny)
            self.assertAlmostEqual(gs.x0, peru_grid_spec.x0)
            self.assertAlmostEqual(gs.y0, peru_grid_spec.y0)
            self.assertAlmostEqual(gs.dx, peru_grid_spec.dx)
            self.assertAlmostEqual(gs.dy, peru_grid_spec.dy)


except ImportError as ie:
    if 'statkraft' in str(ie):
        print("(Test require statkraft.script environment to run: {})".format(ie))
    else:
        print("ImportError: {}".format(ie))

if __name__ == '__main__':
    unittest.main()
