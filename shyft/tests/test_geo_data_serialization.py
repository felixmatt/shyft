import unittest
import numpy as np
from shyft.api import pt_gs_k
from shyft.repository.service.gis_region_model_repository import \
    (GridSpecification, get_grid_spec_from_catch_poly, RegionModelConfig, GisRegionModelRepository)

try:
    
    from statkraft.ssa.environment import SMG_PREPROD as PREPROD

    def import_check():
        return PREPROD  # just to silence the module unused

    class SerializationTestCase(unittest.TestCase):

        def test_serialize_and_deserialize(self):
            rm_type = pt_gs_k.PTGSKModel
            reg_param = rm_type.parameter_t()
            epsg_id = 32633
            catchment_type = 'regulated'
            identifier = 'SUBCATCH_ID'
            dxy = 1000.
            pad = 5
            # LTM5-Tya
            id_list = [172, 174, 177, 185, 187, 190]
            grid_specification = get_grid_spec_from_catch_poly(id_list, catchment_type, identifier, epsg_id, dxy, pad)
            # LTM5-Nea
            id_list_1 = [180, 188, 191, 196]
            grid_specification_1 = get_grid_spec_from_catch_poly(id_list_1, catchment_type, identifier, epsg_id, dxy, pad)

            cfg_list = [
                RegionModelConfig('Tya', rm_type, reg_param, grid_specification, catchment_type, identifier, id_list),
                RegionModelConfig('Nea', rm_type, reg_param, grid_specification_1, catchment_type, identifier, id_list_1),
            ]
            rm_cfg_dict = {x.name: x for x in cfg_list}
            rm_repo = GisRegionModelRepository(rm_cfg_dict, use_cache=True)

            # Start by removing existing cache file
            GisRegionModelRepository.remove_cache(identifier, grid_specification)
            # Update cache with Nea cell_data
            GisRegionModelRepository.update_cache(catchment_type, identifier, grid_specification_1, id_list_1)
            # Construct a region_model for Tya - this process should automatically append Tya cell_data to cache
            rm = rm_repo.get_region_model('Tya')
            cells_from_rm = rm.get_cells()
            # Get Tya cell_data from the region_model
            cell_data_from_region_model = cells_from_rm.geo_cell_data_vector(cells_from_rm).to_numpy().reshape(-1, 11)
            # Get Tya cell_data from auto generated cache
            cell_data_from_auto_cache = GisRegionModelRepository.get_cell_data_from_cache(
                identifier, grid_specification, id_list)
            # Remove the cache file
            GisRegionModelRepository.remove_cache(identifier, grid_specification)
            # Update cache with Tya cell_data
            GisRegionModelRepository.update_cache(catchment_type, identifier, grid_specification, id_list)
            # Update cache with Nea cell_data
            GisRegionModelRepository.update_cache(catchment_type, identifier, grid_specification_1, id_list_1)
            # Get Tya cell_data from updated cache
            cell_data_from_updated_cache = GisRegionModelRepository.get_cell_data_from_cache(
                identifier, grid_specification, id_list)
            # Get Nea cell_data from updated cache
            cell_data_from_updated_cache_1 = GisRegionModelRepository.get_cell_data_from_cache(
                identifier, grid_specification_1, id_list_1)
            # Remove the cache file
            GisRegionModelRepository.remove_cache(identifier, grid_specification)
            # Get Nea cell_data from GIS
            cell_data_from_gis_1 = GisRegionModelRepository.get_cell_data_from_gis(
                catchment_type, identifier, grid_specification_1, id_list_1)
            # Get Tya cell_data from GIS
            cell_data_from_gis = GisRegionModelRepository.get_cell_data_from_gis(
                catchment_type, identifier, grid_specification, id_list)
            # Update cache with Tya cell_data
            GisRegionModelRepository.update_cache(catchment_type, identifier, grid_specification, id_list)
            # Construct a region_model for Nea - this process should automatically append Nea cell_data to cache')
            rm_1 = rm_repo.get_region_model('Nea')
            cells_from_rm_1 = rm_1.get_cells()
            # Get Nea cell_data from the region_model
            cell_data_from_region_model_1 = cells_from_rm_1.geo_cell_data_vector(cells_from_rm_1).to_numpy().reshape(-1, 11)
            # Get Nea cell_data from auto generated cache
            cell_data_from_auto_cache_1 = GisRegionModelRepository.get_cell_data_from_cache(
                identifier, grid_specification_1, id_list_1)
            # Remove the cache file
            GisRegionModelRepository.remove_cache(identifier, grid_specification)

            # Comparing results...
            # Arrays to compare
            # Tya
            # cell_data_from_region_model, cell_data_from_auto_cache, cell_data_from_updated_cache, cell_data_from_gis
            atol = 1e-08
            self.assertTrue(np.allclose(cell_data_from_region_model, cell_data_from_gis['geo_data'], atol=atol))
            self.assertTrue(np.allclose(cell_data_from_auto_cache['geo_data'], cell_data_from_gis['geo_data'], atol=atol))
            self.assertTrue(np.allclose(cell_data_from_updated_cache['geo_data'], cell_data_from_gis['geo_data'], atol=atol))

            self.assertTrue(np.allclose(rm.catchment_id_map, cell_data_from_gis['cid_map'], atol=atol))
            self.assertTrue(np.allclose(cell_data_from_auto_cache['cid_map'], cell_data_from_gis['cid_map'], atol=atol))
            self.assertTrue(np.allclose(cell_data_from_updated_cache['cid_map'], cell_data_from_gis['cid_map'], atol=atol))
            # Nea
            # cell_data_from_region_model_1, cell_data_from_auto_cache_1, cell_data_from_updated_cache_1, cell_data_from_gis_1
            self.assertTrue(np.allclose(cell_data_from_region_model_1, cell_data_from_gis_1['geo_data'], atol=atol))
            self.assertTrue(np.allclose(cell_data_from_auto_cache_1['geo_data'], cell_data_from_gis_1['geo_data'], atol=atol))
            self.assertTrue(np.allclose(cell_data_from_updated_cache_1['geo_data'], cell_data_from_gis_1['geo_data'], atol=atol))

            self.assertTrue(np.allclose(rm_1.catchment_id_map, cell_data_from_gis_1['cid_map'], atol=atol))
            self.assertTrue(np.allclose(cell_data_from_auto_cache_1['cid_map'], cell_data_from_gis_1['cid_map'], atol=atol))
            self.assertTrue(np.allclose(cell_data_from_updated_cache_1['cid_map'], cell_data_from_gis_1['cid_map'], atol=atol))

except ImportError as ie:
    if 'statkraft' in str(ie):
        print("(Test require statkraft.script environment to run: {})".format(ie))
    else:
        print("ImportError: {}".format(ie))

if __name__ == '__main__':
    unittest.main()