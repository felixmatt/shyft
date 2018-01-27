from builtins import range
import unittest
import os
import glob
from shyft.api import Calendar,CellStateId
from shyft.api import GammaSnowState, KirchnerState
from shyft.repository.yaml_state_repository import YamlStateRepository, StateSerializer
from shyft.api.pt_gs_k import PTGSKState,PTGSKStateWithId, PTGSKStateWithIdVector


class YamlStateRepositoryTestCase(unittest.TestCase):
    @property
    def _test_state_directory(self):
        return os.path.join(os.path.dirname(__file__), "state_tmp")

    def _clean_test_directory(self):
        files = glob.glob(os.path.join(self._test_state_directory, '*.*'))
        for f in files:
            os.remove(f)
        if os.path.exists(self._test_state_directory):
            os.rmdir(self._test_state_directory)

    def setUp(self):
        if not os.path.exists(self._test_state_directory):
            os.makedirs(self._test_state_directory)
        else:
            self._clean_test_directory()

    def tearDown(self):
        self._clean_test_directory()

    def _create_state_vector(self, n):
        r = PTGSKStateWithIdVector()
        for i in range(n):
            sid =PTGSKStateWithId()
            sid.id = CellStateId(cid=i%3,x=i*1000,y=i*1000,area=1000*1000)
            sid.state=PTGSKState(GammaSnowState(albedo=0.1 * i, lwc=0.1 * i), KirchnerState(q=0.3 + i))
            r.append(sid)
        return r

    def test_create_empty_gives_no_state(self):
        state_repository = YamlStateRepository(directory_path=self._test_state_directory, state_serializer=StateSerializer(PTGSKStateWithIdVector))
        self.assertIsNotNone(state_repository)
        self.assertEqual(len(state_repository.find_state()), 0, "We expect 0 states for empty repository")

    def test_crudf_cycle(self):
        """
        Verify we can create, store, read, find and delete state in the state repository
        
        """

        # arrange, by creating one State
        cal = Calendar()
        utc_timestamp = cal.time(2001, 1, 1)
        region_model_id = "neanidelv-ptgsk"
        n_cells = 10
        tags = ["initial", "unverified"]
        state_vector = self._create_state_vector(n_cells)
        self.assertIsNotNone(state_vector, "we should have a valid state vector object at this spot")
        # now start state_repository test
        state_repository = YamlStateRepository(directory_path=self._test_state_directory,state_serializer=StateSerializer(PTGSKStateWithIdVector))
        # put in two states, record the unique state_id..
        state_id_1 = state_repository.put_state(region_model_id, utc_timestamp, state_vector, tags)
        state_id_2 = state_repository.put_state(region_model_id, utc_timestamp, state_vector, tags)
        # assert that we got two unique state_id
        self.assertIsNotNone(state_id_1, "We expect back a unique id")
        self.assertIsNotNone(state_id_2, "We expect back a unique id")
        self.assertNotEqual(state_id_1, state_id_2,
                            "storing two state, same model, same time, each state should be stored with a unique id")
        # now we should have two states in the repository
        state_infos = state_repository.find_state()
        self.assertEqual(2, len(state_infos), "We just stored two, expect two back..")
        # extra test, verify that we really stored the state (using kirchner q)
        state_1 = state_repository.get_state(state_id_1)
        self.assertEqual(n_cells,len( state_1), "expect to get back state with same number of cells")
        for i in range(n_cells):
            self.assertAlmostEqual(state_1[i].state.kirchner.q, state_vector[i].state.kirchner.q, 3,
                                   "state repository should preserve state...")
        # now remove state
        state_repository.delete_state(state_id_1)
        # check that we got just one left, and that it is the correct one..
        state_list = state_repository.find_state()
        self.assertEqual(1, len(state_list))
        self.assertEqual(state_list[0].region_model_id, region_model_id)
        self.assertEqual(state_list[0].utc_timestamp, utc_timestamp)
        self.assertEqual(state_list[0].state_id, state_id_2)

    def test_find_with_region_model_filter(self):
        cal = Calendar()
        utc_timestamp = cal.time(2001, 1, 1)
        region_model_id = "neanidelv-ptgsk"
        n_cells = 10
        tags = ["initial", "unverified"]
        state_vector = self._create_state_vector(n_cells)
        # now start state_repository test
        state_repository = YamlStateRepository(directory_path=self._test_state_directory,state_serializer=StateSerializer(PTGSKStateWithIdVector))
        # put in two states, record the unique state_id..
        state_repository.put_state(region_model_id, utc_timestamp, state_vector, tags)
        state_repository.put_state("tokke-ptgsk", utc_timestamp, state_vector, tags)
        all_states = state_repository.find_state()
        neanidelv_states = state_repository.find_state(region_model_id)
        self.assertEqual(2, len(all_states))
        self.assertEqual(1, len(neanidelv_states))
        self.assertEqual(neanidelv_states[0].region_model_id, region_model_id)

    def test_find_with_region_model_and_time_filter(self):
        cal = Calendar()
        region_model_id = "neanidelv-ptgsk"
        n_cells = 10
        tags = ["initial", "unverified"]
        state_vector = self._create_state_vector(n_cells)
        # now start state_repository test
        state_repository = YamlStateRepository(directory_path=self._test_state_directory,state_serializer=StateSerializer(PTGSKStateWithIdVector))
        # put in two states, record the unique state_id..
        state_id_1 = state_repository.put_state(region_model_id, cal.time(2001, 1, 1, 0, 0, 0), state_vector, tags)
        state_id_2 = state_repository.put_state(region_model_id, cal.time(2001, 1, 2, 0, 0, 0), state_vector,tags)
        all_states = state_repository.find_state()
        neanidelv_states = state_repository.find_state(region_model_id)
        self.assertEqual(2, len(all_states))
        self.assertEqual(2, len(neanidelv_states))
        most_recent_state_before_time = state_repository.find_state(region_model_id, cal.time(2001, 1, 1, 0, 0, 0))
        self.assertEqual(1, len(most_recent_state_before_time))
        self.assertEqual(state_id_1, most_recent_state_before_time[0].state_id)
        self.assertEqual(0,len(state_repository.find_state(region_model_id, cal.time(2000, 12, 31, 23, 59, 59))))
        self.assertEqual(state_id_2,state_repository.find_state(region_model_id, cal.time(2002, 1, 1, 0, 0, 0))[0].state_id)


if __name__ == '__main__':
    unittest.main()
