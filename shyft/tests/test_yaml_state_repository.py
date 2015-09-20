import unittest
from os import path
from os import makedirs
from os import remove
import glob
#from shyft import shyftdata_dir
from shyft.api import Calendar,YMDhms
#from shyft.api import UtcPeriod
from shyft.api import GammaSnowState,KirchnerState
from shyft.repository.interfaces import StateInfo
from shyft.repository.yaml_state_repository import YamlStateRepository
from shyft.api.pt_gs_k import PTGSKState,PTGSKStateVector

class YamlStateRepositoryTestCase(unittest.TestCase):

    @property
    def _test_state_directory(self):
        return path.join(path.dirname(__file__), "state_tmp")
            
    def _clean_test_directory(self):
        files=glob.glob(path.join(self._test_state_directory,"*.yaml"))
        for f in files:
            remove(f)
            
    def setUp(self):
        self._clean_test_directory()

    def tearDown(self):
        pass

    def _create_state_vector(self,n):
        r=PTGSKStateVector()
        for i in xrange(n):
            r.push_back(PTGSKState(GammaSnowState(albedo=0.1*i,lwc=0.1*i),KirchnerState(q=0.3+i)))
        return r;
    
    def test_create_empty_gives_no_state(self):
        state_repository= YamlStateRepository(self._test_state_directory)
        self.assertIsNotNone(state_repository)
        self.assertEquals(len(state_repository.find_state()),0,"We expect 0 states for empty repository")
        
    def test_crudf_cycle(self):
        """
        Verify we can create, store, read, find and delete state in the state repository
        
        """
        
        # arrange, by creating one State
        cal=Calendar()
        utc_timestamp=cal.time(YMDhms(2001,1,1))
        region_model_id="neanidelv-ptgsk"
        n_cells=10
        tags=["initial","unverified"]
        state_vector=self._create_state_vector(n_cells)
        self.assertIsNotNone(state_vector, "we should have a valid state vector object at this spot")
        # now start state_repository test
        state_repository= YamlStateRepository(self._test_state_directory)
        # put in two states, record the unique state_id..
        state_id_1=state_repository.put_state(region_model_id, utc_timestamp, state_vector, tags)
        state_id_2=state_repository.put_state(region_model_id, utc_timestamp, state_vector, tags)
        # assert that we got two unique state_id
        self.assertIsNotNone(state_id_1,"We expect back a unique id")
        self.assertIsNotNone(state_id_2,"We expect back a unique id")
        self.assertNotEquals(state_id_1,state_id_2,"storing two state, same model, same time, each state should be stored with a unique id")
        # now we should have two states in the repository
        state_infos=state_repository.find_state()
        self.assertEquals(2,len(state_infos),"We just stored two, expect two back..")
        # extra test, verify that we really stored the state (using kirchner q)
        state_1=state_repository.get_state(state_id_1)
        self.assertEquals(n_cells,state_1.size(),"expect to get back state with same number of cells")
        for i in xrange(n_cells):
            self.assertAlmostEqual(state_1[i].kirchner.q, state_vector[i].kirchner.q, 3, "state repository should preserve state...")
        #now remove state
        state_repository.delete_state(state_id_1)
        # check that we got just one left, and that it is the correct one..
        state_list=state_repository.find_state()
        self.assertEquals(1,len(state_list))
        self.assertEquals(state_list[0].region_model_id,region_model_id)
        self.assertEquals(state_list[0].utc_timestamp,utc_timestamp)
        self.assertEquals(state_list[0].state_id,state_id_2)
        
    def test_find_with_region_model_filter(self):
        cal=Calendar()
        utc_timestamp=cal.time(YMDhms(2001,1,1))
        region_model_id="neanidelv-ptgsk"
        n_cells=10
        tags=["initial","unverified"]
        state_vector=self._create_state_vector(n_cells)
        # now start state_repository test
        state_repository= YamlStateRepository(self._test_state_directory)
        # put in two states, record the unique state_id..
        state_repository.put_state(region_model_id, utc_timestamp, state_vector, tags)
        state_repository.put_state("tokke-ptgsk", utc_timestamp, state_vector, tags)
        all_states= state_repository.find_state()
        neanidelv_states=state_repository.find_state(region_model_id)
        self.assertEquals(2,len(all_states))
        self.assertEquals(1,len(neanidelv_states))
        self.assertEquals(neanidelv_states[0].region_model_id,region_model_id)
        
    def test_find_with_region_model_and_time_filter(self):
        cal=Calendar()
        region_model_id="neanidelv-ptgsk"
        n_cells=10
        tags=["initial","unverified"]
        state_vector=self._create_state_vector(n_cells)
        # now start state_repository test
        state_repository= YamlStateRepository(self._test_state_directory)
        # put in two states, record the unique state_id..
        state_id_1=state_repository.put_state(region_model_id, cal.time(YMDhms(2001,1,1,0,0,0)), state_vector, tags)
        state_id_2=state_repository.put_state(region_model_id, cal.time(YMDhms(2001,1,2,0,0,0)), state_vector, tags)
        all_states= state_repository.find_state()
        neanidelv_states=state_repository.find_state(region_model_id)
        self.assertEquals(2,len(all_states))
        self.assertEquals(2,len(neanidelv_states))
        most_recent_state_before_time=state_repository.find_state(region_model_id,cal.time(YMDhms(2001,1,1,0,0,0)))
        self.assertEquals(1,len(most_recent_state_before_time))
        self.assertEquals(state_id_1,most_recent_state_before_time[0].state_id)
        self.assertEquals(0,len(state_repository.find_state(region_model_id,cal.time(YMDhms(2000,12,31,23,59,59)))))
        self.assertEquals(state_id_2,state_repository.find_state(region_model_id,cal.time(YMDhms(2002,1,1,0,0,0)))[0].state_id)
        
        