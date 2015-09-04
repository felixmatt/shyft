from numpy import random 
from datetime import datetime
import unittest
from shyft import api
import os
import yaml
from shyft.orchestration.state import set_ptgsk_model_state
from shyft.orchestration.state import extract_ptgsk_model_state
from shyft.orchestration.state import State
from shyft.orchestration.state import save_state_as_yaml_file
from shyft.orchestration.input_source import InputSource
from shyft.orchestration.repository.state_repository import TimeCondition
from shyft.orchestration.repository.state_repository import combine_conditions
from shyft.orchestration.repository.testsupport.mocks import MockRepository
from shyft.orchestration.repository.testsupport.mocks import MockStateRepository
from shyft.orchestration.repository.testsupport.mocks import MockInputSourceRepository
from shyft.orchestration.repository.testsupport.mocks import mock_cell_data
from shyft.orchestration.repository.testsupport.mocks import state_repository_factory
from shyft.orchestration.repository.testsupport.time_series import create_mock_station_data
from shyft.orchestration.repository.cell_read_only_repository import CellReadOnlyRepository
from shyft.orchestration.repository.cell_read_only_repository import FileCellRepository
from shyft.orchestration.repository.arome_data_repository import AromeDataRepository
from shyft.orchestration.repository.state_repository import yaml_file_storage_factory


class StateIOTestCase(unittest.TestCase):

    @staticmethod
    def build_model(model_t, model_size, num_catchments=1):

        cells = model_t.cell_t.vector_t()
        cell_area=1000*1000
        region_parameter=api.PTGSKParameter()
        for i in xrange(model_size):
            loc = (10000*random.random(2)).tolist() + (500*random.random(1)).tolist()
            gp = api.GeoPoint(*loc)
            geo_cell_data=api.GeoCellData(gp,cell_area,random.randint(0, num_catchments))
            cell=model_t.cell_t()
            cell.geo=geo_cell_data
            cells.append(cell)
        return model_t(region_parameter, cells)

    @staticmethod
    def build_mock_state_dict(**kwargs):
        pt = {}
        gs = {"albedo": 0.4, 
              "lwc": 0.1,
              "surface_heat": 30000,
              "alpha": 1.26,
              "sdc_melt_mean": 1.0,
              "acc_melt": 0.0,
              "iso_pot_energy": 0.0,
              "temp_swe": 0.0,
             }
        kirchner = {"q": 0.25}
        pt.update({(k,v) for k,v in kwargs.iteritems() if k in pt})
        gs.update({(k,v) for k,v in kwargs.iteritems() if k in gs})
        kirchner.update({(k,v) for k,v in kwargs.iteritems() if k in kirchner})
        state=api.PTGSKState()
        state.gs.albedo=gs["albedo"]
        state.gs.lwc=gs["lwc"]
        state.gs.surface_heat=gs["surface_heat"]
        state.gs.alpha=gs["alpha"]
        state.gs.sdc_melt_mean=gs["sdc_melt_mean"]
        state.gs.acc_melt=gs["acc_melt"]
        state.gs.iso_pot_energy=gs["iso_pot_energy"]
        state.gs.temp_swe=gs["temp_swe"]
        state.kirchner.q=kirchner["q"]
        sio=api.PTGSKStateIo()
        return sio.to_string(state) #{"pt": pt, "gs": gs, "kirchner": kirchner}

    def _create_constant_geo_ts(self,geoTsType,geo_point,utc_period,value):
        """ creates a time point ts, with one value at the start of the supplied utc_period """
        tv=api.UtcTimeVector()
        tv.push_back(utc_period.start)
        vv=api.DoubleVector()
        vv.push_back(value)
        cts=api.TsFactory().create_time_point_ts(utc_period,tv,vv)
        return geoTsType(geo_point,cts)
        
    def create_dummy_region_environment(self,time_axis,mid_point):
        re=api.ARegionEnvironment()
        
        re.precipitation=api.PrecipitationSourceVector()
        re.precipitation.append(self._create_constant_geo_ts(api.PrecipitationSource,mid_point,time_axis.total_period(),5.0))

        re.temperature=api.TemperatureSourceVector()
        re.temperature.append(self._create_constant_geo_ts(api.TemperatureSource,mid_point,time_axis.total_period(),10.0))
        
        re.wind_speed=api.WindSpeedSourceVector()
        re.wind_speed.append(self._create_constant_geo_ts(api.WindSpeedSource,mid_point,time_axis.total_period(),2.0))
        
        re.rel_hum=api.RelHumSourceVector()
        re.rel_hum.append(self._create_constant_geo_ts(api.RelHumSource,mid_point,time_axis.total_period(),0.7))
        
        re.radiation=api.RadiationSourceVector()
        re.radiation.append(self._create_constant_geo_ts(api.RadiationSource,mid_point,time_axis.total_period(),300.0))
        return re
        
        


    def test_model_initialize_and_run(self):
        num_cells = 20
        model_type=api.PTGSKModel
        model = self.build_model(model_type, num_cells)
        self.assertEqual(model.size(), num_cells)
        cal=api.Calendar()
        time_axis=api.Timeaxis(cal.time(api.YMDhms(2015,1,1,0,0,0)),api.deltahours(1),240)
        model.run_interpolation(api.InterpolationParameter(),time_axis,self.create_dummy_region_environment(time_axis,model.get_cells()[num_cells/2].geo.mid_point()))
        model.set_state_collection(-1,True) # enable state collection for all cells
        model.run_cells()
        cids=api.IntVector() # optional, we can add selective catchment_ids here
        sum_discharge=model.statistics.discharge(cids)
        avg_temperature=model.statistics.temperature(cids)
        avg_precipitation=model.statistics.precipitation(cids)
        for time_step in xrange(time_axis.size()):
            precip_raster= model.statistics.precipitation(cids,time_step) # example for raster output
            self.assertEquals(precip_raster.size(),num_cells)
        avg_gs_sca = model.gamma_snow_response.sca(cids) # swe output
        
        avg_gs_albedo = model.gamma_snow_state.albedo(cids) # lwc surface_heat alpha melt_mean melt iso_pot_energy temp_sw
        self.assertEqual(avg_temperature.size(), time_axis.size(), "expect results equal to time-axis size")
        

    def test_model_state_io(self):
        num_cells = 2
        for model_type in [api.PTGSKModel, api.PTGSKOptModel]:
            model = self.build_model(model_type, num_cells)
            state_list = []
            x=""
            for i in xrange(num_cells):
                state_list.append(self.build_mock_state_dict(q=(i + 1)*0.5/num_cells))
            initial_states=x.join(state_list)
            set_ptgsk_model_state(model, State(initial_states, datetime.strftime(datetime.utcnow(), "%Y-%m-%d-%M-%S")))
            retrieved_states = extract_ptgsk_model_state(model)
            self.assertEqual(initial_states,retrieved_states.state_list)

            # Test that the state can be serialized and de-serialized:
            serialized_states = yaml.dump(retrieved_states, default_flow_style=False)
            self.assertTrue(isinstance(serialized_states, str))
            deserialized_states = yaml.load(serialized_states)

            self.assertEqual(retrieved_states.state_list,deserialized_states.state_list)

            # Finally, set the deserialized states into the model:
            set_ptgsk_model_state(model, deserialized_states)

    def test_set_too_few_model_states(self):
        num_cells = 20
        for model_type in [api.PTGSKModel, api.PTGSKOptModel]:
            model = self.build_model(model_type, num_cells)
            states = []
            x=""
            for i in xrange(num_cells - 1):
                states.append(self.build_mock_state_dict(q=(i + 1)*0.5/num_cells))
            statestr=x.join(states)
            self.assertRaises(RuntimeError, set_ptgsk_model_state, model, State(statestr))
            for i in xrange(num_cells + 1):
                states.append(self.build_mock_state_dict(q=(i + 1)*0.5/num_cells))
            statestr=x.join(states)
            self.assertRaises(RuntimeError, set_ptgsk_model_state, model, State(statestr))

    #def test_pthsk_state_io(self):
    #    """ Just to verify it is pthsk and state io is working """
    #    sio=api.pthsk_state_io()
    #    s0=api.PTHSKStat()
    #    s0.hbv_snow.swe=30.0
    #    s0.hbv_snow.sca=3.0

    #   str= sio.to_string(s0)
    #   self.assertEqual(len(str),35)
    #    statev=api.PTHSKStateVector()
    #    s1=api.PTHSKStat()
    #    s1.hbv_snow.sca=3.0
    #    s1.hbv_snow.swe=40

    #   statev.push_back(s0)
    #   statev.push_back(s1)
    #    sstr=sio.to_string(statev)

    #    self.assertEqual(sstr,'pthsk:3.000000 30.000000 0.000100 \npthsk:3.000000 40.000000 0.000100 \n')

    #    stv=sio.vector_from_string(sstr)
    #    self.assertAlmostEqual(stv[0].hbv_snow.sca,statev[0].hbv_snow.sca)
    #    self.assertAlmostEqual(stv[1].hbv_snow.swe,statev[1].hbv_snow.swe)

class MockRepositoryTestCase(unittest.TestCase):

    def test_put_entry(self):
        repository = MockRepository()
        repository.put("foobar", None)
        keys = repository.find()
        self.assertTrue("foobar" in keys)
        self.assertEqual(len(keys), 1)

    def test_get(self):
        repository = MockRepository()
        repository.put("foo", None)
        repository.put("bar", None)
        try:
            repository.get("foo")
            repository.get("bar")
        except RuntimeError:
            self.fail("Reading existing entry failed with RuntimeError")
        self.assertRaises(RuntimeError, repository.get, "spam")

    def test_delete(self):
        repository = MockRepository()
        entry = "Mockdata"
        repository.put("foo", entry)
        data = repository.delete("foo")
        self.assertEqual(data, entry)
        self.assertEqual(len(repository.find()), 0)
        self.assertRaises(RuntimeError, repository.delete, "foo")


class MockStateRepositoryTestCase(unittest.TestCase):

    def setUp(self):
        repository = MockStateRepository()
        repository.generate_mock_entry("ptgsk-state-0", "2014-09-30-0-0-0", tags=["unittest", "mock"])
        repository.generate_mock_entry("ptgsk-state-1", "2014-10-04-0-0-0", tags=["unittest", "foo"])
        self.repository = repository

    def test_compare(self):

        class mock_state(object):
            def __init__(self, utc_timestamp):
                self.utc_timestamp = utc_timestamp

        condition = 5 < TimeCondition()
        self.assertTrue(condition(mock_state(6)))
        self.assertFalse(condition(mock_state(5)))

        condition = 5 <= TimeCondition()
        self.assertTrue(condition(mock_state(5)))
        self.assertFalse(condition(mock_state(4)))

        condition = 5 > TimeCondition()
        self.assertTrue(condition(mock_state(4)))
        self.assertFalse(condition(mock_state(5)))

        condition = 5 >= TimeCondition()
        self.assertTrue(condition(mock_state(5)))
        self.assertFalse(condition(mock_state(6)))

        condition = combine_conditions(5 < TimeCondition(), TimeCondition() < 10)
        self.assertTrue(condition(mock_state(6)))
        self.assertFalse(condition(mock_state(5)))
        self.assertFalse(condition(mock_state(10)))

        condition = combine_conditions(5 <= TimeCondition(), TimeCondition() < 10)
        self.assertTrue(condition(mock_state(6)))
        self.assertTrue(condition(mock_state(5)))
        self.assertFalse(condition(mock_state(10)))


    def test_find_all_entries(self):
        keys = self.repository.find()
        self.assertEqual(len(keys), 2)

    def test_find_all_and_select_one_entry(self):
        keys = self.repository.find()
        try:
            self.repository.get(keys[-1])
        except RuntimeError:
            self.fail("Reading existing entry failed with RuntimeError")
        
    def test_find_with_simple_condition(self):
        condition = TimeCondition() > "2014-10-01-0-0-0"
        keys = self.repository.find(condition=condition)
        self.assertEqual(len(keys), 1)

    def test_find_with_combined_conditions(self):
        condition = combine_conditions( "2014-10-01-0-0-0" < TimeCondition(), TimeCondition() < "2014-10-02-0-0-0")
        keys = self.repository.find(condition=condition)
        self.assertEqual(len(keys), 0)

    def test_find_with_tags(self):
        keys = self.repository.find(tags=["unittest"])
        self.assertEqual(len(keys), 2)
        keys = self.repository.find(tags=["foo"])
        self.assertEqual(len(keys), 1)
        keys = self.repository.find(tags=["bar"])
        self.assertEqual(len(keys), 0)

    def test_combine_condition_and_tags(self):
        condition = TimeCondition() > "2014-10-01-0-0-0"
        keys = self.repository.find(condition=condition, tags=["foo"])
        self.assertEqual(len(keys), 1)
        keys = self.repository.find(condition=condition, tags=["mock"])
        self.assertEqual(len(keys), 0)


class MockInputSourceRepositoryTestCase(unittest.TestCase):
    
    def test_create_mock_station_data(self):
        data = create_mock_station_data(0, 3600, 24)
        self.assertEqual(len(data), 5)
        self.assertTrue("temperature" in data)
        self.assertTrue("precipitation" in data)
        self.assertTrue("wind_speed" in data)
        self.assertTrue("relative_humidity" in data)
        self.assertTrue("radiation" in data)

    def test_put_input_source(self):
        repository = MockInputSourceRepository()
        station = InputSource([0.0, 0.0, 0.0], create_mock_station_data(0, 3600, 100), tags=["synthetic_data"])
        repository.put("Hylen", station)
        self.assertTrue("Hylen" in repository.find())


class CellReadOnlyRepositoryTestCase(unittest.TestCase):

    def test_create_mock_cell_read_only_repository(self):
        config = {"x_min": 0,
                  "y_min": 0,
                  "dx": 1000,
                  "dy": 1000,
                  "n_x": 10,
                  "n_y": 10}
        
        class Config(object):
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        repository = CellReadOnlyRepository(**mock_cell_data(Config(**config)))


    def test_raster_cell_repository_construction(self):
        config_file = "../../doc/example/file_configs/region.yaml"
        with open(config_file, "r") as cf:
            config = yaml.load(cf.read())
        r = config['repository']
        data = r['constructor'][0](None, *r['constructor'][1:])
        self.assertTrue(isinstance(data, FileCellRepository))


class AromeDataRepositoryTestCase(unittest.TestCase):

    def test_create_reader(self):
        """
        Simple regression test of arome data respository.
        """
        from os.path import dirname
        from os.path import pardir
        from os.path import join
        from shyft import __file__ as shyft_file
        EPSG = 32633
        upper_left_x = 436100.0
        upper_left_y = 7417800.0
        nx = 74
        ny = 94
        dx = 1000.0
        dy = 1000.0
        base_dir = join(dirname(shyft_file), pardir, pardir, "shyft-data", "netcdf", "arome-testdata")
        pth1 = join(base_dir, "arome_metcoop_red_default2_5km_20150823_06.nc")
        pth2 = join(base_dir, "arome_metcoop_red_test2_5km_20150823_06.nc")
        bounding_box = ([upper_left_x, upper_left_x + nx*dx, upper_left_x + nx*dx, upper_left_x],
                        [upper_left_y, upper_left_y, upper_left_y - ny*dy, upper_left_y - ny*dy])
        ar1 = AromeDataRepository(pth1, EPSG, bounding_box)
        ar2 = AromeDataRepository(pth2, EPSG, bounding_box)
        ar1.add_time_series(ar2)
        sources = ar1.get_sources()
        self.assertTrue(len(sources) > 0)
        data_names = "temperature", "radiation", "wind_speed", "precipitation", "relative_humidity"
        self.assertTrue(all([n in sources for n in data_names]))
        self.assertTrue(sources["temperature"][0].ts.size() == 67)
        r0 = sources["radiation"][0].ts
        t0 = sources["temperature"][0].ts
        self.assertTrue(r0.size() == 66)
        self.assertTrue(r0.time(0) == t0.time(0))
        self.assertTrue(r0.time(r0.size()-1) < t0.time(t0.size()-1))
    

class LocalStateRepositoryTestCase(unittest.TestCase):

    def setUp(self):
        self.mock_state_repository = state_repository_factory({"t_start": 0, "num_cells": 10})
        self.state_dir = "."
        self.state_file_name = "state_test.yaml"

    def test_save(self):
        key = self.mock_state_repository.find()[0]
        save_state_as_yaml_file(self.mock_state_repository.get(key), os.path.join(self.state_dir, self.state_file_name))

    def test_load(self):
        key = self.mock_state_repository.find()[0]
        d1 = self.mock_state_repository.get(key)
        save_state_as_yaml_file(self.mock_state_repository.get(key), os.path.join(self.state_dir, self.state_file_name))
        state_repository = yaml_file_storage_factory({}, self.state_dir, "state_test.yaml")
        d2 = state_repository.get(state_repository.find()[0])
        self.assertDictEqual(d1.__dict__, d2.__dict__)

    #def test_speed_convert(self):
    #    state_repository = yaml_file_storage_factory({}, "D:/Users/sih/enki_config_for_test/states", "ptgsk_state.yaml")
    #    k0=state_repository.find()[0]
    #    s0=state_repository.get(k0)
    #    self.assertTrue(len(s0)>0,"Assume we got some states")


if __name__ == "__main__":
    unittest.main()
