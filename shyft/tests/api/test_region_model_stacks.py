﻿from __future__ import print_function
from numpy import random
import unittest

from shyft import api
from shyft.api import pt_gs_k
from shyft.api import pt_ss_k


class BasicModelCreateAndRun(unittest.TestCase):

    @staticmethod
    def build_model(model_t,parameter_t, model_size, num_catchments=1):

        cells = model_t.cell_t.vector_t()
        cell_area = 1000*1000
        region_parameter = parameter_t()
        for i in xrange(model_size):
            loc = (10000*random.random(2)).tolist() + \
                (500*random.random(1)).tolist()
            gp = api.GeoPoint(*loc)
            geo_cell_data = api.GeoCellData(gp, cell_area,
                                            random.randint(0, num_catchments))
            cell = model_t.cell_t()
            cell.geo = geo_cell_data
            cells.append(cell)
        return model_t(cells, region_parameter)

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
              "temp_swe": 0.0}
        kirchner = {"q": 0.25}
        pt.update({(k, v) for k, v in kwargs.iteritems() if k in pt})
        gs.update({(k, v) for k, v in kwargs.iteritems() if k in gs})
        kirchner.update({(k, v) for k, v in kwargs.iteritems()
                         if k in kirchner})
        state = pt_gs_k.PTGSKState()
        state.gs.albedo = gs["albedo"]
        state.gs.lwc = gs["lwc"]
        state.gs.surface_heat = gs["surface_heat"]
        state.gs.alpha = gs["alpha"]
        state.gs.sdc_melt_mean = gs["sdc_melt_mean"]
        state.gs.acc_melt = gs["acc_melt"]
        state.gs.iso_pot_energy = gs["iso_pot_energy"]
        state.gs.temp_swe = gs["temp_swe"]
        state.kirchner.q = kirchner["q"]
        return pt_gs_k.PTGSKStateIo().to_string(state)

    def _create_constant_geo_ts(self, geoTsType, geo_point, utc_period, value):
        """Create a time point ts, with one value at the start
        of the supplied utc_period."""
        tv = api.UtcTimeVector()
        tv.push_back(utc_period.start)
        vv = api.DoubleVector()
        vv.push_back(value)
        cts = api.TsFactory().create_time_point_ts(utc_period, tv, vv)
        return geoTsType(geo_point, cts)

    def create_dummy_region_environment(self, time_axis, mid_point):
        re = api.ARegionEnvironment()
        re.precipitation = api.PrecipitationSourceVector()
        re.precipitation.append(self._create_constant_geo_ts(
            api.PrecipitationSource, mid_point, time_axis.total_period(), 5.0))

        re.temperature = api.TemperatureSourceVector()
        re.temperature.append(self._create_constant_geo_ts(
            api.TemperatureSource, mid_point, time_axis.total_period(), 10.0))

        re.wind_speed = api.WindSpeedSourceVector()
        re.wind_speed.append(self._create_constant_geo_ts(
            api.WindSpeedSource, mid_point, time_axis.total_period(), 2.0))

        re.rel_hum = api.RelHumSourceVector()
        re.rel_hum.append(self._create_constant_geo_ts(
            api.RelHumSource, mid_point, time_axis.total_period(), 0.7))

        re.radiation = api.RadiationSourceVector()
        re.radiation.append(self._create_constant_geo_ts(
            api.RadiationSource, mid_point, time_axis.total_period(), 300.0))
        return re
    def test_pt_ss_k_model_init(self):
        num_cells = 20
        model_type = pt_ss_k.PTSSKModel
        model = self.build_model(model_type,pt_ss_k.PTSSKParameter, num_cells)
        self.assertEqual(model.size(), num_cells)
        
    def test_model_initialize_and_run(self):
        num_cells = 20
        model_type = pt_gs_k.PTGSKModel
        model = self.build_model(model_type,pt_gs_k.PTGSKParameter, num_cells)
        self.assertEqual(model.size(), num_cells)
        cal = api.Calendar()
        time_axis = api.Timeaxis(cal.time(api.YMDhms(2015, 1, 1, 0, 0, 0)),
                                 api.deltahours(1), 240)
        model_interpolation_parameter = api.InterpolationParameter()
        # degC/m, so -0.5 degC/100m
        model_interpolation_parameter.temperature_idw.default_temp_gradient = -0.005
        # Max number of temperature sources used for one interpolation
        model_interpolation_parameter.temperature_idw.max_members = 6
        # 20 km is max distance
        model_interpolation_parameter.temperature_idw.max_distance = 20000
        # Pure linear interpolation
        model_interpolation_parameter.temperature_idw.distance_measure_factor = 1.0
        # This enables IDW with default temperature gradient.
        model_interpolation_parameter.use_idw_for_temperature = True

        model.run_interpolation(
            model_interpolation_parameter, time_axis,
            self.create_dummy_region_environment(time_axis,
                                                 model.get_cells()[num_cells/2].geo.mid_point()))
        model.set_state_collection(-1, True)  # enable state collection for all cells
        model.run_cells()
        cids = api.IntVector()  # optional, we can add selective catchment_ids here
        sum_discharge = model.statistics.discharge(cids)
        self.assertIsNotNone(sum_discharge)
        avg_temperature = model.statistics.temperature(cids)
        avg_precipitation = model.statistics.precipitation(cids)
        self.assertIsNotNone(avg_precipitation)
        for time_step in xrange(time_axis.size()):
            precip_raster = model.statistics.precipitation(cids, time_step)  # example raster output
            self.assertEquals(precip_raster.size(), num_cells)
        avg_gs_sca = model.gamma_snow_response.sca(cids)  # swe output
        self.assertIsNotNone(avg_gs_sca)
        # lwc surface_heat alpha melt_mean melt iso_pot_energy temp_sw
        avg_gs_albedo = model.gamma_snow_state.albedo(cids)
        self.assertIsNotNone(avg_gs_albedo)
        self.assertEqual(avg_temperature.size(), time_axis.size(),
                         "expect results equal to time-axis size")

    def test_model_state_io(self):
        num_cells = 2
        for model_type in [pt_gs_k.PTGSKModel, pt_gs_k.PTGSKOptModel]:
            model = self.build_model(model_type,pt_gs_k.PTGSKParameter, num_cells)
            state_list = []
            x = ""
            for i in xrange(num_cells):
                state_list.append(self.build_mock_state_dict(q=(i + 1)*0.5/num_cells))
            initial_states = x.join(state_list)
            sio=model_type.state_t.serializer_t()
            state_vector=sio.vector_from_string(initial_states)
            model.set_states(state_vector)
            m_state_vector=model_type.state_t.vector_t()
            model.get_states(m_state_vector)
            retrieved_states=sio.to_string(m_state_vector)
            self.assertEqual(initial_states,retrieved_states)

    def test_set_too_few_model_states(self):
        num_cells = 20
        for model_type in [pt_gs_k.PTGSKModel, pt_gs_k.PTGSKOptModel]:
            model = self.build_model(model_type, pt_gs_k.PTGSKParameter,num_cells)
            states = []
            x = ""
            for i in xrange(num_cells - 1):
                states.append(self.build_mock_state_dict(q=(i + 1)*0.5/num_cells))
            statestr = x.join(states)
            sio=model_type.state_t.serializer_t()
            state_vector=sio.vector_from_string(statestr)

            self.assertRaises(RuntimeError, model.set_states,state_vector)
            for i in xrange(num_cells + 1):
                states.append(self.build_mock_state_dict(q=(i + 1)*0.5/num_cells))
            statestr = x.join(states)
            state_vector=sio.vector_from_string(statestr)

            self.assertRaises(RuntimeError, model.set_states,state_vector)





if __name__ == "__main__":
    unittest.main()
