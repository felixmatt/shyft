from numpy import random
import unittest
import tempfile
from os import path

from shyft import api
from shyft.api import pt_gs_k
from shyft.api import pt_ss_k
from shyft.api import pt_hs_k
from shyft.api import hbv_stack


class RegionModel(unittest.TestCase):
    @staticmethod
    def build_model(model_t, parameter_t, model_size, num_catchments=1):
        cell_area = 1000 * 1000
        region_parameter = parameter_t()
        gcds = api.GeoCellDataVector()  # creating models from geo_cell-data is easier and more flexible
        for i in range(model_size):
            gp = api.GeoPoint(500+ 1000.0*i,500.0, 500.0*i/model_size)
            cid = 0
            if num_catchments > 1:
                cid = random.randint(1, num_catchments + 1)
            geo_cell_data = api.GeoCellData(gp, cell_area, cid, 0.9, api.LandTypeFractions(0.01, 0.05, 0.19, 0.3, 0.45))
            geo_cell_data.land_type_fractions_info().set_fractions(glacier=0.01, lake=0.05, reservoir=0.19, forest=0.3)
            gcds.append(geo_cell_data)

        return model_t(gcds, region_parameter)

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
        pt.update({(k, v) for k, v in kwargs.items() if k in pt})
        gs.update({(k, v) for k, v in kwargs.items() if k in gs})
        kirchner.update({(k, v) for k, v in kwargs.items() if k in kirchner})
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

    def _create_constant_geo_ts(self, geo_ts_type, geo_point, utc_period, value):
        """Create a time point ts, with one value at the start
        of the supplied utc_period."""
        tv = api.UtcTimeVector()
        tv.push_back(utc_period.start)
        vv = api.DoubleVector()
        vv.push_back(value)
        cts = api.TsFactory().create_time_point_ts(utc_period, tv, vv, api.POINT_AVERAGE_VALUE)
        return geo_ts_type(geo_point, cts)

    def test_source_uid(self):
        cal = api.Calendar()
        time_axis = api.TimeAxisFixedDeltaT(cal.time(api.YMDhms(2015, 1, 1, 0, 0, 0)), api.deltahours(1), 240)
        mid_point = api.GeoPoint(1000, 1000, 100)
        precip_source = self._create_constant_geo_ts(api.PrecipitationSource, mid_point, time_axis.total_period(), 5.0)
        self.assertIsNotNone(precip_source.uid)
        precip_source.uid = 'abc'
        self.assertEqual(precip_source.uid, 'abc')

    def create_dummy_region_environment(self, time_axis, mid_point):
        re = api.ARegionEnvironment()
        re.precipitation.append(self._create_constant_geo_ts(api.PrecipitationSource, mid_point, time_axis.total_period(), 5.0))
        re.temperature.append(self._create_constant_geo_ts(api.TemperatureSource, mid_point, time_axis.total_period(), 10.0))
        re.wind_speed.append(self._create_constant_geo_ts(api.WindSpeedSource, mid_point, time_axis.total_period(), 2.0))
        re.rel_hum.append(self._create_constant_geo_ts(api.RelHumSource, mid_point, time_axis.total_period(), 0.7))
        re.radiation = api.RadiationSourceVector()  # just for testing BW compat
        re.radiation.append(self._create_constant_geo_ts(api.RadiationSource, mid_point, time_axis.total_period(), 300.0))
        return re

    def test_create_region_environment(self):
        cal = api.Calendar()
        time_axis = api.TimeAxisFixedDeltaT(cal.time(api.YMDhms(2015, 1, 1, 0, 0, 0)), api.deltahours(1), 240)
        re = self.create_dummy_region_environment(time_axis, api.GeoPoint(1000, 1000, 100))
        self.assertIsNotNone(re)
        self.assertEqual(len(re.radiation), 1)
        self.assertAlmostEqual(re.radiation[0].ts.value(0), 300.0)
        vv = re.radiation.values_at_time(time_axis.time(0))  # verify .values_at_time(t)
        self.assertEqual(len(vv), len(re.radiation))
        self.assertAlmostEqual(vv[0], 300.0)


    def verify_state_handler(self, model):
        cids_unspecified = api.IntVector()
        states = model.state.extract_state(cids_unspecified)
        self.assertEqual(len(states), model.size())
        unapplied_list = model.state.apply_state(states, cids_unspecified)
        self.assertEqual(len(unapplied_list), 0)

    def test_pt_ss_k_model_init(self):
        num_cells = 20
        model_type = pt_ss_k.PTSSKModel
        model = self.build_model(model_type, pt_ss_k.PTSSKParameter, num_cells)
        self.assertEqual(model.size(), num_cells)
        self.verify_state_handler(model)
        self.assertIsNotNone(model.skaugen_snow_response)
        self.assertIsNotNone(model.skaugen_snow_state)

    def test_pt_hs_k_model_init(self):
        num_cells = 20
        model_type = pt_hs_k.PTHSKModel
        model = self.build_model(model_type, pt_hs_k.PTHSKParameter, num_cells)
        self.assertEqual(model.size(), num_cells)
        self.verify_state_handler(model)

    def test_hbv_stack_model_init(self):
        num_cells = 20
        model_type = hbv_stack.HbvModel
        model = self.build_model(model_type, hbv_stack.HbvParameter, num_cells)
        self.assertEqual(model.size(), num_cells)
        self.verify_state_handler(model)

    def test_extract_geo_cell_data_vector(self):
        num_cells = 20
        model_type = hbv_stack.HbvModel
        model = self.build_model(model_type, hbv_stack.HbvParameter, num_cells)
        self.assertEqual(model.size(), num_cells)
        gcdv = model.extract_geo_cell_data()
        self.assertEqual(len(gcdv), num_cells)

    def test_model_area_functions(self):
        num_cells = 20
        model_type = pt_gs_k.PTGSKModel
        model = self.build_model(model_type, pt_gs_k.PTGSKParameter, num_cells)
        # demo how to get area statistics.
        cids = api.IntVector()
        total_area = model.statistics.total_area(cids)
        forest_area = model.statistics.forest_area(cids)
        glacier_area = model.statistics.glacier_area(cids)
        lake_area = model.statistics.lake_area(cids)
        reservoir_area = model.statistics.reservoir_area(cids)
        unspecified_area = model.statistics.unspecified_area(cids)
        self.assertAlmostEqual(total_area, forest_area + glacier_area + lake_area + reservoir_area + unspecified_area)
        cids.append(3)
        total_area_no_match = model.statistics.total_area(cids)  # now, cids contains 3, that matches no cells
        self.assertAlmostEqual(total_area_no_match, 0.0)

    def test_model_initialize_and_run(self):
        num_cells = 20
        model_type = pt_gs_k.PTGSKModel
        model = self.build_model(model_type, pt_gs_k.PTGSKParameter, num_cells)
        self.assertEqual(model.size(), num_cells)
        self.verify_state_handler(model)
        # demo of feature for threads
        self.assertGreaterEqual(model.ncore, 1)  # defaults to hardware concurrency
        model.ncore = 4  # set it to 4, and
        self.assertEqual(model.ncore, 4)  # verify it works

        # now modify snow_cv forest_factor to 0.1
        region_parameter = model.get_region_parameter()
        region_parameter.gs.snow_cv_forest_factor = 0.1
        region_parameter.gs.snow_cv_altitude_factor = 0.0001
        self.assertEqual(region_parameter.gs.snow_cv_forest_factor, 0.1)
        self.assertEqual(region_parameter.gs.snow_cv_altitude_factor, 0.0001)

        self.assertAlmostEqual(region_parameter.gs.effective_snow_cv(1.0, 0.0), region_parameter.gs.snow_cv + 0.1)
        self.assertAlmostEqual(region_parameter.gs.effective_snow_cv(1.0, 1000.0), region_parameter.gs.snow_cv + 0.1 + 0.1)
        cal = api.Calendar()
        time_axis = api.TimeAxisFixedDeltaT(cal.time(2015, 1, 1, 0, 0, 0), api.deltahours(1), 240)
        model_interpolation_parameter = api.InterpolationParameter()
        # degC/m, so -0.5 degC/100m
        model_interpolation_parameter.temperature_idw.default_temp_gradient = -0.005
        # if possible use closest neighbor points and solve gradient using equation,(otherwise default min/max height)
        model_interpolation_parameter.temperature_idw.gradient_by_equation = True
        # Max number of temperature sources used for one interpolation
        model_interpolation_parameter.temperature_idw.max_members = 6
        # 20 km is max distance
        model_interpolation_parameter.temperature_idw.max_distance = 20000
        # zscale is used to discriminate neighbors at different elevation than target point
        self.assertAlmostEqual(model_interpolation_parameter.temperature_idw.zscale, 1.0)
        model_interpolation_parameter.temperature_idw.zscale = 0.5
        self.assertAlmostEqual(model_interpolation_parameter.temperature_idw.zscale, 0.5)
        # Pure linear interpolation
        model_interpolation_parameter.temperature_idw.distance_measure_factor = 1.0
        # This enables IDW with default temperature gradient.
        model_interpolation_parameter.use_idw_for_temperature = True
        self.assertAlmostEqual(model_interpolation_parameter.precipitation.scale_factor, 1.02)  # just verify this one is as before change to scale_factor
        model.initialize_cell_environment(time_axis)  # just show how we can split the run_interpolation into two calls(second one optional)
        model.interpolate(
            model_interpolation_parameter,
            self.create_dummy_region_environment(time_axis,
                                                 model.get_cells()[int(num_cells / 2)].geo.mid_point()))
        m_ip_parameter = model.interpolation_parameter  # illustrate that we can get back the passed interpolation parameter as a property of the model
        self.assertEqual(m_ip_parameter.use_idw_for_temperature, True)  # just to ensure we really did get back what we passed in
        self.assertAlmostEqual(m_ip_parameter.temperature_idw.zscale, 0.5)
        s0 = pt_gs_k.PTGSKStateVector()
        for i in range(num_cells):
            si = pt_gs_k.PTGSKState()
            si.kirchner.q = 40.0
            s0.append(si)
        model.set_states(s0)
        model.set_state_collection(-1, True)  # enable state collection for all cells
        model2 = model_type(model)  # make a copy, so that we in the stepwise run below get a clean copy with all values zero.
        opt_model = pt_gs_k.create_opt_model_clone(model)  # this is how to make a model suitable for optimizer
        model.run_cells()  # the default arguments applies: thread_cell_count=0,start_step=0,n_steps=0)
        cids = api.IntVector()  # optional, we can add selective catchment_ids here
        sum_discharge = model.statistics.discharge(cids)
        sum_discharge_value = model.statistics.discharge_value(cids, 0)  # at the first timestep
        sum_charge = model.statistics.charge(cids)
        sum_charge_value=model.statistics.charge_value(cids, 0)
        opt_model.run_cells()  # starting out with the same state, same interpolated values, and region-parameters, we should get same results
        sum_discharge_opt_value= opt_model.statistics.discharge_value(cids, 0)
        self.assertAlmostEqual(sum_discharge_opt_value,sum_discharge_value,3)  # verify the opt_model clone gives same value
        self.assertGreaterEqual(sum_discharge_value, 130.0)
        opt_model.region_env.temperature[0].ts.set(0,23.2)  # verify that region-env is different (no aliasing, a true copy is required)
        self.assertFalse( abs( model.region_env.temperature[0].ts.value(0)-opt_model.region_env.temperature[0].ts.value(0)) >0.5)

        #
        # check values
        #
        self.assertIsNotNone(sum_discharge)
        # now, re-run the process in 24-hours steps x 10
        model.set_states(s0)  # restore state s0
        self.assertEqual(s0.size(), model.initial_state.size())
        for do_collect_state in [False,True]:
            model2.set_state_collection(-1,do_collect_state)  # issue reported by Yisak, prior to 21.3, this would crash
            model2.set_states(s0)
            # now  after fix, it works Ok
            for section in range(10):
                model2.run_cells(use_ncore=0, start_step=section * 24, n_steps=24)
                section_discharge = model2.statistics.discharge(cids)
                self.assertEqual(section_discharge.size(), sum_discharge.size())  # notice here that the values after current step are 0.0
        stepwise_sum_discharge = model2.statistics.discharge(cids)
        # assert stepwise_sum_discharge == sum_discharge
        diff_ts = sum_discharge.values.to_numpy() - stepwise_sum_discharge.values.to_numpy()
        self.assertAlmostEqual((diff_ts * diff_ts).max(), 0.0, 4)
        # Verify that if we pass in illegal cids, then it raises exception(with first failing
        try:
            illegal_cids = api.IntVector([0, 4, 5])
            model.statistics.discharge(illegal_cids)
            self.assertFalse(True, "Failed test, using illegal cids should raise exception")
        except RuntimeError as rte:
            pass

        avg_temperature = model.statistics.temperature(cids)
        avg_precipitation = model.statistics.precipitation(cids)
        self.assertIsNotNone(avg_precipitation)
        for time_step in range(time_axis.size()):
            precip_raster = model.statistics.precipitation(cids, time_step)  # example raster output
            self.assertEqual(precip_raster.size(), num_cells)
        # example single value spatial aggregation (area-weighted) over cids for a specific timestep
        avg_gs_sc_value = model.gamma_snow_response.sca_value(cids, 1)
        self.assertGreaterEqual(avg_gs_sc_value, 0.0)
        avg_gs_sca = model.gamma_snow_response.sca(cids)  # swe output
        self.assertIsNotNone(avg_gs_sca)
        # lwc surface_heat alpha melt_mean melt iso_pot_energy temp_sw
        avg_gs_albedo = model.gamma_snow_state.albedo(cids)
        self.assertIsNotNone(avg_gs_albedo)
        self.assertEqual(avg_temperature.size(), time_axis.size(), "expect results equal to time-axis size")
        copy_region_model = model.__class__(model)
        self.assertIsNotNone(copy_region_model)
        copy_region_model.run_cells()  # just to verify we can copy and run the new model
        #
        # Play with routing and river-network
        #
        # 1st: add a river, with 36.000 meter hydro length, a UHGParameter with 1m/hour speed, alpha/beta suitable
        model.river_network.add(
            api.River(1, api.RoutingInfo(0, 3000.0), api.UHGParameter(1 / 3.60, 7.0, 0.0)))  # river id =1
        # 2nd: let cells route to the river
        model.connect_catchment_to_river(0, 1)  # now all cells in catchment 0 routes to river with id 1.
        self.assertTrue(model.has_routing())
        # 3rd: now we can have a look at water coming in and out
        river_out_m3s = model.river_output_flow_m3s(1)  # should be delayed and reshaped
        river_local_m3s = model.river_local_inflow_m3s(
            1)  # should be equal to cell outputs (no routing stuff from cell to river)
        river_upstream_inflow_m3s = model.river_upstream_inflow_m3s(
            1)  # should be 0.0 in this case, since we do not have a routing network
        self.assertIsNotNone(river_out_m3s)
        self.assertAlmostEqual(river_out_m3s.value(8), 31.57297, 0)
        self.assertIsNotNone(river_local_m3s)
        self.assertIsNotNone(river_upstream_inflow_m3s)
        model.connect_catchment_to_river(0, 0)
        self.assertFalse(model.has_routing())

    def test_optimization_model(self):
        num_cells = 20
        model_type = pt_gs_k.PTGSKOptModel
        model = self.build_model(model_type, pt_gs_k.PTGSKParameter, num_cells)
        cal = api.Calendar()
        t0 = cal.time(2015, 1, 1, 0, 0, 0)
        dt = api.deltahours(1)
        n = 240
        time_axis = api.TimeAxisFixedDeltaT(t0, dt, n)
        model_interpolation_parameter = api.InterpolationParameter()
        model.initialize_cell_environment(time_axis)  # just show how we can split the run_interpolation into two calls(second one optional)
        model.interpolate(
            model_interpolation_parameter,
            self.create_dummy_region_environment(time_axis,
                                                 model.get_cells()[int(num_cells / 2)].geo.mid_point()))
        s0 = pt_gs_k.PTGSKStateVector()
        for i in range(num_cells):
            si = pt_gs_k.PTGSKState()
            si.kirchner.q = 40.0
            s0.append(si)
        model.set_states(s0)  # at this point the intial state of model is established as well
        model.run_cells()
        cids = api.IntVector.from_numpy([0])  # optional, we can add selective catchment_ids here
        sum_discharge = model.statistics.discharge(cids)
        sum_discharge_value = model.statistics.discharge_value(cids, 0)  # at the first timestep
        self.assertGreaterEqual(sum_discharge_value, 130.0)
        # verify we can construct an optimizer
        optimizer = model_type.optimizer_t(model)  # notice that a model type know it's optimizer type, e.g. PTGSKOptimizer
        self.assertIsNotNone(optimizer)
        #
        # create target specification
        #
        model.revert_to_initial_state()  # set_states(s0)  # remember to set the s0 again, so we have the same initial condition for our game
        tsa = api.TsTransform().to_average(t0, dt, n, sum_discharge)
        t_spec_1 = api.TargetSpecificationPts(tsa, cids, 1.0, api.KLING_GUPTA, 1.0, 0.0, 0.0, api.DISCHARGE, 'test_uid')

        target_spec = api.TargetSpecificationVector()
        target_spec.append(t_spec_1)
        upper_bound = model_type.parameter_t(model.get_region_parameter())  # the model_type know it's parameter_t
        lower_bound = model_type.parameter_t(model.get_region_parameter())
        upper_bound.kirchner.c1 = -1.9
        lower_bound.kirchner.c1 = -3.0
        upper_bound.kirchner.c2 = 0.99
        lower_bound.kirchner.c2 = 0.80

        optimizer.set_target_specification(target_spec, lower_bound, upper_bound)
        # Not needed, it will automatically get one.
        # optimizer.establish_initial_state_from_model()
        # s0_0 = optimizer.get_initial_state(0)
        # optimizer.set_verbose_level(1000)
        p0 = model_type.parameter_t(model.get_region_parameter())
        orig_c1 = p0.kirchner.c1
        orig_c2 = p0.kirchner.c2
        # model.get_cells()[0].env_ts.precipitation.set(0, 5.1)
        # model.get_cells()[0].env_ts.precipitation.set(1, 4.9)
        p0.kirchner.c1 = -2.4
        p0.kirchner.c2 = 0.91
        opt_param = optimizer.optimize(p0, 1500, 0.1, 1e-5)
        goal_fx = optimizer.calculate_goal_function(opt_param)
        p0.kirchner.c1 = -2.4
        p0.kirchner.c2 = 0.91
        # goal_fx1 = optimizer.calculate_goal_function(p0)

        self.assertLessEqual(goal_fx, 10.0)
        self.assertAlmostEqual(orig_c1, opt_param.kirchner.c1, 4)
        self.assertAlmostEqual(orig_c2, opt_param.kirchner.c2, 4)

    def test_hbv_model_initialize_and_run(self):
        num_cells = 20
        model_type = hbv_stack.HbvModel
        model = self.build_model(model_type, hbv_stack.HbvParameter, num_cells)
        self.assertEqual(model.size(), num_cells)
        opt_model = model.create_opt_model_clone()
        self.assertIsNotNone(opt_model)
        # now modify snow_cv forest_factor to 0.1
        region_parameter = model.get_region_parameter()
        # region_parameter.gs.snow_cv_forest_factor = 0.1
        # region_parameter.gs.snow_cv_altitude_factor = 0.0001
        # self.assertEqual(region_parameter.gs.snow_cv_forest_factor, 0.1)
        # self.assertEqual(region_parameter.gs.snow_cv_altitude_factor, 0.0001)

        # self.assertAlmostEqual(region_parameter.gs.effective_snow_cv(1.0, 0.0), region_parameter.gs.snow_cv + 0.1)
        # self.assertAlmostEqual(region_parameter.gs.effective_snow_cv(1.0, 1000.0), region_parameter.gs.snow_cv + 0.1 + 0.1)
        cal = api.Calendar()
        time_axis = api.TimeAxisFixedDeltaT(cal.time(2015, 1, 1, 0, 0, 0), api.deltahours(1), 240)
        model_interpolation_parameter = api.InterpolationParameter()
        # degC/m, so -0.5 degC/100m
        model_interpolation_parameter.temperature_idw.default_temp_gradient = -0.005
        # if possible use closest neighbor points and solve gradient using equation,(otherwise default min/max height)
        model_interpolation_parameter.temperature_idw.gradient_by_equation = True
        # Max number of temperature sources used for one interpolation
        model_interpolation_parameter.temperature_idw.max_members = 6
        # 20 km is max distance
        model_interpolation_parameter.temperature_idw.max_distance = 20000
        # zscale is used to discriminate neighbors at different elevation than target point
        self.assertAlmostEqual(model_interpolation_parameter.temperature_idw.zscale, 1.0)
        model_interpolation_parameter.temperature_idw.zscale = 0.5
        self.assertAlmostEqual(model_interpolation_parameter.temperature_idw.zscale, 0.5)
        # Pure linear interpolation
        model_interpolation_parameter.temperature_idw.distance_measure_factor = 1.0
        # This enables IDW with default temperature gradient.
        model_interpolation_parameter.use_idw_for_temperature = True
        self.assertAlmostEqual(model_interpolation_parameter.precipitation.scale_factor, 1.02)  # just verify this one is as before change to scale_factor
        model.run_interpolation(
            model_interpolation_parameter, time_axis,
            self.create_dummy_region_environment(time_axis,
                                                 model.get_cells()[int(num_cells / 2)].geo.mid_point()))
        s0 = hbv_stack.HbvStateVector()
        for i in range(num_cells):
            si = hbv_stack.HbvState()
            si.tank.uz = 40.0
            si.tank.lz = 40.0
            s0.append(si)
        model.set_states(s0)
        model.set_state_collection(-1, True)  # enable state collection for all cells
        model.run_cells()
        cids = api.IntVector()  # optional, we can add selective catchment_ids here
        sum_discharge = model.statistics.discharge(cids)
        sum_discharge_value = model.statistics.discharge_value(cids, 0)  # at the first timestep
        self.assertGreaterEqual(sum_discharge_value, 32.0)
        self.assertIsNotNone(sum_discharge)
        # Verify that if we pass in illegal cids, then it raises exception(with first failing
        try:
            illegal_cids = api.IntVector([0, 4, 5])
            model.statistics.discharge(illegal_cids)
            self.assertFalse(True, "Failed test, using illegal cids should raise exception")
        except RuntimeError as rte:
            pass

        avg_temperature = model.statistics.temperature(cids)
        avg_precipitation = model.statistics.precipitation(cids)
        self.assertIsNotNone(avg_precipitation)
        for time_step in range(time_axis.size()):
            precip_raster = model.statistics.precipitation(cids, time_step)  # example raster output
            self.assertEqual(precip_raster.size(), num_cells)
        # example single value spatial aggregation (area-weighted) over cids for a specific timestep
        # avg_gs_sc_value = model.gamma_snow_response.sca_value(cids, 1)
        # self.assertGreaterEqual(avg_gs_sc_value,0.0)
        # avg_gs_sca = model.gamma_snow_response.sca(cids)  # swe output
        # self.assertIsNotNone(avg_gs_sca)
        # lwc surface_heat alpha melt_mean melt iso_pot_energy temp_sw
        # avg_gs_albedo = model.gamma_snow_state.albedo(cids)
        # self.assertIsNotNone(avg_gs_albedo)
        self.assertEqual(avg_temperature.size(), time_axis.size(), "expect results equal to time-axis size")
        copy_region_model = model.__class__(model)
        self.assertIsNotNone(copy_region_model)
        copy_region_model.run_cells()  # just to verify we can copy and run the new model

    def test_model_state_io(self):
        num_cells = 2
        for model_type in [pt_gs_k.PTGSKModel, pt_gs_k.PTGSKOptModel]:
            model = self.build_model(model_type, pt_gs_k.PTGSKParameter, num_cells)
            state_list = []
            x = ""
            for i in range(num_cells):
                state_list.append(self.build_mock_state_dict(q=(i + 1) * 0.5 / num_cells))
            initial_states = x.join(state_list)
            sio = model_type.state_t.serializer_t()
            state_vector = sio.vector_from_string(initial_states)
            model.set_states(state_vector)
            m_state_vector = model_type.state_t.vector_t()
            model.get_states(m_state_vector)
            retrieved_states = sio.to_string(m_state_vector)
            self.assertEqual(initial_states, retrieved_states)

    def test_set_too_few_model_states(self):
        num_cells = 20
        for model_type in [pt_gs_k.PTGSKModel, pt_gs_k.PTGSKOptModel]:
            model = self.build_model(model_type, pt_gs_k.PTGSKParameter, num_cells)

            states = []
            x = ""
            for i in range(num_cells - 1):
                states.append(self.build_mock_state_dict(q=(i + 1) * 0.5 / num_cells))
            statestr = x.join(states)
            sio = model_type.state_t.serializer_t()
            state_vector = sio.vector_from_string(statestr)

            self.assertRaises(RuntimeError, model.set_states, state_vector)
            for i in range(num_cells + 1):
                states.append(self.build_mock_state_dict(q=(i + 1) * 0.5 / num_cells))
            statestr = x.join(states)
            state_vector = sio.vector_from_string(statestr)

            self.assertRaises(RuntimeError, model.set_states, state_vector)

    def test_geo_cell_data_serializer(self):
        """
        This test the bulding block for the geo-cell caching mechanism that can be
        implemented in GeoCell repository to cache complex information from the GIS system.
        The test illustrates how to convert existing cell-vector geo info into a DoubleVector,
        that can be converted .to_nump(),
        and then how to re-create the cell-vector,(of any given type actually) based on
        the geo-cell data DoubleVector (that can be created from .from_numpy(..)

        Notice that the from_numpy(np array) could have limited functionality when it comes
        to strides etc, so if problem flatten out the np.array before passing it.

        """
        n_cells = 3
        n_values_pr_gcd = 11  # number of values in a geo_cell_data stride
        model = self.build_model(pt_gs_k.PTGSKModel, pt_gs_k.PTGSKParameter, n_cells)
        cell_vector = model.get_cells()
        geo_cell_data_vector = cell_vector.geo_cell_data_vector(cell_vector)  # This gives a string, ultra fast, containing the serialized form of all geo-cell data
        self.assertEqual(len(geo_cell_data_vector), n_values_pr_gcd * n_cells)
        cell_vector2 = pt_gs_k.PTGSKCellAllVector.create_from_geo_cell_data_vector(
            geo_cell_data_vector)  # This gives a cell_vector, of specified type, with exactly the same geo-cell data as the original
        self.assertEqual(len(cell_vector), len(cell_vector2))  # just verify equal size, and then geometry, the remaining tests are covered by C++ testing
        for i in range(len(cell_vector)):
            self.assertAlmostEqual(cell_vector[i].geo.mid_point().z, cell_vector2[i].mid_point().z)
            self.assertAlmostEqual(cell_vector[i].geo.mid_point().x, cell_vector2[i].mid_point().x)
            self.assertAlmostEqual(cell_vector[i].geo.mid_point().y, cell_vector2[i].mid_point().y)

    def test_state_with_id_handler(self):
        num_cells = 20
        model_type = pt_gs_k.PTGSKModel
        model = self.build_model(model_type, pt_gs_k.PTGSKParameter, num_cells, 2)
        cids_unspecified = api.IntVector()
        cids_1 = api.IntVector([1])
        cids_2 = api.IntVector([2])

        model_state_12 = model.state.extract_state(cids_unspecified)  # this is how to get all states from model
        model_state_1 = model.state.extract_state(cids_1)  # this is how to get only specified states from model
        model_state_2 = model.state.extract_state(cids_2)
        self.assertEqual(len(model_state_1) + len(model_state_2), len(model_state_12))
        self.assertGreater(len(model_state_1), 0)
        self.assertGreater(len(model_state_2), 0)
        for i in range(len(model_state_1)):  # verify selective extract catchment 1
            self.assertEqual(model_state_1[i].id.cid, 1)
        for i in range(len(model_state_2)):  # verify selective extract catchment 2
            self.assertEqual(model_state_2[i].id.cid, 2)
        for i in range(len(model_state_12)):
            model_state_12[i].state.kirchner.q = 100 + i
        model.state.apply_state(model_state_12, cids_unspecified)  # this is how to put all states into  model
        ms_12 = model.state.extract_state(cids_unspecified)
        for i in range(len(ms_12)):
            self.assertAlmostEqual(ms_12[i].state.kirchner.q, 100 + i)
        for i in range(len(model_state_2)):
            model_state_2[i].state.kirchner.q = 200 + i
        unapplied = model.state.apply_state(model_state_2, cids_2)  # this is how to put a limited set of state into model
        self.assertEqual(len(unapplied), 0)
        ms_12 = model.state.extract_state(cids_unspecified)
        for i in range(len(ms_12)):
            if ms_12[i].id.cid == 1:
                self.assertAlmostEqual(ms_12[i].state.kirchner.q, 100 + i)

        ms_2 = model.state.extract_state(cids_2)
        for i in range(len(ms_2)):
            self.assertAlmostEqual(ms_2[i].state.kirchner.q, 200 + i)

        # serialization support, to and from bytes

        bytes = ms_2.serialize_to_bytes()  # first make some bytes out of the state
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = str(path.join(tmpdirname, "pt_gs_k_state_test.bin"))
            api.byte_vector_to_file(file_path, bytes)  # stash it into a file
            bytes = api.byte_vector_from_file(file_path)  # get it back from the file and into ByteVector
        ms_2x = pt_gs_k.deserialize_from_bytes(bytes)  # then restore it from bytes to a StateWithIdVector

        self.assertIsNotNone(ms_2x)
        for i in range(len(ms_2x)):
            self.assertAlmostEqual(ms_2x[i].state.kirchner.q, 200 + i)


if __name__ == "__main__":
    unittest.main()
