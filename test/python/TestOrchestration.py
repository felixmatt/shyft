from __future__ import print_function
from numpy import random
from datetime import datetime
import unittest
import yaml
from os import path

from shyft import shyftdata_dir
from shyft import api
from shyft.api import pt_gs_k
from shyft.orchestration.state import set_ptgsk_model_state
from shyft.orchestration.state import extract_ptgsk_model_state
from shyft.orchestration.state import State
from shyft.repository.netcdf.arome_data_repository import AromeDataRepository


class StateIOTestCase(unittest.TestCase):

    @staticmethod
    def build_model(model_t, model_size, num_catchments=1):

        cells = model_t.cell_t.vector_t()
        cell_area = 1000*1000
        region_parameter = pt_gs_k.PTGSKParameter()
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

    def test_model_initialize_and_run(self):
        num_cells = 20
        model_type = pt_gs_k.PTGSKModel
        model = self.build_model(model_type, num_cells)
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
            model = self.build_model(model_type, num_cells)
            state_list = []
            x = ""
            for i in xrange(num_cells):
                state_list.append(self.build_mock_state_dict(q=(i + 1)*0.5/num_cells))
            initial_states = x.join(state_list)
            set_ptgsk_model_state(model, State(initial_states,
                                               datetime.strftime(datetime.utcnow(),
                                                                 "%Y-%m-%d-%M-%S")))
            retrieved_states = extract_ptgsk_model_state(model)
            self.assertEqual(initial_states, retrieved_states.state_list)

            # Test that the state can be serialized and de-serialized:
            serialized_states = yaml.dump(retrieved_states, default_flow_style=False)
            self.assertTrue(isinstance(serialized_states, str))
            deserialized_states = yaml.load(serialized_states)

            self.assertEqual(retrieved_states.state_list, deserialized_states.state_list)

            # Finally, set the deserialized states into the model:
            set_ptgsk_model_state(model, deserialized_states)

    def test_set_too_few_model_states(self):
        num_cells = 20
        for model_type in [pt_gs_k.PTGSKModel, pt_gs_k.PTGSKOptModel]:
            model = self.build_model(model_type, num_cells)
            states = []
            x = ""
            for i in xrange(num_cells - 1):
                states.append(self.build_mock_state_dict(q=(i + 1)*0.5/num_cells))
            statestr = x.join(states)
            self.assertRaises(RuntimeError, set_ptgsk_model_state, model, State(statestr))
            for i in xrange(num_cells + 1):
                states.append(self.build_mock_state_dict(q=(i + 1)*0.5/num_cells))
            statestr = x.join(states)
            self.assertRaises(RuntimeError, set_ptgsk_model_state, model, State(statestr))


class AromeDataRepositoryTestCase(unittest.TestCase):

    def test_get_timeseries(self):
        """
        Simple regression test of arome data respository.
        """
        EPSG = 32633
        upper_left_x = 436100.0
        upper_left_y = 7417800.0
        nx = 74
        ny = 94
        dx = 1000.0
        dy = 1000.0

        # Period start
        year = 2015
        month = 8
        day = 23
        hour = 6
        n_hours = 30
        date_str = "{}{:02}{:02}_{:02}".format(year, month, day, hour)
        utc = api.Calendar()  # No offset gives Utc
        t0 = api.YMDhms(year, month, day, hour)
        period = api.UtcPeriod(utc.time(t0), utc.time(t0) + api.deltahours(n_hours))

        base_dir = path.join(shyftdata_dir, "repository", "arome_data_repository")
        f1 = "arome_metcoop_red_default2_5km_{}.nc".format(date_str)
        f2 = "arome_metcoop_red_test2_5km_{}.nc".format(date_str)

        bbox = ([upper_left_x, upper_left_x + nx*dx,
                 upper_left_x + nx*dx, upper_left_x],
                [upper_left_y, upper_left_y,
                 upper_left_y - ny*dy, upper_left_y - ny*dy])
        ar1 = AromeDataRepository(EPSG, period, base_dir, filename=f1, bounding_box=bbox)
        ar2 = AromeDataRepository(EPSG, period, base_dir, filename=f2, elevation_file=f1)
        ar1_data_names = ("temperature", "wind_speed", "precipitation", "relative_humidity")
        ar2_data_names = ("radiation",)
        sources = ar1.get_timeseries(ar1_data_names, period, None)
        self.assertTrue(len(sources) > 0)
        sources2 = ar2.get_timeseries(ar2_data_names, period, geo_location_criteria=bbox)

        self.assertTrue(set(sources) == set(ar1_data_names))
        self.assertTrue(set(sources2) == set(ar2_data_names))
        self.assertTrue(sources["temperature"][0].ts.size() == n_hours + 1)
        r0 = sources2["radiation"][0].ts
        p0 = sources["precipitation"][0].ts
        temp0 = sources["temperature"][0].ts
        self.assertTrue(r0.size() == n_hours + 1)
        self.assertTrue(p0.size() == n_hours + 1)
        self.assertTrue(r0.time(0) == temp0.time(0))
        self.assertTrue(p0.time(0) == temp0.time(0))
        self.assertTrue(r0.time(r0.size() - 1) == temp0.time(temp0.size() - 1))
        self.assertTrue(p0.time(r0.size() - 1) == temp0.time(temp0.size() - 1))

    def test_get_forecast(self):
        EPSG = 32633
        upper_left_x = 436100.0
        upper_left_y = 7417800.0
        nx = 74
        ny = 94
        dx = 1000.0
        dy = 1000.0

        # Period start
        year = 2015
        month = 8
        day = 23
        hour = 6
        n_hours = 30
        utc = api.Calendar()  # No offset gives Utc
        t0 = api.YMDhms(year, month, day, hour)
        period = api.UtcPeriod(utc.time(t0), utc.time(t0) + api.deltahours(n_hours))
        t_c1 = utc.time(t0) + api.deltahours(1)
        t_c2 = utc.time(t0) + api.deltahours(7)

        base_dir = path.join(shyftdata_dir, "repository", "arome_data_repository")
        pattern = "arome_metcoop_red_default2_5km_*.nc"
        bbox = ([upper_left_x, upper_left_x + nx*dx,
                 upper_left_x + nx*dx, upper_left_x],
                [upper_left_y, upper_left_y,
                 upper_left_y - ny*dy, upper_left_y - ny*dy])
        repos = AromeDataRepository(EPSG, period, base_dir, filename=pattern, bounding_box=bbox)
        data_names = ("temperature", "wind_speed", "precipitation", "relative_humidity")
        tc1_sources = repos.get_forecast(data_names, period, t_c1, None)
        tc2_sources = repos.get_forecast(data_names, period, t_c2, None)

        self.assertTrue(len(tc1_sources) == len(tc2_sources))
        self.assertTrue(set(tc1_sources) == set(data_names))
        self.assertTrue(tc1_sources["temperature"][0].ts.size() == n_hours + 1)

        tc1_precip = tc1_sources["precipitation"][0].ts
        tc2_precip = tc2_sources["precipitation"][0].ts

        self.assertTrue(tc1_precip.size() == n_hours + 1)
        self.assertTrue(tc1_precip.time(0) != tc2_precip.time(0))

    def test_get_ensemble(self):
        EPSG = 32633
        upper_left_x = 436100.0
        upper_left_y = 7417800.0
        nx = 74
        ny = 94
        dx = 1000.0
        dy = 1000.0

        # Period start
        year = 2015
        month = 7
        day = 26
        hour = 0
        n_hours = 30
        utc = api.Calendar()  # No offset gives Utc
        t0 = api.YMDhms(year, month, day, hour)
        period = api.UtcPeriod(utc.time(t0), utc.time(t0) + api.deltahours(n_hours))
        t_c = utc.time(t0) + api.deltahours(1)

        base_dir = path.join(shyftdata_dir, "netcdf", "arome")
        pattern = "fc*.nc"
        bbox = ([upper_left_x, upper_left_x + nx*dx,
                 upper_left_x + nx*dx, upper_left_x],
                [upper_left_y, upper_left_y,
                 upper_left_y - ny*dy, upper_left_y - ny*dy])
        repos = AromeDataRepository(EPSG, period, base_dir, filename=pattern, bounding_box=bbox)
        data_names = ("temperature", "wind_speed", "relative_humidity")
        ensemble = repos.get_forecast_ensemble(data_names, period, t_c, None)
        self.assertTrue(isinstance(ensemble, list))
        self.assertEqual(len(ensemble), 10)


if __name__ == "__main__":
    unittest.main()
