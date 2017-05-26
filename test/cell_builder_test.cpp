#include "test_pch.h"
#include "core/experimental.h"
#include "core/model_calibration.h"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include <boost/filesystem.hpp>

// Figure out the complete path based on rel_path to the shyft/test directory
using namespace std;

namespace shyfttest {

	using namespace shyft::experimental;
	using namespace shyft::experimental::io;

	/** \brief scratch class for keeping the information from files, so that we can
	 * process it into shyft-usable types.
	 * TODO: make sure that we let the io-adapters provide the types we need directly.
	 *
	 *
	 */
	struct region_test_data {
		multi_polygon_ forest;
		multi_polygon_ glacier;
		multi_polygon_ lake;
		multi_point_   rsv_points;
		vector<dtm>    dtmv;
		map<int, multi_polygon_> catchment_map;///< map of catchment_id .. multipolygon_
		map<int, observation_location> location_map; ///< map of location_id|geo_id and observation_location ( id, name, position, remember?)
		shared_ptr<vector<geo_xts_t>> temperatures;///<  geo located temperatures, observation/forecast [degC]
		shared_ptr<vector<geo_xts_t>> precipitations;///< geo located precipitations, [mm/h]
		shared_ptr<vector<geo_xts_t>> radiations; ///< geo located radiations,  [W/m^2]
		shared_ptr<vector<geo_xts_t>> discharges;///< geo|id located discharges from catchments,  [m3/s] .. not really geo located, it's catchment id associated.. but for test/mock that is ok for now

		region_environment_t get_region_environment()  {
			region_environment_t re;
			re.temperature = temperatures;
			re.precipitation = precipitations;
			re.radiation = radiations;
			return re;
		}
	private:

	};

	region_test_data load_test_data_from_files(wkt_reader& wkt_io)  {
		region_test_data r;
		r.forest = wkt_io.read("forest", slurp(test_path("neanidelv/landtype_forest_wkt.txt")));
		r.glacier = wkt_io.read("glacier", slurp(test_path("neanidelv/landtype_glacier_wkt.txt")));
		r.lake = wkt_io.read("lake", slurp(test_path("neanidelv/landtype_lake_wkt.txt")));
		r.rsv_points = wkt_io.read_points("rsv_mid_points", slurp(test_path("neanidelv/landtype_rsv_midpoints_wkt.txt")));
		r.dtmv = wkt_io.read_dtm("dtm", slurp(test_path("neanidelv/dtm_xtra.txt")));
		r.catchment_map = wkt_io.read_catchment_map("catchment_map", slurp(test_path("neanidelv/catchments_wkt.txt")));
		r.location_map = wkt_io.read_geo_point_map("met_stations", slurp(test_path("neanidelv/geo_point_map.txt")));
		function<ec::geo_point(int)> geo_map;
		geo_map = [&r](int id) {return r.location_map[id].point; };
		r.temperatures = load_from_directory(wkt_io, geo_map, "neanidelv", "temperature");
		r.precipitations = load_from_directory(wkt_io, geo_map, "neanidelv", "precipitation");
		r.discharges = load_from_directory(wkt_io, geo_map, "neanidelv", "discharge");
		r.radiations = load_from_directory(wkt_io, geo_map, "neanidelv", "radiation");
		return r;
	}

}
TEST_SUITE("cell_builder") {
TEST_CASE("cell_builder_test::test_read_geo_region_data_from_files") {
	using namespace shyft::experimental;
	using namespace shyfttest;
	TS_TRACE("geo testing not activated yet, experimental");
	return;
	wkt_reader wkt_io;
	region_test_data t(load_test_data_from_files(wkt_io));


	TS_ASSERT(t.dtmv.size() > 0);
	TS_ASSERT(t.rsv_points->size() > 0);
	TS_ASSERT(t.forest->size() > 0);
	TS_ASSERT(t.lake->size() > 0);
	TS_ASSERT(t.catchment_map.size() > 0);
	TS_ASSERT(t.location_map.size() >= 0);
	TS_ASSERT(t.temperatures->size() >= 0);
	TS_ASSERT(t.precipitations->size() > 0);
	TS_ASSERT(t.discharges->size() > 0);
	if (wkt_io.reported_errors.size()) {
		TS_TRACE("Some reported errors on the test data read from file");
		cerr << "Reported errors on test-data:" << endl;
		for (auto e : wkt_io.reported_errors) {
			cerr << e << endl;
		}
	}


}

TEST_CASE("cell_builder_test::test_read_geo_point_map") {
	using namespace shyft::experimental;
	using namespace shyfttest;
	namespace ec = shyft::core;
	wkt_reader wio;
	map<int, observation_location> obs;
	obs = wio.read_geo_point_map("met_stations", slurp(test_path("neanidelv/geo_point_map.txt")));
	TS_ASSERT_DIFFERS(obs.size(), 0u);
	for (const auto& kv : obs) {
		TS_ASSERT_DIFFERS(kv.first, 0);
		TS_ASSERT_DIFFERS(kv.second.point.x, 0.0);
		TS_ASSERT_DIFFERS(kv.second.point.y, 0.0);
		TS_ASSERT_DIFFERS(kv.second.point.z, 0.0);

	}
}

TEST_CASE("cell_builder_test::test_read_geo_located_ts") {
	using namespace shyft::experimental;
	using namespace shyfttest;
	using namespace shyft::core;
	wkt_reader wio;
	map<int, observation_location> obs;
	obs = wio.read_geo_point_map("met_stations", slurp(test_path("neanidelv/geo_point_map.txt")));

	geo_xts_t tmp = wio.read_geo_xts_t(
		"gts.121.temperature",
		[&obs](int id) {return obs[id].point; },
		slurp(test_path("neanidelv/gts.121.temperature")));
	TS_ASSERT_LESS_THAN(geo_point::distance2(tmp.location, obs[121].point), 0.1);
	//TS_ASSERT_EQUALS(tmp.ts_type, "temperature");
	TS_ASSERT(tmp.ts.size() > 365 * 2 * 24);
	utctime t = no_utctime;
	for (size_t i = 0; i < tmp.ts.size(); ++i) {
		auto p = tmp.ts.get(i);
		if (t == no_utctime) {
			t = p.t;
		} else {
			TS_ASSERT(p.t > t);
		}
		TS_ASSERT(p.v< 40.0 && p.v > -40.0);
		t = p.t;
	}
}

TEST_CASE("cell_builder_test::test_io_performance") {
	auto t0 = shyft::core::utctime_now();

	auto temp = shyfttest::find("neanidelv", "temperature");
	auto prec = shyfttest::find("neanidelv", "precipitation");
	auto disc = shyfttest::find("neanidelv", "discharge");
	auto rad = shyfttest::find("neanidelv", "radiation");
	TS_ASSERT_EQUALS(temp.size(), 10u);
	TS_ASSERT_DIFFERS(prec.size(), 0u);
	TS_ASSERT_DIFFERS(disc.size(), 0u);
	TS_ASSERT_EQUALS(rad.size(), 1u);
	auto dt = shyft::core::utctime_now() - t0;
	TS_ASSERT_LESS_THAN(dt, 10);
}

template<class ts_t>
static void print(ostream&os, const ts_t& ts, size_t i0, size_t max_sz) {
	for (size_t i = i0; i < min(i0 + max_sz, ts.size()); ++i)
		os << (i == i0 ? "\n" : ",") << ts.value(i);
	os << endl;
}

TEST_CASE("cell_builder_test::test_read_and_run_region_model") {

	//
	// Arrange
	//
	const char *test_path = "neanidelv";
	using namespace shyft::experimental;
	using namespace shyft::experimental::repository;
	// define a cell type
	typedef ec::pt_gs_k::cell_discharge_response_t cell_t;
	// and a region model for that cell-type
	typedef ec::region_model<cell_t, region_environment_t> region_model_t;
	// Step 1: read cells from cell_file_repository
	cout << endl << "1. Reading cells from files" << endl;
    auto cells = make_shared<vector<cell_t>>();
    auto global_parameter = make_shared<cell_t::parameter_t>();
#ifdef _WIN32
    const char *cell_path = "neanidelv/geo_cell_data.v2.win.bin";
#else
    const char *cell_path = "neanidelv/geo_cell_data.v2.bin";
#endif
    bool verbose = getenv("SHYFT_VERBOSE") != nullptr;
    std::string geo_xml_fname = shyft::experimental::io::test_path(cell_path, false);
    if ( !boost::filesystem::is_regular_file(boost::filesystem::path(geo_xml_fname))) {
        cout << "-> bin file missing,   regenerating xml file (could take some time)" << endl;
        cell_file_repository<cell_t> cfr(test_path, 557600.0, 6960000.0, 122, 75, 1000.0, 1000.0);
        TS_ASSERT(cfr.read(cells));
        //-- stream cell.geo to a store
        std::vector<shyft::core::geo_cell_data> gcd;
        gcd.reserve(cells->size());
        for (const auto&c : *cells) gcd.push_back(c.geo);
        std::ofstream geo_cell_xml_file(geo_xml_fname,ios::binary);
        boost::archive::binary_oarchive oa(geo_cell_xml_file);
        oa << BOOST_SERIALIZATION_NVP(gcd);
    }

    {
        std::vector<shyft::core::geo_cell_data> gcd;gcd.reserve(5000);
        std::ifstream geo_cell_xml_file(geo_xml_fname,ios::binary);
        boost::archive::binary_iarchive ia(geo_cell_xml_file);
        ia >> BOOST_SERIALIZATION_NVP(gcd);
        cells->reserve(gcd.size());
        cell_t::state_t s0;
        s0.kirchner.q = 100.0;
        for (const auto& g : gcd) {
            cells->push_back(cell_t{ g, global_parameter, s0 });
        }
    }

    auto cal = ec::calendar();
	auto start = cal.time(2010, 9, 1, 0, 0, 0);
	auto dt_hours=3;
	auto dt = ec::deltahours(dt_hours);
	auto ndays = atoi(getenv("NDAYS") ? getenv("NDAYS") : "90");
	size_t n = 24 * ndays/dt_hours;//365;// 365 takes 20 seconds at cell stage 8core
	ta::fixed_dt ta(start, dt, n);
    ta::fixed_dt ta_one_step(start, dt*n, 1);

    // Step 2: read geo located ts
	cout << "2. Reading geo-located time-series from file (could take a short time)" << endl;
	geo_located_ts_file_repository geo_f_ts(test_path);
	region_environment_t re;
	TS_ASSERT(geo_f_ts.read(re));
	if(!re.rel_hum) {
        re.rel_hum=make_shared<vector<geo_cts_t>>();
        re.wind_speed= make_shared<vector<geo_cts_t>>();
        auto gp = (*re.temperature)[0].location;
        re.rel_hum->push_back(geo_cts_t{ gp,cts_t(ta,0.8)});
        re.wind_speed->push_back(geo_cts_t{gp,cts_t(ta,2.0)});
	}
    //return;
	// Step 3: make a region model
	cout << "3. creating a region model and run t0="<<cal.to_string(start)<<",dt="<<dt_hours<<"h, n= "<<n<<" ~"<<ndays<<" days" << endl;
	region_model_t rm(cells, *global_parameter);
	rm.ncore = atoi(getenv("NCORE") ? getenv("NCORE") : "8");
	cout << " - ncore set to " << rm.ncore << endl;
	ec::interpolation_parameter ip;
	ip.use_idw_for_temperature = false;
    vector<int> all_catchment_ids;// empty vector means no filtering
    vector<int> catchment_ids{ 87,115 };
    vector<int> other_ids{ 38,188,259,295,389,465,496,516,551,780 };
	auto ti1 = timing::now();
    rm.initialize_cell_environment(ta);
    rm.set_catchment_calculation_filter(catchment_ids);
	rm.interpolate(ip, re);
	auto ipt =elapsed_ms(ti1,timing::now());
	cout << "3. a Done with interpolation step two catchments used = " << ipt << "[ms]" << endl;
    auto avg_precip_ip_set = ec::cell_statistics::average_catchment_feature(*rm.get_cells(), catchment_ids, [](const cell_t&c) {return c.env_ts.precipitation; });
    auto avg_precip_ip_set_value = et::average_accessor<et::pts_t, ta::fixed_dt>(avg_precip_ip_set, ta_one_step).value(0);
    auto avg_precip_ip_o_set = ec::cell_statistics::average_catchment_feature(*rm.get_cells(), other_ids, [](const cell_t&c) {return c.env_ts.precipitation; });
    auto avg_precip_ip_o_set_value = et::average_accessor<et::pts_t, ta::fixed_dt>(avg_precip_ip_o_set, ta_one_step).value(0);
    //cout << "partial:avg precip for selected    catchments is:" << avg_precip_ip_set_value << endl;
    //cout << "partial:avg precip for unselected  catchments is:" << avg_precip_ip_o_set_value << endl;
    FAST_CHECK_GT(avg_precip_ip_set_value, 0.05);
    FAST_CHECK_EQ(std::isfinite(avg_precip_ip_o_set_value),false);
    rm.set_catchment_calculation_filter(all_catchment_ids);
    ti1 = timing::now();
    auto ok_ip = rm.interpolate(ip, re);
    ipt = elapsed_ms(ti1, timing::now());
    cout << "3. b Done with interpolation step *all* catchments used = " << ipt << "[ms]" << endl;
    REQUIRE(ok_ip);
    // now verify we got same results for the limited set, and non-zero for the others.
    auto avg_precip_ip_set2 = ec::cell_statistics::average_catchment_feature(*rm.get_cells(), catchment_ids, [](const cell_t&c) {return c.env_ts.precipitation; });
    auto avg_precip_ip_set_value2 = et::average_accessor<et::pts_t, ta::fixed_dt>(avg_precip_ip_set2, ta_one_step).value(0);
    auto avg_precip_ip_o_set2 = ec::cell_statistics::average_catchment_feature(*rm.get_cells(), other_ids, [](const cell_t&c) {return c.env_ts.precipitation; });
    auto avg_precip_ip_o_set_value2 = et::average_accessor<et::pts_t, ta::fixed_dt>(avg_precip_ip_o_set2, ta_one_step).value(0);
    FAST_CHECK_LT(std::abs(avg_precip_ip_set_value - avg_precip_ip_set_value2), 0.0001);
    FAST_CHECK_GT(avg_precip_ip_o_set_value2, 0.05);
    //cout << "full:avg precip for selected    catchments is:" << avg_precip_ip_set_value2 << endl;
    //cout << "full:avg precip for unselected  catchments is:" << avg_precip_ip_o_set_value2 << endl;
    cout << "3. c Verify best_effort interpolation";
    auto re_temperature = *(re.temperature);
    for(auto& gts:*(re.temperature)) {
        for(size_t i=100;i<gts.ts.size();++i)
            gts.ts.set(i,shyft::nan);
    }
    auto ok=rm.interpolate(ip,re);
    cout<<"\ndone "<<(ok?", all ok":", with problems")<<", exit"<<endl;
    REQUIRE(!ok);
    try {
        ok=rm.interpolate(ip,re,false);
        REQUIRE(false);// REQUIRE the above to throw
    } catch(exception const &ex) {
        // we should get an exception here
        if(verbose) cout<<"ok, got expected exception : "<<ex.what()<<endl;
    }
    ip.use_idw_for_temperature = true;// test for idw (more forgiving)
    try {
        ok=rm.interpolate(ip,re,false);
    } catch(exception const &ex) {
        // we should get an exception here
        if(verbose) cout<<"ok, got expected exception : "<<ex.what()<<endl;
        REQUIRE(false);// REQUIRE the above to not throw (idw is best effort by design)
    }
    *re.temperature = re_temperature; // restore ok temperatures
    if (getenv("SHYFT_IP_ONLY"))
		return;
	ok_ip = rm.interpolate(ip, re);
	REQUIRE(ok_ip);
	vector<shyft::core::pt_gs_k::state_t> s0;
	// not needed, the rm will provide the initial_state for us.rm.get_states(s0);
    //
    auto t0 = timing::now();

    //rm.set_catchment_calculation_filter(catchment_ids);
	rm.set_snow_sca_swe_collection(-1, true);
	rm.run_cells();

    auto ms = elapsed_ms(t0,timing::now());

	cout << "3. b Done with cell-step :" << ms << " [ms]" << endl;
	auto sum_discharge = ec::cell_statistics::sum_catchment_feature(*rm.get_cells(), all_catchment_ids, [](const cell_t&c) {return c.rc.avg_discharge; });
	auto snow_sca = ec::cell_statistics::average_catchment_feature(*rm.get_cells(), all_catchment_ids, [](const cell_t &c) {return c.rc.snow_sca; });
	auto snow_swe = ec::cell_statistics::average_catchment_feature(*rm.get_cells(), all_catchment_ids, [](const cell_t &c) {return c.rc.snow_swe; });
    size_t i0 = 8 * 30;
    size_t n_steps = 8 * 30;
    if (verbose) {
        cout << "4. Print results" << endl;
        cout << "discharge:" << endl; print(cout, *sum_discharge, i0, n_steps);
        cout << "snow_sca :" << endl; print(cout, *snow_sca, i0, n_steps);
        cout << "snow_swe :" << endl; print(cout, *snow_swe, i0, n_steps);
    }

    auto sum_dischargex = ec::cell_statistics::sum_catchment_feature(*rm.get_cells(), catchment_ids, [](const cell_t&c) {return c.rc.avg_discharge; });
    auto snow_scax = ec::cell_statistics::average_catchment_feature(*rm.get_cells(), catchment_ids, [](const cell_t &c) {return c.rc.snow_sca; });
    auto snow_swex = ec::cell_statistics::average_catchment_feature(*rm.get_cells(), catchment_ids, [](const cell_t &c) {return c.rc.snow_swe; });
    if (verbose) {
        cout << "discharge:" << endl; print(cout, *sum_dischargex, i0, n_steps);
        cout << "snow_sca :" << endl; print(cout, *snow_scax, i0, n_steps);
        cout << "snow_swe :" << endl; print(cout, *snow_swex, i0, n_steps);
    }
	cout << endl << "5. now a run with just two catchments" << endl;

    auto sum_discharge2x = ec::cell_statistics::sum_catchment_feature(*rm.get_cells(), catchment_ids, [](const cell_t&c) {return c.rc.avg_discharge; });
    // also setup a river network for these two catchments
    // so that they end up in a common river
	rm.set_catchment_calculation_filter(catchment_ids);
    int common_river_id = 1;
    ec::routing::river sum_river_115_87(common_river_id, ec::routing_info(0, 0.0));// sum river of 38 and 87
    ec::routing::river river_115(115, ec::routing_info(common_river_id, 0.0));// use river-id eq. catchment-id, but any value would do
    ec::routing::river river_87(87, ec::routing_info(common_river_id, 0.0));
    // add to region model:
    rm.river_network.add(sum_river_115_87).add(river_115).add(river_87);
    rm.connect_catchment_to_river(115, 115);// make the connections for the cells
    rm.connect_catchment_to_river(87, 87);// currently, the effective routing is zero, should be equal:
    // and both these river
    auto sum_discharge_115_87 = ec::cell_statistics::sum_catchment_feature(*rm.get_cells(), catchment_ids, [](const cell_t&c) {return c.rc.avg_discharge; });
    auto sum_river_discharge = rm.river_output_flow_m3s(common_river_id);// verify it's equal
    for (size_t i = 0;i < sum_discharge_115_87->size();++i) {
        TS_ASSERT_DELTA(sum_discharge_115_87->value(i), sum_river_discharge->value(i), 0.001);
    }
	cout << "5. b Done, now compute new sum" << endl;
	rm.revert_to_initial_state();
	rm.get_states(s0);// so that we start at same state.
    rm.set_catchment_calculation_filter(catchment_ids);
    ta::fixed_dt tax = rm.time_axis; tax.n = 0; // force zero length time-axis for non calc cells.
    for (auto&c : *rm.get_cells())
        c.begin_run(tax, 0, 0);
    rm.set_snow_sca_swe_collection(-1, true);
	rm.run_cells();// this time only two catchments
	cout << "6. Done, now compute new sum" << endl;
	auto sum_discharge2 = ec::cell_statistics::sum_catchment_feature(*rm.get_cells(), catchment_ids, [](const cell_t&c) {return c.rc.avg_discharge; });
	auto snow_sca2 = ec::cell_statistics::average_catchment_feature(*rm.get_cells(), catchment_ids, [](const cell_t &c) {return c.rc.snow_sca; });
	auto snow_swe2 = ec::cell_statistics::average_catchment_feature(*rm.get_cells(), catchment_ids, [](const cell_t &c) {return c.rc.snow_swe; });
    auto sum_river_discharge2 = rm.river_output_flow_m3s(common_river_id);
    TS_ASSERT_EQUALS(sum_discharge2->size(), ta.size());// ensure we get complete results when running with catch.calc filter
    TS_ASSERT_EQUALS(snow_swex->size(), snow_swe2->size());
    for (size_t i = 0;i < sum_discharge2->size();++i) { // should still be equal
        TS_ASSERT_DELTA(sum_discharge2->value(i), sum_river_discharge2->value(i), 0.001);
        TS_ASSERT_DELTA(sum_discharge_115_87->value(i), sum_discharge2->value(i), 0.001);// the two runs are equal
        TS_ASSERT_DELTA(snow_swex->value(i), snow_swe2->value(i), 0.001);
    }
    if (verbose) {
        cout << "7. Sum discharge for " << catchment_ids[2] << " and " << catchment_ids[1] << " is:" << endl;

        cout << "discharge:" << endl; print(cout, *sum_discharge2, i0, n_steps);
        cout << "snow_sca :" << endl; print(cout, *snow_sca2, i0, n_steps);
        cout << "snow_swe :" << endl; print(cout, *snow_swe2, i0, n_steps);
    }
	cout << endl << "Done test read and run region-model" << endl;
    if (!getenv("SHYFT_FULL_TEST")) {
        TS_TRACE("Please define SHYFT_FULL_TEST, export SHYFT_FULL_TEST=TRUE; or win: set SHYFT_FULL_TEST=TRUE to enable calibration run of nea-nidelv in this test");
        cout << "Please define SHYFT_FULL_TEST, export SHYFT_FULL_TEST = TRUE; or win: set SHYFT_FULL_TEST = TRUE to enable calibration run of nea - nidelv in this test"<<endl;
        return;
    }
    // To enable cell-to river calibration, introduce a routing-effect between the cells and the river.
    // we do this by setting the routing distance in the cells, and then also adjust the parameter.routing.velocity
    //
    double hydro_distance = 1000.0;//m
    for (auto &c : *rm.get_cells()) {
        c.geo.routing.distance = hydro_distance;// all set to 1000 meter
    }
    double n_timesteps_delay = 7.0;// make the uhg approx 7 time steps long (3hourx 7 ~ 21 hours)
    rm.get_region_parameter().routing.velocity = hydro_distance / dt/ n_timesteps_delay;

    // now we can pull out the routed flow (delayed and smoothed)
    auto routed_flow = rm.river_output_flow_m3s(common_river_id);

	rm.revert_to_initial_state();//set_states(s0);// get back initial state
	cout << "Calibration/parameter optimization" << endl;
	using namespace shyft::core::model_calibration;
	typedef shyft::core::pt_gs_k::parameter_t parameter_accessor_t;
	typedef target_specification<pts_t> target_specification_t;
	target_specification_t discharge_target(*sum_discharge2, catchment_ids, 1.0, KLING_GUPTA, 1.0, 1.0, 1.0, DISCHARGE);
	target_specification_t snow_sca_target(*snow_sca2, catchment_ids, 1.0, KLING_GUPTA, 1.0, 1.0, 1.0, SNOW_COVERED_AREA);
	target_specification_t snow_swe_target(*snow_swe2, catchment_ids, 1.0, KLING_GUPTA, 1.0, 1.0, 1.0, SNOW_WATER_EQUIVALENT);
    target_specification_t routed_target(*routed_flow, common_river_id, 1.0, KLING_GUPTA, 1.0, 1.0, 1.0);

	vector<target_specification_t> target_specs;
	target_specs.push_back(discharge_target);
	target_specs.push_back(snow_sca_target);
	target_specs.push_back(snow_swe_target);
    target_specs.push_back(routed_target);
    *global_parameter = rm.get_region_parameter();//refresh the values to current
	parameter_accessor_t& pa(*global_parameter);
	// Define parameter ranges
	const size_t n_params = pa.size();
	std::vector<double> lower; lower.reserve(n_params);
	std::vector<double> upper; upper.reserve(n_params);
    auto orig_parameter= *global_parameter;
	vector<bool> calibrate_parameter(n_params, false);
    // 25 is routing velocity
	for (auto i : vector<int>{ 0,4,14,16,25 }) calibrate_parameter[i] = true;
	for (size_t i = 0; i < n_params; ++i) {
		double v = pa.get(i);
		lower.emplace_back(calibrate_parameter[i] ? 0.7*v : v);
		upper.emplace_back(calibrate_parameter[i] ? 1.2*v : v);
	}
	// Perturb parameter set
	std::vector<double> x(n_params);
	//SiH: We don't use random, would like to repeat exactly (if possible) the tests.
	// std::default_random_engine rnd; rnd.seed(1023);
	for (size_t i = 0; i < n_params; ++i) {
		if (calibrate_parameter[i]) { // put the parameter outside optimal value
			x[i] = 0.9*(lower[i] + upper[i])*0.5;//lower[i]< upper[i] ? std::uniform_real_distribution<double>(lower[i], upper[i])(rnd) : std::uniform_real_distribution<double>(upper[i], lower[i])(rnd);
		} else {
			x[i] = (lower[i] + upper[i])*0.5;
		}
	}
    optimizer<region_model_t, parameter_accessor_t, pts_t > rm_opt(rm);
    parameter_accessor_t lwr;lwr.set(lower);
    parameter_accessor_t upr;upr.set(upper);
    parameter_accessor_t px; px.set(x);
    rm_opt.set_target_specification(target_specs, lwr, upr);
	rm_opt.set_verbose_level(1);
	auto tz = timing::now();
	auto x_optimized = rm_opt.optimize(px, 2500, 0.2, 5e-4);
	auto used = elapsed_ms(tz,timing::now());
	cout << "results: " << used << " ms, nthreads = " << rm.ncore << endl;
	cout << " goal function value:" << rm_opt.calculate_goal_function(x_optimized) << endl;
	cout << " x-parameters before and after" << endl;
	for (size_t i = 0; i < x.size(); ++i) {
		if(rm_opt.active_parameter(i) )
            cout<< "'" << pa.get_name(i) << "' = " << px.get(i) << " -> " << x_optimized.get(i) <<
            " (orig:"<<orig_parameter.get(i)<<")"<< endl;
	}
	cout << " done" << endl;
	cout <<"Retry with sceua:\n";
	tz=timing::now();
	auto x_optimized2=rm_opt.optimize_sceua(x_optimized);
	used= elapsed_ms(tz,timing::now());

	cout << "results: " << used << " ms, nthreads = " << rm.ncore << endl;
	cout << " goal function value:" << rm_opt.calculate_goal_function(x_optimized2) << endl;
	cout << " x-parameters before and after" << endl;
	for (size_t i = 0; i < x.size(); ++i) {
		if(rm_opt.active_parameter(i) )
            cout<< "'" << pa.get_name(i) << "' = " << px.get(i) << " -> " << x_optimized2.get(i) <<
            " (orig:"<<orig_parameter.get(i)<<")"<< endl;
	}
	cout<< "done"<<endl;
}
}

