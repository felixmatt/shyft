#include "test_pch.h"
#include "core/core_pch.h"
#include "cell_builder_test.h"
#include "core/experimental.h"
#include "core/model_calibration.h"

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
        shared_ptr<vector<geo_xts_t>> radiations; ///< geo located radiations,  [W/m²]
        shared_ptr<vector<geo_xts_t>> discharges;///< geo|id located discharges from catchments,  [m3/s] .. not really geo located, it's catchment id associated.. but for test/mock that is ok for now

        region_environment_t get_region_environment() {
            region_environment_t re;
            re.temperature = temperatures;
            re.precipitation = precipitations;
            re.radiation = radiations;
            return re;
        }
      private:

    };

    region_test_data load_test_data_from_files(wkt_reader& wkt_io) {
        region_test_data r;
        r.forest = wkt_io.read("forest", slurp(test_path("neanidelv/landtype_forest_wkt.txt")));
        r.glacier  = wkt_io.read("glacier", slurp(test_path("neanidelv/landtype_glacier_wkt.txt")));
        r.lake  = wkt_io.read("lake", slurp(test_path("neanidelv/landtype_lake_wkt.txt")));
        r.rsv_points = wkt_io.read_points("rsv_mid_points", slurp(test_path("neanidelv/landtype_rsv_midpoints_wkt.txt")));
        r.dtmv = wkt_io.read_dtm("dtm", slurp(test_path("neanidelv/dtm_xtra.txt")));
        r.catchment_map = wkt_io.read_catchment_map("catchment_map", slurp(test_path("neanidelv/catchments_wkt.txt")));
        r.location_map = wkt_io.read_geo_point_map("met_stations", slurp(test_path("neanidelv/geo_point_map.txt")));
        function<ec::geo_point(int)> geo_map;
        geo_map = [&r] (int id) {return r.location_map[id].point;};
        r.temperatures = load_from_directory(wkt_io, geo_map, "neanidelv", "temperature");
        r.precipitations = load_from_directory(wkt_io, geo_map, "neanidelv", "precipitation");
        r.discharges = load_from_directory(wkt_io, geo_map, "neanidelv", "discharge");
        r.radiations = load_from_directory(wkt_io, geo_map, "neanidelv", "radiation");
        return r;
    }

}

void cell_builder_test::test_read_geo_region_data_from_files(void) {
    using namespace shyft::experimental;
    using namespace shyfttest;
    TS_TRACE("geo testing not activated yet, experimental");
    return;
    wkt_reader wkt_io;
    region_test_data t(load_test_data_from_files(wkt_io));


    TS_ASSERT(t.dtmv.size()>0);
    TS_ASSERT(t.rsv_points->size()>0);
    TS_ASSERT(t.forest->size()>0);
    TS_ASSERT(t.lake->size()>0);
    TS_ASSERT(t.catchment_map.size()>0);
    TS_ASSERT(t.location_map.size()>= 0);
    TS_ASSERT(t.temperatures->size()>= 0);
    TS_ASSERT(t.precipitations->size()>0);
    TS_ASSERT(t.discharges->size()>0);
    if(wkt_io.reported_errors.size()) {
        TS_TRACE("Some reported errors on the test data read from file");
        cerr << "Reported errors on test-data:" << endl;
        for(auto e:wkt_io.reported_errors) {
            cerr << e << endl;
        }
    }


}

/* \note test below is currently just a play-ground test while searching for minimalistic interfaces and methods */
void cell_builder_test::test_build_geo_cell_data_from_geo_region_data(void) {

    using namespace shyft::experimental;
    using namespace shyfttest;
    namespace ec = shyft::core;
    TS_TRACE("geo testing not activated yet, experimental");
    return;

    wkt_reader wkt_io;
    region_test_data t(load_test_data_from_files(wkt_io));
    cout << "Done reading files, now making a region:" << endl;
    region_grid rg(point_xy(557600.0, 6960000.0), 122, 75, 1000.0, 1000.0);//THese matches exactly the dtm file in test data.
    int misses = rg.set_dtm(t.dtmv);
    TS_ASSERT_EQUALS(misses, 0);// meaning we got the test-dtm data right aligned with the grid.
    // Now pick the reservoirs from lake
    cout <<"Mark the reservoirs" << endl;
    multi_polygon_ l = make_shared<multi_polygon>();
    multi_polygon_ r = make_shared<multi_polygon>();
    for(const auto& lr:*t.lake) {
        for(const auto& rp:*t.rsv_points) {
            polygon mpl;
            geo::simplify(lr, mpl, 300.0);// simplify, down to a x m. resolution, to gain speed
            if(geo::covered_by(rp, lr)) {
                r->push_back(mpl);
            } else {
                l->push_back(mpl);
            }
        }
    }
    cout << "Region data created, verify result" << endl;
    TS_ASSERT(r->size()>0);
    TS_ASSERT(l->size()>0);
    if(wkt_io.reported_errors.size()) {
        TS_TRACE("Some reported errors on the test data read from file");
        cerr << "Reported errors on test-data:" << endl;
        for(auto e:wkt_io.reported_errors) {
            cerr << e << endl;
        }
    }
    cout << "Compute region cells" << endl;
    geo_cell_data_computer region;
    typedef vector<ec::geo_cell_data> geo_cell_data_vector;
    typedef map<int, geo_cell_data_vector> catchment_gcd_map;
    vector<int> internal_to_catchment_id;// we keep core internal id 0-based compact.
    catchment_gcd_map  gcd_map;
    for(const auto&kv:t.catchment_map) {
        internal_to_catchment_id.push_back(kv.first);
        size_t internal_id = internal_to_catchment_id.size()-1;
        gcd_map.insert(make_pair(kv.first, region.catchment_geo_cell_data(
                                rg,
                                internal_id,
                                0.9,
                                *kv.second,
                                *r,
                                *l,
                                *t.glacier,
                                *t.forest
                                )));
    }
    cout << "done creating the region cells, verify results" << endl;
    TS_ASSERT_DIFFERS(gcd_map.size(), 0);
    TS_ASSERT_EQUALS(region.area_errors, 0);
    size_t n_cells = 0;
    typedef ec::pt_gs_k::cell_discharge_response_t cell_t;
    typedef ec::region_model<cell_t> region_model_t;
    typedef vector<cell_t> cell_vector_t;
    auto cells = make_shared<cell_vector_t>();
    cells->reserve(rg.n_cells());
    shyft::core::pt_gs_k::state_t cell_state;
    cell_state.kirchner.q = 100.0;
    auto global_parameter = make_shared<shyft::core::pt_gs_k::parameter_t>();
    for(const auto& kv:gcd_map) {

        double cell_sum = 0.0;
        double z_sum = 0.0;
        TS_ASSERT_DIFFERS(kv.second.size(), 0);
        for(const auto& gcd:kv.second){
            cell_sum += gcd.area();
            z_sum += gcd.mid_point().z;
            n_cells++;
            cells->emplace_back(cell_t{gcd, global_parameter, cell_state});
        }
        TS_ASSERT_LESS_THAN(0.0, z_sum);
        // verify agains original areal
        //const auto& c_geom=t.catchment_map[kv.first];
        //double shape_area= geo::area(*c_geom);
        //TS_ASSERT_DELTA(cell_sum, shape_area, 1000.0);
    }
    cout << "Finished reading, create region, verify results" << endl;
    region_model_t rm(cells, *global_parameter);
    auto re = t.get_region_environment();
    auto cal = ec::calendar();
    auto start = cal.time(ec::YMDhms(2010, 9, 1, 0, 0, 0));
    auto dt = ec::deltahours(1);
    size_t n = 24*10*2;// 365 takes 20 seconds at cell stage 8core
    et::timeaxis ta(start, dt, n);
    ec::interpolation_parameter ip;
    rm.run_interpolation(ip, ta, re);
    cout << "Done with interpolation step" << endl;
    vector<shyft::core::pt_gs_k::state_t> s0;
    rm.get_states(s0);
    rm.run_cells();
    cout << "Done with cellstep" << endl;
    vector<int> cids{0, 2};
    auto avg_temperature = ec::cell_statistics::average_catchment_feature(*cells, cids, [](const cell_t&c) {return c.env_ts.temperature;});
    vector<int> all_cids;
    auto sum_discharge = ec::cell_statistics::sum_catchment_feature(*cells, all_cids, [](const cell_t &c) {return c.rc.avg_discharge;});
    //pts_t sum_discharge(ta, 0.0);
    //for(auto &c:*cells) {
    //    sum_discharge.add(c.rc.avg_discharge);
    //}
    cout << "Sum discharge" << endl;
    for(size_t i=0;i< (n<30?n:30);++i) {
        cout << "\t" << i << ":" << sum_discharge->value(i) << endl;
    }
    cout << "now a run with just two catchments" << endl;
    vector<int> catchment_ids{0, 2};
    rm.set_catchment_calculation_filter(catchment_ids);
    cout << "Done, now compute new sum" << endl;
    rm.set_states(s0);// so that we start at same state.
    rm.run_cells();// this time only two catchments
    pts_t sum_discharge2(ta, 0.0);
    cout << "Done, now compute new sum" << endl;
    for(auto &cx:*cells) {
        if(find(begin(catchment_ids), end(catchment_ids), cx.geo.catchment_id())!= end(catchment_ids))
            sum_discharge2.add(cx.rc.avg_discharge);
    }
    cout << "Sum discharge for "<< internal_to_catchment_id[catchment_ids[0]] << " and " << internal_to_catchment_id[catchment_ids[1]] << " is:" << endl;
    for(size_t i=0;i< (n<30?n:30);++i) {
        cout << "\t" << i << ":" << sum_discharge2.value(i) << endl;
    }

    cout << "finito" << endl;

}
void cell_builder_test::test_read_geo_point_map(void){
    using namespace shyft::experimental;
    using namespace shyfttest;
    namespace ec = shyft::core;
    wkt_reader wio;
    map<int, observation_location> obs;
    obs = wio.read_geo_point_map("met_stations", slurp(test_path("neanidelv/geo_point_map.txt")));
    TS_ASSERT_DIFFERS(obs.size(), 0);
    for(const auto& kv:obs){
        TS_ASSERT_DIFFERS(kv.first, 0);
        TS_ASSERT_DIFFERS(kv.second.point.x, 0.0);
        TS_ASSERT_DIFFERS(kv.second.point.y, 0.0);
        TS_ASSERT_DIFFERS(kv.second.point.z, 0.0);

    }
}
void cell_builder_test::test_read_geo_located_ts() {
    using namespace shyft::experimental;
    using namespace shyfttest;
    using namespace shyft::core;
    wkt_reader wio;
    map<int, observation_location> obs;
    obs = wio.read_geo_point_map("met_stations", slurp(test_path("neanidelv/geo_point_map.txt")));

    geo_xts_t tmp = wio.read_geo_xts_t(
            "gts.121.temperature",
            [&obs] (int id) {return obs[id].point;},
             slurp(test_path("neanidelv/gts.121.temperature")));
    TS_ASSERT_LESS_THAN(geo_point::distance2(tmp.location, obs[121].point), 0.1);
    //TS_ASSERT_EQUALS(tmp.ts_type, "temperature");
    TS_ASSERT(tmp.ts.size()>365*2*24);
    utctime t = no_utctime;
    for(size_t i=0;i<tmp.ts.size();++i) {
        auto p = tmp.ts.get(i);
        if( t== no_utctime) {
            t = p.t;
        } else {
            TS_ASSERT( p.t > t );
        }
        TS_ASSERT( p.v< 40.0 && p.v > -40.0);
        t = p.t;
    }
}

void cell_builder_test::test_io_performance() {
    auto t0 = shyft::core::utctime_now();

    auto temp = shyfttest::find("neanidelv", "temperature");
    auto prec = shyfttest::find("neanidelv", "precipitation");
    auto disc = shyfttest::find("neanidelv", "discharge");
    auto rad = shyfttest::find("neanidelv", "radiation");
    TS_ASSERT_EQUALS(temp.size(), 10);
    TS_ASSERT_DIFFERS(prec.size(), 0);
    TS_ASSERT_DIFFERS(disc.size(), 0);
    TS_ASSERT_EQUALS(rad.size(), 1);
    auto dt = shyft::core::utctime_now()-t0;
    TS_ASSERT_LESS_THAN(dt, 10);


}
void cell_builder_test::test_read_and_run_region_model(void) {
    if(!getenv("SHYFT_FULL_TEST")) {
        TS_TRACE("Please define SHYFT_FULL_TEST, export SHYFT_FULL_TEST=TRUE; or win: set SHYFT_FULL_TEST=TRUE to enable real run of nea-nidelv in this test");
        return;
    }
    //
    // Arrange
    //
    const char *test_path = "neanidelv";
    using namespace shyft::experimental;
    using namespace shyft::experimental::repository;
    // define a cell type
    typedef ec::pt_gs_k::cell_discharge_response_t cell_t;
    // and a region model for that cell-type
    typedef ec::region_model<cell_t> region_model_t;
    // Step 1: read cells from cell_file_repository
    cout << endl << "1. Reading cells from files (could take some time)" << endl;
    cell_file_repository<cell_t> cfr(test_path, 557600.0, 6960000.0, 122, 75, 1000.0, 1000.0);
    auto cells = make_shared<vector<cell_t>>();
    vector<int> internal_to_catchment_id;
    TS_ASSERT(cfr.read(cells, internal_to_catchment_id));
    // Step 2: read geo located ts
    cout << "2. Reading geo-located time-series from file (could take a short time)" << endl;
    geo_located_ts_file_repository geo_f_ts(test_path);
    region_environment_t re;
    TS_ASSERT(geo_f_ts.read(re));
    // Step 3: make a region model
    cout << "3. creating a region model and run it for a short period" << endl;
    auto global_parameter = make_shared<shyft::core::pt_gs_k::parameter_t>();
    region_model_t rm(cells, *global_parameter);
	rm.ncore = atoi(getenv("NCORE") ? getenv("NCORE") : "32");
	cout << " - ncore set to " << rm.ncore << endl;
    auto cal = ec::calendar();
    auto start = cal.time(ec::YMDhms(2010, 9, 1, 0, 0, 0));
    auto dt = ec::deltahours(1);
	auto ndays = atoi(getenv("NDAYS") ? getenv("NDAYS") : "20");
    size_t n = 24*ndays;//365;// 365 takes 20 seconds at cell stage 8core
    et::timeaxis ta(start, dt, n);
    ec::interpolation_parameter ip;
    ip.use_idw_for_temperature = true;
    auto ti1 = ec::utctime_now();
    rm.run_interpolation(ip, ta, re);
    auto ipt = ec::utctime_now()-ti1;
    cout << "3. a Done with interpolation step used = " << ipt << "[s]" << endl;
    if(getenv("SHYFT_IP_ONLY"))
        return;
    vector<shyft::core::pt_gs_k::state_t> s0;
    rm.get_states(s0);
    auto t0 = ec::utctime_now();

	vector<int> catchment_ids{ 0, 2 };
	rm.set_catchment_calculation_filter(catchment_ids);

    rm.run_cells();


	cout << "3. b Done with cellstep :" << ec::utctime_now()-t0 << " [s]" << endl;
    pts_t sum_discharge(ta, 0.0);
    for(auto &c:*cells) {
        if(rm.is_calculated(c.geo.catchment_id()))
			sum_discharge.add(c.rc.avg_discharge);
    }
    cout << "4. Print Sum discharge" << endl;
    for(size_t i=0;i< (n<30?n:30);++i) {
        cout << (i == 0 ? "\t" : ",") << sum_discharge.value(i);
    }
    cout << endl << "5. now a run with just two catchments" << endl;

	//vector<int> catchment_ids{0, 2};
    //rm.set_catchment_calculation_filter(catchment_ids);

    cout << "5. b Done, now compute new sum" << endl;
    rm.set_states(s0);// so that we start at same state.
    rm.run_cells();// this time only two catchments
    pts_t sum_discharge2(ta, 0.0);
    cout << "6. Done, now compute new sum" << endl;
    for(auto &cx:*cells) {
        if(find(begin(catchment_ids), end(catchment_ids), cx.geo.catchment_id())!= end(catchment_ids))
            sum_discharge2.add(cx.rc.avg_discharge);
    }
    cout << "7. Sum discharge for "<< internal_to_catchment_id[catchment_ids[0]] << " and " << internal_to_catchment_id[catchment_ids[1]] << " is:" << endl;
    for(size_t i=0;i< (n<30?n:30);++i) {
        cout << (i == 0 ? "\t" : ",") << sum_discharge2.value(i);
    }

    cout << endl << "Done test read and run region-model" << endl;
    rm.set_states(s0);// get back initial state
    cout << "Calibration/parameter optimization" << endl;
    using namespace shyft::core::model_calibration;
    typedef shyft::core::pt_gs_k::parameter_t parameter_accessor_t;
    typedef target_specification<pts_t> target_specification_t;
    target_specification_t spec1(sum_discharge2, catchment_ids, 1.0, KLING_GUPTA);
    vector<target_specification_t> target_specs;
    target_specs.push_back(spec1);
    parameter_accessor_t& pa(*global_parameter);
        // Define parameter ranges
    const size_t n_params = pa.size();
    std::vector<double> lower; lower.reserve(n_params);
    std::vector<double> upper; upper.reserve(n_params);
    const size_t n_calib_params = 4;
    for (size_t i = 0; i < n_params; ++i) {
        double v = pa.get(i);
        lower.emplace_back(i < n_calib_params?0.7*v:v);
        upper.emplace_back(i< n_calib_params?1.2*v:v);
    }
       // Perturb parameter set
    std::vector<double> x(n_params);
    std::default_random_engine rnd; rnd.seed(1023);
	for (size_t i = 0; i < n_params; ++i) {
        if(i<n_calib_params) {
            x[i] = lower[i]< upper[i] ? std::uniform_real_distribution<double>(lower[i], upper[i])(rnd) : std::uniform_real_distribution<double>(upper[i], lower[i])(rnd);
        } else {
            x[i] = (lower[i] + upper[i])*0.5;
        }
	}
    optimizer<region_model_t, parameter_accessor_t , pts_t > rm_opt(rm, target_specs, lower, upper);
    rm_opt.set_verbose_level(1);
	auto tz = ec::utctime_now();
    auto x_optimized = rm_opt.optimize(x,2500,0.1,5e-6);
	auto used = ec::utctime_now() - tz;
    cout<< "results: " << used << " seconds, nthreads = "<< rm.ncore << endl;
    cout<< " goal function value:" << rm_opt.calculate_goal_function(x_optimized) << endl;
    cout<< " x-parameters before and after" << endl;
    for(size_t i=0;i<x.size();++i) {
        cout << "(" << i << ") = " << x[i] << " -> " << x_optimized[i] << endl;
    }
    cout << " done" << endl;
}
