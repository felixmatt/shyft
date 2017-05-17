#include "test_pch.h"


// from core pull in the basic templated algorithms
#include "core/utctime_utilities.h"
#include "core/geo_cell_data.h"


#include "core/time_series.h"
#include "core/inverse_distance.h"
#include "core/bayesian_kriging.h"

#include "core/precipitation_correction.h"
#include "core/priestley_taylor.h"
#include "core/gamma_snow.h"

#include "core/kirchner.h"
#include "core/pt_gs_k.h"


#include "core/region_model.h"

#include "core/cell_model.h"
#include "core/pt_gs_k_cell_model.h"

// define namespace shorthands
using namespace std;
namespace sc = shyft::core;
namespace st = shyft::time_series;
namespace pt = shyft::core::priestley_taylor;
namespace pc = shyft::core::precipitation_correction;
namespace gs = shyft::core::gamma_snow;
namespace ae = shyft::core::actual_evapotranspiration;
namespace kr = shyft::core::kirchner;
namespace pt_gs_k = shyft::core::pt_gs_k;
namespace ta = shyft::time_axis;

// and typedefs for commonly used types in this test
typedef st::point_ts<ta::fixed_dt> pts_t;
typedef st::constant_timeseries<ta::fixed_dt> cts_t;
typedef ta::fixed_dt ta_t;

TEST_SUITE("region_model") {
TEST_CASE("test_build") {

    // arrange
    sc::calendar cal;
    auto start = cal.time(sc::YMDhms(2015, 1, 1, 12, 00, 00));
    auto dt = sc::deltahours(1);
    size_t n = 24*10;

    ta_t ta(start, dt, n);
    pts_t x;
    pts_t prec(ta, 2.0);
    pts_t rad (ta, 100.0);
    pts_t temp(ta, -5.0);
    pts_t windspd(ta, 2.0);
    pts_t rhum(ta, 60.0);
    cts_t chum(ta, 60.0);
    auto env = sc::create_cell_environment<ta_t>(x, temp, rad, windspd, rhum);
    env.init(ta);
    TS_ASSERT_EQUALS(env.temperature.size(), n);

    // create one cell..

    // GeoCell, easy to understand
    sc::geo_cell_data gc1(sc::geo_point(100.0, 100.0, 10.0), 1000.0*1000.0, 0);
    sc::geo_cell_data gc2(sc::geo_point(100.0, 100.0, 10.0), 1000.0*1000.0, 1);
    // The model stack needs its parameters:
    pt::parameter pt;
    gs::parameter gs;
    ae::parameter ae;
    kr::parameter kp;
    pc::parameter scp;
    shared_ptr<pt_gs_k::parameter_t> gp(new pt_gs_k::parameter_t{pt, gs, ae, kp, scp});

    // And there has to be a start state

    gs::state gss;
    kr::state ks;ks.q = 30.0;
    pt_gs_k::state_t state{gss, ks};

    pt_gs_k::cell_complete_response_t c1;
    pt_gs_k::cell_discharge_response_t c2;

    c1.geo = gc1;
    c1.env_ts = env;
    c1.set_parameter(gp);

    c1.init_env_ts(ta); // zero,
    c1.env_ts.precipitation = prec; // fix some numbers
    c1.env_ts.temperature = temp;
    c1.env_ts.rel_hum = rhum;
    c1.set_state(state);
    c1.run(ta,0,0);

    c2.geo = gc2;
    c2.env_ts = env;
    c2.set_parameter(gp);

    c2.init_env_ts(ta); // zero,
    c2.env_ts.precipitation.fill(5.0);// fix some numbers
    c2.env_ts.temperature.fill(3.0);
    c2.env_ts.rel_hum = rhum;
    c2.set_state(state);
    c2.run(ta, 0, 0);

    TS_ASSERT_EQUALS(c1.rc.snow_swe.size(), ta.size());

    // now build region environment
    typedef sc::geo_point_ts<pts_t> gpts_t;
    typedef sc::geo_point_ts<cts_t> gcts_t;

    sc::geo_point s1(2000.0, 2000.0, 10.0);
    sc::geo_point s2(250.0, 1500.0, 200.0);
    gpts_t gprec{s1, prec};
    gpts_t gtemp {s1, temp};
    pts_t temp2 = temp;
    temp2.fill(3.0);
    gpts_t gtemp2{s2, temp2};
    //<class PS,class TS,class RS,class HS,class WS>
    typedef sc::region_environment<gpts_t, gpts_t, gpts_t, gcts_t, gcts_t> test_env_t;
    test_env_t testenv;
    testenv.temperature = make_shared<vector<gpts_t>>();
    testenv.temperature->push_back(gtemp);
    testenv.temperature->push_back(gtemp2);
    testenv.precipitation = make_shared<vector<gpts_t>>();
    testenv.precipitation->push_back(gtemp);
    testenv.precipitation->push_back(gtemp2);

    typedef sc::region_model<pt_gs_k::cell_complete_response_t, test_env_t> ptgsk_region_model_t;
    auto ptgsk_cells = make_shared<std::vector<pt_gs_k::cell_complete_response_t>> ();//ptgsk_cells;
    auto c1b = c1;
    c1b.geo.set_catchment_id(1);
    ptgsk_cells->push_back(c1);
    ptgsk_cells->push_back(c1b);
    map<int, pt_gs_k::parameter_t> catchment_params;
    auto c1p = pt_gs_k::parameter_t{pt, gs, ae, kp, scp};
    c1p.kirchner.c1 = -2.5;
    catchment_params[1] = c1p;

    ptgsk_region_model_t rm(ptgsk_cells, *gp, catchment_params);
    TS_ASSERT(rm.has_catchment_parameter(1));
    auto &c1pr = rm.get_catchment_parameter(1);
    TS_ASSERT_DELTA(c1pr.kirchner.c1, -2.5, 0.0001);

    sc::interpolation_parameter ip;
    rm.run_interpolation(ip, ta, testenv);
    auto tsz1 = c1.env_ts.temperature.size();
    rm.run_cells();
    tsz1 = c1.env_ts.temperature.size();
    TS_ASSERT(tsz1>0);
    SUBCASE("re_init_ts_test") { // test case that cover issue reported by Yisak, re-init/re-run did not fixup result time-axis
        pts_t ts(ta, 2.0);
        FAST_CHECK_EQ(ts.size(), ta.size());
        FAST_CHECK_EQ(ta, ts.ta);
        auto ta2 = ta;
        ta2.t += sc::deltahours(1);
        sc::ts_init(ts, ta2, 0, ta.size(), sc::ts_point_fx::POINT_AVERAGE_VALUE);
        FAST_CHECK_EQ(ts.ta, ta2);
    }
    SUBCASE("change_ta_start_only") {
        auto ta2 = ta;
        ta2.t += sc::deltahours(1);
        rm.run_interpolation(ip, ta2, testenv);
        rm.run_cells();
        FAST_CHECK_EQ((*rm.get_cells())[0].rc.avg_discharge.ta, ta2);
    }
    ptgsk_region_model_t rm_copy(rm);
    auto p1 = rm.get_region_parameter();
    auto p2 = rm_copy.get_region_parameter();
    TS_ASSERT(rm.ncore == rm_copy.ncore);
    TS_ASSERT(p1.kirchner.c1 == p2.kirchner.c1);
    p1.kirchner.c1 += 0.1;
    TS_ASSERT(p1.kirchner.c1 != p2.kirchner.c1);

}

TEST_CASE("test_region_vs_catchment_parameters") {
    using cell_t=pt_gs_k::cell_complete_response_t;
    using region_model_t=sc::region_model<cell_t>;
    using parameter_t=cell_t::parameter_t;

    // The model stack needs its parameters:
    parameter_t gp;

    // And there has to be a start state
    gs::state gss;
    kr::state ks;ks.q = 30.0;
    pt_gs_k::state_t state{gss, ks};

    // most important, there is cells, that are geo-located, have area, mid-point,catchment-id
    sc::geo_cell_data gc1(sc::geo_point(500.0,  50.0,   10.0), 1000.0* 100.0, 0);
    sc::geo_cell_data gc2(sc::geo_point(1500.0, 500.0, 100.0), 1000.0*1000.0, 1);

    cell_t c1,c2;
    c1.geo=gc1;
    c2.geo=gc2;

    auto cells = make_shared<std::vector<cell_t>> ();//ptgsk_cells;
    cells->push_back(c1);
    cells->push_back(c2);
    region_model_t rm(cells,gp);
    TS_ASSERT_EQUALS(rm.number_of_catchments(),2u);
    TS_ASSERT(rm.has_catchment_parameter(0)==false);
    TS_ASSERT(rm.has_catchment_parameter(1)==false); // by default, all should share the global rm parameter
    parameter_t c1p;                                 // now, put a specific parameter to catchment 0
    c1p.kirchner.c1 = -2.5;
    rm.set_catchment_parameter(0,c1p);
    TS_ASSERT(rm.has_catchment_parameter(0)==true);
    TS_ASSERT(rm.has_catchment_parameter(1)==false);
    rm.remove_catchment_parameter(0);
    TS_ASSERT(rm.has_catchment_parameter(0)==false);
    TS_ASSERT(rm.has_catchment_parameter(1)==false);
    TS_ASSERT((*cells)[0].parameter==(*cells)[1].parameter );//ensure they now share the common region-model parameter
    // now test alternate constructor that takes catchment-parameter map as input
    map<int, parameter_t> catchment_params;
    c1p.kirchner.c1 = -2.59;
    catchment_params[1] = c1p;
    rm=region_model_t(cells,gp,catchment_params);// so catchment-id 1 should have different parameters!
    TS_ASSERT(rm.has_catchment_parameter(0)==false);
    TS_ASSERT(rm.has_catchment_parameter(1)==true);
    TS_ASSERT_DELTA(rm.get_region_parameter().kirchner.c1,gp.kirchner.c1,0.000001);// should equal our global constant
    TS_ASSERT_DELTA(rm.get_catchment_parameter(1).kirchner.c1,c1p.kirchner.c1,0.00001);// should equal our special catch-id 1 parameter


}
}

