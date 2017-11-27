#include "test_pch.h"
#include "core/pt_gs_k.h"
#include "core/cell_model.h"
#include "core/pt_gs_k_cell_model.h"
#include "core/geo_cell_data.h"
#include "core/geo_point.h"
#include "mocks.h"
#include "core/time_series.h"
#include "core/utctime_utilities.h"

// Some typedefs for clarity
using namespace shyft::core;
using namespace shyft::time_series;

using namespace shyft::core::pt_gs_k;
using namespace shyfttest;
using namespace shyfttest::mock;

namespace pt = shyft::core::priestley_taylor;
namespace gs = shyft::core::gamma_snow;
namespace kr = shyft::core::kirchner;
namespace ae = shyft::core::actual_evapotranspiration;
namespace pc = shyft::core::precipitation_correction;
namespace ta = shyft::time_axis;
typedef TSPointTarget<ta::point_dt> catchment_t;

static std::ostream& operator<<(std::ostream& os, const point& pt) {
    os << calendar().to_string(pt.t) << ", " << pt.v;
    return os;
}
TEST_SUITE("pt_gs_k")  {
TEST_CASE("test_call_stack") {
    xpts_t temp;
    xpts_t prec;
    xpts_t rel_hum;
    xpts_t wind_speed;
    xpts_t radiation;

    calendar cal;
    utctime t0 = cal.time(YMDhms(2014, 8, 1, 0, 0, 0));
    size_t n_ts_points = 3;
    utctimespan dt=deltahours(24);
    utctime t1 = t0 + n_ts_points*dt;
    shyfttest::create_time_series(temp, prec, rel_hum, wind_speed, radiation, t0, dt, n_ts_points);

    utctime model_dt = deltahours(24);
    vector<utctime> times;
    for (utctime i=t0; i <= t1; i += model_dt)
        times.emplace_back(i);

    ta::point_dt time_axis(times);
    times.emplace_back(t1+model_dt);
    ta::point_dt state_axis(times);

    // Initialize parameters
    pt::parameter pt_param;
    gs::parameter gs_param;
    ae::parameter ae_param;
    kr::parameter k_param;
    //CellParameter c_param;
    pc::parameter p_corr_param;

    // Initialize the state vectors
    kr::state kirchner_state{5.0};
    gs::state gs_state(0.0, 1.0, 0.0, 1.0/(gs_param.snow_cv*gs_param.snow_cv), 10.0, -1.0, 0.0, 0.0);

    // Initialize response
    //response response;

    // Initialize collectors
    shyfttest::mock::PTGSKResponseCollector response_collector(time_axis.size());
    shyfttest::mock::StateCollector<ta::point_dt> state_collector(state_axis);

    state state{gs_state, kirchner_state};
    parameter parameter{pt_param, gs_param, ae_param, k_param,  p_corr_param};
    geo_cell_data geo_cell;
    pt_gs_k::run_pt_gs_k<shyft::time_series::direct_accessor,
                         response>(geo_cell, parameter, time_axis,0,0, temp, prec, wind_speed,
                                     rel_hum, radiation, state, state_collector, response_collector);
    for(size_t i=0;i<n_ts_points+1;++i) { // state have one extra point
        TS_ASSERT(state_collector._inst_discharge.value(i)>0.0001);// verify there are different from 0.0 filled in for all time-steps
    }
}

TEST_CASE("test_raster_call_stack") {
    //using shyfttest::TSSource;

    typedef MCell<response, state, parameter, xpts_t> PTGSKCell;
    calendar cal;
    utctime t0 = cal.time(YMDhms(2014, 8, 1, 0, 0, 0));
    const int nhours=1;
    utctimespan dt=1*deltahours(nhours);
    utctimespan model_dt = 1*deltahours(nhours);
    size_t n_ts_points = 365;
    utctime t1 = t0 + n_ts_points*dt;

    vector<utctime> times;
    for (utctime i=t0; i <= t1; i += model_dt)
        times.emplace_back(i);
    ta::point_dt time_axis(times);
    times.emplace_back(t1+model_dt);
    ta::point_dt state_axis(times);

    // 10 catchments numbered from 0 to 9.
    std::vector<catchment_t> catchment_discharge;
    catchment_discharge.reserve(10);
    for (size_t i = 0; i < 10; ++i)
        catchment_discharge.emplace_back(time_axis);

    size_t n_dests = 10*100;
    std::vector<PTGSKCell> model_cells;
    model_cells.reserve(n_dests);

    pt::parameter pt_param;
    gs::parameter gs_param;
    ae::parameter ae_param;
    kr::parameter k_param;
    pc::parameter p_corr_param;

    xpts_t temp;
    xpts_t prec;
    xpts_t rel_hum;
    xpts_t wind_speed;
    xpts_t radiation;

    kr::state kirchner_state{5.0};
    gs::state gs_state(0.6, 1.0, 0.0, 1.0/(gs_param.snow_cv*gs_param.snow_cv), 10.0, -1.0, 0.0, 0.0);
    state state{gs_state, kirchner_state};


    parameter parameter{pt_param, gs_param, ae_param, k_param, p_corr_param};

    for (size_t i = 0; i < n_dests; ++i) {
        shyfttest::create_time_series(temp, prec, rel_hum, wind_speed, radiation, t0, dt, n_ts_points);
        state.gs.albedo += 0.3*(double)i/(n_dests - 1); // Make the snow albedo differ at each destination.
        model_cells.emplace_back(temp, prec, wind_speed, rel_hum, radiation, state, parameter, i % 3);
    }


    const std::clock_t start = std::clock();
    for_each(model_cells.begin(), model_cells.end(), [&time_axis,&state_axis] (PTGSKCell& d) mutable {
        auto time = time_axis.time(0);

        shyfttest::mock::StateCollector<ta::point_dt> sc(state_axis);
        shyfttest::mock::DischargeCollector<ta::point_dt> rc(1000 * 1000, time_axis);
        //PTGSKResponseCollector rc(time_axis.size());

        pt_gs_k::run_pt_gs_k<shyft::time_series::direct_accessor, response>(d.geo_cell_info(), d.parameter(), time_axis,0,0,
              d.temperature(),
              d.precipitation(),
              d.wind_speed(),
              d.rel_hum(),
              d.radiation(),
              d.get_state(time),
              sc,
              rc
              );
    });
    bool verbose= getenv("SHYFT_VERBOSE")!=nullptr;
    if(verbose) {
        for (size_t i=0; i < 3; ++i)
            std::cout << "Catchment "<< i << " first total discharge = " << catchment_discharge.at(i).value(0) << std::endl;
        for (size_t i=0; i < 3; ++i)
            std::cout << "Catchment "<< i << " second total discharge = " << catchment_discharge.at(i).value(1) << std::endl;
        for (size_t i=0; i < 3; ++i)
            std::cout << "Catchment "<< i << " third total discharge = " << catchment_discharge.at(i).value(2) << std::endl;

        const std::clock_t total = std::clock() - start;
        std::cout << "One year and " << n_dests << " destinatons with catchment discharge aggregation took: " << 1000*(total)/(double)(CLOCKS_PER_SEC) << " ms" << std::endl;
    }

}

TEST_CASE("test_mass_balance") {
    calendar cal;
    utctime t0 = cal.time(2014, 8, 1, 0, 0, 0);
    utctimespan dt=deltahours(1);
    const int n=1;
    ta::fixed_dt tax(t0,dt,n);
	ta::fixed_dt tax_state(t0, dt, n + 1);
    pt::parameter pt_param;
    gs::parameter gs_param;
    ae::parameter ae_param;
    kr::parameter k_param;
    pc::parameter p_corr_param;
    parameter parameter{pt_param, gs_param, ae_param, k_param, p_corr_param};

    pts_t temp(tax,15.0);
    pts_t prec(tax,3.0);
    pts_t rel_hum(tax,0.8);
    pts_t wind_speed(tax,2.0);
    pts_t radiation(tax,300.0);

    kr::state kirchner_state{5.0};// 5 mm in storage/state
    gs::state gs_state;// zero snow (0.6, 1.0, 0.0, 1.0/(gs_param.snow_cv*gs_param.snow_cv), 10.0, -1.0, 0.0, 0.0);
    gs_state.lwc=0.0;
    gs_state.acc_melt=-1; // zero snow, precipitation goes straight through
    state state{gs_state, kirchner_state};
    pt_gs_k::state_collector sc;
    pt_gs_k::all_response_collector rc;
    const double cell_area=1000*1000;
    sc.collect_state=true;
    sc.initialize(tax_state,0,0,cell_area);
    rc.initialize(tax,0,0,cell_area);
    geo_cell_data gcd(geo_point(1000,1000,100));
    pt_gs_k::run_pt_gs_k<direct_accessor,pt_gs_k::response>(gcd,parameter,tax,0,0,temp,prec,wind_speed,rel_hum,radiation,state,sc,rc);

    // test strategy:
    // let it rain constantly, let's say 3 mm/h,
    //   when t goes to +oo the kirchner output should be 3 mm/h - act. evapotrans..
    //   so precipitation goes in to the system, and out goes actual evapotranspiration and kirchner response q.
    for(size_t i=0;i<10000;i++) {
        pt_gs_k::run_pt_gs_k<direct_accessor,pt_gs_k::response>(gcd,parameter,tax,0,0,temp,prec,wind_speed,rel_hum,radiation,state,sc,rc);
    }
    TS_ASSERT_DELTA(rc.avg_discharge.value(0)*dt*1000/cell_area + rc.ae_output.value(0), prec.value(0),0.0000001);
	TS_ASSERT_DELTA(rc.snow_outflow.value(0)*dt * 1000 / cell_area, prec.value(0), 0.0000001);//verify snow out is m3/s
    SUBCASE("direct_response_on_bare_lake_only") {
        // verify that when it rains, verify that only (1-snow.sca)*(lake+rsv).fraction * precip_mm *cell_area goes directly to response.
        // case 1: no snow-covered area, close to zero in kirchner q, expect all rain directly
        land_type_fractions ltf(0.0,0.5,0.5,0.0,0.0);// all is lake and reservoir
        gcd.set_land_type_fractions(ltf);
        state.kirchner.q=1e-4;//
        state.gs.lwc=0.0;// no snow, should have direct response, 3 mm/h, over the cell_area
        state.gs.acc_melt=-1;// should be no snow, should go straight through
        pt_gs_k::run_pt_gs_k<direct_accessor,pt_gs_k::response>(gcd,parameter,tax,0,0,temp,prec,wind_speed,rel_hum,radiation,state,sc,rc);
        TS_ASSERT_DELTA(rc.avg_discharge.value(0)*dt*1000.0/cell_area,prec.value(0),0.001);
        state.gs.lwc=1.0;
        state.gs.acc_melt=300.0;
        temp.v[0]=-10.0;//it's cold
        pt_gs_k::run_pt_gs_k<direct_accessor,pt_gs_k::response>(gcd,parameter,tax,0,0,temp,prec,wind_speed,rel_hum,radiation,state,sc,rc);
        TS_ASSERT_DELTA(rc.snow_sca.value(0),0.96,0.01);// almost entirely covered by snow, so we should have 0.04 of rain direct response
        TS_ASSERT_DELTA(rc.avg_discharge.value(0)*dt*1000.0/cell_area,0.04*prec.value(0),0.02);
        temp.v[0]=10.0;// heat is on, melt snow, .9 should end up in kirchner it's cold
        //state.gs.sdc_melt_mean=10.0;
        state.gs.acc_melt=5.0;//simulate early autumn, melt out everything
        state.gs.temp_swe = 3.0;//
        state.gs.lwc = 10.0;//
        for(size_t i=0;i<5000;++i){
            pt_gs_k::run_pt_gs_k<direct_accessor,pt_gs_k::response>(gcd,parameter,tax,0,0,temp,prec,wind_speed,rel_hum,radiation,state,sc,rc);
            if(rc.snow_sca.value(0)<0.1)
                break;// melt done, everything should be in kirchner response by now
        }
        TS_ASSERT_DELTA(rc.snow_sca.value(0),0.0,0.1);// almost entirely covered by snow, so we should have 0.04 of rain direct response
        TS_ASSERT_DELTA(rc.avg_discharge.value(0),0.8333,0.001);//empirical

    }

}


}
