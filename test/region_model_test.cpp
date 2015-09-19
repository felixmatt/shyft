#include "test_pch.h"
#include "region_model_test.h"



// from core pull in the basic templated algorithms
#include "core/utctime_utilities.h"
#include "core/geo_cell_data.h"


#include "core/timeseries.h"
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
namespace ec = shyft::core;
namespace et = shyft::timeseries;
namespace em = shyft::core;
namespace pt = shyft::core::priestley_taylor;
namespace pc = shyft::core::precipitation_correction;
namespace gs = shyft::core::gamma_snow;
namespace ae = shyft::core::actual_evapotranspiration;
namespace kr = shyft::core::kirchner;
namespace pt_gs_k = shyft::core::pt_gs_k;

// and typedefs for commonly used types in this test
typedef et::point_timeseries<et::timeaxis> pts_t;
typedef et::constant_timeseries<et::timeaxis> cts_t;
typedef et::timeaxis ta_t;

void region_model_test::test_build(void) {

    // arrange
    ec::calendar cal;
    auto start=cal.time(ec::YMDhms(2015,1,1,12,00,00));
    auto dt   =ec::deltahours(1);
    size_t n  =24*10;

    ta_t ta(start,dt,n);
    pts_t x;
    pts_t prec(ta,2.0);
    pts_t rad (ta,100.0);
    pts_t temp(ta,-5.0);
    pts_t windspd(ta,2.0);
    pts_t rhum(ta,60.0);
    cts_t chum(ta,60.0);
    auto env=em::create_cell_environment<ta_t>(x,temp,rad,windspd,rhum);
    env.init(ta);
    TS_ASSERT_EQUALS(env.temperature.size(),n);

    // create one cell..

    // GeoCell, easy to understand
    ec::geo_cell_data gc1(ec::geo_point(100.0,100.0,10.0),1000.0*1000.0,0);
    ec::geo_cell_data gc2(ec::geo_point(100.0,100.0,10.0),1000.0*1000.0,1);
    // The model stack needs its parameters:
    pt::parameter pt;
    gs::parameter gs;
    ae::parameter ae;
    kr::parameter kp;
    pc::parameter scp;
    shared_ptr<pt_gs_k::parameter_t> gp(new pt_gs_k::parameter_t{pt,gs,ae,kp,scp});


    // And there has to be a start state

    gs::state gss;
    kr::state ks;ks.q=30.0;
    pt_gs_k::state_t state{gss,ks};


    pt_gs_k::cell_complete_response_t c1;
    pt_gs_k::cell_discharge_response_t c2;

    c1.geo=gc1;
    c1.env_ts=env;
    c1.set_parameter(gp);

    c1.init_env_ts(ta);// zero,
    c1.env_ts.precipitation=prec;// fix some numbers
    c1.env_ts.temperature=temp;
    c1.env_ts.rel_hum=rhum;
    c1.set_state(state);
    c1.run(ta);

    c2.geo=gc2;
    c2.env_ts=env;
    c2.set_parameter(gp);

    c2.init_env_ts(ta);// zero,
    c2.env_ts.precipitation.fill(5.0);// fix some numbers
    c2.env_ts.temperature.fill(3.0);
    c2.env_ts.rel_hum=rhum;
    c2.set_state(state);
    c2.run(ta);

    TS_ASSERT_EQUALS(c1.rc.snow_swe.size(),ta.size());

    // now build region environment
    typedef em::geo_point_ts<pts_t> gpts_t;
    typedef em::geo_point_ts<cts_t> gcts_t;

    ec::geo_point s1(2000.0,2000.0,10.0);
    ec::geo_point s2(250.0,1500.0,200.0);
    gpts_t gprec{s1,prec};
    gpts_t gtemp {s1,temp};
    pts_t temp2=temp;
    temp2.fill(3.0);
    gpts_t gtemp2 {s2,temp2};
    //<class PS,class TS,class RS,class HS,class WS>
    typedef em::region_environment<gpts_t,gpts_t,gpts_t,gcts_t,gcts_t> test_env_t;
    test_env_t testenv;
    testenv.temperature= make_shared<vector<gpts_t>>();
    testenv.temperature->push_back(gtemp);
    testenv.temperature->push_back(gtemp2);
    testenv.precipitation=make_shared<vector<gpts_t>>();
    testenv.precipitation->push_back(gtemp);
    testenv.precipitation->push_back(gtemp2);

    typedef em::region_model<pt_gs_k::cell_complete_response_t> ptgsk_region_model_t;
    auto ptgsk_cells= make_shared<std::vector<pt_gs_k::cell_complete_response_t>> ();//ptgsk_cells;
    auto c1b=c1;
    c1b.geo.set_catchment_id(1);
    ptgsk_cells->push_back(c1);
    ptgsk_cells->push_back(c1b);
    map<size_t,pt_gs_k::parameter_t> catchment_params;
    auto c1p=pt_gs_k::parameter_t{pt,gs,ae,kp,scp};
    c1p.kirchner.c1=-2.5;
    catchment_params[1]=c1p;

    ptgsk_region_model_t rm(ptgsk_cells, *gp,catchment_params);
    TS_ASSERT(rm.has_catchment_parameter(1));
    auto &c1pr=rm.get_catchment_parameter(1);
    TS_ASSERT_DELTA(c1pr.kirchner.c1,-2.5,0.0001);

    ec::interpolation_parameter ip;
    rm.run_interpolation(ip,ta,testenv);
    auto tsz1=c1.env_ts.temperature.size();
    rm.run_cells();
    tsz1=c1.env_ts.temperature.size();
    TS_ASSERT(tsz1>0);


}
