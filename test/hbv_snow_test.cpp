#include "test_pch.h"
#include "core/hbv_snow.h"

using namespace shyft::core;
using namespace shyft::core::hbv_snow;
using namespace std;

using SnowModel = calculator<parameter, state>;
TEST_SUITE("hbv_snow") {
TEST_CASE("test_integral_calculations") {
    vector<double> f = {0.0, 0.5, 1.0};
    vector<double> x = {0.0, 0.5, 1.0};

    double pt0 = 0.25;
    double pt1 = 0.75;
    TS_ASSERT_DELTA(integrate(f, x, x.size(), 0.0, 1.0), 0.5, 1.0e-12);
    TS_ASSERT_DELTA(integrate(f, x, x.size(), 0.0, pt0) + integrate(f, x, x.size(), pt0, 1.0),
                    integrate(f, x, x.size(), 0.0, 1.0), 1.0e-12);

    TS_ASSERT_DELTA(integrate(f, x, x.size(), pt0, pt1),
                    integrate(f, x, x.size(), 0.0, 1.0) - (integrate(f, x, x.size(), 0.0, pt0)
                                                                   + integrate(f, x, x.size(), pt1, 1.0)), 1.0e-12);

}

TEST_CASE("test_mass_balance_at_snowpack_reset") {
    vector<double> s = {1.0, 1.0, 1.0, 1.0, 1.0};
    vector<double> a = {0.0, 0.25, 0.5, 0.75, 1.0};
    parameter p(s, a);
    state state;
    response r;
    utctime t0 = 0;
    utctime t1 = 3600; // One hour
    double precipitation = 0.04;
    double temperature = 1.0;
    double sca = 1.0;
    double swe = 0.05;
    state.swe=swe;
    state.sca=sca;
    double total_water_before = precipitation + swe;
    SnowModel snow_model(p, state);
    snow_model.step(state, r, t0, t1, p, precipitation, temperature);
    double total_water_after = state.swe + r.outflow;
    TS_ASSERT_DELTA(total_water_before, total_water_after, 1.0e-8);
}

TEST_CASE("test_mass_balance_at_snowpack_buildup") {
    vector<double> s = {1.0, 1.0, 1.0, 1.0, 1.0};
    vector<double> a = {0.0, 0.25, 0.5, 0.75, 1.0};
    parameter p(s, a);
    state state;
    response r;

    utctime t0 = 0;
    utctime t1 = 3600; // One hour
    double precipitation = 0.15;
    double temperature = -1.0;
    double sca = 0.6;
    double swe = 0.2;
    state.swe=swe;
    state.sca=sca;
    double total_water_before = precipitation + swe;
    SnowModel snow_model(p, state);
    snow_model.step(state, r, t0, t1, p, precipitation, temperature);
    double total_water_after = state.swe + r.outflow;
    TS_ASSERT_DELTA(total_water_before, total_water_after, 1.0e-8);
    state.swe = 0.2;
    state.sca = 0.6;
    temperature=p.tx;// special check fo tx
    SnowModel snow_model2(p, state);
    snow_model2.step(state, r, t0, t1, p, precipitation, temperature);
    TS_ASSERT_DELTA(total_water_before,state.swe+r.outflow, 1.0e-8);

}

TEST_CASE("test_mass_balance_rain_no_snow") {
    vector<double> s = {1.0, 1.0, 1.0, 1.0, 1.0};
    vector<double> a = {0.0, 0.25, 0.5, 0.75, 1.0};
    parameter p(s, a);
    state state;
    response r;

    utctime t0 = 0;
    utctime t1 = 3600; // One hour
    double precipitation = 0.15;
    double temperature = p.tx;
    double sca = 0.0;
    double swe = 0.0;
    state.swe=swe;
    state.sca=sca;
    double total_water_before = precipitation + swe;
    SnowModel snow_model(p, state);
    snow_model.step(state, r, t0, t1, p, precipitation, temperature);
    double total_water_after = state.swe + r.outflow;
    TS_ASSERT_DELTA(total_water_before, total_water_after, 1.0e-8);
    TS_ASSERT_DELTA(state.sca, 0.0, 1.0e-8);
    TS_ASSERT_DELTA(state.swe, 0.0, 1.0e-8);
}

TEST_CASE("test_mass_balance_melt_no_precip") {
    vector<double> s = {1.0, 1.0, 1.0, 1.0, 1.0};
    vector<double> a = {0.0, 0.25, 0.5, 0.75, 1.0};
    parameter p(s, a);
    state state;
    response r;

    utctime t0 = 0;
    utctime t1 = 3600; // One hour
    double precipitation = 0.0;
    double temperature = 3.0;
    double sca = 0.5;
    double swe = 10.0;
    state.swe=swe;
    state.sca=sca;
    double total_water_before = precipitation + swe;
    SnowModel snow_model(p, state);

    snow_model.step(state, r, t0, t1, p, precipitation, temperature);
    double total_water_after = state.swe + r.outflow;
    TS_ASSERT_DELTA(total_water_before, total_water_after, 1.0e-8);
}

}
/* vim: set filetype=cpp: */
