#include "test_pch.h"
#include "core/skaugen.h"


using namespace shyft::core::skaugen;
typedef calculator<parameter, state, response> SkaugenModel;
TEST_SUITE("skaugen") {
TEST_CASE("test_accumulation") {
    // Model parameters
    const double d_range = 113.0;
    const double unit_size = 0.1;
    const double alpha_0 = 40.77;
    const double max_water_fraction = 0.1;
    const double tx = 0.16;
    const double cx = 2.50;
    const double ts = 0.14;
    const double cfr = 0.01;
    parameter p(alpha_0, d_range, unit_size, max_water_fraction, tx, cx, ts, cfr);

    // Model state variables
    const double alpha = alpha_0;
    const double nu = alpha_0*unit_size;
    const double sca = 0.0;
    const double swe = 0.0;
    const double free_water = 0.0;
    const double residual = 0.0;
    const unsigned long nnn = 0;
    state s(nu, alpha, sca, swe, free_water, residual, nnn);

    // Model input
    shyft::time_series::utctimespan dt = 60*60;
    double temp = -10.0;
    double prec = 10.0;
    double radiation = 0.0;
    double wind_speed = 0.0;
    std::vector<std::pair<double, double>> tp(10, std::pair<double, double>(temp, prec));

    // Accumulate snow
    SkaugenModel model;
    response r;
	for_each(tp.begin(), tp.end(), [&dt, &p, &model, &radiation, &wind_speed, &s, &r](std::pair<double, double> pair) {
            model.step(dt, p, pair.first, pair.second, radiation, wind_speed, s, r);
        });
    TS_ASSERT_DELTA(s.swe*s.sca, prec*tp.size(), 1.0e-6);
    TS_ASSERT_DELTA(s.sca, 1.0, 1.0e-6);
    TS_ASSERT(s.nu < alpha_0*unit_size);
}


TEST_CASE("test_melt") {
    // Model parameters
    const double d_range = 113.0;
    const double unit_size = 0.1;
    const double alpha_0 = 40.77;
    const double max_water_fraction = 0.1;
    const double tx = 0.16;
    const double cx = 2.50;
    const double ts = 0.14;
    const double cfr = 0.01;
    parameter p(alpha_0, d_range, unit_size, max_water_fraction, tx, cx, ts, cfr);

    // Model state variables
    const double alpha = alpha_0;
    const double nu = alpha_0*unit_size;
    const double sca = 0.0;
    const double swe = 0.0;
    const double free_water = 0.0;
    const double residual = 0.0;
    const unsigned long nnn = 0;
    state s(nu, alpha, sca, swe, free_water, residual, nnn);

    // Model input
    shyft::time_series::utctimespan dt = 24*60*60;
    double temp = -10.0;
    double prec = 10.0;
    double radiation = 0.0;
    double wind_speed = 0.0;
    std::vector<std::pair<double, double>> tp(10, std::pair<double, double>(temp, prec));

    // Accumulate snow
    SkaugenModel model;
    response r;
	for_each(tp.begin(), tp.end(), [&dt, &p, &model, &radiation, &wind_speed, &s, &r](std::pair<double, double> pair) {
            model.step(dt, p, pair.first, pair.second, radiation, wind_speed, s, r);
        });

    const double total_water = s.swe*s.sca;
    double agg_outflow(0.0); // For checking mass balance

    // Single melt event
    tp = std::vector<std::pair<double, double>>(1, std::pair<double, double>(10.0, 0.0)); // No precip, but 10.0 degrees for one day
	for_each(tp.begin(), tp.end(), [&dt, &p, &model, &radiation, &wind_speed, &s, &r, &agg_outflow](std::pair<double, double> pair) {
            model.step(dt, p, pair.first, pair.second, radiation, wind_speed, s, r);
            agg_outflow += r.outflow;
        });
    const double total_water_after_melt = s.sca*(s.swe + s.free_water);
    TS_ASSERT(total_water_after_melt <= total_water); // Less water after melt due to runoff
    TS_ASSERT(r.outflow + s.free_water >= 1.0); // Some runoff or free water in snow
    TS_ASSERT_DELTA(r.outflow + s.sca*(s.free_water + s.swe), total_water, 1.0e-6);

    // One hundred melt events, that should melt everything
    tp = std::vector<std::pair<double, double>>(100, std::pair<double, double>(10.0, 0.0));
    for_each(tp.begin(), tp.end(), [&dt , &p, &model, &radiation, &wind_speed, &s, &r, &agg_outflow] (std::pair<double, double> pair) {
            model.step(dt, p, pair.first, pair.second, radiation, wind_speed, s, r);
            agg_outflow += r.outflow;
        });

    TS_ASSERT_DELTA(s.sca, 0.0, 1.0e-6);
    TS_ASSERT_DELTA(s.swe, 0.0, 1.0e-6);
    TS_ASSERT_DELTA(agg_outflow, total_water, 1.0e-10);
    TS_ASSERT_DELTA(s.alpha, alpha_0, 1.0e-6);
    TS_ASSERT_DELTA(s.nu, alpha_0*unit_size, 1.0e-6);
}


TEST_CASE("test_lwc") {
    // Model parameters
    const double d_range = 113.0;
    const double unit_size = 0.1;
    const double alpha_0 = 40.77;
    const double max_water_fraction = 0.1;
    const double tx = 0.16;
    const double cx = 2.50;
    const double ts = 0.14;
    const double cfr = 0.01;
    parameter p(alpha_0, d_range, unit_size, max_water_fraction, tx, cx, ts, cfr);

    // Model state variables
    const double alpha = alpha_0;
    const double nu = alpha_0*unit_size;
    const double sca = 0.0;
    const double swe = 0.0;
    const double free_water = 0.0;
    const double residual = 0.0;
    const unsigned long nnn = 0;
    state s(nu, alpha, sca, swe, free_water, residual, nnn);

    // Model input
    shyft::time_series::utctimespan dt = 24*60*60;
    double temp = -10.0;
    double prec = 10.0;
    double radiation = 0.0;
    double wind_speed = 0.0;
    std::vector<std::pair<double, double>> tp(10, std::pair<double, double>(temp, prec));

    // Accumulate snow
    SkaugenModel model;
    response r;
    for_each(tp.begin(), tp.end(), [&dt , &p, &model, &radiation, &wind_speed, &s, &r] (std::pair<double, double> pair) {
            model.step(dt, p, pair.first, pair.second, radiation, wind_speed, s, r);
        });

    TS_ASSERT_DELTA(s.free_water, 0.0, 1.0e-6);  // No free water when dry snow precip
    model.step(dt, p, 10.0, 0.0, radiation, wind_speed, s, r);
    TS_ASSERT(s.free_water <= s.swe*max_water_fraction);  // Can not have more free water in the snow than the capacity of the snowpack
    tp = std::vector<std::pair<double, double>>(5, std::pair<double, double>(10.0, 0.0));
    for_each(tp.begin(), tp.end(), [&dt , &p, &model, &radiation, &wind_speed, &s, &r] (std::pair<double, double> pair) {
            model.step(dt, p, pair.first, pair.second, radiation, wind_speed, s, r);
        });
    TS_ASSERT_DELTA(s.free_water, s.swe*max_water_fraction, 1.0e-6);  // Test that snow is saturated with free water

    return;
}
}
