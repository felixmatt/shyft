#include "test_pch.h"
#include "mocks.h"
#include "core/gamma_snow.h"
//#include "core/region_model.h"

namespace shyfttest {

    const double EPS = 1.0e-10;

};
using namespace shyft::core;
namespace gs=shyft::core::gamma_snow;
class gamma_snow_test {
public:
    template <class GS_CALC,class P>
    void reset_snow_pack(GS_CALC&gsc, double& sca, double& lwc, double& alpha, double& sdc_melt_mean, double& acc_melt,
        double& temp_swe, const double storage, const P& p) const {
        gsc.reset_snow_pack(sca, lwc, alpha, sdc_melt_mean, acc_melt, temp_swe, storage, p);
    }
    template <class GS_CALC>
    void calc_snow_state(GS_CALC&gsc, const double shape, const double scale, const double y0, const double lambda,
        const double lwd, const double max_water_frac, const double temp_swe,
        double& swe, double& sca) const {
        gsc.calc_snow_state(shape, scale, y0, lambda, lwd, max_water_frac, temp_swe, swe, sca);
    }
    template <class GS_CALC>
    double corr_lwc(GS_CALC&gsc, const double z1, const double a1, const double b1,
        double z2, const double a2, const double b2) const {
        return gsc.corr_lwc(z1, a1, b1, z2, a2, b2);
    }
};
static gamma_snow_test fix; // access to private scope
TEST_SUITE("gamma_snow") {
TEST_CASE("test_reset_snow_pack_zero_storage") {

    gs::calculator<gs::parameter, gs::state, gs::response> gs;
    gs::parameter p;
    double sca;
    double lwc;
    double alpha;
    double sdc_melt_mean;
    double acc_melt;
    double temp_swe;
    const double storage = 0.0;
    fix.reset_snow_pack(gs,sca, lwc, alpha, sdc_melt_mean, acc_melt, temp_swe, storage, p);
    TS_ASSERT_DELTA(sca, 0.0, shyfttest::EPS);
    TS_ASSERT_DELTA(lwc, 0.0, shyfttest::EPS);
    TS_ASSERT_DELTA(alpha, 1.0/(p.snow_cv*p.snow_cv), shyfttest::EPS);
    TS_ASSERT_DELTA(sdc_melt_mean, 0.0, shyfttest::EPS);
    TS_ASSERT_DELTA(acc_melt, -1.0, shyfttest::EPS);
    TS_ASSERT_DELTA(temp_swe, 0.0, shyfttest::EPS);
}


TEST_CASE("test_reset_snow_pack_with_storage") {

	gs::calculator<gs::parameter, gs::state, gs::response> gs;
	gs::parameter p;
    double sca;
    double lwc;
    double alpha;
    double sdc_melt_mean;
    double acc_melt;
    double temp_swe;
    const double storage = 1.0;

    fix.reset_snow_pack(gs,sca, lwc, alpha, sdc_melt_mean, acc_melt, temp_swe, storage, p);
    TS_ASSERT_DELTA(sca, 1.0 - p.initial_bare_ground_fraction, shyfttest::EPS);
    TS_ASSERT_DELTA(lwc, 0.0, shyfttest::EPS);
    TS_ASSERT_DELTA(alpha, 1.0/(p.snow_cv*p.snow_cv), shyfttest::EPS);
    TS_ASSERT_DELTA(sdc_melt_mean, storage/sca, shyfttest::EPS);
    TS_ASSERT_DELTA(acc_melt, -1.0, shyfttest::EPS);
    TS_ASSERT_DELTA(temp_swe, 0.0, shyfttest::EPS);
}

TEST_CASE("test_calculate_snow_state") {
    // This is a pure regression test, based on some standard values and recorded response

	gs::calculator<gs::parameter, gs::state, gs::response> gs;

    const double shape = 1.0/(0.4*0.4);
    const double scale = 0.4/shape;
    const double y0 = 0.04;
    const double lambda = 0.0;
    const double lwd = 0.0;
    const double max_water_frac = 0.1;
    const double temp_swe = 0.0;
    double swe; // Output from gs.calc_snow_state(...)
    double sca; // Output from gs.calc_snow_state(...)
    fix.calc_snow_state(gs,shape, scale, y0, lambda, lwd, max_water_frac, temp_swe, swe, sca);
    TS_ASSERT_DELTA(swe, 0.384, shyfttest::EPS);
    TS_ASSERT_DELTA(sca, 0.96, shyfttest::EPS);
}

TEST_CASE("test_correct_lwc") {
    // This is a pure regression test, based on a set of input values and recorded response

	gs::calculator<gs::parameter, gs::state, gs::response> gs;
    const double z1 = 4.0;
    const double a1 = 6.0;
    const double b1 = 1.0;
    const double z2 = 5.0;
    const double a2 = 5.0;
    const double b2 = 2.0;

    const double result = fix.corr_lwc(gs,z1, a1, b1, z2, a2, b2);
// as a result of using lower resolution, less accuracy
    TS_ASSERT_DELTA(result, 3.8411,0.0001);
    // would throw in previous versions
    fix.corr_lwc(gs, 1.0, 6.25, 0.005358, 0.5, 6.25, 0.005358);

    fix.corr_lwc(gs, 0.0, a1, b1, z2, a2, b2);
    fix.corr_lwc(gs, z1, a1, b1, 0.0, a2, b2);

}

TEST_CASE("test_warm_winter_effect") {
	gs::calculator<gs::parameter, gs::state, gs::response> gs;
    double tx = -0.5;
    double wind_scale = 2.0;
    double wind_const = 1.0;
    double max_lwc = 0.1;
    double surf_mag = 30.0;
    double max_alb = 0.9;
    double min_alb = 0.6;
    double fadr = 5.0;
    double sadr = 5.0;
    double srd = 5.0;
    double glac_alb = 0.4;
	gs::parameter param(100, tx, wind_scale, wind_const, max_lwc, surf_mag, max_alb, min_alb, fadr, sadr, srd, glac_alb);
    param.initial_bare_ground_fraction=0.04;
    param.snow_cv=0.5;
    // Value for node 94
    //param.set_glacier_fraction(0.12);
    //double albedo = 0.6;
    //double lwc = 3133.46;
    //double surface_heat = 0.0;
    //double alpha = 3.9589693546295166;
    //double sdc_melt_mean = 1474.5162353515625;
    //double acc_melt = 2669.209716796875;
    //double iso_pot_energy = 1698.521484375;
    //double temp_swe = 0.0;

    // Value for node 231
    double albedo = 0.6;
    double lwc = 3148.9609375;
    double surface_heat = 0.0;
    double alpha = 3.960848093032837;
    double sdc_melt_mean = 1525.66064453125;
    double acc_melt = 2753.03076171875;
    double iso_pot_energy = 1752.56396484375;
    double temp_swe = 0.0;
    gs::state states(albedo, lwc, surface_heat, alpha, sdc_melt_mean, acc_melt, iso_pot_energy, temp_swe);
    gs::response response;

    double temp = 7.99117956; // With this temperature, the GammaSnow method fails at winter end
    double rad = 0;
    double prec = 0.0;
    double wind_speed = 2.0;
    double rel_hum = 0.70;
    auto dt = shyft::core::deltahours(1);
    size_t num_days = 100;
	bool verbose = getenv("SHYFT_VERBOSE") ? true : false;
    for (size_t i = 0; i < 24*num_days; ++i) {
        gs.step(states, response, dt*24*232+i*dt, dt, param, temp, rad, prec, wind_speed, rel_hum,0.0,0.0);

		if (verbose && (i % 10 == 0)) {
            std::cout << "Time step: " << i << ", ";
            std::cout << response.storage << ", ";
            std::cout << response.sca << ", ";
            std::cout << states.lwc << ", ";
            std::cout << response.outflow << std::endl;
        }
    }
}

TEST_CASE("test_step") {

	gs::calculator<gs::parameter, gs::state, gs::response> gs;
	gs::parameter param;
    shyft::time_series::utctime dt = shyft::core::deltahours(1);

	gs::state states(0.0, 1.0, 0.0, 1.0 / (param.snow_cv*param.snow_cv), 10.0, -1.0, 0.0, 0.0);
	gs::response response;

    double temp = 1.0;
    double rad = 10.0;
    double prec = 5.0;
    double wind_speed = 2.0;
    double rel_hum = 0.70;

    const std::clock_t start = std::clock();
    for (size_t i=0; i<365; ++i) {
        gs.step(states, response, i*dt, dt, param, i&1 ?temp:param.tx, rad, prec, wind_speed, rel_hum,0.0,0.0);
    }
    const std::clock_t total = std::clock() - start;
    if(getenv("SHYFT_VERBOSE")) {
        std::cout << "Final result: " << response.outflow << std::endl;
        std::cout << "One year of (simple) snow algorithm took: " << 1000*(total)/(double)(CLOCKS_PER_SEC) << " ms" << std::endl;
    }
}


TEST_CASE("test_output_independent_of_timestep") {
	gs::calculator<gs::parameter, gs::state, gs::response> gs;
	gs::parameter param;
    auto dt = shyft::core::deltahours(1);
    auto dt3h=shyft::core::deltahours(3);

	gs::state initial_state(0.0, 0.0, 0.0, 1.0 / (param.snow_cv*param.snow_cv), 0.0, -1.0, 0.0, 0.0);
    double rad = 10.0;
    double prec = 5.0;
    double wind_speed = 2.0;
    double rel_hum = 0.70;

    for(size_t j=0;j<2;j++) {

        double temp = j==0?10.0:-10.0;
        // first test with hot scenario, no snow, to get equal output (response is nonlinear)
        // then with cold,
        gs::state state1h(initial_state);
        gs::response response1h;
        double output_hour=0;
        for(size_t i=0;i<3;++i) {
            gs.step(state1h,response1h,i*dt,dt,param,temp,rad,prec,wind_speed,rel_hum,0.0,0.0);
            output_hour+=response1h.outflow; // mm_h
        }
        output_hour/=3.0; // avg hour output
        gs::state state3h(initial_state);
        gs::response response3h;
        gs.step(state3h,response3h,0*dt3h,dt3h,param,temp,rad,prec,wind_speed,rel_hum,0.0,0.0);
        // require same output on the average, and same lwc
        TS_ASSERT_DELTA(output_hour,response3h.outflow,0.00001);
        TS_ASSERT_DELTA(state1h.lwc,state3h.lwc,0.000001);
        TS_ASSERT_DELTA(response1h.storage,response3h.storage,0.00001);
    }

}
TEST_CASE("test_forest_altitude_dependent_snow_cv") {
    gs::parameter p;
    TS_ASSERT_DELTA(p.effective_snow_cv(1.0,1000),p.snow_cv,0.0000001);// assume default no effect, backward compatible
    p.snow_cv_forest_factor=0.1; // forest fraction 1.0 should add 0.1 to the snow_cv
    p.snow_cv_altitude_factor=0.0001;// at 1000m, add 0.1 to the factor
    TS_ASSERT_DELTA(p.effective_snow_cv(0.0,0.0),p.snow_cv,0.0000001);// verify no increase of forest=0 and altitude=0
    TS_ASSERT_DELTA(p.effective_snow_cv(1.0,0.0),p.snow_cv+0.1,0.0000001);// verify increase in forest direction
    TS_ASSERT_DELTA(p.effective_snow_cv(0.0,1000.0),p.snow_cv +0.1, 0.0000001);// verify increase in altitude direction
}
}
