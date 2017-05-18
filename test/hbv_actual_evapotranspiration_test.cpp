#include "test_pch.h"
#include "core/hbv_actual_evapotranspiration.h"
#include "core/utctime_utilities.h"

namespace shyfttest {
	const double EPS = 1.0e-8;
}

using namespace shyft::core;
using namespace shyft::core::hbv_actual_evapotranspiration;
TEST_SUITE("hbv_actual_evapotranspiration") {
TEST_CASE("test_soil_moisture") {
	const double sca = 0.0;
	const double pot_evap = 5.0; // [mm/h]
	const double lp = 150.0;
	const utctime dt = deltahours(3);
	double act_evap;
	act_evap = calculate_step(0.0, pot_evap, lp, sca, dt);
	TS_ASSERT_DELTA(act_evap, 0.0, shyfttest::EPS);

	act_evap = calculate_step(1.0e8, pot_evap, lp, sca, dt);
	TS_ASSERT_DELTA(act_evap, pot_evap, shyfttest::EPS);
}

TEST_CASE("test_snow") {
	const double soil_moisture = 100.0;
	const double pot_evap = 5.0; // [mm/h]
	const double lp = 150.0;
	const utctime dt = deltahours(1);
	double act_evap_no_snow = calculate_step(soil_moisture, pot_evap, lp, 0.0, dt);
	double act_evap_some_snow = calculate_step(soil_moisture, pot_evap, lp, 0.1, dt);

	TS_ASSERT(act_evap_no_snow > act_evap_some_snow);
}
TEST_CASE("test_evap_from_non_snow_only") {
    const double soil_moisture = 200.0;
    const double pot_evap = 5.0; // [mm/h]
    const double lp = 150.0;
    const utctime dt = deltahours(1);
    double act_evap_no_snow = calculate_step(soil_moisture, pot_evap, lp, 0.0, dt);
    double act_evap_some_snow = calculate_step(soil_moisture, pot_evap, lp, 0.1, dt);

    TS_ASSERT(act_evap_no_snow > act_evap_some_snow);

}

TEST_CASE("test_soil_moisture_threshold") {
	const double sca = 0.0;
	const double pot_evap = 5.0; // [mm/h]
	const double lp = 150.0;
	const utctime dt = deltahours(1);
	double act_evap_less_moisture = calculate_step(50, pot_evap, lp, sca, dt);
	double act_evap_more_moisture = calculate_step(100, pot_evap, lp, sca, dt);

	TS_ASSERT(act_evap_less_moisture < act_evap_more_moisture);

}
}
