
#include "test_pch.h"
#include "core/hbv_soil.h"


using namespace shyft::core;
TEST_SUITE("hbv_soil") {
TEST_CASE("test_regression") {
	hbv_soil::parameter p;
	hbv_soil::calculator<hbv_soil::parameter> calc(p);
	hbv_soil::state s;
	hbv_soil::response r;
	calendar utc;
	utctime t0 = utc.time(2015, 1, 1);
	auto previous_value = s.sm;
	calc.step(s, r, t0, t0 + deltahours(1), 0, 0);
	TS_ASSERT_DELTA(r.outflow, 0.0, 0.0); //TODO: verify some more numbers
	TS_ASSERT_DELTA(s.sm, previous_value, 0.0);
	calc.step(s, r, t0, t0 + deltahours(1), 50.0, 0);
	TS_ASSERT_DELTA(s.sm, 48.6111, 0.0001);
	TS_ASSERT_DELTA(r.outflow, 1.38888000, 0.05);
}

TEST_CASE("test_dry_soil_case") {
    hbv_soil::parameter p;
    hbv_soil::calculator<hbv_soil::parameter> calc(p);
    hbv_soil::state s;
    hbv_soil::response r;
    calendar utc;
    utctime t0 = utc.time(2015, 1, 1);
    s.sm = 1.0;
    calc.step(s, r, t0, t0 + deltahours(1), 1.0, 20.0);// feed in actual evap
    TS_ASSERT_DELTA(s.sm, 0.0, 0.0);
    TS_ASSERT_DELTA(r.outflow, 4.4444e-5, 1.0e-6);

}
}
