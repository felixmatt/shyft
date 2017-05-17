#include "test_pch.h"
#include "core/hbv_tank.h"


using namespace shyft::core;
TEST_SUITE("hbv_tank"){
TEST_CASE("test_regression") {
	hbv_tank::parameter p;
	hbv_tank::calculator<hbv_tank::parameter> calc(p);
	hbv_tank::state s;
	hbv_tank::response r;
	calendar utc;
	utctime t0 = utc.time(2015, 1, 1);

	calc.step(s, r, t0, t0 + deltahours(1), 0);
	TS_ASSERT_DELTA(r.outflow, 6.216, 0.0); //TODO: verify some more numbers
	TS_ASSERT_DELTA(s.uz, 13.2, 0.0);
	TS_ASSERT_DELTA(s.lz, 10.584, 0.0001);
	calc.step(s, r, t0, t0 + deltahours(1), 20.0);
	TS_ASSERT_DELTA(s.uz, 20.8, 0.0002);
	TS_ASSERT_DELTA(r.outflow, 11.82768, 0.00005);
}
}
