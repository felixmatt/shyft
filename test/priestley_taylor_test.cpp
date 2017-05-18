#include "test_pch.h"
#include "core/priestley_taylor.h"

using namespace shyft::core;
namespace pt = shyft::core::priestley_taylor;
TEST_SUITE("priestley_taylor") {
TEST_CASE("priestley_taylor_test::test_regression") {
    pt::calculator pt(0.2, 1.26);
    TS_ASSERT_DELTA(pt.potential_evapotranspiration(20.5, 445, 64/100.0)*24.0*3600 , 11.0 , 1.0); //TODO: verify some more numbers
    TS_ASSERT_DELTA(pt.potential_evapotranspiration(-20, 200, 30/100.0)*24*3600,0.0, .5);// at very low temperature, expect 0.0
    TS_ASSERT_DELTA(pt.potential_evapotranspiration(0, 200, 30/100.0)*24*3600, 1.0, .5);// at low temperature, expect close to 0.0
    // just verify that pet increase with temp.
    for(double t=0.0; t < 30.0; t += 0.5)
        TS_ASSERT(pt.potential_evapotranspiration(t, 400, 50/100.0) < pt.potential_evapotranspiration(t + 0.5, 400, 50/100.0));
    // and increase with rh
    for(double rh=0.01; rh < 1.0; rh += 0.01)
        TS_ASSERT(pt.potential_evapotranspiration(15, 400, rh) < pt.potential_evapotranspiration(15, 400, rh + 1.0));

    // and increase with radiation
    for(double r=10.0; r < 900.0; r += 50.0)
        TS_ASSERT(pt.potential_evapotranspiration(15, r, 60)<pt.potential_evapotranspiration(15, r + 50.0, 60));
}
}
