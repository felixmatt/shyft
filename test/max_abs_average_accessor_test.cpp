#include <memory>
#include <algorithm>

#include "test_pch.h"
#include "core/time_series.h"
#include "core/utctime_utilities.h"
TEST_SUITE("time_series") {
    TEST_CASE("max_abs_average_accessor") {
        using namespace shyft::time_axis;
        using namespace shyft::time_series;
        using namespace shyft::core;
        using std::vector;
        fixed_dt ta_s{0,10,6};
        fixed_dt ta_a{0,20,3};
        point_ts<decltype(ta_s)> ts{ta_s,vector<double>{ 1,-4,-1,5,3,shyft::nan},POINT_AVERAGE_VALUE};
        max_abs_average_accessor<decltype(ts),decltype(ta_a)> ma{ts,ta_a};
        FAST_CHECK_EQ(ma.value(0), doctest::Approx(2)); // because --4 ==> 4 and over two intervals ->2
        FAST_CHECK_EQ(ma.value(1), doctest::Approx(2.5));// because 5 is higher and two intervals -> 2.5
        FAST_CHECK_EQ(ma.value(2), doctest::Approx(3.0));// average of non-nan area->3
    }
}

