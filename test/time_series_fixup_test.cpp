#include "test_pch.h"

#include <cmath>
#include <vector>

#include "core/utctime_utilities.h"
#include "core/time_axis.h"
#include "api/time_series.h"



TEST_SUITE("time_series") {

    using shyft::core::no_utctime;
    using std::numeric_limits;
    const double eps = numeric_limits<double>::epsilon();
    using shyft::api::apoint_ts;
    using shyft::time_axis::generic_dt;
    using shyft::time_series::ts_point_fx;
    using std::vector;
    using std::make_shared;
    using std::isfinite;

    using shyft::api::qac_parameter;
    using shyft::api::qac_ts;

    TEST_CASE("qac_parameter") {

        qac_parameter q;

        SUBCASE("no limits set, allow all values, except nan") {
            FAST_CHECK_EQ(q.is_ok_quality(shyft::nan),false);
            FAST_CHECK_EQ(q.is_ok_quality(1.0),true);
        }

        SUBCASE("min/max abs limits") {
            q.max_x=1.0;
            FAST_CHECK_EQ(q.is_ok_quality(1.0),true);
            FAST_CHECK_EQ(q.is_ok_quality(1.0+eps),false);
            q.min_x=-1.0;
            FAST_CHECK_EQ(q.is_ok_quality(-1.0),true);
            FAST_CHECK_EQ(q.is_ok_quality(-1.0-eps),false);
        }
    }

    TEST_CASE("qac_ts") {
        generic_dt ta{0,10,5};
        //                 0    1       2     3    4
        vector<double> v {0.0,1.0,shyft::nan,3.0,-20.1};
        vector<double>ev {0.0,1.0,       2.0,3.0,-20.1};
        vector<double>cv {1.0,0.0,      -1.0,3.0,-20.1};
        apoint_ts src(ta,v,ts_point_fx::POINT_AVERAGE_VALUE);
        apoint_ts cts{ta,cv,ts_point_fx::POINT_AVERAGE_VALUE};
        qac_parameter qp;
        auto ts = make_shared<qac_ts>(src,qp);


        // verify simple min-max limit cases
        FAST_CHECK_UNARY(ts.get()!=nullptr);
        FAST_CHECK_EQ(ts->value(2),doctest::Approx(2.0));
        FAST_CHECK_EQ(ts->value_at(ts->time(2)),doctest::Approx(2.0));
        FAST_CHECK_EQ(ts->value_at(ts->time(2)+1),doctest::Approx(2.0));
        src.set_point_interpretation(ts_point_fx::POINT_INSTANT_VALUE);
        FAST_CHECK_EQ(ts->value_at(ts->time(2)+1),doctest::Approx(2.1));
        FAST_CHECK_EQ(ts->value_at(ts->time(2)-1),doctest::Approx(1.9));
        ts->p.min_x = 0.0;
        FAST_CHECK_UNARY(!isfinite(ts->value_at(ts->time(3)+1)));
        ts->p.min_x = -40.0;
        FAST_CHECK_UNARY(isfinite(ts->value_at(ts->time(3)+1)));
        ts->p.max_x = 2.9; // clip 3.0 out of range
        FAST_CHECK_EQ(ts->value(2),
                      doctest::Approx(
                         v[1] +10*(v[4]-v[1])/(40-10)
                      )
        );

        FAST_CHECK_EQ(ts->value(3),doctest::Approx(
                         v[1] +20*(v[4]-v[1])/(40-10)
                      )
        );

        ts->p.max_x = shyft::nan;
        auto qv=ts->values();
        for(size_t i=0;i<ts->size();++i)
            FAST_CHECK_EQ(qv[i], doctest::Approx(ev[i]));
        src.set(0,shyft::nan);
        src.set(1,shyft::nan);
        FAST_CHECK_UNARY(!isfinite(ts->value(0)));
        FAST_CHECK_UNARY(!isfinite(ts->value(1)));
        FAST_CHECK_UNARY(!isfinite(ts->value(2)));
        FAST_CHECK_UNARY(isfinite(ts->value(3)));
        FAST_CHECK_UNARY(isfinite(ts->value(4)));

        ts->cts=cts.ts;// now set in replacement values as time-series
        FAST_CHECK_EQ(ts->value(0),doctest::Approx(cts.value(0)));
        FAST_CHECK_EQ(ts->value(1),doctest::Approx(cts.value(1)));
        FAST_CHECK_EQ(ts->value(2),doctest::Approx(cts.value(2)));
        FAST_CHECK_EQ(ts->value(3),doctest::Approx(v[3]));// own value
        FAST_CHECK_EQ(ts->value(4),doctest::Approx(v[4]));// own value
        ts->p.min_x = 0.0;// clip out neg values
        FAST_CHECK_EQ(ts->value(4),doctest::Approx(cts.value(4)));// own value replaces -20.1
    }

}
