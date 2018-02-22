#include "test_pch.h"
#define _USE_MATH_DEFINES
#include "core/utctime_utilities.h"
#include "core/time_axis.h"
#include "core/time_series.h"
//#include "core/time_series_average.h"
#include "core/time_series_dd.h"

using namespace shyft;
using namespace shyft::core;
using namespace shyft::time_series;
using namespace shyft::time_axis;
using shyft::time_series::dd::apoint_ts;
using shyft::time_series::dd::ice_packing_recession_parameters;
using shyft::time_series::dd::ice_packing_recession_ts;

TEST_SUITE("time_series") {
    TEST_CASE("ice_packing") {
        /// this test should cover all ice-packing detection cases

        fixed_dt ta(0,3600,24);
        point_ts<fixed_dt> temperature{ta,0.0,ts_point_fx::POINT_AVERAGE_VALUE};
        temperature.set(2,-10.0);
        temperature.set(3,-20.0);
        temperature.set(4,-15.0);
        temperature.set(10,shyft::nan);

        //-- verify empty ts
        ice_packing_ts<decltype(temperature)> ice_pack_1;
        FAST_CHECK_EQ(ice_pack_1.ip_param.threshold_temp,doctest::Approx(0.0));
        FAST_CHECK_EQ(ice_pack_1.ip_param.window,0); // is this well defined ?
        FAST_CHECK_EQ(ice_pack_1.ipt_policy,ice_packing_temperature_policy::DISALLOW_MISSING);
        FAST_CHECK_EQ(ice_pack_1.needs_bind(),false);
        FAST_CHECK_EQ(ice_pack_1.size(),0);

        //-- verify simple cases
        ice_packing_parameters ipp{deltahours(2),-10.0};
        ice_packing_ts<decltype(temperature)> ice_pack_2(temperature,ipp);
        FAST_CHECK_EQ(ice_pack_2.size(),temperature.size());
        FAST_CHECK_EQ(isfinite(ice_pack_2.value(0)),false);// 0
        FAST_CHECK_EQ(isfinite(ice_pack_2.value(1)),false);// 0
        ice_pack_2.ipt_policy=ice_packing_temperature_policy::ALLOW_INITIAL_MISSING;
        FAST_CHECK_EQ(ice_pack_2.value(0),doctest::Approx(0.0));// 0
        FAST_CHECK_EQ(ice_pack_2.value(1),doctest::Approx(0.0));//-5
        ice_pack_2.ipt_policy=ice_packing_temperature_policy::ALLOW_ANY_MISSING;
        FAST_CHECK_EQ(ice_pack_2.value(0),doctest::Approx(0.0));// 0
        FAST_CHECK_EQ(ice_pack_2.value(1),doctest::Approx(0.0));//-5
        ice_pack_2.ipt_policy=ice_packing_temperature_policy::ALLOW_INITIAL_MISSING;

        FAST_CHECK_EQ(ice_pack_2.value(2),doctest::Approx(0.0));// 0
        FAST_CHECK_EQ(ice_pack_2.value(3),doctest::Approx(0.0));//-5
        FAST_CHECK_EQ(ice_pack_2.value(4),doctest::Approx(1.0));//-15
        FAST_CHECK_EQ(ice_pack_2.value(5),doctest::Approx(1.0));//-17.5
        FAST_CHECK_EQ(ice_pack_2.value(6),doctest::Approx(0.0));//-7.5

        FAST_CHECK_EQ(ice_pack_2.value(9),doctest::Approx(0));
        FAST_CHECK_EQ(isfinite(ice_pack_2.value(10)),true);
        FAST_CHECK_EQ(ice_pack_2.value(10),doctest::Approx(0.0));
        FAST_CHECK_EQ(isfinite(ice_pack_2.value(11)),false);// nan because #10 is nan, and we don't allow nans
        FAST_CHECK_EQ(isfinite(ice_pack_2.value(12)),false);// nan beause overlap
        FAST_CHECK_EQ(ice_pack_2.value(13),doctest::Approx(0));
        ice_pack_2.ipt_policy =ice_packing_temperature_policy::ALLOW_ANY_MISSING; // now it should allow some nans
        FAST_CHECK_EQ(ice_pack_2.value(11),doctest::Approx(0.0));// because we allow nan
        FAST_CHECK_EQ(ice_pack_2.value(12),doctest::Approx(0.0));
        //FAST_CHECK_EQ(isfinite(ice_pack_2.value(11)),false);// nan because #10 is nan


        FAST_CHECK_EQ(ice_pack_2.value(23),doctest::Approx(0.0));
    }
    TEST_CASE("ice_recession") {
        // default ct
        ice_packing_recession_ts no_fix;
        FAST_CHECK_EQ(no_fix.needs_bind(),false);
        FAST_CHECK_EQ(no_fix.ipr_param.alpha,doctest::Approx(0.0));
        FAST_CHECK_EQ(no_fix.size(),0);


        // using the same setup as test above
        generic_dt ta(0,3600,24);
        apoint_ts temperature{ta,0.0,ts_point_fx::POINT_AVERAGE_VALUE};
        temperature.set(2,-10.0);
        temperature.set(3,-20.0);
        temperature.set(4,-15.0);
        temperature.set(10,shyft::nan);

        //-- verify empty ts
        ice_packing_parameters ipp{deltahours(2),-10.0};
        auto ice_pack=temperature.ice_packing(ipp,ice_packing_temperature_policy::ALLOW_ANY_MISSING);

        apoint_ts flow{ta,10.0,ts_point_fx::POINT_AVERAGE_VALUE};
        for(size_t i=0;i<ta.size();++i) // fill with 10.0, 11,... 34.0
            flow.set(i,10 + i*1.0);
        ice_packing_recession_parameters ipr;
        FAST_CHECK_EQ(ipr.alpha,doctest::Approx(0.0));
        FAST_CHECK_EQ(ipr.recession_minimum,doctest::Approx(0.0));
        ipr.alpha = 1/3600.0; // down to 30% after 1 h
        ipr.recession_minimum = 0.75; // m3/s min flow
        auto ice_fixup= flow.ice_packing_recession(ice_pack,ipr);
        FAST_CHECK_EQ(ice_pack.needs_bind(),false);
        FAST_CHECK_EQ(ice_fixup.needs_bind(),false);
        for(size_t i=0;i<ice_fixup.size();++i) {
            //double ice = ice_pack.value(i);
            double v =ice_fixup.value(i);
            double e = 10.0+i*1.0;

            if(i==4) {
                e = ipr.recession_minimum + ( (10.0 + (i-1)*1.0) - ipr.recession_minimum)*exp(-ipr.alpha*(ice_fixup.time(i)-ice_fixup.time(i-1)));
            } else if (i==5) {
                e = ipr.recession_minimum + ( (10.0 + (i-2)*1.0) - ipr.recession_minimum)*exp(-ipr.alpha*(ice_fixup.time(i)-ice_fixup.time(i-2)));
            }

            FAST_CHECK_EQ(v,doctest::Approx(e));

        }

    }

    TEST_CASE("ice_speed") {
        utctime t0=0;
        calendar utc;
        generic_dt ta(t0,3600,24*365*1);// approx 1y hour ts
        apoint_ts temperature{ta,0.0,ts_point_fx::POINT_AVERAGE_VALUE};
        for(size_t i=0;i<ta.size();++i) { // create something that should give ice
            auto w=utc.calendar_week_units(ta.time(i)).iso_week;
            if(w==4) {
                temperature.set(i,-15.0);//ice
            } else if(w==6) {
                temperature.set(i,-16.0); // another ice
            }
        }
        apoint_ts flow{ta,10.0,ts_point_fx::POINT_AVERAGE_VALUE};

        auto ice_fixup= flow.ice_packing_recession(
            temperature.ice_packing(
                ice_packing_parameters{deltahours(24*5),-10.0},// 5days freezing cold -10 ->ice
                ice_packing_temperature_policy::ALLOW_ANY_MISSING
            ),
            ice_packing_recession_parameters{1/(3600.0*24*7),0.75}
        );
        auto t_start=timing::now();
        auto v= ice_fixup.values();//force evaluation of all values
        auto us_used=  elapsed_us(t_start,timing::now());
        size_t nan_count=0;
        size_t recession_count=0;
        for(auto x:v) {
            if(!isfinite(x)) {++nan_count;continue;}
            if(x <10.0)
                ++recession_count;
        }
        FAST_CHECK_EQ(nan_count,0);
        FAST_CHECK_GE(recession_count,200);//empirical
        if(getenv("SHYFT_VERBOSE")){
            cout<<"ice_pack_recession: time-used for 10 year hour fixups :"<<us_used/1e3<<" [ms]\n"
            <<"\tnan_count:"<<nan_count<<"\n"
            <<"\trecession_count:"<<recession_count<<"\n";
        }
        FAST_WARN_LE(us_used,200000);// 0.2second.. a lot of time.
    }
}
