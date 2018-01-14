#include "test_pch.h"
#define _USE_MATH_DEFINES
#include "core/time_axis.h"
#include "core/time_series.h"
#include "core/time_series_average.h"

namespace shyft {namespace time_series {

//-- for test; run the original shyft-accumulate
template < class TA,class TS>
vector<double> old_accumulate_linear(const TA&ta, const TS& ts, bool avg) {
    vector<double> r; r.reserve(ta.size());
    size_t ix_hint=0;
    utctimespan t_sum{0};
    if(avg) {
        for(size_t i=0;i<ta.size();++i) {
            double v = accumulate_value(ts,ta.period(i),ix_hint,t_sum,true,true);
            r.emplace_back(t_sum?v/double(t_sum):shyft::nan);
        }
    } else {
        for(size_t i=0;i<ta.size();++i) {
            r.emplace_back( accumulate_value(ts,ta.period(i),ix_hint,t_sum,true,true));
        }
    }
    return r;
}

template < class TA,class TS>
vector<double> old_accumulate_stair_case(const TA&ta, const TS& ts, bool avg) {
    vector<double> r; r.reserve(ta.size());
    size_t ix_hint=0;
    utctimespan t_sum{0};
	utcperiod tp = ts.total_period();
    if(avg) {
        for(size_t i=0;i<ta.size();++i) {
			auto p = ta.period(i);
			if (p.end > tp.end) p.end = tp.end;
			if (p.start < tp.end) {
				double v = accumulate_value(ts, p, ix_hint, t_sum, false, false);
				r.emplace_back(t_sum ? v / double(t_sum) : shyft::nan);
			}  else
				r.emplace_back(shyft::nan);
            ;
        }
    } else {
        for(size_t i=0;i<ta.size();++i) {
			auto p = ta.period(i);
			if (p.end > tp.end) p.end = tp.end;
			if (p.start < tp.end)
				r.emplace_back(accumulate_value(ts, ta.period(i), ix_hint, t_sum, false, false));
			else
				r.emplace_back(shyft::nan);
        }
    }
    return r;
}


}
}

using namespace shyft::core;
using namespace shyft::time_series;
using namespace shyft::time_axis;
using namespace std;
#define TEST_SECTION(x)
template<class F>
void test_linear_fx(F&& acc_fn) {
    utctimespan dt{deltahours(1)};
    fixed_dt ta{0,dt,6};
    ts_point_fx linear{POINT_INSTANT_VALUE};
    point_ts<decltype(ta)> ts{ta,vector<double>{1.0,2.0,shyft::nan,4.0,3.0,6.0},linear};
	TEST_SECTION("own_axis_average") {
        auto r = acc_fn(ta,ts,true);
        FAST_REQUIRE_EQ(r.size(),ta.size());
        CHECK(r[0]==doctest::Approx(1.5));
        FAST_CHECK_UNARY(!isfinite(r[1]));
        FAST_CHECK_UNARY(!isfinite(r[2]));
        CHECK(r[3]==doctest::Approx(3.5));
        CHECK(r[4]==doctest::Approx(4.5));
        FAST_CHECK_UNARY(!isfinite(r[5]));
    }
	TEST_SECTION("own_axis_integral") {
        auto r = acc_fn(ta,ts,false);
        FAST_REQUIRE_EQ(r.size(),ta.size());
        CHECK(r[0]==doctest::Approx(1.5*dt));
        FAST_CHECK_UNARY(!isfinite(r[1]));
        FAST_CHECK_UNARY(!isfinite(r[2]));
        CHECK(r[3]==doctest::Approx(3.5*dt));
        CHECK(r[4]==doctest::Approx(4.5*dt));
        FAST_CHECK_UNARY(!isfinite(r[5]));
    }
	TEST_SECTION("zero_axis") {
        fixed_dt zta{0,dt,0};
        auto r=acc_fn(zta,ts,true);
        FAST_CHECK_EQ(r.size(),0);
    }
	TEST_SECTION("zero_ts") {
        point_ts<decltype(ta)> zts(fixed_dt{0,dt,0},1.0,linear);
        auto r=acc_fn(ta,zts,true);
        FAST_CHECK_EQ(r.size(),ta.size());
        for(const auto&v:r) FAST_CHECK_UNARY(!isfinite(v));
    }
	TEST_SECTION("axis_before") {
        fixed_dt bta{-10000,dt,1};
        auto r=acc_fn(bta,ts,true);
        FAST_CHECK_EQ(r.size(),bta.size());
        for(const auto&v:r) FAST_CHECK_UNARY(!isfinite(v));
    }
    SUBCASE("axis_after") {
        fixed_dt ata{ta.total_period().end,dt,10};
        auto r=acc_fn(ata,ts,true);
        FAST_CHECK_EQ(r.size(),ata.size());
        for(const auto&v:r) FAST_CHECK_UNARY(!isfinite(v));
    }
	TEST_SECTION("aligned_x2_axis") {
        fixed_dt ta2(0,2*dt,ta.size()/2);
        auto r = acc_fn(ta2,ts,true);
        FAST_REQUIRE_EQ(r.size(),ta2.size());
        CHECK(r[0]==doctest::Approx(1.5));
        CHECK(r[1]==doctest::Approx(3.5));
        CHECK(r[2]==doctest::Approx(4.5));
    }
	TEST_SECTION("aligned_/2_axis") {
        fixed_dt ta2(0,dt/2,ta.size()*2-2);
        auto r = acc_fn(ta2,ts,true);
        FAST_REQUIRE_EQ(r.size(),ta2.size());
        CHECK(r[0]==doctest::Approx(1.25));
        CHECK(r[1]==doctest::Approx(1.75));
        FAST_CHECK_UNARY(!isfinite(r[2]));
        FAST_CHECK_UNARY(!isfinite(r[3]));
        FAST_CHECK_UNARY(!isfinite(r[4]));
        FAST_CHECK_UNARY(!isfinite(r[5]));
        CHECK(r[6]==doctest::Approx(3.75));
        CHECK(r[7]==doctest::Approx(3.25));
        CHECK(r[8]==doctest::Approx(0.5*(3.0+4.5)));
        CHECK(r[9]==doctest::Approx(0.5*(4.5+6.0)));
    }
	TEST_SECTION("aligned_one_interval") {
        fixed_dt ta2 {0,ta.total_period().timespan(),1};
        auto r = acc_fn(ta2,ts,true);
        FAST_REQUIRE_EQ(r.size(),ta2.size());
        CHECK(r[0]==doctest::Approx((1.5+3.5+4.5)/3.0));
    }
	TEST_SECTION("un_aligned_/2_axis_begins_in_interval") {
        fixed_dt ta2(+dt/4,dt/2,ta.size()*2);
        auto r = acc_fn(ta2,ts,true);
        FAST_REQUIRE_EQ(r.size(),ta2.size());
        CHECK(r[0]==doctest::Approx(1.500));
        CHECK(r[1]==doctest::Approx(1.875));
        FAST_CHECK_UNARY(!isfinite(r[2]));
        FAST_CHECK_UNARY(!isfinite(r[3]));
        FAST_CHECK_UNARY(!isfinite(r[4]));
        CHECK(r[5]==doctest::Approx(3.875));
        CHECK(r[6]==doctest::Approx(3.500));
        CHECK(r[7]==doctest::Approx(3.250));
        CHECK(r[8]==doctest::Approx(4.500));
        CHECK(r[9]==doctest::Approx(5.625));
        FAST_CHECK_UNARY(!isfinite(r[10]));
        FAST_CHECK_UNARY(!isfinite(r[11]));
    }
	TEST_SECTION("un_aligned_/2_axis_begins_before_interval") {
        fixed_dt ta2(-dt/4,dt/2,ta.size()*2-2);
        auto r = acc_fn(ta2,ts,true);
        FAST_REQUIRE_EQ(r.size(),ta2.size());
        CHECK(r[0]==doctest::Approx(1.125));
        CHECK(r[1]==doctest::Approx(1.500));
        CHECK(r[2]==doctest::Approx(1.875));
        FAST_CHECK_UNARY(!isfinite(r[3]));
        FAST_CHECK_UNARY(!isfinite(r[4]));
        FAST_CHECK_UNARY(!isfinite(r[5]));
        CHECK(r[6]==doctest::Approx(3.875));
        CHECK(r[7]==doctest::Approx(3.500));
        CHECK(r[8]==doctest::Approx(3.250));
        CHECK(r[9]==doctest::Approx(4.500));
    }
	TEST_SECTION("out_of_points_searching_for_start") {
        point_ts<fixed_dt> nts(fixed_dt{0,dt,5},vector<double>{shyft::nan,1.0,1.0,shyft::nan,shyft::nan},linear);
        fixed_dt ta2{0,dt*4,2};
        auto r = acc_fn(ta2,nts,true);
        FAST_REQUIRE_EQ(r.size(),ta2.size());
        CHECK(r[0]==doctest::Approx(1.0));
        FAST_CHECK_UNARY(!isfinite(r[1]));
    }
	TEST_SECTION("just_nans") {
        point_ts<fixed_dt> nts(fixed_dt{0,dt,5},vector<double>{shyft::nan,shyft::nan,shyft::nan,shyft::nan,shyft::nan},linear);
        fixed_dt ta2{0,dt*4,2};
        auto r = acc_fn(ta2,nts,true);
        FAST_REQUIRE_EQ(r.size(),ta2.size());
        FAST_CHECK_UNARY(!isfinite(r[0]));
        FAST_CHECK_UNARY(!isfinite(r[1]));
    }
	TEST_SECTION("just_one_value_in_source") {
        point_ts<fixed_dt> nts(fixed_dt{0,dt,1},vector<double>{1.0},linear);
        fixed_dt ta2{0,dt*4,2};
        auto r = acc_fn(ta2,nts,true);
        FAST_REQUIRE_EQ(r.size(),ta2.size());
        FAST_CHECK_UNARY(!isfinite(r[0]));
        FAST_CHECK_UNARY(!isfinite(r[1]));
    }

}



template<class F>
void test_stair_case_fx(F&& acc_fn) {
    utctimespan dt{deltahours(1)};
    fixed_dt ta{0,dt,6};
    ts_point_fx stair_case{POINT_AVERAGE_VALUE};
    point_ts<decltype(ta)> ts{ta,vector<double>{1.0,2.0,shyft::nan,4.0,3.0,6.0},stair_case};
	TEST_SECTION("own_axis_average") {
        auto r = acc_fn(ta,ts,true);
        FAST_REQUIRE_EQ(r.size(),ta.size());
        CHECK(r[0]==doctest::Approx(1.0));
        CHECK(r[1]==doctest::Approx(2.0));
        FAST_CHECK_UNARY(!isfinite(r[2]));
        CHECK(r[3]==doctest::Approx(4.0));
        CHECK(r[4]==doctest::Approx(3.0));
        CHECK(r[5]==doctest::Approx(6.0));
    }
	TEST_SECTION("own_axis_integral") {
        auto r = acc_fn(ta,ts,false);
        FAST_REQUIRE_EQ(r.size(),ta.size());
        CHECK(r[0]==doctest::Approx(1.0*dt));
        CHECK(r[1]==doctest::Approx(2.0*dt));
        FAST_CHECK_UNARY(!isfinite(r[2]));
        CHECK(r[3]==doctest::Approx(4.0*dt));
        CHECK(r[4]==doctest::Approx(3.0*dt));
        CHECK(r[5]==doctest::Approx(6.0*dt));
    }
	TEST_SECTION("zero_axis") {
        fixed_dt zta{0,dt,0};
        auto r= acc_fn(zta,ts,true);
        FAST_CHECK_EQ(r.size(),0);
    }
	TEST_SECTION("zero_ts") {
        point_ts<decltype(ta)> zts(fixed_dt{0,dt,0},1.0,stair_case);
        auto r= acc_fn(ta,zts,true);
        FAST_CHECK_EQ(r.size(),ta.size());
        for(const auto&v:r) FAST_CHECK_UNARY(!isfinite(v));
    }
	TEST_SECTION("one_point_ts") {
        point_ts<decltype(ta)> zts(fixed_dt{0,dt,1},1.0,stair_case);
        auto r= acc_fn(ta,zts,true);
        FAST_CHECK_EQ(r.size(),ta.size());
        CHECK(r[0]==doctest::Approx(1.0));
        for(size_t i=1;i<r.size();++i) FAST_CHECK_UNARY(!isfinite(r[i]));
    }
	TEST_SECTION("last_interval_handling") {
        point_ts<decltype(ta)> ots(fixed_dt{0,dt*(utctimespan)ta.size(),1},1.0,stair_case);
        auto r=acc_fn(ta,ots,false);
        FAST_CHECK_EQ(r.size(),ta.size());
        for(size_t i=0;i<ta.size();++i) {
            FAST_CHECK_EQ(r[i],doctest::Approx(dt));
        }
        
    }
	TEST_SECTION("axis_before") {
        fixed_dt bta{-10000,dt,1};
        auto r= acc_fn(bta,ts,true);
        FAST_CHECK_EQ(r.size(),bta.size());
        for(const auto&v:r) FAST_CHECK_UNARY(!isfinite(v));
    }
	TEST_SECTION("axis_after") {
        fixed_dt ata{ta.total_period().end,dt,10};
        auto r= acc_fn(ata,ts,true);
        FAST_CHECK_EQ(r.size(),ata.size());
        for(const auto&v:r) FAST_CHECK_UNARY(!isfinite(v));
    }
	TEST_SECTION("aligned_x2_axis") {
        fixed_dt ta2(0,2*dt,ta.size()/2);
        auto r = acc_fn(ta2,ts,true);
        FAST_REQUIRE_EQ(r.size(),ta2.size());
        CHECK(r[0]==doctest::Approx(1.5));
        CHECK(r[1]==doctest::Approx(4.0));
        CHECK(r[2]==doctest::Approx(4.5));
    }
	TEST_SECTION("aligned_/2_axis") {
        fixed_dt ta2(0,dt/2,ta.size()*2-2);
        auto r = acc_fn(ta2,ts,true);
        FAST_REQUIRE_EQ(r.size(),ta2.size());
        CHECK(r[0]==doctest::Approx(1.00));
        CHECK(r[1]==doctest::Approx(1.00));
        CHECK(r[2]==doctest::Approx(2.00));
        CHECK(r[3]==doctest::Approx(2.00));
        FAST_CHECK_UNARY(!isfinite(r[4]));
        FAST_CHECK_UNARY(!isfinite(r[5]));
        CHECK(r[6]==doctest::Approx(4.00));
        CHECK(r[7]==doctest::Approx(4.00));
        CHECK(r[8]==doctest::Approx(3.00));
        CHECK(r[9]==doctest::Approx(3.00));
    }
	TEST_SECTION("aligned_one_interval") {
        fixed_dt ta2 {0,ta.total_period().timespan(),1};
        auto r = acc_fn(ta2,ts,true);
        FAST_REQUIRE_EQ(r.size(),ta2.size());
        CHECK(r[0]==doctest::Approx((1+2+4+3+6)/5.0));
    }
	TEST_SECTION("un_aligned_/2_axis_begins_in_interval") {
        fixed_dt ta2(+dt/4,dt/2,ta.size()*2);
        auto r = acc_fn(ta2,ts,true);
        FAST_REQUIRE_EQ(r.size(),ta2.size());
        CHECK(r[0]==doctest::Approx(1.000));
        CHECK(r[1]==doctest::Approx(1.500));
        CHECK(r[2]==doctest::Approx(2.000));
        CHECK(r[3]==doctest::Approx(2.000));
        FAST_CHECK_UNARY(!isfinite(r[4]));
        CHECK(r[5]==doctest::Approx(4.000));
        CHECK(r[6]==doctest::Approx(4.000));
        CHECK(r[7]==doctest::Approx(3.500));
        CHECK(r[8]==doctest::Approx(3.000));
        CHECK(r[9]==doctest::Approx(4.500));
		CHECK(r[10]==doctest::Approx(6.000));
		CHECK(r[11]==doctest::Approx(6.000));
		//FAST_CHECK_UNARY(!isfinite(r[10]));
        //FAST_CHECK_UNARY(!isfinite(r[11]));
    }
	TEST_SECTION("un_aligned_/2_axis_begins_before_interval") {
        fixed_dt ta2(-dt/4,dt/2,ta.size()*2-2);
        auto r = acc_fn(ta2,ts,true);
        FAST_REQUIRE_EQ(r.size(),ta2.size());
        CHECK(r[0]==doctest::Approx(1.000));
        CHECK(r[1]==doctest::Approx(1.000));
        CHECK(r[2]==doctest::Approx(1.500));
        CHECK(r[3]==doctest::Approx(2.000));
        CHECK(r[4]==doctest::Approx(2.000));
        FAST_CHECK_UNARY(!isfinite(r[5]));
        CHECK(r[6]==doctest::Approx(4.000));
        CHECK(r[7]==doctest::Approx(4.000));
        CHECK(r[8]==doctest::Approx(3.500));
        CHECK(r[9]==doctest::Approx(3.000));
    }
	TEST_SECTION("out_of_points_searching_for_start") {
        point_ts<fixed_dt> nts(fixed_dt{0,dt,5},vector<double>{shyft::nan,1.0,1.0,shyft::nan,shyft::nan},stair_case);
        fixed_dt ta2{0,dt*4,2};
        auto r = acc_fn(ta2,nts,true);
        FAST_REQUIRE_EQ(r.size(),ta2.size());
        CHECK(r[0]==doctest::Approx(1.0));
        FAST_CHECK_UNARY(!isfinite(r[1]));
    }
	TEST_SECTION("just_nans") {
        point_ts<fixed_dt> nts(fixed_dt{0,dt,5},vector<double>{shyft::nan,shyft::nan,shyft::nan,shyft::nan,shyft::nan},stair_case);
        fixed_dt ta2{0,dt*4,2};
        auto r = acc_fn(ta2,nts,true);
        FAST_REQUIRE_EQ(r.size(),ta2.size());
        FAST_CHECK_UNARY(!isfinite(r[0]));
        FAST_CHECK_UNARY(!isfinite(r[1]));
    }
	TEST_SECTION("just_one_value_in_source") {
        point_ts<fixed_dt> nts(fixed_dt{0,dt,1},vector<double>{1.0},stair_case);
        fixed_dt ta2{0,dt*4,2};
        auto r = acc_fn(ta2,nts,true);
        FAST_REQUIRE_EQ(r.size(),ta2.size());
        CHECK(r[0]==doctest::Approx(1.0));
        FAST_CHECK_UNARY(!isfinite(r[1]));
    }
}

TEST_SUITE("time_series") {
    TEST_CASE("ts_avg_linear") {
        auto f_lin=accumulate_linear<fixed_dt,point_ts<fixed_dt>>;
        test_linear_fx(f_lin);
    }
    TEST_CASE("old_ts_avg_linear") {
        auto f_lin_old=old_accumulate_linear<fixed_dt,point_ts<fixed_dt>>;
        test_linear_fx(f_lin_old);
    }
    TEST_CASE("ts_avg_stair_case") {
        auto f_stair_case=accumulate_stair_case<fixed_dt,point_ts<fixed_dt>>;
        test_stair_case_fx(f_stair_case);
    }
    TEST_CASE("old_ts_avg_stair_case") {
        auto f_stair_case_old=old_accumulate_stair_case<fixed_dt,point_ts<fixed_dt>>;
        test_stair_case_fx(f_stair_case_old);
    }
    TEST_CASE("ts_avg_speed_test") {
        using ts_t = point_ts<fixed_dt>;
        size_t n = 5 * 365 * 8;// 0.1 MB pr ts.
        size_t n_ts = 1000;//~100 MB for each 1000 ts.
        utctimespan dt_h{ deltahours(1) };
        utctimespan dt_h24{ 24 * dt_h };
        fixed_dt ta_h{ 0,dt_h,n };
        ts_point_fx linear{ POINT_INSTANT_VALUE };
        vector<ts_t> tsv;
        for (size_t i = 0; i<n_ts; ++i) tsv.emplace_back(ta_h, double(i), linear);

        // time ts->vector
        fixed_dt ta_h24(0, dt_h24, n / 24);
        double s = 0.0;
        auto t0 = timing::now();
        for (size_t i = 0; i < n_ts; ++i) {
            auto r = accumulate_linear(ta_h24, tsv[i], true);
            s += r[0];
        }
        auto t1 = timing::now();
        for (size_t i = 0; i < n_ts; ++i) {
            auto r = accumulate_stair_case(ta_h24, tsv[i], true);
            s += r[0];
        }
        auto t2 = timing::now();
        for (size_t i = 0; i < n_ts; ++i) {
            average_accessor<ts_t,fixed_dt> avg(tsv[i], ta_h24, extension_policy::USE_NAN);
            vector<double> r;r.reserve(ta_h24.size());
            for (size_t t = 0; t<ta_h24.size();++t) {
                r.push_back(avg.value(t));
            }
            s += r[0];
        }
        auto t3 = timing::now();
        auto us_l = elapsed_us(t0, t1);
        auto us_s = elapsed_us(t1, t2);
        auto us_o = elapsed_us(t2, t3);
        FAST_CHECK_GE(s, 0.0);
        double mpts_s = n_ts*n / 1e6;//mill pts in source dim
        //double mmpt_r = n_ts*n / 24 / 1e6;// mill points result dim
        std::cout << "Linear mill pts/ sec " << mpts_s / (us_l / 1e6) << "\n";
        std::cout << "Stairc mill pts/ sec " << mpts_s / (us_s / 1e6) << "\n";
        std::cout << "Old    mill pts/ sec " << mpts_s / (us_o / 1e6) << "\n";
    }
    TEST_CASE("ts_core_ts_last_interval_study_case") {
        using ts_t = point_ts<fixed_dt>;
        calendar utc{};
        fixed_dt ta_s{utc.time(2017,10,16),deltahours(24*7),219};
        fixed_dt ta{utc.time(2017,10,16),deltahours(3),12264};
        ts_t src(ta_s,1.0,POINT_AVERAGE_VALUE);
        
        average_ts<ts_t,fixed_dt> i_src{src,ta};
        for(size_t i=0;i<ta.size();++i)
            CHECK( i_src.value(i) == doctest::Approx(1.0));

    }
}
