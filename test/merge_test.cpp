#include "test_pch.h"
#define _USE_MATH_DEFINES
#include "core/time_series.h"
#include "core/time_axis.h"
#include "api/api.h"
#include "api/time_series.h"
#include "core/time_series_statistics.h"
#include "core/time_series_merge.h"


TEST_SUITE("ts_merge") {
    TEST_CASE("vector_merge") {
        using shyft::time_axis::merge_info;
        using shyft::time_axis::merge;
        using std::vector;
        FAST_CHECK_EQ(vector<int>{1,1,2,2}, merge(vector<int>{1,1}, vector<int>{2,2}, merge_info{0,0,2}));
        FAST_CHECK_EQ(vector<int>{1, 1}, merge(vector<int>{1, 1}, vector<int>{2, 2}, merge_info{ 0,0,0 }));
        FAST_CHECK_EQ(vector<int>{2,2,1, 1}, merge(vector<int>{1, 1}, vector<int>{2, 2}, merge_info{ 2,0,0 }));
        FAST_CHECK_EQ(vector<int>{2, 1, 1,2}, merge(vector<int>{1, 1}, vector<int>{2, 2}, merge_info{ 1,1,1 }));
        FAST_CHECK_EQ(vector<int>{2, 1, 1}, merge(vector<int>{1, 1}, vector<int>{2, 2}, merge_info{ 1,0,0 }));
        FAST_CHECK_EQ(vector<int>{1, 1, 2}, merge(vector<int>{1, 1}, vector<int>{2, 2}, merge_info{ 0,1,1 }));
    }
    TEST_CASE("point_merge") {
        using shyft::time_series::point_ts;
        using shyft::core::utcperiod;
        using shyft::core::utctime;
        using shyft::core::calendar;
        using shyft::core::deltahours;
        using shyft::time_series::merge;
        const auto stair_case = shyft::time_series::POINT_AVERAGE_VALUE;
        calendar c;
        auto t1 = c.time(2017, 1, 1,0);
        auto t2 = c.time(2017, 1, 1, 3);
        auto dt = deltahours(1);
        size_t n = 3;
        SUBCASE("fixed_dt") {
            using shyft::time_axis::fixed_dt;
            fixed_dt ta1{ t1,dt,n };
            fixed_dt ta2{ t2,dt,n };
            fixed_dt ta12{ t1,dt,2 * n };
            point_ts<fixed_dt> a(ta1, 1.0, stair_case);
            point_ts<fixed_dt> b(ta2, 2.0, stair_case);
            point_ts<fixed_dt> ab(ta12, { 1.0,1.0,1.0,2.0,2.0,2.0 }, stair_case);
            auto m_ab = merge(a, b);
            auto m_ba = merge(b, a);// in this case, gives same result
            FAST_CHECK_EQ(m_ab.v, ab.v);
            FAST_CHECK_EQ(m_ba.v, ab.v);
        }
        SUBCASE("calendar_dt") {
            using shyft::time_axis::calendar_dt;
            using std::make_shared;
            auto u = make_shared<calendar>(deltahours(1));
            calendar_dt ta1{u, t1,dt,n };
            calendar_dt ta2{u, t2,dt,n };
            calendar_dt ta12{u, t1,dt,2 * n };
            point_ts<calendar_dt> a(ta1, 1.0, stair_case);
            point_ts<calendar_dt> b(ta2, 2.0, stair_case);
            point_ts<calendar_dt> ab(ta12, { 1.0,1.0,1.0,2.0,2.0,2.0 }, stair_case);
            auto m_ab = merge(a, b);
            auto m_ba = merge(b, a);// in this case, gives same result
            FAST_CHECK_EQ(m_ab.v, ab.v);
            FAST_CHECK_EQ(m_ba.v, ab.v);
        }
        SUBCASE("point_dt") {
            using shyft::time_axis::point_dt;
            point_dt ta1{ {t1 + 0*dt,t1 + 1*dt,t1 + 2*dt}, t1+3*dt };
            point_dt ta2{ { t2 + 0*dt,t2 + 1*dt,t2 + 2*dt }, t2 + 3*dt };
            point_dt ta12{ { t1 + 0*dt,t1 + 1*dt,t1 + 2*dt, t1 + 3*dt, t1 + 4*dt, t1+5*dt }, t1 + 6*dt };
            point_ts<point_dt> a(ta1, 1.0, stair_case);
            point_ts<point_dt> b(ta2, 2.0, stair_case);
            point_ts<point_dt> ab(ta12, { 1.0,1.0,1.0,2.0,2.0,2.0 }, stair_case);
            auto m_ab = merge(a, b);
            auto m_ba = merge(b, a);// in this case, gives same result
            FAST_CHECK_EQ(m_ab.v, ab.v);
            FAST_CHECK_EQ(m_ba.v, ab.v);
            FAST_CHECK_EQ(m_ba.ta.t, ab.ta.t);
        }
        SUBCASE("generic_dt") {
            using shyft::time_axis::fixed_dt;
            using shyft::time_axis::generic_dt;
            generic_dt ta1{ fixed_dt{ t1,dt,n } };
            generic_dt ta2{ fixed_dt { t2,dt,n } };
            generic_dt ta12{ fixed_dt { t1,dt,2*n } };
            point_ts<generic_dt> a(ta1, 1.0, stair_case);
            point_ts<generic_dt> b(ta2, 2.0, stair_case);
            point_ts<generic_dt> ab(ta12, { 1.0,1.0,1.0,2.0,2.0,2.0 }, stair_case);
            auto m_ab = merge(a, b);
            auto m_ba = merge(b, a);// in this case, gives same result
            FAST_CHECK_EQ(m_ab.v, ab.v);
            FAST_CHECK_EQ(m_ba.v, ab.v);
        }
    }
    TEST_CASE("forecast_merge") {
		using namespace shyft;
		using namespace std;
		using ta_t = time_axis::fixed_dt;
		using ts_t = time_series::point_ts<ta_t>;
		using tsv_t = std::vector<ts_t>;
        // standard triple A testing

        // #1:Arrange
        size_t n_fc=100;
        size_t fc_steps =66;

        core::calendar utc;
        auto t0 = utc.time(2017,1,1,0,0,0);
        auto dt=deltahours(1);
        size_t n_dt_fc =6;
        auto dt_fc= deltahours(n_dt_fc); // typical arome
        tsv_t fc;
        auto point_fx= time_series::ts_point_fx::POINT_AVERAGE_VALUE;
        for(size_t i=0;i<n_fc;++i) {
            vector<double> v;for(size_t t=0;t<fc_steps;++t) v.push_back(double(i)+ t/double(fc_steps));
            fc.emplace_back(ta_t(t0+i*dt_fc,dt,fc_steps),v,point_fx);
        }
        // #2: Act
        for( size_t lead_time_hours=0;lead_time_hours<12;++lead_time_hours) {
            auto m = time_series::forecast_merge<time_series::point_ts<time_axis::point_dt>>(fc,deltahours(lead_time_hours),dt_fc);
            // #3: Assert
            FAST_REQUIRE_EQ(m.size(),n_fc*n_dt_fc);
            for(size_t i=0;i<m.size();++i) {
                int fc_number = i/n_dt_fc;
                double expected = double(fc_number) + double(i+lead_time_hours-fc_number*n_dt_fc )/double(fc_steps);
                TS_ASSERT_DELTA(expected, m.value(i),1e-10);
                utctime p_start=t0+deltahours(lead_time_hours)+i*dt;
                FAST_CHECK_EQ(m.time_axis().period(i),utcperiod(p_start,p_start + dt));
            }
        }
        SUBCASE("empty_vector_gives_empty_ts") {
            auto m = time_series::forecast_merge<time_series::point_ts<time_axis::point_dt>>(tsv_t(),deltahours(0),dt_fc);
            FAST_CHECK_EQ(m.size(),0);
        }
        SUBCASE("missing_fc") {
            // this test is kind of hard to put up with expected values,
            // but with some hacks, it do produce the expected values for verification
            fc.erase(fc.begin()+1,fc.begin()+2); // remove 2'nd forecast, so that we force 1'st to be used until 3rd starts
            FAST_REQUIRE_EQ(n_fc-1,fc.size());
            for( size_t lead_time_hours=0;lead_time_hours<20;++lead_time_hours) {
                auto m = time_series::forecast_merge<time_series::point_ts<time_axis::point_dt>>(fc,deltahours(lead_time_hours),dt_fc);
                // #3: Assert
                FAST_REQUIRE_EQ(m.size(),n_fc*n_dt_fc); // still same size of output, since we borrow data from first ts.
                for(size_t i=0;i<m.size();++i) {
                    int fc_number = i/n_dt_fc;
                    int fc_hour_number = (i-fc_number*n_dt_fc);
                    if(fc_number==1 ) {
                        if(fc_hour_number+lead_time_hours < n_dt_fc ) // this is missing, and replaced with values from fc_number 0
                            fc_number=0;
                        else
                            fc_number=2;//pick remaining values from the 3rd fc.
                    }
                    double expected = double(fc_number) + double(i+lead_time_hours-fc_number*n_dt_fc )/double(fc_steps);
                    TS_ASSERT_DELTA(expected, m.value(i),1e-10);
                    utctime p_start=t0+deltahours(lead_time_hours)+i*dt;
                    FAST_CHECK_EQ(m.time_axis().period(i),utcperiod(p_start,p_start + dt));
                }
            }

        }
    }
    TEST_CASE("tsv_nash_sutcliffe") {
        // arrange
		using namespace shyft;
		using namespace std;
		using ta_t = time_axis::fixed_dt;
		using ts_t = time_series::point_ts<ta_t>;
		using tsv_t = std::vector<ts_t>;
        size_t n_fc=100;
        size_t fc_steps =66;

        core::calendar utc;
        auto t0 = utc.time(2017,1,1,0,0,0);
        auto dt=deltahours(1);
        size_t n_dt_fc =6;
        auto dt_fc= deltahours(n_dt_fc); // typical arome
        tsv_t fc;
        auto point_fx= time_series::ts_point_fx::POINT_AVERAGE_VALUE;
        for(size_t i=0;i<n_fc;++i) {
            vector<double> v;
            for(size_t t=0;t<fc_steps;++t) {
                auto tt= i*n_dt_fc+t;
                v.push_back( tt + sin(0.314+ 3.14*tt/240.0));
            }
            fc.emplace_back(ta_t(t0+i*dt_fc,dt,fc_steps),v,point_fx);
        }
        ts_t obs_ts(ta_t(t0,dt,n_dt_fc*n_fc+fc_steps),0.0,point_fx);
        for(size_t i=0;i<obs_ts.size();++i)
            obs_ts.set(i,i+1.5*cos(3.1415*i/240.0));
        auto  ns=time_series::nash_sutcliffe(fc,obs_ts,0,deltahours(1),6);
        FAST_CHECK_LE(ns,1.0);
        FAST_CHECK_GE(ns,0.99996);
        cout<<"ns:"<<ns<<endl;
    }
}


