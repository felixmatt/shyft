#include "test_pch.h"
#define _USE_MATH_DEFINES
#include "timeseries_test.h"
#include "mocks.h"
#include "core/timeseries.h"
#include <armadillo>
#include <cmath>
#include <functional>


namespace shyfttest {
    const double EPS = 1.0e-8;
    using namespace std;
    using namespace shyft::timeseries;

        /// for testing, this one helps verifying the behavior of the algorithms.
        class test_timeseries {
            vector<point> points;
          public:
                         /** point intepretation: how we should map points to f(t) */
            point_interpretation_policy point_fx=POINT_AVERAGE_VALUE;///< this is special for these test
            point_interpretation_policy point_interpretation() const {return point_fx;}
            void set_point_interpretation(point_interpretation_policy point_interpretation) {point_fx=point_interpretation;}

            test_timeseries() {}
            // make it move-able
            test_timeseries(const test_timeseries &c) : points(c.points) {}
            test_timeseries(test_timeseries&& c) : points(std::move(c.points)) {}
            test_timeseries& operator=(test_timeseries&& c) {points=std::move(c.points);return *this;}

            // constructor from interators
            template< typename S>
            test_timeseries( S points_begin,  S points_end)
              : points(points_begin, points_end) {}

            // Source read interface
            mutable size_t get_count=0;// for testing
            point get(size_t i) const {
                get_count++;
                return points[i];
            }

            /**
             * \note Never use in inner loops, can be extremely costly.
             * \param tx
             * \return Return index i where t(i) <= tx
             */
            mutable size_t index_of_count=0;// for testing
            size_t index_of(const utctime tx) const {
                index_of_count++;
                if (tx < points[0].t) return std::string::npos;
                auto r = lower_bound(points.cbegin(), points.cend(), tx,
                                          [](const point& pt, const utctime& val){ return pt.t <= val; });
                return  static_cast<size_t>(r - points.cbegin()) - 1;
            }
            size_t size() const { return points.size(); }

            // Source write interface
            void set(size_t i, point p) { points[i] = p; }
            void add_point(const point& p) {points.emplace_back(p);}
            void reserve(size_t sz) {points.reserve(sz);}
        };

} // namespace test

using namespace shyft::core;
using namespace shyft::timeseries;

using namespace shyfttest;

typedef point_timeseries<point_timeaxis> xts_t;

void timeseries_test::test_point_timeaxis() {
    point_timeaxis ts0; //zero points
    TS_ASSERT_EQUALS(ts0.size(),0);
    TS_ASSERT_EQUALS(ts0.index_of(12),std::string::npos);
    vector<utctime> t2={3600*1};//just one point
    point_timeaxis ts1(begin(t2),end(t2));
    TS_ASSERT_EQUALS(ts0.size(),0);
    TS_ASSERT_EQUALS(ts0.index_of(12),std::string::npos);

    vector<utctime> t={3600*1,3600*2,3600*3};
    point_timeaxis tx(begin(t),end(t));
    TS_ASSERT_EQUALS(tx.size(),2);// number of periods, - two .. (unless we redefined the last to be last point .. +oo)
    TS_ASSERT_EQUALS(tx(0),utcperiod(t[0],t[1]));
    TS_ASSERT_EQUALS(tx(1),utcperiod(t[1],t[2]));
    TS_ASSERT_EQUALS(tx.index_of(-3600),std::string::npos);//(1),utcperiod(t[1],t[2]);
    TS_ASSERT_EQUALS(tx.index_of(t[0]),0);
    TS_ASSERT_EQUALS(tx.index_of(t[1]-1),0);
    TS_ASSERT_EQUALS(tx.index_of(t[2]+1),1);


}
void timeseries_test::test_timeaxis() {
    auto t0=calendar().time(YMDhms(2000,1,1,0,0,0));
    auto dt=deltahours(1);
    size_t n=3;
    timeaxis tx(t0,dt,n);
    TS_ASSERT_EQUALS(tx.size(),n);
    for(size_t i=0;i<n;++i) {
        TS_ASSERT_EQUALS(tx(i), utcperiod(t0+i*dt,t0+(i+1)*dt));
        TS_ASSERT_EQUALS(tx[i], t0+i*dt );
        TS_ASSERT_EQUALS(tx.index_of(tx[i]),i);
        TS_ASSERT_EQUALS(tx.index_of(tx[i]+dt/2),i)
    }
    TS_ASSERT_EQUALS(tx.index_of(t0+(n+1)*dt),n-1);
    TS_ASSERT_EQUALS(tx.index_of(t0-1),string::npos);

}

void timeseries_test::test_point_source_with_timeaxis() {
    calendar utc;
    auto t=utc.time(YMDhms(2015,5,1,0,0,0));
    auto d=deltahours(1);
    size_t n=10;
    vector<utctime> time_points;for(size_t i=0;i<=n;i++)time_points.emplace_back(t+i*d);
    vector<double> values;for(size_t i=0;i<n;++i) values.emplace_back(i*1.0);
    //Two equal timeaxis representations
    timeaxis fixed_ta(t,d,n);
    point_timeaxis point_ta(time_points);

    point_timeseries<timeaxis> a(fixed_ta,values);
    point_timeseries<point_timeaxis> b(point_ta,values);

    TS_ASSERT_EQUALS(a.total_period(),b.total_period());
    TS_ASSERT_EQUALS(a.size(),b.size());
    TS_ASSERT_EQUALS(a.get_time_axis().size(),b.get_time_axis().size());
    auto t_after= t+ deltahours(24*365*10);
    TS_ASSERT_EQUALS(fixed_ta.index_of(t_after),point_ta.index_of(t_after));

    for(size_t i=0;i<n;i++) {
        // Verify values
        TS_ASSERT_DELTA(values[i],a.get(i).v,shyfttest::EPS);
        TS_ASSERT_DELTA(values[i],b.get(i).v,shyfttest::EPS);
        TS_ASSERT_DELTA(a.value(i),b.value(i),shyfttest::EPS);
        TS_ASSERT_EQUALS(a.get(i),b.get(i));
        // Verify time
        TS_ASSERT_EQUALS(a.time(i),b.time(i));
        TS_ASSERT_EQUALS(a.get(i).t,b.get(i).t);
        // time-index of
        auto ti=fixed_ta[i] + d/2;
        auto ix= fixed_ta.index_of(ti);
        TS_ASSERT_EQUALS(ix,a.index_of(ti));
        TS_ASSERT_EQUALS(ix,b.index_of(ti));
        //add..
        a.add(i,10.0);b.add(i,10.0);
        TS_ASSERT_DELTA(values[i]+10.0,a.value(i),shyfttest::EPS);
        TS_ASSERT_DELTA(values[i]+10.0,b.value(i),shyfttest::EPS);
    }

}
void timeseries_test::test_point_source_scale_by_value() {
    calendar utc;
    auto t=utc.time(YMDhms(2015,5,1,0,0,0));
    auto d=deltahours(1);
    size_t n=10;
    vector<double> values;for(size_t i=0;i<n;++i) values.emplace_back(i*1.0);
    point_timeseries<timeaxis> a(timeaxis(t,d,n),values);
    auto b=a;
    a.scale_by(2.0);
    b.fill(1.0);
    for(size_t i=0;i<n;++i) {
        TS_ASSERT_DELTA(2*values[i],a.value(i),shyfttest::EPS);
        TS_ASSERT_DELTA(1.0,b.value(i),shyfttest::EPS);
    }
}
namespace shyfttest {
    class ts_source {
            utctime start;
            utctimespan dt;
            size_t n;
          public:
            ts_source(utctime start=no_utctime, utctimespan dt=0, size_t n=0) : start(start),dt(dt),n(n) {}
            utcperiod total_period() const { return utcperiod(start,start+n*dt);}
            size_t size() const { return n; }
            utctimespan delta() const {return dt;}

            mutable size_t n_period_calls=0;
            mutable size_t n_time_calls=0;
            mutable size_t n_index_of_calls=0;
            mutable size_t n_get_calls=0;
            size_t total_calls()const {return n_get_calls+n_period_calls+n_time_calls+n_index_of_calls;}
            void reset_call_count() {n_get_calls=n_period_calls=n_time_calls=n_index_of_calls=0;}

            utcperiod operator()(size_t i) const {
                n_period_calls++;
                if(i>n) throw runtime_error("index out of range called");
                return utcperiod(start + i*dt, start + (i + 1)*dt);
            }
            utctime   operator[](size_t i) const {
                n_time_calls++;
                if(i>n) throw runtime_error("index out of range called");
                return utctime(start + i*dt);
            }
            point get(size_t i) const {
                n_get_calls++;
                if(i>n) throw runtime_error("index out of range called");

                return point(start+i*dt,i);}
            size_t index_of(utctime tx) const {
                n_index_of_calls++;
                if(tx < start) return string::npos;
                auto ix = size_t((tx - start)/dt);
                return ix < n ? ix : n - 1;
            }
        };
};
void timeseries_test::test_hint_based_bsearch() {
    calendar utc;
    auto t=utc.time(YMDhms(2015,5,1,0,0,0));
    auto d=deltahours(1);
    size_t n=20;
    shyfttest::ts_source ta(t,d,n);
    shyfttest::ts_source ta_null(t,d,0);
    TS_ASSERT_EQUALS(hint_based_search(ta_null,ta.total_period(),-1),string::npos);//trivial.
    size_t ix;
    TS_ASSERT_EQUALS(ix=hint_based_search(ta,utcperiod(t+d,t+2*d),-1),1);

    TS_ASSERT_EQUALS(ta.n_index_of_calls,1);ta.reset_call_count();
    TS_ASSERT_EQUALS(hint_based_search(ta,utcperiod(t+d,t+2*d),ix),1);
    TS_ASSERT_EQUALS(ta.n_index_of_calls,0);
    TS_ASSERT_EQUALS(hint_based_search(ta,utcperiod(t+2*d,t+3*d),ix),2);
    TS_ASSERT_EQUALS(ta.n_index_of_calls,0);

    TS_ASSERT_EQUALS(hint_based_search(ta,utcperiod(t+7*d,t+8*d),4),7);
    TS_ASSERT_EQUALS(ta.n_index_of_calls,0);// great using linear approach to find near solution upward.

    TS_ASSERT_EQUALS(hint_based_search(ta,utcperiod(t+0*d,t+1*d),4),0);
    TS_ASSERT_EQUALS(ta.n_index_of_calls,0);// great using linear approach to find near solution upward.

    ta.reset_call_count();
    TS_ASSERT_EQUALS(hint_based_search(ta,utcperiod(t+1*d,t+2*d),0),1);
    TS_ASSERT_EQUALS(ta.n_index_of_calls,0);// great using linear approach to find near solution upward.
    TS_ASSERT_EQUALS(ta.n_get_calls,3);// great using linear approach to find near solution upward.

    ta.reset_call_count();

    TS_ASSERT_EQUALS(hint_based_search(ta,utcperiod(t-1*d,t-0*d),5),string::npos);
    TS_ASSERT_EQUALS(ta.n_index_of_calls,1);// great using linear approach to find near solution upward.

    ta.reset_call_count();


    TS_ASSERT_EQUALS(hint_based_search(ta,utcperiod(t+20*d,t+21*d),5),ta.size()-1);
    TS_ASSERT_EQUALS(ta.n_index_of_calls,1);// great using linear approach to find near solution upward.

    ta.reset_call_count();

    TS_ASSERT_EQUALS(hint_based_search(ta,utcperiod(t+2*d,t+3*d),ta.size()+5),2);//shall survive bad usage..
    TS_ASSERT_EQUALS(ta.n_index_of_calls,1);// great using linear approach to find near solution upward.

}

void timeseries_test::test_average_value_staircase() {
    auto t0=calendar().time(YMDhms(2000,1,1,0,0,0));
    auto dt=deltahours(1);
    size_t n=3;
    timeaxis tx(t0,dt,n);
    vector<point> points={point(t0,1.0),point(t0+dt/2,2.0),point(t0+2*dt,3)};


    shyfttest::test_timeseries ps(begin(points),end(points));
    TS_ASSERT_EQUALS(-1,ps.index_of(t0-deltahours(1)));
    TS_ASSERT_EQUALS(0,ps.index_of(t0+deltahours(0)));
    TS_ASSERT_EQUALS(1,ps.index_of(t0+deltahours(1)));
    TS_ASSERT_EQUALS(2,ps.index_of(t0+deltahours(2)));
    TS_ASSERT_EQUALS(2,ps.index_of(t0+deltahours(3)));

    // case 1: just check it can compute true average..
    size_t ix=0;
    auto avg1_2_3=average_value(ps,utcperiod(t0,t0+3*dt),ix,false);
    TS_ASSERT_DELTA((1*0.5+2*1.5+3*1.0)/3.0,avg1_2_3,0.0000001);

    // case 2: check that it can deliver true average at a slice of a stair-case
    ix=-1;
    ps.index_of_count=0;
    TS_ASSERT_DELTA(1.0,average_value(ps,utcperiod(t0+deltaminutes(10),t0+ deltaminutes(11)),ix,false),0.0000001);
    TS_ASSERT_EQUALS(1,ps.index_of_count);
    TS_ASSERT_EQUALS(ix,1);
    // case 3: check that it deliver/keep average value after last value observed
    ps.index_of_count=0;
    ix=2;
    TS_ASSERT_DELTA(3.0,average_value(ps,utcperiod(t0+5*dt,t0+ 60*dt),ix,false),0.0000001);
    TS_ASSERT_EQUALS(0,ps.index_of_count);
    TS_ASSERT_EQUALS(ix,2);
    // case 4: ask for data before first point -> nan
    ix=2;
    ps.index_of_count=0;
    double v=average_value(ps,utcperiod(t0-5*dt,t0- 4*dt),ix,false);
    TS_ASSERT(!std::isfinite(v));
    TS_ASSERT_EQUALS(ps.index_of_count,0);
    TS_ASSERT_EQUALS(ix,0);

    // case 5: check it eats nans (nan-0)
    vector<point> points_with_nan0={point(t0,shyft::nan),point(t0+dt/2,2.0),point(t0+dt,shyft::nan),point(t0+2*dt,3)};
    vector<point> points_with_nan1={point(t0,1.0),point(t0+dt/2,2.0),point(t0+dt,shyft::nan),point(t0+2*dt,3)};
    vector<point> points_with_nan2={point(t0,1.0),point(t0+dt/2,2.0),point(t0+dt,shyft::nan),point(t0+2*dt,shyft::nan)};
    vector<point> points_1={point(t0,1.0)};
    vector<point> points_0;
    vector<point> points_10;for(utctime t=t0;t<10*dt+t0;t+=dt) points_10.push_back(point(t,double(t-t0)/dt) );

    // other corner cases and behaviour testing for the ix
    utcperiod full_period(t0,t0+3*dt);
    shyfttest::test_timeseries ps1(begin(points_with_nan0),end(points_with_nan0));
    shyfttest::test_timeseries ps2(begin(points_with_nan1),end(points_with_nan1));
    shyfttest::test_timeseries ps3(begin(points_with_nan2),end(points_with_nan2));
    shyfttest::test_timeseries ps4(begin(points_1),end(points_1));
    shyfttest::test_timeseries ps5(begin(points_0),end(points_0));
    shyfttest::test_timeseries ps6(begin(points_10),end(points_10));



    TS_ASSERT_DELTA((0.5*2+3.0)/1.5,average_value(ps1,full_period,ix,false),0.00001);
    TS_ASSERT_DELTA((1.0*0.5+0.5*2+3.0)/2.0,average_value(ps2,full_period,ix,false),0.00001);
    TS_ASSERT_DELTA((1.0*0.5+0.5*2+0.0)/1.0,average_value(ps3,full_period,ix,false),0.00001);
    TS_ASSERT_DELTA(1,average_value(ps4,full_period,ix,false),0.00001);
    ix=-1;
    v=average_value(ps5,full_period,ix,false);// no points should give nan
    TS_ASSERT(!std::isfinite(v));
    ix=-1;
    v=average_value(ps4,utcperiod(t0-dt*10,t0-dt*9),ix,false);//before any point should give nan
    TS_ASSERT(!std::isfinite(v));

    ix=7;// ensure negative direction search ends up in doing a binary -search
    v=average_value(ps6,full_period,ix,false);
    TS_ASSERT_EQUALS(ps6.index_of_count,1);
    TS_ASSERT_DELTA((0*1+1*1+2*1.0)/3.0,v,0.00001);
    v=average_value(ps6,utcperiod(t0+3*dt,t0+4*dt),ix,false);
    TS_ASSERT_EQUALS(ps6.index_of_count,1);
    TS_ASSERT_DELTA((3)/1.0,v,0.00001);
    TS_ASSERT_EQUALS(ix,4);
    ix=0;ps6.index_of_count=0;
    v=average_value(ps6,utcperiod(t0+7*dt,t0+8*dt),ix,false);
    TS_ASSERT_EQUALS(ps6.index_of_count,1);
    TS_ASSERT_DELTA((7)/1.0,v,0.00001);
    ps.index_of_count=0;
    average_accessor<shyfttest::test_timeseries,timeaxis> avg_a(ps,tx);
    TS_ASSERT_DELTA(avg_a.value(0),(1*0.5+2*0.5)/1.0,0.000001);//(1*0.5+2*1.5+3*1.0)/3.0
    TS_ASSERT_DELTA(avg_a.value(1),(2*1.0)/1.0,0.000001);//(1*0.5+2*1.5+3*1.0)/3.0
    TS_ASSERT_DELTA(avg_a.value(2),(3*1.0)/1.0,0.000001);//(1*0.5+2*1.5+3*1.0)/3.0
    TS_ASSERT_EQUALS(ps.index_of_count,0);
}


void timeseries_test::test_average_value_linear_between_points() {
	auto t0 = calendar().time(YMDhms(2000, 1, 1, 0, 0, 0));
	auto dt = deltahours(1);
	size_t n = 3;
	timeaxis tx(t0, dt, n);
	vector<point> points = { point(t0, 1.0), point(t0 + dt / 2, 2.0), point(t0 + 2 * dt, 3) };


	shyfttest::test_timeseries ps(begin(points), end(points));
	TS_ASSERT_EQUALS(-1, ps.index_of(t0 - deltahours(1)));
	TS_ASSERT_EQUALS(0, ps.index_of(t0 + deltahours(0)));
	TS_ASSERT_EQUALS(1, ps.index_of(t0 + deltahours(1)));
	TS_ASSERT_EQUALS(2, ps.index_of(t0 + deltahours(2)));
	TS_ASSERT_EQUALS(2, ps.index_of(t0 + deltahours(3)));

	// case 1: just check it can compute true average..
	size_t ix = 0;
	auto avg1_2_3 = average_value(ps, utcperiod(t0, t0 + 3 * dt), ix);
	TS_ASSERT_DELTA((1.5 * 0.5 + 2.5 * 1.5 + 3 * 1.0) / 3.0, avg1_2_3, 0.0000001);

	// case 2: check that it can deliver true average at a slice of a stair-case
	ix = -1;
	ps.index_of_count = 0;
	TS_ASSERT_DELTA(1.0+ 2*10.5/60, average_value(ps, utcperiod(t0 + deltaminutes(10), t0 + deltaminutes(11)), ix), 0.0000001);
	TS_ASSERT_EQUALS(1, ps.index_of_count);
	TS_ASSERT_EQUALS(ix, 1);
	// case 3: check that it deliver/keep average value after last value observed
	ps.index_of_count = 0;
	ix = 2;
	TS_ASSERT_DELTA(3.0, average_value(ps, utcperiod(t0 + 5 * dt, t0 + 60 * dt), ix), 0.0000001);
	TS_ASSERT_EQUALS(0, ps.index_of_count);
	TS_ASSERT_EQUALS(ix, 2);
	// case 4: ask for data before first point -> nan
	ix = 2;
	ps.index_of_count = 0;
	double v = average_value(ps, utcperiod(t0 - 5 * dt, t0 - 4 * dt), ix);
	TS_ASSERT(!std::isfinite(v));
	TS_ASSERT_EQUALS(ps.index_of_count, 0);
	TS_ASSERT_EQUALS(ix, 0);

	// case 5: check it eats nans (nan-0)
	vector<point> points_with_nan0 = { point(t0, shyft::nan), point(t0 + dt / 2, 2.0), point(t0 + dt, shyft::nan), point(t0 + 2 * dt, 3) };
	vector<point> points_with_nan1 = { point(t0, 1.0), point(t0 + dt / 2, 2.0), point(t0 + dt, shyft::nan), point(t0 + 2 * dt, 3) };
	vector<point> points_with_nan2 = { point(t0, 1.0), point(t0 + dt / 2, 2.0), point(t0 + dt, shyft::nan), point(t0 + 2 * dt, shyft::nan) };
	vector<point> points_1 = { point(t0, 1.0) };
	vector<point> points_0;
	vector<point> points_10; for (utctime t = t0; t < 10 * dt + t0; t += dt) points_10.push_back(point(t, double(t - t0) / dt));

	// other corner cases and behaviour testing for the ix
	utcperiod full_period(t0, t0 + 3 * dt);
	shyfttest::test_timeseries ps1(begin(points_with_nan0), end(points_with_nan0));
	shyfttest::test_timeseries ps2(begin(points_with_nan1), end(points_with_nan1));
	shyfttest::test_timeseries ps3(begin(points_with_nan2), end(points_with_nan2));
	shyfttest::test_timeseries ps4(begin(points_1), end(points_1));
	shyfttest::test_timeseries ps5(begin(points_0), end(points_0));
	shyfttest::test_timeseries ps6(begin(points_10), end(points_10));



	TS_ASSERT_DELTA((0.5 * 2 + 3.0) / 1.5, average_value(ps1, full_period, ix), 0.00001);
	TS_ASSERT_DELTA(2.3750, average_value(ps2, full_period, ix), 0.00001);
	TS_ASSERT_DELTA(1.75, average_value(ps3, full_period, ix), 0.00001);
	TS_ASSERT_DELTA(1, average_value(ps4, full_period, ix), 0.00001);
	ix = -1;
	v = average_value(ps5, full_period, ix);// no points should give nan
	TS_ASSERT(!std::isfinite(v));
	ix = -1;
	v = average_value(ps4, utcperiod(t0 - dt * 10, t0 - dt * 9), ix);//before any point should give nan
	TS_ASSERT(!std::isfinite(v));

	ix = 7;// ensure negative direction search ends up in doing a binary -search
	v = average_value(ps6, full_period, ix);
	TS_ASSERT_EQUALS(ps6.index_of_count, 1);
	TS_ASSERT_DELTA(1.5, v, 0.00001);
	v = average_value(ps6, utcperiod(t0 + 3 * dt, t0 + 4 * dt), ix);
	TS_ASSERT_EQUALS(ps6.index_of_count, 1);
	TS_ASSERT_DELTA(3.5, v, 0.00001);
	TS_ASSERT_EQUALS(ix, 4);
	ix = 0; ps6.index_of_count = 0;
	v = average_value(ps6, utcperiod(t0 + 7 * dt, t0 + 8 * dt), ix);
	TS_ASSERT_EQUALS(ps6.index_of_count, 1);
	TS_ASSERT_DELTA(7.5, v, 0.00001);
	ps.index_of_count = 0;
	ps.set_point_interpretation(POINT_INSTANT_VALUE);
	average_accessor<shyfttest::test_timeseries, timeaxis> avg_a(ps, tx);
	TS_ASSERT_DELTA(avg_a.value(0), 1.83333333333, 0.000001);//(1*0.5+2*1.5+3*1.0)/3.0
	TS_ASSERT_DELTA(avg_a.value(1), 2.6666666666, 0.000001);//(1*0.5+2*1.5+3*1.0)/3.0
	TS_ASSERT_DELTA(avg_a.value(2), (3 * 1.0) / 1.0, 0.000001);//(1*0.5+2*1.5+3*1.0)/3.0
	TS_ASSERT_EQUALS(ps.index_of_count, 0);
}




using namespace shyft::core;
using namespace shyft::timeseries;
/// average_value_staircase_fast:
/// Just keept for the reference now, but
/// the generic/complex does execute at similar speed
/// combining linear/stair-case into one function.
//
        template <class S>
        double average_value_staircase_fast(const S& source, const utcperiod& p, size_t& last_idx) {
            const size_t n=source.size();
            if (n == 0) // early exit if possible
                return shyft::nan;
            size_t i=hint_based_search(source,p,last_idx);  // use last_idx as hint, allowing sequential periodic average to execute at high speed(no binary searches)

            if(i==std::string::npos) // this might be a case
                return shyft::nan; // and is equivalent to no points, or all points after requested period.

            point l;          // Left point
            l = source.get(i);
            if (l.t >= p.end) { // requested period before first point,
                last_idx=i;
                return shyft::nan; // defined to return nan
            }

            if(l.t<p.start) l.t=p.start;//project left value to start of interval
            if(!std::isfinite(l.v))
                l.v=0.0;// nan-is 0 kind of def. fixes issues if period start with some nan-points, then area will be zero until first point..

            if( ++i >= n) { // advance to next point and check for early exit
                last_idx=--i;
                return l.v*(p.end-l.t)/p.timespan(); // nan-is 0 kind of def..
            }
            double area = 0.0;  // Integrated area
            point r(l);
            do {
                if( i<n ) { // if possible, advance to next point
                    r=source.get(i++);
                    if(r.t>p.end)
                        r.t=p.end;//clip to p.end to ensure correct integral time-range
                } else {
                    r.t=p.end; // else set right hand side time to p.end(we are done)
                }
                area += l.v*(r.t -l.t);// add area for the current stair-case
                if(std::isfinite(r.v)) { // could be a new value, which establish the new stair-case
                    l=r;
                } else { // not a valid new value, so
                    l.t=r.t;// just keep l.value, adjust time(ignore nans)
                }
            } while(l.t < p.end );
            last_idx = --i; // Store last index, so that next call starts at a smart position.
            return area/p.timespan();
        }

template <class S, class TA>
class average_staircase_accessor_fast {
private:
	static const size_t npos = -1;  // msc doesn't allow this std::basic_string::npos;
	mutable size_t last_idx;
	mutable size_t q_idx;// last queried index
	mutable double q_value;// outcome of
	const TA& time_axis;
	const S& source;
public:
	average_staircase_accessor_fast(const S& source, const TA& time_axis)
		: last_idx(0), q_idx(npos), q_value(0.0), time_axis(time_axis), source(source) { /* Do nothing */
	}

	size_t get_last_index() { return last_idx; }  // TODO: Testing utility, remove later.

	double value(const size_t i) const {
		if (i == q_idx)
			return q_value;// 1.level cache, asking for same value n-times, have cost of 1.
		q_value = average_value_staircase_fast(source, time_axis(q_idx = i), last_idx);
		return q_value;
	}

	size_t size() const { return time_axis.size(); }
};



void timeseries_test::test_TxFxSource() {
    utctime t0 = calendar().time(YMDhms(1940, 1, 1, 0,0, 0));
    utctimespan dt = deltahours(1);
	size_t n = 4*100*24*365;
	typedef std::function<double(utctime)> f_t;
    f_t fx_sin = [t0, dt](utctime t) { return 0.2*(t - t0)/dt; }; //sin(2*3.14*(t - t0)/dt);
    typedef function_timeseries<timeaxis, f_t> txfx_t;

    timeaxis tx(t0, dt, n);
    txfx_t fsin(tx, fx_sin,POINT_AVERAGE_VALUE);
    TS_ASSERT_EQUALS(fsin.size(), tx.size());
    TS_ASSERT_DELTA(fsin(t0 + dt), fx_sin(t0+dt), shyfttest::EPS);
    TS_ASSERT_EQUALS(fsin.get(0), point(t0, fx_sin(t0)));
    /// Some speedtests to check/verify that even more complexity could translate into fast code:
	timeaxis td(t0, dt * 25, n / 24);
	average_accessor<txfx_t, timeaxis> favg(fsin, td);
	average_staircase_accessor_fast<txfx_t, timeaxis> gavg(fsin, td);
	auto f1 = [&favg, &td](double &sum) {sum = 0.0; for (size_t i = 0; i < td.size(); ++i) sum += favg.value(i); };
	auto f2 = [&gavg, &td](double &sum) { sum = 0.0; for (size_t i = 0; i < td.size(); ++i) sum += gavg.value(i); };
	double s1,s2;
	auto msec1 = measure<>::execution(f1,s1);
	auto msec2 = measure<>::execution(f2, s2);
	cout<<"Timing results:"<<endl;
	cout << "\tT generic :" << msec1 << "(" << s1 << ")" << endl;
	cout << "\tT fast    :" << msec2 << "(" << s2 << ")" << endl;

}

void timeseries_test::test_point_timeseries_with_point_timeaxis() {
    vector<utctime> times={3600*1,3600*2,3600*3,3600*4};
    vector<double> points={1.0,2.0,3.0};
    point_timeseries<point_timeaxis> ps(point_timeaxis(times),points);
    TS_ASSERT_EQUALS(ps.size(),3);
    for(size_t i=0;i<ps.size();++i) {
        TS_ASSERT_EQUALS(ps.get(i).v,points[i]);

    }
}



void timeseries_test::test_time_series_difference() {
    using namespace shyfttest::mock;
    const size_t n = 10;
    vector<utctime> ta_times(n);
    const utctime T0 = 0;
    const utctime T1 = 100;
    for (size_t i = 0; i < n; ++i)
        ta_times[i] = T0 + (T1 - T0)*i/(n - 1);
    point_timeaxis time_axis(ta_times.cbegin(), ta_times.cend());

}

void timeseries_test::test_ts_weighted_average(void) {
    using pts_t=point_timeseries<timeaxis>;
	calendar utc;
	utctime start = utc.time(YMDhms(2000, 1, 1, 0, 0, 0));
	utctimespan dt = deltahours(1);
	size_t n = 10;
	timeaxis ta(start, dt, n);
	pts_t r(ta,0.0);
	pts_t c2(ta, 1.0); double a2 = 1.0;
	pts_t c1(ta, 10.0); double a1 = 10.0;
	r.add(c1);
	r.add(c2);
	r.scale_by(1 / 11.0);
	for (size_t i = 0; i < r.size(); ++i) {
		TS_ASSERT_DELTA(r.value(i), 1.0, 0.0001);
	}
	pts_t rs(ta, 0.0);
	rs.add_scale(c1, a1);
	rs.add_scale(c2, a2);
	rs.scale_by(1 / (a1 + a2));
	for (size_t i = 0; i < rs.size(); ++i) {
		TS_ASSERT_DELTA(rs.value(i), (1.0*a2+10.0*a1)/(a1+a2), 0.0001);
	}
}


void timeseries_test::test_sin_fx_ts() {
	sin_fx fx(10.0, 0.0, 10.0, 10.0, 0, deltahours(24));
	TS_ASSERT_DELTA(fx(0), 10.0, 0.000001);
	TS_ASSERT_DELTA(fx(deltahours(24)), 10.0, 0.000001);
	calendar utc;
	utctime start = utc.time(YMDhms(2000, 1, 1, 0, 0, 0));
	utctimespan dt = deltahours(1);
	size_t n = 10;
	timeaxis ta(start, dt, n);
	function_timeseries<timeaxis, sin_fx> tsfx(ta, fx);

	TS_ASSERT_DELTA(tsfx(0), 10.0, 0.0000001);
	TS_ASSERT_DELTA(tsfx(deltahours(24)), 10.0, 0.0000001);
	TS_ASSERT_DELTA(tsfx(deltahours(18)), 0.0, 0.000001);

}
/**
thoughts for ts operations.
struct ts {
  // most general level:
  double operator()(utctime t)const  {..}
  // point/index based operations
  size_t size();
  size_t index_of(utctime t) const {}
  utctime time(i) const {}
  double  value(i) const {}
  //maybe this ?
  timeaxis& timeaxis() const {}

}

Given a point ts is described as

struct pts {
    timeaxis ta; // could be of type start,dt,n, optional calendar, or list of periods, or both.
    value<double> v;
};

outside timeaxis, we could say f(t), f(i) = nan,

if timeaxis of a and b ::

   - all equal --> trivial & high speed
   - a or b is a constant -> trivial & high speed
   - equal dt --> use ix-offset, high speed (question about nan handling)
   - != dt --> a lot of considerations:
                the 'effective' timeaxis, the size and the points
                translations from effective timeaxis to a-timeaxis & b-timeaxis
                when the above is done, -its quite straight forward.
                 e.g. t= effective.timeaxis.index_of(i), .. etc.

   conclusion: complexity and speed depends on how efficient we can handle time-axis operations.
    a class that combines two time-axis a+b into a new one, still keeping refs. to a&b
    seems useful.
   time-axis for a constant ? .size()=0 total_period() -oo .. +oo

expression tree & references vs. shared_ptr:
     c= a + b*d

     should we keep references to terminals(ts), or should we use reference counting,
     maybe we should support both ?

      (assuming copy  or by value is ruled out due to performance hit?
       or should we just do that and wrap shared_ptr<ts> into 'by value' ts ?

       ).

     references is extreme fast , shared_ptr is quite fast, and in the context of handling ts
     a minor contributor to the overall performance.

     If we are going for shared_ptr, it 'bleeds' out the terminals, they need to be shared_ptr as
     well.





*/
namespace shyft{namespace think {
    using namespace std;
    using namespace timeseries;


    //--- to avoid duplicating algorithms in classes where the stored references are
    // either by shared_ptr, or by value, we use template function:
    //  to make shared_ptr<T> appear equal to T
    template<typename T>
    struct is_shared_ptr {static const bool value=false;};
    template<typename T>
    struct is_shared_ptr<shared_ptr<T>> {static const bool value=true;};

    ///< T d_ref(T) template to return a ref or const ref to T, if T is shared_ptr<T> or just T
    template< typename U>
    typename  enable_if<!is_shared_ptr<U>::value,U>::type &
      d_ref(U& t) {return t;}

    template< typename U>
    typename  enable_if<!is_shared_ptr<U>::value,U>::type const &
      d_ref(const U& t) {return t;}

    template< typename U>
    typename  enable_if<is_shared_ptr<U>::value,U>::type::element_type &
      d_ref(U& t) {return *t;}
    template< typename U>
    typename  enable_if<is_shared_ptr<U>::value,U>::type::element_type const &
      d_ref(const U& t) {return *t;}

    ///< \brief d_ref_t template to rip out T of shared_ptr<T> or T if T specified
    template <class T, class P = void >
    struct d_ref_t { };

    template <class T >
    struct d_ref_t<T,typename enable_if<is_shared_ptr<T>::value>::type> {
        typedef typename T::element_type type;
    };
    template <class T >
    struct d_ref_t<T,typename enable_if<!is_shared_ptr<T>::value>::type > {
        typedef T type;
    };
    //---

    enum point_fx_policy { // how to interpret f(t) between two points
        LINEAR=0,
        STAIR_CASE=1
    };
    inline point_fx_policy result_policy(point_fx_policy a, point_fx_policy b) {
        return a==point_fx_policy::LINEAR || b==point_fx_policy::LINEAR?point_fx_policy::LINEAR:point_fx_policy::STAIR_CASE;
    }

    namespace time_axis {

        //time_axis-traits we can use to generate more efficient code
        template <bool T>
        struct continuous {
            static const bool value=false;
        };
        template<>
        struct continuous<true> {
            static const bool value=true;
        };


        /**
        timeaxis questions:

            def: an ordered sequence of non-overlapping periods

            notice:
               a) the most usual time-axis is the fixed_dt time-axis
                  in the SHyFT core, this is the one we use(could even need a highspeed-nocompromise version)

               b) continuous time-axis: there are no holes in total_period
                                   types: fixed_dt,calendar_dt, point_dt
               c) sparse time-axis: there are holes within the span of total_period
                                   types: period_list and calendar_dt_p

            Q&A:

            Should we go for one catch-all generic timeaxis or should we gain speed using
            specialized timeaxis :
                ta_period_list(period-list)
                ta_points_list(total_period + points)
                ta_regular(t0,dt,n),
                ta_cal(cal,t0,dt,n)
                ta_full_flex(cal,t0,dt,n, sub-period-list..)
            If we do full_flex time-axis, as "one catch all cases",
            - we can't use templates to gain speed.
             * (but we need some runtime anyhow, since they can still be 'different').
            + easy to expose to python.
            * combining timeaxis a and b if full-flex timeaxis:
                code-driven (if regular&regular .. code-size is larger?)
                worst-case ?
                  a full merge of periods,
                   and then we need to define semantics.
                easiest case ?
                  a and b timeaxis are equal
                usual cases?
                  a and b have same dt&cal, but different start&n
                  a and b have different dt (hour, 3hour), same cal
            * polymorfism ? i_time_axis etc, no- the cost of virtual call >> inline regular 90%.

            q:when is it useful to do
             for(size_t i= first; i<last;++i)
                op(a(ix_of_a(i)),b(ix_of_b(i)) ?
            a: evaluating expression trees, e.g. c= 3*a + b*d
                case specified result time-axis
                case unspecified result time-axis (derived from expression).

            finally: should the ts have a timeaxis ?
             + probably,  it's a central tool when doing arithmetic
             + what about a constant ts timeaxis ?
               - it can be size()==0 and total_period(-oo..+oo) for a constant ts
             + null timeaxis ? total_period() == null period, size()=0
             + should timeaxis be value-type or reference/shared_ptr ?
               regular (start,dt,n) is typical value type case
               even a
               regular (cal,dt,n) is a typical value type case, given cal is reference
               for point-timeaxis: well, they arise from uneven sampled values, so they are by definition different.
                                   we *could* try to convert them to fixed_dt time-axis (most will fit in there)
               for period-timeaxis: they play a role mainly in analytics, compute production during sun-hours

        */

        /**\brief a simple regular time-axis, starting at t,
         * with n consecutive periods of fixed length dt
         * there are 5 different useful types of calendars, this is the most efficient and popular one
         */
        struct fixed_dt:continuous<true> {

            utctime t;
            utctimespan dt;
            size_t n;

            fixed_dt():t(no_utctime),dt(0),n(0){}
            fixed_dt(utctime t,utctimespan dt,size_t n):t(t),dt(dt),n(n) {}

            size_t size() const {return n;}

            utcperiod total_period() const {
                return n==0?
                 utcperiod(min_utctime,min_utctime):// maybe just a non-valid period?
                 utcperiod(t,t+n*dt);
            }

            utctime time(size_t i) const {
                if(i<n) return t+i*dt;
                throw std::out_of_range("fixed_dt.time(i)");
            }

            utcperiod period(size_t i) const {
                if(i<n) return utcperiod(t+i*dt,t+(i+1)*dt);
                throw std::out_of_range("fixed_dt.period(i)");
            }

            size_t index_of(utctime tx) const {
                if(tx<t) return std::string::npos;
                size_t r= (tx-t)/dt;
                if(r<n) return r;
                return std::string::npos;
            }
            size_t open_range_index_of(utctime tx) const {return n>0 && (tx>=t+utctimespan(n*dt))?n-1: index_of(tx);}
            static fixed_dt full_range() {return fixed_dt(min_utctime,max_utctime,2);} //Hmm.
            static fixed_dt null_range() {return fixed_dt(0,0,0);}
        };

        /** A variant of time_axis that adheres to calendar periods, possibly including DST handling
         *  e.g.: a calendar day might be 23,24 or 25 hour long in a DST calendar.
         *  If delta-t is less or equal to one hour, it's close to as efficient as time_axis
         */
        struct calendar_dt:continuous<true>  {
            static constexpr utctimespan dt_h=3600;
            shared_ptr<const calendar> cal;
            utctime t;
            utctimespan dt;
            size_t n;

            calendar_dt():t(no_utctime),dt(0),n(0){}
            calendar_dt(const shared_ptr<const calendar>& cal, utctime t,utctimespan dt,size_t n):cal(cal),t(t),dt(dt),n(n) {}

            size_t size() const {return n;}

            utcperiod total_period() const {
                return n==0?
                 utcperiod(min_utctime,min_utctime):// maybe just a non-valid period?
                 utcperiod(t,dt<=dt_h?t+n*dt:cal->add(t,dt,n) );
            }

            utctime time(size_t i) const {
                if(i<n) return dt<=dt_h?t+i*dt:cal->add(t,dt,i);
                throw out_of_range("calendar_dt.time(i)");
            }

            utcperiod period(size_t i) const {
                if(i<n) return dt<dt_h ? utcperiod(t+i*dt,t+(i+1)*dt) : utcperiod(cal->add(t,dt,i),cal->add(t,dt,i+1));
                throw out_of_range("calendar_dt.period(i)");
            }

            size_t index_of(utctime tx) const {
                auto p=total_period();
                if(!p.contains(tx))
                    return string::npos;
                return dt<=dt_h?
                    (size_t) ((tx-t)/dt):
                    (size_t) cal->diff_units(t,tx,dt);
            }
            size_t open_range_index_of(utctime tx) const {return tx>=total_period().end && n>0 ?n-1:index_of(tx);}

        };

        /** Another popular representation of time-axis, using n time points + end-point,
        * where interval(i) utcperiod(t[i],t[i+1])
        *          except for the last
        *                   utcperiod(t[i],te)
        *
        * very flexible, but inefficient in space and time
        * to minimize the problem the, .index_of() provides
        *  'ix_hint' to hint about the last location used
        *    then if specified, search +- 10 positions to see if we get a hit
        *   otherwise just a binary-search for the correct index.
        *  TODO: maybe even do some smarter partitioning, guessing equidistance on average between points.
        *        then guess the area (+-10), if within range, search there ?
        */
        struct point_dt:continuous<true>  {
            vector<utctime> t;
            utctime  t_end;// need one extra, after t.back(), to give the last period!
            point_dt(){}
            point_dt(const vector<utctime>& t,utctime t_end):t(t),t_end(t_end) {
             //TODO: throw if t.back()>= t_end
             // consider t_end==no_utctime , t_end=t.back()+tick.
            }

            size_t size() const {return t.size();}

            utcperiod total_period() const {
                return t.size()==0?
                 utcperiod(min_utctime,min_utctime):// maybe just a non-valid period?
                 utcperiod(t[0],t_end);
            }

            utctime time(size_t i) const {
                if(i<t.size()) return t[i];
                throw std::out_of_range("point_dt.time(i)");
            }

            utcperiod period(size_t i) const {
                if(i<t.size())  return  utcperiod(t[i],i+1<t.size()?t[i+1]:t_end);
                throw std::out_of_range("point_dt.period(i)");
            }

            size_t index_of(utctime tx, size_t ix_hint=std::string::npos) const {
                if(t.size()==0 || tx<t[0] || tx >= t_end) return std::string::npos;
                if(tx >= t.back()) return t.size()-1;

                if(ix_hint != std::string::npos && ix_hint<t.size()) {
                    if(t[ix_hint]==tx) return ix_hint;
                    const size_t max_directional_search=10;// just  a wild guess
                    if(t[ix_hint]<tx ) {
                        size_t j=0;
                        while(t[ix_hint] < tx && ++j <max_directional_search && ix_hint<t.size()) {
                            ix_hint++;
                        }
                        if(t[ix_hint]>=tx || ix_hint==t.size() ) // we startet below p.start, so we got one to far(or at end), so revert back one step
                            return ix_hint-1;
                        // give up and fall through to binary-search
                    } else {
                        size_t j=0;
                        while(t[ix_hint]>tx && ++j <max_directional_search && ix_hint >0) {
                            --ix_hint;
                        }
                        if(t[ix_hint]>tx && ix_hint>0) // if we are still not before p.start, and i is >0, there is a hope to find better index, otherwise we are at/before start
                          ; // bad luck searching downward, need to use binary search.
                        else
                            return ix_hint;
                    }
                }

                auto r = lower_bound(t.cbegin(), t.cend(), tx,
                            [](utctime pt, utctime val){ return pt <= val; });
                return static_cast<size_t>(r - t.cbegin()) - 1;
            }
            size_t open_range_index_of(utctime tx) const {return size()>0 && tx>=t_end? size()-1:index_of(tx);}

            static point_dt null_range() {
                return point_dt{};
            }

        };

        /** \brief a generic (not sparse) time interval time-axis
         * that can be anyone of the dense (non sparse) time-axis types
         * we need this one to resolve into a best possible time-axis runtime
         * since we can not decide everything compile-time
         */
        struct generic_dt :continuous<true> {
            //--possible implementation types:
            enum generic_type { FIXED=0,CALENDAR=1,POINT=2};
            generic_type gt;
            fixed_dt f;
            calendar_dt c;
            point_dt p;
            //---------------
            generic_dt():gt(FIXED){}
            generic_dt(const fixed_dt&f):gt(FIXED),f(f){}
            generic_dt(const calendar_dt &c):gt(CALENDAR),c(c){}
            generic_dt(const point_dt& p):gt(POINT),p(p){}
            bool is_fixed_dt() const {return gt==POINT;}
            //const generic_dt& most_optimal(const generic_dt& other)const { if(gt==FIXED)return *this;if(other.gt==FIXED)return other;if(gt==CALENDAR)return *this;if(other.gt==CALENDAR)return other;return *this;}
            size_t size() const          {switch(gt){default:case FIXED:return f.size();case CALENDAR:return c.size();case POINT:return p.size();}}
            utcperiod total_period()const  {switch(gt){default:case FIXED:return f.total_period();case CALENDAR:return c.total_period();case POINT:return p.total_period();}}
            utcperiod period(size_t i)const {switch(gt){default:case FIXED:return f.period(i);case CALENDAR:return c.period(i);case POINT:return p.period(i);}}
            utctime     time(size_t i)const {switch(gt){default:case FIXED:return f.time(i);case CALENDAR:return c.time(i);case POINT:return p.time(i);}}
            size_t index_of(utctime t)const {switch(gt){default:case FIXED:return f.index_of(t);case CALENDAR:return c.index_of(t);case POINT:return p.index_of(t);}}
            size_t open_range_index_of(utctime t) const {switch(gt){default:case FIXED:return f.open_range_index_of(t);case CALENDAR:return c.open_range_index_of(t);case POINT:return p.open_range_index_of(t);}}

        };

        /** \brief Yet another variant of calendar time-axis,
         * This one is similar to calendar_dt, except, each main-period given by
         * calendar_dt have sub-periods.
         * E.g. you would like to have a weekly time-axis that represents all working-hours
         * etc.
         * If the sub-period(s) specified are null, then this time_axis is equal to calendar_dt.
         *
         */
        struct calendar_dt_p:continuous<false>  {
            calendar_dt cta;
            vector<utcperiod> p;// sub-periods within each cta.period, using cta.period.start as t0
            calendar_dt_p(const shared_ptr<const calendar>& cal, utctime t,utctimespan dt,size_t n,vector<utcperiod> p)
             :cta(cal,t,dt,n),p(move(p)) {
              // TODO: validate p, each p[i] non-overlapping and within ~ dt
              // possibly throw if invalid period, but
              // this could be tricky, because the 'gross period' will vary over the timeaxis,
              // so worst case we would have to run through all gross-periods and verify that
              // sub-periods fits-within each gross period.

             }

            size_t size() const { return cta.size()*( p.size() ? p.size() : 1 );}

            utcperiod total_period() const { return cta.total_period();}

            utctime time(size_t i) const { return period(i).start;}

            utcperiod period(size_t i) const {
                if( i < size() ) {
                    size_t main_ix= i/p.size();
                    size_t p_ix= i - main_ix*p.size();
                    auto pi=cta.period(main_ix);
                    return p.size()?
                        utcperiod(cta.cal->add(pi.start,p[p_ix].start,1),cta.cal->add(pi.start,p[p_ix].end,1)):
                        pi;
                }
                throw out_of_range("calendar_dt_p.period(i)");
            }

            size_t index_of(utctime tx,bool period_containment=true) const {
                auto tp=total_period();
                if(!tp.contains(tx))
                    return string::npos;
                size_t main_ix=cta.index_of(tx);
                if(p.size()==0)
                    return main_ix;

                utctime t0=cta.time(main_ix);
                for(size_t i=0;i<p.size();++i) { //important: p is assumed to have size like 5, 7 (workdays etc).
                    utcperiod pi(cta.cal->add(t0,p[i].start,1),cta.cal->add(t0,p[i].end,1));
                    if(pi.contains(tx))
                        return p.size()*main_ix + i;
                    if(!period_containment) { // if just searching for the closest period back in time, then:
                        if(pi.start>tx ) // the previous period is before tx, then we know we have a hit
                            return p.size()*main_ix + (i>0?i-1:0);// it might be a hit just before first sub-period as well. Kind of special case(not consistent,but useful)
                        if( i+1 == p.size()) // the current period is the last one, then we also know we have a hit
                            return p.size()*main_ix +i;

                    }
                }

                return string::npos;
            }
            size_t open_range_index_of(utctime tx) const {
                return size()>0 && tx>=total_period().end?size()-1:index_of(tx,false);
            }
        };




        /**\brief The by definition ultimate time-axis using
         * the canonical definition of time-axis:
         * an ordered sequence of non-overlapping periods
         *
         */
        struct period_list:continuous<false> {
            vector<utcperiod> p;
            period_list(){}
            period_list(const vector<utcperiod>& p):p(p) {}
            template< class TA>
            static period_list convert(const TA& ta) {
                period_list r;
                r.p.reserve(ta.size());
                for(size_t i=0;i<ta.size();++i)
                    r.p.push_back(ta.period(i));
                return r;
            }
            size_t size() const {return p.size();}

            utcperiod total_period() const {
                return p.size()==0?
                 utcperiod(min_utctime,min_utctime):// maybe just a non-valid period?
                 utcperiod(p.front().start,p.back().end);
            }

            utctime time(size_t i) const {
                if(i<p.size()) return p[i].start;
                throw std::out_of_range("period_list.time(i)");
            }

            utcperiod period(size_t i) const {
                if(i<p.size())  return  p[i];;
                throw std::out_of_range("period_list.period(i)");
            }

            size_t index_of(utctime tx, size_t ix_hint=std::string::npos,bool period_containment=true) const {
                if(p.size()==0 || tx<p.front().start || tx >= p.back().end ) return std::string::npos;

                if(ix_hint != string::npos && ix_hint<p.size()) {
                    if( (period_containment && p[ix_hint].contains(tx)) ||
                       ( !period_containment && (p[ix_hint].start>=tx && (ix_hint+1<p.size()?tx<p[ix_hint+1].start:true)    )) ) return ix_hint;
                    const size_t max_directional_search=10;// just  a wild guess
                    if(p[ix_hint].end < tx ) {
                        size_t j=0;
                        while(p[ix_hint].end < tx && ++j <max_directional_search && ix_hint<p.size()) {
                            ix_hint++;
                        }
                        if(ix_hint==p.size())
                            return string::npos;
                        if(p[ix_hint].contains(tx)) // ok this is it.
                            return ix_hint;
                        else if(!period_containment && ix_hint+1<p.size() && tx>=p[ix_hint].start && tx<p[ix_hint+1].start)
                            return ix_hint;// this is the closest period <= to tx
                        // give up and fall through to binary-search
                    } else {
                        size_t j=0;
                        while(p[ix_hint].start>tx && ++j <max_directional_search && ix_hint >0) {
                            --ix_hint;
                        }
                        if(p[ix_hint].contains(tx))
                            return ix_hint;
                        else if(!period_containment && ix_hint+1<p.size() && tx>=p[ix_hint].start && tx<p[ix_hint+1].start)
                            return ix_hint;// this is the closest period <= to tx

                        // give up, binary search.
                    }
                }

                auto r = lower_bound(p.cbegin(), p.cend(), tx,
                            [](utcperiod pt, utctime val){ return pt.start <= val; });
                size_t ix= static_cast<size_t>(r - p.cbegin()) - 1;
                if( ix ==string::npos || p[ix].contains(tx))
                    return ix; //cover most cases, including period containment
                if (!period_containment && tx >= p[ix_hint].start && ix_hint+1 < p.size() && tx<p[ix_hint+1].start)
                    return ix;// ok this is the closest period that matches
                return string::npos;// no match to period,what so ever
            }
            size_t open_range_index_of(utctime tx) const {
                return size()>0&&tx>=total_period().end?size()-1:index_of(tx,string::npos,false);
            }

            static period_list null_range() {
                return period_list();
            }
        };


        /// combining time_axis types are important for
        /// ts-math operations (speed&efficiency)
        /// Number of combine() functions we have to write with this approach is equal
        /// to 1 + 2 + 3 +..n types.
        /// number of time-axis types are
        ///  (1)      t,dt,n  -- this is the most common
        ///  (2) cal t,dt,n   -- also very common, if dt<=1h, it's equal to (1)
        ///  (3) cal t,dt,n, (sub) period_list -- less common, if dt<=1h, it's equal to (1)
        ///  (4) period_list  -- the ultimate generic definition, but not common
        ///  (5) vector<t> plus t_end -- very common at the input/sample side, break-point type of ts.
        ///= 15
        ///  we could write one generic algorithm(that do have some penalty cost)
        ///   and then specialize for the most popular one ?
        ///    then number are down to 3-5 ?


        /** \brief fast&efficient combine for two fixed_dt time-axis */
        inline fixed_dt combine(const fixed_dt& a,const fixed_dt& b)  {
            // 0. check if they overlap (todo: this could hide dt-errors checked for later)
            utcperiod pa=a.total_period();
            utcperiod pb=b.total_period();
            if( !pa.overlaps(pb) || a.size()==0 || b.size()==0 )
                return fixed_dt::null_range();
            if( a.dt == b.dt) {
                if(a.t == b.t && a.n == b.n) return a;
                utctime t0=max(pa.start,pb.start);
                return fixed_dt(t0,a.dt,(min(pa.end,pb.end) - t0)/a.dt);
            } if( a.dt > b.dt) {
                if((a.dt % b.dt)!=0) throw std::runtime_error("combine(fixed_dt a,b) needs dt to align");
                utctime t0=max(pa.start,pb.start);
                return fixed_dt(t0,b.dt,(min(pa.end,pb.end) - t0)/b.dt);
            } else {
                if ((b.dt % a.dt)!=0)
                    throw std::runtime_error("combine(fixed_dt a,b) needs dt to align");
                utctime t0=max(pa.start,pb.start);
                return fixed_dt(t0,a.dt,(min(pa.end,pb.end) - t0)/a.dt);
            }
        }

        /** \brief combine continuous (time-axis,time-axis) template
         * for combining any continuous time-axis with another continuous time-axis
         * \note this could have potentially linear cost of n-points
         */
        template<class TA,class TB>
        inline generic_dt combine(const TA& a,const TB & b, typename enable_if< TA::continuous::value && TB::continuous::value >::type* x=0) {
            utcperiod pa=a.total_period();
            utcperiod pb=b.total_period();
            if( !pa.overlaps(pb) || a.size()==0 || b.size()==0 )
                return generic_dt(point_dt::null_range());
            if(pa==pb && a.size() == b.size()) { //possibly exact equal ?
                bool all_equal=true;
                for(size_t i=0;i<a.size();++i) {
                    if(a.period(i)!=b.period(i)) {
                        all_equal=false;break;
                    }
                }
                if(all_equal)
                    return generic_dt(a);
            }
            // the hard way merge points in the intersection of periods
            utctime t0= max(pa.start,pb.start);
            utctime te= min(pa.end,pb.end);
            size_t ia= a.open_range_index_of(t0);
            size_t ib= b.open_range_index_of(t0);
            size_t ea= 1 + a.open_range_index_of(te);
            size_t eb= 1 + b.open_range_index_of(te);
            point_dt r;
            r.t.reserve((ea-ia)+(eb-ib));//assume worst case here, avoid realloc
            r.t_end = te;// last point set

            while(ia<ea && ib < eb) {
                utctime ta=a.time(ia);
                utctime tb=b.time(ib);

                if(ta==tb) {
                    r.t.push_back(ta);++ia;++ib;// common point,push&incr. both
                } else if(ta<tb) {
                    r.t.push_back(ta);++ia;// a contribution only, incr. a
                } else {
                    r.t.push_back(tb);++ib;// b contribution only, incr. b
                }
            }
            // a or b (or both) are empty for time-points,
            if (ia<ea) { // more to fill in from a ?
                while(ia<ea)
                    r.t.push_back(a.time(ia++));
            } else { // more to fill in from b ?
                while(ib<eb)
                    r.t.push_back(b.time(ib++));
            }

            if(r.t.back()==r.t_end) // make sure we leave t_end as the last point.
                r.t.pop_back();
            return generic_dt(r);
        }

        /**\brief TODO write the combine sparse time-axis algorithm
         *  for calendar_p_dt and period-list.
         * period_list+period_list -> period_list
         *
         */
         template<class TA,class TB>
         inline period_list combine(const TA& a,const TB & b, typename enable_if< !TA::continuous::value || !TB::continuous::value >::type* x=0) {
            utcperiod pa=a.total_period();
            utcperiod pb=b.total_period();
            if( !pa.overlaps(pb) || a.size()==0 || b.size()==0 )
                return period_list::null_range();
            if(pa==pb && a.size() == b.size()) { //possibly exact equal ?
                bool all_equal=true;
                for(size_t i=0;i<a.size();++i) {
                    if(a.period(i)!=b.period(i)) {
                        all_equal=false;break;
                    }
                }
                if(all_equal)
                    return period_list::convert(a);
            }

            // the hard way merge points in the intersection of periods
            utctime t0= max(pa.start,pb.start);
            utctime te= min(pa.end,pb.end);
            //TODO: Really! We need index_of(t0,open_range=true), or we need sparse timeaxis to deliver index of closest internal period!
            size_t ia= a.open_range_index_of(t0);//notice these have to be non-nan!
            size_t ib= b.open_range_index_of(t0);
            if(ia==string::npos || ib==string::npos)
                throw runtime_error("period_list::combine algorithmic error npos not expected here" );
            size_t ea= 1 + a.open_range_index_of(te);
            size_t eb= 1 + b.open_range_index_of(te);
            period_list r;
            r.p.reserve((ea-ia)+(eb-ib));//assume worst case here, avoid realloc

            while(ia<ea && ib < eb) { // while both do have contributions
                utcperiod ta=a.period(ia);
                utcperiod tb=b.period(ib);
                if(!ta.overlaps(tb)){
                    ++ia;++ib;
                    continue;
                }

                if(ta==tb) {
                    r.p.push_back(ta);++ia;++ib;// common period, push&incr. both
                } else if( tb.contains(ta) )  {
                    r.p.push_back(ta);++ia;// a contribution only, incr. a
                } else if( ta.contains(tb) ) {
                    r.p.push_back(tb);++ib;// b contribution only, incr. b
                } else { //ok, not equal, and neither a or b contains the other
                    if( ta.start < tb.start) {
                        r.p.push_back(intersection(ta,tb));++ia;// a contribution only, incr. a
                    } else {
                        r.p.push_back(intersection(ta,tb));++ib;// b contribution only, incr. b
                    }
                }
            }
            // a or b (or both) are empty for periods,
            if (ia<ea) { // more to fill in from a ?
                while(ia<ea)
                    r.p.push_back(a.period(ia++));
            } else { // more to fill in from b ?
                while(ib<eb)
                    r.p.push_back(b.period(ib++));
            }
            return r;
         }


        /** \brief time-axis combine type deduction system
         * The goal here is to deduce the fastest possible representation type of
         * two time-axis to combine.
         */

        template <typename T_A,typename T_B,typename C=void >
        struct combine_type { // generic fallback to period_list type, very general, but expensive
            typedef period_list type;
        };

        template<> // make sure that the fast fixed_dt combination is propagated
        struct combine_type<fixed_dt,fixed_dt,void> {typedef fixed_dt type;};
        template<typename T_A,typename T_B> // then take care of all the continuous type of time-axis, they all goes into generic_dt type
        struct combine_type<T_A,T_B,typename enable_if<T_A::continuous::value && T_B::continuous::value>::type> {typedef generic_dt type;};

     }

    /**\brief point time-series, pts, defined by
     * its
     * templated time-axis, ta
     * the values corresponding periods in time-axis (same size)
     * the point_fx_policy that determine how to compute the
     * f(t) on each interval of the time-axis (linear or stair-case)
     * and
     * value of the i'th interval of the time-series.
     */
    template <class TA>
    struct pts {
        typedef TA ta_t;
        point_fx_policy fx_policy;
        TA ta;
        vector<double> v;
        pts(const TA& ta, double fill_value):ta(ta),v(ta.size(),fill_value) {}
        //TODO: move/cp constructors needed ?
        //TODO should we provide/hide v ?
        // TA ta, ta is expected to provide 'time_axis' functions as needed
        // so we do not re-pack and provide functions like .size(), .index_of etc.
        /**\brief the function value f(t) at time t, fx_policy taken into account */
        double operator()(utctime t) const {
            size_t i = ta.index_of(t);
            if(i == string::npos) return nan;
            if( fx_policy==point_fx_policy::LINEAR && i+1<ta.size() && isfinite(v[i+1])) {
                utctime t1=ta.time(i);
                utctime t2=ta.time(i+1);
                double f= double(t2-t)/double(t2-t1);
                return v[i]*f + (1.0-f)*v[i+1];
            }
            return v[i]; // just keep current value flat to +oo or nan
        }

        /**\brief value of the i'th interval fx_policy taken into account,
         * Hmm. is that policy every useful in this context ?
         */
        double value(size_t i) const  {
            if( fx_policy==point_fx_policy::LINEAR && i+1<ta.size() && isfinite(v[i+1]))
                return 0.5*(v[i] + v[i+1]); // average of the value, linear between points(is that useful ?)
            return v[i];
        }
        /** TODO: Consider having function like
        vector<double> extract_points(TA ta) const {}// returns f(t) according to supplied ta
        vector<double> extract_points() const {} //returns points pr. internal ta
        */
    };


    /**\brief ats, average time-series, represents a ts that for
     * the specified time-axis returns the true-average of the
     * underlying specified TS ts.
     *
     */
    template<class TS,class TA>
    struct ats {
        typedef TA ta_t;
        TA ta;
        TS ts;
        point_fx_policy fx_policy;
        ats(const TS&ts,const TA& ta)
        :ta(ta),ts(ts)
        ,fx_policy(point_fx_policy::STAIR_CASE) {} // because true-average of periods is per def. STAIR_CASE
        // to help average_value method for now!
        point get(size_t i) const {return point(ta.time(i),ts.value(i));}
        size_t size() const { return ta.size();}
        size_t index_of(utctime t) const {return ta.index_of(t);}
        //--
        double value(size_t i) const {
            if(i >= ta.size())
                return nan;
            size_t ix_hint=(i*d_ref(ts).ta.size())/ta.size();// assume almost fixed delta-t.
            //TODO: make specialized pr. time-axis average_value, since average of fixed_dt is trivial compared to other ta.
            return average_value(*this,ta.period(i),ix_hint,d_ref(ts).fx_policy == point_fx_policy::LINEAR);// also note: average of non-nan areas !
        }
        double operator()(utctime t) const {
            size_t i=ta.index_of(t);
            if( i==string::npos)
                return nan;
            return value(i);
        }
    };




    /// Basic math operators:
    ///
    template<class A, class B, class O, class TA>
    struct bin_op {
        typedef TA ta_t;
        O op;
        A lhs;
        B rhs;
        TA ta;
        point_fx_policy fx_policy;

        bin_op(const A&lhs,O op,const B& rhs):op(op),lhs(lhs),rhs(rhs) {
            ta=time_axis::combine(d_ref(lhs).ta,d_ref(rhs).ta);
            fx_policy = result_policy(d_ref(lhs).fx_policy,d_ref(rhs).fx_policy);
        }


        double operator()(utctime t) const {
            if(!ta.total_period().contains(t))
                return nan;
            return op(d_ref(lhs)(t),d_ref(rhs)(t));
        }
        double value(size_t i) const {
            if(i==string::npos || i>=ta.size() )
                return nan;
            if(fx_policy==point_fx_policy::STAIR_CASE)
                return (*this)(ta.time(i));
            utcperiod p=ta.period(i);
            double v0= (*this)(p.start);
            double v1= (*this)(p.end);
            if(isfinite(v1)) return 0.5*(v0 + v1);
            return v0;
        }
    };

    //specialize for double bin_op ts
    template<class B,class O,class TA>
    struct bin_op<double,B,O,TA> {
        typedef TA ta_t;
        double lhs;
        B rhs;
        O op;
        TA ta;
        point_fx_policy fx_policy;

        bin_op(double lhs,O op,const B& rhs):lhs(lhs),rhs(rhs),op(op) {
            ta=d_ref(rhs).ta;
            fx_policy = d_ref(rhs).fx_policy;
        }

        double operator()(utctime t) const {return op(lhs,d_ref(rhs)(t));}
        double value(size_t i) const {return op(lhs,d_ref(rhs).value(i));}
    };

    // specialize for ts bin_op ts
    template<class A,class O,class TA>
    struct bin_op<A,double,O,TA> {
        typedef TA ta_t;
        A lhs;
        double rhs;
        O op;
        TA ta;
        point_fx_policy fx_policy;

        bin_op(const A& lhs,O op,double rhs):lhs(lhs),rhs(rhs),op(op) {
            ta=d_ref(lhs).ta;
            fx_policy = d_ref(lhs).fx_policy;
        }
        double operator()(utctime t) const {return op(d_ref(lhs)(t),rhs);}
        double value(size_t i) const {return op(d_ref(lhs).value(i),rhs);}
    };


    // When doing binary-ts operations, we need to deduce at runtime what will be the
    // most efficient time-axis type of the binary result
    // We use time_axis::combine_type<..> if we have two time-series
    // otherwise we specialize for double binop ts, and ts binop double.
    template<typename L,typename R>
    struct op_axis {
        typedef typename time_axis::combine_type< typename d_ref_t<L>::type::ta_t, typename d_ref_t<R>::type::ta_t>::type type;
    };

    template<typename R>
    struct op_axis<double,R> {
        typedef typename d_ref_t<R>::type::ta_t type;
    };

    template<typename L>
    struct op_axis<L,double> {
        typedef typename d_ref_t<L>::type::ta_t type;
    };

    struct op_max {
        double operator()(const double&a,const double&b) const {return max(a,b);}
    };
    struct op_min {
        double operator()(const double&a,const double&b) const {return min(a,b);}
    };

    template <class A,class B>
    auto operator+ (const A& lhs, const B& rhs) {
        return bin_op<A,B,plus<double>,typename op_axis<A,B>::type> (lhs,plus<double>(),rhs);
    }

    template <class A,class B>
    auto operator- (const A& lhs, const B& rhs) {
        return bin_op<A,B,minus<double>,typename op_axis<A,B>::type> (lhs,minus<double>(),rhs);
    }

    template <class A,class B>
    auto operator* (const A& lhs, const B& rhs) {
        return bin_op<A,B,multiplies<double>,typename op_axis<A,B>::type> (lhs,multiplies<double>(),rhs);
    }

    template <class A,class B>
    auto operator/ (const A& lhs, const B& rhs) {
        return bin_op<A,B,divides<double>,typename op_axis<A,B>::type> (lhs,divides<double>(),rhs);
    }

    template <class A,class B>
    auto max(const A& lhs, const B& rhs) {
        return bin_op<A,B,op_max,typename op_axis<A,B>::type> (lhs,op_max(),rhs);
    }

    template <class A,class B>
    auto min(const A& lhs, const B& rhs) {
        return bin_op<A,B,op_min,typename op_axis<A,B>::type> (lhs,op_min(),rhs);
    }



    }
}

void timeseries_test::test_binary_operator() {
    using namespace shyft::think;
    // regular time-axis time-series
    typedef pts<time_axis::fixed_dt> pts_t;
    /*
    Demo to show how to use ts-math binary operations,
    Working for fixed_dt, point_dt and shared_ptr or by value ts.

    */
    calendar utc;
    utctime start = utc.time(YMDhms(2016,3,8));
    auto dt = deltahours(1);
    int  n = 10;
    time_axis::fixed_dt ta(start,dt,n);
    time_axis::point_dt bta({start,start+dt*3,start+dt*6},start+dt*7);
    time_axis::calendar_dt_p c_dt_p_ta(make_shared<calendar>(),start,dt,n,{utcperiod(deltaminutes(10),deltaminutes(20))});
    pts<time_axis::point_dt> bts(bta,10.0);
    pts<time_axis::calendar_dt_p> c_dt_p_ts(c_dt_p_ta,14.0);
    auto a=make_shared<pts_t>(ta,1.0);
    auto b=make_shared<pts_t>(ta,2.0);

    auto c= a+a*a + 2.0*(b+1.0)/a;
    time_axis::fixed_dt ta2(start,dt*2,n/2);

    auto avg_c = make_shared<ats<decltype(c),decltype(ta2)>>(c,ta2); //true average ts of c
    ats<decltype(c),decltype(ta2)> avg_c2(c,ta2);
    auto d = avg_c*3.0 + avg_c2 + bts+max(a,b)*min(b,c);
    //DO NOT RUN YET:
    auto e = d + c_dt_p_ts;
    TS_ASSERT_EQUALS(n,c.ta.size());
    TS_ASSERT_DELTA(8.0,c.value(0),0.00000001);
    TS_ASSERT_EQUALS(n/2,avg_c->ta.size());
    TS_ASSERT(d.value(0)>0.0?true:false);

}
template <class TA>
static bool test_if_equal(shyft::think::time_axis::fixed_dt& e,const TA& t) {
    using namespace std;
    if(e.size()!=t.size())
        return false;
    if(e.total_period() != t.total_period())
        return false;
    if(e.index_of(e.total_period().end) != t.index_of(e.total_period().end))
        return false;
    if(e.open_range_index_of(e.total_period().end) != t.open_range_index_of(e.total_period().end))
        return false;

    for(size_t i=0;i<e.size();++i) {
        if(e.time(i)!=t.time(i))
            return false;
        if(e.period(i)!= t.period(i))
            return false;
        if(e.index_of(e.time(i)+deltaminutes(30)) != t.index_of(e.time(i)+deltaminutes(30)))
            return false;
        if(e.index_of(e.time(i)-deltaminutes(30)) != t.index_of(e.time(i)-deltaminutes(30)))
            return false;
        if(e.open_range_index_of(e.time(i)+deltaminutes(30)) != t.open_range_index_of(e.time(i)+deltaminutes(30)))
            return false;
        if(e.open_range_index_of(e.time(i)-deltaminutes(30)) != t.open_range_index_of(e.time(i)-deltaminutes(30)))
            return false;
    }
    return true;
}

void timeseries_test::test_time_axis() {
    // Verify that if all types of time-axis are setup up to have the same periods
    // they all have the same properties.
    // test-strategy: Have one fixed time-axis that the other should equal
    using namespace shyft::think;
    auto utc=make_shared<calendar>();
    utctime start = utc->time(YMDhms(2016,3,8));
    auto dt = deltahours(3);
    int  n = 9*3;
    time_axis::fixed_dt expected(start,dt,n); // this is the simplest possible time axis
    //
    // STEP 0: verify that the expected time-axis is correct
    //
    TS_ASSERT_EQUALS(n,expected.size());
    TS_ASSERT_EQUALS(utcperiod(start,start+n*dt),expected.total_period());
    TS_ASSERT_EQUALS(string::npos,expected.index_of(start-1));
    TS_ASSERT_EQUALS(string::npos,expected.open_range_index_of(start-1));
    TS_ASSERT_EQUALS(string::npos,expected.index_of(start+n*dt));
    TS_ASSERT_EQUALS(n-1,expected.open_range_index_of(start+n*dt));
    for(int i=0;i<n;++i) {
        TS_ASSERT_EQUALS(start+i*dt,expected.time(i));
        TS_ASSERT_EQUALS(utcperiod(start+i*dt,start+(i+1)*dt),expected.period(i));
        TS_ASSERT_EQUALS(i,expected.index_of(start+i*dt));
        TS_ASSERT_EQUALS(i,expected.index_of(start+i*dt+dt-1));
        TS_ASSERT_EQUALS(i,expected.open_range_index_of(start+i*dt));
        TS_ASSERT_EQUALS(i,expected.open_range_index_of(start+i*dt+dt-1));
    }
    //
    // STEP 1: construct all the other types of time-axis, with equal content, but represented differently
    //
    time_axis::calendar_dt c_dt(utc,start,dt,n);
    vector<utctime> tp;for(int i=0;i<n;++i)tp.push_back(start + i*dt);
    time_axis::point_dt p_dt(tp,start+n*dt);
    vector<utcperiod> sub_period;
    for(int i=0;i<3;++i) sub_period.emplace_back(i*dt,(i+1)*dt);
    time_axis::calendar_dt_p c_dt_p(utc,start,3*dt,n/3,sub_period);
    vector<utcperiod> periods;
    for(int i=0;i<n;++i) periods.emplace_back(start+i*dt,start+(i+1)*dt);
    time_axis::period_list ta_of_periods(periods);
    //
    // STEP 2: Verify that all the other types are equal to the now verified correct expected time_axis
    //
    TS_ASSERT(test_if_equal(expected,c_dt));
    TS_ASSERT(test_if_equal(expected,p_dt));
    TS_ASSERT(test_if_equal(expected,c_dt_p));
    TS_ASSERT(test_if_equal(expected,ta_of_periods));
    TS_ASSERT(test_if_equal(expected,time_axis::generic_dt(expected)));
    TS_ASSERT(test_if_equal(expected,time_axis::generic_dt(p_dt)));
    TS_ASSERT(test_if_equal(expected,time_axis::generic_dt(c_dt)));
    //
    // STEP 3: Verify the time_axis::combine algorithm when equal time-axis are combined
    //
    TS_ASSERT(test_if_equal(expected,time_axis::combine(expected,expected)));
    TS_ASSERT(test_if_equal(expected,time_axis::combine(c_dt,expected)));
    TS_ASSERT(test_if_equal(expected,time_axis::combine(c_dt,p_dt)));
    TS_ASSERT(test_if_equal(expected,time_axis::combine(c_dt,p_dt)));
    TS_ASSERT(test_if_equal(expected,time_axis::combine(ta_of_periods,p_dt)));
    TS_ASSERT(test_if_equal(expected,time_axis::combine(ta_of_periods,c_dt_p)));

    //
    // STEP 4: Verify the time_axis::combine algorithm for non-overlapping timeaxis(should give null-timeaxis)
    //
    time_axis::fixed_dt f_dt_null=time_axis::fixed_dt::null_range();
    time_axis::point_dt p_dt_x({start+n*dt,start+(n+1)*dt},start+(n+2)*dt);
    TS_ASSERT(test_if_equal(f_dt_null,time_axis::combine(c_dt,p_dt_x)));
    TS_ASSERT(test_if_equal(f_dt_null,time_axis::combine(expected,p_dt_x)));
    TS_ASSERT(test_if_equal(f_dt_null,time_axis::combine(p_dt,p_dt_x)));
    //TODO: when we implement combine of non-continuous time_axis, verify it here

    //
    // STEP 5: Verify the time_axis::combine algorithm for overlapping time-axis
    //
    time_axis::fixed_dt overlap1(start+dt,dt,n);
    time_axis::fixed_dt expected_combine1(start+dt,dt,n-1);
    TS_ASSERT(test_if_equal(expected_combine1,time_axis::combine(expected,overlap1)));
    TS_ASSERT(test_if_equal(expected_combine1,time_axis::combine(c_dt,overlap1)));
    TS_ASSERT(test_if_equal(expected_combine1,time_axis::combine(p_dt,overlap1)));
    TS_ASSERT(test_if_equal(expected_combine1,time_axis::combine(overlap1,time_axis::generic_dt(c_dt))));
    TS_ASSERT(test_if_equal(expected_combine1,time_axis::combine(overlap1,time_axis::generic_dt(p_dt))));
    TS_ASSERT(test_if_equal(expected_combine1,time_axis::combine(overlap1,time_axis::generic_dt(expected))));
    //TODO: when we implement combine of non-continuous time_axis, verify it here
}


