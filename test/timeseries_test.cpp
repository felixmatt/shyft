#include "test_pch.h"
#define _USE_MATH_DEFINES
#include "timeseries_test.h"
#include "mocks.h"
#include "core/timeseries.h"
#include "core/time_axis.h"
#include <armadillo>
#include <cmath>
#include <functional>


namespace shyfttest {
    const double EPS = 1.0e-8;
    using namespace std;
    using namespace shyft::timeseries;
    using namespace shyft::core;

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
template <class TA,class TB>
static bool test_if_equal(const TA& e,const TB& t) {
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
        utctime tx=e.time(i)-deltaminutes(30);
        size_t ei=e.open_range_index_of(tx);
        size_t ti=t.open_range_index_of(tx);
        TS_ASSERT_EQUALS( ei, ti);
        if(ei!=ti)
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
    TS_ASSERT(test_if_equal(f_dt_null,time_axis::combine(ta_of_periods,f_dt_null)));
    TS_ASSERT(test_if_equal(f_dt_null,time_axis::combine(p_dt_x,c_dt_p)));


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
    TS_ASSERT(test_if_equal(expected_combine1,time_axis::combine(overlap1,c_dt_p)));
    TS_ASSERT(test_if_equal(expected_combine1,time_axis::combine(ta_of_periods,overlap1)));

    //
    // STEP 6: Verify the time_axis::combine algorithm for sparse time-axis period_list|calendar_dt_p
    //

    // create sparse time-axis samples
    vector<utcperiod> sparse_sub_period;for(int i=0;i<3;++i)  sparse_sub_period.emplace_back(i*dt+deltahours(1),(i+1)*dt - deltahours(1));
    vector<utcperiod> sparse_period;for(int i=0;i<n;++i) sparse_period.emplace_back(start+i*dt+deltahours(1),start+(i+1)*dt - deltahours(1));

    time_axis::calendar_dt_p sparse_c_dt_p(utc,start,3*dt,n/3,sparse_sub_period);
    time_axis::period_list sparse_period_list(sparse_period);
    TS_ASSERT(test_if_equal(sparse_c_dt_p,sparse_period_list)); // they should be equal
    // now combine a sparse with a dense time-axis, the result should be equal to the sparse (given they cover same period)
    TS_ASSERT(test_if_equal(sparse_c_dt_p,time_axis::combine(expected,sparse_c_dt_p)));// combine to a dense should give the sparse result
    TS_ASSERT(test_if_equal(sparse_c_dt_p,time_axis::combine(sparse_c_dt_p,expected)));// combine to a dense should give the sparse result

    TS_ASSERT(test_if_equal(sparse_c_dt_p,time_axis::combine(expected,sparse_period_list)));// combine to a dense should give the sparse result
    TS_ASSERT(test_if_equal(sparse_c_dt_p,time_axis::combine(sparse_c_dt_p,sparse_period_list)));// combine to a dense should give the sparse result
    TS_ASSERT(test_if_equal(sparse_c_dt_p,time_axis::combine(c_dt_p,sparse_period_list)));// combine to a dense should give the sparse result
    //final tests: verify that if we combine two sparse time-axis, we get the distinct set of the periods.
    {
        time_axis::period_list ta1({utcperiod(1,3),utcperiod(5,7),utcperiod(9,11)});
        time_axis::period_list ta2({utcperiod(0,2),utcperiod(4,10),utcperiod(11,12)});
        time_axis::period_list exp({utcperiod(1,2),utcperiod(5,7),utcperiod(9,10)});
        TS_ASSERT(test_if_equal(exp,time_axis::combine(ta1,ta2)));

    }

}


