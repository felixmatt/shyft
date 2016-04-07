#include "test_pch.h"
#define _USE_MATH_DEFINES
#include "timeseries_test.h"
#include "mocks.h"
#include "core/timeseries.h"
#include "core/time_axis.h"
#include "api/api.h"
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

typedef point_ts<point_timeaxis> xts_t;

void timeseries_test::test_point_timeaxis() {
    point_timeaxis ts0; //zero points
    TS_ASSERT_EQUALS(ts0.size(),0);
    TS_ASSERT_EQUALS(ts0.index_of(12),std::string::npos);
    vector<utctime> t2={3600*1};//just one point
    try {
    point_timeaxis ts1(t2);
    TS_ASSERT(false);
    //TS_ASSERT_EQUALS(ts0.size(),0);
    //TS_ASSERT_EQUALS(ts0.index_of(12),std::string::npos);
    } catch (const exception & ex) {

    }
    vector<utctime> t={3600*1,3600*2,3600*3};
    point_timeaxis tx(t);
    TS_ASSERT_EQUALS(tx.size(),2);// number of periods, - two .. (unless we redefined the last to be last point .. +oo)
    TS_ASSERT_EQUALS(tx.period(0),utcperiod(t[0],t[1]));
    TS_ASSERT_EQUALS(tx.period(1),utcperiod(t[1],t[2]));
    TS_ASSERT_EQUALS(tx.index_of(-3600),std::string::npos);//(1),utcperiod(t[1],t[2]);
    TS_ASSERT_EQUALS(tx.index_of(t[0]),0);
    TS_ASSERT_EQUALS(tx.index_of(t[1]-1),0);
    TS_ASSERT_EQUALS(tx.index_of(t[2]+1),std::string::npos);
    TS_ASSERT_EQUALS(tx.open_range_index_of(t[2]+1),1);


}
void timeseries_test::test_timeaxis() {
    auto t0=calendar().time(YMDhms(2000,1,1,0,0,0));
    auto dt=deltahours(1);
    size_t n=3;
    timeaxis tx(t0,dt,n);
    TS_ASSERT_EQUALS(tx.size(),n);
    for(size_t i=0;i<n;++i) {
        TS_ASSERT_EQUALS(tx.period(i), utcperiod(t0+i*dt,t0+(i+1)*dt));
        TS_ASSERT_EQUALS(tx.time(i), t0+i*dt );
        TS_ASSERT_EQUALS(tx.index_of(tx.time(i)),i);
        TS_ASSERT_EQUALS(tx.index_of(tx.time(i)+dt/2),i);
    }
    TS_ASSERT_EQUALS(tx.open_range_index_of(t0+(n+1)*dt),n-1);
    TS_ASSERT_EQUALS(tx.index_of(t0+(n+1)*dt),string::npos);
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

    point_ts<timeaxis> a(fixed_ta,values);
    point_ts<point_timeaxis> b(point_ta,values);

    TS_ASSERT_EQUALS(a.ta.total_period(),b.ta.total_period());
    TS_ASSERT_EQUALS(a.size(),b.size());
    TS_ASSERT_EQUALS(a.ta.size(),b.ta.size());
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
        auto ti=fixed_ta.time(i) + d/2;
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
    point_ts<timeaxis> a(timeaxis(t,d,n),values);
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
		q_value = average_value_staircase_fast(source, time_axis.period(q_idx = i), last_idx);
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
    point_ts<point_timeaxis> ps(point_timeaxis(times),points);
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
    point_timeaxis time_axis(ta_times);

}

void timeseries_test::test_ts_weighted_average(void) {
    using pts_t=point_ts<timeaxis>;
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
namespace shyft {
    namespace api {
            /**
                time-series math to be exposed to python

                This provide functionality like

                a = TsFactory.create_ts(..)
                b = TsFactory.create_ts(..)
                c = a + 3*b
                d = max(c,0.0)

                implementation strategy

                provide a type apoint_ts, that always appears as a time-series.

                It could however represent either a
                  point_ts<time_axis:generic_dt>
                or a
                 abin_op_ts( lhs,op,rhs) , i.e. an expression

                Then we provide operators:
                apoint_ts bin_op a_point_ts
                and
                double bin_op a_points_ts


             */
            typedef shyft::time_axis::generic_dt gta_t;
            typedef shyft::timeseries::point_ts<gta_t> gts_t;

            /** virtual abstract base for point_ts */
            struct  ipoint_ts {
                virtual ~ipoint_ts(){}
                virtual point_interpretation_policy point_interpretation() const =0;
                virtual void set_point_interpretation(point_interpretation_policy point_interpretation) =0;
                virtual const gta_t& time_axis() const =0;
                virtual utcperiod total_period() const=0;   ///< Returns period that covers points, given
                virtual size_t index_of(utctime t) const=0; ///< we might want index_of(t,ix-hint..)
                virtual size_t size() const=0;        ///< number of points that descr. y=f(t) on t ::= period
                virtual utctime time(size_t i) const=0;///< get the i'th time point
                virtual double value(size_t i) const=0;///< get the i'th value
                virtual double value_at(utctime t) const =0;
                virtual std::vector<double> values() const =0;// corresponding to time_axis()
                // to be removed:
                point get(size_t i) const {return point(time(i),value(i));}

                // will not work, operator is for each class :double operator()(utctime t) const =0;
                // we might want:
                // vector<double> values() const; ///< return the values for the complete ts
                //
            };

            /** apoint_ts, a value-type concrete ts that we build all the semantics on */
            class apoint_ts {
                /** a ref to the real implementation, could be a concrete point ts, or an expression */
                std::shared_ptr<ipoint_ts> ts;// consider unique pointer instead,possibly public, to ease transparency in python
                friend struct average_ts;
               public:
                // constructors that we want to expose
                // like

                apoint_ts(const gta_t& ta,double fill_value);
                apoint_ts(const gta_t& ta,const std::vector<double>& values);
                apoint_ts(const gta_t& ta,std::vector<double>&& values);
                apoint_ts(gta_t&& ta,std::vector<double>&& values);
                apoint_ts(gta_t&& ta,double fill_value);
                apoint_ts(const std::shared_ptr<ipoint_ts>& c):ts(c) {}
                // some more exotic stuff like average_ts


                // std ct/= stuff, that might be ommitted if c++ do the right thing.
                apoint_ts(){}
                apoint_ts(const apoint_ts&c):ts(c.ts){}
                apoint_ts(apoint_ts&& c):ts(std::move(c.ts)){}
                apoint_ts& operator=(const apoint_ts& c) {
                    if( this != &c )
                        ts=c.ts;
                    return *this;
                }
                apoint_ts& operator=(apoint_ts&& c) {
                    ts=std::move(c.ts);
                    return *this;
                }


                // interface we want to expose
                point_interpretation_policy point_interpretation() const {return ts->point_interpretation();}
                void set_point_interpretation(point_interpretation_policy point_interpretation) { ts->set_point_interpretation(point_interpretation); };
                const gta_t& time_axis() const { return ts->time_axis();};
                utcperiod total_period() const {return ts->total_period();};   ///< Returns period that covers points, given
                size_t index_of(utctime t) const {return ts->index_of(t);};
                size_t size() const {return ts->size();};        ///< number of points that descr. y=f(t) on t ::= period
                utctime time(size_t i) const {return ts->time(i);};///< get the i'th time point
                double value(size_t i) const {return ts->value(i);};///< get the i'th value
                double operator()(utctime t) const  {return ts->value_at(t);};
                std::vector<double> values() const {return ts->values();}
                apoint_ts average(const gta_t &ta) const;
                //-- if apoint_ts is everything we want to expose,
                // these are needed in the case of gpoint_ts (exception if not..)
                void set(size_t i, double x) ;
                void fill(double x) ;
                void scale_by(double x) ;


            };



            /** gpoint_ts a generic concrete point_ts, with a generic time-axis */
            struct gpoint_ts:ipoint_ts {
                gts_t rep;
                // To create gpoint_ts, we use const ref, move ct wherever possible:
                // note (we would normally use ct template here, but we are aiming at exposing to python)
                gpoint_ts(const gta_t&ta,double fill_value):rep(ta,fill_value){}
                gpoint_ts(const gta_t&ta,const std::vector<double>& v):rep(ta,v) {}
                gpoint_ts(gta_t&&ta,double fill_value):rep(std::move(ta),fill_value){}
                gpoint_ts(gta_t&&ta,std::vector<double>&& v):rep(std::move(ta),std::move(v)) {}
                gpoint_ts(const gta_t& ta,std::vector<double>&& v):rep(ta,std::move(v)) {}

                // now for the gpoint_ts it self, constructors incl. move
                gpoint_ts(const gpoint_ts& c):rep(c.rep){}
                gpoint_ts(gts_t&& c):rep(std::move(c)){}
                gpoint_ts& operator=(const gpoint_ts&c) {
                    if(this != &c)
                        rep=c.rep;
                    return *this;
                }
                gpoint_ts& operator=(gpoint_ts&& c) {
                    rep=std::move(c.rep);
                    return *this;
                }

                // implement ipoint_ts contract:
                virtual point_interpretation_policy point_interpretation() const {return rep.point_interpretation();}
                virtual void set_point_interpretation(point_interpretation_policy point_interpretation) {rep.set_point_interpretation(point_interpretation);}
                virtual const gta_t& time_axis() const {return rep.time_axis();}
                virtual utcperiod total_period() const {return rep.total_period();}
                virtual size_t index_of(utctime t) const {return rep.index_of(t);}
                virtual size_t size() const {return rep.size();}
                virtual utctime time(size_t i) const {return rep.time(i);};
                virtual double value(size_t i) const {return rep.value(i);}
                virtual double value_at(utctime t) const {return rep(t);}
                virtual std::vector<double> values() const {return rep.v;}
                // implement some extra functions to manipulate the points
                void set(size_t i, double x) {rep.set(i,x);}
                void fill(double x) {rep.fill(x);}
                void scale_by(double x) {rep.scale_by(x);}
            };

            struct average_ts:ipoint_ts {
                gta_t ta;
                std::shared_ptr<ipoint_ts> ts;
                // useful constructors
                average_ts(gta_t&& ta,const apoint_ts& ats):ta(std::move(ta)),ts(ats.ts) {}
                average_ts(gta_t&& ta,apoint_ts&& ats):ta(std::move(ta)),ts(std::move(ats.ts)) {}
                average_ts(const gta_t& ta,apoint_ts&& ats):ta(ta),ts(std::move(ats.ts)) {}
                average_ts(const gta_t& ta,const apoint_ts& ats):ta(ta),ts(ats.ts) {}
                average_ts(const gta_t& ta,const std::shared_ptr<ipoint_ts> &ts ):ta(ta),ts(ts){}
                average_ts(gta_t&& ta,const std::shared_ptr<ipoint_ts> &ts ):ta(std::move(ta)),ts(ts){}
                // std copy ct and assign
                average_ts(const average_ts &c):ta(c.ta),ts(c.ts) {}
                average_ts(average_ts&&c):ta(std::move(ta)),ts(std::move(c.ts)) {}
                average_ts& operator=(const average_ts&c) {
                    if( this != &c) {
                        ta=c.ta;
                        ts=c.ts;
                    }
                    return *this;
                }
                average_ts& operator=(average_ts&& c) {
                    ta=std::move(c.ta);
                    ts=std::move(c.ts);
                    return *this;
                }
                                // implement ipoint_ts contract:
                virtual point_interpretation_policy point_interpretation() const {return point_interpretation_policy::POINT_AVERAGE_VALUE;}
                virtual void set_point_interpretation(point_interpretation_policy point_interpretation) {;}
                virtual const gta_t& time_axis() const {return ta;}
                virtual utcperiod total_period() const {return ta.total_period();}
                virtual size_t index_of(utctime t) const {return ta.index_of(t);}
                virtual size_t size() const {return ta.size();}
                virtual utctime time(size_t i) const {return ta.time(i);};
                virtual double value(size_t i) const {
                    if(i>ta.size())
                        return nan;
                    size_t ix_hint=(i*ts->size())/ta.size();// assume almost fixed delta-t.
                    return average_value(*ts,ta.period(i),ix_hint,ts->point_interpretation() == point_interpretation_policy::POINT_INSTANT_VALUE);
                }
                virtual double value_at(utctime t) const {
                    // return true average at t
                    if(!ta.total_period().contains(t))
                        return nan;
                    return value(index_of(t));
                }
                virtual std::vector<double> values() const {
                    std::vector<double> r;r.reserve(ta.size());
                    size_t ix_hint=ts->index_of(ta.time(0));
                    bool linear_interpretation=ts->point_interpretation() == point_interpretation_policy::POINT_INSTANT_VALUE;
                    for(size_t i=0;i<ta.size();++i) {
                        r.push_back(average_value(*ts,ta.period(i),ix_hint,linear_interpretation));
                    }
                    return std::move(r);//needed ?
                }
                // to help the average function, return the i'th point of the underlying timeseries
                //point get(size_t i) const {return point(ts->time(i),ts->value(i));}

            };

            // alternatives for operator types
            // a) overload op(a,b) in bin_op --> gives bin_op_add,bin_op_sub etc... quite flexible to add more data needed during op
            // b) make op a class, that we *refer* in bin_op (at the cost of referring bin-ops), hmm. involves heap, even if we use a unique_ptr
            // c) just use pure function pointer, ok, but somewhat limits what the op() is and can do(no heap involvement
            // d) combination of a) and c) ?
            // e) let binop keep a ref. to op, and use static for +- etc. and then for the others? hmm.. no solution
            // current approach c), since it's the simplest possible solution that solves the current use.

            typedef double (*iop_t)(double a,double b);

            static double iop_add(double a,double b) {return a + b;}
            static double iop_sub(double a,double b) {return a - b;}
            static double iop_div(double a,double b) {return a/b;}
            static double iop_mul(double a,double b) {return a*b;}
            static double iop_min(double a,double b) {return std::min(a,b);}
            static double iop_max(double a,double b) {return std::max(a,b);}

            /** binary operation for type ts op ts */
            struct abin_op_ts:ipoint_ts {

                  apoint_ts lhs;
                  iop_t op;
                  apoint_ts rhs;
                  gta_t ta;
                  point_interpretation_policy fx_policy;
                  point_interpretation_policy point_interpretation() const {return fx_policy;}
                  void set_point_interpretation(point_interpretation_policy x) {fx_policy=x;}

                  abin_op_ts(const apoint_ts &lhs,iop_t op,const apoint_ts& rhs)
                  :lhs(lhs),op(op),rhs(rhs) {
                      ta=time_axis::combine(lhs.time_axis(),rhs.time_axis());
                      fx_policy= result_policy(lhs.point_interpretation(),rhs.point_interpretation());
                  }
                  abin_op_ts(const abin_op_ts& c)
                    :lhs(c.lhs),op(c.op),rhs(c.rhs),ta(c.ta),fx_policy(c.fx_policy) {
                  }
                  abin_op_ts(abin_op_ts&& c)
                    :lhs(std::move(c.lhs)),op(c.op),rhs(std::move(c.rhs)),ta(std::move(c.ta)),fx_policy(c.fx_policy)
                     {
                  }

                  abin_op_ts& operator=(const abin_op_ts& c) {
                    if( this != & c) {
                        lhs = c.lhs;
                        op = c.op;
                        rhs = c.rhs;
                        ta = c.ta;
                        fx_policy = c.fx_policy;
                    }
                    return *this;
                  }

                  abin_op_ts& operator=(abin_op_ts&& c) {
                    if( this != & c) {
                        lhs = std::move(c.lhs);
                        op = c.op;
                        rhs = std::move(c.rhs);
                        ta  = std::move(c.ta);
                        fx_policy = c.fx_policy;
                    }
                    return *this;
                  }

                  virtual utcperiod total_period() const {return ta.total_period();}
                  const gta_t& time_axis() const {return ta;};// combine lhs,rhs
                  size_t index_of(utctime t) const{return ta.index_of(t);};
                  size_t size() const {return ta.size();};// use the combined ta.size();
                  utctime time( size_t i) const {return ta.time(i);}; // reeturn combined ta.time(i)
                  double value_at(utctime t) const ;
                  double value(size_t i) const;// return op( lhs(t), rhs(t)) ..
                  std::vector<double> values() const;

            };

            /** binary operation for type double op ts */
            struct abin_op_scalar_ts:ipoint_ts {
                  double lhs;
                  iop_t op;
                  apoint_ts rhs;
                  gta_t ta;
                  point_interpretation_policy fx_policy;
                  point_interpretation_policy point_interpretation() const {return fx_policy;}
                  void set_point_interpretation(point_interpretation_policy x) {fx_policy=x;}

                  abin_op_scalar_ts(double lhs,iop_t op,const apoint_ts& rhs)
                  :lhs(lhs),op(op),rhs(rhs) {
                      ta=rhs.time_axis();
                      fx_policy= rhs.point_interpretation();
                  }
                  abin_op_scalar_ts(const abin_op_scalar_ts& c)
                    :lhs(c.lhs),op(c.op),rhs(c.rhs),ta(c.ta),fx_policy(c.fx_policy) {
                  }
                  abin_op_scalar_ts(abin_op_scalar_ts&& c)
                    :lhs(c.lhs),op(c.op),rhs(std::move(c.rhs)),ta(std::move(c.ta)),fx_policy(c.fx_policy)
                     {
                  }

                  abin_op_scalar_ts& operator=(const abin_op_scalar_ts& c) {
                    if( this != & c) {
                        lhs = c.lhs;
                        op = c.op;
                        rhs = c.rhs;
                        ta = c.ta;
                        fx_policy = c.fx_policy;
                    }
                    return *this;
                  }

                  abin_op_scalar_ts& operator=(abin_op_scalar_ts&& c) {
                    if( this != & c) {
                        lhs = c.lhs;
                        op = c.op;
                        rhs = std::move(c.rhs);
                        ta  = std::move(c.ta);
                        fx_policy = c.fx_policy;
                    }
                    return *this;
                  }

                  virtual utcperiod total_period() const {return ta.total_period();}
                  const gta_t& time_axis() const {return ta;};// combine lhs,rhs
                  size_t index_of(utctime t) const{return ta.index_of(t);};
                  size_t size() const {return ta.size();};
                  utctime time( size_t i) const {return ta.time(i);};
                  double value_at(utctime t) const {return op(lhs,rhs(t));}
                  double value(size_t i) const {return op(lhs,rhs.value(i));}
                  std::vector<double> values() const {
                      std::vector<double> r(rhs.values());
                      for(auto& v:r)
                        v=op(lhs,v);
                      return r;
                  }

            };

            /** binary operation for type ts op double */
            struct abin_op_ts_scalar:ipoint_ts {
                  apoint_ts lhs;
                  iop_t op;
                  double rhs;
                  gta_t ta;
                  point_interpretation_policy fx_policy;
                  point_interpretation_policy point_interpretation() const {return fx_policy;}
                  void set_point_interpretation(point_interpretation_policy x) {fx_policy=x;}

                  abin_op_ts_scalar(const apoint_ts &lhs,iop_t op,double rhs)
                  :lhs(lhs),op(op),rhs(rhs) {
                      ta=lhs.time_axis();
                      fx_policy= lhs.point_interpretation();
                  }
                  abin_op_ts_scalar(const abin_op_ts_scalar& c)
                    :lhs(c.lhs),op(c.op),rhs(c.rhs),ta(c.ta),fx_policy(c.fx_policy) {
                  }
                  abin_op_ts_scalar(abin_op_ts_scalar&& c)
                    :lhs(std::move(c.lhs)),op(c.op),rhs(c.rhs),ta(std::move(c.ta)),fx_policy(c.fx_policy)
                     {
                  }

                  abin_op_ts_scalar& operator=(const abin_op_ts_scalar& c) {
                    if( this != & c) {
                        lhs = c.lhs;
                        op = c.op;
                        rhs = c.rhs;
                        ta = c.ta;
                        fx_policy = c.fx_policy;
                    }
                    return *this;
                  }

                  abin_op_ts_scalar& operator=(abin_op_ts_scalar&& c) {
                    if( this != & c) {
                        lhs = std::move(c.lhs);
                        op = c.op;
                        rhs = c.rhs;
                        ta  = std::move(c.ta);
                        fx_policy = c.fx_policy;
                    }
                    return *this;
                  }

                  virtual utcperiod total_period() const {return ta.total_period();}
                  const gta_t& time_axis() const {return ta;};
                  size_t index_of(utctime t) const{return ta.index_of(t);};
                  size_t size() const {return ta.size();};
                  utctime time( size_t i) const {return ta.time(i);};
                  double value_at(utctime t) const {return op(lhs(t),rhs);}
                  double value(size_t i) const {return op(lhs.value(i),rhs);}
                  std::vector<double> values() const {
                      std::vector<double> r(lhs.values());
                      for(auto& v:r)
                        v=op(rhs,v);
                      return r;
                  }

            };

            // add operators and functions to the apoint_ts class, of all variants that we want to expose
            apoint_ts average(const apoint_ts& ts,const gta_t& ta/*fx-type */)  { return apoint_ts(std::make_shared<average_ts>(ta,ts));}
            apoint_ts average(apoint_ts&& ts,const gta_t& ta)  { return apoint_ts(std::make_shared<average_ts>(ta,std::move(ts)));}


            apoint_ts operator+(const apoint_ts& lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_ts       >( lhs,iop_add,rhs )); }
            apoint_ts operator+(const apoint_ts& lhs,double           rhs) {return apoint_ts(std::make_shared<abin_op_ts_scalar>( lhs,iop_add,rhs )); }
            apoint_ts operator+(double           lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_scalar_ts>( lhs,iop_add,rhs )); }

            apoint_ts operator-(const apoint_ts& lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_ts       >( lhs,iop_sub,rhs )); }
            apoint_ts operator-(const apoint_ts& lhs,double           rhs) {return apoint_ts(std::make_shared<abin_op_ts_scalar>( lhs,iop_sub,rhs )); }
            apoint_ts operator-(double           lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_scalar_ts>( lhs,iop_sub,rhs )); }
            apoint_ts operator-(const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_scalar_ts>( -1.0,iop_mul,rhs )); }

            apoint_ts operator/(const apoint_ts& lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_ts       >( lhs,iop_div,rhs )); }
            apoint_ts operator/(const apoint_ts& lhs,double           rhs) {return apoint_ts(std::make_shared<abin_op_ts_scalar>( lhs,iop_div,rhs )); }
            apoint_ts operator/(double           lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_scalar_ts>( lhs,iop_div,rhs )); }

            apoint_ts operator*(const apoint_ts& lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_ts       >( lhs,iop_mul,rhs )); }
            apoint_ts operator*(const apoint_ts& lhs,double           rhs) {return apoint_ts(std::make_shared<abin_op_ts_scalar>( lhs,iop_mul,rhs )); }
            apoint_ts operator*(double           lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_scalar_ts>( lhs,iop_mul,rhs )); }


            apoint_ts max(const apoint_ts& lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_ts       >( lhs,iop_max,rhs ));}
            apoint_ts max(const apoint_ts& lhs,double           rhs) {return apoint_ts(std::make_shared<abin_op_ts_scalar>( lhs,iop_max,rhs ));}
            apoint_ts max(double           lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_scalar_ts>( lhs,iop_max,rhs ));}

            apoint_ts min(const apoint_ts& lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_ts>( lhs,iop_min,rhs ));}
            apoint_ts min(const apoint_ts& lhs,double           rhs) {return apoint_ts(std::make_shared<abin_op_ts_scalar>( lhs,iop_min,rhs ));}
            apoint_ts min(double           lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_scalar_ts>( lhs,iop_min,rhs ));}

            double abin_op_ts::value(size_t i) const {
                if(i==std::string::npos || i>=ta.size() )
                    return nan;
                if(fx_policy==point_interpretation_policy::POINT_AVERAGE_VALUE)
                    return value_at(ta.time(i));
                utcperiod p=ta.period(i);
                double v0= value_at(p.start);
                double v1= value_at(p.end);
                if(isfinite(v1)) return 0.5*(v0 + v1);
                return v0;
            }

            double abin_op_ts::value_at(utctime t) const {
                if(!ta.total_period().contains(t))
                    return nan;
                return op( lhs(t),rhs(t) );// this might cost a 2xbin-search if not the underlying ts have smart incremental search (at the cost of thread safety)
            }

            std::vector<double> abin_op_ts::values() const {
                std::vector<double> r;r.reserve(ta.size());
                for(size_t i=0;i<ta.size();++i) {
                    r.push_back(value(i));//TODO: improve speed using accessors with ix-hint for lhs/rhs stepwise traversal
                }
                return std::move(r);
            }

            // implement popular ct for apoint_ts to make it easy to expose & use
            apoint_ts::apoint_ts(const time_axis::generic_dt& ta,double fill_value)
                :ts(std::make_shared<gpoint_ts>(ta,fill_value)) {
            }
            apoint_ts::apoint_ts(const time_axis::generic_dt& ta,const std::vector<double>& values)
                :ts(std::make_shared<gpoint_ts>(ta,values)) {
            }
            apoint_ts::apoint_ts(const time_axis::generic_dt& ta,std::vector<double>&& values)
                :ts(std::make_shared<gpoint_ts>(ta,std::move(values))) {
            }
            apoint_ts::apoint_ts(time_axis::generic_dt&& ta,std::vector<double>&& values)
                :ts(std::make_shared<gpoint_ts>(std::move(ta),std::move(values))) {
            }
            apoint_ts::apoint_ts(time_axis::generic_dt&& ta,double fill_value)
                :ts(std::make_shared<gpoint_ts>(std::move(ta),fill_value))
                {
            }
            apoint_ts apoint_ts::average(const gta_t &ta) const {
                return shyft::api::average(*this,ta);
            }
            void apoint_ts::set(size_t i, double x) {
                gpoint_ts *gpts=dynamic_cast<gpoint_ts*>(ts.get());
                if(!gpts)
                    throw std::runtime_error("apoint_ts::set(i,x) only allowed for ts of non-expression types");
                gpts->set(i,x);
            }
            void apoint_ts::fill(double x) {
                gpoint_ts *gpts=dynamic_cast<gpoint_ts*>(ts.get());
                if(!gpts)
                    throw std::runtime_error("apoint_ts::fill(x) only allowed for ts of non-expression types");
                gpts->fill(x);
            }
            void apoint_ts::scale_by(double x) {
                gpoint_ts *gpts=dynamic_cast<gpoint_ts*>(ts.get());
                if(!gpts)
                    throw std::runtime_error("apoint_ts::scale_by(x) only allowed for ts of non-expression types");
                gpts->scale_by(x);
            }

    }
}

template <class A,class B>
static bool is_equal_ts(const A& a,const B& b) {
    const auto &a_ta=a.time_axis();
    const auto &b_ta=b.time_axis();
    if(a_ta.size()!=b_ta.size())
        return false;
    for(size_t i=0;i<a_ta.size();++i) {
        if(a_ta.period(i)!= b_ta.period(i))
            return false;
        if (fabs(a.value(i)-b.value(i))>1e-6)
            return false;
    }
    return true;
}

template < class TS_E,class TS_A,class TS_B,class TA>
static void test_bin_op(const TS_A& a, const TS_B &b, const TA ta,double a_value,double b_value) {
    // Excpected results are time-series with ta and a constant value equal to the standard operators
    TS_E a_plus_b(ta,a_value+b_value);
    TS_E a_minus_b(ta,a_value-b_value);
    TS_E a_mult_b(ta,a_value*b_value);
    TS_E a_div_b(ta,a_value/b_value);
    TS_E max_a_b(ta,std::max(a_value,b_value));
    TS_E min_a_b(ta,std::min(a_value,b_value));

    // Step 1:   ts bin_op ts
    TS_ASSERT(is_equal_ts(a_plus_b,a+b));
    TS_ASSERT(is_equal_ts(a_minus_b,a-b));
    TS_ASSERT(is_equal_ts(a_mult_b,a*b));
    TS_ASSERT(is_equal_ts(a_div_b,a/b));
    TS_ASSERT(is_equal_ts(max_a_b,max(a,b)));
    TS_ASSERT(is_equal_ts(min_a_b,min(a,b)));
    // Step 2:  ts bin_op double
    TS_ASSERT(is_equal_ts(a_plus_b,a+b_value));
    TS_ASSERT(is_equal_ts(a_minus_b,a-b_value));
    TS_ASSERT(is_equal_ts(a_mult_b,a*b_value));
    TS_ASSERT(is_equal_ts(a_div_b,a/b_value));
    TS_ASSERT(is_equal_ts(max_a_b,max(a,b_value)));
    TS_ASSERT(is_equal_ts(min_a_b,min(a,b_value)));
    // Step 3: double bin_op ts
    TS_ASSERT(is_equal_ts(a_plus_b,a_value+b));
    TS_ASSERT(is_equal_ts(a_minus_b,a_value-b));
    TS_ASSERT(is_equal_ts(a_mult_b,a_value*b));
    TS_ASSERT(is_equal_ts(a_div_b,a_value/b));
    TS_ASSERT(is_equal_ts(max_a_b,max(a_value,b)));
    TS_ASSERT(is_equal_ts(min_a_b,min(a_value,b)));

}


void timeseries_test::test_binary_operator() {
    /** Test strategy here is to ensure that
       a) it compiles
       b) it gives the expected results for all combinations
       The test_bin_op template function does the hard work,
       this method only provides suitable parameters and types
       to the test_bin_op function

    */
    using namespace shyft::timeseries;
    using namespace shyft;
    calendar utc;
    utctime start = utc.time(YMDhms(2016,3,8));
    utctimespan dt = deltahours(1);
    size_t  n = 4;
    double a_value=3.0;
    double b_value=2.0;
    {
        typedef time_axis::fixed_dt ta_t;
        typedef point_ts<ta_t> ts_t;
        ta_t ta(start,dt,n);
        auto  a = make_shared<ts_t>(ta,a_value);
        ts_t b(ta,b_value); // test by-value as well as shared_ptr<TS>
        test_bin_op<ts_t>(a,b,ta,a_value,b_value);
    }

    {
        typedef time_axis::point_dt ta_t;
        typedef point_ts<ta_t> ts_t;
        ta_t ta({start,start+dt,start+2*dt,start+3*dt},start+4*dt);
        auto  a = make_shared<ts_t>(ta,a_value);
        ts_t b(ta,b_value); // test by-value as well as shared_ptr<TS>
        test_bin_op<ts_t>(a,b,ta,a_value,b_value);
    }

    {
        typedef time_axis::calendar_dt ta_t;
        typedef point_ts<ta_t> ts_t;
        ta_t ta(make_shared<calendar>(),start,dt,n);
        auto  a = make_shared<ts_t>(ta,a_value);
        ts_t b(ta,b_value); // test by-value as well as shared_ptr<TS>
        test_bin_op<ts_t>(a,b,ta,a_value,b_value);
    }

    {
        typedef time_axis::calendar_dt_p ta_t;
        typedef point_ts<ta_t> ts_t;
        vector<utcperiod> sub_periods({utcperiod(deltaminutes(10),deltaminutes(20))});
        ta_t ta(make_shared<calendar>(),start,dt,n,sub_periods);
        auto  a = make_shared<ts_t>(ta,a_value);
        ts_t b(ta,b_value); // test by-value as well as shared_ptr<TS>
        test_bin_op<ts_t>(a,b,ta,a_value,b_value);
    }
    {
        typedef time_axis::period_list ta_t;
        typedef point_ts<ta_t> ts_t;
        vector<utcperiod> periods;for(size_t i=0;i<n;++i) periods.emplace_back(start+i*dt,start+(i+1)*dt);
        ta_t ta(periods);
        auto  a = make_shared<ts_t>(ta,a_value);
        ts_t b(ta,b_value); // test by-value as well as shared_ptr<TS>
        test_bin_op<ts_t>(a,b,ta,a_value,b_value);
    }

}
void timeseries_test::test_api_ts() {
    using namespace shyft::api;
    calendar utc;
    utctime start = utc.time(YMDhms(2016,3,8));
    utctimespan dt = deltahours(1);
    size_t  n = 4;
    double a_value=3.0;
    double b_value=2.0;

    gta_t ta(shyft::time_axis::fixed_dt(start,dt,n));
    apoint_ts a(ta,a_value);
    apoint_ts b(ta,b_value);
    test_bin_op<apoint_ts>(a,b,ta,a_value,b_value);

    gta_t ta2(shyft::time_axis::fixed_dt(start,dt*2,n/2));
    auto c= average( a+3*b,ta2) * 4 + (b*a/2.0).average(ta2);
    auto cv= c.values();
    TS_ASSERT_EQUALS(cv.size(),ta2.size());
    for(const auto&v:cv) {
        TS_ASSERT_DELTA(v,(a_value+3*b_value)*4 +(a_value*b_value/2.0), 0.001);
    }
    // verify some special fx
    a.set(0,a_value*b_value);
    TS_ASSERT_DELTA(a.value(0),a_value*b_value,0.0001);
    a.fill(b_value);
    for(auto v:a.values())
        TS_ASSERT_DELTA(v,b_value,0.00001);


}
namespace shyft {
   namespace timeseries {

   }
}

typedef shyft::time_axis::fixed_dt tta_t;
typedef shyft::timeseries::point_ts<tta_t> tts_t;

template<typename Fx>
std::vector<tts_t> create_test_ts(size_t n,tta_t ta, Fx&& f) {
    std::vector<tts_t> r;r.reserve(n);
    for (size_t i = 0;i < n;++i) {
        std::vector<double> v;v.reserve(ta.size());
        for (size_t j = 0;j < ta.size();++j)
            v.push_back(f(i, ta.time(j)));
        r.push_back(tts_t(ta, v));
    }
    return std::move(r);
}

/** just verify that it calculate the right things */
void timeseries_test::test_ts_statistics_calculations() {
    calendar utc;
    auto t0 = utc.time(2015, 1, 1);
    auto fx_1 = [t0](size_t i, utctime t)->double {return double( i );};// should generate 0..9 constant ts.
    auto n_days=1;
    auto n_ts=10;
    tta_t  ta(t0, calendar::HOUR, n_days*24);
    tta_t tad(t0, calendar::DAY, n_days);
    auto tsv1 = create_test_ts(n_ts, ta, fx_1);
    auto r1 = calculate_percentiles(tad, tsv1, {0,10,50,-1,70,100});
    TS_ASSERT_EQUALS(size_t(6), r1.size());//, "expect ts equal to percentiles wanted");
    // using excel percentile.inc(..) as correct answer, values 0..9 incl
    TS_ASSERT_DELTA(r1[0].value(0), 0.0,0.0001);// " 0-percentile");
    TS_ASSERT_DELTA(r1[1].value(0), 0.9,0.0001);// "10-percentile");
    TS_ASSERT_DELTA(r1[2].value(0), 4.5,0.0001);// "50-percentile");
    TS_ASSERT_DELTA(r1[3].value(0), 4.5,0.0001);// "avg");
    TS_ASSERT_DELTA(r1[4].value(0), 6.3,0.0001);// "70-percentile");
    TS_ASSERT_DELTA(r1[5].value(0), 9.0,0.0001);// "100-percentile");
    //cout<<"Done statistics tests!"<<endl;
}
/** just verify that it calculate the right things */
void timeseries_test::test_ts_statistics_speed() {
    calendar utc;
    auto t0 = utc.time(2015, 1, 1);

    auto fx_1 = [t0](size_t i, utctime t)->double {return double( rand()/36000.0 );};// should generate 0..9 constant ts.
    auto n_days=365*10;
    auto n_ts=100;
    tta_t  ta(t0, calendar::HOUR, n_days*24);
    tta_t tad(t0, calendar::DAY, n_days);
    auto tsv1 = create_test_ts(n_ts, ta, fx_1);
    cout<<"\nStart calc percentiles "<< n_days<<" days, x "<<n_ts<< " ts\n";
    //auto r1 = calculate_percentiles(tad, tsv1, {0,10,50,-1,70,100});
    vector<tts_t> r1;
    auto f1 = [&tad,&tsv1,&r1]() {r1=calculate_percentiles(tad, tsv1, {0,10,50,-1,70,100});};
    auto msec1 = measure<>::execution(f1);
	//auto f1 = [&favg, &td](double &sum) {sum = 0.0; for (size_t i = 0; i < td.size(); ++i) sum += favg.value(i); };
    TS_ASSERT_EQUALS(size_t(6), r1.size());//, "expect ts equal to percentiles wanted");
    cout<<"Done statistics speed tests, "<<msec1<<" ms"<<endl;
}


