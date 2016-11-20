#include "test_pch.h"
#define _USE_MATH_DEFINES
#include "timeseries_test.h"
#include "mocks.h"
#include "core/timeseries.h"
#include "core/time_axis.h"
#include "api/api.h"
#include "api/timeseries.h"
#include "api/ts_serialization.h"


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
    } catch (const exception & ) {

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
    TS_ASSERT_EQUALS(ta.n_get_calls,2);// great using linear approach to find near solution upward.

    ta.reset_call_count();

    TS_ASSERT_EQUALS(hint_based_search(ta,utcperiod(t-1*d,t-0*d),5),string::npos);
    TS_ASSERT_EQUALS(ta.n_index_of_calls,0);// in this case, local search ends up with conclusive npos

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
	TS_ASSERT_LESS_THAN(std::abs(msec1-msec2),100);

	//cout << "\nTiming results:" << endl;
	//cout << "    T generic : " << msec1 << " (" << s1 << ")" << endl;
	//cout << "    T fast    : " << msec2 << " (" << s2 << ")" << endl;
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

    // verify expression is updated when terminals changes
    a.fill(a_value);
    b.fill(b_value);
    auto d= a + b;
    TS_ASSERT_DELTA(d.value(0),a_value+b_value,0.00001);
    a.set(0,b_value);
    TS_ASSERT_DELTA(d.value(0),b_value+b_value,0.00001);
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

/** just verify that it calculate at full speed */
void timeseries_test::test_ts_statistics_speed() {
    calendar utc;
    auto t0 = utc.time(2015, 1, 1);

    auto fx_1 = [t0](size_t i, utctime t)->double {return double( rand()/36000.0 );};// should generate 0..9 constant ts.
#ifdef _DEBUG
    auto n_days=365*100;
#else
	auto n_days = 365 * 10;// fewer for debug
#endif
	auto n_ts=10;
    tta_t  ta(t0, calendar::HOUR, n_days*24);
    tta_t tad(t0, deltahours(24), n_days);
    auto tsv1 = create_test_ts(n_ts, ta, fx_1);
    bool verbose = getenv("SHYFT_VERBOSE") != nullptr;
    if(verbose) cout << "\nStart calc percentiles " << n_days << " days, x " << n_ts << " ts\n";
    //auto r1 = calculate_percentiles(tad, tsv1, {0,10,50,-1,70,100});
    vector<tts_t> r1;
    auto f1 = [&tad, &tsv1, &r1](int min_t_steps) {r1=calculate_percentiles(tad, tsv1, {0,10,50,-1,70,100},min_t_steps);};
    for (int sz = tad.size(); sz > 100; sz /= 2) {
        auto msec1 = measure<>::execution(f1,sz);
        if(verbose) cout<<"statistics speed tests, "<< tad.size() <<" steps, pr.thread = "<< sz << " steps: "<< msec1 << " ms" <<endl;
    }
    //auto msec2= measure<>::execution(f1,tad.size()/4);
    //cout<<"Done statistics speed tests,2 threads "<<msec2<<" ms"<<endl;
}

void timeseries_test::test_timeshift_ts() {
    using namespace shyft;
    using namespace shyft::core;
    using namespace shyft::timeseries;
    typedef point_ts<time_axis::fixed_dt> pts_t;
    calendar utc;
    utctime t0=utc.time(2016,1,1);
    utctime t1=utc.time(2016,1,1, 1);
    utctimespan dt=deltahours(1);
    size_t n = 24;
    time_axis::fixed_dt ta0(t0,dt,n);
    pts_t ts0(ta0,0.0);
    auto ts1 = time_shift(ts0,t1-t0);
    auto ts2 = time_shift(ts1,t0-t1);

    TS_ASSERT(is_equal_ts(ts0,ts2));// time-shift back and forth, should give same ts
    TS_ASSERT(!is_equal_ts(ts0,ts1));// the time-shifted ts is different from the other

    for(size_t i=0;i<ts0.size();++i) {
        TS_ASSERT_DELTA(ts0.value(i),ts1.value(i),1e-7);//The values should be exactly equal
        TS_ASSERT_EQUALS(ts0.time(i)+(t1-t0), ts1.time(i));// just to verify the offsets are ok (also tested by time_axis fx)
    }

    auto c = time_shift(4.0*ts1-ts0,t0-t1); // if it compiles!, then ts-operators are working ok.
}

void timeseries_test::test_periodic_ts_t() {
	vector<double> v = { 1, 2, 3, 4, 5, 6, 7, 8 };
	calendar utc;
	utctime t0 = utc.time(2015, 1, 1);
	timeaxis ta(t0, deltahours(10), 1000);

	typedef periodic_ts<timeaxis> periodic_ts_t;
	periodic_ts_t pts(v, deltahours(3), ta);

	TS_ASSERT_EQUALS(pts.size(), 1000);
	TS_ASSERT_EQUALS(pts.index_of(t0), 0);
}

void timeseries_test::test_periodic_ts_over_sampled() {
	vector<double> v = { 1, 2, 3, 4, 5, 6, 7, 8 };
	calendar utc;
	utctime t0 = utc.time(2015, 1, 1);
	timeaxis ta(t0, deltahours(1), 1000);

	typedef periodic_ts<timeaxis> periodic_ts_t;
	periodic_ts_t pts(v, deltahours(3), ta);

	TS_ASSERT_EQUALS(pts.size(), 1000);
	TS_ASSERT_EQUALS(pts.index_of(t0), 0);
	TS_ASSERT_EQUALS(pts.value(0), v[0]);
	TS_ASSERT_EQUALS(pts.value(1), v[0]);
	TS_ASSERT_EQUALS(pts.value(2), v[0]);
	TS_ASSERT_EQUALS(pts.value(3), v[1]);
}

void timeseries_test::test_periodic_ts_concept() {
	// Periodic profile having 8 samples spaced by 3 h
	std::array<double, 8> profile = { 1, 2, 3, 4, 5, 6, 7, 8 };
	const utctimespan dt = deltahours(3);
	calendar utc;
	utctime t0 = utc.time(2016, 1, 1);

	struct periodic_ts {
		int nt;
		utctimespan dt;
		utctimespan period;
		utctime t0;
		std::array<double, 8> profile;

		int map_index(utctime t) const {
			t -= t0;
			if (t < 0 || period < t) {
				t = fmod(t, period);
				if (t < 0) t += period;
			}
			return round(double(t) / double(dt));
		}

		double operator() (utctime t) const {
			int i = map_index(t);
			if (i == nt) i = 0;
			return profile[i];
		}
	}
	fun = { 8, dt, 8*dt, t0, profile };

	// Test folding forward and backward
	for (int i = 0; i<8; i++) {
		TS_ASSERT_DELTA(fun(t0 + i*dt), profile[i], 1e-9);
		TS_ASSERT_DELTA(fun(t0 + 3*8*dt + i*dt), profile[i], 1e-9);
		TS_ASSERT_DELTA(fun(t0 - 3*8*dt + i*dt), profile[i], 1e-9);
	}

	// Test stair and folding
	TS_ASSERT_DELTA(fun(t0 - dt), profile[7], 1e-9);
	TS_ASSERT_DELTA(fun(t0 + dt/2 - 1), profile[0], 1e-9);
	TS_ASSERT_DELTA(fun(t0 + dt/2), profile[1], 1e-9);
	TS_ASSERT_DELTA(fun(t0 - dt/2 - 1), profile[7], 1e-9);
	TS_ASSERT_DELTA(fun(t0 - dt/2), profile[0], 1e-9);
}

void timeseries_test::test_periodic_template_ts() {
	std::vector<double> pv = { 1, 2, 3, 4, 5, 6, 7, 8 };
	const utctimespan dt = deltahours(3);
	calendar utc;
	utctime t0 = utc.time(2015, 1, 1);
	timeaxis ta(t0, deltahours(10), 1000);

	profile_description pd(t0, dt, pv);

	TS_ASSERT_DELTA(pd(0), pv[0], 1e-9);
	TS_ASSERT_DELTA(pd.size(), pv.size(), 1e-9);

	periodic_ts< timeaxis> fun(pd, ta, point_interpretation_policy::POINT_AVERAGE_VALUE);
	// case 0: time-axis delta t covers several steps/values of the pattern
	TS_ASSERT_EQUALS(fun.size(), 1000);
	TS_ASSERT_EQUALS(fun.index_of(t0), 0);
	TS_ASSERT_EQUALS(fun.value(0), 2.2);
	TS_ASSERT_EQUALS(fun.value(1), 5.5);
	TS_ASSERT_EQUALS(fun.value(2), 4.0);
	// case 1: as case 0, but
	// t0 of the profile do *not* match the time-axis tstart
	// and time-of day is also different:
	//
	profile_description pd1(utc.time(2000,1,1,2), dt, pv);
	periodic_ts<timeaxis> fx(pd1, ta, POINT_AVERAGE_VALUE);
	TS_ASSERT_EQUALS(fx.value(0), 3.1);// verified using excel
	TS_ASSERT_EQUALS(fx.value(1), 4.8);
	TS_ASSERT_EQUALS(fx.value(2), 5.0);// overlaps next day as well
	// case 2: now test another case where the time-axis is 1hour, and we have POINT_AVERAGE_VALUE(stair-case-type f(t))
	// between points.
	timeaxis ta_hour(t0, deltahours(1), 24);
	periodic_ts<timeaxis> fs(pd1, ta_hour, POINT_AVERAGE_VALUE);
	vector<double> expected_fs = { 8.0,8.0,1.0,1.0,1.0,2.0,2.0,2.0,3.0,3.0,3.0,4.0,4.0,4.0,5.0,5.0,5.0,6.0,6.0,6.0,7.0,7.0,7.0,8.0 };
	for (size_t i = 0;i < expected_fs.size();++i)
		TS_ASSERT_DELTA(expected_fs[i], fs.value(i), 0.00001);
	// case 3: as case 2, but POINT_INSTANT_VALUE and linear-between points type of f(t)
	//
	periodic_ts< timeaxis> fl(pd1, ta_hour, POINT_INSTANT_VALUE);
	vector<double> expected_fl = { 4.500000,2.166667,1.166667,1.500000,1.833333,2.166667,2.500000,2.833333,3.166667,3.500000,3.833333,4.166667,4.500000,4.833333,5.166667,5.500000,5.833333,6.166667,6.500000,6.833333,	7.166667,7.500000,7.833333, 8.0 /*	ok, if we consider f(t) keep value at the end, otherwise it's 6.833333 */	};
	for (size_t i = 0;i < expected_fl.size();++i)
		TS_ASSERT_DELTA(expected_fl[i], fl.value(i), 0.00001);
}

void timeseries_test::test_periodic_ts_values() {
	std::vector<double> pv = { 1, 2, 3, 4, 5, 6, 7, 8 };
	const utctimespan dt = deltahours(3);
	calendar utc;
	utctime t0 = utc.time(2015, 1, 1);
	timeaxis ta(t0, deltahours(10), 1000);

	profile_description pd(t0, dt, pv);
	periodic_ts< timeaxis> fun(pd, ta, point_interpretation_policy::POINT_AVERAGE_VALUE);

	auto v = fun.values();
	TS_ASSERT_EQUALS(v.size(), ta.size());
	TS_ASSERT_EQUALS(v[0], 2.2);
	TS_ASSERT_EQUALS(v[1], 5.5);
	TS_ASSERT_EQUALS(v[2], 4.0);
}

void timeseries_test::test_accumulate_value() {
	calendar utc;
	auto t = utc.time(2015, 5, 1, 0, 0, 0);
	auto d = deltahours(1);
	size_t n = 10;
	vector<double> values;for (size_t i = 0;i < n;++i) values.emplace_back(i!= 5?i*1.0:shyft::nan);//0, 1,2,4,nan,6..9
	//Two equal timeaxis representations
	timeaxis ta(t, d, n);
	point_ts<timeaxis> a(ta, values,point_interpretation_policy::POINT_INSTANT_VALUE);// so a is a straight increasing line
	accumulate_accessor<point_ts<timeaxis>, timeaxis> aa(a, ta); // while we have the test-setup, we test both the function, the accessor
	accumulate_ts<point_ts<timeaxis>, timeaxis> ats(a, ta);// and even the core time-series implementation

	utctimespan tsum;
	size_t last_ix = 0;
	auto y1 = accumulate_value(a, ta.period(0), last_ix, tsum, true);
	TS_ASSERT_DELTA(0.5*deltahours(1), y1, 0.0001);
	TS_ASSERT_EQUALS(tsum, deltahours(1));
	auto y2 = accumulate_value(a, utcperiod(ta.time(0), ta.time(2)), last_ix, tsum, true);
	TS_ASSERT_DELTA(1.0*deltahours(2), y2, 0.0001);
	TS_ASSERT_EQUALS(tsum, deltahours(2));
	auto y3 = accumulate_value(a, utcperiod(ta.time(0), ta.time(6)), last_ix, tsum, true);// last part of accumulation is nan, so 4 flattens out
	TS_ASSERT_DELTA(2.0*deltahours(4)+4.0*deltahours(1), y3, 0.0001);
	// besides, since average_value and friends uses accumulate_value, this function is pretty well tested from those tests.
}

void timeseries_test::test_accumulate_ts_and_accessor() {
	calendar utc;
	auto t = utc.time(2015, 5, 1, 0, 0, 0);
	auto d = deltahours(1);
	size_t n = 10;
	vector<double> values;for (size_t i = 0;i < n;++i) values.emplace_back(i != 5 ? i*1.0 : shyft::nan);//0, 1,2,4,nan,6..9
																										//Two equal timeaxis representations
	timeaxis ta(t, d, n);
	point_ts<timeaxis> a(ta, values, point_interpretation_policy::POINT_INSTANT_VALUE);// so a is a straight increasing line
	accumulate_accessor<point_ts<timeaxis>, timeaxis> aa(a, ta); //  the accessor
	accumulate_ts<point_ts<timeaxis>, timeaxis> ats(a, ta);// and even the core time-series implementation
	TS_ASSERT_DELTA(0.0, aa.value(0), 0.001);
	TS_ASSERT_DELTA(0.0, ats.value(0), 0.001);
	// simple test at ix=1:
	TS_ASSERT_DELTA(0.5*deltahours(1), aa.value(1), 0.001);
	TS_ASSERT_DELTA(0.5*deltahours(1), ats.value(1), 0.001);
	// now the ts have some extra feature through it's f(t), operator()
	TS_ASSERT_DELTA(0.25*deltaminutes(30), ats(t + deltaminutes(30)), 0.0001);
	// the accessor should be smart, trying to re-use prior computation, I verify the result here,
	TS_ASSERT_DELTA(1.0*deltahours(2), aa.value(2), 0.0001);// and using step-debug to verify it's really doing the right thing
}

void timeseries_test::test_partition_by() {
	calendar utc;
	auto t = utc.time(1930, 9, 1, 0, 0, 0);
	auto d = deltahours(1);
	size_t n = utc.diff_units(t, utc.time(2016, 10, 1), d);

	shyft::api::gta_t ta(t, d, n);
	size_t n_years = 80;
	vector<double> values;values.reserve(n);
	for (size_t i = 0;i < n;++i)
		values.push_back(i);//0, 1,2,4,5,6..9
	shyft::api::apoint_ts src_a(ta, values, point_interpretation_policy::POINT_AVERAGE_VALUE);// so a is a straight increasing stair-case

  // core version : auto mk_time_shift = [](const decltype(src_a)  &ts, utctimespan dt)-> time_shift_ts<decltype(src_a)> {return time_shift(ts,dt);};
	// below is the raw- time-shift version
	auto mk_raw_time_shift = [](const decltype(src_a)& ts, utctimespan dt)->shyft::api::apoint_ts {
		return shyft::api::apoint_ts(std::make_shared<shyft::api::time_shift_ts>(ts, dt));
	};
	/** A class that makes an average partition over the supplied time-axis*/
	struct mk_average_partition {
		shyft::api::gta_t ta;///< the time-axis that is common for all generated partitions
		mk_average_partition(const shyft::api::gta_t& ta) : ta(ta) {}
		shyft::api::apoint_ts operator()(const shyft::api::apoint_ts& ts, utctimespan dt_shift) const {
			shyft::api::apoint_ts r(std::make_shared<shyft::api::time_shift_ts>(ts, dt_shift));
			return r.average(ta);
		}
	};
	auto common_t0 = utc.time(2015, 9, 1);
	shyft::api::gta_t ta_y2015(common_t0, deltahours(1), 365 * 24);
	mk_average_partition mk_avg_time_shift(ta_y2015);// callable that make partitions one year long

	// core version: auto  partitions=partition_by<time_shift_ts<decltype(src_a)>>(src_a,utc,t,calendar::YEAR,n_years,common_t0,mk_time_shift);
	auto partitions_avg_year  = partition_by<shyft::api::apoint_ts>(src_a, utc, t, calendar::YEAR, n_years, common_t0, mk_avg_time_shift);
	auto partitions_raw_shift = partition_by<shyft::api::apoint_ts>(src_a, utc, t, calendar::YEAR, n_years, common_t0, mk_raw_time_shift);
	TS_ASSERT_EQUALS(n_years, partitions_avg_year.size());
	TS_ASSERT_EQUALS(n_years, partitions_raw_shift.size());
	utctime ty = t;
	// verify the avg case, where all the partition time-axis are just one year (2015.09.01 + hourly 365*24 steps
	for (const auto& ts : partitions_avg_year) {
		// verify that the value at common_t0, equals value(t0)
		auto src_ix = src_a.index_of(ty);
		auto ts_ix = ts.index_of(common_t0);
		auto v_common_t0 = ts.value(ts_ix);
		auto v_year_start = src_a.value(src_ix);
		TS_ASSERT_EQUALS(ts.size(), ta_y2015.size());
		TS_ASSERT_EQUALS(ts_ix, 0);// because we make a separate one 365 day time-axis, so the first value should be at 0'th index
		TS_ASSERT_DELTA(v_common_t0, v_year_start, 0.01);// verify from source exactly equal to the partition value
		TS_ASSERT_DELTA(v_year_start, double(src_ix), 0.01); // verify we did get the right value
		ty = utc.add(ty, calendar::YEAR, 1);
	}
	ty = t;
	// verify the raw shift case, where the time-axis are just shifted to align same-value at common_t0 time
	for (const auto& ts : partitions_raw_shift) {
		// verify that the value at common_t0, equals value(t0)
		auto src_ix = src_a.index_of(ty);
		auto ts_ix = ts.index_of(common_t0);
		auto v_common_t0 = ts.value(ts_ix);
		auto v_year_start = src_a.value(src_ix);
		TS_ASSERT_EQUALS(ts.size(), src_a.size()); // because basically, it's the same time-shifted time-series
		TS_ASSERT_DELTA(v_common_t0, v_year_start, 0.01);// verify from source exactly equal to the partition value
		TS_ASSERT_DELTA(v_year_start, double(src_ix), 0.01); // verify we did get the right value
		ty = utc.add(ty, calendar::YEAR, 1);
	}

}
void timeseries_test::test_unit_conversion() {
    double area_m2 =1000;
    double q_m3s = 1;
    TS_ASSERT_DELTA(shyft::m3s_to_mmh(q_m3s,area_m2),q_m3s/area_m2*(1000.0*3600.0),1e-10);
    TS_ASSERT_DELTA(shyft::mmh_to_m3s(shyft::m3s_to_mmh(q_m3s,area_m2),area_m2),q_m3s,1e-10);
}

void timeseries_test::test_ts_ref() {
    using ta_t=shyft::time_axis::fixed_dt;
    using ts_t=shyft::timeseries::point_ts<shyft::time_axis::fixed_dt>;
    using rts_t=shyft::timeseries::ref_ts<ts_t>;
    using calendar=shyft::core::calendar;
    calendar utc;
    ta_t ta(utc.time(2016,10,1),deltahours(1),10);
    ts_t a(ta,1.0,fx_policy_t::POINT_AVERAGE_VALUE);
    auto b=make_shared<rts_t>(); // note that ref_ts b need to be some kind of shared_ptr
    string b_ref_key("netcdf://group/a/b");
    string x_ref_key("fame://nordic_main/price_forecast_NO1_eur_MWh");
    auto x=make_shared<rts_t>();
    x->ref=x_ref_key;
    b->ref=b_ref_key;
    auto c=3.0*a+b*a/4.0*x*x; // otherwise, we would have to bind the b copied in the expression here
                // (also possible)
    size_t n;   // but not practical now.
    try
     {
        n=c.time_axis().size();
        TS_FAIL("unbound ref_ts should throw on access");
    } catch (const runtime_error& ) {
        ;// ok
    } catch (...) {
        TS_FAIL("Expected runtime_error here");
    }
    auto bb= make_shared<ts_t>(ta,2.0,fx_policy_t::POINT_AVERAGE_VALUE);// this is what we would like to bind
    auto bx= make_shared<ts_t>(ta,10.0,fx_policy_t::POINT_AVERAGE_VALUE);// this is what we would like to bind

    auto resolve_sym_ref = [&bb,&b_ref_key,&bx,&x_ref_key](rts_t& rts) {
        if(rts.ref == b_ref_key) // mimic of a lookup that could have been done here.
            rts.ts = bb;
        if(rts.ref== x_ref_key)
            rts.ts = bx;
    };

    bind_ref_ts(c,resolve_sym_ref);
    n=c.time_axis().size(); // this time expr. c is valid
    TS_ASSERT_EQUALS(n,a.time_axis().size());// have proper size
    TS_ASSERT_DELTA(c.value(0), 3*1 + 2*1/4.0*10.0*10.0,0.0001);// and expected value(s)

}
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>


template <class T>
static T serialize_loop(const T& o) {
    using namespace std;
    std::ostringstream xmls;
    boost::archive::xml_oarchive oa(xmls);
    oa << BOOST_SERIALIZATION_NVP(o);
    xmls.flush();
    string ss=xmls.str();
    istringstream xmli(ss);
    boost::archive::xml_iarchive ia(xmli);
    T o2;
    ia>>BOOST_SERIALIZATION_NVP(o2);
    return o2;
}



template<class TA>
static bool is_equal(const shyft::timeseries::point_ts<TA>& a,const shyft::timeseries::point_ts<TA>&b) {
    if(a.size()!=b.size())
        return false;
    if(a.time_axis().total_period()!=b.time_axis().total_period())
        return false;
    if(a.fx_policy!= b.fx_policy)
        return false;
    const double eps=1e-12;
    for(size_t i=0;i<a.size();++i)
        if(fabs(a.v[i]-b.v[i])> eps)
            return false;
    return true;
}


template<class Ts>
static bool is_equal(const Ts& a,const Ts &b) {
    if(a.size()!=b.size())
        return false;
    if(a.time_axis().total_period()!=b.time_axis().total_period())
        return false;
    if(a.point_interpretation()!= b.point_interpretation())
        return false;
    const double eps=1e-12;
    for(size_t i=0;i<a.size();++i)
        if(fabs(a.value(i)-b.value(i))> eps)
            return false;
    return true;
}



void timeseries_test::test_serialization() {
    using namespace std;
    using namespace shyft;

    // testing serialization in the order of appearance/dependency

    //
    // 1. time & calendar
    //

    calendar utc;
    utcperiod p_1(utc.time(2016,1,1),utc.time(2017,1,1));
    auto p_2 = serialize_loop(p_1);
    TS_ASSERT_EQUALS(p_1,p_2);

    time_zone::tz_info_t tzi_1;
    tzi_1.base_tz=deltahours(1);
    tzi_1.tz.tz_name="ANY+01:00";
    tzi_1.tz.dst.emplace_back(utc.time(2016,3,1),utc.time(2016,10,1));
    tzi_1.tz.dt.push_back(deltahours(3600));

    auto tzi_2 = serialize_loop(tzi_1);
    TS_ASSERT_EQUALS(tzi_1.base_tz,tzi_2.base_tz);
    TS_ASSERT_EQUALS(tzi_1.tz.tz_name,tzi_2.tz.tz_name);
    TS_ASSERT_EQUALS(tzi_1.tz.dst,tzi_2.tz.dst);
    TS_ASSERT_EQUALS(tzi_1.tz.dt,tzi_2.tz.dt);

    auto osl=make_shared<calendar>("Europe/Oslo");
    auto osl2 = serialize_loop(osl);

    TS_ASSERT_EQUALS(osl->tz_info->base_tz,osl2->tz_info->base_tz);
    TS_ASSERT_EQUALS(osl->tz_info->tz.tz_name,osl2->tz_info->tz.tz_name);

    //
    // 2. time-axis
    //
    time_axis::fixed_dt ta(utc.time(2016,1,1),deltahours(1),24);
    auto ta2 = serialize_loop(ta);
    TS_ASSERT_EQUALS(ta.t,ta2.t);
    TS_ASSERT_EQUALS(ta.dt,ta2.dt);
    TS_ASSERT_EQUALS(ta.n,ta2.n);

    time_axis::calendar_dt tac(osl,osl->time(2016,7,1),deltahours(1),24);

    auto tac2 = serialize_loop(tac);
    TS_ASSERT_EQUALS(tac.size(),tac2.size());
    TS_ASSERT_EQUALS(tac.total_period(),tac2.total_period());

    time_axis::generic_dt tag(ta);
    auto tag2 = serialize_loop(tag);
    TS_ASSERT_EQUALS(tag.gt,tag2.gt);
    TS_ASSERT_EQUALS(tag.size(),tag2.size());
    TS_ASSERT_EQUALS(tag.total_period(),tag2.total_period());

    //
    // 3. time-series
    //

    timeseries::point_ts<time_axis::fixed_dt> ts(ta,1.0,timeseries::fx_policy_t::POINT_AVERAGE_VALUE);
    auto ts2 = serialize_loop(ts);
    TS_ASSERT(is_equal(ts,ts2));
    time_axis::point_dt tap(vector<utctime>{0,3600},3600*2);
    auto tsp=make_shared<timeseries::point_ts<time_axis::point_dt>>(tap,2.0,timeseries::fx_policy_t::POINT_INSTANT_VALUE);
    auto tsp2 = serialize_loop(tsp);
    TS_ASSERT(is_equal(*tsp,*tsp2));

    timeseries::time_shift_ts<decltype(ts)> tsts(ts,deltahours(3600));
    auto tsts2 = serialize_loop(tsts);
    TS_ASSERT(is_equal(tsts,tsts2));

    timeseries::average_ts<decltype(ts),decltype(ta) > tsavg(ts,ta);
    auto tsavg2=serialize_loop(tsavg);
    TS_ASSERT(is_equal(tsts,tsts2));

    timeseries::accumulate_ts<decltype(ts),decltype(ta) > tsacc(ts,ta);
    auto tsacc2=serialize_loop(tsacc);
    TS_ASSERT(is_equal(tsacc,tsacc2));


    timeseries::periodic_ts<decltype(ta)> tspp(vector<double>{1.0,10.0,2.0,3.0},deltahours(1),utc.time(2016,1,1),ta);
    auto tspp2=serialize_loop(tspp);
    TS_ASSERT(is_equal(tspp,tspp2));

    timeseries::glacier_melt_ts<decltype(ts)> tsgm(ts,ts,1000.0,6.2);
    auto tsgm2=serialize_loop(tsgm);
    TS_ASSERT(is_equal(tsgm,tsgm2));

    auto c = max(ts+ts*2.0,(1.0 + tsp)*tsavg);
    auto c2 = serialize_loop(c);
    TS_ASSERT(is_equal(c,c2));

    //-- api time-series

    api::gpoint_ts gts(tag,10.0);
    auto gts2=serialize_loop(gts);
    TS_ASSERT(is_equal(gts,gts2));

    shared_ptr<api::ipoint_ts> igts=make_shared<api::gpoint_ts>(tag,2.5);
    auto igts2 = serialize_loop(igts);
    TS_ASSERT(is_equal(*igts,*igts2));

    api::apoint_ts agts(tag,20.0);
    auto agts2 = serialize_loop(agts);
    TS_ASSERT(is_equal(agts,agts2));

    api::average_ts gtsavg(tag,agts);
    auto gtsavg2 = serialize_loop(gtsavg);
    TS_ASSERT(is_equal(gtsavg,gtsavg2));

    api::accumulate_ts gtsacc(tag,agts);
    auto gtsacc2 = serialize_loop(gtsacc);
    TS_ASSERT(is_equal(gtsacc,gtsacc2));

    api::time_shift_ts atsts(igts,deltahours(24));
    auto atsts2= serialize_loop(atsts);
    TS_ASSERT(is_equal(atsts,atsts2));

    api::periodic_ts apts(vector<double>{1.0,10.0,5.0,2.0},deltahours(1),tag);
    auto apts2= serialize_loop(apts);
    TS_ASSERT(is_equal(apts,apts2));

    api::aref_ts arts("netcdf://file.nc");
    arts.rep.ts=make_shared<api::gts_t>(tag,1.0,timeseries::fx_policy_t::POINT_AVERAGE_VALUE);
    auto arts2=serialize_loop(arts);
    TS_ASSERT_EQUALS(arts.rep.ref,arts2.rep.ref);
    TS_ASSERT(is_equal(arts,arts2));

    auto aexpr = (agts*2.0 + agts/4.0 + 12)/agts;
    auto aexpr2 = serialize_loop(aexpr);
    TS_ASSERT(is_equal(aexpr,aexpr2));
    
    // verify vector stuff.
    vector<api::apoint_ts> tsv;
    tsv.push_back(agts);
    tsv.push_back(3.0*agts+agts);
    tsv.push_back(10.0*agts+ 1.0/agts);
    auto tsv2 = serialize_loop(tsv);

    TS_ASSERT_EQUALS(tsv.size(), tsv2.size());
    for (size_t i = 0;i < tsv.size();++i)
        TS_ASSERT(is_equal(tsv[i], tsv2[i]));

}



void timeseries_test::test_api_ts_ref_binding() {
    using namespace shyft;
    using namespace std;
    calendar utc;
    time_axis::generic_dt ta(utc.time(2016,1,1),deltahours(1),24);
    api::apoint_ts a(ta,1.0);
    api::apoint_ts b(ta,2.0);
    string s_c="fm::/nordic_main/xyz";
    api::apoint_ts c(s_c);
    string s_d="netcdf://arome_2016_01_01T00:00/UTM32/E12.123/N64.222";
    api::apoint_ts d(s_d);
    auto f = 3.0*a*(b+(c*d)*4);
    auto tsr=f.find_ts_bind_info();
    TS_ASSERT_EQUALS(tsr.size(),2);
    try {
        f.value(0);
        TS_FAIL("Expected runtine_error here");

    } catch (const runtime_error&) {
        ;//OK!
    }
    auto xmls_unbound = f.serialize();
    //cout<<"expression xml before bind\n";
    //cout<<serialize(f);
    //cout<<"\n";
    // -now bind the variables
    api::apoint_ts b_c(ta,5.0);
    api::apoint_ts b_d(ta,3.0);

    for (auto&bind_info : tsr) {
        if (bind_info.reference == s_c)
            bind_info.ts.bind(b_c);
        else if (bind_info.reference == s_d)
            bind_info.ts.bind(b_d);
        else
            TS_FAIL("ref not found");
    }
    // then retry evaluate
    try {
        double v0=f.value(0);
        TS_ASSERT_DELTA(v0,3.0*1.0*(2.0+(5.0*3.0)*4),1e-9);
    } catch (const runtime_error&) {
        TS_FAIL("Sorry, still not bound values");
    }
    auto a_f = api::apoint_ts::deserialize(xmls_unbound);
    auto unbound_ts = a_f.find_ts_bind_info();
    for (auto&bind_info : unbound_ts) {
        if (bind_info.reference == s_c)
            bind_info.ts.bind(b_c);
        else if (bind_info.reference == s_d)
            bind_info.ts.bind(b_d);
        else
            TS_FAIL("ref not found");
    }
    TS_ASSERT_DELTA(f.value(0), a_f.value(0), 1e-9);

    //cout<<"expression xml after bind\n";
    //cout<<serialize(f);
    //cout<<"\n";

}

void timeseries_test::test_serialization_performance() {
    using namespace shyft;
    using namespace shyft::core;
    using namespace std;
    bool verbose = getenv("SHYFT_VERBOSE") ? true : false;
    //
    // 1. create one large ts, do loop it.
    //
    calendar utc;
    size_t n = 1*1000*1000;// gives 8 Mb memory
    vector<double> x;x.reserve(n);
    double x0 = -double(n) / 2.0;
    for (size_t i = 0;i < n;++i) 
        x.push_back(-double(n)/2.0 + i);
    api::apoint_ts aa(api::gta_t(utc.time(2016, 1, 1), deltahours(1), n), x);
    auto a = aa*3.0 + aa;
    //
    // 2. serialize it
    //
    std::clock_t t0 = std::clock();
    auto xmls = a.serialize();
    auto ms = (std::clock() - t0)*1000.0 / double(CLOCKS_PER_SEC);
    if(verbose)cout << "\nserialization took " << ms << "ms\n";
    TS_ASSERT_LESS_THAN(ms, 200.0); // i7 ~ 10 ms
    t0 = std::clock();
    auto b = api::apoint_ts::deserialize(xmls);
    ms = (std::clock() - t0)*1000.0 / double(CLOCKS_PER_SEC);
    TS_ASSERT_LESS_THAN(ms, 200.0);// i7 ~ 10 ms
    if(verbose) cout  << "de-serialization took " << ms << "ms\n\tsize:"<<xmls.size()<<" bytes \n";
    TS_ASSERT(is_equal(a, b));
}