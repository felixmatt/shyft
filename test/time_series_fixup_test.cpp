#include "test_pch.h"

#include <cmath>
#include <vector>

#include "core/utctime_utilities.h"
#include "core/time_axis.h"
#include "api/time_series.h"


namespace shyft {
namespace api {

using std::shared_ptr;
using std::isfinite;
using std::string;
using std::move;

/** \brief quality and correction parameters
 *
 *  Controls how we consider the quality of the time-series,
 *  and in what condition to give up to put in a correction value.
 *
 */
struct qac_parameter {
    utctimespan max_timespan{max_utctime};///< max time span to fix
    double min_x{shyft::nan};    ///< x < min_x                 -> nan
    double max_x{shyft::nan};    ///< x > max_x                 -> nan
    double max_dx_dt{shyft::nan};///< x > x0 + max_dx_dt*(t-t0) -> nan
    double min_dx_dt{shyft::nan};///< x < x0 - min_dx_dt*(t-t0) -> nan
    qac_parameter()=default;
    bool dx_dt_active() const {return isfinite(min_dx_dt)|| isfinite(max_dx_dt);}

    /** check agains min-max is set */
    bool is_ok_quality(const double& x) const noexcept {
        if(isfinite(min_x) && x < min_x)
            return false;
        if(isfinite(max_x) && x > max_x)
            return false;
        return true;
    }

    /** a naive straight forward check routine
     *
     * consider: replace with template generated specialized routines
     * that only check value against configured limits.(faster!)
     * \param t0 previous time-step time-value
     * \param x0 previous time-step x-value
     * \param t  current time-step time-value
     * \param x  current time-step x-value
     * \return false if the (t,x) does not meet criteria limits or is nan, otherwise true
     */
    bool is_ok_quality(const utctime t0,const double& x0, const utctime t,const double& x) const noexcept {
        if(!isfinite(x))
            return false;
        if(isfinite(x0)) {
            if(isfinite(max_dx_dt)) {
                if( x > x0 + (t-t0)*max_dx_dt )
                    return false;
            }
            if(isfinite(min_dx_dt)) {
                if( x < x0 - (t-t0)*min_dx_dt )
                    return false;
            }
        }
        return is_ok_quality(x);
    }

    //x_serialize_decl();
};

/** \brief The average_ts is used for providing ts average values over a time-axis
 *
 * Given a source ts, apply qac criteria, and replace nan's with
 * correction values as specified by the parameters, or the
 * intervals as provided by the specified time-axis.
 *
 * true average for each period in the time-axis is defined as:
 *
 *   integral of f(t) dt from t0 to t1 / (t1-t0)
 *
 * using the f(t) interpretation of the supplied ts (linear or stair case).
 *
 * The \ref ts_point_fx is always POINT_AVERAGE_VALUE for the result ts.
 *
 * \note if a nan-value intervals are excluded from the integral and time-computations.
 *       E.g. let's say half the interval is nan, then the true average is computed for
 *       the other half of the interval.
 *
 */
struct qac_ts:ipoint_ts {
    shared_ptr<ipoint_ts> ts;///< the source ts
    shared_ptr<ipoint_ts> cts;///< optional ts with replacement values
    qac_parameter p;///< the parameters that control how the qac is done

    // useful constructors

    qac_ts(const apoint_ts& ats):ts(ats.ts) {}
    qac_ts(apoint_ts&& ats):ts(move(ats.ts)) {}
    qac_ts(const shared_ptr<ipoint_ts> &ts ):ts(ts){}

    qac_ts(const apoint_ts& ats, const qac_parameter& qp,const apoint_ts& cts):ts(ats.ts),cts(cts.ts),p(qp) {}
    qac_ts(const shared_ptr<ipoint_ts>& ats, const qac_parameter& qp,const shared_ptr<ipoint_ts>& cts):ts(ats),cts(cts),p(qp) {}

    // std copy ct and assign
    qac_ts()=default;

    // implement ipoint_ts contract, these methods just forward to source ts
    virtual ts_point_fx point_interpretation() const {return ts->point_interpretation();}
    virtual void set_point_interpretation(ts_point_fx pfx) {ts->set_point_interpretation(pfx);}
    virtual const gta_t& time_axis() const {return ts->time_axis();}
    virtual utcperiod total_period() const {return ts->time_axis().total_period();}
    virtual size_t index_of(utctime t) const {return ts->index_of(t);}
    virtual size_t size() const {return ts->size();}
    virtual utctime time(size_t i) const {return ts->time(i);};

    // methods that needs special implementation according to qac rules
    virtual double value(size_t i) const {
        auto t=ts->time(i);
        double x=ts->value(i);
        if(i>0) { // do we have a previous value ?
            double t0= ts->time(i-1);
            double x0= ts->value(i-1);
            if(p.is_ok_quality(t0,x0,t,x))
                return x;
        } else {
            if(p.is_ok_quality(x))
                return x;
        }
        // fill in replacement value
        if(cts) // use a correction value ts if available
            return cts->value_at(t); // we do not check this value, assume ok!
        size_t n =ts->size();
        if(i==0 ||p.max_timespan == 0) // lack possible previous value
            return shyft::nan;
        size_t j=i;
        while(j--) { // find previous ok point
            utctime t0=ts->time(j);
            if( t-t0 > p.max_timespan)
                return shyft::nan;// exceed configured max timespan,->nan
            double x0=ts->value(j);//
            if(isfinite(x0)) { // got a previous point
                // check that it is valid ..(min-max)
                if(!p.is_ok_quality(x0))
                    continue;
                // then if we have min/max_dx_dt limits, also check previous point.. and dx-dt..
                if(p.dx_dt_active() && j>0) {
                    double x00=ts->value(j-1);
                    utctime t00=ts->time(j-1);
                    if(!p.is_ok_quality(t00,x00,t0,x0))
                        continue;
                }
                // here we are at a point where t0,x0 is valid ,(or at the beginning)
                for(size_t k=i+1;k<n;++k) { // then find next ok point
                    utctime t1=ts->time(k);
                    if(t1-t0 > p.max_timespan)
                        return shyft::nan;// exceed configured max time span ->nan
                    double x1= ts->value(k);//
                    if(isfinite(x1)) { // got a next  point
                        // check that it is min-max valid
                        if(!p.is_ok_quality(t0,x0,t1,x1))
                            continue;
                        double a= (x1-x0)/(t1-t0);
                        double b= x0 - a*t0;// x= a*t + b -> b= x- a*t
                        double xt= a*t + b;
                        return xt;
                    }
                }
            }
        }
        return shyft::nan; // if we reach here, we failed to find substitute
    }

    virtual double value_at(utctime t) const {
        size_t i = index_of(t);
        if(i == string::npos)
            return shyft::nan;
        double x0 = value(i);
        if(ts->point_interpretation()== ts_point_fx::POINT_AVERAGE_VALUE)
            return x0;
        // linear interpolation between points
        utctime t0 = time(i);
        if(t0==t) // value at endpoint is exactly the point-value
            return x0;
        if(i+1>=size())
            return shyft::nan;// no next point, ->nan
        double x1= value(i+1);
        if(!isfinite(x1))
            return shyft::nan;//  next point is nan ->nan
        utctime t1= ts->time(i+1);
        double a= (x1 - x0)/(t1 - t0);
        double b= x0 - a*t0;
        return a*t + b; // otherwise linear interpolation
    }
    virtual vector<double> values() const {
        const size_t n{size()};
        vector<double> r;r.reserve(n);
        for(size_t i=0;i<n;++i)
            r.emplace_back(value(i));
        return r;
    }

    // methods for binding and symbolic ts
    virtual bool needs_bind() const {
        return ts->needs_bind() || (cts && cts->needs_bind());
    }
    virtual void do_bind() {
        ts->do_bind();
        if(cts)
            cts->do_bind();
    }

    //x_serialize_decl();

};


}
}

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
            FAST_CHECK_EQ(q.is_ok_quality(no_utctime,shyft::nan,0,shyft::nan),false);
            FAST_CHECK_EQ(q.is_ok_quality(no_utctime,shyft::nan,0,1.0),true);
        }

        SUBCASE("min/max abs limits") {
            q.max_x=1.0;
            FAST_CHECK_EQ(q.is_ok_quality(no_utctime,shyft::nan,0,1.0),true);
            FAST_CHECK_EQ(q.is_ok_quality(no_utctime,shyft::nan,0,1.0+eps),false);
            q.min_x=-1.0;
            FAST_CHECK_EQ(q.is_ok_quality(no_utctime,shyft::nan,0,-1.0),true);
            FAST_CHECK_EQ(q.is_ok_quality(no_utctime,shyft::nan,0,-1.0-eps),false);
        }

        SUBCASE("min-max dx/dt change") {
            q.max_x=10;
            q.min_x=-10;
            q.max_dx_dt=0.5;
            FAST_CHECK_EQ(q.is_ok_quality(0,1.0, 1,1.5),true);
            FAST_CHECK_EQ(q.is_ok_quality(0,1.0, 1,1.5+eps),false);
            q.min_dx_dt=0.5;
            FAST_CHECK_EQ(q.is_ok_quality(0,1.0, 1,0.5),true);
            FAST_CHECK_EQ(q.is_ok_quality(0,1.0, 1,0.5-eps),false);
            //-- also check that we can hit abs-max limits at the same time
            FAST_CHECK_EQ(q.is_ok_quality(0,10.0, 1,10.0),true);
            FAST_CHECK_EQ(q.is_ok_quality(0,10.0, 1,10.1),false);
            FAST_CHECK_EQ(q.is_ok_quality(0,-10.0, 1,-10.0),true);
            FAST_CHECK_EQ(q.is_ok_quality(0,-10.0, 1,-10.1),false);
        }
    }

    TEST_CASE("qac_ts") {
        generic_dt ta{0,10,5};
        //                 0    1       2     3    4
        vector<double> v {0.0,1.0,shyft::nan,3.0,-20.1};
        apoint_ts src(ta,v,ts_point_fx::POINT_AVERAGE_VALUE);
        apoint_ts cts;
        qac_parameter qp;
        auto ts = make_shared<qac_ts>(src,qp,cts);

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

        // verify min/max dx_dt limits
        ts->p.min_dx_dt=1;
        FAST_CHECK_UNARY(!isfinite(ts->value_at(ts->time(3)+1)));
        ts->p.min_dx_dt=100;
        FAST_CHECK_UNARY(isfinite(ts->value_at(ts->time(3)+1)));
        ts->p.max_dx_dt=0;
        FAST_CHECK_UNARY(isfinite(ts->value_at(ts->time(3)+1)));
        FAST_CHECK_UNARY(isfinite(ts->value_at(ts->time(3)-1)));
        // verify
    }

}
