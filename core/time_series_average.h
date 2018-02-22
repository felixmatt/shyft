#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include "utctime_utilities.h"
#include "time_series_common.h"

namespace shyft {namespace time_series {

using std::vector;
using std::isfinite;
using std::max;
using std::min;
using std::string;
using core::utcperiod;
using core::utctime;
using core::utctimespan;

/** \brief compute non_nan_integral|average for a linear interpreted time-series
  *
  * For each interval in the specified time-axis (continuous)
  * compute the non-nan integral|average in the interval.
  *
  * The points in the supplied time-series is interpreted according
  * to it's ts_point_fx policy.
  *
  *  #) stair_case|POINT_AVERAGE_VALUE:
  *     it represents the average value of the interval
  *     so it's value
  *       f(t) = c for t in [start..end>
  *
  *     f(t) is nan for all points outside ts.total_period()
  *
  *  #) linear| POINT_INSTANT_VALUE:
  *     the point represent the value at the time specified
  *     and f(t) is *linear between* points:
  *
  *       f(t) = a*t + b, for t in [start..end>
  *              where a,b describes the straight line
  *              between points (start,f(start)) (end,f(end))
  *              where we require:
  *               1. a finite left-hand value of f(start)
  *               2. a finite right-hand value of f(end)
  *
  *     first-point, last-point and nan-point considerations:
  *
  *     from the above definition of interval values:
  *     f(t) is nan before the first non-nan point (t,v).
  *     f(t) is nan immediately after the last non-nan point(t,v)
  *     f(t) *left-hand value* is nan if the following point is nan
  *
  *     note: from this follows that you need two-points to have
  *           a non-nan value for linear type of integral
  *
  *     e.g
  *
  *      ^
  *      |      o  x
  *      |    /       o
  *      |  o          \
  *      |              o
  *      ...[---].....[-].......>
  *
  *      The f(t) contributes to the
  *      non-nan integral values only
  *      for where it is defined.
  *      Also note that the left-hand side values at the end of
  *      intervals, or last point, is considered as nan.
  *
  *  Performance consideration/intentions:
  *  this algorithm will call
  *   ts.index_of() just once
  *   and ts.get(i) just once for each point i needed.
  *
  * \tparam TA is the time-axis type to compute values for
  *        The required signatures is:
  *
  *      #) .size() ->size_t
  *      #) .period(size_t i) -> utcperiod
  *
  * \tparam TS is the point-source that we are integrating/averaging
  *        The required signature is:
  *
  *      #) .size() ->size_t : number of points in the point source
  *      #) .index_of(utctime t)->size_t: left-hand index of point at t
  *      #) .get(size_t i) ->point: time,value of the i'th point
  *
  *
  * \param ta the time-axis to compute values for
  * \param ts the point source to use for computation
  * \param avg if true compute the average else compute integral
  * \return a vector<double>[ta.size()] with computed values, integral or true average
 */
template < class TA,class TS>
vector<double> accumulate_linear(const TA&ta, const TS& ts, bool avg) {
    vector<double> r(ta.size(),nan);
    // all good reasons for quitting early goes here:
    if(    ta.size()==0
        || ts.size() < 2 // needs two points, otherwise ->nan
        || ts.time(0) >= ta.total_period().end  // entirely after ->nan
        || ts.time(ts.size()-1) <= ta.total_period().start) // entirely before ->nan
        return r;
    // s= start point in our algorithm
    size_t s=ts.index_of(ta.period(0).start);
    if(s == string::npos) // ts might start after our time-axis begin
        s=0;// no prob. then we start where it begins.
    point s_p{ts.get(s)};
    bool s_finite{isfinite(s_p.v)};
    // e = end point for a partition in our algorithm
    size_t e{0};
    point e_p;
    bool e_finite{false};
    const size_t n=ts.size();

    double a{0};// the constants for line
    double b{0};// f(t) = a*t + b, computed only when needed

    for(size_t i=0;i<ta.size();++i) {
        double area{0.0}; // integral of non-nan f(x), area
        utctimespan t_sum{0};  // sum of non-nan time-axis
        const auto p {ta.period(i)};

        //---- find first finite point of a partition
        search_s_finite: // we loop here from advance_e_point if we hit a nan
            while(!s_finite) {
                ++s;
                if(s+1>=n) {// we are out of points searching for non-nan
                    if(t_sum)
                        r[i] = avg? area/t_sum: area;
                    return r;//-> we are completely done
                }
                s_p=ts.get(s); // we need only value here.. could optimize
                s_finite= isfinite(s_p.v);
            }
            // ok! here we got one finite point, possibly with one more after

        if(s_p.t >= p.end) { // are we out of this interval? skip to next ta.period
            if(t_sum)
                r[i] = avg? area/t_sum: area;// stash this result if any
            continue;
        }

        //---- find end-point of a partition
            if(e != s+1) { // only if needed!
        advance_e_point: // we loop into this point at the end of compute partition below
                e= s+1;// get next point from s
                if(e==n) {// we are at the end, and left-value of s is nan
                    if(t_sum)
                        r[i] = avg? area/t_sum: area;//stash result if any
                    return r;// -> we are completely done
                }
                e_p = ts.get(e);
                e_finite = isfinite(e_p.v);
                if(e_finite) {// yahoo! two points, we can then..
                    // compute equation for the line f(t) = a*t + b
                    // given points s_p and e_p
                    a = (e_p.v - s_p.v)/double(e_p.t - s_p.t);
                    b = s_p.v - a*double(s_p.t);
                } else {
                    s=e;// got nan, restart search s_finite
                    s_finite=false;//
                    goto search_s_finite;
                }
            }

        //compute_partition: we got a valid partition, defined by two points
            auto s_t = max(s_p.t,p.start);// clip to interval p
            auto e_t = min(e_p.t,p.end); // recall that the points can be anywhere
            // then compute non-nan area and non-nan t_sum
            utctimespan dt{e_t-s_t};
            area +=  (a*(s_t + e_t)*0.5 + b)*dt;// avg.value * dt
            t_sum += dt;
            if(e_p.t >= p.end) { // are we done in this time-step ?
                r[i] = avg? area/t_sum: area;// stash result
                continue; // using same s, as it could extend into next p
            }
            // else advance start to next point, that is; the current end-point
            s_p=e_p;
            s=e;
            goto advance_e_point;
    }
    return r;
}

template < class TA,class TS>
vector<double> accumulate_stair_case(const TA&ta, const TS& ts, bool avg) {
    vector<double> r(ta.size(),nan);
    // all good reasons for quitting early goes here:
    if(    ta.size()==0
        || ts.size() ==0 // needs at least one point, otherwise ->nan
        || ts.time(0) >= ta.total_period().end  // entirely after ->nan
        || ts.total_period().end <= ta.total_period().start) // entirely before ->nan
        return r;
    const utcperiod tp{ts.total_period()};
    // s= start point in our algorithm
    size_t s=ts.index_of(ta.period(0).start);
    if(s == string::npos) // ts might start after our time-axis begin
        s=0;// no prob. then we start where it begins.
    point s_p{ts.get(s)};
    bool s_finite{isfinite(s_p.v)};
    // e = end point for a partition in our algorithm
    point e_p;
    bool e_finite{false};
    const size_t n=ts.size();

    for(size_t i=0;i<ta.size();++i) {
        double area{0.0}; // integral of non-nan f(x), area
        utctimespan t_sum{0};  // sum of non-nan time-axis
        const auto p {ta.period(i)};

        //---- find first finite point of a partition
        search_s_finite:
            while(!s_finite) {
                ++s;
                if(s>=n) {// we are out of points searching for non-nan
                    if(t_sum)
                        r[i] = avg? area/t_sum: area;
                    return r;//-> we are completely done
                }
                s_p=ts.get(s); // we need only value here.. could optimize
                s_finite= isfinite(s_p.v);
            }
            // ok! here we got one finite point, possibly with one more after

        if(s_p.t >= p.end) { // are we out of this interval? skip to next ta.period
            if(t_sum)
                r[i] = avg? area/t_sum: area;// stash this result if any
            continue;
        }
        //---- find end-point of a partition
        find_partition_end:
            if(s+1<n) {
                e_p = ts.get(s+1);
                e_finite = isfinite(e_p.v);
            } else {
                e_p.t = tp.end;// total-period end
                e_finite=false;
            }
        //compute_partition: we got a valid partition, defined by two points
        auto s_t = max(s_p.t,p.start);// clip to interval p
        auto e_t = min(e_p.t,p.end); // recall that the points can be anywhere
        utctimespan dt{e_t-s_t};
        area +=  s_p.v*dt;
        t_sum += dt;
        if ( e_p.t <= p.end && s+1 <n) {// should&can we advance s
            s_p=e_p;
            s_finite=e_finite;
            ++s;
            if(e_p.t == p.end) {
                r[i] = avg? area/t_sum: area;// stash result
                continue;// skip to next interval
            }
            if(s_finite)
                goto find_partition_end;
            else
                goto search_s_finite;
        }
        // keep s, next interval.
        r[i] = avg? area/t_sum: area;// stash result

        if(s+1>=n && p.end >= tp.end)
           return r;// finito
    }
    return r;
}
}} // shyft.time_series
