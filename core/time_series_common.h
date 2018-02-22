#pragma once
#include "utctime_utilities.h"
#include <cmath>
namespace shyft {
const double nan = std::numeric_limits<double>::quiet_NaN();

namespace time_series {

using core::utctime;
const double EPS=1e-12; ///< used some places for comparison to equality, \ref point

/** \brief simply a point identified by utctime t and value v */
struct point {
    utctime t;
    double v;
    point(utctime t=0, double v=0.0) : t(t), v(v) { /* Do nothing */}
};

/** \brief point a and b are considered equal if same time t and value-diff less than EPS
*/
inline bool operator==(const point &a,const point &b)  {return (a.t==b.t) && std::fabs(a.v-b.v)< EPS;}


}}// shyft.time_series
