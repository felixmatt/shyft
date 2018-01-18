#pragma once
#include <vector>
#include <algorithm>
#include <utility>

#include "time_series.h"
namespace shyft{namespace time_series {

/** \brief merge points from b into a 
 *
 * The result of the merge operation is the unique union set of time-points
 * with corresponding values from ts a or b, b has the priority for
 * equal time-points, replaces points in a.
 * 
 * The a.total_period() is extended to max of a and b
 * 
 * The function is assumed to be useful in the data-collection or
 * time-series point manipulation tasks
 * 
 */
template<class t_axis_a,class ts_b>    
void ts_point_merge (point_ts<t_axis_a>&a, const ts_b& b) {
    if(b.size()==0) return;//noop
    if(a.size()==0) { // -> assign
        a.ta=b.time_axis();
        a.v=b.values();
        a.fx_policy=b.point_interpretation();//kind of ok, practical?
    } else { // a straight forward, not neccessary optimal algorithm for merge:
        vector<utctime> t;t.reserve(a.size()+b.size());//assume worst case
        vector<double> v;v.reserve(a.size()+b.size());
        size_t ia=0;size_t ib=0;
        while(ia<a.size() && ib<b.size()) {
            auto ta=a.time(ia);
            auto tb=b.time(ib);
            if(ta==tb) { // b replaces value in a
                t.emplace_back(tb);
                v.emplace_back(b.value(ib));
                ++ia;++ib;
            } else if(ta<tb) { // a contribute with it's own point
                t.emplace_back(ta);
                v.emplace_back(a.value(ia));
                ++ia;
            } else { // b contribute with it's own point
                t.emplace_back(tb);
                v.emplace_back(b.value(ib));
                ++ib;
            }
        }
        // fill in remaining a or b points(one of them got empty of points above)
        while(ia<a.size()) {
            t.emplace_back(a.time(ia));
            v.emplace_back(a.value(ia++));
        }
        while(ib<b.size()) {
            t.emplace_back(b.time(ib));
            v.emplace_back(b.value(ib++));
        }
        a.v=move(v);
        utctime t_end = std::max(a.ta.total_period().end,b.time_axis().total_period().end);
        a.ta=time_axis::generic_dt(move(t),t_end);
    }            
};

}}
