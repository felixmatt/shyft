#pragma once
//assume core_pch.h is included, so no vector etc.
#include "utctime_utilities.h"


namespace shyft {
    namespace time_series {
        using namespace std;
        /** \brief the merge function helps converting parts of forecasts into one time-series
         *
         * forecast ts definition:
         *  a time-series (time-axis, values, point-intepretation) - plain ts in this context
         *
         * Given a ts-vector,
         * representing forecast time-series,
         *  requirement #1: ordered in time.
         *  requirement #2: ts.time(0) should be increasing by parameter dt
         *
         * E.g. met.no Arome forecast are generated every 6th hour,
         * starting at 00, 06, 12, 18 each lasting for 66 hours, with hourly data.
         * or ECWMF EC generated twice a day, starting at 00 and lasting for 240 hours of 1h..3h resolution
         *
         * The merge-algorithms picks a slice from each forecast in the vector,
         * starting at t0_offset time into the forecast, with length dt.
         *
         * if a forecast is missing, the algorithm try to
         * fix this extending the slice it
         * takes from each of time-series,
         * to form a mostly continuous time-series
         *
         * The returned time-series will have time-axis using time-points, where
         * each point exactly resemble an underlying point and value extracted from
         * the forecast time-series input
         *
         * \tparam ts_t return type of time-series, requirement to constructor(time-axis,value,fx_policy)
         * \tparam tsv_t vector type for forecasts
         * \param tsv an order by start-time vector with forecasts, equidistance at least as large as dt
         * \param t0_offset time-span to skip into each forecast
         * \param dt length of slice to pick from each forecast, as well as expected distance between each forecast t0
         * \return time-series of type ts_t, required to handle point-type (irregular) type of time-axis
         */
        template< class ts_t,class tsv_t>
        ts_t forecast_merge(tsv_t const& tsv, utctimespan t0_offset,utctimespan dt) {
            if(tsv.size()==0)
                return ts_t();
            typedef typename ts_t::ta_t ta_t;
            size_t n_estimate = tsv.size()*(1+dt/deltahours(1));
            vector<utctime> tv;tv.reserve(n_estimate);
            vector<double> vv;vv.reserve(n_estimate);
            auto t_previous_end=tsv.front().time(0)+t0_offset;
            for(size_t i=0;i<tsv.size();++i) {
                auto const& ts=tsv[i];
                // we want to extract ts [t0_offset,.. +. dt>
                // but there might be gap in front.. due to previous ts lacks data
                // .. and gap at the end because the next ts lacks data
                auto t_start = ts.time(0)+t0_offset; // t_start where we want to extract data
                size_t ix;
                if(t_start > t_previous_end) { // have to fill gap until the interesting slice starts?
                    ix=ts.index_of(t_previous_end);// may be ensure t_previous_end >tv.back()
                    if(ix==string::npos) // ok, the ts start AFTER t_previous_end, we are have to accept that
                        ix=0; // and start using the first point available
                    while(ts.time(ix)<t_start && ix <ts.size()) { // then fill the points up to t_start
                        tv.push_back(tsv[i].time(ix));vv.push_back(tsv[i].value(ix));++ix;
                    }
                } else { // ordinary case, this ts just fall exactly into place

                    if(t_start < t_previous_end) // one extra check,to keep sanity
                        t_start = t_previous_end; // force next start to continue after next
                    ix = ts.index_of(t_start);// guaranteed to give a valid index?
                    if(ix==string::npos) { // in case ts.time(0)+t0_offset is AFTER the last interval
                        if(ts.size()) // and there is some data
                            ix=ts.size()-1;// try to use the last value of the ts
                        else
                            continue; // just give up, no data to collect here
                    }
                }
                // if forecasts is missing, we might also need to fill *after* the slice
                // from this fc time-series, and we try to compute the amount extra here:
                utctimespan dt_extra(0);
                if(i+1<tsv.size()) {
                    if(tsv[i+1].size()) {
                        auto t_0_next=tsv[i+1].time(0); // pick next ts first time-point
                        dt_extra = std::max(utctimespan(0),t_0_next-(t_start+dt));//
                    }
                }
                auto t_end = t_start +  dt +dt_extra;// will be previous end *after* the next loop
                while(ts.time(ix) < t_end && ix < ts.size()) {
                    tv.push_back(ts.time(ix));vv.push_back(ts.value(ix));
                    ++ix;
                }
                t_previous_end=t_end;
            }
            return ts_t(ta_t(tv,t_previous_end),vv,tsv.front().point_interpretation());
        }

        /** \brief nash-sutcliffe evaluate slices from a set of forecast against observed
         *
         *  Given a ts-vector, and an observed ts, run nash-sutcliffe
         *  over each forecast, using specified slice specified as (lead-time, dt, n)
         *  return n.s (1-E)
         * \see shyft::time_series::nash_sutcliffe_goal_function
         */
        template< class ts_t,class tsv_t>
        double nash_sutcliffe(tsv_t const& tsv,ts_t const &obs, utctimespan t0_offset,utctimespan dt,size_t n) {
            using namespace std;
            if(tsv.size()==0 ||obs.size()==0)
                return shyft::nan;
            typedef typename ts_t::ta_t ta_t;
            vector<double> vs;vs.reserve(n*tsv.size());
            vector<double> vo;vo.reserve(n*tsv.size());
            for(auto const&ts:tsv) {
                ta_t ta(ts.time_axis().time(0) + t0_offset, dt, n);
                average_accessor<ts_t,ta_t> s_ts(ts,ta);
                average_accessor<ts_t,ta_t> o_ts(obs,ta);
                for(size_t i=0;i<ta.size();++i) {
                    vs.push_back(s_ts.value(i));
                    vo.push_back(o_ts.value(i));
                }
            }
            ta_t txa(0,dt,vs.size());// just a fake time-axis to resolve nash_sutcliffe_goal function
            ts_t ss(txa,vs,ts_point_fx::POINT_AVERAGE_VALUE);
            ts_t os(txa,vo,ts_point_fx::POINT_AVERAGE_VALUE);
            return 1.0-nash_sutcliffe_goal_function(os,ss); // 1-E -> n.s
        }

    }
}
