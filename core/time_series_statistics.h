#pragma once
#ifdef SHYFT_NO_PCH
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <future>
#include <utility>

#endif // SHYFT_NO_PCH

#include "compiler_compatiblity.h"
#include "utctime_utilities.h"
#include "time_axis.h"
#include "time_series.h"

namespace shyft {
    namespace time_series {
        using namespace std;
        using namespace shyft;

        /** specialized max function that ignores nan*/
        inline double nan_max(double const& r, double const& x) {
            if (!isfinite(x))
                return r;
            if (!isfinite(r))
                return x;
            return std::max(r, x);
        }
        /** specialized min function that ignores nan*/
        inline double nan_min(double const&r, double const& x) {
            if (!isfinite(x))
                return r;
            if (!isfinite(r))
                return x;
            return std::min(r, x);
        }

        template <class Ts, class Ta, typename Fx
            //, typename = enable_if_t<is_ts<Ts>::value>
        >
        inline vector<double> extract_statistics(Ts const&ts,Ta const&ta, Fx&& fx) {
            auto ix_map = time_axis::make_time_axis_map(ts.time_axis(), ta);
            size_t is_max = ts.size();//optimize out the end index here
            vector<double> r;r.reserve(ta.size());
            size_t is = ix_map.src_index(0);
            for (size_t i = 0; i < ta.size(); ++i) {
                auto p = ta.period(i);
                double rv = nan;
                if (is == string::npos) { //in the beginning of interval, nothing
                    ;// maybe check if ts.time(0) is within interval, then work the way through
                    if (p.contains(ts.time(0))) {
                        is = 0; //proceed as normal in this interval
                    } else {
                        r.push_back(rv);//emit result for this interval and go to next
                        continue;
                    }
                }
                // process all values relevant for this interval,
                // the first value could be lhs value of the interval (could be interesting)
                // the next value could be in the interval or first on rhs(exit condition)
                if (is < is_max && ts.time(is) < p.start) {
                    //point is left of current interval
                    ++is;//now we are: a) inside interval, b) after interval or end
                }
                while (is < is_max && ts.time(is) < p.end) {
                    rv = fx(rv, ts.value(is));
                    ++is;//advance is, next could be in a) interval, or first right of interval
                }
                r.push_back(rv);// we are at end of is
            }
            return r;
        }

        template <class Ts, class Ta, typename Fx>
        inline vector<double> extract_statistic_from_vector(std::vector<Ts> const&ts_list, Ta const&ta, Fx&& fx) {
            std::vector<double> v_x;
            for (size_t i = 0;i < ts_list.size();++i) {
                if (i == 0) {
                    v_x = extract_statistics(ts_list[i], ta, forward<Fx>(fx));
                } else {
                    auto i_x = extract_statistics(ts_list[i], ta, forward<Fx>(fx));
                    for (size_t j = 0;j < i_x.size();++j)
                        v_x[j] = fx(v_x[j], i_x[j]);
                }
            }
            return v_x;
        }

        template <class Ts, class Ta, typename = enable_if_t<is_ts<Ts>::value>>
        struct statistics {
            Ts ts;
            Ta ta;
            template <class Ts_, class Ta_>
            statistics(Ts_&&tsx, Ta_&tax) :ts(forward<Ts_>(tsx)), ta(forward<Ta_>(tax)) {}
            template <class Ts_>
            explicit statistics(Ts_&&tsx) : ts(forward<Ts_>(tsx)) {
                ta = ts.time_axis();
            }
            template <typename Fx>
            vector<double> extract(Fx&& fx)const {
                return extract_statistics(ts, ta, fx);
            }
        };
        enum statistics_property {
            AVERAGE=-1,
            MIN_EXTREME=-1000,
            MAX_EXTREME=+1000
        };
        /// http://en.wikipedia.org/wiki/Percentile NIST definitions, we use R7, as R and excel
        /// http://www.itl.nist.gov/div898/handbook/prc/section2/prc262.htm
        /// calculate percentile using full sort.. works nice for a larger set of percentiles.
        inline vector<double> calculate_percentiles_excel_method_full_sort(vector<double>& samples, const vector<int>& percentiles) {
            vector<double> result; result.reserve(percentiles.size());
            const int n_samples = (int)samples.size();
            const double silent_nan = std::numeric_limits<double>::quiet_NaN();
            if (n_samples == 0) {
                for (size_t i = 0; i < percentiles.size(); ++i)
                    result.emplace_back(silent_nan);
            } else {
                //TODO: filter out Nans
                sort(begin(samples), end(samples));
                for (auto i : percentiles) {
                    // use NIST definition for percentile
                    if (i == statistics_property::AVERAGE ) { // hack: -1,  aka. the mean value..
                        double sum = 0; int n = 0;
                        for (auto x : samples) {
                            if (std::isfinite(x)) { sum += x; ++n; }
                        }
                        result.emplace_back(n > 0 ? sum / n : silent_nan);
                    } else if (i >= 0 && i <= 100) {
                        const double eps = 1e-30;
                        // use Hyndman and fam R7 definition, excel, R, and python
                        double nd = 1.0 + (n_samples - 1)*double(i) / 100.0;
                        int  n = int(nd);
                        double delta = nd - n;
                        --n;//0 based index
                        if (n <= 0 && delta <= eps) result.emplace_back(samples.front());
                        else if (n >= n_samples) result.emplace_back(samples.back());
                        else {

                            if (delta < eps) { //direct hit on the index, use just one.
                                result.emplace_back(samples[n]);
                            } else { // in-between two samples, use positional weight
                                auto lower = samples[n];
                                if (n < n_samples - 1)
                                    n++;
                                auto upper = samples[n];
                                result.emplace_back(lower + (delta)*(upper - lower));
                            }
                        }
                    } else {
                        result.emplace_back(silent_nan);//some other statistics property we don't compute here
                    }
                }
            }
            return result;
        }

        /** \brief calculate specified percentiles for supplied list of time-series over the specified time-axis

        Percentiles for a set of timeseries, over a time-axis
        we would like to :
        percentiles_timeseries = calculate_percentiles(ts-id-list,time-axis,percentiles={0,25,50,100})
        done like this
        1..m ts-id, time-axis( start,dt, n),
        read 1..m ts into memory
        percentiles specified is np
        result is percentiles_timeseries
        accessor "accumulate" on time-axis to dt, using stair-case or linear between points
        create result vector[1..n] (the time-axis dimension)
        where each element is vector[1..np] (for each timestep, we get the percentiles
         for each timestep_i in time-axis
          for each tsa_i: accessors(time-axis,ts)
            sample vector[timestep_i].emplace_back( tsa_i(time_step_i) ) (possibly skipping nans if selected)
            percentiles_timeseries[timestep_i]= calculate_percentiles(..)


        \return percentiles_timeseries

        */
        template <class ts_t, class ta_t>
        inline std::vector< point_ts<ta_t> > calculate_percentiles(const ta_t& ta, const std::vector<ts_t>& ts_list, const std::vector<int>& percentiles, size_t min_t_steps = 1000,bool skip_nans=true) {
            std::vector<point_ts<ta_t>> result;
            auto fx_p = ts_list.size() ? ts_list.front().point_interpretation() : ts_point_fx::POINT_AVERAGE_VALUE;
            for (size_t r = 0; r < percentiles.size(); ++r) // pre-init the result ts that we are going to fill up
                result.emplace_back(ta, 0.0, fx_p);

            auto partition_calc = [&result, &ts_list, &ta, &percentiles,skip_nans](size_t i0, size_t n) {

                std::vector < average_accessor<ts_t, ta_t>> tsa_list; tsa_list.reserve(ts_list.size());
                for (const auto& ts : ts_list) // initialize the ts accessors to we can accumulate to time-axis ta e.g.(hour->day)
                    tsa_list.emplace_back(ts, ta);

                std::vector<double> samples;samples.reserve(tsa_list.size());

                for (size_t t = i0; t < i0 + n; ++t) {//each time step t in the time-axis, here we could do parallel partition
                    samples.clear();
                    for (size_t i = 0; i < tsa_list.size(); ++i) { // get samples from all the "tsa"
                        auto v=tsa_list[i].value(t);
                        if(!skip_nans || isfinite(v))
                            samples.emplace_back(v);
                    }
                    // possible with pipe-line to percentile calc here !
                    std::vector<double> percentiles_at_t(calculate_percentiles_excel_method_full_sort(samples, percentiles));
                    for (size_t p = 0; p < result.size(); ++p) {
                        if(!(percentiles[p]==statistics_property::MAX_EXTREME || percentiles[p]==statistics_property::MIN_EXTREME))
                            result[p].set(t, percentiles_at_t[p]);
                    }
                }
            };
            auto extreme_calc = [&result, &ts_list, &ta, &percentiles](size_t x) {
                result[x].v = extract_statistic_from_vector(ts_list, ta, percentiles[x] == statistics_property::MIN_EXTREME?nan_min:nan_max);
            };

            if (ta.size() < min_t_steps) {
                partition_calc(0, ta.size());
                //if mi-ma extreme calc, do it here
                for (size_t i = 0;i < percentiles.size();++i) {
                    if (percentiles[i] == statistics_property::MIN_EXTREME) {// min-extremes
                        result[i].v = extract_statistic_from_vector(ts_list, ta, nan_min);
                    } else if (percentiles[i] == statistics_property::MAX_EXTREME) {// max-extremes
                        result[i].v = extract_statistic_from_vector(ts_list, ta, nan_max);
                    }
                }
            } else {
                vector<future<void>> calcs;
                for (size_t p = 0;p < ta.size(); ) {
                    size_t np = p + min_t_steps <= ta.size() ? min_t_steps : ta.size() - p;
                    calcs.push_back(std::async(std::launch::async, partition_calc, p, np));
                    p += np;
                }

                for (size_t i = 0;i < percentiles.size();++i) {
                    if (percentiles[i] == statistics_property::MIN_EXTREME || percentiles[i] == statistics_property::MAX_EXTREME)
                        calcs.push_back(std::async(std::launch::async,extreme_calc,i));
                }
                for (auto &f : calcs)
                    f.get();

            }

            return result;
        }
    }
}
