#include "test_pch.h"
#define _USE_MATH_DEFINES
#include "core/time_series.h"
#include "core/time_axis.h"
#include "core/utctime_utilities.h"
//#include "api/api.h"
//#include "api/time_series.h"
//#include "core/time_series_statistics.h"

// workbench for qm data model and algorithms goes into shyft::qm
// promoted to core/qm.h when done

namespace shyft {
    namespace qm {
        using namespace std;

        /**\brief quantile_index generates a pr. time-step index for order by value
         * \tparam tsa_t time-series accessor type that fits to tsv_t::value_type and ta_t, thread-safe fast access to the value aspect
         *               of the time-series for each period in the specified time-axis
         *               requirement to the type is:
         *               constructor tsa_t(ts,ta) including just enough stuff for trivial copy-ct, move-ct etc.
         *                tsa_t.value(size_t i) -> double gives the i'th time-axis period value of ts (cache/lookup if needed)
         *
         * \tparam tsv_t time-series vector type
         * \tparam ta_t time-axis type, require .size() (the tsa_t hides other methods required)
         * \param tsv time-series vector that keeps .size() time-series
         * \param ta time-axis
         * \return vector<vector<int>> that have dimension [ta.size()][tsv.size()] holding the indexes of the by-value in time-step ordered list
         *
         */
        template <class tsa_t,class tsv_t,class ta_t>
        vector<vector<int>> quantile_index(tsv_t const &tsv,ta_t const &ta ) {
            vector < tsa_t> tsa; tsa.reserve(tsv.size()); // create ts-accessors to enable fast thread-safe access to ts value aspect
            for (const auto& ts : tsv) tsa.emplace_back(ts, ta, shyft::time_series::extension_policy::USE_NAN);

            vector<vector<int>> qi(ta.size()); // result vector, qi[i] -> a vector of index-order tsv[i].value(t)
            vector<int> pi(tsv.size());for(size_t i=0;i<tsv.size();++i) pi[i]=i;//initial order 0..n-1, and we re-use it in the loop
            for(size_t t=0;t<ta.size();++t) {
                sort(begin(pi),end(pi),[&tsa,t](int a,int b)->int {return tsa[a].value(t)<tsa[b].value(t);});
                // Filter away the nans, which signal that we have time series that don't extend this far
                vector<int> final_inds;
                for (size_t i=0; i<tsa.size(); ++i) {
                    if (isfinite(tsa[i].value(t))) {
                        final_inds.emplace_back(pi[i]);
                    }
                }
                qi[t]=final_inds; // it's a copy assignment, ok, since sort above will start out with previous time-step order
            }
            return qi;
        }


        /** \brief compute quantile-values based on weigh_value ordered (wvo) items
         *
         *
         * \note The requirement to the wvo_accessor
         *  (1) sum weights == 1.0
         *  (2) weight(i), value(i) are already ordered by value asc
         *   are the callers responsibility
         *   no checks/asserts are done
         *
         *  TODO: we might want to rework the output of this algorithm
         *        so that it streams/put data directly into the
         *        target-time-series (instead of building a vector
         *        and then spread that across time-series)
         *
         *
         * \tparam WVO Weight Value Ordered accessor, see also wvo_accessor
         *  .size() -> number of weight,value items
         *  .value(i)-> double the i'th value
         *  .weight(i)-> double the i'th weight
         * \param n_q number of quantiles,>1, evenly distributed over 0..1.0
         * \param items the weighted items, weight,value, ordered asc value by i'th index,
         *        assuming that sum weights = 1.0
         * \return vector<double> of size n_q, where the i'th value is the i'th computed quantile
         */
        template <class WVO> // to-consider: maybe just iterators to weight-value pairs ?
        vector<double> compute_weighted_quantiles(size_t n_q,WVO const& wv ) {
            const double q_step= 1.0/(n_q - 1); // iteration is 0-based..
            vector<double> q;q.reserve(n_q);
            // tip: 'j' denotes the 'source' quantile material, 'i' denotes the wanted quantiles
            size_t j=0;// the j'th item in wv
            double q_j=0.0;// since the 'weights' are normalized, each weight represent the length of quantile-segment the value is valid for
            double w_j=wv.weight(j);// We try to have only one access for each indexed element
            double v_j=wv.value(j); // (we *could* trust the compiler optimizer here)
            for(size_t i=0;i<n_q;++i) { // ensure we are precise on number of quantiles we provide
                double q_i = q_step*i;  // multiplication or addition, almost same performance(?)
                while(q_j + w_j  < q_i ) { //  as long as current q_j+w_j are LESS than target q_i
                    q_j += w_j;            // climb along the 'j' quantile axis until we reach wanted level
                    if( j + 1 < wv.size()) {  // only get next element if there are more left
                        w_j=wv.weight(++j);
                        v_j=wv.value(j);
                    }
                }
                q.emplace_back(v_j);// note: we *could* do weighted interpolation q_j < q_i <q_j+1, but we do simple rank here
            }
            return q;
        }

        /**\brief a Weight Value Ordered collection for ts-vector type
         *
         * This type is a stack-context only, light weight wrapper,
         *  for ts-vector type, to be feed into the
         * computed_weighted_quantiles algorithm.
         *
         * The idea here is to give a class that is useful for
         * anything that resembles typical shyft time-series or time-series accessor
         * Note that we hide the mechanism for how 'things' are sorted
         * keeping the indirect-indexing array an internal detail away from the
         * compute_weighted_quantile function.
         *
         * \see computed_weighted_quantiles
         */
        template <class tsa_t>
        struct wvo_accessor {
            vector<vector<int>>const& ordered_ix; ///<index ordering of the i'th timestep
            vector<double>const & w;///< weights of each ts - assumed to be same for all timesteps
            vector<tsa_t>const& tsv;///< access to .value(t), where t is size_t
            size_t t_ix=0;///< current time step index
            vector<double> w_sum;///< sum of weights per timestep so we can return normalized .weight(i). Note that it must be per timestep even if we have the same weights for all timestep - this is because ordered_idx can contain varying length-vectors of indices, since certain time series might be invalid at certain times.
            //-------------------
            wvo_accessor(vector<vector<int>> const&ordered_ix,vector<double> const&w,vector<tsa_t> const &tsv)
            :ordered_ix(ordered_ix),w(w),tsv(tsv) 
            {
                for (size_t t=0; t<ordered_ix.size(); ++t) {
                    double currsum = 0.0;
                    for (size_t i=0; i<ordered_ix[t].size(); ++i) {
                        currsum += w[ordered_ix[t][i]];
                    }
                    w_sum.emplace_back(currsum);
                }
            }

            //-- here goes the template contract requirements
            size_t size() const { return tsv.size();}
            double weight(size_t i ) const {return w[ordered_ix[t_ix][i]]/w_sum[t_ix];}
            double value(size_t i ) const { return tsv[ordered_ix[t_ix][i]].value(t_ix);}
        };

        /**\brief The main quantile mapping function, which, using quantile
         * calculations, maps the values of the weighted 'prognosis' time
         * series vectors onto the 'prior' time series. This mapping is done
         * for each time step in the specified time axis.
         * \tparam tsa_t The time-series accessor type.
         * \tparam tsv_t The time-series vector type.
         * \tparam ta_t The time-axis type.
         * \param pri_tsv The time-series vector containing the 'prior'
         *      observations. The values in this vector will be more or less
         *      overwritten by the values in prog_tsv that, using the quantile
         *      mapping, correspond to the values in pri_tsv.
         * \param prog_tsv The time-series vector containing the 'prognosis'.
         *      The values in this vector will overwrite the corresponding ones
         *      (in the quantile mapping sense) in pri_tsv.
         * \param pri_idx_v The vector containing the indices that will order
         *      the observations in pri_tsv according to their value, so that
         *      if the current time axis index is t, then pri_idx_v[t] will
         *      give a vector sorting pri_tsv for that timestep.
         * \param prog_idx_v Same as pri_idx_v, but applies to prog_tsv.
         * \param prog_weights Contains the weights to apply to each
         *      value in prog_tsv. Assumed to be the same for all timesteps.
         * \param time_axis The time axis over which to perform the mapping.
         * \param interpolation_start The start of the period over which we
         *      want to interpolate between the prognosis and prior. We assume
         *      that the end of the interpolation period coincides with the
         *      last point of time_axis. For all times before
         *      interpolation_start, the corresponding value in pri_tsv will
         *      only contain the corresponding quantile-mapped value in
         *      prog_tsv, while for times after interpolation_start, the value
         *      in pri_tsv will contain a linearly interpolated mix of the
         *      value already in pri_tsv and the corresponding quantile-mapped
         *      value in prog_tsv. If this parameter is set to
         *      core::no_utctime, interpolation will not take place.
         * \return tsv_t Containing the quantile mapped values at the times
         *      indicated by time_axis, and interpolated after
         *      interpolation_start.
         **/
        template <class tsa_t, class tsv_t, class ta_t>
        tsv_t quantile_mapping(tsv_t const &pri_tsv, tsv_t const &prog_tsv,
                               vector<vector<int>> const &pri_idx_v,
                               vector<vector<int>> const &prog_idx_v,
                               vector<double> const &prog_weights,
                               ta_t const &time_axis,
                               core::utctime const &interpolation_start) {
            vector<tsa_t> pri_accessor_vec;
            pri_accessor_vec.reserve(pri_tsv.size());
            for (const auto& ts : pri_tsv) 
                pri_accessor_vec.emplace_back(ts, time_axis);

            core::utcperiod interpolation_period;
            if (core::is_valid(interpolation_start)) {
                interpolation_period = core::utcperiod(
                        interpolation_start,
                        time_axis.time(time_axis.size()-1)
                        );
            } else {
                interpolation_period = core::utcperiod();
            }
            auto wvo_prog = wvo_accessor<typename tsv_t::value_type>(prog_idx_v,
                    prog_weights, prog_tsv);


            tsv_t output;
            output.reserve(pri_tsv.size());
            for (size_t i=0; i<pri_tsv.size(); ++i) {
                output.emplace_back(time_axis, nan,
                        pri_tsv[i].point_interpretation());
            }
            for (size_t t=0; t<time_axis.size(); ++t) {
                wvo_prog.t_ix = t;
                size_t num_pri_cases = pri_idx_v[t].size();
                vector<double> quantile_vals = compute_weighted_quantiles(
                        num_pri_cases,
                        wvo_prog);
                double interp_weight = 0.0;
                if (interpolation_period.valid() && 
                        (interpolation_period.contains(time_axis.time(t)) ||
                         interpolation_period.end == time_axis.time(t))) {
                    core::utctime start = interpolation_period.start;
                    core::utctime end = interpolation_period.end;
                    interp_weight = (
                            static_cast<double>(time_axis.time(t) - start) /
                            (end - start));
                }

                for (size_t i=0; i<num_pri_cases; ++i) {
                    double setval;
                    if (interp_weight != 0.0) {
                        setval = ((1.0 - interp_weight) * quantile_vals[i] +
                                  interp_weight * 
                                  pri_accessor_vec[pri_idx_v[t][i]].value(t));
                    } else {
                        setval = quantile_vals[i];
                    }
                    output[pri_idx_v[t][i]].set(t, setval);
                }
            }

            return output;
        }

        template <class tsa_t, class tsv_t, class ta_t>
        tsv_t quantile_map_main(vector<tsv_t> const &forecast_sets, 
                vector<double> const &set_weights, tsv_t const &historical_data,
                ta_t const &time_axis, 
                core::utctime const &interpolation_start) {
            tsv_t forecasts_unpacked;
            vector<double> weights_unpacked;
            for (size_t i=0; i<forecast_sets.size(); ++i) {
                forecasts_unpacked.reserve(forecasts_unpacked.size() +
                        forecast_sets[i].size());
                weights_unpacked.reserve(weights_unpacked.size() + 
                        forecast_sets[i].size());
                for (size_t j=0; j<forecast_sets[i].size(); ++j) {
                    forecasts_unpacked.emplace_back(forecast_sets[i][j]);
                    weights_unpacked.emplace_back(set_weights[i]);
                }
            }

            auto historical_indices = quantile_index<tsa_t>(historical_data,
                    time_axis);
            auto forecast_indices = quantile_index<tsa_t>(forecasts_unpacked,
                    time_axis);

            return quantile_mapping<tsa_t>(historical_data, forecasts_unpacked,
                    historical_indices, forecast_indices, weights_unpacked,
                    time_axis, interpolation_start);
        }
    }
}


using namespace shyft;
using namespace std;
using ta_t = time_axis::fixed_dt;
using ts_t = time_series::point_ts<ta_t>;
using tsa_t = time_series::average_accessor<ts_t,ta_t>;
using tsv_t = std::vector<ts_t>;

TEST_SUITE("qm") {
    TEST_CASE("ts_vector_to_quantile_ix_list") { // to run this test: test_shyft -tc=ts_vector_to_quantile_ix_list
        // standard triple A testing

        // #1:Arrange
        size_t n_prior=100;
        core::calendar utc;
        ta_t ta(utc.time(2017,1,1,0,0,0),core::deltahours(24),14);
        tsv_t prior;
        for(size_t i=0;i<n_prior;++i) {
            vector<double> v;
            for(size_t t=0;t<ta.size();++t)
                v.push_back(double((i+t)%n_prior));
            prior.emplace_back(ta,v,time_series::ts_point_fx::POINT_AVERAGE_VALUE);
        }

        // #2: Act
        auto qi=qm::quantile_index<tsa_t>(prior,ta);

        // #3: Assert
        FAST_REQUIRE_EQ(qi.size(),ta.size());
        for(size_t t=0;t<ta.size();++t) {
            FAST_REQUIRE_EQ(qi[t].size(),prior.size());//each time-step, have indexes for prior.size() (ordered) index entries
            for(size_t i=0;i<qi[t].size()-1;++i) {
                double v1 = prior[qi[t][i]].value(t);
                double v2 = prior[qi[t][i+1]].value(t);
                FAST_CHECK_LE(v1 ,v2 ); // just ensure all values are ordered in ascending order, based on index-lookup and original ts.
            }
        }
    }

    TEST_CASE("compute_weighted_quantiles") {
        // Arrange
        const auto fx_avg= time_series::ts_point_fx::POINT_AVERAGE_VALUE;
        size_t n_quantiles = 100;
        core::calendar utc;
        ta_t ta(utc.time(2017, 1, 1, 0, 0, 0), core::deltahours(24), 2);
        //-----------------
        vector<double> weights;//weight of each forecast below,two with equal weights
        tsv_t fcv;// forecast-vector, there are 4 of them in this case, with two values each
        weights.push_back(5.0);fcv.emplace_back(ta,vector<double>{14.2,  4.1},fx_avg);
        weights.push_back(2.0);fcv.emplace_back(ta,vector<double>{13.0,  2.9},fx_avg);
        weights.push_back(2.0);fcv.emplace_back(ta,vector<double>{15.8, 20.4},fx_avg);
        weights.push_back(1.0);fcv.emplace_back(ta,vector<double>{ 9.1, 11.2},fx_avg);

        auto q_order = qm::quantile_index<tsa_t>(fcv,ta); // here we use already tested function to get ordering, all steps
        qm::wvo_accessor<ts_t> wvo(q_order,weights,fcv);// stash it into wvo_accessor, so we are ready to go

        // Act
        vector<double> q0 = qm::compute_weighted_quantiles(n_quantiles,wvo);
        wvo.t_ix++; // increment to next time-step
        vector<double> q1 = qm::compute_weighted_quantiles(n_quantiles,wvo);

        // Assert
        // to consider: could we select values etc. to make a data-driven approach?
        FAST_REQUIRE_EQ(q0.size(), n_quantiles);
        for (size_t i=0; i<n_quantiles; ++i) {
            if (i <= 9) {
                FAST_CHECK_EQ(q0[i], 9.1);
            } else if (i <= 29) {
                FAST_CHECK_EQ(q0[i], 13.0);
            } else if (i <= 79) {
                FAST_CHECK_EQ(q0[i], 14.2);
            } else {
                FAST_CHECK_EQ(q0[i], 15.8);
            }
        }
        FAST_REQUIRE_EQ(q1.size(), n_quantiles);
        for (size_t i=0; i<n_quantiles; ++i) {
            if (i <= 19) {
                FAST_CHECK_EQ(q1[i], 2.9);
            } else if (i <= 69) {
                FAST_CHECK_EQ(q1[i], 4.1);
            } else if (i <= 79) {
                FAST_CHECK_EQ(q1[i], 11.2);
            } else {
                FAST_CHECK_EQ(q1[i], 20.4);
            }
        }
    }

    TEST_CASE("quantile_mapping") {

        const auto fx_avg = time_series::ts_point_fx::POINT_AVERAGE_VALUE;
        // Arrange
        core::calendar utc;
        ta_t ta(utc.time(2017, 1, 1, 0, 0, 0), core::deltahours(24), 2);
        tsv_t prior_ts_v;
        tsv_t forecast_ts_v;

        vector<double> weights;//weight of each forecast below,two with equal weights
        weights.push_back(39.0); forecast_ts_v.emplace_back(ta,vector<double>{32.1,  1.2},fx_avg);
        weights.push_back(32.0); forecast_ts_v.emplace_back(ta,vector<double>{21.0,  34.2},fx_avg);
        weights.push_back(8.0); forecast_ts_v.emplace_back(ta,vector<double>{ 10.2, 12.4},fx_avg);
        weights.push_back(73.0); forecast_ts_v.emplace_back(ta,vector<double>{ 71.0, 89.2},fx_avg);
        weights.push_back(14.0); forecast_ts_v.emplace_back(ta,vector<double>{ 35.4, 83.4},fx_avg);
        auto q_order = qm::quantile_index<tsa_t>(forecast_ts_v, ta); // here we use already tested function to get ordering, all steps

        // We now just create a random prior time series vector. The values
        // that we put in do not matter, since they will be overwritten if
        // everything works. However, the ordering matters when we are going to
        // compare so we have to keep track of the ordering.

        size_t num_priors = 43;
        for (size_t i=0; i<num_priors; ++i) {
            // Generate random no between 0 and 50
            vector<double> insertvec { static_cast<double>(std::rand())/RAND_MAX * 50.0,
                                       static_cast<double>(std::rand())/RAND_MAX * 50.0 };
            prior_ts_v.emplace_back(ta, insertvec, fx_avg);
        }

        auto pri_q_order = qm::quantile_index<tsa_t>(prior_ts_v, ta);

        // We're not testing interpolation here, so just make a start time that
        // will give a non-valid interpolation period
        core::utctime interp_start(core::no_utctime);

        // Act
        auto result = qm::quantile_mapping<tsa_t>(prior_ts_v, forecast_ts_v,
                pri_q_order, q_order, weights, ta, interp_start);

        // Assert
        for (size_t i=0; i<num_priors; ++i) {
            if (i < 3) {
                FAST_CHECK_EQ(result[pri_q_order[0][i]].value(0), 10.2);
            } else if (i < 11) {
                FAST_CHECK_EQ(result[pri_q_order[0][i]].value(0), 21.0);
            } else if (i < 20) {
                FAST_CHECK_EQ(result[pri_q_order[0][i]].value(0), 32.1);
            } else if (i < 24) {
                FAST_CHECK_EQ(result[pri_q_order[0][i]].value(0), 35.4);
            } else {
                FAST_CHECK_EQ(result[pri_q_order[0][i]].value(0), 71.0);
            }

            if (i < 10) {
                FAST_CHECK_EQ(result[pri_q_order[1][i]].value(1), 1.2);
            } else if (i < 12) {
                FAST_CHECK_EQ(result[pri_q_order[1][i]].value(1), 12.4);
            } else if (i < 20) {
                FAST_CHECK_EQ(result[pri_q_order[1][i]].value(1), 34.2);
            } else if (i < 24) {
                FAST_CHECK_EQ(result[pri_q_order[1][i]].value(1), 83.4);
            } else {
                FAST_CHECK_EQ(result[pri_q_order[1][i]].value(1), 89.2);
            }
        }
    }

    TEST_CASE("quantile_mapping_interp") {
        const auto fx_avg = time_series::ts_point_fx::POINT_AVERAGE_VALUE;
        //Arrange

        // This time we want a longer time series so that we can examine the interpolation
        core::calendar utc;
        ta_t ta(utc.time(2017, 1, 1, 0, 0, 0), core::deltahours(24), 14);
        tsv_t prior_ts_v;
        tsv_t forecast_ts_v;
        core::utctime interp_start(utc.time(2017, 1, 10, 0, 0, 0));
        vector<double> weights;
        // We make only two time series for the forecast, with unity weights.
        for (size_t i=0; i<2; ++i) {
            weights.push_back(1.0);
            vector<double> currvals;
            for (size_t j=0; j<14; ++j) {
                if (i == 0) {
                    currvals.emplace_back(5.0);
                } else if (i == 1) {
                    currvals.emplace_back(20.0);
                }
            }
            forecast_ts_v.emplace_back(ta, currvals, fx_avg);
        }
        auto q_order = qm::quantile_index<tsa_t>(forecast_ts_v, ta);

        size_t num_priors = 18;
        for (size_t i=0; i<num_priors; ++i) {
            vector<double> insertvec;
            for (size_t t=0; t<14; ++t) {
                // Just give the prior the same value as its index
                insertvec.emplace_back(i);
            }
            prior_ts_v.emplace_back(ta, insertvec, fx_avg);
        }
        auto pri_q_order = qm::quantile_index<tsa_t>(prior_ts_v, ta);

        // Act
        auto result = qm::quantile_mapping<tsa_t>(prior_ts_v, forecast_ts_v,
                pri_q_order, q_order, weights, ta, interp_start);

        // Assert
        for (size_t i=0; i<num_priors; ++i) {
            for (size_t t=0; t<14; ++t) {
                // The first half of the priors should have been mapped to 5,
                // and interpolated between 5 and i during the interpolation
                // period. For the second half, the value is 20.
                if (i < num_priors / 2) {
                    if (t < 9) {
                        FAST_CHECK_EQ(result[pri_q_order[t][i]].value(t),
                                      5.0);
                    } else {
                        double weight = (t - 9) / 4.0;
                        FAST_CHECK_EQ(result[pri_q_order[t][i]].value(t),
                                      ((1.0 - weight) * 5.0 + weight * i));
                    }
                } else {
                    if (t < 9) {
                        FAST_CHECK_EQ(result[pri_q_order[t][i]].value(t),
                                      20.0);
                    } else {
                        double weight = (t - 9) / 4.0;
                        FAST_CHECK_EQ(result[pri_q_order[t][i]].value(t),
                                      ((1.0 - weight) * 20.0 + weight * i));
                    }
                }
            }
        }
    }

    TEST_CASE("qm_partial_prognoses") {

        // Test for the cases where we have certain prognoses that don't
        // extend to the whole mapping period. We'll make three prognoses
        // series, with varying lengths.


        //Arrange
        const auto fx_avg = time_series::ts_point_fx::POINT_AVERAGE_VALUE;
        core::calendar utc;
        tsv_t prior_ts_v;
        tsv_t forecast_ts_v;
        //We don't want to interpolate for this one
        core::utctime interp_start(core::no_utctime);
        vector<double> weights;
        // we make three series
        for (size_t i=0; i<3; ++i) {
            vector<double> currvals;
            ta_t ta;
            weights.emplace_back(1.0);
            if (i == 0) {
                ta = ta_t(utc.time(2017, 1, 1, 0, 0, 0), core::deltahours(24), 2);
                currvals = {1.0, 1.0};
            } else if (i == 1) {
                ta = ta_t(utc.time(2017, 1, 1, 0, 0, 0), core::deltahours(24), 3);
                currvals = {2.0, 2.0, 2.0};
            } else if (i == 2) {
                ta = ta_t(utc.time(2017, 1, 1, 0, 0, 0), core::deltahours(24), 4);
                currvals = {3.0, 3.0, 3.0, 3.0};
            }
            forecast_ts_v.emplace_back(ta, currvals, fx_avg);
        }

        ta_t ta(utc.time(2017, 1, 1, 0, 0, 0), core::deltahours(24), 4);
        auto q_order = qm::quantile_index<tsa_t>(forecast_ts_v, ta);

        // Create the prior series. These will all have four data points
        for (size_t i=0; i<9; ++i) {
            vector<double> currvals;
            for (size_t t=0; t<4; ++t) {
                currvals.emplace_back(i);
            }
            prior_ts_v.emplace_back(ta, currvals, fx_avg);
        }

        auto pri_q_order = qm::quantile_index<tsa_t>(prior_ts_v, ta);

        //Act
        auto result = qm::quantile_mapping<tsa_t>(prior_ts_v, forecast_ts_v,
                pri_q_order, q_order, weights, ta, interp_start);

        //Assert
        for (size_t i=0; i<9; ++i) {
            for (size_t t=0; t<3; ++t) {
                if (t < 2) {
                    if (i < 3) {
                        FAST_CHECK_EQ(result[pri_q_order[t][i]].value(t), 1.0);
                    } else if (i < 6) {
                        FAST_CHECK_EQ(result[pri_q_order[t][i]].value(t), 2.0);
                    } else if (i < 9) {
                        FAST_CHECK_EQ(result[pri_q_order[t][i]].value(t), 3.0);
                    }
                } else if (t < 3) {
                    if (i < 5) {
                        FAST_CHECK_EQ(result[pri_q_order[t][i]].value(t), 2.0);
                    } else if (i < 9) {
                        FAST_CHECK_EQ(result[pri_q_order[t][i]].value(t), 3.0);
                    }
                } else {
                    FAST_CHECK_EQ(result[pri_q_order[t][i]].value(t), 3.0);
                }
            }
        }
    }

    TEST_CASE("qm_main") {
        
        //Arrange
        const auto fx_avg = time_series::ts_point_fx::POINT_AVERAGE_VALUE;
        core::calendar utc;
        ta_t ta(utc.time(2017, 1, 1, 0, 0, 0), core::deltahours(24), 4);
        tsv_t historical_data;
        vector<tsv_t> forecast_sets;
        vector<double> weight_sets;
        size_t num_historical_data = 56;

        //Let's make three sets, one of two elements, one of three, and one of
        //four.
        tsv_t forecasts_1, forecasts_2, forecasts_3;

        vector<double> currvals = {13.4, 15.6, 17.1, 19.1};
        forecasts_1.emplace_back(ta, currvals, fx_avg);
        currvals = {34.1, 2.4, 43.9, 10.2};
        forecasts_1.emplace_back(ta, currvals, fx_avg);
        forecast_sets.emplace_back(forecasts_1);
        weight_sets.emplace_back(5.0);
        currvals = {83.1, -42.2, 0.4, 23.4};
        forecasts_2.emplace_back(ta, currvals, fx_avg);
        currvals = {15.1, 6.5, 4.2, 2.9};
        forecasts_2.emplace_back(ta, currvals, fx_avg);
        currvals = {53.1, 87.9, 23.8, 5.6};
        forecasts_2.emplace_back(ta, currvals, fx_avg);
        forecast_sets.emplace_back(forecasts_2);
        weight_sets.emplace_back(9.0);
        currvals = {1.5, -1.9, -17.2, -10.0};
        forecasts_3.emplace_back(ta, currvals, fx_avg);
        currvals = {4.7, 18.2, 15.3, 8.9};
        forecasts_3.emplace_back(ta, currvals, fx_avg);
        currvals = {-45.2, -2.3, 80.2, 71.0};
        forecasts_3.emplace_back(ta, currvals, fx_avg);
        currvals = {45.1, -92.0, 34.4, 65.8};
        forecasts_3.emplace_back(ta, currvals, fx_avg);
        forecast_sets.emplace_back(forecasts_3);
        weight_sets.emplace_back(3.0);


        for (size_t i=0; i<num_historical_data; ++i) {
            vector<double> insertvec { static_cast<double>(std::rand())/RAND_MAX * 50.0,
                                       static_cast<double>(std::rand())/RAND_MAX * 50.0 ,
                                       static_cast<double>(std::rand())/RAND_MAX * 50.0 ,
                                       static_cast<double>(std::rand())/RAND_MAX * 50.0 };
            historical_data.emplace_back(ta, insertvec, fx_avg);
        }

        auto historical_order = qm::quantile_index<tsa_t>(historical_data, ta);

        core::utctime interpolation_start(core::no_utctime);

        //Act
        auto result = qm::quantile_map_main<tsa_t>(forecast_sets, weight_sets, 
                historical_data, ta, interpolation_start);

        //Assert
        for (size_t i=0; i<num_historical_data; ++i) {
            if (i < 4) {
                FAST_CHECK_EQ(result[historical_order[0][i]].value(0), -45.2);
            } else if (i < 7) {
                FAST_CHECK_EQ(result[historical_order[0][i]].value(0), 1.5);
            } else if (i < 11) {
                FAST_CHECK_EQ(result[historical_order[0][i]].value(0), 4.7);
            } else if (i < 16) {
                FAST_CHECK_EQ(result[historical_order[0][i]].value(0), 13.4);
            } else if (i < 26) {
                FAST_CHECK_EQ(result[historical_order[0][i]].value(0), 15.1);
            } else if (i < 32) {
                FAST_CHECK_EQ(result[historical_order[0][i]].value(0), 34.1);
            } else if (i < 35) {
                FAST_CHECK_EQ(result[historical_order[0][i]].value(0), 45.1);
            } else if (i < 45) {
                FAST_CHECK_EQ(result[historical_order[0][i]].value(0), 53.1);
            } else {
                FAST_CHECK_EQ(result[historical_order[0][i]].value(0), 83.1);
            } 

            if (i < 4) {
                FAST_CHECK_EQ(result[historical_order[1][i]].value(1), -92.0);
            } else if (i < 14) {
                FAST_CHECK_EQ(result[historical_order[1][i]].value(1), -42.2);
            } else if (i < 17) {
                FAST_CHECK_EQ(result[historical_order[1][i]].value(1), -2.3);
            } else if (i < 21) {
                FAST_CHECK_EQ(result[historical_order[1][i]].value(1), -1.9);
            } else if (i < 26) {
                FAST_CHECK_EQ(result[historical_order[1][i]].value(1), 2.4);
            } else if (i < 36) {
                FAST_CHECK_EQ(result[historical_order[1][i]].value(1), 6.5);
            } else if (i < 42) {
                FAST_CHECK_EQ(result[historical_order[1][i]].value(1), 15.6);
            } else if (i < 45) {
                FAST_CHECK_EQ(result[historical_order[1][i]].value(1), 18.2);
            } else {
                FAST_CHECK_EQ(result[historical_order[1][i]].value(1), 87.9);
            }

            if (i < 4) {
                FAST_CHECK_EQ(result[historical_order[2][i]].value(2), -17.2);
            } else if (i < 14) {
                FAST_CHECK_EQ(result[historical_order[2][i]].value(2), 0.4);
            } else if (i < 24) {
                FAST_CHECK_EQ(result[historical_order[2][i]].value(2), 4.2);
            } else if (i < 27) {
                FAST_CHECK_EQ(result[historical_order[2][i]].value(2), 15.3);
            } else if (i < 33) {
                FAST_CHECK_EQ(result[historical_order[2][i]].value(2), 17.1);
            } else if (i < 43) {
                FAST_CHECK_EQ(result[historical_order[2][i]].value(2), 23.8);
            } else if (i < 47) {
                FAST_CHECK_EQ(result[historical_order[2][i]].value(2), 34.4);
            } else if (i < 52) {
                FAST_CHECK_EQ(result[historical_order[2][i]].value(2), 43.9);
            } else {
                FAST_CHECK_EQ(result[historical_order[2][i]].value(2), 80.2);
            }

            if (i < 4) {
                FAST_CHECK_EQ(result[historical_order[3][i]].value(3), -10.0);
            } else if (i < 14) {
                FAST_CHECK_EQ(result[historical_order[3][i]].value(3), 2.9);
            } else if (i < 24) {
                FAST_CHECK_EQ(result[historical_order[3][i]].value(3), 5.6);
            } else if (i < 27) {
                FAST_CHECK_EQ(result[historical_order[3][i]].value(3), 8.9);
            } else if (i < 33) {
                FAST_CHECK_EQ(result[historical_order[3][i]].value(3), 10.2);
            } else if (i < 39) {
                FAST_CHECK_EQ(result[historical_order[3][i]].value(3), 19.1);
            } else if (i < 49) {
                FAST_CHECK_EQ(result[historical_order[3][i]].value(3), 23.4);
            } else if (i < 52) {
                FAST_CHECK_EQ(result[historical_order[3][i]].value(3), 65.8);
            } else {
                FAST_CHECK_EQ(result[historical_order[3][i]].value(3), 71.0);
            }
        }

    }
}
