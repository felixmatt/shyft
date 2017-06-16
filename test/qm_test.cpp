#include "test_pch.h"
#define _USE_MATH_DEFINES
#include "core/time_series.h"
#include "core/time_axis.h"
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
            for (const auto& ts : tsv) tsa.emplace_back(ts, ta);

            vector<vector<int>> qi(ta.size()); // result vector, qi[i] -> a vector of index-order tsv[i].value(t)
            vector<int> pi(tsv.size());for(size_t i=0;i<tsv.size();++i) pi[i]=i;//initial order 0..n-1, and we re-use it in the loop
            for(size_t t=0;t<ta.size();++t) {
                sort(begin(pi),end(pi),[&tsa,t](int a,int b)->int {return tsa[a].value(t)<tsa[b].value(t);});
                qi[t]=pi; // it's a copy assignment, ok, since sort above will start out with previous time-step order
            }
            return qi;
        }


        template <class tsa_t>
        vector<double> get_sample_quantiles(size_t num_pri_cases,
                                            vector<int> const &quantile_indices,
                                            vector<double> const &weights,
                                            vector<tsa_t> const &accessor_vec,
                                            size_t time_idx) {
            // This function does not interpolate at all, simply does the 
            // nearest-rank assignment.
            vector<double> cumsum;
            for (size_t i=0; i<weights.size(); ++i) {
                if (i == 0) {
                    cumsum.emplace_back(weights[quantile_indices[i]]);
                } else {
                    cumsum.emplace_back(cumsum[i-1] + weights[quantile_indices[i]]);
                }
            }
            for (size_t i=0; i<cumsum.size(); ++i) {
                cumsum[i] = cumsum[i] / cumsum.back();
            }
            double curr_quantile = 0.0;
            double quantile_inc = 1.0 / (num_pri_cases - 1);
            int currind = 0;
            vector<double> quantile_values;
            quantile_values.reserve(num_pri_cases);
            double const eps = std::numeric_limits<double>::epsilon();
            while (curr_quantile <= 1.0 || abs(curr_quantile - 1.0) < eps) {
                // This is not the best way of comparing floating points,
                // see e.g. https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
                // but it seems a bit overkill for this case, so we just
                // add epsilon.
                while (cumsum[currind] + eps < curr_quantile) {
                    ++currind;
                }
                quantile_values.emplace_back(
                        accessor_vec[quantile_indices[currind]].value(time_idx));
                curr_quantile += quantile_inc;
            }
            return quantile_values;
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

    TEST_CASE("weighted_sample_quantiles") {
        // Arrange
        size_t num_priors = 73;
        core::calendar utc;
        ta_t ta(utc.time(2017, 1, 1, 0, 0, 0), core::deltahours(24), 2);
        vector<int> quantile_indices_firststep {3, 1, 0, 2};
        vector<int> quantile_indices_secondstep {1, 0, 3, 2};
        vector<double> weights {100.0, 32.0, 48.0, 90.0};

        tsv_t weighted_prognoses;
        vector<double> firststep {14.2, 13.0, 15.8, 9.1};
        vector<double> secondstep {4.1, 2.9, 20.4, 11.2};
        for (size_t i=0; i<firststep.size(); ++i) {
            vector<double> insertvec {firststep[i], secondstep[i]};
            weighted_prognoses.emplace_back(
                ta, insertvec, time_series::ts_point_fx::POINT_AVERAGE_VALUE);
        }
        vector<tsa_t> wp_accessors;
        wp_accessors.reserve(weights.size());
        for (const auto& ts : weighted_prognoses) 
            wp_accessors.emplace_back(ts, ta);


        // Act
        vector<double> result = qm::get_sample_quantiles(
                num_priors,
                quantile_indices_firststep,
                weights,
                wp_accessors,
                0);

        // Assert
        FAST_REQUIRE_EQ(result.size(), num_priors);
        for (size_t i=0; i<num_priors; ++i) {
            double curres = result[i];
            if (i <= 24) {
                FAST_REQUIRE_EQ(result[i], 9.1);
            } else if (i <= 32) {
                FAST_REQUIRE_EQ(result[i], 13.0);
            } else if (i <= 59) {
                FAST_REQUIRE_EQ(result[i], 14.2);
            } else {
                FAST_REQUIRE_EQ(result[i], 15.8);
            }
        }
    }
}

