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
         * \param weights The weights for each time-series. Same size as tsv_t, or size zero (unity weights).
         * \return vector<vector<int>> that have dimension [ta.size()][tsv.size() or accumulated sum of weights] holding the indexes of the by-value in time-step ordered list
         *
         */
        template <class tsa_t, class tsv_t, class ta_t>
        vector<vector<int>> quantile_index(tsv_t const &tsv, ta_t const &ta, vector<int> const &weights) {

            bool unity_weights = false;
            if (weights.size() == 0) {
                unity_weights = true;
            }
            if (!unity_weights && weights.size() != tsv.size()) {
                throw runtime_error("Weights and time series are not equal length");
            }

            size_t final_size = unity_weights ? tsv.size() : std::accumulate(weights.begin(), weights.end(), 0);

            vector <tsa_t> tsa; tsa.reserve(final_size); // create ts-accessors to enable fast thread-safe access to ts value aspect
            vector<int> pi(final_size);
            size_t currind = 0;
            for (size_t i=0; i < tsv.size(); i++) {
                if (unity_weights) {
                    tsa.emplace_back(tsv[i], ta);
                    pi[currind] = currind;
                    currind++;
                } else {
                    for (size_t j=0; j < weights[i]; j++) {
                        tsa.emplace_back(tsv[i], ta);
                        pi[currind] = currind;
                        currind++;
                    }
                }
            }

            vector<vector<int>> qi(ta.size()); // result vector, qi[i] -> a vector of index-order tsv[i].value(t)
            for(size_t t = 0; t<ta.size(); ++t) {
                sort(begin(pi),end(pi),[&tsa,t](int a,int b)->int {return tsa[a].value(t)<tsa[b].value(t);});
                qi[t]=pi; // it's a copy assignment, ok, since sort above will start out with previous time-step order
            }
            return qi;
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
        vector<int> weights;

        // #2: Act
        auto qi=qm::quantile_index<tsa_t>(prior, ta, weights);

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
}

