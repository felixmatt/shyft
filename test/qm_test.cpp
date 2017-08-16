#include "test_pch.h"
#define _USE_MATH_DEFINES
#include "core/time_series.h"
#include "core/time_axis.h"
#include "core/utctime_utilities.h"
#include "core/time_series_qm.h"
//#include "api/api.h"
//#include "api/time_series.h"
//#include "core/time_series_statistics.h"

// workbench for qm data model and algorithms goes into shyft::qm
// promoted to core/time_series_qm.h when done

namespace shyft {
    namespace qm {
        using namespace std;
        // new work, modification can go into this place before promoting it(save compile time!)
    }
}


using namespace shyft;
using namespace std;
using ta_t = time_axis::fixed_dt;
using ts_t = time_series::point_ts<ta_t>;
using tsa_t = time_series::average_accessor<ts_t, ta_t>;
using tsv_t = std::vector<ts_t>;

TEST_SUITE("qm") {
    TEST_CASE("ts_vector_to_quantile_ix_list") { // to run this test: test_shyft -tc=ts_vector_to_quantile_ix_list
                                                 // standard triple A testing

                                                 // #1:Arrange
        size_t n_prior = 100;
        core::calendar utc;
        ta_t ta(utc.time(2017, 1, 1, 0, 0, 0), core::deltahours(24), 14);
        tsv_t prior;
        for (size_t i = 0; i<n_prior; ++i) {
            vector<double> v;
            for (size_t t = 0; t<ta.size(); ++t)
                v.push_back(double((i + t) % n_prior));
            prior.emplace_back(ta, v, time_series::ts_point_fx::POINT_AVERAGE_VALUE);
        }

        // #2: Act
        auto qi = qm::quantile_index<tsa_t>(prior, ta);

        // #3: Assert
        FAST_REQUIRE_EQ(qi.size(), ta.size());
        for (size_t t = 0; t<ta.size(); ++t) {
            FAST_REQUIRE_EQ(qi[t].size(), prior.size());//each time-step, have indexes for prior.size() (ordered) index entries
            for (size_t i = 0; i<qi[t].size() - 1; ++i) {
                double v1 = prior[qi[t][i]].value(t);
                double v2 = prior[qi[t][i + 1]].value(t);
                FAST_CHECK_LE(v1, v2); // just ensure all values are ordered in ascending order, based on index-lookup and original ts.
            }
        }
    }

    TEST_CASE("compute_weighted_quantiles") {
        // Arrange
        const auto fx_avg = time_series::ts_point_fx::POINT_AVERAGE_VALUE;
        size_t n_quantiles = 100;
        core::calendar utc;
        ta_t ta(utc.time(2017, 1, 1, 0, 0, 0), core::deltahours(24), 2);
        //-----------------
        vector<double> weights;//weight of each forecast below,two with equal weights
        tsv_t fcv;// forecast-vector, there are 4 of them in this case, with two values each
        weights.push_back(5.0); fcv.emplace_back(ta, vector<double>{14.2, 4.1}, fx_avg);
        weights.push_back(2.0); fcv.emplace_back(ta, vector<double>{13.0, 2.9}, fx_avg);
        weights.push_back(2.0); fcv.emplace_back(ta, vector<double>{15.8, 20.4}, fx_avg);
        weights.push_back(1.0); fcv.emplace_back(ta, vector<double>{ 9.1, 11.2}, fx_avg);

        auto q_order = qm::quantile_index<tsa_t>(fcv, ta); // here we use already tested function to get ordering, all steps
        qm::wvo_accessor<ts_t> wvo(q_order, weights, fcv);// stash it into wvo_accessor, so we are ready to go

                                                          // Act
        vector<double> q0 = qm::compute_weighted_quantiles(n_quantiles, wvo);
        wvo.t_ix++; // increment to next time-step
        vector<double> q1 = qm::compute_weighted_quantiles(n_quantiles, wvo);

        // Assert
        // to consider: could we select values etc. to make a data-driven approach?
        FAST_REQUIRE_EQ(q0.size(), n_quantiles);
        for (size_t i = 0; i<n_quantiles; ++i) {
            if (i <= 9) {
                FAST_CHECK_EQ(q0[i], 9.1);
            }
            else if (i <= 29) {
                FAST_CHECK_EQ(q0[i], 13.0);
            }
            else if (i <= 79) {
                FAST_CHECK_EQ(q0[i], 14.2);
            }
            else {
                FAST_CHECK_EQ(q0[i], 15.8);
            }
        }
        FAST_REQUIRE_EQ(q1.size(), n_quantiles);
        for (size_t i = 0; i<n_quantiles; ++i) {
            if (i <= 19) {
                FAST_CHECK_EQ(q1[i], 2.9);
            }
            else if (i <= 69) {
                FAST_CHECK_EQ(q1[i], 4.1);
            }
            else if (i <= 79) {
                FAST_CHECK_EQ(q1[i], 11.2);
            }
            else {
                FAST_CHECK_EQ(q1[i], 20.4);
            }
        }
    }

    TEST_CASE("interpolated_weighted_quantiles") {
        // Arrange
        const auto fx_avg = time_series::ts_point_fx::POINT_AVERAGE_VALUE;
        size_t n_quantiles = 5;
        core::calendar utc;
        ta_t ta(utc.time(2017, 1, 1, 0, 0, 0), core::deltahours(24), 2);
        //-----------------
        vector<double> weights;//weight of each forecast below,two with equal weights
        tsv_t fcv;// forecast-vector, there are 4 of them in this case, with two values each
        weights.push_back(5.0); fcv.emplace_back(ta, vector<double>{14.2, 4.1}, fx_avg);
        weights.push_back(2.0); fcv.emplace_back(ta, vector<double>{13.0, 2.9}, fx_avg);
        weights.push_back(2.0); fcv.emplace_back(ta, vector<double>{15.8, 20.4}, fx_avg);
        weights.push_back(1.0); fcv.emplace_back(ta, vector<double>{ 9.1, 11.2}, fx_avg);

        auto q_order = qm::quantile_index<tsa_t>(fcv, ta); // here we use already tested function to get ordering, all steps
        qm::wvo_accessor<ts_t> wvo(q_order, weights, fcv);// stash it into wvo_accessor, so we are ready to go

        // Act
        vector<double> q0 = qm::compute_interp_weighted_quantiles(n_quantiles, wvo);
        wvo.t_ix++; // increment to next time-step
        vector<double> q1 = qm::compute_interp_weighted_quantiles(n_quantiles, wvo);

        // Assert
        FAST_REQUIRE_EQ(q0.size(), n_quantiles);
        FAST_CHECK_EQ(q0[0], 9.1);
        TS_ASSERT_DELTA(q0[1], 13.171428571428571, 1e-15);
        TS_ASSERT_DELTA(q0[2], 14.028571428571428, 1e-15);
        TS_ASSERT_DELTA(q0[3], 15.114285714285714, 1e-15);
        FAST_CHECK_EQ(q0[4], 15.8);

        FAST_REQUIRE_EQ(q1.size(), n_quantiles);
        FAST_CHECK_EQ(q1[0], 2.9);
        TS_ASSERT_DELTA(q1[1], 3.4142857142857141, 1e-15);
        TS_ASSERT_DELTA(q1[2], 5.2833333333333332, 1e-15);
        TS_ASSERT_DELTA(q1[3], 11.2, 1e-15);
        FAST_CHECK_EQ(q1[4], 20.4);
    }

    TEST_CASE("quantile_mapping") {
        const auto fx_avg = time_series::ts_point_fx::POINT_AVERAGE_VALUE;
        // Arrange
        core::calendar utc;
        ta_t ta(utc.time(2017, 1, 1, 0, 0, 0), core::deltahours(24), 2);
        tsv_t prior_ts_v;
        tsv_t forecast_ts_v;

        vector<double> weights;//weight of each forecast below,two with equal weights
        weights.push_back(39.0); forecast_ts_v.emplace_back(ta, vector<double>{32.1, 1.2}, fx_avg);
        weights.push_back(32.0); forecast_ts_v.emplace_back(ta, vector<double>{21.0, 34.2}, fx_avg);
        weights.push_back(8.0); forecast_ts_v.emplace_back(ta, vector<double>{ 10.2, 12.4}, fx_avg);
        weights.push_back(73.0); forecast_ts_v.emplace_back(ta, vector<double>{ 71.0, 89.2}, fx_avg);
        weights.push_back(14.0); forecast_ts_v.emplace_back(ta, vector<double>{ 35.4, 83.4}, fx_avg);
        auto q_order = qm::quantile_index<tsa_t>(forecast_ts_v, ta); // here we use already tested function to get ordering, all steps

                                                                     // We now just create a random prior time series vector. The values
                                                                     // that we put in do not matter, since they will be overwritten if
                                                                     // everything works. However, the ordering matters when we are going to
                                                                     // compare so we have to keep track of the ordering.

        size_t num_priors = 43;
        for (size_t i = 0; i<num_priors; ++i) {
            // Generate random no between 0 and 50
            vector<double> insertvec{ static_cast<double>(std::rand()) / RAND_MAX * 50.0,
                static_cast<double>(std::rand()) / RAND_MAX * 50.0 };
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
        for (size_t i = 0; i<num_priors; ++i) {
            if (i < 3) {
                FAST_CHECK_EQ(result[pri_q_order[0][i]].value(0), 10.2);
            }
            else if (i < 11) {
                FAST_CHECK_EQ(result[pri_q_order[0][i]].value(0), 21.0);
            }
            else if (i < 20) {
                FAST_CHECK_EQ(result[pri_q_order[0][i]].value(0), 32.1);
            }
            else if (i < 24) {
                FAST_CHECK_EQ(result[pri_q_order[0][i]].value(0), 35.4);
            }
            else {
                FAST_CHECK_EQ(result[pri_q_order[0][i]].value(0), 71.0);
            }

            if (i < 10) {
                FAST_CHECK_EQ(result[pri_q_order[1][i]].value(1), 1.2);
            }
            else if (i < 12) {
                FAST_CHECK_EQ(result[pri_q_order[1][i]].value(1), 12.4);
            }
            else if (i < 20) {
                FAST_CHECK_EQ(result[pri_q_order[1][i]].value(1), 34.2);
            }
            else if (i < 24) {
                FAST_CHECK_EQ(result[pri_q_order[1][i]].value(1), 83.4);
            }
            else {
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
        for (size_t i = 0; i<2; ++i) {
            weights.push_back(1.0);
            vector<double> currvals;
            for (size_t j = 0; j<14; ++j) {
                if (i == 0) {
                    currvals.emplace_back(5.0);
                }
                else if (i == 1) {
                    currvals.emplace_back(20.0);
                }
            }
            forecast_ts_v.emplace_back(ta, currvals, fx_avg);
        }
        auto q_order = qm::quantile_index<tsa_t>(forecast_ts_v, ta);

        size_t num_priors = 18;
        for (size_t i = 0; i<num_priors; ++i) {
            vector<double> insertvec;
            for (size_t t = 0; t<14; ++t) {
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
        for (size_t i = 0; i<num_priors; ++i) {
            for (size_t t = 0; t<14; ++t) {
                // The first half of the priors should have been mapped to 5,
                // and interpolated between 5 and i during the interpolation
                // period. For the second half, the value is 20.
                if (i < num_priors / 2) {
                    if (t < 9) {
                        FAST_CHECK_EQ(result[pri_q_order[t][i]].value(t),
                            5.0);
                    }
                    else {
                        double weight = (t - 9) / 4.0;
                        FAST_CHECK_EQ(result[pri_q_order[t][i]].value(t),
                            ((1.0 - weight) * 5.0 + weight * i));
                    }
                }
                else {
                    if (t < 9) {
                        FAST_CHECK_EQ(result[pri_q_order[t][i]].value(t),
                            20.0);
                    }
                    else {
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
        for (size_t i = 0; i<3; ++i) {
            vector<double> currvals;
            ta_t ta;
            weights.emplace_back(1.0);
            if (i == 0) {
                ta = ta_t(utc.time(2017, 1, 1, 0, 0, 0), core::deltahours(24), 2);
                currvals = { 1.0, 1.0 };
            }
            else if (i == 1) {
                ta = ta_t(utc.time(2017, 1, 1, 0, 0, 0), core::deltahours(24), 3);
                currvals = { 2.0, 2.0, 2.0 };
            }
            else if (i == 2) {
                ta = ta_t(utc.time(2017, 1, 1, 0, 0, 0), core::deltahours(24), 4);
                currvals = { 3.0, 3.0, 3.0, 3.0 };
            }
            forecast_ts_v.emplace_back(ta, currvals, fx_avg);
        }

        ta_t ta(utc.time(2017, 1, 1, 0, 0, 0), core::deltahours(24), 6);
        auto q_order = qm::quantile_index<tsa_t>(forecast_ts_v, ta);

        // Create the prior series. These will all have four data points
        for (size_t i = 0; i<9; ++i) {
            vector<double> currvals;
            for (size_t t = 0; t<ta.size(); ++t) {
                currvals.emplace_back(i);
            }
            prior_ts_v.emplace_back(ta, currvals, fx_avg);
        }

        auto pri_q_order = qm::quantile_index<tsa_t>(prior_ts_v, ta);

        //Act
        auto result = qm::quantile_mapping<tsa_t>(prior_ts_v, forecast_ts_v,
            pri_q_order, q_order, weights, ta, interp_start);

        //Assert
        for (size_t i = 0; i<9; ++i) {
            for (size_t t = 0; t<3; ++t) {
                if (t < 2) {
                    if (i < 3) {
                        FAST_CHECK_EQ(result[pri_q_order[t][i]].value(t), 1.0);
                    }
                    else if (i < 6) {
                        FAST_CHECK_EQ(result[pri_q_order[t][i]].value(t), 2.0);
                    }
                    else if (i < 9) {
                        FAST_CHECK_EQ(result[pri_q_order[t][i]].value(t), 3.0);
                    }
                }
                else if (t < 3) {
                    if (i < 5) {
                        FAST_CHECK_EQ(result[pri_q_order[t][i]].value(t), 2.0);
                    }
                    else if (i < 9) {
                        FAST_CHECK_EQ(result[pri_q_order[t][i]].value(t), 3.0);
                    }
                }
                else {
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

        vector<double> currvals = { 13.4, 15.6, 17.1, 19.1 };
        forecasts_1.emplace_back(ta, currvals, fx_avg);
        currvals = { 34.1, 2.4, 43.9, 10.2 };
        forecasts_1.emplace_back(ta, currvals, fx_avg);
        forecast_sets.emplace_back(forecasts_1);
        weight_sets.emplace_back(5.0);
        currvals = { 83.1, -42.2, 0.4, 23.4 };
        forecasts_2.emplace_back(ta, currvals, fx_avg);
        currvals = { 15.1, 6.5, 4.2, 2.9 };
        forecasts_2.emplace_back(ta, currvals, fx_avg);
        currvals = { 53.1, 87.9, 23.8, 5.6 };
        forecasts_2.emplace_back(ta, currvals, fx_avg);
        forecast_sets.emplace_back(forecasts_2);
        weight_sets.emplace_back(9.0);
        currvals = { 1.5, -1.9, -17.2, -10.0 };
        forecasts_3.emplace_back(ta, currvals, fx_avg);
        currvals = { 4.7, 18.2, 15.3, 8.9 };
        forecasts_3.emplace_back(ta, currvals, fx_avg);
        currvals = { -45.2, -2.3, 80.2, 71.0 };
        forecasts_3.emplace_back(ta, currvals, fx_avg);
        currvals = { 45.1, -92.0, 34.4, 65.8 };
        forecasts_3.emplace_back(ta, currvals, fx_avg);
        forecast_sets.emplace_back(forecasts_3);
        weight_sets.emplace_back(3.0);


        for (size_t i = 0; i<num_historical_data; ++i) {
            vector<double> insertvec{ static_cast<double>(std::rand()) / RAND_MAX * 50.0,
                static_cast<double>(std::rand()) / RAND_MAX * 50.0 ,
                static_cast<double>(std::rand()) / RAND_MAX * 50.0 ,
                static_cast<double>(std::rand()) / RAND_MAX * 50.0 };
            historical_data.emplace_back(ta, insertvec, fx_avg);
        }

        auto historical_order = qm::quantile_index<tsa_t>(historical_data, ta);

        core::utctime interpolation_start(core::no_utctime);

        //Act
        auto result = qm::quantile_map_forecast<tsa_t>(forecast_sets, weight_sets,
            historical_data, ta, interpolation_start);

        //Assert
        for (size_t i = 0; i<num_historical_data; ++i) {
            if (i < 4) {
                FAST_CHECK_EQ(result[historical_order[0][i]].value(0), -45.2);
            }
            else if (i < 7) {
                FAST_CHECK_EQ(result[historical_order[0][i]].value(0), 1.5);
            }
            else if (i < 11) {
                FAST_CHECK_EQ(result[historical_order[0][i]].value(0), 4.7);
            }
            else if (i < 16) {
                FAST_CHECK_EQ(result[historical_order[0][i]].value(0), 13.4);
            }
            else if (i < 26) {
                FAST_CHECK_EQ(result[historical_order[0][i]].value(0), 15.1);
            }
            else if (i < 32) {
                FAST_CHECK_EQ(result[historical_order[0][i]].value(0), 34.1);
            }
            else if (i < 35) {
                FAST_CHECK_EQ(result[historical_order[0][i]].value(0), 45.1);
            }
            else if (i < 45) {
                FAST_CHECK_EQ(result[historical_order[0][i]].value(0), 53.1);
            }
            else {
                FAST_CHECK_EQ(result[historical_order[0][i]].value(0), 83.1);
            }

            if (i < 4) {
                FAST_CHECK_EQ(result[historical_order[1][i]].value(1), -92.0);
            }
            else if (i < 14) {
                FAST_CHECK_EQ(result[historical_order[1][i]].value(1), -42.2);
            }
            else if (i < 17) {
                FAST_CHECK_EQ(result[historical_order[1][i]].value(1), -2.3);
            }
            else if (i < 21) {
                FAST_CHECK_EQ(result[historical_order[1][i]].value(1), -1.9);
            }
            else if (i < 26) {
                FAST_CHECK_EQ(result[historical_order[1][i]].value(1), 2.4);
            }
            else if (i < 36) {
                FAST_CHECK_EQ(result[historical_order[1][i]].value(1), 6.5);
            }
            else if (i < 42) {
                FAST_CHECK_EQ(result[historical_order[1][i]].value(1), 15.6);
            }
            else if (i < 45) {
                FAST_CHECK_EQ(result[historical_order[1][i]].value(1), 18.2);
            }
            else {
                FAST_CHECK_EQ(result[historical_order[1][i]].value(1), 87.9);
            }

            if (i < 4) {
                FAST_CHECK_EQ(result[historical_order[2][i]].value(2), -17.2);
            }
            else if (i < 14) {
                FAST_CHECK_EQ(result[historical_order[2][i]].value(2), 0.4);
            }
            else if (i < 24) {
                FAST_CHECK_EQ(result[historical_order[2][i]].value(2), 4.2);
            }
            else if (i < 27) {
                FAST_CHECK_EQ(result[historical_order[2][i]].value(2), 15.3);
            }
            else if (i < 33) {
                FAST_CHECK_EQ(result[historical_order[2][i]].value(2), 17.1);
            }
            else if (i < 43) {
                FAST_CHECK_EQ(result[historical_order[2][i]].value(2), 23.8);
            }
            else if (i < 47) {
                FAST_CHECK_EQ(result[historical_order[2][i]].value(2), 34.4);
            }
            else if (i < 52) {
                FAST_CHECK_EQ(result[historical_order[2][i]].value(2), 43.9);
            }
            else {
                FAST_CHECK_EQ(result[historical_order[2][i]].value(2), 80.2);
            }

            if (i < 4) {
                FAST_CHECK_EQ(result[historical_order[3][i]].value(3), -10.0);
            }
            else if (i < 14) {
                FAST_CHECK_EQ(result[historical_order[3][i]].value(3), 2.9);
            }
            else if (i < 24) {
                FAST_CHECK_EQ(result[historical_order[3][i]].value(3), 5.6);
            }
            else if (i < 27) {
                FAST_CHECK_EQ(result[historical_order[3][i]].value(3), 8.9);
            }
            else if (i < 33) {
                FAST_CHECK_EQ(result[historical_order[3][i]].value(3), 10.2);
            }
            else if (i < 39) {
                FAST_CHECK_EQ(result[historical_order[3][i]].value(3), 19.1);
            }
            else if (i < 49) {
                FAST_CHECK_EQ(result[historical_order[3][i]].value(3), 23.4);
            }
            else if (i < 52) {
                FAST_CHECK_EQ(result[historical_order[3][i]].value(3), 65.8);
            }
            else {
                FAST_CHECK_EQ(result[historical_order[3][i]].value(3), 71.0);
            }
        }

    }
}
