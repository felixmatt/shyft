#include "test_pch.h"

#include <cmath>
#include <vector>

#include "core/dtss.h"
#include "core/predictions.h"
#include "core/utctime_utilities.h"
#include "core/time_axis.h"
#include "core/time_series.h"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>


namespace {

namespace sp = shyft::prediction;
namespace core = shyft::core;
namespace sta = shyft::time_axis;
namespace sts = shyft::time_series;

template <class T>
static T serialize_loop(const T& o) {
    std::ostringstream xmls;

    boost::archive::binary_oarchive oa(xmls);
    oa << BOOST_SERIALIZATION_NVP(o);

    xmls.flush();

    std::string ss=xmls.str();
    std::istringstream xmli(ss);
    boost::archive::binary_iarchive ia(xmli);

    T o2;
    ia >> BOOST_SERIALIZATION_NVP(o2);

    return o2;
}

template < typename TA >
std::vector<double> make_sine(const TA & time_axis) {
    std::vector<double> sine;
    sine.reserve(time_axis.size());
    double scale = 2*M_PI/(time_axis.size()-1);
    for ( std::size_t i = 0; i < time_axis.size(); ++i ) {
        sine.emplace_back(std::sin(i*scale));
    }

    return sine;
}

}

TEST_SUITE("predictors") {

TEST_CASE("predictor_serialization") {

    core::utctime t0 = core::utctime_now();
    core::utctimespan dt = core::deltahours(3);
    std::size_t n = 1000u;

    SUBCASE("serialize krls_rbf_predictor") {
        sta::fixed_dt time_ax = sta::fixed_dt(t0, dt, n);
        sts::point_ts<sta::fixed_dt> sine_ts = sts::point_ts<sta::fixed_dt>{ time_ax, make_sine(time_ax) ,sts::ts_point_fx::POINT_AVERAGE_VALUE};
        sts::point_ts<sta::fixed_dt> sine_ts2 = sts::point_ts<sta::fixed_dt>{ time_ax, make_sine(time_ax) ,sts::ts_point_fx::POINT_INSTANT_VALUE};

        sp::krls_rbf_predictor pred{ core::deltahours(3), 1E-6, 0.001, 100000u };
        pred.train(sine_ts);

        sp::krls_rbf_predictor deserialized_pred = serialize_loop(pred);
        auto pred_vec = pred.predict_vec(time_ax);
        auto pred_ts = pred.predict<decltype(sine_ts)>(time_ax);
        auto ds_pred_vec = deserialized_pred.predict_vec(time_ax);
        FAST_CHECK_EQ(pred_ts.point_interpretation(),sine_ts.point_interpretation());
        FAST_REQUIRE_EQ(ds_pred_vec.size(), pred_vec.size());
        for ( std::size_t i = 0u; i < n; ++i ) {
            FAST_CHECK_EQ(ds_pred_vec.at(i), pred_vec.at(i));
        }
        pred.train(sine_ts2);
        auto pred_ts2 = pred.predict<decltype(sine_ts2)>(time_ax);
        FAST_CHECK_EQ(pred_ts.point_interpretation(),sine_ts.point_interpretation());

    }
}

}
