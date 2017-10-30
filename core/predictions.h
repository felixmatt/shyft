#pragma once


#include <cstddef>
#include <vector>
#include <algorithm>
#include <utility>
#include <limits>
#include <dlib/svm.h>


#include "time_axis.h"
#include "time_series.h"

namespace shyft {
namespace prediction {

namespace ta = shyft::time_axis;
namespace ts = shyft::time_series;
using utctime=shyft::core::utctime;


/** \brief A time-series predictor based on the krls algorithm with a rbf kernel, with dt-normalizing.
 * \detail This predictor normalizes time-axes as to reduce the span in the time dimension.
 */
class krls_rbf_predictor
{
public:  // static/type api
    using krls_sample_type = dlib::matrix<double, 1, 1>;
    using kernel_type = dlib::radial_basis_kernel<krls_sample_type>;
    // -----
    using scalar_type = typename kernel_type::scalar_type;  // becomes the type specified in `krls_sample_type` -> double

private:
    core::utctimespan _dt;
    dlib::krls<dlib::radial_basis_kernel<dlib::matrix<double, 1, 1>>> _krls = dlib::krls<kernel_type>{ kernel_type{} };
    ts::ts_point_fx train_point_fx=ts::ts_point_fx::POINT_AVERAGE_VALUE;
public:  // con-/de-struction, move & copy
    krls_rbf_predictor() = default;
    // -----
    /** Construct a new krls predictor with a rbf kernel using the spesified gamma.
    */
    krls_rbf_predictor(
        const core::utctimespan dt,
        const scalar_type radial_kernel_gamma,
        const scalar_type tolerance,
        std::size_t max_dict_size = 1000000u
    ) : krls_rbf_predictor{ kernel_type{ radial_kernel_gamma }, dt, tolerance, max_dict_size } { }
    /** Construct a new krls predictor with a given kernel.
     */
    krls_rbf_predictor(
        kernel_type && kernel,
        const core::utctimespan dt,
        const scalar_type tolerance,
        const std::size_t max_dict_size = 1000000u
    ) : _dt{ dt }, _krls{ std::forward<kernel_type>(kernel), tolerance, max_dict_size } { }
    krls_rbf_predictor(
        const kernel_type & kernel,
        const core::utctimespan dt,
        const scalar_type tolerance,
        const std::size_t max_dict_size = 1000000u
    ) : _dt { dt }, _krls{ kernel, tolerance, max_dict_size } { }
    // -----
    ~krls_rbf_predictor() = default;
    // -----
    krls_rbf_predictor(const krls_rbf_predictor &) = default;
    krls_rbf_predictor & operator= (const krls_rbf_predictor &) = default;
    // -----
    krls_rbf_predictor(krls_rbf_predictor &&) = default;
    krls_rbf_predictor & operator= (krls_rbf_predictor &&) = default;

public:
    /** \brief Train the krls prediction algorithm on samples taken from a time-axis.
     *
     * \tparam TS  Input time-series type. Must at least support:
     *
     *     * `TS::size()` to get the number of points.
     *     * `TS::time(std::size_t )` to get the i'th time-point.
     *     * `TS::value(std::size_t )` to get the i'th value.
     *
     * \param ts          Time-series to train on.
     * \param offset      Offset from the start of the time-series. Default to `0u`.
     * \param count       Number of samples from the beginning of `ts` to use. Default to maximum value.
     * \param stride      Stride between samples from the time-series. Defaults to `1u`.
     * \param iterations  Maximum number of times to train on the samples. Defaults to `10u`.
     * \param mse_tol     Tolerance for the mean-squared error over the training data. If the
     *                    mse after a training session is less than this skip training further.
     *                    Defaults to `1E-9`.
     */
    template < typename TS >
    scalar_type train(
        const TS & ts,
        const std::size_t offset = 0u,
        const std::size_t count = std::numeric_limits<std::size_t>::max(),
        const std::size_t stride = 1u,
        const std::size_t iterations = 1u,
        const scalar_type mse_tol = 0.001
    ) {
        std::size_t nan_count;
        scalar_type diff_v, mse = 0.;
        std::size_t dim = std::min(offset + count*stride, ts.size());
        krls_sample_type x;
        const scalar_type scaling_f = 1./_dt;  // compute time scaling factor
        train_point_fx = ts.point_interpretation();
        // training iteration
        core::utctime tp;
        scalar_type pv;
        std::size_t iter_count = 0u;
        while ( iter_count++ < iterations ) {
            mse = 0.;
            nan_count = 0u;
            for ( std::size_t i = offset; i < dim; i += stride ) {
                tp = ts.time(i);
                pv = ts.value(i);

                if ( ! std::isnan(pv) ) {
                    x(0) = static_cast<scalar_type>(tp*scaling_f);  // NB: utctime -> double conversion !!!
                    _krls.train(x, pv);

                    diff_v = pv - _krls(x);
                    mse += diff_v * diff_v;
                } else {
                    nan_count += 1;
                }
            }

            mse /= std::max(static_cast<scalar_type>(dim - nan_count), 1.);
            if ( mse < mse_tol ) {
                return mse;
            }
        }
        return mse;
    }
    /** \brief Given a time-axis generate a point_ts prediction.
    *
    * \tparam TA  Time-axis type. Must at least support:
    *
    *    * `TA::size()` to get the number of points.
    *    * `TA::time(std::size_t )` to get the i'th time-point.
    *
    * \param ta  Time-axis with time-points to predict values at.
    * \return    A vector with predicted values. Is of equal leght as ta.
    */
    template < typename TA >
    std::vector<scalar_type> predict_vec(
        const TA & ta
    ) const {
        std::vector<scalar_type> predictions;
        predictions.reserve(ta.size());
        const scalar_type scaling_f = 1./_dt;  // compute time scaling factor

        krls_sample_type x_sample;
        for ( std::size_t i = 0, dim = ta.size(); i < dim; ++i ) {
            x_sample(0) = static_cast<scalar_type>(ta.time(i)*scaling_f);  // NB: utctime -> double conversion !!!
            predictions.emplace_back(_krls(x_sample));
        }

        return predictions;
    }
    /** \brief Given a time-axis generate a point_ts prediction.
     *
     * \tparam TA  Time-axis type. Must at least support:
     *
     *    * `TA::size()` to get the number of points.
     *    * `TA::time(std::size_t )` to get the i'th time-point.
     *	  * Should be able to be used by `TS` as the time-axis implementation.
     *
     * \tparam TS  Time-series type. Must be able to be initialized with a TA and a std::vector.
     *
     * \param ta  Time-axis with time-points to predict values at.
     * \return  A time-series using a copy of `ta` for time-points with predicted data-points.
     */
    template <typename TS,typename TA >
    TS predict(
        const TA & ta
    ) const {
        return TS{ ta, predict_vec(ta), train_point_fx };
    }
    /** \brief Given a time-axis generate a point_ts prediction.
    *
    * \param t  Time-point to predict at.
    * \return   Prediction at t.
    */
    scalar_type predict(
        const core::utctime t
    ) const {
        krls_sample_type x_sample{ t*1./_dt };  // NB: utctime -> double conversion !!!
        return _krls(x_sample);
    }
    /** \brief Compute the mean-squared error (_mse_) over the time-series.
     *
     * \tparam TS  Time-series type. Must at least support:
     *
     *     * `TS::size()` to get the number of points.
     *     * `TS::time(std::size_t )` to get the i'th time-point.
     *     * `TS::value(std::size_t )` to get the i'th value.
     *
     * \param ts      Time-series to compute mse over.
     * \param offset  Offset from the start of the time-series. Default to `0u`.
     * \param count   Number of samples from the beginning of `ts` to use. Default to maximum value.
     * \param stride  Stride between samples from the time-series. Defaults to `1u`.
     *
     * \return  The mse of the predictor for `ts`.
     */
    template < typename TS >
    scalar_type predictor_mse(
        const TS & ts,
        const std::size_t offset = 0u,
        const std::size_t count = std::numeric_limits<std::size_t>::max(),
        const std::size_t stride = 1u
    ) const {
        std::size_t nan_count = 0u;
        scalar_type mse = 0.;
        std::size_t dim = std::min(offset + count*stride, ts.size());
        const scalar_type scaling_f = 1./_dt;  // compute time scaling factor

        utctime tp;
        scalar_type pv;
        krls_sample_type x_sample;
        for ( std::size_t i = offset; i < dim; ++i ) {
            tp = ts.time(i);
            pv = ts.value(i);

            if ( ! std::isnan(pv) ) {
                x_sample(0) = static_cast<scalar_type>(tp*scaling_f);  // NB: utctime -> double conversion !!!
                scalar_type diff_v = pv - _krls(x_sample);
                mse += diff_v * diff_v;
            } else {
                nan_count += 1;
            }
        }

        return mse / std::max(static_cast<scalar_type>(dim - nan_count), 1.);
    }
    /** \brief Compute a gliding window mean-squared error estimate of the predictor.
     *
     * \tparam TS_i  Input time-series type. Must atleast support:
     *
     *    * `TS_i::size()` to get the number of points.
     *    * `TS_i::time(std::size_t )` to get the i'th time-point.
     *	  * `TS_i::time_axis()` to get (a reference to) the underlying time-axis.
     *
     * \tparam TS_o  Output time-axis type. Should be constructible from the time-axis of TS_i
     *               and a double vector with the same length as the time-axis.
     *
     * \param ta      Time-axis with time-points to compute mse for.
     * \param points  Number of extra points to use before/after eash points when computing.
     *
     * \return A time-series using a copy of `ta` for time-points, containing
     *         the mean-squared error for each point in ta.
     */
    template < typename TS_i, typename TS_o >
    TS_o mse_ts(
        const TS_i & ts,
        const std::size_t points
    ) const {
        std::vector<scalar_type> mse_vec( ts.size(), 0. );
        const scalar_type scaling_f = 1./_dt;  // compute time scaling factor

        std::size_t count;
        scalar_type ts_v;
        scalar_type diff_v;
        krls_sample_type x_sample;
        for ( std::size_t i = 0, dim = ts.size(); i < dim; ++i ) {
            count = 0u;
            std::size_t j_low = (i < points ? 0u : i - points);
            std::size_t j_high = (i + points + 1u > dim ? dim : i + points + 1u);
            for ( std::size_t j = j_low; j < j_high; ++j ) {
                ts_v = ts.value(j);
                if ( ! std::isnan(ts_v) ) {
                    count += 1;
                    x_sample(0) = static_cast<scalar_type>(scaling_f*ts.time(j));
                    // -----
                    diff_v = ts_v - _krls(x_sample);
                    mse_vec[i] += diff_v*diff_v;
                }
            }
            if ( count != 0 ) {
                mse_vec[i] /= count;
            } else {
                mse_vec[i] = shyft::nan;
            }
        }

        return TS_o{ ts.time_axis(), std::move(mse_vec), ts.point_interpretation() };
    }

public:
    /** \brief Clear all training data from the predictor.
     */
    void clear() {
        _krls.clear_dictionary();
    }

    x_serialize_decl();
};

} }  // shyft::prediction

x_serialize_export_key(shyft::prediction::krls_rbf_predictor);
