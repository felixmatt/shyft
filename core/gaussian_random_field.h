///	Copyright 2014 Statkraft Energi A/S
///
///	This file is part of Shyft.
///
///	Shyft is free software: you can redistribute it and/or modify it under the terms of
/// the GNU Lesser General Public License as published by the Free Software Foundation,
/// either version 3 of the License, or (at your option) any later version.
///
///	Shyft is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
/// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
/// PURPOSE. See the GNU Lesser General Public License for more details.
///
///	You should have received a copy of the GNU Lesser General Public License along with
/// Shyft, usually located under the Shyft root directory in two files named COPYING.txt
/// and COPYING_LESSER.txt.	If not, see <http://www.gnu.org/licenses/>.
///
/// Inspired by early enki method programmed by Kolbjørn Engeland and Sjur Kolberg, and 
/// the later changes performed by Kalkulo AS.
///

/*
 * Some Refences to be included in code later
 * About the pointlessness of burn out: http://users.stat.umn.edu/~geyer/mcmc/burn.html
 *
 *
 */

#pragma once

#include <armadillo>
#define _USE_MATH_DEFINES // microsoft need this to get M_PI in place
#include <cmath>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/gamma.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include "RandomFields.h"
#include "geo_point.h"

namespace shyft {
    namespace core {
        namespace grf {

            class semivariogram_error : public std::exception {};

            inline arma::mat::fixed<2, 2> anisotropy_mapping(double anisotropy_ratio, double anisotropy_direction) {
                arma::mat::fixed<2, 2> T;

                if (anisotropy_ratio < 1.0)
                    throw semivariogram_error();

                if (anisotropy_ratio > 1.0 + 1.0e-10) {
                    const double kernel = (90.0 - anisotropy_direction)*M_PI/180.0;
                    T.at(0, 0) =  cos(kernel);
                    T.at(0, 1) =  sin(kernel);
                    T.at(1, 0) = -sin(kernel)*anisotropy_ratio;
                    T.at(1, 1) =  cos(kernel)*anisotropy_ratio;
                } else 
                    T.operator=( arma::eye(2, 2));
                return T;
            }

            /** Calculate the (symmetric) anisotropy distance matrix for stations
             *
             * Only compute upper half of matrix for speed.
             */
            template<typename SP>
            void calculate_anisotropy_distances(const SP& source_begin,
                                                const SP& source_end,
                                                double anisotropy_ratio,
                                                double anisotropy_direction,
                                                arma::mat& anisotropy_distances) {
                auto T = anisotropy_mapping(anisotropy_ratio, anisotropy_direction);
                const size_t n = std::distance(source_begin, source_end);
                anisotropy_distances.set_size(n, n);
                auto s_i = source_begin;
                for (size_t i = 0; i < n; ++i) {
                    auto p_i = (*s_i++)->geo_point();
                    auto s_j = s_i;
                    for (size_t j = i+1; j < n; ++j) {
                        auto p_j = (*s_j++)->geo_point();
                        arma::vec::fixed<2> d = {p_i.x - p_j.x, p_i.y - p_j.y};
                        anisotropy_distances.at(i, j) = arma::norm(T*d);
                    }
                }
                anisotropy_distances.diag() *= 0.0;
                anisotropy_distances = arma::symmatu(anisotropy_distances);
            }

            template<typename SP, typename D>
            void calculate_anisotropy_distances(const SP& source_begin,
                                                const SP& source_end,
                                                const D& destination_begin,
                                                const D& destination_end,
                                                double anisotropy_ratio,
                                                double anisotropy_direction,
                                                arma::mat& anisotropy_distances) {
                auto T = anisotropy_mapping(anisotropy_ratio, anisotropy_direction);
                const size_t m = std::distance(destination_begin, destination_end);
                const size_t n = std::distance(source_begin, source_end);
                anisotropy_distances.set_size(m, n);
                auto d_i = destination_begin;
                for (size_t i = 0; i < m; ++i) {
                    auto p_i = (d_i++)->geo_point();
                    auto s_j = source_begin;
                    for (size_t j = 0; j < n; ++j) {
                        auto p_j = (*s_j++)->geo_point();
                        arma::vec::fixed<2> d = {p_i.x - p_j.x, p_i.y - p_j.y};
                        anisotropy_distances.at(i, j) = arma::norm(T*d);
                    }
                }
            }

            void distance_sort_order(const arma::mat& distances, size_t sort_length, arma::umat& sort_order) {
                const size_t num_neighbours = std::min((size_t)(distances.n_cols), sort_length);
                sort_order.reshape(distances.n_rows, distances.n_cols);
                arma::uvec idx(distances.n_cols);
                for (arma::uword i = 0; i < distances.n_cols; ++i) idx.at(i) = i;
                for (arma::uword i = 0; i < distances.n_rows; ++i) {
                    arma::uvec row = idx;
                    std::partial_sort(std::begin(row), std::begin(row) + num_neighbours, 
                                      std::end(row), [&distances, i] (arma::uword a, arma::uword b) {
                        return distances.at(i, a) < distances.at(i, b);
                    });
                    sort_order.row(i) = row.t();
                }
            }


            struct gaussian_model {

                template <typename P>
                static inline void covariance(const arma::mat& distances, const P& semi_var, arma::mat& covariances) {
                    covariances = (semi_var.sill() - semi_var.nugget())*arma::exp(-arma::square(distances/semi_var.range()));
                }

                template <typename P>
                static inline void semivariance(const arma::mat& distances, const P& semi_var, arma::mat& semivariances) {
                    semivariances = 0.5*(semi_var.sill() - (semi_var.sill() -
                                         semi_var.nugget())*arma::exp(-arma::square(distances/semi_var.range()))
                                        );
                }

                template <typename P>
                static inline void semivariogram(const arma::mat& distances, const P& semi_var, arma::mat& gammas) {
                    gammas = semi_var.sill() - (semi_var.sill() - semi_var.nugget())*arma::exp(-arma::square(distances/semi_var.range()));
                }

                static inline double range_correction(double range, double shape) {
                    return 3.0*range/sqrt(3.0);
                }

                static const char* name() { return "gauss"; }

                template <typename SV>
                static std::vector<double> parameter(const SV& semi_var) {
                    return {semi_var.sill(), semi_var.range()} ;
                }

            };


            struct spherical_model {

                template <typename P>
                static inline void covariance(const arma::mat& distances, const P& semi_var, arma::mat& covariances) {
                    arma::mat r_dists = distances/semi_var.range();
                    covariances = 1.0 - 1.5*r_dists + 0.5*arma::pow(r_dists, 3);
                    covariances = (semi_var.sill() - semi_var.nugget())*arma::clamp(covariances, 0.0, covariances.max());
                }

                template <typename P>
                static inline void semivariance(const arma::mat& distances, const P& semi_var, arma::mat& semivariances) {
                    // Spherical model's semivariance equals the covariance
                    covariance(distances, semi_var, semivariances);
                }

                template <typename P>
                static inline void semivariogram(const arma::mat& distances, const P& semi_var, arma::mat& gammas) {
                    arma::mat tmp = 1.5*distances/semi_var.range() - 0.5*arma::pow(distances/semi_var.range(), 3);
                    gammas = semi_var.nugget() + (semi_var.sill() - semi_var.nugget())*arma::clamp(tmp, tmp.min(), semi_var.sill());
                }

                static inline double range_correction(double range, double shape) {
                    return 3.0*range/0.81;
                }

                static const char* name() { return "spherical"; }

                template <typename SV>
                static std::vector<double> parameter(const SV& semi_var) {
                    return {semi_var.sill(), semi_var.range()} ;
                }

            };


            struct exponential_model {

                template <typename P>
                static inline void covariance(const arma::mat& distances, const P& semi_var, arma::mat& covariances) {
                    covariances = (semi_var.sill() - semi_var.nugget())*arma::exp(-distances/semi_var.range());
                }

                template <typename P>
                static inline void semivariance(const arma::mat& distances, const P& semi_var, arma::mat& semivariances) {
                    semivariances = 0.5*(semi_var.sill() - (semi_var.sill() - semi_var.nugget())*arma::exp(-distances/semi_var.range()));
                }

                template <typename P>
                static inline void semivariogram(const arma::mat& distances, const P& semi_var, arma::mat& gammas) {
                    gammas = semi_var.sill() - (semi_var.sill() - semi_var.nugget())*exp(-distances/semi_var.range());
                }

                static inline double range_correction(double range, double shape) {
                    return range;
                }

                static const char* name() { return "exponenial"; }

                template <typename SV>
                static std::vector<double> parameter(const SV& semi_var) {
                    return {semi_var.sill(), semi_var.range()} ;
                }

            };


            struct matern_model {
                typedef boost::math::policies::policy<boost::math::policies::digits10<10>> boost_policy_type;

                template <typename P>
                static inline void covariance(const arma::mat& distances, const P& semi_var, arma::mat& covariances) {
                    const boost_policy_type policy = boost_policy_type();
                    covariances = distances/semi_var.range();
                    const double r = semi_var.sill() - semi_var.nugget();
                    const double p = pow(2, 1 - semi_var.shape());
                    const double e = exp(-boost::math::lgamma(semi_var.shape(), policy));
                    std::for_each(std::begin(covariances), std::end(covariances), [r, p, e, semi_var, policy] (double& v)
                        {
                            v = v > 0.0 ? r*p*e*boost::math::cyl_bessel_k(semi_var.shape(), v, policy)*pow(v, semi_var.shape()) : r;
                        });
                }

                template <typename P>
                static inline void semivariance(const arma::mat& distances, const P& semi_var, arma::mat& semivariances) {
                    return covariance(distances, semi_var, semivariances);
                }

                template <typename P>
                static inline void semivariogram(const arma::mat& distances, const P& semi_var, arma::mat& gammas) {
                    const boost_policy_type policy = boost_policy_type();
                    gammas = distances/semi_var.range();
                    const double p = pow(2, 1.0 - semi_var.shape());
                    const double e = exp(-boost::math::lgamma(semi_var.shape(), policy));
                    std::for_each(std::begin(gammas), std::end(gammas), [p, e, semi_var, policy] (double& v)
                        {
                            v = v > 0.0 ? semi_var.nugget() + (semi_var.sill() - semi_var.nugget())*(1.0 -
                                          boost::math::cyl_bessel_k(semi_var.shape(), v, policy)*pow(v, semi_var.shape())*p*e)
                                        : semi_var.nugget();
                        });
                }

                static inline double range_correction(double range, double shape) {
                    return 0.75*range/pow(shape, 0.4233);
                }

                static const char* name() { return "whittlematern"; }
                template <typename SV>
                static std::vector<double> parameter(const SV& semi_var) {
                    return {semi_var.sill(), semi_var.shape(), semi_var.range()} ;
                }
           };

            /** \brif Simple krig method to calculate gibbs weights for
             *        the sub set of indices given by sub_set.
             *
             * For an excellent introduction to kriging interpolation,
             * see http://people.ku.edu/~gbohling/cpe940/Kriging.pdf
             * This method assumes that the trend component of the random
             * field is known and constant mean, m(u) = m, see pages
             * 7 and 8 in the reference.
             */
            void calculate_gibbs_weights(const arma::mat& cov,
                                         const std::vector<unsigned int>& sub_inds,
                                         arma::mat& krig_weights) {
                const size_t m = cov.n_rows - 1; // Matrix dimension for krig LA problem
                krig_weights.set_size(sub_inds.size(), m + 1);
                arma::vec x(m);
                arma::vec weights(m + 1);
                arma::uvec indices(cov.n_rows);
                for (unsigned int i = 0; i < cov.n_rows; ++i) indices.at(i) = i;
                for (size_t idx = 0; idx < sub_inds.size(); ++idx) {
                    const unsigned int zero_index = sub_inds[idx];
                    // Build submatrix index vector by excluding zero_index station
                    std::vector<unsigned int> sub_vec(indices.begin(), indices.begin() + zero_index);
                    sub_vec.insert(sub_vec.end(), indices.begin() + zero_index + 1, indices.end());
                    const arma::uvec sub_indices(sub_vec);
                    const arma::uvec::fixed<1> col = {zero_index};

                    // Solve simple krig problem
                    arma::solve(x, cov(sub_indices, sub_indices), cov(sub_indices, col));

                    // Insert 0.0 as weight at (zero_index, zero_index) to preserve index mapping
                    // This allows efficient matrix-vector arithmetics when calculating the gibbs estimates
                    if (zero_index > 0)
                        weights.subvec(0, zero_index - 1) = x.subvec(0, zero_index - 1);
                    weights.at(zero_index) = 0.0;
                    if (zero_index < m)
                        weights.subvec(zero_index + 1, m) = x.subvec(zero_index, m - 1);
                    krig_weights.row(idx) = weights.t();
                }
            }

            /** \brief Calculate krig estimates and variances for all stations
             *
             * At each point, the estimate is given by z_j = k_j + sum_i!=j k_i*z_i and k_j = mu(1-sum_i!=j k_i.
             * See Analyzing Environmental Data by Walter W. Piegorsch,A. John Bailer, page 305.
             */
            void calculate_gibbs_estimates(const std::vector<unsigned int>& sub_inds,  // Sub indices to calculate over
                                           const std::vector<double>& data,            // Precipitations
                                           const arma::mat& weights,                   // Kriging weights
                                           const arma::mat& cov,                       // Full covariance matrix
                                           const double sill,
                                           std::vector<double>& mu,                    // Kriging estimate
                                           std::vector<double>& var) {                 // Kriging variance
                typedef arma::conv_to<std::vector<double>> to_std_vec;
                typedef arma::conv_to<arma::vec> to_arma_vec;
                typedef arma::conv_to<arma::uvec> to_arma_uvec;

                const double obs_mean = arma::mean(to_arma_vec::from(data));

                const arma::uvec sub_indices = to_arma_uvec::from(sub_inds);
                mu = to_std_vec::from(
                        obs_mean*(1.0 - arma::sum(weights, 1)) + weights*to_arma_vec::from(data)
                     );
                var.resize(sub_inds.size());

                for (size_t i = 0; i < sub_indices.n_rows; ++i) {
                    // sigma^2 = C(0) - lambda^T*k, see page 8 in http://people.ku.edu/~gbohling/cpe940/Kriging.pdf
                    const size_t idx = sub_indices.at(i);
                    var[i] = sill - to_arma_vec::from(weights.row(i)*cov.col(idx)).at(0);
                }
            }

            /** Destination D must provide
             * \tparam const arma::uvec& source_sort_order() const: sort order of distances to valid sources
             * \tparam const arma::vec& source_covariances() const: covariances for valid sources
             * \tparam void set_weights(const arma::vec& weights, const arma::uvec& indices): Set the weights for the stations given in indices
             */
            template<typename D>
            void calculate_local_weights_data(const arma::mat& station_covariances,
                                              size_t max_neighbours,
                                              bool simple_kriging,
                                              D& dest) {
                // Double sample space for the distance pre-sort.
                const size_t num_valid_stations = station_covariances.n_rows;
                const size_t n_dist_neighbours = std::min(num_valid_stations, 2*max_neighbours);

                // First kriging problem, to go from distances to weights
                arma::uvec sub_indices = dest.source_sort_order().subvec(0, n_dist_neighbours - 1);
                arma::vec w;
                if (simple_kriging)  {
                    w = arma::solve(station_covariances(sub_indices, sub_indices), dest.source_covariances()(sub_indices));
                }
                else {
                    //w += (1 - arma::sum(w))/w.n_rows; // TODO: Can we do this instead of solving the full problem?
                    // Ordinary Kriging with Lagrange multiplier that enforces \sum w_i = 1.0.
                    const arma::uword m = n_dist_neighbours + 1;
                    arma::mat K(m, m);
                    arma::vec k(m);
                    K.submat(0, 0, m - 2, m - 2) = station_covariances(sub_indices, sub_indices);
                    K.row(m - 1) = arma::rowvec(m , arma::fill::ones);
                    K.col(m - 1) = arma::vec(m, arma::fill::ones);
                    K.at(m - 1, m - 1) = 0.0;
                    k.subvec(0, m - 2) = dest.source_covariances()(sub_indices);
                    k.at(m - 1) = 1.0;
                    arma::vec l = arma::solve(K, k);
                    w = l.subvec(0, m - 2);
                }
                // TODO: This method is greatly simplyfied as compared to the intention of the original method, but equivalent to
                // the actual implementation of it. Figure out if solving a second kringing problem is needed, and if so, recalc 
                // indices.
                dest.set_weights(w, sub_indices);
            }

            static inline void gamma_transform(const std::vector<double>& x,
                                               const std::vector<double>& m,
                                               const std::vector<double>& cv,
                                               const std::vector<double>& p0,
                                               std::vector<double>& res) {
                typedef boost::math::policies::policy<boost::math::policies::digits10<10>> acc_policy;
                typedef boost::math::gamma_distribution<double, acc_policy> gamma_dist;
                typedef boost::math::normal_distribution<double, acc_policy> normal_dist;
                normal_dist n_d;
                res.reserve(x.size());
                for(size_t i = 0; i < x.size(); ++i) {
                    const double a = 1.0/(cv[i]*cv[i]);
                    const gamma_dist g_d(a, m[i]/a);
                    const double y = p0[i] + (1.0 - p0[i])*boost::math::cdf(g_d, x[i]);
                    res.emplace_back(boost::math::quantile(n_d, y));
                }
            }

            static inline void inv_gamma_transform(const std::vector<double>& x,
                                                   const std::vector<double>& m,
                                                   const std::vector<double>& cv,
                                                   const std::vector<double>& p0,
                                                   std::vector<double>& res) {
                typedef boost::math::policies::policy<boost::math::policies::digits10<3>> acc_policy;
                typedef boost::math::gamma_distribution<double, acc_policy> gamma_dist;
                typedef boost::math::normal_distribution<double, acc_policy> normal_dist;
                normal_dist n_d;
                res.reserve(x.size());
                for(size_t i = 0; i < x.size(); ++i) {
                    const double a = 1.0/(cv[i]*cv[i]);
                    const gamma_dist g_d(a, m[i]/a);
                    const double y = (boost::math::cdf(n_d, x[i]) - p0[i])/(1.0 - p0[i]);
                    res.emplace_back(boost::math::quantile(g_d, y));
                }
            }

            /** \brief Calculate mean of values restriced to indices
             */
            static inline double subset_mean(const std::vector<double>& values,
                                             const std::vector<unsigned int> indices) {
                if (indices.size() == 0) throw semivariogram_error();
                double sum = 0.0;
                for (size_t i = 0; i < indices.size(); ++i)
                    sum += values[indices[i]];
                return sum/indices.size();
            }

            /** \brief Calculate mean of values
             */
            static inline double mean(const std::vector<double>& values) {
                if (values.size() == 0) throw semivariogram_error();
                double sum = 0.0;
                for (size_t i = 0; i < values.size(); ++i)
                    sum += values[i];
                return sum/values.size();
            }

            /** \brief Calculate variance of values restricted to indices
             */
            static inline double subset_var(const std::vector<double>& values,
                                            const std::vector<unsigned int>& indices) {
                if (indices.size() < 2) throw semivariogram_error();
                const double mean = subset_mean(values, indices);
                double sum = 0.0;
                for (size_t i = 0; i < indices.size(); ++i) {
                    const double tmp = values[indices[i]] - mean;
                    sum += tmp*tmp;
                }
                return sum/(indices.size() - 1);
            }

            /** \brief Calculate variance of all values
             */
            static inline double variance(const std::vector<double>& values) {
                if (values.size() < 2) throw semivariogram_error();
                const double _mean = mean(values);
                double sum = 0.0;
                for (size_t i = 0; i < values.size(); ++i) {
                    const double tmp = values[i] - _mean;
                    sum += tmp*tmp;
                }
                return sum/(values.size() - 1);
            }

            template<typename M, typename P>
            void gibbs_sampler(size_t num_iter,
                               size_t num_var_update,
                               arma::mat& weights,
                               arma::mat& covariances,
                               arma::mat& distances,
                               std::vector<unsigned int>& zero_inds,
                               std::vector<double>& max_zero_transformed_precipitation,
                               P& semi_var,
                               std::vector<double>& transformed_precipitation) {
                std::vector<double> mu;
                std::vector<double> var;
                typedef boost::math::policies::policy<boost::math::policies::digits10<10>> acc_policy;
                typedef boost::math::normal_distribution<double, acc_policy> normal_dist;
                normal_dist n_d;

                boost::mt19937 mg; // Mersenne Twister random number engine
                boost::uniform_01<boost::mt19937> r_gen(mg); // Uniform [0,1) generator

                for (size_t iter = 0; iter < num_iter; ++iter) {
                    std::cout << "iter = " << iter << std::endl;
                    calculate_gibbs_estimates(zero_inds, transformed_precipitation, weights, covariances, semi_var.sill(), mu, var);
                    std::cout << "var[0] = " << var[0] << std::endl;
                    for (size_t j = 0; j < zero_inds.size(); ++j) {
                        const double std_dev = sqrt(var[j]);
                        const double p_upper = boost::math::cdf(n_d, (max_zero_transformed_precipitation[j] - mu[j])/std_dev);
                        const double q = boost::math::quantile(n_d, p_upper*r_gen());
                        transformed_precipitation[zero_inds[j]] = mu[j] + q*std_dev;
                    }
                    if (iter < num_var_update) {
                        const double sill = variance(transformed_precipitation) + semi_var.nugget();
                        //std::cout << "Gibbs sampler new sill = " << sill << std::endl;
                        semi_var.set_sill(sill);
                        M::covariance(distances, semi_var, covariances);
                        calculate_gibbs_weights(covariances, zero_inds, weights);
                        std::cout << weights;
                    }
                }
            }


            template<typename M, typename D, typename P, typename SV>
            void conditional_random_field_simulation(size_t num_realizations,
                                                     const P& parameter,
                                                     const D& destination_begin,
                                                     const D& destination_end, 
                                                     const SV& semi_var,
                                                     std::vector<double>& gauss_field) {

                // Call R-functionality to run random field simulation. No const-correctness implemented there, unfortunately.

                std::vector<double> grid_info = {parameter.x_min() + 0.5*parameter.dx(), 
                                                 parameter.x_max() - 0.5*parameter.dx(), 
                                                 parameter.dx(), 
                                                 parameter.y_min() + 0.5*parameter.dy(), 
                                                 parameter.y_max() - 0.5*parameter.dy(), 
                                                 parameter.dy()};
                double T = 1.0;
                int dim = 2;
                int lx = 3;
                int grid = 1;
                int with_time = 0;
                int* model_number = GetModeNr_Mod(M::name());
                std::vector<double> sim_parameter = M::parameter(semi_var);
                int n_cov = 1;
                int anisotropy = 0;
                int op = 0;
                int* method_number = GetMethodNr_Mod("circulant embedding");
                int distribution = 0;
                int key = 0;
                int error = 0;

                // TODO: Do we really need to check each and every result to know if we had success, and wrap this code in a while-loop?

                InitSimulateRF(grid_info.data(),    // X
                               &T,                  //
                               &dim,                //
                               &lx,                 //
                               &grid,               //
                               &with_time,          //
                               model_number,        //
                               sim_parameter.data(),//
                               sim_parameter.size(),//
                               &n_cov,              //
                               &anisotropy,         //
                               &op,                 //
                               method_number,       //
                               &distribution,       //
                               &key,                //
                               &error);             //

                int modus = 0;
                double lambda = 0.0; // Mean value of field we want to simulate.

                StoreTrend_mod(&key,                //
                               &modus,               //
                               &lambda,              //
                               &error);              //

                int paired = 0;
                int n_r = num_realizations;

                std::fill(begin(gauss_field), end(gauss_field), 0.0);
                DoSimulateRF(&key, &n_r, &paired, gauss_field.data(), &error);
                DeleteKey(&key);

                const double std_dev = sqrt(semi_var.nugget());
                //if (std_dev > 0.0) 
                //    for_each(begin(gauss_field), end(gauss_field), [&std_dev] (double& value) { value += rcore::rnorm(0.0, std_dev); });




                return;
            }

            class semi_variogram {
              private:
                double _nugget = 0.0;
                double _sill = 0.0;
                double _range = 0.0;
                double _shape = 0.0;
                double _anisotropy_ratio = 0.0;
                double _anisotropy_direction = 0.0;
                arma::mat anisotropy_distances;
                arma::mat station_covariances;
                arma::mat station_angles;
              public:
                double range() const { return _range; }
                double shape() const { return _shape; }
                double sill() const { return _sill; }
                double nugget() const { return _nugget; }
                double anisotropy_ratio() const { return _anisotropy_ratio; }
                double anisotropy_direction() const { return _anisotropy_direction; }
                void set_range(double value) { _range = value; }
                void set_shape(double value) { _shape = value; }
                void set_sill(double value) { _sill = value; }
                void set_nugget(double value) { _nugget = value; }

                semi_variogram(double nugget, double sill, double range) : _nugget(nugget), _sill(sill), _range(range) { /* Do nothing */ }
                semi_variogram(double nugget, double sill, double range, double shape)
                  :  _nugget(nugget), _sill(sill), _range(range), _shape(shape) { /* Do nothing */ }

                /*
                void set_anisotropy_values(const double ratio, const double direction, const S& source_begin, const S& source_end) {
                    if (ratio < 1.0 || direction < 0.0 || direction >= 180.0)
                        throw semivariogram_error();
                    anisotropy_ratio = ratio;
                    anisotropy_direction = direction;
                    arma::mat anisotropy_distances;
                    grf::calculate_anisotropy_distances(anisotropy_distances, source_begin, source_end,
                                                        anisotropy_ratio, anisotropy_direction);
                    if (sill > 0.0 && nugget >= 0.0 && (shape > 0.0))
                        M::covariance(station_covariances, anisotropy_distances, nugget, sill, range, shape);
                    else
                        station_covariances.fill(0.0);
                }
                */

                template<typename M>
                void calculate_station_covariances() {
                    M::covariance(station_covariances, anisotropy_distances, _nugget, _sill, _range, _shape);
                }
            };



        /** \brief Gaussian Random Field Interpolation
         */
        template<class M, class TSA, class S, class D, class T, class P>
        void run_interpolation(S source_begin, S source_end,
                              D destination_begin, D destination_end,
                              const T& time_axis, const P& parameter) {
            const size_t num_sources = std::distance(source_begin, source_end);
            std::vector<TSA> source_accessors;
            source_accessors.reserve(num_sources);
            std::for_each(source_begin, source_end, [&] (const typename S::value_type& source)
                          { source_accessors.emplace_back(TSA(source.precipitations(), time_axis)); });

            const size_t num_timesteps = time_axis.size();
            std::vector<size_t> valid_indices; valid_indices.reserve(num_sources);
            std::vector<size_t> prev_valid_indices; prev_valid_indices.reserve(num_sources);
            typedef typename S::value_type* s_p_type;
            std::vector<s_p_type> valid_sources;

            // Variables used to store the values needed to change from gamma to normal distribution
            // Mean, cv and p0 need to be determined based on a separate pass through the source data.
            std::vector<double> precipitations;
            std::vector<double> precipitations_mean;
            std::vector<double> precipitations_cv;
            std::vector<double> precipitations_p0;
            precipitations.reserve(num_sources);
            precipitations_mean.reserve(num_sources);
            precipitations_cv.reserve(num_sources);
            precipitations_p0.reserve(num_sources);

            for (size_t t_step = 0; t_step < num_timesteps; ++t_step) {
                std::vector<unsigned int> pos_inds;
                std::vector<unsigned int> zero_inds;
                pos_inds.reserve(num_sources);
                zero_inds.reserve(num_sources);
                valid_sources.clear();
                auto source_iter = source_begin;
                for(size_t i = 0; i < num_sources; ++i) {
                    const double precipitation = source_accessors[i].value(t_step);
                    const double precipitation_mean = 0.5; // TODO: Estimate!
                    const double precipitation_cv = 0.2; // TODO: Estimate!
                    const double precipitation_p0 = 0.3; // TODO: Estimate!
                    if (std::isfinite(precipitation)) {
                        valid_indices.emplace_back(i);
                        valid_sources.emplace_back(&(*source_iter));
                        precipitations.emplace_back(precipitation);
                        precipitations_mean.emplace_back(precipitation_mean);
                        precipitations_cv.emplace_back(precipitation_cv);
                        precipitations_p0.emplace_back(precipitation_p0);
                        if (precipitation > 0.0) pos_inds.emplace_back(i);
                        else zero_inds.emplace_back(i);
                    }
                    ++source_iter;
                }
                std::vector<double> transformed_precipitation;
                grf::gamma_transform(precipitations,
                                     precipitations_mean,
                                     precipitations_cv,
                                     precipitations_p0,
                                     transformed_precipitation);
                const double positive_mean = pos_inds.size() > 0 ? grf::subset_mean(transformed_precipitation, pos_inds) : 0.0;
                const double positive_var = pos_inds.size() > 1 ? grf::subset_var(transformed_precipitation, pos_inds) : 0.0;

                const double zero_var = 0.35; // TODO: Why?
                const double zero_mean = -0.75; // TODO: Why?
                const size_t n_z = zero_inds.size();
                const size_t n_p = pos_inds.size();

                const double tmp = (positive_mean - zero_mean)/(n_p + n_z);
                const double total_var = (positive_var*n_p + zero_var*n_z)/(n_p + n_z) + n_p*n_z*tmp*tmp;
                const double nugget = parameter.nugget();

                arma::mat station_distances;
                grf::calculate_anisotropy_distances(begin(valid_sources), end(valid_sources),
                                                    parameter.anisotropy_ratio(), parameter.anisotropy_direction(), station_distances);
                grf::semi_variogram semi_var(nugget, total_var + nugget, parameter.range());

                typedef boost::math::policies::policy<boost::math::policies::digits10<3>> acc_policy;
                typedef boost::math::normal_distribution<double, acc_policy> normal_dist;
                normal_dist n_d;

                std::vector<double> max_zero_transformed_precipitation; // Maximum values in gaussian space for zero precip stations
                max_zero_transformed_precipitation.reserve(zero_inds.size());
                if (n_z > 0 ) {
                    for (size_t i = 0; i < n_z; ++i) {
                        const double t_p = transformed_precipitation[zero_inds[i]];
                        max_zero_transformed_precipitation.emplace_back(t_p);
                        const double p_norm = boost::math::cdf(n_d, t_p);
                        // Initialize to truncated expectation
                        transformed_precipitation[zero_inds[i]] = boost::math::quantile(n_d, p_norm/2.0);
                    }
                    arma::mat krig_weights;
                    arma::mat station_covariances;
                    M::covariance(station_distances, semi_var, station_covariances);
                    grf::calculate_gibbs_weights(station_covariances, zero_inds, krig_weights);
                    grf::gibbs_sampler<M>(5, 0, krig_weights, station_covariances, station_distances,
                                          zero_inds, max_zero_transformed_precipitation,
                                          semi_var, transformed_precipitation);
                }
                arma::mat source_dest_dists;
                if (valid_indices != prev_valid_indices) {
                    grf::calculate_anisotropy_distances(begin(valid_sources), 
                                                        end(valid_sources),
                                                        destination_begin, 
                                                        destination_end, 
                                                        parameter.anisotropy_ratio(), 
                                                        parameter.anisotropy_direction(), 
                                                        source_dest_dists);
                }
                arma::mat source_dest_covariances;
                M::covariance(source_dest_dists, semi_var, source_dest_covariances);
                auto dest_iter = destination_begin;
                const size_t n_dests = std::distance(destination_begin, destination_end);
                for (size_t i = 0; i < n_dests; ++i)
                    grf::calculate_local_weights_data(source_dest_covariances, parameter.max_neighbours(), true, *(dest_iter++));

                // Random field simulation

                arma::vec computed_mean(n_dests, arma::fill::zeros);
                arma::vec computed_std_dev(n_dests, arma::fill::zeros);
                arma::uvec positives(n_dests, arma::fill::zeros);
                typedef arma::conv_to<arma::vec> to_arma_vec;

                for (size_t i = 0; i < parameter.number_of_random_field_realizations(); ++i) {

                    std::vector<double> gauss_prec;
                    std::vector<double> gamma_prec;
                    grf::conditional_random_field_simulation<M>(2, parameter, destination_begin, destination_end, gauss_prec);
                    grf::inv_gamma_transform(gauss_prec, precipitations_mean, precipitations_cv, precipitations_cv, gamma_prec);

                    computed_mean += to_arma_vec::from(gamma_prec);
                    computed_std_dev += to_arma_vec::from(gamma_prec)*to_arma_vec::from(gamma_prec);
                    positives.elem(arma::find(to_arma_vec::from(gamma_prec) > 0.0)) += 1;


                }


                    //for_each(destination_begin, destination_end, [parameter&, weights&] (const typename D::value_type& destination) {
                    //    arma::mat w;
                    //    grf::calculate_local_weights_data(dest, station_covars, paramter.max_neighbours(), true, w);

                    //});

                //
                // Garbageish from this point on

                if (valid_indices != prev_valid_indices) {
                    // First iteration solution; only save previous index set, and recaluclate when the
                    // set of valid stations change.
                }
            //    for (D d = destination_begin; d != destination_end; ++d)
            //        d->set_precipitation(t_step, E_temp_post.at(dist++));
                prev_valid_indices.swap(valid_indices);
                valid_indices.clear();
            }
        }
		} // End namespace grf

    } // Namespace core
} // shyft

/* vim: set filetype=cpp: */
