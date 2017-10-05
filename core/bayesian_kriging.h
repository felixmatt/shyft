#pragma once
#ifdef SHYFT_NO_PCH
#include <string>
#include <vector>
#include <iterator>
//#include <cmath>
//#include <limits>
#include <stdexcept>
#include <armadillo>
#endif // SHYFT_NO_PCH

#include "time_series.h"

/**
 * contains all BayesianKriging stuff, like concrete useful Parameters and the templated BTK algorithm
 *
 */


namespace shyft {
    namespace core {
		namespace bayesian_kriging {

	        namespace utils {

	            /** \brief Scaled distance between points
	             *
	             * Multiply elevation distance with a scale factor and compute the corresponding difference between the points p1 and p2.
	             *
	             * \tparam P
	             * Parameter that supplies:
	             *  -# P.zscale() const
	             * \tparam GP
	             *  Parameter GeoPoint, that should have the zscaled_distance(GP p1,GPp2 ,double zscale) method defined
	             * \param p1 first point
	             * \param p2 second point
	             * \param 'param' the parameter containing zscale used
	             * \return scaled distance.
	             */
	            template<class P,class GP> double scaled_dist(const GP& p1, const GP& p2, const P& param)
	            {
	                return GP::zscaled_distance(p1,p2,param.zscale());
	            }

	            template<class P> double zero_dist_cov(const P& parameter)
	            {
	                return parameter.sill() - parameter.nug();
	            }

	            // Cov(d) = (sill - nug)*exp(-d/range), where d is a height-scaled measure of distance.
	            // TODO: Check out http://www.seas.upenn.edu/~ese502/NOTEBOOK/Part_II/4_Variograms.pdf
	            template<class TP, class P> void cov(const TP& dists, TP& cov, const P& parameter)
	            {
	                cov = (parameter.sill() - parameter.nug())*arma::exp(-dists/parameter.range());
	            }

	            /** \brief Build the covariance matrices from sources and destinations
	             *
	             * The Bayesian Temperature Kriging algorithm needs source-source and source-destination covariance
	             * matrices. Define \f$p_1 = (x_1, y_1, z_1)\f$  and \f$p_2 = (x_2, y_2, z_2)\f$, two 3D points for the
	             * source/destination. Furthermore, define \f$d=\sqrt{(x_1-x_2)^2 + (y_1-y_2)^2 + (l(z_1-z_2))^2}\f$, where \f$l\f$
	             * is a height scaling factor. Then,
	             * \f[
	             *  cov(p_1, p_2) = (s-n)e^{-\frac{d}{r}}
	             * \f]
	             * where \f$s\f$ is the sill, \f$n\f$ is the nugget and \f$r\f$ is the range of the semivariogram.
	             *
	             * \tparam S
	             *  Source (location)
	             *  Type that implements:
	             *    -# S.mid_point() const --> 3D point, GeoPoint, specifies the location
	             * \tparam D
	             *  Destination (location/area)
	             *  Type that implements:
	             *    -# D.mid_point() const --> 3D point, GeoPoint, specifies the location
	             *    -# D.set_temperature() --> 3D point, arma::vec3, specifies the location
	             * \tparam P
	             * Parameters for the algorithm, that supplies:
	             *    -# parameter.sill() const --> Sill value of the semivariogram
	             *    -# parameter.nug() const --> Nugget value of the semivariogram
	             *    -# parameter.range() const --> Range of the semivariogram
	             *    -# P.zscale() const --> Height scaling factor
	             * \param source_begin,source_end is specifying the range for the input sources
	             * \param destination_begin,destination_end is specifying the range for the target of the interpolation
	             * \param parameter contains the parameter for the function
	             * \param K
	             * Symmetric, positive semi-definite source-source covariance matrix
	             * \param k
	             * source-destination covariance matrix
	             *
	             */
	            template<class S, class D, class P> void build_covariance_matrices(S source_begin, S source_end,
	                                                                               D destination_begin, D destination_end,
	                                                                               const P& parameter, arma::mat& K, arma::mat& k)
	            {
	                const arma::uword n = (arma::uword)std::distance(source_begin, source_end);
					const arma::uword m = (arma::uword)std::distance(destination_begin, destination_end);
	                arma::vec source_dists(n*(n-1)/2);
	                arma::vec K_nz(n*(n-1)/2);
	                arma::uword idx = 0;
	                for (auto i=source_begin; i != source_end; ++i)
	                    for (auto j=std::next(i); j != source_end; ++j)
	                        source_dists.at(idx++) = scaled_dist(i->mid_point(), j->mid_point(), parameter);

	                cov(source_dists, K_nz, parameter);
	                idx = 0;
	                K.eye(n,n);
					for (arma::uword i = 0; i<n; ++i)
						for (arma::uword j = i + 1; j<n; ++j)
	                        K.at(i, j) = K_nz.at(idx++);
	                K.diag() *= zero_dist_cov(parameter);
	                K = arma::symmatu(K);

	                arma::mat k_dists(n,m);
					arma::uword i = 0;

	                for (auto  s=source_begin; s != source_end; ++s) {
						arma::uword j = 0;
	                    for (auto  d=destination_begin; d != destination_end; ++d)
	                        k_dists.at(i, j++) = scaled_dist(s->mid_point(), d->mid_point(), parameter);
						++i;
	                }
	                cov(k_dists, k, parameter);
	            }

	            /** \brief Build the elevation matrices from  sources and destinations
	             *
	             * Build the elevation matrices needed by the Bayesian Temperature Kriging algorithm.
	             *
	             * \tparam S
	             *  Source (location)
	             *  Type that implements:
	             *    -# S.mid_point() const --> 3D point,GeoPoint, specifies the location
	             * \tparam D
	             *  Destination (location/area)
	             *  Type that implements:
	             *    -# D.mid_point() const --> 3D point, GeoPoint, specifies the location
                 * \param source_begin,source_end is specifying the range for the input sources
	             * \param destination_begin,destination_end is specifying the range for the target of the interpolation
	             * \param F
	             * Matrix \f$ F\in{} \mathcal{R}^{n,2} \f$, where \f$n\f$ is the number of sources.
	             * \param f
	             * Matrix \f$ F\in{} \mathcal{R}^{2,m} \f$, where \f$m\f$ is the number of destinations.
	             */
	            template<class S, class D> void build_elevation_matrices(S source_begin, S source_end, D destination_begin,
	                                                                     D destination_end, arma::mat& F, arma::mat& f)
	            {
					const arma::uword n = (arma::uword)std::distance(source_begin, source_end);
					const arma::uword m = (arma::uword)std::distance(destination_begin, destination_end);
	                F.set_size(n, 2);
	                f.set_size(2, m);
	                F.col(0) = arma::ones(n);
	                f.row(0) = arma::rowvec(m, arma::fill::ones);
					arma::uword i = 0;
	                std::for_each(source_begin, source_end, [&] (const typename S::value_type& source)
	                                                       { F.at(i++, 1) = source.mid_point().z; });
	                i = 0;
	                std::for_each(destination_begin, destination_end, [&] (const typename D::value_type& dest)
	                                                                 { f.at(1, i++) = dest.mid_point().z; });

	            }
	        } // End namespace btk_utils

	        /** \brief Simple BTKParameter class with constant temperature gradient
	         * \sa BayesianKriging
	         */
	        class const_parameter {
	            double gradient = -0.006; //< Prior expectation of temperature gradient [C/m]
	            double gradient_sd = 0.0025; //< Prior standard deviation of temperature gradient in [C/m]
	            double sill_value = 25.0; //< Value of semivariogram at range
	            double nug_value = 0.5; //< Nugget magnitude
	            double range_value = 200000.0; //< Point where semivariogram flattens out
	            double zscale_value = 20.0; //< Height scale used during distance computations.
	          public:
	              //< Construct a complete parameter with some reasonable values
	            const_parameter() {}
	            const_parameter(double temperature_gradient, double temperature_gradient_sd)
	              : gradient(temperature_gradient/100),
	                gradient_sd(temperature_gradient_sd/100) {}
	            const_parameter(double temperature_gradient, double temperature_gradient_sd, double sill, double nugget, double range, double zscale)
	              : gradient(temperature_gradient/100),
	                gradient_sd(temperature_gradient_sd/100),
	                sill_value(sill), nug_value(nugget), range_value(range), zscale_value(zscale) { /* Do nothing */ }

	            //< returns the fixed period independent prior gradient parameter
	            const double temperature_gradient(utcperiod p) const {
	                return gradient;
	            }
	            //< returns the standart deviation parameter
	            const double temperature_gradient_sd() const { return gradient_sd;}

	            double sill() const { return sill_value; }
	            double nug() const { return nug_value; }
	            double range() const { return range_value; }
	            double zscale() const { return zscale_value; }
	        };


	        /** \brief BTKParameter class with time varying gradient based on day no.
	         *
	         * \sa BayesianKriging
	         */
	        class parameter {
	            double gradient_sd = 0.0025; //< Prior standard deviation of temperature gradient in [C/m]
	            double sill_value = 25.0; //< Value of semivariogram at range
	            double nug_value = 0.5; //< Nugget magnitude
	            double range_value = 200000.0; //< Point where semivariogram flattens out
	            double zscale_value = 20.0; //< Height scale used during distance computations.
	            calendar cal;
	          public:
	            parameter() {}
	            parameter(double temperature_gradient, double temperature_gradient_sd)
	              : gradient_sd(temperature_gradient_sd/100) {}
	            parameter(double temperature_gradient, double temperature_gradient_sd, double sill, double nugget, double range, double zscale)
	              : gradient_sd(temperature_gradient_sd/100),
	                sill_value(sill), nug_value(nugget), range_value(range), zscale_value(zscale) { /* Do nothing */ }

	            //< Returns temperature gradient based on day of year for the supplied period, using midpoint
	            const double temperature_gradient(utcperiod p) const {
	                const double doy = (double)cal.day_of_year((p.start + p.end)/2);
	                return 1.18e-3*sin(6.2831/365*(doy + 79.0)) - 5.48e-3;
	            }
	            const double temperature_gradient_sd() const { return gradient_sd;}

	            double sill() const { return sill_value; }
	            double nug() const { return nug_value; }
	            double range() const { return range_value; }
	            double zscale() const { return zscale_value; }
	        };


	        /** \brief Bayesian Temperature Kriging Interpolation
	         *
	         * Extracted from the Enki method BayesTKrig by Sjur Kolberg/Sintef.
	         *
	         * Scatters a set of time-series data at given source locations (X,Y,Z) to a
	         * new set of time-series data at target locations (X,Y,Z) using a Bayesian Temperature Kriging method
	         * where the vertical distances used during the semivariogram's covariance calculations are scaled with a height factor.
	         *
	         * In addition to interpolating values at the destinations, estimates on the sea level temperature and temperature
	         * gradients are computed at each time step.
	         *
	         *  Preconditions:
	         *   -# There should be at least one source location with a valid value for all t.
	         * \tparam TSA TimeSeriesAccessor
	         *  Type that takes a time source and a time axis and provides a TimeAccessor type
	         *  the TSA provides, when instantiated:
	         *    -# TSA::.value(size_t index) --> time series data at index in time axis.
	         *
	         * \tparam S
	         *  Source (location)
	         *  Type that implements:
	         *    -# S.geo_point() const --> 3D point, GeoPoint, specifies the location.
	         *    -# S.temperatures() const --> a ts that can go into the supplied TSA type as ts source
	         *
	         * \tparam D
	         *  Destination (location/area)
	         *  Type that implements:
	         *    -# D.geo_point() const --> 3D point, GeoPoint
	         *    -# D.set_temperature(size_t index, double value), function that is called to set the temperature at position index
	         *       in the time_axis.
	         * \tparam T
	         * TimeAxis providing:
	         *    -# T.size() const --> size_t, number of non-overlapping time intervals
	         *    -# T(const size_t i) const --> shyft::time_series::utctimeperiod period of a with
	         *       shyft::time_series::utctime start and shyft::time_series::utctime end.
	         * \tparam P
	         * Parameters for the algorithm, that supplies:
	         *    -# P.zscale() const --> double elevation scale factor used when computing covariances.
	         *    -# P.sill() const --> double semivariogram parameter.
	         *    -# P.nug() const --> double semivariogram parameter.
	         *    -# P.range() const --> double semivariogram parameter.
	         *    -# P.temperature_gradient() const --> prior temperature gradient. TODO: Should this be a time series?
	         *    -# P.temperature_gradient_sd() const --> prior standard deviation of temperature gradient.
	         *       TODO: Should this be a time series?
	         *   \sa BTKConstParameter \sa BTKParameter
	         *
	         */
	        template<class TSA, class S, class D, class T, class P>
	        void btk_interpolation(S source_begin, S source_end,
	                              D destination_begin, D destination_end,
	                              const T& time_axis, const P& parameter)
	        {
	            // Armadillo's notion of submatrix views are used for slicing out portions of F, k, and K
	            // to minimize the number of matrix allocations.

	            // Allocate matrices of known sizes:
	            arma::mat22 H, H_inv, G, G_inv, GH_inv;
	            arma::mat::fixed<2,1> E_beta_pri, E_beta_w_pri,/* E_beta_post,*/ beta_hat;
	            // These matrices sizes vary with the number valid sources and the number of destinations.
	            arma::mat K, k, F, f, K_inv, E_beta_w, omega, BM, T_obs, E_temp_post;

	            // Prior data
	            E_beta_pri(0, 0) = 0.0; // Old code says this is ok. TODO: Check assumption.
	            arma::mat22 eye22 = arma::diagmat(arma::vec(2, arma::fill::ones));

	            // Gather spatial data for all stations and destinations
	            utils::build_elevation_matrices(source_begin, source_end, destination_begin, destination_end, F, f);
	            utils::build_covariance_matrices(source_begin, source_end, destination_begin, destination_end, parameter, K, k);

	            // Build full operators

	            K_inv = K.i();
	            H_inv = F.t()*K_inv*F;
	            if (arma::rank(H_inv) == 1) {
	                throw std::runtime_error("The bayestian temperature kriging algorithm needs at least two sources at different heights.");
	            }
	            H = H_inv.i();
	            G_inv = H_inv;
	            G_inv.at(1, 1) += 1/(parameter.temperature_gradient_sd()*parameter.temperature_gradient_sd());
	            G = G_inv.i();
	            GH_inv = G*H_inv;
	            BM = (f - F.t()*K_inv*k).t()*(eye22 - GH_inv);
	            E_beta_w = H*F.t()*K_inv; // beta_est_weights
	            omega = k.t()*K_inv;    // krig_weights
	            // Reduced matrices
	            arma::mat F_r, E_beta_w_r, omega_r, GH_inv_r, BM_r;

	            // Matrix pointers used in the time loop
	            arma::mat *F_p=nullptr, *E_beta_w_p=nullptr, *omega_p=nullptr, *GH_inv_p=nullptr, *BM_p=nullptr;

	            const size_t num_sources = std::distance(source_begin, source_end);
	            std::vector<TSA> source_accessors;
	            source_accessors.reserve(num_sources);
	            std::for_each(source_begin, source_end, [&] (const typename S::value_type& source)
	                          { source_accessors.emplace_back(TSA(source.temperatures(), time_axis)); });
	            // Time step loop
	            const size_t num_timesteps = time_axis.size();
	            std::vector<double> temperatures;
	            temperatures.reserve(num_sources);

	            std::vector<arma::uword> valid_inds, prev_valid_inds;
	            valid_inds.reserve(num_sources);
	            for (size_t t_step=0; t_step < num_timesteps; ++t_step) {
	                temperatures.clear();
	                prev_valid_inds = valid_inds;
	                valid_inds.clear();
	                size_t idx = 0;
	                for (auto& a: source_accessors) {
	                    double v = a.value(t_step);
	                    if (std::isfinite(v)) {
	                        valid_inds.push_back((arma::uword)idx);
	                        temperatures.push_back(v);
	                    }
	                    ++idx;
	                }
	                if (valid_inds != prev_valid_inds || valid_inds.size()==0) {
	                    if (valid_inds.size() == 0) {
	                        //std::cout << "period("<< t_step <<"| " << num_timesteps << ") = " << time_axis.period(t_step) << std::endl;
	                        throw std::runtime_error(std::string("bayesian kriging temperature: No valid sources for time period, giving up.") + calendar().to_string(time_axis.period(t_step)));
	                    }
	                    if (valid_inds.size() == num_sources) {
	                        // Use full operators
	                        F_p = &F;
	                        E_beta_w_p = &E_beta_w;
	                        omega_p = &omega;
	                        GH_inv_p = &GH_inv;
	                        BM_p = &BM;
	                    } else {
	                        // Build new reduced operators from full operators
	                        arma::uvec sub_idx(valid_inds);
	                        F_r = F.rows(sub_idx);
							K_inv = K.submat(sub_idx, sub_idx).i();
	                        H_inv = F_r.t()*K_inv*F_r;
	                        H = H_inv.i();
	                        G_inv = H_inv;
	                        G_inv(1, 1) += 1/(parameter.temperature_gradient_sd()*parameter.temperature_gradient_sd());
	                        G = G_inv.i();
	                        GH_inv_r = G*H_inv;
	                        arma::mat k_red = k.rows(sub_idx);
	                        BM_r = (f - F_r.t()*K_inv*k_red).t()*(eye22 - GH_inv_r);
	                        E_beta_w_r = H*F_r.t()*K_inv; // beta_est_weights
	                        omega_r = k_red.t()*K_inv;    // krieg_weights
	                        // Assign pointers
	                        F_p = &F_r;
	                        E_beta_w_p = &E_beta_w_r;
	                        omega_p = &omega_r;
	                        GH_inv_p = &GH_inv_r;
	                        BM_p = &BM_r;
	                    }
	                }

	                // Build prior data for time step:
	                E_beta_pri(1, 0) = parameter.temperature_gradient(time_axis.period(t_step));
	                E_beta_w_pri = ((eye22 - *GH_inv_p)*E_beta_pri);

	                // Fill T_obs with valid temperatures
	                T_obs.set_size((arma::uword)valid_inds.size(), 1);
	                std::copy(std::begin(temperatures), std::end(temperatures), T_obs.begin_col(0));
	                // Core computational work here:
	                beta_hat = (*E_beta_w_p)*T_obs;
	                arma::mat T_hat = f.t()*beta_hat + (*omega_p)*(T_obs - (*F_p)*beta_hat);
	                //E_beta_post = (*GH_inv_p)*beta_hat + E_beta_w_pri;
	                E_temp_post = arma::vec(T_hat - (*BM_p)*(beta_hat - E_beta_pri));

	                arma::uword dist = 0;
	                for (D d=destination_begin; d != destination_end; ++d)
	                    d->set_temperature(t_step, E_temp_post.at(dist++));

	            }
	        }
		}
    } // End namespace core
} // End namespace shyft

/* vim: set filetype=cpp: */
