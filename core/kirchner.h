
#include <cmath>

#ifndef _MSC_VER
// SiH: On ubuntu 15.04, with boost 1.59, gcc 4.9.2 this was needed
// to avoid undefined static_gcd_type
#include <boost/math_fwd.hpp>
typedef boost::math::static_gcd_type static_gcd_type;
#endif
#include <boost/numeric/odeint.hpp>
#include "timeseries.h"

#pragma once

namespace shyft {
    namespace core {
        namespace kirchner {

            /** \brief Compute the average of the solution using a simple trapezoidal rule
             *
             * During the integration of the Kirchner model from t_0 to t_1, this routine uses the intermediate solution
             * steps at t_i where t_0 < t_i <= t_1, implicitly given by the tolerance of the ode solver, to apply a simple
             * trapezoidal rule to compute the average. This routine is not very costly, as it only uses the already computed
             * intermediate solution steps during the calculations.
             * \tparam S stepper
             */
            template<class S>
			class trapezoidal_average {
                private:
                    double area = 0.0;
                    double f_a = 0.0;; // Left hand side of next integration subinterval
                    double t_start = 0.0; // Start of integration period
                    double t_a = 0.0; // Left hand side time of next integration subinterval
                public:
                    trapezoidal_average(S* stepper) {}

                    /** \brief initialize must be called to reset states before being used during ode integration.
                     */
                    void initialize(double f0, double t_start) {
                        this->f_a = f0;
                        this->t_start = t_start;
                        t_a = t_start;
                        area = 0.0;
                    }

                    /** \brief Add contribution to average using a simple trapezoidal rule
                     *
                     * See: http://en.wikipedia.org/wiki/Numerical_integration
                     */
                    void add(double f, double t) {
                        area += 0.5*(f_a + f)*(t - t_a);
                        f_a = f;
                        t_a = t;
                    }

                    double result() const { return area/(t_a - t_start); }
            };

            /** \brief Compute the average of the solution using a n step composite trapezoidal rule
             *
             * During the integration of the Kirchner model from \f$t_0\f$ to \f$t_1\f$ using a dense solver, we can
             * interpolate the solution on the time sub-interval \f$[t_i, t_i + dt_i]\f$, where \f$dt_i\f$ is implicitly determined
             * by the tolerance of the ode solver for some internal time step \f$t_i\f$, where \f$t_{i+1} = t_i + dt_i\f$.
             * At this sub interval, we use the composite trapezoidal rule to compute the average of each sub-interval using
             * \f$n\f$ sub-sub-intervals. This is a costly routine, requiring \f$n-1\f$ interpolations at each sub-interval.
             * \tparam S Stepper that provides:
             *   -# .calc_state ..
             *   -# ::state_type
             * \sa Kirchner
             */
            template<class S> class
			composite_trapezoidal_average {
                private:
                    double area = 0.0;  // Integrated area
                    double f_a = 0.0;
                    double t_start = 0.0;
                    double t_a = 0.0;
                    const size_t n{5}; // Number of internal subintervals
                    S* stepper;
                    typename S::state_type x; // Storage for the intermediate state
                public:
                    composite_trapezoidal_average(S* stepper): stepper(stepper) {}

                    /** \brief initialize must be called to reset states before being used during ode integration.
                     */
                    void initialize(double f0, double t_start) {
                        this->f_a = f0;
                        this->t_start = t_start;
                        t_a = t_start;
                        area = 0.0;
                    }

                    /** \brief Add contribution to average using a composite trapezoidal rule
                     *
                     * For more information regarding the numerical method, see: http://en.wikipedia.org/wiki/Numerical_integration
                     */
                    void add(double f, double t) {
                        const double dt = (t - t_a)/n;
                        double sum = (f_a + f)*0.5;
                        for (size_t k=1; k < n; ++k) {
                            stepper->calc_state(t_a + k*dt, x);
                            // The Kirchner method is solving the log transform of the
                            // original problem formulation, hence the result at the
                            // internal steps are inverted back by applying the
                            // exponential function at each sub time steps.
                            sum += exp(x /*.at(0)*/);
                        }
                        area += sum*(t - t_a)/n;
                        f_a = f;
                        t_a = t;
                    }

                    double result() const { return area/(t_a - t_start); }
            };


			/** \brief kirchner parameters as defined in reference
			 *
			 * In operation, these parameters could be estimated based on time
			 * periods when precipitation and evapo-transpiration are much smaller
			 * than then discharge.
			 * \note reasonable default values are provided
			 */
			struct parameter {
				double c1 = -2.439;
				double c2 = 0.966;
				double c3 = -0.10;
				parameter(double c1=-2.439,double c2=0.966,double c3=-0.1):c1(c1),c2(c2),c3(c3){}
			};


			struct state {
				double q=0.0001; //< water content in [mm/h], it defaults to 0.0001 mm, zero is not a reasonable valid value
				state(double q=0.0001):q(q){}
			};


			struct response {
				double q_avg = 0.0; //< average discharge over time-step in [mm/h]
			};


			/** \brief Kirchner model for computing the discharge based on precipitation and evapotranspiration data.
			 *
			 * This algorithm is based on the log transform of the ode formulation of the time change in discharge as a function
			 * of measured precipitation, evapo-transpiration and discharge, i.e. equation 19 in the publication
			 * "Catchments as simple dynamic systems: Catchment characterization, rainfall-runoff modeling, and doing
			 * hydrology backward" by James W. Kirchner, published in Water Resources Research, vol. 45, W02429,
			 * doi: 10.1029/2008WR006912, 2009.
			 *
			 * \tparam AC Average Computer type, implementing the interface:
			 *    - AC(S* stepper) --> Constructor taking a Kirchner::dense_stepper_type as argument.
			 *    - AC.initialize(double f0, double t_start) --> void, initialize average computer
			 *      with initial function value and start time.
			 *    - AC.add(double f, double t) --> void, add contribution f at next time t.
			 * \tparam P Parameter type, implementing the interface:
			 *    - P.c1 --> double, first parameter in the Kirchner model
			 *    - P.c2 --> double, second parameter in the Kirchner model
			 *    - P.c3 --> double, third parameter in the Kirchner model
			 *    \sa kirchner::TrapezoidalAverage \sa kirchner::CompositeTrapezoidalAverage
			 */
			template<template<class> class AC, class P>
			class calculator {
			  public:
				//typedef arma::vec::fixed<1> state_type;
				typedef double state_type;
				typedef boost::numeric::odeint::result_of::make_dense_output<
						boost::numeric::odeint::runge_kutta_dopri5<state_type> >::type dense_stepper_type;
				typedef std::pair<dense_stepper_type::time_type, dense_stepper_type::time_type> interval_type;
			  private:
				dense_stepper_type dense_stepper = boost::numeric::odeint::make_dense_output(1.0e-7, 1.0e-8,// early tests indicate that 1.0e-6 and 1.0.e-7 gives robust results
												   boost::numeric::odeint::runge_kutta_dopri5<state_type>());
				AC<dense_stepper_type> average_computer{&dense_stepper};
				const P param;

				/** \brief Sensitivity function g(q)
				 *
				 * The sensitivity function expresses the sensitivity of discharge to changes in the storage. As
				 * the changes is the storage can be expressed as a (pure) function of q, it can be estimated from
				 * observational data (p, e, and q).
				 */
				double g(double ln_q) const {
					return exp(param.c1 + param.c2*ln_q + param.c3*ln_q*ln_q);
				}

				/** \brief Log transform of equation 18 in Kirchner, aka equation 19.
				 *
				 * When solving the Kirchner model, the log transform formulation is desireable due to numercial
				 * instabilities of the original formulation, \f$\frac{dQ}{dt} = g(Q)(P-E-Q)\f$.
				 */
				double log_transform_f(double ln_q, double p, double e) const {
					const double gln_q = g(ln_q);
					if (gln_q < 1.0e-30) /// TODO:a numerical challenge, but why is 1.0e-30 a suitable number?
						return 0.0;
					const double rest = (p - e)*exp(-ln_q) - 1.0;
					return gln_q*rest;
				}

			  public:
				calculator(const P& param) : param(param) { /* Do nothing */ }

				calculator(double abs_err, double rel_err, P& param)
					: dense_stepper(boost::numeric::odeint::make_dense_output(abs_err, rel_err,
						boost::numeric::odeint::runge_kutta_dopri5<state_type>())),
					  average_computer(&dense_stepper), param(param) {}


				/** \brief step Kirchner model forward from time t0 to time t1
				 * \note if the supplied q (state) is less than min_q(0.00001, it represents mm water..), it is forced to min_q to ensure numerical stability
				 */
				void step(shyft::timeseries::utctime T0, shyft::timeseries::utctime T1, double& q, double& q_avg, double p, double e) {
					state_type x_tmp;
					const double min_q= 0.00001;// ref note above
					if(q<min_q) q= min_q;
					x_tmp = log(q); // Log transform

					double t0 = 0.0;
					double t1 = double(T1 - T0)/deltahours(1); // units in kirchner is volume mm/hour, hour like 1..n
					dense_stepper.initialize(x_tmp, t0, t1 - t0);
					average_computer.initialize(q, t0);
					double current_time = dense_stepper.current_time();
					while (current_time < t1) {
						dense_stepper.do_step([this, p, e](const state_type x, state_type& dxdt, double) {
											  dxdt= log_transform_f(x, p, e); });
						current_time = dense_stepper.current_time();
						if (current_time < t1)
							average_computer.add(exp(dense_stepper.current_state()), current_time);
					}
					dense_stepper.calc_state(t1, x_tmp);
					q = exp(x_tmp); // Invert log transform
					average_computer.add(q, t1);
					q_avg = average_computer.result();
				}

			};
		} // End namespace kirchner
    };
};

/* vim: set filetype=cpp: */
