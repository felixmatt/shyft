#pragma once
///	Copyright 2012 Statkraft Energi A/S
///
///	This file is part of SHyFT.
///
///	SHyFT is free software: you can redistribute it and/or modify it under the terms of
/// the GNU Lesser General Public License as published by the Free Software Foundation,
/// either version 3 of the License, or (at your option) any later version.
///
///	SHyFT is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
/// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
/// PURPOSE. See the GNU Lesser General Public License for more details.
///
///	You should have received a copy of the GNU Lesser General Public License along with
/// SHyFT, usually located under the SHyFT root directory in two files named COPYING.txt
/// and COPYING_LESSER.txt.	If not, see <http://www.gnu.org/licenses/>.
///
/// This implementation is a slightly improved and ported version of Skaugen's snow routine
/// implemented in R, see [ref].
#ifdef SHYFT_NO_PCH
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/math/tools/minima.hpp>

#include "core_pch.h"
#endif // SHYFT_NO_PCH

#include "utctime_utilities.h"
#include "time_series.h"

namespace shyft {
    namespace core {
        namespace skaugen {

            class mass_balance_error : public std::exception {};

            typedef boost::math::policies::policy<boost::math::policies::digits10<16> > acc_policy;
            typedef boost::math::gamma_distribution<double, acc_policy> gamma_dist;

            struct statistics {
                const double alpha_0;
                const double d_range;
                const double unit_size;

                statistics(double alpha_0, double d_range, double unit_size)
                  : alpha_0(alpha_0), d_range(d_range), unit_size(unit_size) { /* Do nothing */ }

                static inline double c(unsigned long n, double d_range) {
                    return exp(-(double)n/d_range);
                }

                static inline double sca_rel_red(unsigned long u, unsigned long n, double unit_size, double nu_a, double alpha) {
                    const double nu_m = ((double)u/n)*nu_a;
                    // Note; We use m (melt) in stead of s (smelt) due to language conversion
                    // Compute: Find X such that f_a(X) = f_m(X)
                    const gamma_dist g_m(nu_m, 1.0/alpha);
                    const gamma_dist g_a(nu_a, 1.0/alpha);
                    const double g_a_mean = boost::math::mean(g_a);
                    auto zero_func = [&] (const double& x) {
                        return boost::math::pdf(g_m, x) - boost::math::pdf(g_a, x); } ;

                    double lower = boost::math::mean(g_m);
                    double upper = boost::math::tools::brent_find_minima(zero_func, 0.0, g_a_mean, 2).first;  // TODO: Is this robust enough??
                    while (boost::math::pdf(g_m, lower) < boost::math::pdf(g_a, lower))
                        lower *= 0.9;

                    boost::uintmax_t max_iter = 100;
                    boost::math::tools::eps_tolerance<double> tol(10); // 10 bit precition on result
                    typedef std::pair<double, double> result_t;
                    result_t res = boost::math::tools::bisect(zero_func, lower, upper, tol, max_iter);
                    //result_t res = boost::math::tools::toms748_solve(zero_func, lower, upper, tol, max_iter); // TODO: Boost bug causes this to crash!!
                    const double x = (res.first + res.second)*0.5; // TODO: Check that we converged
                    // Compute: {m,a} = \int_0^x f_{m,a} dx
                    const double m = boost::math::cdf(g_m, x);
                    const double a = boost::math::cdf(g_a, x);
                    return a + 1.0 - m;
                }

                double c(unsigned long n) const { return c(n, d_range); }

                double sca_rel_red(unsigned long u, unsigned long n, double nu_a, double alpha) const {
                    return sca_rel_red(u, n, unit_size, nu_a, alpha);
                }
            };

            struct parameter {
                ///<note that the initialization is not the most elegant yet, due to diffs swig-python/ms-win/gcc
                double alpha_0=40.77;
                double d_range=113.0;
                double unit_size=0.1;
                double max_water_fraction=0.1;
                double tx=0.16;
                double cx=2.5;
                double ts=0.14;
                double cfr=0.01;

                parameter(double alpha_0, double d_range, double unit_size,
                          double max_water_fraction, double tx, double cx, double ts, double cfr)
                 : alpha_0(alpha_0), d_range(d_range),
                   unit_size(unit_size), max_water_fraction(max_water_fraction),
                   tx(tx), cx(cx), ts(ts), cfr(cfr) { /* Do nothing */ }
                parameter() {}
            };


            struct state {
                double nu;
                double alpha;
                double sca;
                double swe;
                double free_water;
                double residual;
                unsigned long num_units;
                state(double nu=4.077, double alpha=40.77, double sca=0.0,
                      double swe=0.0, double free_water=0.0, double residual=0.0, unsigned long num_units=0)
                 : nu(nu), alpha(alpha), sca(sca), swe(swe), free_water(free_water),
                   residual(residual), num_units(num_units) {}
                bool operator==(const state &x) const {
                    const double eps=1e-6;
                    return fabs(nu - x.nu)<eps
                        && fabs(alpha - x.alpha)<eps
                        && fabs(sca - x.sca) < eps
                        && fabs(swe - x.swe)<eps
                        && fabs(free_water - x.free_water)<eps
                        && fabs(residual - x.residual)<eps
                        && num_units==x.num_units;


                }
                x_serialize_decl();
            };


            struct response {
                double outflow = 0.0;///< m^3/s
                double total_stored_water = 0.0;// def. as sca*(swe+lwc)
                double sca =0.0;// fraction, sih-note: we need it for snow calibration collection
                double swe= 0.0;// mm, as noted above, for calibration, def. as (swe+lwc)
            };

            template<class P, class S, class R>
            class calculator {
              private:
                const double snow_tol = 1.0e-10;
              public:
                calculator() { /* Do nothing */ }
                void step(shyft::time_series::utctimespan dt,
                          const P& p,
                          const double T,
                          const double prec,
                          const double rad,
                          const double wind_speed,
                          S& s,
                          R& r) const {
                    const double unit_size = p.unit_size;

                    // Redistribute residual, if possible:
                    const double corr_prec = std::max(0.0, prec + s.residual);
                    s.residual = std::min(0.0, prec + s.residual);

                    // Simple degree day model physics
                    const double step_in_days = dt/86400.0;
                    const double snow = T < p.tx ? corr_prec : 0.0;
                    const double rain = T < p.tx ? 0.0 : corr_prec;

                    if (s.sca*s.swe < unit_size && snow < snow_tol) {
                        // Set state and response and return
                        r.outflow = rain + s.sca*(s.swe + s.free_water) + s.residual;
                        r.total_stored_water = 0.0;

                        s.residual = 0.0;
                        if (r.outflow < 0.0) {
                            s.residual = r.outflow;
                            r.outflow = 0.0;
                        }
                        s.nu = p.alpha_0*unit_size;
                        s.alpha = p.alpha_0;
                        s.sca = 0.0;
                        s.swe = 0.0;
                        s.free_water = 0.0;
                        s.num_units = 0;
                        r.sca=s.sca;
                        r.swe=s.swe;
                        return;
                    }

                    const double alpha_0 = p.alpha_0;
                    double swe = s.swe; // Conditional value!

                    unsigned long nnn = s.num_units;
                    double sca = s.sca;
                    double nu = s.nu;
                    double alpha = s.alpha;

                    if (nnn > 0)
                        nu *= nnn; // Scaling nu with the number of accumulated units internally. Back scaled before returned.
                    else {
                        nu = alpha_0*p.unit_size;
                        alpha = alpha_0;
                    }

                    double total_new_snow = snow;
                    double lwc = s.free_water;  // Conditional value
                    const double total_storage = swe + lwc; // Conditional value

                    double pot_melt = p.cx*step_in_days*(T - p.ts);
                    const double refreeze = std::min(std::max(0.0, -pot_melt*p.cfr), lwc);
                    total_new_snow += sca*refreeze;
                    lwc -= refreeze;
                    pot_melt = std::max(0.0, pot_melt);
                    const double new_snow_reduction = std::min(pot_melt, total_new_snow);
                    pot_melt -= new_snow_reduction;
                    total_new_snow -= new_snow_reduction;

                    statistics stat(alpha_0, p.d_range, unit_size);

                    unsigned long n = 0;
                    //xx unsigned long u = 0;

                    // Accumulation
                    if (total_new_snow > unit_size) {
                        n = lrint(total_new_snow/unit_size);
                        compute_shape_vars(stat, nnn, n, 0, sca, 0.0, alpha, nu);
                        nnn = lrint(nnn*sca) + n;
                        sca = 1.0;
                        swe = nnn*unit_size;
                    }

                    // Melting
                    if (pot_melt > unit_size) {
                        unsigned long u = lrint(pot_melt/unit_size);
                        if (nnn < u + 2) {
                            nnn = 0;
                            alpha = alpha_0;
                            nu = alpha_0*unit_size;
                            swe = 0.0;
                            lwc = 0.0;
                            sca = 0.0;
                        } else {
                            const double rel_red_sca = stat.sca_rel_red(u, nnn, nu, alpha);
                            const double sca_scale_factor = 1.0 - rel_red_sca;
                            sca = s.sca*sca_scale_factor;
                            swe = (nnn - u)/sca_scale_factor*unit_size;

                            if (swe >= nnn*unit_size) {
                                u = long( nnn*rel_red_sca) + 1;
                                swe = (nnn - u)/sca_scale_factor*unit_size;
                                if (nnn == u)
                                    sca = 0.0;
                            }
                            if (sca < 0.005) {
                                nnn = 0;
                                alpha = alpha_0;
                                nu = alpha_0*unit_size;
                                swe = 0.0;
                                lwc = 0.0;
                                sca = 0.0;
                            } else {
                                compute_shape_vars(stat, nnn, n, u, sca, rel_red_sca, alpha, nu);
                                nnn = lrint(swe/unit_size);
                                swe = nnn*unit_size;
                            }
                        }
                    }

                    // Eli and Ola, TODO:
                    // This section is quite hackish, but makes sure that we have control on the local mass balance that is constantly
                    // violated due to discrete melting/accumulation events. A trained person should rethink the logic and have a goal of
                    // minimizing the number of if-branches below. Its embarrasingly many of them...
                    // The good news is that the residual is in the interval [-0.1, 0.1] for the cases we've investigated, and that 15
                    // years of simulation on real data gives a O(1.0e-11) accumulated mass balance violation (in mm).

                    if (s.sca*s.swe > sca*swe) { // (Unconditional) melt
                        lwc += std::max(0.0, s.swe - swe); // Add melted water to free water in snow
                    }
                    lwc *= std::min(1.0, s.sca/sca); // Scale lwc (preserve total free water when sca increases)
                    lwc = std::min(lwc, swe*p.max_water_fraction); // Limit by parameter
                    double discharge = s.sca*total_storage + snow - sca*(swe + lwc); // We consider rain later

                    // If discharge is negative, recalculate new lwc to take the slack
                    if (discharge < 0.0) {
                        lwc = (s.sca*total_storage + snow)/sca - swe;
                        discharge = 0.0;
                        // Not enough water in the snow for the negative outflow
                        if (lwc < 0.0) {
                            lwc = 0.0;
                            discharge = s.sca*total_storage + snow - sca*swe;
                            // Giving up and adding to residual. TODO: SHOULD BE CHECKED, and preferably REWRITTEN!!
                            if (discharge < 0.0) {
                                s.residual += discharge;
                                discharge = 0.0;
                            }
                        }
                    }

                    // If no runoff, discharge is numerical noise and should go into residual.
                    if (snow > 0.0 && rain == 0.0 && pot_melt <= 0.0) {
                        s.residual += discharge;
                        discharge = 0.0;
                    }

                    // Add rain
                    if (rain > swe*p.max_water_fraction - lwc) {
                        discharge += sca*(rain - (swe*p.max_water_fraction - lwc)) + rain*(1.0 - sca); // exess water in snowpack + uncovered
                        lwc = swe*p.max_water_fraction; // No more room for water
                    } else {
                        lwc += rain; // Rain becomes water in the snow
                        discharge += rain*(1.0 - sca); // Only uncovered area gives rain discharge
                    }

                    // Negative discharge may come from roundoff due to discrete model, add to residual and get rid of it later
                    if (discharge <= 0.0) {
                        s.residual += discharge;
                        discharge = 0.0;
                    } else { // Discharge is positive
                        // We have runoff, so get rid of as much residual as possible (without mass balance violation)
                        if (discharge >= -s.residual) {
                            discharge += s.residual;
                            s.residual = 0.0;
                        // More residual than discharge - get rid of as much as possible and zero out the runoff
                        } else {
                            s.residual += discharge;
                            discharge = 0.0;
                        }
                    }

                    if (nnn > 0)
                        nu /= nnn;

                    r.outflow = discharge;
                    r.total_stored_water = sca*(swe + lwc);
                    r.swe=(swe+lwc);//sih: needed for calibration, and most likely what other routines consider as swe
                    r.sca=sca; //sih: needed for calibration on snow sca
                    s.nu = nu;
                    s.alpha = alpha;
                    s.sca = sca;
                    s.swe = swe;
                    s.free_water = lwc;
                    s.num_units = nnn;
                }

                static inline void compute_shape_vars(const statistics& stat,
                                                      unsigned long nnn,
                                                      unsigned long n,
                                                      unsigned long u,
                                                      double sca,
                                                      double rel_red_sca,
                                                      double& alpha,
                                                      double& nu) {
                    const double alpha_0 = stat.alpha_0;
                    const double nu_0 = stat.alpha_0*stat.unit_size;
                    const double dyn_var = nu/(alpha*alpha);
                    const double init_var = nu_0/(alpha_0*alpha_0);
                    double tot_var = 0.0;
                    double tot_mean = 0.0;
                    if (n > 0) { // Accumulation
                        if (nnn == 0) { // No previous snow pack
                            tot_var = n*init_var*(1 + (n - 1)*stat.c(n));
                            tot_mean = n*nu_0/alpha_0;
                        } else  {
                            const double old_var_cov = (nnn + n)*init_var*(1 + ((nnn + n) - 1)*stat.c(nnn + n));
                            const double new_var_cov = n*init_var*(1 + (n - 1)*stat.c(n));
                            tot_var = old_var_cov*sca*sca + new_var_cov*(1.0 - sca)*(1.0 - sca);
                            tot_mean = (sca*(nnn + n) + (1.0 - sca)*n)*stat.unit_size;
                        }
                    }
                    if (u > 0) { // Ablation
                        const double factor = (dyn_var/(nnn*init_var) + 1.0 + (nnn - 1)*stat.c(nnn))/(2*nnn);
                        const double non_cond_mean = (nnn - u)*stat.unit_size;
                        tot_mean = non_cond_mean/(1.0 - rel_red_sca);
                        const unsigned long cond_u = lrint((1.0 - rel_red_sca)*nnn - (nnn - u));
                        const double auto_var  = cond_u > 0 ? init_var*cond_u*(1.0 + (cond_u - 1.0)*stat.c(cond_u)) : 0.0;
                        const double cross_var = cond_u > 0 ? init_var*cond_u*2.0*factor*cond_u : 0.0;
                        tot_var = dyn_var + auto_var - cross_var;
                    }

                    if (fabs(tot_mean) < 1.0e-7) {
                        nu = nu_0;
                        alpha = alpha_0;
                        return;
                    }
                    nu = tot_mean*tot_mean/tot_var;
                    alpha = nu/(stat.unit_size*lrint(tot_mean/stat.unit_size));
                }
            };
       } // skaugen
    } // core
} // shyft
  //-- serialization support shyft
x_serialize_export_key(shyft::core::skaugen::state);
