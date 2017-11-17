#pragma once
///    Copyright 2012 Statkraft Energi A/S
///
///    This file is part of Shyft.
///
///    Shyft is free software: you can redistribute it and/or modify it under the terms of
/// the GNU Lesser General Public License as published by the Free Software Foundation,
/// either version 3 of the License, or (at your option) any later version.
///
///    Shyft is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
/// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
/// PURPOSE. See the GNU Lesser General Public License for more details.
///
///    You should have received a copy of the GNU Lesser General Public License along with
/// Shyft, usually located under the Shyft root directory in two files named COPYING.txt
/// and COPYING_LESSER.txt.    If not, see <http://www.gnu.org/licenses/>.
///
/// Adapted from early enki method programmed by Kolbjørn Engeland and Sjur Kolberg
///

#include <string>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/math/tools/minima.hpp>

#include "core_pch.h"

#include "time_series.h"
#include "utctime_utilities.h"

#ifdef __UNIT_TEST__
class gamma_snow_test;
#endif

namespace shyft {
    namespace core {
        namespace gamma_snow {
            static const double tol = 1.0e-10;

            struct parameter {
                calendar cal;
                size_t winter_end_day_of_year = 100;///< approx 10th april
                double initial_bare_ground_fraction = 0.04;
                double snow_cv = 0.4;
                double tx = -0.5;
                double wind_scale = 2.0;
                double wind_const = 1.0;
                double max_water = 0.1;
                double surface_magnitude = 30.0;
                double max_albedo = 0.9;
                double min_albedo = 0.6;
                double fast_albedo_decay_rate = 5.0;
                double slow_albedo_decay_rate = 5.0;
                double snowfall_reset_depth = 5.0;
                double glacier_albedo = 0.4; // TODO: Remove from GammaSnow and put into glacier method parameter?
                bool calculate_iso_pot_energy = false;
                double snow_cv_forest_factor=0.0;///< [ratio] the effective snow_cv gets an additional value of geo.forest_fraction()*snow_cv_forest_factor
                double snow_cv_altitude_factor=0.0;///< [1/m] the effective snow_cv gets an additional value of altitude[m]* snow_cv_altitude_factor
                size_t n_winter_days=221;///< # winter is from [winter_end_day_of_year - n_winter_days, winter_end_day_of_year], default yyyy.09.01 yyyy+1.04.10
                parameter(size_t winter_end_day_of_year = 100,double initial_bare_ground_fraction = 0.04,double snow_cv = 0.4,double tx = -0.5,
                          double wind_scale = 2.0,double wind_const = 1.0,double max_water = 0.1,double surface_magnitude = 30.0,double max_albedo = 0.9,
                          double min_albedo = 0.6,double fast_albedo_decay_rate = 5.0,double slow_albedo_decay_rate = 5.0,double snowfall_reset_depth = 5.0,
                          double glacier_albedo = 0.4, // TODO: Remove from GammaSnow and put into glacier method parameter?
                          bool calculate_iso_pot_energy = false,
                          double snow_cv_forest_factor=0.0,
                          double snow_cv_altitude_factor=0.0,
                          size_t n_winter_days=221
                          ):winter_end_day_of_year(winter_end_day_of_year),initial_bare_ground_fraction(initial_bare_ground_fraction),snow_cv(snow_cv),tx(tx),
                          wind_scale(wind_scale),wind_const(wind_const),max_water(max_water),surface_magnitude(surface_magnitude),max_albedo(max_albedo),
                          min_albedo(min_albedo),fast_albedo_decay_rate(fast_albedo_decay_rate),slow_albedo_decay_rate (slow_albedo_decay_rate),
                          snowfall_reset_depth(snowfall_reset_depth),
                          glacier_albedo(glacier_albedo) , // TODO: Remove from GammaSnow and put into glacier method parameter?
                          calculate_iso_pot_energy(calculate_iso_pot_energy),
                          snow_cv_forest_factor(snow_cv_forest_factor),
                          snow_cv_altitude_factor(snow_cv_altitude_factor),
                          n_winter_days(n_winter_days)
                          {}
                /** \returns the effective snow cv, taking the forest_fraction and altitude into the equations using corresponding factors */
                double effective_snow_cv(double forest_fraction,double altitude) const {
                    return snow_cv+ forest_fraction*snow_cv_forest_factor + altitude*snow_cv_altitude_factor;
                }
                /** \returns true if specified t is within the snow season, e.g. sept.. winder_end_day_of_year */
                bool is_snow_season(utctime t) const {
                    utctime t_w_end = cal.trim(t,calendar::YEAR) + deltahours(winter_end_day_of_year*24);
                    utcperiod snow_period{t_w_end - deltahours(n_winter_days*24),t_w_end};
                    return snow_period.contains(t);
                }
                /** \returns true if specified interval t day of year is wind_end_day_of_year */
                bool is_start_melt_season(utctime t, utctimespan dt) const {
                    return cal.day_of_year(t) == winter_end_day_of_year;
                }
            };


            struct state {
                state (double albedo=0.4, double lwc=0.1, double surface_heat=30000.0,
                       double alpha=1.26, double sdc_melt_mean=0.0, double acc_melt=0.0,
                       double iso_pot_energy=0.0, double temp_swe=0.0)
                  :
                    albedo(albedo), lwc(lwc), surface_heat(surface_heat), alpha(alpha), sdc_melt_mean(sdc_melt_mean),
                    acc_melt(acc_melt), iso_pot_energy(iso_pot_energy), temp_swe(temp_swe) {/* Do nothing */}

                double albedo = 0.4;
                double lwc = 0.1;
                double surface_heat = 30000.0;
                double alpha = 1.26;
                double sdc_melt_mean = 0.0;
                double acc_melt = 0.0;
                double iso_pot_energy = 0.0;
                double temp_swe = 0.0;
                bool operator==(const state &x) const {
                    const double eps=1e-6;
                    return fabs(albedo - x.albedo)<eps
                        && fabs(lwc - x.lwc) < eps
                        && fabs(surface_heat - x.surface_heat)<eps
                        && fabs(alpha - x.alpha)<eps
                        && fabs(sdc_melt_mean - x.sdc_melt_mean)<eps
                        && fabs(acc_melt - x.acc_melt)<eps
                        && fabs(iso_pot_energy - x.iso_pot_energy)<eps
                        && fabs(temp_swe - x.temp_swe)<eps;

                }
                x_serialize_decl();
            };


            struct response {
                double sca = 0.0;
                double storage = 0.0;
                double outflow = 0.0;
            };


            /** Gamma Snow model
             *
             * \param P
             * Parameter that supplies:
             * -# P.winter_end --> Last day of accumulation season
             * -# P.initial_bare_ground_fraction --> Bare ground fraction at melt onset
             * -# P.snow_cv --> Spatial coefficient variation of fresh snowfall
             * -# P.tx  --> Snow/rain threshold temperature [C]
             * -# P.wind_scale --> Slope in turbulent wind function [m/s]
             * -# P.wind_const --> Intercept in turbulent wind function
             * -# P.max_water --> Maximum liquid water content
             * -# P.surface_magnitude --> Surface layer magnitude
             * -# P.max_albedo --> Maximum albedo value
             * -# P.min_albedo --> Minimum albedo value
             * -# P.fast_albedo_decay_rate --> Albedo decay rate during melt [days]
             * -# P.slow_albedo_decay_rate --> Albedo decay rate in cold conditions [days]
             * -# P.snowfall_reset_depth --> Snowfall required to reset albedo [mm]
             * -# P.glacier_albedo --> Glacier ice fixed albedo // TODO: Remove from GammaSnow and put into glacier method parameter?
             * -# P.calculate_iso_pot_energy --> bool Whether or not to calculate the potential energy flux
             * -# P::LandType --> Enum; any of {LAKE, LAND}
             * \param S
             * State class that supports
             * -# S.albedo --> albedo (Broadband snow reflectivity fraction)
             * -# S.lwc --> lwc (liquid water content) [mm]
             * -# S.surface_heat --> surface_heat (Snow surface cold content) [J/m2]
             * -# S.alpha --> alpha (Dynamic shape state in the SDC)
             * -# S.sdc_melt_mean --> sdc_melt_mean  (Mean snow storage at melt onset) [mm]
             * -# S.acc_melt --> acc_melt (Accumulated melt depth) [mm]
             * -# S.iso_pot_energy --> iso_pot_energy (Accumulated energy assuming isothermal snow surface) [J/m2]
             * -# S.temp_swe --> temp_swe (Depth of temporary new snow layer during spring) [mm]
             *
             * \param R
             * Response class that supports
             * -# R.sca --> set sca (Snow covered area)
             * -# R.storage --> set snow storage [mm]
             * -# R.outflow--> set water outflow [mm]
             */
            template<class P, class S, class R>
            class calculator {
    #ifdef __UNIT_TEST__
                friend gamma_snow_test;
    #endif
              private:
                calendar cal;
                const double melt_heat = 333660.0;
                const double water_heat = 4180.0;
                const double ice_heat = 2050.0;
                const double sigma = 5.670373e-8;
                const double BB0{0.98*sigma*std::pow(273.15, 4)};
                typedef boost::math::policies::policy< boost::math::policies::digits10<10> > high_precision_type;
                typedef boost::math::policies::policy< boost::math::policies::digits10<5> > low_precision_type;
                const high_precision_type high_precision = high_precision_type();
                const low_precision_type low_precision = low_precision_type();
                const double precision_threshold = 2.0;

                double gamma_p(double a, double b) const {
                    return a < precision_threshold ? boost::math::gamma_p(a, b, high_precision) : boost::math::gamma_p(a, b, low_precision);
                }

                double lgamma(double a) const {
                    return a < precision_threshold ? boost::math::lgamma(a, high_precision) : boost::math::lgamma(a, low_precision);
                }
                /*xx
                inline double calc_df(const double a, const double b, const double z) const {
                    return a*b*exp(-(a + 1.0)*log(b) - lgamma(a + 1.0) + a*log(z) - z/b)
                            + 1.0 - z*exp(-a*log(b) - lgamma(a) + (a - 1.0)*log(z) - z/b)
                            - gamma_p(a, z/b);
                }
                */
                inline double calc_q(const double a, const double b, const double z) const
                {
                    return a*b*gamma_p(a + 1.0, z/b) + z*(1.0 - gamma_p(a, z/b));
                }

                double corr_lwc(const double z1, const double a1, const double b1,
                    double z2, const double a2, const double b2) const {
                    using boost::math::tools::brent_find_minima;
                    uintmax_t iterations = 60;
                    auto digits = 12;// accurate enough,std::numeric_limits<double>::digits;
                    double Q1 = calc_q(a1, b1, z1);
                    auto result = brent_find_minima(
                        [Q1, a2, b2, this](const double&z)->double {
                            double f = this->calc_q(a2, b2, z) - Q1;
                            return f*f;
                        },
                        0.0, z1, digits, iterations);
                    return result.first;
                }


                  void calc_snow_state(const double shape, const double scale, const double y0, const double lambda,
                                       const double lwd, const double max_water_frac, const double temp_swe,
                                       double& swe, double& sca) const {
                      double y = 0.0;
                      double y1 = 0.0;
                      const double m = shape*scale;
                      if (lambda <= 0.0) { // accumulated melt is -1.
                          swe = m;
                          sca = 1.0 - y0;
                      } else if (lambda/scale > 1.3*shape + 20.0) {
                          swe = sca = 0.0;
                          return;
                      } else {
                          const double x = lambda/scale;
                          y = gamma_p(shape, x);
                          y1 = y - exp(shape*log(x) - x - lgamma(shape))/shape;
                          swe = m*(1.0 - y1) - lambda*(1 - y);
                          sca = (1.0 - y)*(1.0 - y0);
                      }
                      if (lwd > m) swe *= 1.0 + max_water_frac;
                      else if (lwd > 0.0) {
                          const double sat = lwd/max_water_frac;
                          const double x = sat/scale;
                          const double ssa = gamma_p(shape, x);
                          const double ssa1 = ssa - exp(shape*log(x) - x - lgamma(shape))/shape;
                          const double liqwat = max_water_frac*(m*(ssa1 - y1) + sat*(1.0 - ssa) - lambda*(1.0 - y));
                          swe += liqwat;
                      }
                      swe += temp_swe;
                      swe *= 1.0 - y0;
                  }

                  void reset_snow_pack(double& sca, double& lwc, double& alpha, double& sdc_melt_mean, double& acc_melt,
                                       double& temp_swe, const double storage, const P& p) const
                  {
                      if (storage > gamma_snow::tol) {
                          sca = 1.0 - p.initial_bare_ground_fraction;
                          sdc_melt_mean = storage/sca;
                      } else {
                          sca = sdc_melt_mean = 0.0;
                      }
                      alpha = 1.0/(p.snow_cv*p.snow_cv);
                      temp_swe = lwc = 0.0;
                      acc_melt = -1.0;
                  }

              public:
                  calculator()=default;
                  /*
                  * \brief step the snow model forward from time t to t+dt, state, parameters and input
                  * updates the state and response upon return.
                  * \param s state of type S,in/out, ref template parameters
                  * \param r result of type R, output only, ref. template parameters
                  * \param T temperature degC, considered constant over timestep dt
                  * \param rad radiation ..
                  * \param prec precipitation in mm/h
                  * \param wind_speed in m/s
                  * \param rel_hum 0..1
                  * \param forest_fraction 0..1, influences calculation of effective snow_cv
                  * \param altitude 0..x [m], influences calculation of effective_snow_cv
                  */
                void step(S& s, R& r, shyft::time_series::utctime t, shyft::time_series::utctimespan dt,
                          const P& p, const double T, const double rad, const double prec_mm_h,
                          const double wind_speed, const double rel_hum, const double forest_fraction,const double altitude) const {
                    // Some special cases treated first for efficiency

                    // Read state vars needed early (possible early return)
                    double sdc_melt_mean = s.sdc_melt_mean;
                    double acc_melt = s.acc_melt;
                    double iso_pot_energy = s.iso_pot_energy;
                    const double prec = prec_mm_h*dt/calendar::HOUR;

                    if( p.is_start_melt_season(t, dt))
                        acc_melt = iso_pot_energy = 0.0;


                    double snow;
                    double rain;
                    if( T < p.tx ) {snow = prec; rain = 0.0;}
                    else           {snow = 0.0;rain = prec;}
                    if (fabs(snow + rain - prec) > 1.0e-8)
                        throw std::runtime_error("Mass balance violation!!!!");

                    if (snow < gamma_snow::tol && sdc_melt_mean < gamma_snow::tol && acc_melt < 0.0) { // Early autumn scenario, no snow
                        // Set some of the states.
                        s.albedo = p.max_albedo;
                        s.surface_heat = 0.0;
                        s.iso_pot_energy = 0.0;
                        r.sca = 0.0;
                        r.storage = 0.0;
                        r.outflow = prec_mm_h;
                        return;
                    }

                    // State vars
                    double albedo = s.albedo;
                    double lwc = s.lwc;
                    double surface_heat = s.surface_heat;
                    double alpha = s.alpha;
                    double temp_swe = s.temp_swe;

                    // Response vars;
                    double sca = 0.0;
                    double storage = 0.0;
                    double outflow = 0.0;

                    // Local variables
                    const double min_albedo = p.min_albedo;
                    const double max_albedo = p.max_albedo;
                    const double snow_cv = p.effective_snow_cv(forest_fraction,altitude);
                    const double albedo_range = max_albedo -  min_albedo;
                    const double dt_in_days = dt/double(calendar::DAY);
                    const double slow_albedo_decay_rate = 0.5*albedo_range*dt_in_days/p.slow_albedo_decay_rate;
                    const double fast_albedo_decay_rate = pow(2.0, -dt_in_days/p.fast_albedo_decay_rate);


                    const double T_k = T + 273.15; // Temperature in Kelvin
                    const double turb = p.wind_scale*wind_speed + p.wind_const;
                    double vapour_pressure = 33.864*(pow(7.38e-3*T + 0.8072, 8) - 1.9e-5*fabs(1.8*T + 48.0)
                                           + 1.316e-3)*rel_hum;
                    if (T < 0.0)  // Change VP over water to VP over ice (Bosen)
                        vapour_pressure *= 1.0 + 9.72e-3*T + 4.2e-5*T*T;

                    if (snow > gamma_snow::tol)
                        albedo += snow*albedo_range/p.snowfall_reset_depth;
                    else {
                        if (T < 0.0)
                            albedo -= slow_albedo_decay_rate;
                        else
                            albedo = min_albedo + fast_albedo_decay_rate*(albedo - min_albedo);
                    }

                    albedo = std::max(std::min(albedo, max_albedo), min_albedo);

                    double effect = rad*(1.0 - albedo);
                    effect += 0.98*sigma*pow(vapour_pressure/T_k, 6.87e-2)*pow(T_k, 4);

                    if (T > 0.0 && snow < gamma_snow::tol) // Why not if (rain > 0.0)?
                        effect += rain*T*water_heat/(double)dt;
                    if (T <= 0.0 && rain < gamma_snow::tol)
                        effect += snow*T*ice_heat/(double)dt;

                    if (p.calculate_iso_pot_energy) {
                        double iso_effect = effect - BB0 + turb*(T + 1.7*(vapour_pressure - 6.12));
                        iso_pot_energy += iso_effect*(double)dt/melt_heat;
                    }

                    double sst = std::min(0.0, 1.16*T - 2.09);
                    if (sst > -gamma_snow::tol)
                        effect += turb*(T + 1.7*(vapour_pressure - 6.12)) - BB0;
                    else
                        effect += turb*(T - sst + 1.7*(vapour_pressure - 6.132*exp(0.103*T - 0.186))) // Sensible heat contribution
                                 - 0.98*sigma*pow(sst + 273.15, 4);     // Upward longwave from colder snowpack.

                    double delta_sh = -surface_heat;                        // Surface heat change, positive during warming
                    surface_heat = p.surface_magnitude*ice_heat*sst*0.5;  // New surface heat; always nonpositive since sst <= 0.
                    delta_sh += surface_heat;

                    double energy = effect*(double)dt;
                    if (delta_sh > 0.0) energy -= delta_sh;                 // Surface energy is a sink, but not a source

                    double potential_melt = std::max(0.0, energy/melt_heat);          // Potential snowmelt in mm.

                    // Done surface layer energy and phase change calculations, now mass balance!
                    // The mass balance section is skipped off-season (when there is no snow pack).
                    // The distributed mass balance simulates the CV during the season. There is a
                    // static/parameter snowfall CV applied to new snow. If no mid-winter melt events
                    // occur, the melt-season SWE_CV will also have this value. If a melt event occurs,
                    // the melt is fastest from the thinnest snow packs, and the CV is thus increased.
                    // This is accomplished by reducing the shape alpha, keeping the scale constant.
                    // Subsequent snowfalls will then adjust the SWE_CV back towards the snowfall CV.

                    // Completely bare ground lead to exit only after acc-melt is computed, in order to
                    // provide lambda values to the updating routine, which estimates elevation gradients.
                    // With smaller evenly-distributed snow packs at low elevation, this gradient would
                    // discontinue increasing with time if low-altitude accumulation were stopped.

                    double sdc_scale = sdc_melt_mean/alpha;

                    calc_snow_state(alpha, sdc_scale, p.initial_bare_ground_fraction, acc_melt,
                                    lwc, p.max_water, temp_swe, storage, sca);

                    double start_storage_value = storage;

                    if (acc_melt < 0.0) {
                        if (snow < gamma_snow::tol) snow = 0.0;
                        else {
                            double alpha_prev = alpha;
                            double sdc_scale_prev = sdc_scale;
                            double sdc_snow = snow/(1.0 - p.initial_bare_ground_fraction);

                            alpha = (sdc_melt_mean*alpha + sdc_snow/(snow_cv*snow_cv))/(sdc_snow + sdc_melt_mean);

                            sdc_melt_mean += sdc_snow;
                            sdc_scale = sdc_melt_mean/alpha;
                            if (lwc > 0.0 && sdc_snow > 0.01*sdc_melt_mean) {
                                double z1 = lwc/p.max_water;
                                double z1_guess = z1*(1.0 - sdc_snow/sdc_melt_mean);
                                //double z1_guess = z1*0.5; // Alternative, simple initial guess
                                if (z1_guess < gamma_snow::tol)
                                    z1_guess = z1*0.5;
                                z1 = corr_lwc(z1, alpha_prev, sdc_scale_prev>0.0?sdc_scale_prev:sdc_scale, z1_guess, alpha, sdc_scale);
                                lwc = z1*p.max_water;
                                calc_snow_state(alpha, sdc_scale, p.initial_bare_ground_fraction,
                                                acc_melt, lwc, p.max_water, temp_swe, storage, sca);
                            }
                        }
                        lwc += rain;
                        if (sdc_melt_mean <= potential_melt) {
                            storage = 0.0;
                            reset_snow_pack(sca, lwc, alpha, sdc_melt_mean, acc_melt, temp_swe, storage, p);
                            sdc_scale = 0.0;
                        } else if (potential_melt > 0.0) {
                            sdc_melt_mean -= potential_melt;
                            lwc += potential_melt;
                            // Update alpha and scale. Alpha in interval [0.1, 1.0/(snow_cv*snow_cv)]
                            alpha = std::max(0.1, sdc_melt_mean/sdc_scale);
                            if (alpha > 1.0/(snow_cv*snow_cv))
                                alpha = 1.0/(snow_cv*snow_cv);
                            sdc_scale = sdc_melt_mean/alpha;
                        }

                    } else { // Spring snow pack
                        temp_swe += snow/(1.0 - p.initial_bare_ground_fraction);
                        if (temp_swe > 0.0) {
                            double melt = std::min(temp_swe, potential_melt);
                            temp_swe -= melt;
                            potential_melt -= melt;
                            lwc += melt;
                            if (temp_swe < gamma_snow::tol) temp_swe = 0.0;
                        }
                        acc_melt += potential_melt;
                        lwc += rain + potential_melt;
                        if (!p.calculate_iso_pot_energy || p.is_snow_season(t)) {
                            if (storage < std::max(0.2, 2*temp_swe) || storage < 0.2*rain) {
                                storage += snow;
                                reset_snow_pack(sca, lwc, alpha, sdc_melt_mean, acc_melt, temp_swe, storage, p);
                                sdc_scale = sdc_melt_mean/alpha;
                            }
                        }
                    }
                    // Establish the snow pack state after this time step
                    calc_snow_state(alpha, sdc_scale, p.initial_bare_ground_fraction, acc_melt,
                                    lwc, p.max_water, temp_swe, storage, sca);

                    outflow = prec + start_storage_value - storage;

                    if (outflow < 0.0) outflow = 0.0;  // Tiny rounding errors may occur.

                    // Store updated state variables
                    s.albedo = albedo;
                    s.lwc = lwc;
                    s.surface_heat = surface_heat;
                    s.alpha = alpha;
                    s.sdc_melt_mean = sdc_melt_mean;
                    s.acc_melt = acc_melt;
                    s.iso_pot_energy = iso_pot_energy;
                    s.temp_swe = temp_swe;

                    // Store updated response variables
                    r.sca = sca;
                    r.storage = storage;
                    r.outflow = outflow*calendar::HOUR/dt;
                }
            }; // End GammaSnow
        }
    } // core
} // shyft
  //-- serialization support shyft
x_serialize_export_key(shyft::core::gamma_snow::state);
