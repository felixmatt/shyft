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
/// PURPOSE.  See the    GNU Lesser General Public License for more details.
///
///    You should have received a copy of the GNU Lesser General Public License along with
/// Shyft, usually located under the Shyft root directory in two files named COPYING.txt
/// and COPYING_LESSER.txt.    If not, see <http://www.gnu.org/licenses/>.
///
/// Adapted from early enki method programmed by Kolbjørn Engeland and Sjur Kolberg
///


#pragma once
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <sstream>

#include "time_series.h"
#include "utctime_utilities.h"
#include "hbv_snow_common.h"

namespace shyft {
    namespace core {
        namespace hbv_physical_snow {
            using namespace std;
            using namespace hbv_snow_common;
            static const double tol = 1.0e-10;

            struct parameter {

                vector<double> s; // snow redistribution vector
                vector<double> intervals; // snow quantiles list 0, 0.25, 0.5, 1.0
                double tx = 0.0;
                double lw = 0.1;
                double cfr = 0.5;
                double wind_scale = 2.0;
                double wind_const = 1.0;
                double surface_magnitude = 30.0;
                double max_albedo = 0.9;
                double min_albedo = 0.6;
                double fast_albedo_decay_rate = 5.0;
                double slow_albedo_decay_rate = 5.0;
                double snowfall_reset_depth = 5.0;
                bool calculate_iso_pot_energy = false;


                void set_std_distribution_and_quantiles() {
                    double si[] = {1.0, 1.0, 1.0, 1.0, 1.0};
                    double ii[] = {0, 0.25, 0.5, 0.75, 1.0};
                    s.clear();s.reserve(5);
                    intervals.clear();intervals.reserve(5);
                    for(size_t i=0; i<5; ++i) {
                        s.push_back(si[i]);
                        intervals.push_back(ii[i]);
                    }
					normalize_snow_distribution();
                }

                void normalize_snow_distribution() {
                    const double mean = integrate(s, intervals, intervals.size(), intervals[0], intervals.back());
                    for (auto &s_ : s) s_ /= mean;
                }

                parameter(double tx=0.0, double lw=0.1, double cfr=0.5,
                        double wind_scale=2.0, double wind_const=1.0,
                        double surface_magnitude=30.0,
                        double max_albedo=0.9, double min_albedo=0.6,
                        double fast_albedo_decay_rate=5.0,
                        double slow_albedo_decay_rate=5.0,
                        double snowfall_reset_depth=5.0,
                        bool calculate_iso_pot_energy=false)
                            : tx(tx), lw(lw), cfr(cfr), wind_scale(wind_scale),
                              wind_const(wind_const),
                              surface_magnitude(surface_magnitude),
                              max_albedo(max_albedo),
                              min_albedo(min_albedo),
                              fast_albedo_decay_rate(fast_albedo_decay_rate),
                              slow_albedo_decay_rate(slow_albedo_decay_rate),
                              snowfall_reset_depth(snowfall_reset_depth),
                              calculate_iso_pot_energy(calculate_iso_pot_energy)
                {
                    set_std_distribution_and_quantiles();
                }

                parameter(const vector<double>& s,
                        const vector<double>& intervals,
                        double tx=0.0, double lw=0.1, double cfr=0.5,
                        double wind_scale=2.0, double wind_const=1.0,
                        double surface_magnitude=30.0,
                        double max_albedo=0.9, double min_albedo=0.6,
                        double fast_albedo_decay_rate=5.0,
                        double slow_albedo_decay_rate=5.0,
                        double snowfall_reset_depth=5.0,
                        bool calculate_iso_pot_energy=false)
                            : s(s), intervals(intervals), tx(tx), lw(lw),
                              cfr(cfr), wind_scale(wind_scale),
                              wind_const(wind_const),
                              surface_magnitude(surface_magnitude),
                              max_albedo(max_albedo),
                              min_albedo(min_albedo),
                              fast_albedo_decay_rate(fast_albedo_decay_rate),
                              slow_albedo_decay_rate(slow_albedo_decay_rate),
                              snowfall_reset_depth(snowfall_reset_depth),
                              calculate_iso_pot_energy(calculate_iso_pot_energy)
                { normalize_snow_distribution(); }

                void set_snow_redistribution_factors(const vector<double>& values) {
                    if (values.size() != intervals.size())
                        throw std::runtime_error("Incompatible size for snowdistribution factors: " + to_string(intervals.size()) + " != " + to_string(values.size()));
                    s = values;
                    normalize_snow_distribution();
                }

                void set_snow_quantiles(const vector<double>& values) {
                    if (values.size() != s.size())
                        throw std::runtime_error("Incompatible size for snowdistribution factors: " + to_string(s.size()) + " != " + to_string(values.size()));
                    intervals = values;
                    normalize_snow_distribution();
                }
            };


            struct state {

                vector<double> sp;
                vector<double> sw;
                vector<double> albedo;
                vector<double> iso_pot_energy;

                double surface_heat = 30000.0;
                double swe = 0.0;
                double sca = 0.0;

                state() = default;

                state(const vector<double>& albedo,
                      const vector<double>& iso_pot_energy,
                      double surface_heat=30000.0, double swe=0.0,
                      double sca=0.0)
                    : sp(albedo.size(), 0.0), sw(albedo.size(), 0.0), albedo(albedo), iso_pot_energy(iso_pot_energy),
                      surface_heat(surface_heat), swe(swe), sca(sca)
                {}

                void distribute(const parameter& p, bool force=true) {
                    if(force || sp.size() != p.s.size() || sw.size()!=p.s.size() ) {// if not force ,but a size miss-match
                        distribute_snow(p, sp, sw,swe,sca);
                    }
                    if(sp.size()!= albedo.size()) { // .. user forgot to pass in correctly sized arrays, fix it
                        albedo= vector<double>(sp.size(),0.4); // same as for gamma-snow
                        iso_pot_energy=vector<double>(sp.size(),0.0);
                    }
                }


                bool operator==(const state &x) const {
                    const double eps = 1e-6;
                    if (albedo.size() != x.albedo.size()) return false;
                    if (sp.size() != sw.size()) return false;

                    if (iso_pot_energy.size() != x.iso_pot_energy.size()) {
                        return false;
                    }
                    for (size_t i=0; i<albedo.size(); ++i) {
                        if (fabs(albedo[i] - x.albedo[i]) >= eps ||
                                fabs(iso_pot_energy[i] - x.iso_pot_energy[i]) >=
                                eps) {
                            return false;
                        }
                        if (fabs(sp[i]-x.sp[i])>= eps || fabs(sw[i]-x.sw[i])>=eps)
                            return false;
                    }

                    return fabs(surface_heat - x.surface_heat) < eps
                        && fabs(swe - x.swe) < eps
                        && fabs(sca - x.sca) < eps;
                }
                x_serialize_decl();
            };

            struct response {
                double sca = 0.0;
                double storage = 0.0;
                double outflow = 0.0;
                state hps_state;
            };


            /** \brief Generalized quantile based HBV Snow model method, using
             * the physical melt model from Gamma-snow (refreeze is treated by multiplying potential_melt by a refreeze coefficient)
             *
             * This algorithm uses arbitrary quartiles to model snow. No checks are performed to assert valid input. The starting points of the
             * quantiles have to partition the unity, include the end points 0 and 1 and must be given in ascending order.
             *
             * \tparam P Parameter type, implementing the interface:
             *    - P.s --> vector<double>, snowfall redistribution vector
             *    - P.intervals --> vector<double>, starting points for the quantiles
             *    - P.lw --> double, max liquid water content of the snow
             *    - P.tx --> double, threshold temperature determining if precipitation is rain or snow
             *    - P.cx --> double, temperature index, i.e., melt = cx(t - ts) in mm per degree C
             *    - P.ts --> double, threshold temperature for melt onset
             *    - P.cfr --> double, refreeze coefficient, refreeze = cfr*cx*(ts - t)
             * \tparam S State type, implementing the interface:
             *    - S.albedo --> vector<double>, broadband snow reflectivity fraction in each snow bin.
             *    - S.iso_pot_energy --> vector<double>, accumulated energy assuming isothermal snow surface [J/m2]
             *    - S.surface_heat --> double, snow surface cold content [J/m2]
             *    - S.swe --> double, snow water equivalent of the snowpack [mm]
             *    - S.sca --> double, fraction of area covered by the snowpack [0,1]
             * \tparam R Response type, containing the member variables:
             *    - R.outflow --> double, the value of the outflow [mm]
             *    - R.storage --> double, the value of the swe [mm]
             *    - R.sca --> double, the value of the snow covered area
             *    - R.state --> shyft::core::hbv_physical_snow::state, containing
             *          the current state.
             */
            template<class P, class S, class R>
            class calculator {

              private:
                const P p;
                const double melt_heat = 333660.0;
                const double water_heat = 4180.0;
                const double ice_heat = 2050.0;
                const double sigma = 5.670373e-8;
                const double BB0{0.98*sigma*pow(273.15,4)};

                static inline void refreeze(double &sp, double &sw, const double rain, const double potmelt, const double lw) {
                    // Note that the above calculations might violate the mass
                    // balance due to rounding errors. A fix might be to
                    // replace sw by a sw_fraction, sp with s_tot, and compute
                    // sw and sp based on these.
                    if (sp > 0.0) {
                        if (sw + rain > -potmelt) {
                            sp -= potmelt;
                            sw += potmelt + rain;
                            if (sw > sp*lw) sw = sp*lw;
                        } else {
                            sp += sw + rain;
                            sw = 0.0;
                        }
                    }
                }


                static inline void update_state(double &sp, double &sw, const double rain,  const double potmelt, const double lw) {
                    if (sp > potmelt) {
                        sw += potmelt + rain;
                        sp -= potmelt;
                        sw = std::min(sw, sp*lw);
                    } else if (sp > 0.0)
                        sp = sw = 0.0;
                }


                inline size_t sca_index(double sca) const {
                    for (size_t i = 0;  i < p.intervals.size() - 1; ++i)
                        if (sca >= p.intervals[i] && sca < p.intervals[i + 1])
                            return i;
                    return p.intervals.size() - 1;
                }


              public:
                calculator(const P& p) : p(p) {}

                  /*
                  * \brief step the snow model forward from time t to t+dt, state, parameters and input
                  * updates the state and response upon return.
                  * \param s state of type S,in/out, ref template parameters
                  * \param r result of type R, output only, ref. template parameters
                  * \param t start time
                  * \param dt timespan
                  * \param p parameter of type P, ref template parameters
                  * \param T temperature degC, considered constant over timestep dt
                  * \param rad radiation
                  * \param prec precipitation in mm/h
                  * \param wind_speed in m/s
                  * \param rel_hum 0..1
                  */
                void step(S& s, R& r, utctime t, utctimespan dt,
                        const double T, const double rad,
                        const double prec_mm_h, const double wind_speed,
                        const double rel_hum) const {

                    const auto& I = p.intervals;
                    const double prec = prec_mm_h*dt/calendar::HOUR;
                    const double total_water = prec + s.swe;

                    double snow;
                    double rain;
                    if( T < p.tx ) {snow = prec; rain = 0.0;}
                    else           {snow = 0.0; rain = prec;}
                    if (fabs(snow + rain - prec) > 1.0e-8)
                        throw std::runtime_error("Mass balance violation!!!!");

                    s.swe += snow + s.sca * rain;
                    //Roughly the same as the 'early autumn scenario' of
                    //gamma_snow - i.e. no stored or precipitated snow
                    if (s.swe < hbv_physical_snow::tol) {
                        // Reset everything
                        r.outflow = total_water;
                        fill(begin(s.sp), end(s.sp), 0.0);
                        fill(begin(s.sw), end(s.sw), 0.0);
                        s.swe = 0.0;
                        s.sca = 0.0;

                        r.sca = 0.0;
                        r.storage = 0.0;

                        std::fill(s.albedo.begin(), s.albedo.end(),
                                  p.max_albedo);
                        s.surface_heat = 0.0;
                        std::fill(s.iso_pot_energy.begin(),
                                  s.iso_pot_energy.end(), 0.0);
                        return;
                    }

                    // Trivial case is out of the way, now more complicated

                    // State vars
                    vector<double> albedo = s.albedo;
                    double surface_heat = s.surface_heat;

                    // Response vars;
                    //double outflow = 0.0;

                    // Local variables

                    const double min_albedo = p.min_albedo;
                    const double max_albedo = p.max_albedo;
                    const double albedo_range = max_albedo - min_albedo;
                    const double dt_in_days = dt/double(calendar::DAY);
                    const double slow_albedo_decay_rate = (0.5 * albedo_range *
                            dt_in_days/p.slow_albedo_decay_rate);
                    const double fast_albedo_decay_rate = pow(2.0,
                            -dt_in_days/p.fast_albedo_decay_rate);

                    const double T_k = T + 273.15; // Temperature in Kelvin
                    const double turb = p.wind_scale*wind_speed + p.wind_const;
                    double vapour_pressure = (
                            33.864 * (pow(7.38e-3*T + 0.8072, 8) -
                                1.9e-5*fabs(1.8*T + 48.0) + 1.316e-3) *
                            rel_hum);
                    if (T < 0.0)  // Change VP over water to VP over ice (Bosen)
                        vapour_pressure *= 1.0 + 9.72e-3*T + 4.2e-5*T*T;

                    // There is some snowfall. Here we update the snowpack and
                    // wet snow to reflect the new snow. We also
                    // update albedo.
                    if (snow > hbv_physical_snow::tol) {
                        auto idx = sca_index(s.sca);
                        if (s.sca > 1.0e-5 && s.sca < 1.0 - 1.0e-5) {
                            if (idx == 0) {
                                s.sp[0] *= s.sca/(I[1] - I[0]);
                                s.sw[0] *= s.sca/(I[1] - I[0]);
                            } else {
                                s.sp[idx] *= (1.0 + (s.sca - I[idx]) / (I[idx] -
                                            I[idx - 1])) / (1.0 + (I[idx + 1] -
                                                I[idx])/(I[idx] - I[idx - 1]));
                                s.sw[idx] *= (1.0 + (s.sca - I[idx]) / (I[idx] -
                                            I[idx - 1])) / (1.0 + (I[idx + 1] -
                                                I[idx])/(I[idx] - I[idx - 1]));
                            }
                        }

                        for (size_t i = 0; i < I.size(); ++i)
                        {
                            double currsnow = snow * p.s[i];
                            s.sp[i] += currsnow;
                            albedo[i] += (currsnow * albedo_range /
                                          p.snowfall_reset_depth);
                        }

                        for (size_t i = I.size() - 2; i > 0; --i)
                            if (p.s[i] > 0.0) {
                                s.sca = p.s[i + 1];
                                break;
                            } else
                                s.sca = p.s[1];
                    } else {
                        // No snowfall: Albedo decays
                        if (T < 0.0) {
                            for (auto &alb: albedo) {
                                alb -= slow_albedo_decay_rate;
                            }
                        } else {
                            for (auto &alb: albedo) {
                                alb = (min_albedo + fast_albedo_decay_rate *
                                       (alb - min_albedo));
                            }
                        }
                    }

                    // We now start calculation of energy content based on
                    // current albedoes etc.

                    for (auto &alb: albedo) {
                        alb = std::max(std::min(alb, max_albedo),
                                       min_albedo);
                    }

                    vector<double> effect;
                    for (auto alb: albedo) {
                        effect.push_back(rad * (1.0 - alb));
                    }


                    for (auto &eff: effect) {
                        eff += (0.98 * sigma *
                                pow(vapour_pressure/T_k, 6.87e-2) *
                                pow(T_k, 4));
                    }

                    if (T > 0.0 && snow < hbv_physical_snow::tol) {
                        for (auto &eff: effect) {
                            eff += rain * T * water_heat/(double)dt;
                        }
                    }
                    if (T <= 0.0 && rain < hbv_physical_snow::tol)
                        for (size_t i=0; i<I.size(); ++i) {
                            effect[i] += snow*p.s[i]*T*ice_heat/(double)dt;
                        }
                    //TODO: Should the snow distribution be included here?
                //        effect += snow*T*ice_heat/(double)dt;

                    if (p.calculate_iso_pot_energy) {
                        for (size_t i=0; i<I.size(); ++i) {
                            double iso_effect = (effect[i] - BB0 + turb *
                                    (T + 1.7 * (vapour_pressure - 6.12)));
                            s.iso_pot_energy[i] += (iso_effect *
                                    (double)dt/melt_heat);
                        }
                    }

                    double sst = std::min(0.0, 1.16*T - 2.09);
                    if (sst > -hbv_physical_snow::tol) {
                        for (auto &eff: effect) {
                            eff += turb * (T + 1.7 *
                                    (vapour_pressure - 6.12)) - BB0;
                        }
                    } else {
                        for (auto &eff: effect) {
                            eff += (turb * (T - sst + 1.7 *
                                        (vapour_pressure - 6.132 *
                                         exp(0.103 * T - 0.186))) -
                                    0.98 * sigma * pow(sst + 273.15, 4));
                        }
                    }

                    // Surface heat change, positive during warming
                    double delta_sh = -surface_heat;

                    // New surface heat; always nonpositive since sst <= 0.
                    surface_heat = p.surface_magnitude*ice_heat*sst*0.5;
                    delta_sh += surface_heat;

                    vector<double> energy;
                    for (auto eff : effect) {
                        energy.push_back(eff * (double)dt);
                    }
                    if (delta_sh > 0.0) {
                        //TODO: Should this condition be removed when we allow refreeze?
                //        energy -= delta_sh;  // Surface energy is a sink, but not a source
                        for (auto &en: energy) en -= delta_sh;
                    }

                    vector<double> potential_melt;
                    for (auto en: energy) {
                        //This is a difference from the physical model: We
                        //allow negative potential melts. This to fit it better
                        //with the HBV model. We treat negative potential melts
                        //as refreeze.
                        potential_melt.push_back(en/melt_heat);
                    }

                    // We have now calculated the potential melt in each bin
                    // and now update the distributions and outflow etc. to
                    // reflect that.
                    const double lw = p.lw;

                    size_t idx = I.size();
                    bool any_melt = false;

                    for (size_t i=0; i<I.size(); ++i) {
                        if (potential_melt[i] >= hbv_physical_snow::tol) {
                            any_melt = true;
                            if (s.sp[i] < potential_melt[i]) {
                                idx = i;
                                break;
                            }
                        }
                    }

                    // If there is melting at all
                    if (any_melt) {
                        if (idx == 0) s.sca = 0.0;
                        else if (idx == I.size()) s.sca = 1.0;
                        else {
                            if (s.sp[idx] > 0.0) {
                                s.sca = (I[idx] - (I[idx] - I[idx - 1]) *
                                        (potential_melt[idx] - s.sp[idx]) /
                                        (s.sp[idx - 1] = s.sp[idx]));
                            } else {
                                s.sca = (1.0 - potential_melt[idx]/s.sp[idx - 1]) *
                                    (s.sca - I[idx - 1]) + I[idx - 1];
                            }
                        }
                    }


                    // If negative melt, we treat it as refreeze,
                    // otherwise we update the snowpack and wet snow.
                    for (size_t i=0; i<I.size(); ++i) {
                        if (potential_melt[i] < hbv_physical_snow::tol) {
                            refreeze(s.sp[i], s.sw[i], rain,
                                     p.cfr*potential_melt[i], lw);
                        } else {
                            update_state(s.sp[i], s.sw[i], rain, potential_melt[i], lw);
                        }
                    }

                    if (s.sca < hbv_physical_snow::tol) s.swe = 0.0;
                    else {
                        bool f_is_zero = s.sca >= 1.0 ? false : true;
                        s.swe = integrate(s.sp, I, I.size(), 0, s.sca, f_is_zero);
                        s.swe += integrate(s.sw, I, I.size(), 0, s.sca, f_is_zero);
                    }

                    if (total_water < s.swe) {
                        if (total_water - s.swe < -hbv_physical_snow::tol) {
                            ostringstream buff;
                            buff << "Negative outflow: total_water (" <<
                                total_water << ") - s.swe (" << s.swe << ") = "
                                << total_water - s.swe;
                            throw runtime_error(buff.str());
                        } else
                            s.swe = total_water;
                    }
                    r.outflow = total_water - s.swe;
                    r.sca = s.sca;
                    r.storage = s.swe;
                }
            };
        }
    }
}

x_serialize_export_key(shyft::core::hbv_physical_snow::state);
