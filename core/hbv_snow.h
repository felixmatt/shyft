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

#include "core_serialization.h"

#include "utctime_utilities.h"
#include "hbv_snow_common.h"

namespace shyft {
    namespace core {
        namespace hbv_snow {
            using namespace std;
            using namespace hbv_snow_common;


            struct parameter {
                vector<double> s;///<snow redistribution factors
                vector<double> intervals;///< snow quantiles list 0, 0.25 0.5 1.0
                double tx;
                double cx;
                double ts;
                double lw;
                double cfr;

                void set_std_distribution_and_quantiles() {
                    double si[] = { 1.0, 1.0, 1.0, 1.0, 1.0 };
                    double ii[] = { 0, 0.25, 0.5, 0.75, 1.0 };
                    s.clear(); s.reserve(5);
                    intervals.clear(); intervals.reserve(5);
                    for (size_t i = 0; i<5; ++i) {
                        s.push_back(si[i]);
                        intervals.push_back(ii[i]);
                    }
                    normalize_snow_distribution();
                }

                void normalize_snow_distribution() {
                    const double mean = integrate(s, intervals, intervals.size(), intervals[0], intervals.back());
                    for (auto &s_ : s) s_ /= mean;
                }


                parameter(double tx=0.0, double cx=1.0, double ts=0.0, double lw=0.1, double cfr=0.5)
                  :  tx(tx), cx(cx), ts(ts), lw(lw), cfr(cfr) {
                      set_std_distribution_and_quantiles();
                }
                parameter(const vector<double>& s, const vector<double>& intervals,
                          double tx=0.0, double cx=1.0, double ts=0.0, double lw=0.1, double cfr=0.5)
                  : s(s), intervals(intervals), tx(tx), cx(cx), ts(ts), lw(lw), cfr(cfr) { /* Do nothing */ }


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
                double swe = 0.0;
                double sca = 0.0;

                state(double swe=0.0, double sca=0.0)
                  : swe(swe), sca(sca) { /* Do nothing */ }

                bool operator==(const state& x) const {
                    const double eps=1e-6;
                    if (sp.size() != sw.size()) return false;
                    for (size_t i = 0; i<sp.size(); ++i) {
                        if (fabs(sp[i]-x.sp[i])>= eps || fabs(sw[i]-x.sw[i])>=eps)
                            return false;
                    }
                    return fabs(swe-x.swe)<eps && fabs(sca-x.sca)<eps;
                }
                void distribute(const parameter& p, bool force=true) {
                    if(force || sp.size() != p.s.size() || sw.size()!=p.s.size() ) {// if not force ,but a size miss-match
                        distribute_snow(p, sp, sw,swe,sca);
                    }
                }
                x_serialize_decl();
            };
            struct response {
                double outflow = 0.0;
                state snow_state;///< SiH we need the state for response during calibration, maybe we should reconsider calibration-pattern ?
            };


            /** \brief Generalized quantile based HBV Snow model method
             *
             * This algorithm uses arbitrary quartiles to model snow. No checks are performed to assert valid input. The starting points of the
             * quantiles have to partition the unity, include the end points 0 and 1 and must be given in ascending order.
             *
             * \tparam P Parameter type, implementing the interface:
             *    - P.s() const --> vector<double>, snowfall redistribution vector
             *    - P.intervals() const --> vector<double>, starting points for the quantiles
             *    - P.lw() const --> double, max liquid water content of the snow
             *    - P.tx() const --> double, threshold temperature determining if precipitation is rain or snow
             *    - P.cx() const --> double, temperature index, i.e., melt = cx(t - ts) in mm per degree C
             *    - P.ts() const --> double, threshold temperature for melt onset
             *    - P.cfr() const --> double, refreeze coefficient, refreeze = cfr*cx*(ts - t)
             * \tparam S State type, implementing the interface:
             *    - S.swe --> double, snow water equivalent of the snowpack [mm]
             *    - S.sca --> double, fraction of area covered by the snowpack [0,1]
             * \tparam R Respone type, implementing the interface:
             *    - S.set_outflow(double value) --> void, set the value of the outflow [mm]
             */
            template<class P, class S>
            struct calculator {
                const P p;

                calculator(const P& p):p(p) {
                }

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


                static inline void update_state(double &sp, double &sw, const double rain, const double potmelt, const double lw) {
                    if (sp > potmelt) {
                        sw += potmelt + rain;
                        sp -= potmelt;
                        sw = std::min(sw, sp*lw);
                    } else if (sp > 0.0)
                        sp = sw = 0.0;
                }


                inline size_t sca_index(double sca) const {
                    for (size_t i = 0; i < p.intervals.size() - 1; ++i)
                        if (sca >= p.intervals[i] && sca < p.intervals[i + 1])
                            return i;
                    return p.intervals.size() - 1;
                }

                size_t melt_index(double potmelt, const state &s) const {
                    for (size_t i = 0; i < p.intervals.size(); ++i)
                        if (s.sp[i] < potmelt)
                            return i;
                    return p.intervals.size();
                }

                template <class R> void step(S& s, R& r, utctime t0, utctime t1, double prec, double temp) const {
                    double swe = s.swe;
                    double sca = s.sca;
                    const auto &I = p.intervals;
                    const double total_water = prec + swe;
                    double snow,rain;
                    if( temp < p.tx ) {snow=prec;rain= 0.0;}
                    else              {snow= 0.0;rain=prec;}
                    double step_in_days = (t1 - t0)/86400.0;
                    swe += snow + sca*rain;
                    if (swe < 0.1) {
                        r.outflow = total_water;
                        fill(begin(s.sp), end(s.sp), 0.0);
                        fill(begin(s.sw), end(s.sw), 0.0);
                        s.swe = 0.0;
                        s.sca = 0.0;
                        return;
                    }
                    if (snow > 0.0) {
                        auto idx = sca_index(sca);
                        if (sca > 1.0e-5 && sca < 1.0 - 1.0e-5) {
                            if (idx == 0) {
                                s.sp[0] *= sca/(I[1] - I[0]);
                                s.sw[0] *= sca/(I[1] - I[0]);
                            } else {
                                s.sp[idx] *= (1.0 + (sca - I[idx])/(I[idx] - I[idx - 1]))/(1.0 + (I[idx + 1] - I[idx])/(I[idx] - I[idx - 1]));
                                s.sw[idx] *= (1.0 + (sca - I[idx])/(I[idx] - I[idx - 1]))/(1.0 + (I[idx + 1] - I[idx])/(I[idx] - I[idx - 1]));
                            }
                        }

                        for (size_t i = 0; i < p.s.size(); ++i) s.sp[i] += snow*p.s[i];
                        sca = I[1]; //  at least one bin filled after snow-fall
                        for (size_t i = I.size() - 2; i > 0; --i)
                            if (p.s[i] > 0.0) {
                                sca = I[i + 1];
                                break;
                            }
                    }

                    double potmelt = p.cx*step_in_days*(temp - p.ts);
                    const double lw = p.lw;

                    if (potmelt < 0.0) {
                        potmelt *= p.cfr;
                        for (size_t i = 0; i < I.size(); ++i)
                            refreeze(s.sp[i], s.sw[i], rain, potmelt, lw);
                    } else {
                        size_t idx = melt_index(potmelt,s);

                        if (idx == 0) sca = 0.0;
                        else if (idx == I.size()) sca = 1.0;
                        else {
                            if (s.sp[idx] > 0.0) sca = I[idx] - (I[idx] - I[idx - 1])*(potmelt - s.sp[idx])/(s.sp[idx - 1] = s.sp[idx]);
                            else sca = (1.0 - potmelt/s.sp[idx - 1])*(sca - I[idx - 1]) + I[idx - 1];
                        }

                        for (size_t i = 0; i < I.size(); ++i)
                            update_state(s.sp[i], s.sw[i], rain, potmelt, lw);
                    }

                    if (sca < 1.0e-6) swe = 0.0;
                    else {
                        bool f_is_zero = sca >= 1.0 ? false : true;
                        swe = integrate(s.sp, I, I.size(), 0, sca, f_is_zero);
                        swe += integrate(s.sw, I, I.size(), 0, sca, f_is_zero);
                    }

                    if (total_water < swe) {
                        if (total_water - swe < -1.0e-6) {// we do have about 15-16 digits accuracy
                            ostringstream buff;
                            buff << "Negative outflow: total_water (" << total_water << ") - swe (" << swe << ") = " << total_water - swe;
                            throw runtime_error(buff.str());
                        } else
                            swe = total_water;
                    }
                    r.outflow = total_water - swe;
                    s.swe = swe;
                    s.sca = sca;
                }
            };
        }
    } // core
} // shyft
  //-- serialization support shyft
x_serialize_export_key(shyft::core::hbv_snow::state);
