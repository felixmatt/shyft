///	Copyright 2012 Statkraft Energi A/S
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
/// This implementation is an adapted version of the temperature index model for glacier
/// ice melt, Eq.(1), in "Hock, R. (2003), Temperature index modelling in mountain areas, J. Hydrol., 282, 104-115."
///

#pragma once
#include "utctime_utilities.h"
namespace shyft {
    namespace core {
		namespace glacier_melt {

		    struct parameter {
                double dtf = 5.0;
                parameter(double dtf=5.0):dtf(dtf) {}
            };

            struct state {
                double glacier_height = 100000.0;

                state(double glacier_height=100000.0)
                  : glacier_height(glacier_heit) { /* Do nothing */ }
                bool operator==(const state& x)const {
                    const double eps=1e-6;
                    return fabs(glacier_height-x.glacier_height)<eps;
                }
            };

            struct response {
                double glacier_melt = 0.0;
            };


            /** Gamma Snow model
             *
             * \param P
             * Parameter that supplies:
             * -# P.dtf() const --> double, Degree Timestep Factor [mm/day/deg.C]; lit. values for Norway: 5.5 - 6.4 in Hock, R. (2003), J. Hydrol., 282, 104-115.
             * \param S
             * State class that supports
             * -# S.glacier_height --> double, glacier ice storage over glaciated cell area [mm ice-water equivalent]
             *
             * \param R
             * Response class that supports
             * -# R.glacier_melt --> double, ice melt [mm ice-water equivalent]
             */
            template<class P, class S>
            class calculator {
                private:
                    double glacier_fraction_ = 0.0;
                    const double glacier_tol = 1.0e-10;
                public:
                    void set_glacier_fraction(double glacier_fraction) {glacier_fraction_=glacier_fraction;}
                    double glacier_fraction() const {return glacier_fraction_;}

                    void step(S& s,
                              R& r,
                              shyft::timeseries::utctimespan dt,
                              const P& p,
                              const double T,
                              const double sca) const {
                        double T_effective = std::max(0,T);
                        double area_effective = std::max(0, glacier_fraction_ - sca);
                        double glacier_melt = p.dtf * T_effective * area_effective * dt/calendar::DAY;
                        double glacier_melt = std:min(s.glacier_height, glacier_melt);
                        s.glacier_height -= glacier_melt;
                        r.glacier_melt = glacier_melt;
                    }

            }

		} // glacier_melt
    } // core
} // shyft
