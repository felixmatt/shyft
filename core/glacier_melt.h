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
/// Implemented by Felix Matt

#pragma once
#include "timeseries.h"
namespace shyft {
    namespace core {
		namespace glacier_melt {

		    struct parameter {
                double dtf = 6.0;
                parameter(double dtf=6.0):dtf(dtf) {}
            };

            struct response {
                double glacier_melt = 0.0;
            };


            /** Gamma Snow model
             *
             * \param P
             * Parameter that supplies:
             * -# P.dtf() const --> double, degree timestep factor [mm/day/deg.C]; lit. values for Norway: 5.5 - 6.4 in Hock, R. (2003), J. Hydrol., 282, 104-115.
             *
             * \param R
             * Response class that supports
             * -# R.glacier_melt --> double, ice melt [mm ice-water equivalent]
             */
            template<class P, class R>
            class calculator {
                private:
                    double glacier_fraction_ = 0.0;
                public:
                    void set_glacier_fraction(double glacier_fraction) {glacier_fraction_=glacier_fraction;}
                    double glacier_fraction() const {return glacier_fraction_;}

                    void step(R& r,
                              shyft::timeseries::utctimespan dt,
                              const P& p,
                              const double T,
                              const double sca) const {

                        /** Input variables:
                         *
                         * T -> air temperature in deg C.
                         * sca -> fraction of snow cover in cell
                         */

                        double T_effective = std::max(0.0,T);
                        double area_effective = std::max(0.0, glacier_fraction_ - sca);
                        double glacier_melt = p.dtf * T_effective * area_effective * dt/calendar::DAY;
                        r.glacier_melt = std::max(0.0,glacier_melt) * calendar::HOUR/dt; // convert to mm/h
                    }

            };

		} // glacier_melt
    } // core
} // shyft
