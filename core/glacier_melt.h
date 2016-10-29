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
#include "utctime_utilities.h"
namespace shyft {
    namespace core {
		namespace glacier_melt {

		    struct parameter {
                double dtf = 6.0;
                parameter(double dtf=6.0):dtf(dtf) {}
            };

            /** Glacier Melt model
             *
             * \param dtf degree timestep factor [mm/day/deg.C]; lit. values for Norway: 5.5 - 6.4 in Hock, R. (2003), J. Hydrol., 282, 104-115.
             *
             * \param t temperature [deg. C]
             *
             * \param sca, fraction of snow cover [0..1]
             *
             * \param glacier_fraction, fraction of glacier cover [0..1]
             *
             * \return glacier_melt, outflow from glacier melt [mm/h]
             */

            inline double step(const double dtf, const double t, const double sca, const double glacier_fraction){
                if(glacier_fraction<=0.0)
                    return 0.0;
                double t_effective = std::max(0.0,t);
                double area_effective = std::max(0.0, glacier_fraction - sca);
                return dtf * t_effective * area_effective/24.0; // convert from mm/day to mm/h
            }

		} // glacier_melt
    } // core
} // shyft
