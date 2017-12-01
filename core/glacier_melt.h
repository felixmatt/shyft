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
namespace shyft {
    namespace core {
		namespace glacier_melt {

		    struct parameter {
                double dtf = 6.0;///<degree timestep factor [mm/day/deg.C]; lit. values for Norway: 5.5 - 6.4 in Hock, R. (2003), J. Hydrol., 282, 104-115.
                double  direct_response=0.0;///< fraction to route directly, (1-direct_response) goes into  into kirchner or similar
                explicit parameter(double dtf = 6.0, double direct_response = 0.0) :dtf(dtf), direct_response(direct_response) {}
            };

            /** Glacier Melt model
             *
             * \param dtf degree timestep factor [mm/day/deg.C]; lit. values for Norway: 5.5 - 6.4 in Hock, R. (2003), J. Hydrol., 282, 104-115.
             *
             * \param t temperature [deg.C]
             *
             * \param snow_covered_area_m2 of the glacier, in unit [m2] if
             *
             * \param glacier_area_m2 the area of the glacier, in unit [m2]
             *
             * \return glacier_melt in [m3/s]
             */

            inline double step(const double dtf, const double t, const double snow_covered_area_m2, const double glacier_area_m2){
                if(glacier_area_m2 <= snow_covered_area_m2 || t <= 0.0) // melt is 0.0 if area uncovered by snow less than 0.0, and t below zero
                    return 0.0;
                const double convert_m2_x_mm_d_to_m3_s= 0.001/86400.0;// ref. input units ,mm=0.001m/d=86400s
                return dtf*t*(glacier_area_m2-snow_covered_area_m2)* convert_m2_x_mm_d_to_m3_s;
            }

		} // glacier_melt
    } // core
} // shyft
