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
/// PURPOSE.  See the	GNU Lesser General Public License for more details.
///
///	You should have received a copy of the GNU Lesser General Public License along with
/// Shyft, usually located under the Shyft root directory in two files named COPYING.txt
/// and COPYING_LESSER.txt.	If not, see <http://www.gnu.org/licenses/>.
///
/// Adapted from early enki method programmed by Kolbjørn Engeland and Sjur Kolberg
///
#pragma once
#ifdef SHYFT_NO_PCH
#include <cmath>
#endif // SHYFT_NO_PCH

#include "utctime_utilities.h"
/**
contains the actual evatransporation parameters and algorithm
*/
namespace shyft {
    namespace core {
		namespace actual_evapotranspiration {
			/**<  keeps the parameters (potential calibration/localization) for the AE */
			struct parameter {
				double ae_scale_factor = 1.5; ///<default value is 1.5
				explicit parameter(double ae_scale_factor=1.5) : ae_scale_factor(ae_scale_factor) {}
			};

			/**<  keeps the formal response of the AE method */
			struct response {
				double ae = 0.0;
			};

			/** \brief actual_evapotranspiration calculates actual evapotranspiration
			 * based on supplied parameters
			 *
			 * \param water_level
			 * \param potential_evapotranspiration
			 * \param scale_factor typically 1.5
			 * \param snow_fraction 0..1 - only snow free areas have evapotranspiration in this model
			 * \param dt delta_t \note currently not used in computation
			 * \return calculated actual evapotranspiration
			 *
			 */

			inline double calculate_step(const double water_level,
				const double potential_evapotranspiration,
				const double scale_factor,
				const double snow_fraction,
				const utctime dt) {
				return potential_evapotranspiration*(1.0 - std::exp(-water_level*3.0/ scale_factor))*(1.0 - snow_fraction);
			}
		};
	};
};
