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
/// Extracted from early enki method GammaSnow programmed by Kolbjørn Engeland and 
/// Sjur Kolberg, and adapted according to discussions with Gaute Lappegard and Eli Alfnes
///

#pragma once
#include "utctime_utilities.h"
namespace shyft {
    namespace core {
		namespace glacier_melt {
			double glacier_melt(double radiation, double albedo, double sca, double temporary_snow, utctimespan dt) {
				const double melt_heat = 333660.0; // Latent heat of ice at T = 0C in [J/kg]
				return temporary_snow > 0.0 ? 0.0 : radiation*(1.0 - albedo)*(double)dt / melt_heat*(1.0 - sca);
			}
		}
    } // End namespace core
} // shyft
