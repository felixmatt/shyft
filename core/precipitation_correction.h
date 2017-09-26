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

namespace shyft {
    namespace core {

		namespace precipitation_correction {
			struct parameter {
				double scale_factor = 1.0;
				explicit parameter(double scale_factor = 1.0):scale_factor(scale_factor) {}
			};


			struct calculator {
				const double scale_factor;

				explicit calculator(double scale_factor = 1.0) : scale_factor(scale_factor) {}

				inline double calc(const double precipitation) {
					return precipitation*scale_factor;
				}
			};
		}
    } // End namespace core
} // shyft
