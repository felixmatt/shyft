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
/// Adapted from early enki method programmed by Kolbjørn Engeland and Sjur Kolberg
///
#pragma once
#ifdef SHYFT_NO_PCH
#include <cmath>
#endif // SHYFT_NO_PCH


namespace shyft {
    namespace core {
		namespace priestley_taylor {

			struct parameter {
				double albedo = 0.2;
				double alpha = 1.26;
				parameter(double albedo=0.2, double alpha=1.26) : albedo(albedo), alpha(alpha) { }
			};


			//struct state {}; // No state variables for this method


			struct response {
				double pot_evapotranspiration = 0.0;
			};


			/** \brief PriestleyTaylor,PT, (google for PriestleyTaylor)
			 *  primitive implementation for calculating the potential evaporation.
			 *  This function is plain and simple, taking land_albedo and PT.alpha
			 *  into the constructor and provides a function that calculates potential evapotransporation
			 *  [mm/s] units.
			 */
			struct calculator {
				/** \brief Constructor
				 *
				 * \param land_albedo
				 *
				 * \param alpha
				 *  PriestleyTaylor alpha, typical range 1.26 +- 0.xx
				 *
				 */
				calculator(double land_albedo, double alpha)
					: land_albedo(land_albedo), alpha(alpha) {
#ifdef WIN32
					ck2[0] = 17.84362; ck2[1] = 17.08085;
					ck3[0] = 245.425; ck3[1] = 234.175;
#endif
				}
				/** \brief Calculate PotentialEvapotranspiration, given specified parameters
				 *
				 * \param temperature in [degC]
				 * \param global_radiation [W/m^2]
				 * \param rhumidity in interval [0,1]
				 * \return PotentialEvapotranspiration in [mm/s] units
				 *
				 */
				double potential_evapotranspiration(double temperature, double global_radiation, double rhumidity) const {
					int i = temperature < 0 ? 0 : 1;  //select negative or positive set of ck[] constants
					double ctt_inv = 1 / (ck3[i] + temperature);
					double sat_pressure = ck1*exp(ck2[i] * temperature*ctt_inv);
					double delta = sat_pressure*ck2[i] * ck3[i] * ctt_inv*ctt_inv;
					double vapour_pressure = sat_pressure*rhumidity;// actual vapour pressure,[kPa]

					double epot = alpha*delta*net_radiation(temperature, global_radiation, rhumidity, vapour_pressure) / (delta + psycr);// main P-T equation
					if (epot < 0.0) return 0.0;
					return epot /
						(2500780 - 2361 * temperature); // Latent heat of vaporisation [J/kg], Energy required per water volume vaporized
				}
			private:
				/** \brief calculate net radiation (long +short) given specfied parameters
				 *
				 * \param temperature in [degC]
				 * \param global_radiation [W/m^2]
				 * \param rhumidity in interval [0,1]
				 * \param vapour_pressure [kPa]
				 * \return net_radiation [W/m^2]
				 *
				 */
				double net_radiation(double temperature, double global_radiation, double rhumidity, double vapour_pressure) const {
					double k_temp = temperature + 273.15;// Temperature in Kelvin
					double e_atm = 1.24*std::pow(10 * vapour_pressure / k_temp, 0.143)* (0.85 + 0.5*rhumidity);// Brutsaert's (1975) clear-sky emissivity model, Cloud factor, see tab 1 in Sicart et al (2006)
					return    bolz*std::pow(k_temp, 4)*(e_atm - 0.98) // net longwave radiatin, Ta2m is used for both up- and downward LW-rad,so net balance is just an emissivity difference.
						+ global_radiation*(1.0 - land_albedo);	// Net shortwave radiation
				}
				const double bolz = 0.0000000567;// Stephan-Bolzman constant [W/(m^2 K^4)]
				const double psycr = 0.066;       // psycometric constant [ kPa/C ]
				const double ck1 = 0.610780;    //cck4&ck7,equal originally(KE)
#ifndef WIN32
				const double ck2[2] = {17.84362,17.08085};//ck5,ck8 neg and pos temp
				const double ck3[2] = {245.425,234.175  };//ck6,ck9
#else
				double ck2[2];//= {17.84362,17.08085};//ck5,ck8 neg and pos temp
				double ck3[2]; //= {245.425,234.175  };//ck6,ck9

#endif
				const double land_albedo;
				const double alpha;

			};
		}
    }
} // shyft
