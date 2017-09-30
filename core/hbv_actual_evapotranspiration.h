#pragma once

#include "utctime_utilities.h"

namespace shyft {
	namespace core {
		namespace hbv_actual_evapotranspiration {
			/**<  keeps the parameters (potential calibration/localization) for the AE */
			struct parameter {
				double lp = 150.0; ///<default value is 150.0
				parameter(double lp = 150.0) : lp(lp) {}
			};

			/**<  keeps the formal response of the AE method */
			struct response {
				double ae = 0.0;
			};

			/** \brief actual_evapotranspiration calculates actual evapotranspiration
			* based on supplied parameters
			*
			* \param soil_moisture, sm
			* \param potential_evapotranspiration, pot_evapo
			* \param soil threshold for evapotranspiration, lp
			* \param snow_fraction 0..1
			* \return calculated actual evapotranspiration
			*
			*/

			inline double calculate_step(const double soil_moisture,
				const double pot_evapo,
				const double lp,
				const double snow_fraction,
				const utctime dt) {
				//double actevap;

				return (1.0 - snow_fraction)*(soil_moisture < lp ? pot_evapo*(soil_moisture / lp):pot_evapo);
				}
		};
	};
};
