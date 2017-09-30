#pragma once
#ifdef SHYFT_NO_PCH
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "core_pch.h"
#endif // SHYFT_NO_PCH

#include "utctime_utilities.h"

namespace shyft {
	namespace core {
		namespace hbv_tank {
			using namespace std;

			struct parameter {
				parameter(double uz1 = 25.0, double kuz2 = 0.5, double kuz1 = 0.3, double perc= 0.8, double klz =0.02)
					:uz1(uz1), kuz2(kuz2), kuz1(kuz1), perc(perc), klz(klz) {
					if (perc < .0)
						throw runtime_error("perc should be > 0.0");
				}
				double uz1 = 25; // mm
				double kuz2 = 0.5;// 1/unit time
				double kuz1 = 0.3;// 1/unit time
				double perc = 0.8; // mm/h
				double klz = 0.02; // 1/unit time
			};

			struct state {
				state(double uz = 20.0, double lz =10) :uz(uz), lz(lz) {}
				double uz = 20.0; // mm
				double lz = 10.0; // mm
				bool operator==(const state&x) const {
					const double eps = 1e-6;
					//return fabs(lz - x.lz)<eps; // example script
					if ((lz - x.lz) < eps && (uz - x.uz) < eps)
						return true;
					else
						return false;
				}
                x_serialize_decl();
			};

			struct response {
				double outflow = 0.0; // mm
			};


			/** \brief Hbv tank
			*
			*  reference somewhere..
			*
			* \tparam P Parameter type, implementing the interface:
			* \tparam S State type, implementing the interface:
			* \tparam R Respone type, implementing the interface:
			*/
			template<class P>
			struct calculator {
				P param;
				explicit calculator(const P& p) :param(p) {}
				template <class R, class S>
				void step(S& s, R& r, shyft::core::utctime t0, shyft::core::utctime t1, double soil_outflow) {
					double temp = s.uz + soil_outflow;				//compute of q11 & q12 at end of time after adding soil_outflow
					double q12 = std::max(0.0, (temp - param.uz1)*param.kuz2);
					double q11 = std::min(temp, param.uz1)*param.kuz1;
					s.uz = s.uz + soil_outflow - param.perc - (q12+q11);

					double q2 = (s.lz + param.perc) *param.klz;  ////compute of q2 at end of timestep after adding perc
					s.lz = s.lz + param.perc - q2 ;
					r.outflow = q12 + q11 + q2;

				}
			};
		}
	} // core
} // shyft
  //-- serialization support shyft
x_serialize_export_key(shyft::core::hbv_tank::state);
