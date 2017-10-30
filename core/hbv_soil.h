#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "core_pch.h"

#include "utctime_utilities.h"
namespace shyft {
	namespace core {
		namespace hbv_soil {
			using namespace std;

			struct parameter {
				parameter(double fc = 300.0, double beta = 2.0)
					:fc(fc), beta(beta) {
					if (fc < .0)
						throw runtime_error("fc should be > 0.0");
				}
				double fc=300; // mm
				double beta=2.0;// unit-less
			};

			struct state {
				explicit state(double sm = 0.0) :sm(sm) {}
				double sm=50.0; // mm
				bool operator==(const state&x) const {
					const double eps = 1e-6;
					return fabs(sm - x.sm)<eps;
				}
                x_serialize_decl();
			};

			struct response {
				double outflow = 0.0; // mm
			};


			/** \brief Hbv soil
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
				explicit calculator(const P& p):param(p) {}
				template <class R,class S>
				void step(S& s, R& r, shyft::core::utctime t0, shyft::core::utctime t1, double insoil, double act_evap) {
					double temp = s.sm + insoil;					//compute fraction at end of time after adding insoil
					double outflow = insoil*pow(temp/param.fc, param.beta);
                    r.outflow = outflow > temp ? temp : outflow;
					s.sm = std::max(0.0,s.sm + insoil - r.outflow - act_evap);
				}
			};
		}
	} // core
} // shyft
  //-- serialization support shyft
x_serialize_export_key(shyft::core::hbv_soil::state);
