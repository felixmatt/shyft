#include "boostpython_pch.h"

#include "core/hbv_soil.h"

namespace expose {

	void hbv_soil() {
		using namespace shyft::core::hbv_soil;
		using namespace boost::python;
		using namespace std;

		class_<parameter>("HbvSoilParameter")
			.def(init<optional<double, double>>(args("fc", "beta"), "create parameter object with specifed values"))
			.def_readwrite("fc", &parameter::fc, "mm, .. , default=300")
			.def_readwrite("beta", &parameter::beta, ",default=2.0")
			;

		class_<state>("HbvSoilState")
			.def(init<optional<double>>(args("sm"), "create a state with specified values"))
			.def_readwrite("sm", &state::sm, "Soil  moisture [mm]")
			;

		class_<response>("HbvSoilResponse")
			.def_readwrite("outflow", &response::outflow, "from Soil-routine in [mm]")
			;

		typedef  calculator<parameter> HbvSoilCalculator;
		class_<HbvSoilCalculator>("HbvSoilCalculator",
			"tobe done.. \n"
			"\n"
			"\n", no_init
			)
			.def(init<const parameter&>(args("parameter"), "creates a calculator with given parameter"))
			.def("step", &HbvSoilCalculator::step<response,state>, args("state", "response", "t0", "t1", "insoil", "actual_evap", "pot_evap"), 
				"steps the model forward from t0 to t1, updating state and response")
			;
#if 0
#endif
	}
}
