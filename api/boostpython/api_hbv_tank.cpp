
#include "boostpython_pch.h"

#include "core/hbv_tank.h"

namespace expose {
	void hbv_tank() {
		using namespace shyft::core::hbv_tank;
		using namespace boost::python;
		using namespace std;

		class_<parameter>("HbvTankParameter")
			.def(init<optional<double, double, double, double, double>>(args("uz1", "kuz2", "kuz1", "perc","klz"), "create parameter object with specifed values"))
			.def_readwrite("uz1", &parameter::uz1, "mm, .. , default=25")
			.def_readwrite("kuz2", &parameter::kuz2, ",default=0.5")
			.def_readwrite("kuz1", &parameter::kuz1, ",default=0.3")
			.def_readwrite("perc", &parameter::perc, ",default=0.8")
			.def_readwrite("klz", &parameter::klz, ",default=0.02")
			;

		class_<state>("HbvTankState")
			.def(init<optional<double,double>>(args("uz", "lz"), "create a state with specified values"))
			.def_readwrite("uz", &state::uz, "Water Level Upper Zone [mm]")
			.def_readwrite("lz", &state::lz, "Water Level Lower Zone [mm]")
			;

		class_<response>("HbvTankResponse")
			.def_readwrite("outflow", &response::outflow, "from Tank-routine in [mm]")
			;

		typedef  calculator<parameter> HbvTankCalculator;
		class_<HbvTankCalculator>("HbvTankCalculator",
			"tobe done.. \n"
			"\n"
			"\n", no_init
			)
			.def(init<const parameter&>(args("parameter"), "creates a calculator with given parameter"))
			.def("step", &HbvTankCalculator::step<response, state>, args("state", "response", "t0", "t1", "soil_outflow"),
				"steps the model forward from t0 to t1, updating state and response")
			;
#if 0
#endif
	}
}