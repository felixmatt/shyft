#include "boostpython_pch.h"

#include "core/hbv_actual_evapotranspiration.h"

namespace expose {
	using namespace shyft::core::hbv_actual_evapotranspiration;
	using namespace boost::python;


	void hbv_actual_evapotranspiration() {
		class_<parameter>("HbvActualEvapotranspirationParameter")
			.def(init<optional<double>>(args("lp"), "a new object with specified parameters"))
			.def_readwrite("lp", &parameter::lp, "typical value 150")
			;
		class_<response>("HbvActualEvapotranspirationResponse")
			.def_readwrite("ae", &response::ae)
			;
		def("ActualEvapotranspirationCalculate_step", calculate_step, args("soil-moisture", "potential_evapotranspiration", "lp", "snow_fraction", "dt"),
			" actual_evapotranspiration calculates actual evapotranspiration, returning same unit as input pot.evap\n"
			" based on supplied parameters\n"
			"\n"
			" * param water_level[mm]\n"
			" * param potential_evapotranspiration[mm/x]\n"
			" * param soil_moisture threshold, lp typically 150[mm]\n"
			" * param snow_fraction 0..1\n"
			" * return calculated actual evapotranspiration[mm/x]\n"
			"\n"
		);
	}
}
