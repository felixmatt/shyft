#include "boostpython_pch.h"

#include "core/actual_evapotranspiration.h"

using namespace shyft::core::actual_evapotranspiration;
using namespace boost::python;
using namespace std;

void def_actual_evapotranspiration() {
    class_<parameter>("ActualEvapotranspirationParameter")
        .def(init<optional<double>>(args("ae_scale_factor"),"a new object with specified parameters"))
        .def_readwrite("ae_scale_factor",&parameter::ae_scale_factor,"typical value 1.5")
        ;
    class_<response>("ActualEvapotranspirationResponse")
        .def_readwrite("ae",&response::ae)
        ;
    def("ActualEvapotranspirationCalculate_step",calculate_step,args("water_level","potential_evapotranspiration","scale_factor","snow_fraction","dt"),
         " actual_evapotranspiration calculates actual evapotranspiration for a timestep dt\n"
         " based on supplied parameters\n"
         "\n"
		 " * param water_level\n"
         " * param potential_evapotranspiration\n"
         " * param scale_factor typically 1.5\n"
         " * param snow_fraction 0..1\n"
		 " * return calculated actual evapotranspiration\n"
         "\n"
        );
}
