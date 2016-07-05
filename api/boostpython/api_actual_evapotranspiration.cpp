#include "boostpython_pch.h"

#include "core/actual_evapotranspiration.h"

namespace expose {
    using namespace shyft::core::actual_evapotranspiration;
    using namespace boost::python;


    void actual_evapotranspiration() {
        class_<parameter>("ActualEvapotranspirationParameter")
            .def(init<optional<double>>(args("ae_scale_factor"),"a new object with specified parameters"))
            .def_readwrite("ae_scale_factor",&parameter::ae_scale_factor,"typical value 1.5")
            ;
        class_<response>("ActualEvapotranspirationResponse")
            .def_readwrite("ae",&response::ae)
            ;
        def("ActualEvapotranspirationCalculate_step",calculate_step,args("water_level","potential_evapotranspiration","scale_factor","snow_fraction","dt"),
             " actual_evapotranspiration calculates actual evapotranspiration for a timestep dt[s]\n"
             " based on supplied parameters\n"
             "\n"
             " * param water_level[mm]\n"
             " * param potential_evapotranspiration[mm/x]\n"
             " * param scale_factor typically 1.5[mm]\n"
             " * param snow_fraction 0..1\n"
             " * return calculated actual evapotranspiration[mm/x]\n"
             "\n"
            );
    }
}
