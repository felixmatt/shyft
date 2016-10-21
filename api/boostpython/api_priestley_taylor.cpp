#include "boostpython_pch.h"

#include "core/priestley_taylor.h"

namespace expose {

    void priestley_taylor() {
        using namespace shyft::core::priestley_taylor;
        using namespace boost::python;

        class_<parameter>("PriestleyTaylorParameter")
            .def(init<optional<double,double>>(args("albedo","alpha"),"a new object with specified parameters"))
            .def_readwrite("albedo",&parameter::albedo,"typical value 0.2")
            .def_readwrite("alpha", &parameter::alpha,"typical value 1.26")
            ;

        class_<response>("PriestleyTaylorResponse")
            .def_readwrite("pot_evapotranspiration",&response::pot_evapotranspiration)
            ;

        class_<calculator>("PriestleyTaylorCalculator",
                "PriestleyTaylor,PT, (google for PriestleyTaylor)\n"
                "primitive implementation for calculating the potential evaporation.\n"
                "This function is plain and simple, taking land_albedo and PT.alpha\n"
                "into the constructor and provides a function that calculates potential evapotransporation\n"
                "[mm/s] units.\n",no_init
            )
            .def(init<double,double>(args("land_albedo","alpha")))
            .def("potential_evapotranspiration",&calculator::potential_evapotranspiration,args("temperature","global_radiation","rhumidity"),
                "Calculate PotentialEvapotranspiration, given specified parameters\n"
                "\n"
                "	 * param temperature in [degC]\n"
                "	 * param global_radiation [W/m2]\n"
                "	 * param rhumidity in interval [0,1]\n"
                "	 * return PotentialEvapotranspiration in [mm/s] units\n"
                "	 *\n"
                 )
            ;
    }
}
