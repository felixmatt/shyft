#include "boostpython_pch.h"

#include "core/glacier_melt.h"

namespace expose {

    using namespace shyft::core::glacier_melt;
    using namespace boost::python;
    using namespace std;
    void glacier_melt() {
        class_<parameter>("GlacierMeltParameter")
            .def(init<double>(args("dtf"),"create parameter object with specified values"))
            .def_readwrite("dtf", &parameter::dtf,"degree timestep factor, default=6.0 [mm/day/degC]")
            ;
        def("glacier_melt_step", step, args("dtf","temperature","sca","glacier_fraction"),
            "Calculates outflow from glacier melt rate [mm/h].\n"
            "Parameters\n"
            "----------\n"
            "dtf : float\n"
            "\t degree timestep factor [mm/day/deg.C]; lit. values for Norway: 5.5 - 6.4 in Hock, R. (2003), J. Hydrol., 282, 104-115.\n"
            "temperature : float\n"
            "\t degC, considered constant over timestep dt\n"
            "sca : float\n"
            "\t fraction of snow cover in cell [0..1], glacier melt occurs only if glacier fraction > snow fraction\n"
            "glacier_fraction : float\n"
            "\t glacier fraction [0..1] in the total area\n"
            "Return\n"
            "------\n"
            "glacier_melt : float\n"
            "\t outflow from glacier melt rate [mm/h].\n"
        );
    }
}
