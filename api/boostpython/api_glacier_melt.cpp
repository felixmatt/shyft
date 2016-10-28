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
        class_<response>("GlacierMeltResponse","The response(output) from galcier-melt for one time-step")
            .def_readwrite("glacier_melt",&response::glacier_melt,"Glacier melt (outflow) [mm/h]")
            ;
        typedef calculator<parameter,response> GlacierMeltCalculator;
        class_<GlacierMeltCalculator>("GlacierMeltCalculator")
            .def("set_glacier_fraction",&GlacierMeltCalculator::set_glacier_fraction,"set the glacier fraction parameter for the calculation")
            .def("glacier_fraction",&GlacierMeltCalculator::glacier_fraction,"get the glacier fraction parameter used for the calculation")
            .def("step",&GlacierMeltCalculator::step,args("response","dt","parameter","temperature","sca"),
                "Step the snow model forward from time t to t+dt, given parameters and input.\n"
                "Updates the response upon return.\n"
                " param response result of type R, output only, ref. template parameters\n"
                " param temperature degC, considered constant over timestep dt\n"
                " param sca, fraction of snow cover in cell [0..1], glacier melt occurs only if glacier fraction > snow fraction\n"
            )
            ;

    }
}
