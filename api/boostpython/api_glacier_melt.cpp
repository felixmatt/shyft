#include "boostpython_pch.h"

#include "core/glacier_melt.h"

namespace expose {

    using namespace shyft::core::glacier_melt;
    namespace py=boost::python;
    using namespace std;
    void glacier_melt() {
        py::class_<parameter>("GlacierMeltParameter")
            .def(py::init<double,py::optional<double>>((py::arg("dtf"),py::arg("direct_response")=0.0),"create parameter object with specified values"))
            .def_readwrite("dtf", &parameter::dtf,"degree timestep factor, default=6.0 [mm/day/degC]")
            .def_readwrite("direct_response",&parameter::direct_response,"fraction that goes as direct response, (1-fraction) is routed through soil/kirchner routine,default=0.0")
            ;
        py::def("glacier_melt_step", step,(py::arg("dtf"),py::arg("temperature"), py::arg("sca"), py::arg("glacier_fraction")),
            doc_intro("Calculates outflow from glacier melt rate [mm/h].")
            doc_parameters()
            doc_parameter("dtf","float","degree timestep factor [mm/day/deg.C]; lit. values for Norway: 5.5 - 6.4 in Hock, R. (2003), J. Hydrol., 282, 104-115.")
            doc_parameter("temperature","float","degC, considered constant over timestep dt")
            doc_parameter("sca","float","fraction of snow cover in cell [0..1], glacier melt occurs only if glacier fraction > snow fraction")
            doc_parameter("glacier_fraction","float","glacier fraction [0..1] in the total area")
            doc_returns("glacier_melt","float","output from glacier, melt rate [mm/h]")
        );
    }
}
