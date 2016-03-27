#include "boostpython_pch.h"

#include "core/precipitation_correction.h"

namespace expose {

    void precipitation_correction() {
        using namespace shyft::core::precipitation_correction;
        using namespace boost::python;

        class_<parameter>("PrecipitationCorrectionParameter")
            .def(init<double>(args("scale_factor"),"creates parameter object according to parameters"))
            .def_readwrite("scale_factor",&parameter::scale_factor,"default =1.0")
            ;
        class_<calculator>("PrecipitationCorrectionCalculator",
                "Scales precipitation with the specified scale factor"
            )
            .def(init<double>(args("scale_factor"),"create a calculator using supplied parameter"))
            .def("calc",&calculator::calc,args("precipitation"),"returns scale_factor*precipitation\n")
            ;
     }
}
