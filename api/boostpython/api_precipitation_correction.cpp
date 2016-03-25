#include <boost/python/def.hpp>
#include <boost/python/args.hpp>
#include <boost/python/class.hpp>

#include "core/precipitation_correction.h"

using namespace shyft::core::precipitation_correction;
using namespace boost::python;
using namespace std;

void def_precipitation_correction() {

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
