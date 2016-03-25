
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/scope.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/overloads.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/handle.hpp>

#include "core/utctime_utilities.h"
#include "core/priestley_taylor.h"
#include "core/actual_evapotranspiration.h"
#include "core/precipitation_correction.h"
#include "core/gamma_snow.h"
#include "core/kirchner.h"
#include "core/pt_gs_k.h"

static char const* version() {
   return "v1.0";
}


using namespace boost::python;
using namespace shyft::core;
using namespace shyft::core::pt_gs_k;

void def_pt_gs_k() {
    class_<parameter>("PTGSKParameter",
                      "Contains the parameters to the methods used in the PTGSK assembly\n"
                      "priestley_taylor,gamma_snow,actual_evapotranspiration,precipitation_correction,kirchner\n"
        )
        .def(init<priestley_taylor::parameter,gamma_snow::parameter,actual_evapotranspiration::parameter,kirchner::parameter,precipitation_correction::parameter>(args("pt","gs","ae","k","p_corr"),"create object with specified parameters"))
        .def_readwrite("pt",&parameter::pt,"priestley_taylor parameter")
        .def_readwrite("gs",&parameter::gs,"gamma-snow parameter")
        .def_readwrite("kirchner",&parameter::kirchner,"kirchner parameter")
        .def_readwrite("p_corr",&parameter::p_corr,"precipitation correction parameter")
        .def("size",&parameter::size,"returns total number of calibration parameters")
        .def("set",&parameter::set,args("p"),"set parameters from vector/list of float, ordered as by get_name(i)")
        .def("get",&parameter::get,args("i"),"return the value of the i'th parameter, name given by .get_name(i)")
        .def("get_name",&parameter::get_name,args("i"),"returns the i'th parameter name, see also .get()/.set() and .size()")
        ;

}
//extern void def_api();
BOOST_PYTHON_MODULE(_pt_gs_k)
{

    scope().attr("__doc__")="SHyFT python api for the pt_gs_k model";
    def("version", version);
    def_pt_gs_k();
}
