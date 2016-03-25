
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
#include "core/hbv_snow.h"
#include "core/kirchner.h"
#include "core/pt_hs_k.h"

static char const* version() {
   return "v1.0";
}


using namespace boost::python;
using namespace shyft::core;
using namespace shyft::core::pt_hs_k;

void def_pt_hs_k() {
    class_<parameter>("PTHSKParameter",
                      "Contains the parameters to the methods used in the PTHSK assembly\n"
                      "priestley_taylor,hbv_snow,actual_evapotranspiration,precipitation_correction,kirchner\n"
        )
        .def(init<const priestley_taylor::parameter&,const hbv_snow::parameter&,const actual_evapotranspiration::parameter&,const kirchner::parameter&,const precipitation_correction::parameter&>(args("pt","snow","ae","k","p_corr"),"create object with specified parameters"))
        .def_readwrite("pt",&parameter::pt,"priestley_taylor parameter")
        .def_readwrite("snow",&parameter::snow,"hbv-snow parameter")
        .def_readwrite("kirchner",&parameter::kirchner,"kirchner parameter")
        .def_readwrite("p_corr",&parameter::p_corr,"precipitation correction parameter")
        .def("size",&parameter::size,"returns total number of calibration parameters")
        .def("set",&parameter::set,args("p"),"set parameters from vector/list of float, ordered as by get_name(i)")
        .def("get",&parameter::get,args("i"),"return the value of the i'th parameter, name given by .get_name(i)")
        .def("get_name",&parameter::get_name,args("i"),"returns the i'th parameter name, see also .get()/.set() and .size()")
        ;


}

BOOST_PYTHON_MODULE(_pt_hs_k)
{

    scope().attr("__doc__")="SHyFT python api for the pt_hs_k model";
    def("version", version);
    def_pt_hs_k();
}

