#include "boostpython_pch.h"

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
    class_<state>("PTHSKState")
        .def(init<hbv_snow::state,kirchner::state>(args("snow","k"),"initializes state with hbv-snow gs and kirchner k"))
        .def_readwrite("snow",&state::snow,"hbv-snow state")
        .def_readwrite("kirchner",&state::kirchner,"kirchner state")
        ;

    class_<response>("PTHSKResponse","This struct contains the responses of the methods used in the PTHSK assembly")
        .def_readwrite("pt",&response::pt,"priestley_taylor response")
        .def_readwrite("snow",&response::snow,"hbc-snow response")
        .def_readwrite("ae",&response::ae,"actual evapotranspiration response")
        .def_readwrite("kirchner",&response::kirchner,"kirchner response")
        .def_readwrite("total_discharge",&response::total_discharge,"total stack response")
        ;


}

BOOST_PYTHON_MODULE(_pt_hs_k)
{

    scope().attr("__doc__")="SHyFT python api for the pt_hs_k model";
    def("version", version);
    def_pt_hs_k();
}

