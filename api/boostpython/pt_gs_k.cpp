
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/scope.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/overloads.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/handle.hpp>

#include "core/core_pch.h"
#include "core/utctime_utilities.h"
#include "core/priestley_taylor.h"
#include "core/actual_evapotranspiration.h"
#include "core/precipitation_correction.h"
#include "core/gamma_snow.h"
#include "core/kirchner.h"
#include "core/pt_gs_k.h"
#include "api/pt_gs_k.h"
#include "core/pt_gs_k_cell_model.h"
#include "core/region_model.h"
#include "core/model_calibration.h"

static char const* version() {
   return "v1.0";
}


using namespace boost::python;
using namespace shyft::core;
using namespace shyft::core::pt_gs_k;
static void expose_state_io() {
    namespace sa=shyft::api;
    typedef shyft::api::pt_gs_k_state_io PTGSKStateIo;
    //using state=shyft::core::pt_gs_k::state;
    std::string (PTGSKStateIo::*to_string1)(const state& ) const = &PTGSKStateIo::to_string;
    std::string (PTGSKStateIo::*to_string2)(const std::vector<state>& ) const = &PTGSKStateIo::to_string;
    class_<PTGSKStateIo>("PTGSKStateIo")
     .def("from_string",&PTGSKStateIo::from_string,args("str","state"), "returns true if succeeded convertion string into state")
     .def("to_string",to_string1,args("state"),"convert a state into readable string")
     .def("to_string",to_string2,args("state_vector"),"convert a vector of state to a string" )
     .def("vector_from_string",&PTGSKStateIo::vector_from_string,args("s"),"given string s, convert it to a state vector")
     ;
}

static void expose_pt_gs_k() {
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

    class_<state>("PTGSKState")
        .def(init<gamma_snow::state,kirchner::state>(args("gs","k"),"initializes state with gamma-snow gs and kirchner k"))
        .def_readwrite("gs",&state::gs,"gamma-snow state")
        .def_readwrite("kirchner",&state::kirchner,"kirchner state")
        ;

    class_<response>("PTGSKResponse","This struct contains the responses of the methods used in the PTGSK assembly")
        .def_readwrite("pt",&response::pt,"priestley_taylor response")
        .def_readwrite("gs",&response::gs,"gamma-snnow response")
        .def_readwrite("ae",&response::ae,"actual evapotranspiration response")
        .def_readwrite("kirchner",&response::kirchner,"kirchner response")
        .def_readwrite("total_discharge",&response::total_discharge,"total stack response")
        ;
    typedef std::vector<state> PTGSKStateVector;
    class_<PTGSKStateVector>("PTGSKStateVector")
     .def(vector_indexing_suite<PTGSKStateVector>())
        ;
  #if 0
            %template(PTGSKCellAll)  cell<parameter, environment_t, state, state_collector, all_response_collector>;
        typedef cell<parameter, environment_t, state, state_collector, all_response_collector> PTGSKCellAll;
        %template(PTGSKCellOpt)     cell<parameter, environment_t, state, null_collector, discharge_collector>;
        typedef cell<parameter, environment_t, state, null_collector, discharge_collector> PTGSKCellOpt;
#endif
    //TODO: consider exposing the calculator
}
//extern void def_api();
BOOST_PYTHON_MODULE(_pt_gs_k)
{

    scope().attr("__doc__")="SHyFT python api for the pt_gs_k model";
    def("version", version);
    expose_pt_gs_k();
    expose_state_io();
}
