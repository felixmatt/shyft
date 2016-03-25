
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
    class_<parameter,bases<>,std::shared_ptr<parameter>>("PTGSKParameter",
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
    register_ptr_to_python<std::shared_ptr<parameter> >();

    class_<state>("PTGSKState")
        .def(init<gamma_snow::state,kirchner::state>(args("gs","k"),"initializes state with gamma-snow gs and kirchner k"))
        .def_readwrite("gs",&state::gs,"gamma-snow state")
        .def_readwrite("kirchner",&state::kirchner,"kirchner state")
        ;

    typedef std::vector<state> PTGSKStateVector;
    class_<PTGSKStateVector,bases<>,std::shared_ptr<PTGSKStateVector> >("PTGSKStateVector")
        .def(vector_indexing_suite<PTGSKStateVector>())
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
static void expose_pt_gs_k_collectors() {
    typedef shyft::core::pt_gs_k::all_response_collector PTGSKAllCollector;
    class_<PTGSKAllCollector>("PTGSKAllCollector", "collect all cell response from a run")
        .def_readonly("destination_area",&PTGSKAllCollector::destination_area,"a copy of cell area [m2]")
        .def_readonly("avg_discharge",&PTGSKAllCollector::avg_discharge,"Kirchner Discharge given in [m³/s] for the timestep")
        .def_readonly("snow_sca",&PTGSKAllCollector::snow_sca," gamma snow covered area fraction, sca.. 0..1 - at the end of timestep (state)")
        .def_readonly("snow_swe",&PTGSKAllCollector::snow_swe,"gamma snow swe, [mm] over the cell sca.. area, - at the end of timestep")
        .def_readonly("snow_outflow",&PTGSKAllCollector::snow_outflow," gamma snow output [m³/s] for the timestep")
        .def_readonly("ae_output",&PTGSKAllCollector::ae_output,"actual evap mm/h")
        .def_readonly("pe_output",&PTGSKAllCollector::pe_output,"pot evap mm/h")
        .def_readonly("end_reponse",&PTGSKAllCollector::end_reponse,"end_response, at the end of collected")
    ;

    typedef shyft::core::pt_gs_k::discharge_collector PTGSKDischargeCollector;
    class_<PTGSKDischargeCollector>("PTGSKDischargeCollector", "collect all cell response from a run")
        .def_readonly("cell_area",&PTGSKDischargeCollector::cell_area,"a copy of cell area [m2]")
        .def_readonly("avg_discharge",&PTGSKDischargeCollector::avg_discharge,"Kirchner Discharge given in [m³/s] for the timestep")
        .def_readonly("snow_sca",&PTGSKDischargeCollector::snow_sca," gamma snow covered area fraction, sca.. 0..1 - at the end of timestep (state)")
        .def_readonly("snow_swe",&PTGSKDischargeCollector::snow_swe,"gamma snow swe, [mm] over the cell sca.. area, - at the end of timestep")
        .def_readonly("end_reponse",&PTGSKDischargeCollector::end_response,"end_response, at the end of collected")
        .def_readwrite("collect_snow",&PTGSKDischargeCollector::collect_snow,"controls collection of snow routine")
        ;
    typedef shyft::core::pt_gs_k::null_collector PTGSKNullCollector;
    class_<PTGSKNullCollector>("PTGSKNullCollector","collector that does not collect anything, useful during calibration to minimize memory&maximize speed")
        ;

    typedef shyft::core::pt_gs_k::state_collector PTGSKStateCollector;
    class_<PTGSKStateCollector>("PTGSKStateCollector","collects state, if collect_state flag is set to true")
        .def_readwrite("collect_state",&PTGSKStateCollector::collect_state,"if true, collect state, otherwise ignore (and the state of time-series are undefined/zero)")
        .def_readonly("kirchner_discharge",&PTGSKStateCollector::kirchner_discharge,"Kirchner state instant Discharge given in m^3/s")
        .def_readonly("gs_albedo",&PTGSKStateCollector::gs_albedo,"")
        .def_readonly("gs_lwc",&PTGSKStateCollector::gs_lwc,"")
        .def_readonly("gs_surface_heat",&PTGSKStateCollector::gs_surface_heat,"")
        .def_readonly("gs_alpha",&PTGSKStateCollector::gs_alpha,"")
        .def_readonly("gs_sdc_melt_mean",&PTGSKStateCollector::gs_sdc_melt_mean,"")
        .def_readonly("gs_acc_melt",&PTGSKStateCollector::gs_acc_melt,"")
        .def_readonly("gs_iso_pot_energy",&PTGSKStateCollector::gs_iso_pot_energy,"")
        .def_readonly("gs_temp_swe",&PTGSKStateCollector::gs_temp_swe,"")
    ;

}
template <class T>
static void expose_cell(const char *cell_name,const char* cell_doc) {
  class_<T>(cell_name,cell_doc)
    .def_readwrite("geo",&T::geo,"geo_cell_data information for the cell")
    .def_readwrite("parameter",&T::parameter,"reference to parameter for this cell, typically shared for a catchment")
    .def_readwrite("env_ts",&T::env_ts,"environment time-series as projected to the cell")
    .def_readonly("sc",&T::sc,"state collector for the cell")
    .def_readonly("rc",&T::rc,"response collector for the cell")
  ;

}
static void expose_pt_gs_k_cell() {
  typedef cell<parameter, environment_t, state, state_collector, all_response_collector> PTGSKCellAll;
  typedef cell<parameter, environment_t, state, null_collector, discharge_collector> PTGSKCellOpt;
  expose_cell<PTGSKCellAll>("PTGSKCellAll","tbd: PTGSKCellAll doc");
  expose_cell<PTGSKCellOpt>("PTGSKCellOpt","tbd: PTGSKCellOpt doc");
}

template <class M>
static void expose_model(const char *model_name,const char *model_doc) {

}

static void expose_pt_gs_k_model() {
    typedef region_model<pt_gs_k::cell_discharge_response_t> PTGSKOptModel;
    typedef region_model<pt_gs_k::cell_complete_response_t> PTGSKModel;
}
BOOST_PYTHON_MODULE(_pt_gs_k)
{

    scope().attr("__doc__")="SHyFT python api for the pt_gs_k model";
    def("version", version);
    expose_pt_gs_k();
    expose_state_io();
    expose_pt_gs_k_cell();
    expose_pt_gs_k_collectors();
}
