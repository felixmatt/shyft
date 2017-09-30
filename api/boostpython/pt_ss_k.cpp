
#include "boostpython_pch.h"

#include "core/utctime_utilities.h"
#include "core/priestley_taylor.h"
#include "core/actual_evapotranspiration.h"
#include "core/precipitation_correction.h"
#include "core/skaugen.h"
#include "core/kirchner.h"
#include "core/pt_ss_k.h"
#include "api/api.h"
#include "api/pt_ss_k.h"
#include "core/pt_ss_k_cell_model.h"
#include "core/region_model.h"
#include "core/model_calibration.h"
#include "expose_statistics.h"
#include "expose.h"

static char const* version() {
   return "v1.0";
}

namespace expose {
    namespace pt_ss_k {
        using namespace boost::python;
        using namespace shyft::core;
        using namespace shyft::core::pt_ss_k;

        static void
        parameter_state_response() {

            class_<parameter,bases<>,std::shared_ptr<parameter>>("PTSSKParameter",
                              "Contains the parameters to the methods used in the PTSSK assembly\n"
                              "priestley_taylor,skaugen,actual_evapotranspiration,precipitation_correction,kirchner\n"
                )
                .def(init<priestley_taylor::parameter,skaugen::parameter,actual_evapotranspiration::parameter,kirchner::parameter,precipitation_correction::parameter, optional<glacier_melt::parameter,routing::uhg_parameter>>(args("pt","gs","ae","k","p_corr","gm","routing"),"create object with specified parameters"))
                .def(init<const parameter&>(args("p"),"clone a parameter"))
                .def_readwrite("pt",&parameter::pt,"priestley_taylor parameter")
                .def_readwrite("ae", &parameter::ae, "actual evapotranspiration parameter")
                .def_readwrite("ss",&parameter::ss,"skaugen-snow parameter")
                .def_readwrite("gm", &parameter::gm, "glacier melt parameter")
                .def_readwrite("kirchner",&parameter::kirchner,"kirchner parameter")
                .def_readwrite("p_corr",&parameter::p_corr,"precipitation correction parameter")
                .def_readwrite("routing",&parameter::routing,"routing cell-to-river catchment specific parameters")
                .def("size",&parameter::size,"returns total number of calibration parameters")
                .def("set",&parameter::set,args("p"),"set parameters from vector/list of float, ordered as by get_name(i)")
                .def("get",&parameter::get,args("i"),"return the value of the i'th parameter, name given by .get_name(i)")
                .def("get_name",&parameter::get_name,args("i"),"returns the i'th parameter name, see also .get()/.set() and .size()")
                ;

            typedef std::map<int,parameter> PTSSKParameterMap;
            class_<PTSSKParameterMap>("PTSSKParameterMap","dict (int,parameter)  where the int is 0-based catchment_id")
                .def(map_indexing_suite<PTSSKParameterMap>())
            ;

            class_<state>("PTSSKState")
                .def(init<skaugen::state,kirchner::state>(args("snow","kirchner"),"initializes state with skaugen-snow gs and kirchner k"))
                .def_readwrite("snow",&state::snow,"skaugen-snow state")
                .def_readwrite("kirchner",&state::kirchner,"kirchner state")
                ;

            typedef std::vector<state> PTSSKStateVector;
            class_<PTSSKStateVector,bases<>,std::shared_ptr<PTSSKStateVector> >("PTSSKStateVector")
                .def(vector_indexing_suite<PTSSKStateVector>())
                ;


            class_<response>("PTSSKResponse","This struct contains the responses of the methods used in the PTSSK assembly")
                .def_readwrite("pt",&response::pt,"priestley_taylor response")
                .def_readwrite("snow",&response::snow,"skaugen-snow response")
                .def_readwrite("gm_melt_m3s", &response::gm_melt_m3s, "glacier melt response[m3s]")
                .def_readwrite("ae",&response::ae,"actual evapotranspiration response")
                .def_readwrite("kirchner",&response::kirchner,"kirchner response")
                .def_readwrite("total_discharge",&response::total_discharge,"total stack response")
                ;
        }

        static void
        collectors() {
            typedef shyft::core::pt_ss_k::all_response_collector PTSSKAllCollector;
            class_<PTSSKAllCollector>("PTSSKAllCollector", "collect all cell response from a run")
                .def_readonly("destination_area",&PTSSKAllCollector::destination_area,"a copy of cell area [m2]")
                .def_readonly("avg_discharge",&PTSSKAllCollector::avg_discharge,"Kirchner Discharge given in [m^3/s] for the timestep")
                .def_readonly("snow_total_stored_water",&PTSSKAllCollector::snow_total_stored_water," skaugen aka sca*(swe + lwc) in [mm]")
                .def_readonly("snow_outflow",&PTSSKAllCollector::snow_outflow," skaugen snow output [m^3/s] for the timestep")
                .def_readonly("glacier_melt", &PTSSKAllCollector::glacier_melt, " glacier melt (outflow) [m3/s] for the timestep")
                .def_readonly("ae_output",&PTSSKAllCollector::ae_output,"actual evap mm/h")
                .def_readonly("pe_output",&PTSSKAllCollector::pe_output,"pot evap mm/h")
                .def_readonly("end_reponse",&PTSSKAllCollector::end_reponse,"end_response, at the end of collected")
            ;

            typedef shyft::core::pt_ss_k::discharge_collector PTSSKDischargeCollector;
            class_<PTSSKDischargeCollector>("PTSSKDischargeCollector", "collect all cell response from a run")
                .def_readonly("destination_area",&PTSSKDischargeCollector::destination_area,"a copy of cell area [m2]")
                .def_readonly("avg_discharge",&PTSSKDischargeCollector::avg_discharge,"Kirchner Discharge given in [m^3/s] for the timestep")
                .def_readonly("snow_sca",&PTSSKDischargeCollector::snow_sca," skaugen snow covered area fraction, sca.. 0..1 - at the end of timestep (state)")
                .def_readonly("snow_swe",&PTSSKDischargeCollector::snow_swe,"skaugen snow swe, [mm] over the cell sca.. area, - at the end of timestep")
                .def_readonly("end_reponse",&PTSSKDischargeCollector::end_response,"end_response, at the end of collected")
                .def_readwrite("collect_snow",&PTSSKDischargeCollector::collect_snow,"controls collection of snow routine")
                ;
            typedef shyft::core::pt_ss_k::null_collector PTSSKNullCollector;
            class_<PTSSKNullCollector>("PTSSKNullCollector","collector that does not collect anything, useful during calibration to minimize memory&maximize speed")
                ;

            typedef shyft::core::pt_ss_k::state_collector PTSSKStateCollector;
            class_<PTSSKStateCollector>("PTSSKStateCollector","collects state, if collect_state flag is set to true")
                .def_readwrite("collect_state",&PTSSKStateCollector::collect_state,"if true, collect state, otherwise ignore (and the state of time-series are undefined/zero)")
                .def_readonly("kirchner_discharge",&PTSSKStateCollector::kirchner_discharge,"Kirchner state instant Discharge given in m^3/s")
                .def_readonly("snow_swe",&PTSSKStateCollector::snow_swe,"")
                .def_readonly("snow_sca",&PTSSKStateCollector::snow_sca,"")
                .def_readonly("snow_alpha",&PTSSKStateCollector::snow_alpha,"")
                .def_readonly("snow_nu",&PTSSKStateCollector::snow_nu,"")
                .def_readonly("snow_lwc",&PTSSKStateCollector::snow_lwc,"")
                .def_readonly("snow_residual",&PTSSKStateCollector::snow_residual,"")
            ;

        }

        static void
        cells() {
              typedef shyft::core::cell<parameter, environment_t, state, state_collector, all_response_collector> PTSSKCellAll;
              typedef shyft::core::cell<parameter, environment_t, state, null_collector, discharge_collector> PTSSKCellOpt;
              expose::cell<PTSSKCellAll>("PTSSKCellAll","tbd: PTSSKCellAll doc");
              expose::cell<PTSSKCellOpt>("PTSSKCellOpt","tbd: PTSSKCellOpt doc");
              expose::statistics::skaugen<PTSSKCellAll>("PTSSKCell");//it only gives meaning to expose the *All collect cell-type
              expose::statistics::actual_evapotranspiration<PTSSKCellAll>("PTSSKCell");
              expose::statistics::priestley_taylor<PTSSKCellAll>("PTSSKCell");
              expose::statistics::kirchner<PTSSKCellAll>("PTSSKCell");
              expose::cell_state_etc<PTSSKCellAll>("PTSSK");// just one expose of state
        }

        static void
        models() {
            typedef shyft::core::region_model<pt_ss_k::cell_discharge_response_t, shyft::api::a_region_environment> PTSSKOptModel;
            typedef shyft::core::region_model<pt_ss_k::cell_complete_response_t, shyft::api::a_region_environment> PTSSKModel;
            expose::model<PTSSKModel>("PTSSKModel","PTSSK");
            expose::model<PTSSKOptModel>("PTSSKOptModel","PTSSK");
            def_clone_to_similar_model<PTSSKModel, PTSSKOptModel>("create_opt_model_clone");
            def_clone_to_similar_model<PTSSKOptModel, PTSSKModel>("create_full_model_clone");
        }

        static void
        state_io() {
            expose::state_io<shyft::api::pt_ss_k_state_io,shyft::core::pt_ss_k::state>("PTSSKStateIo");
        }


        static void
        model_calibrator() {
            expose::model_calibrator<shyft::core::region_model<pt_ss_k::cell_discharge_response_t, shyft::api::a_region_environment>>("PTSSKOptimizer");
        }

    }
}


BOOST_PYTHON_MODULE(_pt_ss_k)
{

    boost::python::scope().attr("__doc__")="SHyFT python api for the pt_ss_k model";
    boost::python::def("version", version);
	boost::python::docstring_options doc_options(true, true, false);// all except c++ signatures
    expose::pt_ss_k::state_io();
    expose::pt_ss_k::parameter_state_response();
    expose::pt_ss_k::cells();
    expose::pt_ss_k::models();
    expose::pt_ss_k::collectors();
    expose::pt_ss_k::model_calibrator();
}
