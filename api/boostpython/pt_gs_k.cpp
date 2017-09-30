#include "boostpython_pch.h"

#include "core/utctime_utilities.h"
#include "core/priestley_taylor.h"
#include "core/actual_evapotranspiration.h"
#include "core/precipitation_correction.h"
#include "core/gamma_snow.h"
#include "core/kirchner.h"
#include "core/pt_gs_k.h"
#include "api/api.h"
#include "api/pt_gs_k.h"
#include "core/pt_gs_k_cell_model.h"
#include "core/region_model.h"
#include "core/model_calibration.h"
#include "expose_statistics.h"
#include "expose.h"

static char const* version() {
   return "v1.0";
}

namespace expose {
    namespace pt_gs_k {
        using namespace boost::python;
        using namespace shyft::core;
        using namespace shyft::core::pt_gs_k;

        static void
        parameter_state_response() {

            class_<parameter,bases<>,std::shared_ptr<parameter>>("PTGSKParameter",
                              "Contains the parameters to the methods used in the PTGSK assembly\n"
                              "priestley_taylor,gamma_snow,actual_evapotranspiration,precipitation_correction,kirchner\n"
                )
                .def(init<priestley_taylor::parameter,gamma_snow::parameter,actual_evapotranspiration::parameter,kirchner::parameter,precipitation_correction::parameter, optional<glacier_melt::parameter,routing::uhg_parameter>>(args("pt","gs","ae","k","p_corr","gm","routing"),"create object with specified parameters"))
                .def(init<const parameter&>(args("p"),"clone a parameter"))
                .def_readwrite("pt",&parameter::pt,"priestley_taylor parameter")
                .def_readwrite("gs",&parameter::gs,"gamma-snow parameter")
                .def_readwrite("gm", &parameter::gm, "glacier melt parameter")
				.def_readwrite("ae",&parameter::ae,"actual evapotranspiration parameter")
                .def_readwrite("kirchner",&parameter::kirchner,"kirchner parameter")
                .def_readwrite("p_corr",&parameter::p_corr,"precipitation correction parameter")
                .def_readwrite("routing",&parameter::routing,"routing cell-to-river catchment specific parameters")
                .def("size",&parameter::size,"returns total number of calibration parameters")
                .def("set",&parameter::set,args("p"),"set parameters from vector/list of float, ordered as by get_name(i)")
                .def("get",&parameter::get,args("i"),"return the value of the i'th parameter, name given by .get_name(i)")
                .def("get_name",&parameter::get_name,args("i"),"returns the i'th parameter name, see also .get()/.set() and .size()")
                ;

            typedef std::map<int,parameter> PTGSKParameterMap;
            class_<PTGSKParameterMap>("PTGSKParameterMap","dict (int,parameter)  where the int is the catchment_id")
                .def(map_indexing_suite<PTGSKParameterMap>())
            ;

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
                .def_readwrite("gm_melt_m3s", &response::gm_melt_m3s, "glacier melt response[m3s]")
                .def_readwrite("ae",&response::ae,"actual evapotranspiration response")
                .def_readwrite("kirchner",&response::kirchner,"kirchner response")
                .def_readwrite("total_discharge",&response::total_discharge,"total stack response")
                ;
        }

        static void
        collectors() {
            typedef shyft::core::pt_gs_k::all_response_collector PTGSKAllCollector;
            class_<PTGSKAllCollector>("PTGSKAllCollector", "collect all cell response from a run")
                .def_readonly("destination_area",&PTGSKAllCollector::destination_area,"a copy of cell area [m2]")
                .def_readonly("avg_discharge",&PTGSKAllCollector::avg_discharge,"Kirchner Discharge given in [m^3/s] for the timestep")
                .def_readonly("snow_sca",&PTGSKAllCollector::snow_sca," gamma snow covered area fraction, sca.. 0..1 - at the end of timestep (state)")
                .def_readonly("snow_swe",&PTGSKAllCollector::snow_swe,"gamma snow swe, [mm] over the cell sca.. area, - at the end of timestep")
                .def_readonly("snow_outflow",&PTGSKAllCollector::snow_outflow," gamma snow output [m^3/s] for the timestep")
                .def_readonly("glacier_melt", &PTGSKAllCollector::glacier_melt, " glacier melt (outflow) [m3/s] for the timestep")
                .def_readonly("ae_output",&PTGSKAllCollector::ae_output,"actual evap mm/h")
                .def_readonly("pe_output",&PTGSKAllCollector::pe_output,"pot evap mm/h")
                .def_readonly("end_reponse",&PTGSKAllCollector::end_reponse,"end_response, at the end of collected")
            ;

            typedef shyft::core::pt_gs_k::discharge_collector PTGSKDischargeCollector;
            class_<PTGSKDischargeCollector>("PTGSKDischargeCollector", "collect all cell response from a run")
                .def_readonly("cell_area",&PTGSKDischargeCollector::cell_area,"a copy of cell area [m2]")
                .def_readonly("avg_discharge",&PTGSKDischargeCollector::avg_discharge,"Kirchner Discharge given in [m^3/s] for the timestep")
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

        static void
        cells() {
              typedef shyft::core::cell<parameter, environment_t, state, state_collector, all_response_collector> PTGSKCellAll;
              typedef shyft::core::cell<parameter, environment_t, state, null_collector, discharge_collector> PTGSKCellOpt;
              expose::cell<PTGSKCellAll>("PTGSKCellAll","tbd: PTGSKCellAll doc");
              expose::cell<PTGSKCellOpt>("PTGSKCellOpt","tbd: PTGSKCellOpt doc");
              expose::statistics::gamma_snow<PTGSKCellAll>("PTGSKCell");//it only gives meaning to expose the *All collect cell-type
              expose::statistics::actual_evapotranspiration<PTGSKCellAll>("PTGSKCell");
              expose::statistics::priestley_taylor<PTGSKCellAll>("PTGSKCell");
              expose::statistics::kirchner<PTGSKCellAll>("PTGSKCell");
              expose::cell_state_etc<PTGSKCellAll>("PTGSK");// just one expose of state

        }

        static void
        models() {
            typedef shyft::core::region_model<pt_gs_k::cell_discharge_response_t, shyft::api::a_region_environment> PTGSKOptModel;
            typedef shyft::core::region_model<pt_gs_k::cell_complete_response_t, shyft::api::a_region_environment> PTGSKModel;
            expose::model<PTGSKModel>("PTGSKModel","PTGSK");
            expose::model<PTGSKOptModel>("PTGSKOptModel","PTGSK");
            def_clone_to_similar_model<PTGSKModel, PTGSKOptModel>("create_opt_model_clone");
            def_clone_to_similar_model<PTGSKOptModel,PTGSKModel>("create_full_model_clone");
        }

        static void
        state_io() {
            expose::state_io<shyft::api::pt_gs_k_state_io,shyft::core::pt_gs_k::state>("PTGSKStateIo");
        }


        static void
        model_calibrator() {
            expose::model_calibrator<shyft::core::region_model<pt_gs_k::cell_discharge_response_t,shyft::api::a_region_environment>>("PTGSKOptimizer");
        }
    }
}


BOOST_PYTHON_MODULE(_pt_gs_k)
{

    boost::python::scope().attr("__doc__")="SHyFT python api for the pt_gs_k model";
    boost::python::def("version", version);
	boost::python::docstring_options doc_options(true, true, false);// all except c++ signatures
    expose::pt_gs_k::state_io();
    expose::pt_gs_k::parameter_state_response();
    expose::pt_gs_k::cells();
    expose::pt_gs_k::models();
    expose::pt_gs_k::collectors();
    expose::pt_gs_k::model_calibrator();
}
