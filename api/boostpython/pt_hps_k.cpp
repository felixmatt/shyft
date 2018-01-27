#include "boostpython_pch.h"
#include <boost/python/docstring_options.hpp>
#include "core/utctime_utilities.h"
#include "core/priestley_taylor.h"
#include "core/actual_evapotranspiration.h"
#include "core/precipitation_correction.h"
#include "core/hbv_physical_snow.h"
#include "core/glacier_melt.h"
#include "core/kirchner.h"
#include "core/pt_hps_k.h"
#include "api/api.h"
#include "core/pt_hps_k_cell_model.h"
#include "core/region_model.h"
#include "core/model_calibration.h"
#include "expose_statistics.h"
#include "expose.h"

static char const* version() {
   return "v1.0";
}

namespace expose {
    namespace pt_hps_k {
        using namespace boost::python;
        using namespace shyft::core;
        using namespace shyft::core::pt_hps_k;

        static void parameter_state_response() {
            class_<parameter,bases<>,std::shared_ptr<parameter>>("PTHPSKParameter",
                              "Contains the parameters to the methods used in the PTHPSK assembly\n"
                              "priestley_taylor,hbv_physical_snow,actual_evapotranspiration,precipitation_correction,kirchner\n"
                )
                .def(init<const priestley_taylor::parameter&,const hbv_physical_snow::parameter&,const actual_evapotranspiration::parameter&,const kirchner::parameter&,const precipitation_correction::parameter&,optional<glacier_melt::parameter,routing::uhg_parameter>>(args("pt","hps","ae","k","p_corr","gm","routing"),"create object with specified parameters"))
                .def(init<const parameter&>(args("p"),"clone a parameter"))
                .def_readwrite("pt",&parameter::pt,"priestley_taylor parameter")
				.def_readwrite("ae", &parameter::ae, "actual evapotranspiration parameter")
                .def_readwrite("hps",&parameter::hps,"hbv-physical-snow parameter")
                .def_readwrite("gm",&parameter::gm,"glacier melt parameter")
                .def_readwrite("kirchner",&parameter::kirchner,"kirchner parameter")
                .def_readwrite("p_corr",&parameter::p_corr,"precipitation correction parameter")
                .def_readwrite("routing",&parameter::routing,"routing cell-to-river catchment specific parameters")
                .def("size",&parameter::size,"returns total number of calibration parameters")
                .def("set",&parameter::set,args("p"),"set parameters from vector/list of float, ordered as by get_name(i)")
                .def("get",&parameter::get,args("i"),"return the value of the i'th parameter, name given by .get_name(i)")
                .def("get_name",&parameter::get_name,args("i"),"returns the i'th parameter name, see also .get()/.set() and .size()")
                ;
            typedef std::map<int,parameter> PTHPSKParameterMap;
            class_<PTHPSKParameterMap>("PTHPSKParameterMap","dict (int,parameter)  where the int is 0-based catchment_id")
                .def(map_indexing_suite<PTHPSKParameterMap>())
            ;

            class_<state>("PTHPSKState")
                .def(init<hbv_physical_snow::state,kirchner::state>(args("hps","k"),"initializes state with hbv-physical-snow gs and kirchner k"))
                .def_readwrite("hps",&state::hps,"hbv-physical-snow state")
                .def_readwrite("kirchner",&state::kirchner,"kirchner state")
                ;

            typedef std::vector<state> PTHPSKStateVector;
            class_<PTHPSKStateVector,bases<>,std::shared_ptr<PTHPSKStateVector> >("PTHPSKStateVector")
                .def(vector_indexing_suite<PTHPSKStateVector>())
                ;
            class_<response>("PTHPSKResponse","This struct contains the responses of the methods used in the PTHPSK assembly")
                .def_readwrite("pt",&response::pt,"priestley_taylor response")
                .def_readwrite("hps",&response::hps,"hbv-physical-snow response")
                .def_readwrite("gm_melt_m3s",&response::gm_melt_m3s,"glacier melt response[m3s]")
                .def_readwrite("ae",&response::ae,"actual evapotranspiration response")
                .def_readwrite("kirchner",&response::kirchner,"kirchner response")
                .def_readwrite("total_discharge",&response::total_discharge,"total stack response")
                ;
        }

        static void
        collectors() {
            typedef shyft::core::pt_hps_k::all_response_collector PTHPSKAllCollector;
            class_<PTHPSKAllCollector>("PTHPSKAllCollector", "collect all cell response from a run")
                .def_readonly("destination_area",&PTHPSKAllCollector::destination_area,"a copy of cell area [m2]")
                .def_readonly("avg_discharge",&PTHPSKAllCollector::avg_discharge,"Kirchner Discharge given in [m3/s] for the timestep")
                .def_readonly("hps_sca",&PTHPSKAllCollector::hps_sca," hbv snow covered area fraction, sca.. 0..1 - at the end of timestep (state)")
                .def_readonly("hps_swe",&PTHPSKAllCollector::hps_swe,"hbv snow swe, [mm] over the cell sca.. area, - at the end of timestep")
                .def_readonly("hps_outflow",&PTHPSKAllCollector::hps_outflow," hbv snow output [m^3/s] for the timestep")
                .def_readonly("glacier_melt",&PTHPSKAllCollector::glacier_melt," glacier melt (outflow) [m3/s] for the timestep")
                .def_readonly("ae_output",&PTHPSKAllCollector::ae_output,"actual evap mm/h")
                .def_readonly("pe_output",&PTHPSKAllCollector::pe_output,"pot evap mm/h")
                .def_readonly("end_reponse",&PTHPSKAllCollector::end_reponse,"end_response, at the end of collected")
            ;

            typedef shyft::core::pt_hps_k::discharge_collector PTHPSKDischargeCollector;
            class_<PTHPSKDischargeCollector>("PTHPSKDischargeCollector", "collect all cell response from a run")
                .def_readonly("destination_area",&PTHPSKDischargeCollector::destination_area,"a copy of cell area [m2]")
                .def_readonly("avg_discharge",&PTHPSKDischargeCollector::avg_discharge,"Kirchner Discharge given in [m^3/s] for the timestep")
                .def_readonly("hps_sca",&PTHPSKDischargeCollector::hps_sca," hbv snow covered area fraction, sca.. 0..1 - at the end of timestep (state)")
                .def_readonly("hps_swe",&PTHPSKDischargeCollector::hps_swe,"hbv snow swe, [mm] over the cell sca.. area, - at the end of timestep")
                .def_readonly("end_reponse",&PTHPSKDischargeCollector::end_response,"end_response, at the end of collected")
                .def_readwrite("collect_snow",&PTHPSKDischargeCollector::collect_snow,"controls collection of snow routine")
                ;
            typedef shyft::core::pt_hps_k::null_collector PTHPSKNullCollector;
            class_<PTHPSKNullCollector>("PTHPSKNullCollector","collector that does not collect anything, useful during calibration to minimize memory&maximize speed")
                ;

            typedef shyft::core::pt_hps_k::state_collector PTHPSKStateCollector;
            class_<PTHPSKStateCollector>("PTHPSKStateCollector","collects state, if collect_state flag is set to true")
                .def_readwrite("collect_state",&PTHPSKStateCollector::collect_state,"if true, collect state, otherwise ignore (and the state of time-series are undefined/zero)")
                .def_readonly("kirchner_discharge",&PTHPSKStateCollector::kirchner_discharge,"Kirchner state instant Discharge given in m^3/s")
                .def_readonly("hps_swe",&PTHPSKStateCollector::hps_swe,"")
                .def_readonly("hps_sca",&PTHPSKStateCollector::hps_sca,"")
				.def_readonly("sp", &PTHPSKStateCollector::sp, "")
				.def_readonly("sw", &PTHPSKStateCollector::sw, "")
				.def_readonly("albedo", &PTHPSKStateCollector::albedo, "")
				.def_readonly("iso_pot_energy", &PTHPSKStateCollector::iso_pot_energy, "")
				
            ;

        }

        static void
        cells() {
              typedef shyft::core::cell<parameter, environment_t, state, state_collector, all_response_collector> PTHPSKCellAll;
              typedef shyft::core::cell<parameter, environment_t, state, null_collector, discharge_collector> PTHPSKCellOpt;
              expose::cell<PTHPSKCellAll>("PTHPSKCellAll","tbd: PTHPSKCellAll doc");
              expose::cell<PTHPSKCellOpt>("PTHPSKCellOpt","tbd: PTHPSKCellOpt doc");
              expose::statistics::hbv_physical_snow<PTHPSKCellAll>("PTHPSKCell");//it only gives meaning to expose the *All collect cell-type
              expose::statistics::actual_evapotranspiration<PTHPSKCellAll>("PTHPSKCell");
              expose::statistics::priestley_taylor<PTHPSKCellAll>("PTHPSKCell");
              expose::statistics::kirchner<PTHPSKCellAll>("PTHPSKCell");
              expose::cell_state_etc<PTHPSKCellAll>("PTHPSK");// just one expose of state
        }

        static void
        models() {
            typedef shyft::core::region_model<pt_hps_k::cell_discharge_response_t, shyft::api::a_region_environment> PTHPSKOptModel;
            typedef shyft::core::region_model<pt_hps_k::cell_complete_response_t, shyft::api::a_region_environment> PTHPSKModel;
            expose::model<PTHPSKModel>("PTHPSKModel","PTHPSK");
            expose::model<PTHPSKOptModel>("PTHPSKOptModel","PTHPSK");
            def_clone_to_similar_model<PTHPSKModel, PTHPSKOptModel>("create_opt_model_clone");
            def_clone_to_similar_model<PTHPSKOptModel, PTHPSKModel>("create_full_model_clone");
        }

        static void
        model_calibrator() {
            expose::model_calibrator<shyft::core::region_model<pt_hps_k::cell_discharge_response_t, shyft::api::a_region_environment>>("PTHPSKOptimizer");
        }
    }
}

BOOST_PYTHON_MODULE(_pt_hps_k)
{

    boost::python::scope().attr("__doc__")="SHyFT python api for the pt_hps_k model";
    boost::python::def("version", version);
	boost::python::docstring_options doc_options(true, true, false);// all except c++ signatures
    expose::pt_hps_k::parameter_state_response();
    expose::pt_hps_k::cells();
    expose::pt_hps_k::models();
    expose::pt_hps_k::collectors();
    expose::pt_hps_k::model_calibrator();
}

