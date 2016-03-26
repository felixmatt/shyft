
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/scope.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
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
#include "api/api.h"
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

    typedef std::map<int,parameter> PTGSKParameterMap;
    class_<PTGSKParameterMap>("PTGSKParameterMap","dict (int,parameter)  where the int is 0-based catchment_id")
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
        .def_readwrite("ae",&response::ae,"actual evapotranspiration response")
        .def_readwrite("kirchner",&response::kirchner,"kirchner response")
        .def_readwrite("total_discharge",&response::total_discharge,"total stack response")
        ;
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
namespace expose_statistics {
    typedef shyft::api::result_ts_t_ rts_;
    typedef std::vector<double> vd_;
    typedef const std::vector<int>& cids_;
    typedef size_t ix_;
    using namespace boost::python;

    template<class cell>
    static void kirchner(const char *cell_name) {
        char state_name[200];sprintf(state_name,"%sKirchnerStateStatistics",cell_name);
        typedef typename shyft::api::kirchner_cell_state_statistics<cell>    sc_stat;

        rts_ (sc_stat::*discharge_ts)(cids_) const = &sc_stat::discharge;
        vd_  (sc_stat::*discharge_vd)(cids_,ix_) const =&sc_stat::discharge;
        class_<sc_stat>(state_name,"Kirchner response statistics",no_init)
            .def(init<std::shared_ptr<std::vector<cell>> >(args("cells"),"construct Kirchner cell response statistics object"))
            .def("discharge",discharge_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
            .def("discharge",discharge_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
        ;
    }

    template <class cell>
    static void priestley_taylor(const char *cell_name) {
        char response_name[200];sprintf(response_name,"%sPriestleyTaylorResponseStatistics",cell_name);
        typedef typename shyft::api::priestley_taylor_cell_response_statistics<cell> rc_stat;

        rts_ (rc_stat::*output_ts)(cids_) const = &rc_stat::output;
        vd_  (rc_stat::*output_vd)(cids_,ix_) const =&rc_stat::output;
        class_<rc_stat>(response_name,"PriestleyTaylor response statistics",no_init)
            .def(init<std::shared_ptr<std::vector<cell>> >(args("cells"),"construct PriestleyTaylor cell response statistics object"))
            .def("output",output_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
            .def("output",output_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
        ;
    }

    template <class cell>
    static void actual_evapotranspiration(const char *cell_name) {
        char response_name[200];sprintf(response_name,"%sActualEvapotranspirationResponseStatistics",cell_name);
        typedef typename shyft::api::actual_evapotranspiration_cell_response_statistics<cell> rc_stat;

        rts_ (rc_stat::*output_ts)(cids_) const = &rc_stat::output;
        vd_  (rc_stat::*output_vd)(cids_,ix_) const =&rc_stat::output;
        class_<rc_stat>(response_name,"ActualEvapotranspiration response statistics",no_init)
            .def(init<std::shared_ptr<std::vector<cell>> >(args("cells"),"construct ActualEvapotranspiration cell response statistics object"))
            .def("output",output_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
            .def("output",output_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
        ;
    }

    template <class cell>
    static void gamma_snow(const char *cell_name) {
        char state_name[200];sprintf(state_name,"%sGammaSnowStateStatistics",cell_name);
        char response_name[200];sprintf(response_name,"%sGammaSnowResponseStatistics",cell_name);
        typedef typename shyft::api::gamma_snow_cell_state_statistics<cell>    sc_stat;
        typedef typename shyft::api::gamma_snow_cell_response_statistics<cell> rc_stat;

        rts_ (sc_stat::*albedo_ts)(cids_) const = &sc_stat::albedo;
        vd_  (sc_stat::*albedo_vd)(cids_,ix_) const =&sc_stat::albedo;

        rts_ (sc_stat::*lwc_ts)(cids_) const = &sc_stat::lwc;
        vd_  (sc_stat::*lwc_vd)(cids_,ix_) const =&sc_stat::lwc;

        rts_ (sc_stat::*surface_heat_ts)(cids_) const = &sc_stat::surface_heat;
        vd_  (sc_stat::*surface_heat_vd)(cids_,ix_) const =&sc_stat::surface_heat;

        rts_ (sc_stat::*alpha_ts)(cids_) const = &sc_stat::alpha;
        vd_  (sc_stat::*alpha_vd)(cids_,ix_) const =&sc_stat::alpha;

        rts_ (sc_stat::*sdc_melt_mean_ts)(cids_) const = &sc_stat::sdc_melt_mean;
        vd_  (sc_stat::*sdc_melt_mean_vd)(cids_,ix_) const =&sc_stat::sdc_melt_mean;

        rts_ (sc_stat::*acc_melt_ts)(cids_) const = &sc_stat::acc_melt;
        vd_  (sc_stat::*acc_melt_vd)(cids_,ix_) const =&sc_stat::acc_melt;

        rts_ (sc_stat::*iso_pot_energy_ts)(cids_) const = &sc_stat::iso_pot_energy;
        vd_  (sc_stat::*iso_pot_energy_vd)(cids_,ix_) const =&sc_stat::iso_pot_energy;

        rts_ (sc_stat::*temp_swe_ts)(cids_) const = &sc_stat::temp_swe;
        vd_  (sc_stat::*temp_swe_vd)(cids_,ix_) const =&sc_stat::temp_swe;

        class_<sc_stat>(state_name,"GammaSnow state statistics",no_init)
            .def(init<std::shared_ptr<std::vector<cell>> >(args("cells"),"construct GammaSnow cell state statistics object"))
            .def("albedo",albedo_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
            .def("albedo",albedo_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
            .def("lwc",lwc_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
            .def("lwc",lwc_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
            .def("surface_heat",surface_heat_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
            .def("surface_heat",surface_heat_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
            .def("alpha",alpha_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
            .def("alpha",alpha_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
            .def("sdc_melt_mean",sdc_melt_mean_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
            .def("sdc_melt_mean",sdc_melt_mean_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
            .def("acc_melt",acc_melt_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
            .def("acc_melt",acc_melt_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
            .def("iso_pot_energy",iso_pot_energy_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
            .def("iso_pot_energy",iso_pot_energy_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
            .def("temp_swe",temp_swe_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
            .def("temp_swe",temp_swe_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
        ;


        rts_ (rc_stat::*sca_ts)(cids_) const = &rc_stat::sca;
        vd_  (rc_stat::*sca_vd)(cids_,ix_) const =&rc_stat::sca;

        rts_ (rc_stat::*swe_ts)(cids_) const = &rc_stat::swe;
        vd_  (rc_stat::*swe_vd)(cids_,ix_) const =&rc_stat::swe;

        rts_ (rc_stat::*outflow_ts)(cids_) const = &rc_stat::outflow;
        vd_  (rc_stat::*outflow_vd)(cids_,ix_) const =&rc_stat::outflow;

        class_<rc_stat>(response_name,"GammaSnow response statistics",no_init)
            .def(init<std::shared_ptr<std::vector<cell>> >(args("cells"),"construct GammaSnow cell response statistics object"))
            .def("outflow",outflow_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
            .def("outflow",outflow_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
            .def("swe",swe_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
            .def("swe",swe_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
            .def("sca",sca_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
            .def("sca",sca_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")

            ;

    }

    template <class cell>
    static void basic_cell(const char *cell_name) {
        char base_name[200];sprintf(base_name,"%sStatistics",cell_name);
        typedef typename shyft::api::basic_cell_statistics<cell> bc_stat;

        rts_ (bc_stat::*discharge_ts)(cids_) const = &bc_stat::discharge;
        vd_  (bc_stat::*discharge_vd)(cids_,ix_) const =&bc_stat::discharge;

        rts_ (bc_stat::*temperature_ts)(cids_) const = &bc_stat::temperature;
        vd_  (bc_stat::*temperature_vd)(cids_,ix_) const =&bc_stat::temperature;

        rts_ (bc_stat::*radiation_ts)(cids_) const = &bc_stat::radiation;
        vd_  (bc_stat::*radiation_vd)(cids_,ix_) const =&bc_stat::radiation;

        rts_ (bc_stat::*wind_speed_ts)(cids_) const = &bc_stat::wind_speed;
        vd_  (bc_stat::*wind_speed_vd)(cids_,ix_) const =&bc_stat::wind_speed;

        rts_ (bc_stat::*rel_hum_ts)(cids_) const = &bc_stat::rel_hum;
        vd_  (bc_stat::*rel_hum_vd)(cids_,ix_) const =&bc_stat::rel_hum;

        rts_ (bc_stat::*precipitation_ts)(cids_) const = &bc_stat::precipitation;
        vd_  (bc_stat::*precipitation_vd)(cids_,ix_) const =&bc_stat::precipitation;



        class_<bc_stat>(base_name,"provides statistics for cell environment plus mandatory discharge",no_init)
            .def(init<std::shared_ptr<std::vector<cell>> >(args("cells"),"construct basic cell statistics object"))
            .def("discharge",discharge_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
            .def("discharge",discharge_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
            .def("temperature",temperature_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
            .def("temperature",temperature_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
            .def("precipitation",precipitation_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
            .def("precipitation",precipitation_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
            .def("radiation",radiation_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
            .def("radiation",radiation_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
            .def("wind_speed",wind_speed_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
            .def("wind_speed",wind_speed_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
            .def("rel_hum",rel_hum_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
            .def("rel_hum",rel_hum_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
            ;
    }
}
template <class T>
static void expose_cell(const char *cell_name,const char* cell_doc) {
  class_<T>(cell_name,cell_doc)
    .def_readwrite("geo",&T::geo,"geo_cell_data information for the cell")
    .def_readwrite("parameter",&T::parameter,"reference to parameter for this cell, typically shared for a catchment")
    .def_readwrite("env_ts",&T::env_ts,"environment time-series as projected to the cell")
    .def_readonly("sc",&T::sc,"state collector for the cell")
    .def_readonly("rc",&T::rc,"response collector for the cell")
    .def("set_parameter",&T::set_parameter,args("parameter"),"set the cell method stack parameters, typical operations at region_level, executed after the interpolation, before the run")
    .def("set_state_collection",&T::set_state_collection,args("on_or_off"),"collecting the state during run could be very useful to understand models")
    .def("set_snow_sca_swe_collection",&T::set_snow_sca_swe_collection,"collecting the snow sca and swe on for calibration scenario")
    .def("mid_point",&T::mid_point,"returns geo.mid_point()",return_internal_reference<>())
  ;
  char cv[200];sprintf(cv,"%sVector",cell_name);
  class_<vector<T>,bases<>,shared_ptr<vector<T>> > (cv,"vector of cells")
    .def(vector_indexing_suite<vector<T>>())
    ;
  register_ptr_to_python<std::shared_ptr<std::vector<T>> >();
  expose_statistics::basic_cell<T>(cell_name);//common for all type of cells, so expose it here
}

static void expose_pt_gs_k_cell() {
  typedef cell<parameter, environment_t, state, state_collector, all_response_collector> PTGSKCellAll;
  typedef cell<parameter, environment_t, state, null_collector, discharge_collector> PTGSKCellOpt;
  expose_cell<PTGSKCellAll>("PTGSKCellAll","tbd: PTGSKCellAll doc");
  expose_cell<PTGSKCellOpt>("PTGSKCellOpt","tbd: PTGSKCellOpt doc");
  expose_statistics::gamma_snow<PTGSKCellAll>("PTGSKCell");//it only gives meaning to expose the *All collect cell-type
  expose_statistics::actual_evapotranspiration<PTGSKCellAll>("PTGSKCell");
  expose_statistics::priestley_taylor<PTGSKCellAll>("PTGSKCell");
  expose_statistics::kirchner<PTGSKCellAll>("PTGSKCell");

}

template <class M>
static void expose_model(const char *model_name,const char *model_doc) {
    char m_doc[5000];
    sprintf(m_doc,
        " %s , a region_model is the calculation model for a region, where we can have\n"
        "one or more catchments.\n"
        "The role of the region_model is to describe region, so that we can run the\n"
        "region computational model efficiently for a number of type of cells, interpolation and\n"
        "catchment level algorihtms.\n"
        "\n"
        "The region model keeps a list of cells, of specified type \n"
            ,model_name);
    // NOTE: explicit expansion of the run_interpolate method is needed here, using this specific syntax
    auto run_interpolation_f= &M::template run_interpolation<shyft::api::a_region_environment,shyft::core::interpolation_parameter>;
    class_<M>(model_name,m_doc,no_init)
     .def(init< shared_ptr< vector<typename M::cell_t> >&, const typename M::parameter_t& >(args("cells","region_param"),"creates a model from cells and region model parameters") )
     .def(init< shared_ptr< vector<typename M::cell_t> >&, const typename M::parameter_t&, const map<int,typename M::parameter_t>& >(args("cells","region_param","catchment_parameters"),"creates a model from cells and region model parameters, and specified catchment parameters") )
     .def_readonly("time_axis",&M::time_axis,"the time_axis as set from run_interpolation, determines the time-axis for run")
     .def("number_of_catchments",&M::number_of_catchments,"compute and return number of catchments using info in cells.geo.catchment_id()")
     .def("run_cells",&M::number_of_catchments,"run_cells calculations over specified time_axis, require that run_interpolation is done first")
     .def("run_interpolation",run_interpolation_f,args("interpolation_parameter","time_axis","env"),
                "run_interpolation interpolates region_environment temp,precip,rad.. point sources\n"
                "to a value representative for the cell.mid_point().\n"
                "\n"
                "note: Prior to running all cell.env_ts.xxx are reset to zero, and have a length of time_axis.size().\n"
                "\n"
                "Only supplied vectors of temp, precip etc. are interpolated, thus\n"
                "the user of the class can choose to put in place distributed series in stead.\n"
                "\n"
                "param interpolation_parameter contains wanted parameters for the interpolation\n"
                "param time_axis should be equal to the ref: timeaxis the ref: region_model is prepared running for.\n"
                "param env contains the ref: region_environment type\n"
        )
     .def("set_region_parameter",&M::set_region_parameter,args("p"),
                "set the region parameter, apply it to all cells \n"
                "that do *not* have catchment specific parameters.\n")
     .def("get_region_parameter",&M::get_region_parameter,"provide access to current region parameter-set",return_internal_reference<>())
     .def("set_catchment_parameter",&M::set_catchment_parameter,args("catchment_id","p"),
                "creates/modifies a pr catchment override parameter\n"
                "param catchment_id the 0 based catchment_id that correlates to the cells catchment_id\n"
                "param a reference to the parameter that will be kept for those cells\n"
     )
     .def("remove_catchment_parameter",&M::remove_catchment_parameter,args("catchment_id"),"remove a catchment specific parameter override, if it exists.")
     .def("has_catchment_parameter",&M::has_catchment_parameter,args("catchment_id"),"returns true if there exist a specific parameter override for the specified 0-based catchment_id")
     .def("get_catchment_parameter",&M::get_catchment_parameter,args("catchment_id"),
                "return the parameter valid for specified catchment_id, or global parameter if not found.\n"
                "note Be aware that if you change the returned parameter, it will affect the related cells.\n"
                "param catchment_id 0 based catchment id as placed on each cell\n"
                "returns reference to the real parameter structure for the catchment_id if exists,\n"
                "otherwise the global parameters\n"
     ,return_internal_reference<>())
     .def("set_catchment_calculation_filter",&M::set_catchment_calculation_filter,args("catchment_id_list"),
                "set/reset the catchment based calculation filter. This affects what get simulate/calculated during\n"
                "the run command. Pass an empty list to reset/clear the filter (i.e. no filter).\n"
                "\n"
                "param catchment_id_list is a (zero-based) catchment id vector\n"
     )
     .def("is_calculated",&M::is_calculated,args("catchment_id"),"true if catchment id is calculated during runs, ref set_catchment_calculation_filter")
     .def("get_states",&M::get_states,args("end_states"),
                "collects current state from all the cells\n"
                "note that catchment filter can influence which states are calculated/updated.\n"
                "param end_states a reference to the vector<state_t> that are filled with cell state, in order of appearance.\n"
    )
    .def("set_states",&M::set_states,args("states"),
                "set current state for all the cells in the model.\n"
                "states is a vector<state_t> of all states, must match size/order of cells.\n"
                "note throws runtime-error if states.size is different from cells.size\n"
    )
    .def("set_state_collection",&M::set_state_collection,args("catchment_id","on_or_off"),
                "enable state collection for specified or all cells\n"
                "note that this only works if the underlying cell is configured to\n"
                "do state collection. This is typically not the  case for\n"
                "cell-types that are used during calibration/optimization\n"
    )
    .def("set_snow_sca_swe_collection",&M::set_snow_sca_swe_collection,args("catchment_id","on_or_off"),
                "enable/disable collection of snow sca|sca for calibration purposes\n"
                "param cachment_id to enable snow calibration for, -1 means turn on/off for all\n"
                "param on_or_off true|or false.\n"
                "note if the underlying cell do not support snow sca|swe collection, this \n"
    )
    .def("get_cells",&M::get_cells,"cells as shared_ptr<vector<cell_t>>")
    .def("size",&M::size,"return number of cells")
     ;
}

static void expose_pt_gs_k_model() {
    typedef region_model<pt_gs_k::cell_discharge_response_t> PTGSKOptModel;
    typedef region_model<pt_gs_k::cell_complete_response_t> PTGSKModel;
    expose_model<PTGSKModel>("PTGSKModel","PTGSK");
    expose_model<PTGSKOptModel>("PTGSKOptModel","PTGSK");

}


BOOST_PYTHON_MODULE(_pt_gs_k)
{

    scope().attr("__doc__")="SHyFT python api for the pt_gs_k model";
    def("version", version);
    expose_pt_gs_k();
    expose_state_io();
    expose_pt_gs_k_cell();
    expose_pt_gs_k_collectors();
    expose_pt_gs_k_model();
    //expose_statistics();
}
