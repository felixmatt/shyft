#pragma once
#include "expose_statistics.h"

namespace expose {
    using namespace boost::python;
    using namespace std;

    template <class StateIo,class state>
    static void state_io(const char *state_io_name) {
        std::string (StateIo::*to_string1)(const state& ) const = &StateIo::to_string;
        std::string (StateIo::*to_string2)(const std::vector<state>& ) const = &StateIo::to_string;
        class_<StateIo>(state_io_name)
         .def("from_string",&StateIo::from_string,args("str","state"), "returns true if succeeded convertion string into state")
         .def("to_string",to_string1,args("state"),"convert a state into readable string")
         .def("to_string",to_string2,args("state_vector"),"convert a vector of state to a string" )
         .def("vector_from_string",&StateIo::vector_from_string,args("s"),"given string s, convert it to a state vector")
         ;
    }

    template <class T>
    static void cell(const char *cell_name,const char* cell_doc) {
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
      expose::statistics::basic_cell<T>(cell_name);//common for all type of cells, so expose it here
    }


    template <class M>
    static void model(const char *model_name,const char *model_doc) {
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
}
