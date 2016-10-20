#pragma once
#include "expose_statistics.h"
#include "api/api.h"
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
    static vector<double> geo_cell_data_vector(shared_ptr<vector<T>> cell_vector) {
        vector<double> r; r.reserve(shyft::api::geo_cell_data_io::size()*cell_vector->size());//Assume approx 200 chars pr. cell
        for(const auto& cell:*cell_vector)
            shyft::api::geo_cell_data_io::push_to_vector(r,cell.geo);
        return move(r);
    }
    template <class T>
    static vector<T> create_from_geo_cell_data_vector(const vector<double>& s) {
        if(s.size()==0 || s.size()% shyft::api::geo_cell_data_io::size())
            throw invalid_argument("create_from_geo_cell_data_vector: size of vector of double must be multiple of 11");
        vector<T> r; r.reserve(s.size()/shyft::api::geo_cell_data_io::size());// assume this is ok size for now
        for(size_t i=0;i<s.size();i+=shyft::api::geo_cell_data_io::size()) {
            T cell;
            cell.geo=shyft::api::geo_cell_data_io::from_raw_vector(s.data()+i);
            r.push_back(cell);
        }
        return move(r);
    }


    template <class T>
    static void cell(const char *cell_name,const char* cell_doc) {
      class_<T>(cell_name,cell_doc)
        .def_readwrite("geo",&T::geo,"geo_cell_data information for the cell")
        .add_property("parameter",&T::get_parameter,&T::set_parameter,"reference to parameter for this cell, typically shared for a catchment")
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
        .def("geo_cell_data_vector", geo_cell_data_vector<T>,
             "returns a persistable DoubleVector representation of of geo_cell_data for all cells.\n"
             "that object can in turn be used to construct a <Cell>Vector of any cell type\n"
             "using the <Cell>Vector.create_from_geo_cell_data_vector")
             .staticmethod("geo_cell_data_vector")
        .def("create_from_geo_cell_data_vector",create_from_geo_cell_data_vector<T>,
             "create a cell-vector filling in the geo_cell_data records as given by the DoubleVector.\n"
             "This function works together with the geo_cell_data_vector static method\n"
             "that provides a correctly formatted persistable vector\n"
             "Notice that the context and usage of these two functions is related\n"
             "to python orchestration and repository data-caching\n")
             .staticmethod("create_from_geo_cell_data_vector")

        ;
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
        auto run_interpolation_f= &M::template run_interpolation<shyft::api::a_region_environment>;
		auto interpolate_f = &M::template interpolate<shyft::api::a_region_environment>;
        class_<M>(model_name,m_doc,no_init)
	     .def(init<const M&>(args("other_model"),"create a copy of the model"))
         .def(init< shared_ptr< vector<typename M::cell_t> >&, const typename M::parameter_t& >(args("cells","region_param"),"creates a model from cells and region model parameters") )
         .def(init< shared_ptr< vector<typename M::cell_t> >&, const typename M::parameter_t&, const map<int,typename M::parameter_t>& >(args("cells","region_param","catchment_parameters"),"creates a model from cells and region model parameters, and specified catchment parameters") )
         .def_readonly("time_axis",&M::time_axis,"the time_axis as set from run_interpolation, determines the time-axis for run")
		 .def_readonly("interpolation_parameter",&M::ip_parameter,"the most recently used interpolation parameter as passed to run_interpolation or interpolate routine")
         .def("number_of_catchments",&M::number_of_catchments,"compute and return number of catchments using info in cells.geo.catchment_id()")
		 .def("initialize_cell_environment",&M::initialize_cell_environment,boost::python::arg("time_axis"),
			 "Initializes the cell enviroment (cell.env.ts* )\n" 
			 "\n"
			 "The method initializes the cell environment, that keeps temperature, precipitation etc\n"
			 "that is local to the cell.The initial values of these time - series is set to zero.\n"
			 "The region-model time-axis is set to the supplied time-axis, so that\n"
			 "the any calculation steps will use the supplied time-axis.\n"
			 "This call is needed once prior to call to the .interpolate() or .run_cells() methods\n"
			 "\n"
			 "The call ensures that all cells.env ts are reset to zero, with a time-axis and\n"
			 " value-vectors according to the supplied time-axis.\n"
			 " Also note that the region-model.time_axis is set to the supplied time-axis.\n"
			 "\n"
			 "Parameters\n"
			 "----------\n"
			 " time_axis: Timeaxis\n"
			 "    specifies the time-axis for the region-model, and thus the cells\n"
			 "Returns\n"
			 "-------\n"
			 "Nothing\n"
		 )
		 .def("interpolate", interpolate_f, args("interpolation_parameter", "env"),
				"do interpolation interpolates region_environment temp,precip,rad.. point sources\n"
				"to a value representative for the cell.mid_point().\n"
				"\n"
				"note: initialize_cell_environment should be called once prior to this function\n"
				"\n"
				"Only supplied vectors of temp, precip etc. are interpolated, thus\n"
				"the user of the class can choose to put in place distributed series in stead.\n"
				"\n"
				"Parameters\n"
				"----------\n"
				" interpolation_parameter : InterpolationParameter\n"
				"     contains wanted parameters for the interpolation\n"
				" env: RegionEnvironemnt\n"
				"     contains the ref: region_environment type\n"
		 )
         .def("run_cells",&M::run_cells,(boost::python::arg("thread_cell_count")=0),"run_cells calculations over specified time_axis, require that run_interpolation is done first")
         .def("run_interpolation",run_interpolation_f,args("interpolation_parameter","time_axis","env"),
                    "run_interpolation interpolates region_environment temp,precip,rad.. point sources\n"
                    "to a value representative for the cell.mid_point().\n"
                    "\n"
                    "note: This function is equivalent to\n"
					"    self.initialize_cell_environment(time_axis)\n"
					"    self.interpolate(interpolation_parameter,env)\n"
				    "Parameters\n"
					"----------\n"
					" interpolation_parameter: InterpolationParameter\n"
					"   contains wanted parameters for the interpolation\n"
					" time_axis : Timeaxis\n"
					"    should be equal to the ref: timeaxis the ref: region_model is prepared running for.\n"
					" env : RegionEnvironment\n"
					"    contains the ref: region_environment type\n"
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
        .add_property("cells",&M::get_cells,"cells of the model")
         ;
    }
    template<class RegionModel>
    static void
    model_calibrator(const char *optimizer_name) {

        typedef typename RegionModel::parameter_t parameter_t;
        typedef shyft::core::pts_t pts_t;
        typedef shyft::core::model_calibration::optimizer<RegionModel, parameter_t, pts_t> Optimizer;
        typedef typename Optimizer::target_specification_t target_specification_t;

        class_<Optimizer>(optimizer_name,
            "The optimizer for parameters in a ref: shyft::core::region_model\n"
            "It provides needed functionality to orchestrate a search for the optimal parameters so that the goal function\n"
            "specified by the target_specifications are minimized.\n"
            "The user can specify which parameters (model specific) to optimize, giving range min..max for each of the\n"
            "parameters. Only parameters with min != max are used, thus minimizing the parameter space.\n"
            "\n"
            "Target specification ref: target_specification allows a lot of flexibility when it comes to what\n"
            "goes into the ref: nash_sutcliffe goal function.\n"
            "\n"
            "The search for optimium starts with the current parameter-set, the current start state, over the specified model time-axis.\n"
            "After a run, the goal function is calculated and returned back to the minbobyqa algorithm that continue searching for the minimum\n"
            "value until tolerances/iterations area reached.\n"
            ,no_init
        )
        .def(init< RegionModel&,
                   const std::vector<target_specification_t>&,
                   const std::vector<double>&,
                   const std::vector<double>& >(
                   args("model","targets","p_min","p_max"),
                    "construct an opt model for ptgsk, use p_min=p_max to disable optimization for a parameter\n"
                    "param model reference to the model to be optimized, the model should be initialized, i.e. the interpolation step done.\n"
                    "param vector<target_specification_t> specifies how to calculate the goal-function, \ref shyft::core::model_calibration::target_specification\n"
                    "param p_min minimum values for the parameters to be optimized\n"
                    "param p_max maximum values for the parameters to be  optimized\n"
                  )
        )
        .def("get_initial_state",&Optimizer::get_initial_state,args("i"),"get a copy of the i'th cells initial state")
        .def("optimize",&Optimizer::optimize,args("p","max_n_evaluations","tr_start","tr_stop"),
                "Call to optimize model, starting with p parameter set, using p_min..p_max as boundaries.\n"
                "where p is the full parameter vector.\n"
                "the p_min,p_max specified in constructor is used to reduce the parameterspace for the optimizer\n"
                "down to a minimum number to facilitate fast run.\n"
                "param p contains the starting point for the parameters\n"
                "param max_n_evaluations stop after n calls of the objective functions, i.e. simulations.\n"
                "param tr_start is the trust region start , default 0.1, ref bobyqa\n"
                "param tr_stop is the trust region stop, default 1e-5, ref bobyqa\n"
                "return the optimized parameter vector\n"
        )
        .def("optimize_dream",&Optimizer::optimize_dream,args("p","max_n_evaluations"),
                "Call to optimize model, using DREAM alg., find p, using p_min..p_max as boundaries.\n"
                "where p is the full parameter vector.\n"
                "the p_min,p_max specified in constructor is used to reduce the parameterspace for the optimizer\n"
                "down to a minimum number to facilitate fast run.\n"
                "param p is used as start point (not really, DREAM use random, but we should be able to pass u and q....\n"
                "param max_n_evaluations stop after n calls of the objective functions, i.e. simulations.\n"
                "return the optimized parameter vector\n"
        )
        .def("optimize_sceua",&Optimizer::optimize_sceua,args("p","max_n_evaluations","x_eps","y_eps"),
                "Call to optimize model, using SCE UA, using p as startpoint, find p, using p_min..p_max as boundaries.\n"
                "where p is the full parameter vector.\n"
                "the p_min,p_max specified in constructor is used to reduce the parameter-space for the optimizer\n"
                "down to a minimum number to facilitate fast run.\n"
                "param p is used as start point and is updated with the found optimal points\n"
                "param max_n_evaluations stop after n calls of the objective functions, i.e. simulations.\n"
                "param x_eps is stop condition when all changes in x's are within this range\n"
                "param y_eps is stop condition, and search is stopped when goal function does not improve anymore within this range\n"
                "return the optimized parameter vector\n"
        )
        .def("reset_states",&Optimizer::reset_states,"reset the state of the model to the initial state before starting the run/optimize")
        .def("set_parameter_ranges",&Optimizer::set_parameter_ranges,args("p_min","p_max"),"set the parameter ranges, set min=max=wanted parameter value for those not subject to change during optimization")
        .def("set_verbose_level",&Optimizer::set_verbose_level,args("level"),"set verbose level on stdout during calibration,0 is silent,1 is more etc.")
        .def("calculate_goal_function",&Optimizer::calculate_goal_function,args("full_vector_of_parameters"),
                "calculate the goal_function as used by minbobyqa,etc.,\n"
                "using the full set of  parameters vectors (as passed to optimize())\n"
                "and also ensures that the shyft state/cell/catchment result is consistent\n"
                "with the passed parameters passed\n"
                "param full_vector_of_parameters contains all parameters that will be applied to the run.\n"
                "returns the goal-function, weigthed nash_sutcliffe|Kling-Gupta sum \n"
        )
        ;

    }
}
