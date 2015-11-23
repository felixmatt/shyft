%module(package="shyft.api") pt_hs_k

#define SWIG_FILE_WITH_INIT
%begin %{ // gcc win-compile needs this to avoid problems in cmath, fix: to include first
#include <cmath>
%}
%header %{

#define SWIG_FILE_WITH_INIT
#define SWIG_PYTHON_EXTRA_NATIVE_CONTAINERS
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "api/api.h"
#include "api/pt_hs_k.h"
#include "core/pt_hs_k.h"

%}

%include "numpy.i"
%include <windows.i>
%include <std_string.i>
%include <std_map.i>
%include <std_vector.i>
%include <exception.i>
#define SWIG_SHARED_PTR_NAMESPACE std
%include <shared_ptr.i>

// Add type information from api module
%import "__init__.i"

%shared_ptr(std::vector<shyft::core::pt_hs_k::cell_discharge_response_t>)
%shared_ptr(std::vector<shyft::core::pt_hs_k::cell_complete_response_t>)
%shared_ptr(std::vector<shyft::core::pt_hs_k::state>)
%shared_ptr(shyft::core::pt_hs_k::parameter)
    %shared_ptr(std::vector< shyft::core::cell<shyft::core::pt_hs_k::parameter, shyft::core::environment_t, shyft::core::pt_hs_k::state, shyft::core::pt_hs_k::state_collector, shyft::core::pt_hs_k::all_response_collector> > )
    %shared_ptr(std::vector< shyft::core::cell<shyft::core::pt_hs_k::parameter, shyft::core::environment_t, shyft::core::pt_hs_k::state, shyft::core::pt_hs_k::null_collector, shyft::core::pt_hs_k::discharge_collector> > )

    // exceptions need to go here:
    %exception {
      try {
        $action
      } catch (const std::runtime_error& e) {
        SWIG_exception(SWIG_RuntimeError, e.what());
      } catch ( const std::invalid_argument& e){
        SWIG_exception(SWIG_RuntimeError, e.what());
      } catch (const std::exception& e) {
        SWIG_exception(SWIG_RuntimeError, e.what());
      } catch (...) {
        SWIG_exception(SWIG_UnknownError,"c++ unknown exception");
      }
    }

%rename(PTHSKResponse)    shyft::core::pt_hs_k::response;
%rename(PTHSKState)       shyft::core::pt_hs_k::state;
%rename(PTHSKParameter)   shyft::core::pt_hs_k::parameter;
%include "core/pt_hs_k.h"

%rename(PTHSKDischargeCollector )  shyft::core::pt_hs_k::discharge_collector;
%rename(PTHSKNullCollector)        shyft::core::pt_hs_k::null_collector;
%rename(PTHSKStateCollector)       shyft::core::pt_hs_k::state_collector;
%rename(PTHSKAllCollector)         shyft::core::pt_hs_k::all_response_collector;
%include "core/pt_hs_k_cell_model.h"

%rename (PTHSKStateIo) shyft::api::pt_hs_k_state_io;
%include "pt_hs_k.h"

namespace shyft {
  namespace core {
    namespace pt_hs_k {
        %template(PTHSKCellAll)  cell<parameter, environment_t, state, state_collector, all_response_collector>;
        typedef cell<parameter, environment_t, state, state_collector, all_response_collector> PTHSKCellAll;
        %template(PTHSKCellOpt)     cell<parameter, environment_t, state, null_collector, discharge_collector>;
        typedef cell<parameter, environment_t, state, null_collector, discharge_collector> PTHSKCellOpt;
    }
  }
}

namespace shyft {
  namespace core {
    namespace model_calibration {
        %template(PTHSKOptimizer) optimizer<region_model<pt_hs_k::cell_discharge_response_t>, shyft::core::pt_hs_k::parameter, pts_t>;
    }
  }
}

namespace shyft {
  namespace core {
    %template(PTHSKOptModel) region_model<pt_hs_k::cell_discharge_response_t>;
    typedef region_model<pt_hs_k::cell_discharge_response_t> PTHSKOptModel;
    %template(PTHSKModel) region_model<pt_hs_k::cell_complete_response_t>;
    typedef region_model<pt_hs_k::cell_complete_response_t> PTHSKModel;
    %template(run_interpolation) PTHSKModel::run_interpolation<shyft::api::a_region_environment,shyft::core::interpolation_parameter>;
    %template(run_interpolation) PTHSKOptModel::run_interpolation<shyft::api::a_region_environment,shyft::core::interpolation_parameter>;
  }
}

namespace std {
   %template(PTHSKCellAllVector) vector< shyft::core::cell<shyft::core::pt_hs_k::parameter, shyft::core::environment_t, shyft::core::pt_hs_k::state, shyft::core::pt_hs_k::state_collector, shyft::core::pt_hs_k::all_response_collector> >;
    %template(PTHSKCellOptVector) vector< shyft::core::cell<shyft::core::pt_hs_k::parameter, shyft::core::environment_t, shyft::core::pt_hs_k::state, shyft::core::pt_hs_k::null_collector, shyft::core::pt_hs_k::discharge_collector> >;
    %template(PTHSKStateVector) vector<shyft::core::pt_hs_k::state>;
    %template(PTHSKParameterMap) map<size_t, shyft::core::pt_hs_k::parameter>;
}

namespace shyft {
  namespace api {
    %template(PTHSKCellAllStatistics) basic_cell_statistics<shyft::core::cell<shyft::core::pt_hs_k::parameter, shyft::core::environment_t, shyft::core::pt_hs_k::state, shyft::core::pt_hs_k::state_collector, shyft::core::pt_hs_k::all_response_collector>>;
      %template(PTHSKCellHBVSnowStateStatistics) hbv_snow_cell_state_statistics<shyft::core::cell<shyft::core::pt_hs_k::parameter, shyft::core::environment_t, shyft::core::pt_hs_k::state, shyft::core::pt_hs_k::state_collector, shyft::core::pt_hs_k::all_response_collector>>;
      %template(PTHSKCellHBVSnowResponseStatistics) hbv_cell_response_statistics<shyft::core::cell<shyft::core::pt_hs_k::parameter, shyft::core::environment_t, shyft::core::pt_hs_k::state, shyft::core::pt_hs_k::state_collector, shyft::core::pt_hs_k::all_response_collector>>;
      %template(PTHSKCellPriestleyTaylorResponseStatistics) priestley_taylor_cell_response_statistics<shyft::core::cell<shyft::core::pt_hs_k::parameter, shyft::core::environment_t, shyft::core::pt_hs_k::state, shyft::core::pt_hs_k::state_collector, shyft::core::pt_hs_k::all_response_collector>>;
      %template(PTHSKCellActualEvapotranspirationResponseStatistics) actual_evapotranspiration_cell_response_statistics<shyft::core::cell<shyft::core::pt_hs_k::parameter, shyft::core::environment_t, shyft::core::pt_hs_k::state, shyft::core::pt_hs_k::state_collector, shyft::core::pt_hs_k::all_response_collector>>;
        %template(PTHSKCellKirchnerStateStatistics) kirchner_cell_state_statistics<shyft::core::cell<shyft::core::pt_hs_k::parameter, shyft::core::environment_t, shyft::core::pt_hs_k::state, shyft::core::pt_hs_k::state_collector, shyft::core::pt_hs_k::all_response_collector>>;
      %template(PTHSKCellOptStatistics) basic_cell_statistics<shyft::core::cell<shyft::core::pt_hs_k::parameter, shyft::core::environment_t, shyft::core::pt_hs_k::state, shyft::core::pt_hs_k::null_collector, shyft::core::pt_hs_k::discharge_collector>>;
  }
}
%pythoncode %{
PTHSKModel.cell_t = PTHSKCellAll
PTHSKParameter.map_t=PTHSKParameterMap
PTHSKModel.parameter_t = PTHSKParameter
PTHSKModel.state_t = PTHSKState
PTHSKModel.statistics = property(lambda self: PTHSKCellAllStatistics(self.get_cells()))
PTHSKModel.hbv_snow_state = property(lambda self: PTHSKCellHBVSnowStateStatistics(self.get_cells()))
PTHSKModel.hbv_snow_response = property(lambda self: PTHSKCellHBVSnowResponseStatistics(self.get_cells()))
PTHSKModel.priestley_taylor_response = property(lambda self: PTHSKCellPriestleyTaylorResponseStatistics(self.get_cells()))
PTHSKModel.actual_evaptranspiration_response=property(lambda self: PTHSKCellActualEvapotranspirationResponseStatistics(self.get_cells()))
PTHSKModel.kirchner_state = property(lambda self: PTHSKCellKirchnerStateStatistics(self.get_cells()))
PTHSKOptModel.cell_t = PTHSKCellOpt
PTHSKOptModel.parameter_t = PTHSKParameter
PTHSKOptModel.state_t = PTHSKState
PTHSKOptModel.statistics = property(lambda self:PTHSKCellOptStatistics(self.get_cells()))
PTHSKCellAll.vector_t = PTHSKCellAllVector
PTHSKCellOpt.vector_t = PTHSKCellOptVector
PTHSKState.vector_t = PTHSKStateVector
PTHSKState.serializer_t = PTHSKStateIo

%}




%init %{
    import_array();
%}
