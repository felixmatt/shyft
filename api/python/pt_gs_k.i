%module(package="shyft.api") pt_gs_k

#define SWIG_FILE_WITH_INIT

%header %{

#define SWIG_FILE_WITH_INIT
#define SWIG_PYTHON_EXTRA_NATIVE_CONTAINERS
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "api/api.h"
#include "api/pt_gs_k.h"
#include "core/pt_gs_k.h"

%}

%include "numpy.i"
%include <windows.i>
%include <std_string.i>
%include <std_vector.i>
%include <exception.i>
#define SWIG_SHARED_PTR_NAMESPACE std
%include <shared_ptr.i>

// Add type information from api module
%import "__init__.i"

%shared_ptr(std::vector<shyft::core::pt_gs_k::cell_discharge_response_t>)
%shared_ptr(std::vector<shyft::core::pt_gs_k::cell_complete_response_t>)
%shared_ptr(std::vector<shyft::core::pt_gs_k::state>)
%shared_ptr(shyft::core::pt_gs_k::parameter)
%shared_ptr(std::vector< shyft::core::cell<shyft::core::pt_gs_k::parameter, shyft::core::environment_t, shyft::core::pt_gs_k::state, shyft::core::pt_gs_k::state_collector, shyft::core::pt_gs_k::all_response_collector> > )
%shared_ptr(std::vector< shyft::core::cell<shyft::core::pt_gs_k::parameter, shyft::core::environment_t, shyft::core::pt_gs_k::state, shyft::core::pt_gs_k::null_collector, shyft::core::pt_gs_k::discharge_collector> > )

%rename(PTGSKResponse)    shyft::core::pt_gs_k::response;
%rename(PTGSKState)       shyft::core::pt_gs_k::state;
%rename(PTGSKParameter)   shyft::core::pt_gs_k::parameter;
%include "core/pt_gs_k.h"

%rename(PTGSKDischargeCollector )  shyft::core::pt_gs_k::discharge_collector;
%rename(PTGSKNullCollector)        shyft::core::pt_gs_k::null_collector;
%rename(PTGSKStateCollector)       shyft::core::pt_gs_k::state_collector;
%rename(PTGSKAllCollector)         shyft::core::pt_gs_k::all_response_collector;
%include "core/pt_gs_k_cell_model.h"

%rename (PTGSKStateIo) shyft::api::pt_gs_k_state_io;
%include "pt_gs_k.h"

namespace shyft {
  namespace core {
    namespace pt_gs_k {
        %template(PTGSKCellAll)  cell<parameter, environment_t, state, state_collector, all_response_collector>;
        typedef cell<parameter, environment_t, state, state_collector, all_response_collector> PTGSKCellAll;
        %template(PTGSKCellOpt)     cell<parameter, environment_t, state, null_collector, discharge_collector>;
        typedef cell<parameter, environment_t, state, null_collector, discharge_collector> PTGSKCellOpt;
    }
  }
}

namespace shyft {
  namespace core {
    namespace model_calibration {
        %template(PTGSKOptimizer) optimizer<region_model<pt_gs_k::cell_discharge_response_t>, shyft::core::pt_gs_k::parameter, pts_t>;
    }
  }
}

namespace shyft {
  namespace core {
    %template(PTGSKOptModel) region_model<pt_gs_k::cell_discharge_response_t>;
    typedef region_model<pt_gs_k::cell_discharge_response_t> PTGSKOptModel;
    %template(PTGSKModel) region_model<pt_gs_k::cell_complete_response_t>;
    typedef region_model<pt_gs_k::cell_complete_response_t> PTGSKModel;
    %template(run_interpolation) PTGSKModel::run_interpolation<shyft::api::a_region_environment,shyft::core::interpolation_parameter>;
    %template(run_interpolation) PTGSKOptModel::run_interpolation<shyft::api::a_region_environment,shyft::core::interpolation_parameter>;
  }
}

namespace std {
   %template(PTGSKCellAllVector) vector< shyft::core::cell<shyft::core::pt_gs_k::parameter, shyft::core::environment_t, shyft::core::pt_gs_k::state, shyft::core::pt_gs_k::state_collector, shyft::core::pt_gs_k::all_response_collector> >;
    %template(PTGSKCellOptVector) vector< shyft::core::cell<shyft::core::pt_gs_k::parameter, shyft::core::environment_t, shyft::core::pt_gs_k::state, shyft::core::pt_gs_k::null_collector, shyft::core::pt_gs_k::discharge_collector> >;
    %template(PTGSKStateVector) vector<shyft::core::pt_gs_k::state>;
}

namespace shyft {
  namespace api {
    %template(PTGSKCellAllStatistics) basic_cell_statistics<shyft::core::cell<shyft::core::pt_gs_k::parameter, shyft::core::environment_t, shyft::core::pt_gs_k::state, shyft::core::pt_gs_k::state_collector, shyft::core::pt_gs_k::all_response_collector>>;
      %template(PTGSKCellGammaSnowStateStatistics) gamma_snow_cell_state_statistics<shyft::core::cell<shyft::core::pt_gs_k::parameter, shyft::core::environment_t, shyft::core::pt_gs_k::state, shyft::core::pt_gs_k::state_collector, shyft::core::pt_gs_k::all_response_collector>>;
      %template(PTGSKCellGammaSnowResponseStatistics) gamma_snow_cell_response_statistics<shyft::core::cell<shyft::core::pt_gs_k::parameter, shyft::core::environment_t, shyft::core::pt_gs_k::state, shyft::core::pt_gs_k::state_collector, shyft::core::pt_gs_k::all_response_collector>>;
      %template(PTGSKCellPriestleyTaylorResponseStatistics) priestley_taylor_cell_response_statistics<shyft::core::cell<shyft::core::pt_gs_k::parameter, shyft::core::environment_t, shyft::core::pt_gs_k::state, shyft::core::pt_gs_k::state_collector, shyft::core::pt_gs_k::all_response_collector>>;
      %template(PTGSKCellActualEvapotranspirationResponseStatistics) actual_evapotranspiration_cell_response_statistics<shyft::core::cell<shyft::core::pt_gs_k::parameter, shyft::core::environment_t, shyft::core::pt_gs_k::state, shyft::core::pt_gs_k::state_collector, shyft::core::pt_gs_k::all_response_collector>>;
        %template(PTGSKCellKirchnerStateStatistics) kirchner_cell_state_statistics<shyft::core::cell<shyft::core::pt_gs_k::parameter, shyft::core::environment_t, shyft::core::pt_gs_k::state, shyft::core::pt_gs_k::state_collector, shyft::core::pt_gs_k::all_response_collector>>;
      %template(PTGSKCellOptStatistics) basic_cell_statistics<shyft::core::cell<shyft::core::pt_gs_k::parameter, shyft::core::environment_t, shyft::core::pt_gs_k::state, shyft::core::pt_gs_k::null_collector, shyft::core::pt_gs_k::discharge_collector>>;
  }
}
%pythoncode %{
PTGSKModel.cell_t = PTGSKCellAll
PTGSKModel.statistics = property(lambda self: PTGSKCellAllStatistics(self.get_cells()))
PTGSKModel.gamma_snow_state = property(lambda self: PTGSKCellGammaSnowStateStatistics(self.get_cells()))
PTGSKModel.gamma_snow_response = property(lambda self: PTGSKCellGammaSnowResponseStatistics(self.get_cells()))
PTGSKModel.priestley_taylor_response = property(lambda self: PTGSKCellPriestleyTaylorResponseStatistics(self.get_cells()))
PTGSKModel.actual_evaptranspiration_response=property(lambda self: PTGSKCellActualEvapotranspirationResponseStatistics(self.get_cells()))
PTGSKModel.kirchner_state = property(lambda self: PTGSKCellKirchnerStateStatistics(self.get_cells()))
PTGSKOptModel.cell_t = PTGSKCellOpt
PTGSKOptModel.statistics = property(lambda self:PTGSKCellOptStatistics(self.get_cells()))
PTGSKCellAll.vector_t = PTGSKCellAllVector
PTGSKCellOpt.vector_t = PTGSKCellOptVector
%}

%exception {
  try {
    $action
  } catch (const std::runtime_error& e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  } catch (const std::exception& e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  }
}


%init %{
    import_array();
%}
