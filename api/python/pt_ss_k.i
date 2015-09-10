%module(package="shyft.api") pt_ss_k

#define SWIG_FILE_WITH_INIT

%header %{

#define SWIG_FILE_WITH_INIT
#define SWIG_PYTHON_EXTRA_NATIVE_CONTAINERS
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "api/api.h"
#include "core/pt_ss_k.h"

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

%shared_ptr(std::vector<shyft::core::pt_ss_k::cell_discharge_response_t>)
%shared_ptr(std::vector<shyft::core::pt_ss_k::cell_complete_response_t>)
%shared_ptr(std::vector<shyft::core::pt_ss_k::state>)
%shared_ptr(shyft::core::pt_ss_k::parameter)
%shared_ptr(std::vector< shyft::core::cell<shyft::core::pt_ss_k::parameter, shyft::core::environment_t, shyft::core::pt_ss_k::state, shyft::core::pt_ss_k::state_collector, shyft::core::pt_ss_k::all_response_collector> > )
%shared_ptr(std::vector< shyft::core::cell<shyft::core::pt_ss_k::parameter, shyft::core::environment_t, shyft::core::pt_ss_k::state, shyft::core::pt_ss_k::null_collector, shyft::core::pt_ss_k::discharge_collector> > )

%rename(PTSSKResponse)    shyft::core::pt_ss_k::response;
%rename(PTSSKState)       shyft::core::pt_ss_k::state;
%rename(PTSSKParameter)   shyft::core::pt_ss_k::parameter;
%include "core/pt_ss_k.h"

%rename(PTSSKDischargeCollector )  shyft::core::pt_ss_k::discharge_collector;
%rename(PTSSKNullCollector)        shyft::core::pt_ss_k::null_collector;
%rename(PTSSKStateCollector)       shyft::core::pt_ss_k::state_collector;
%rename(PTSSKAllCollector)         shyft::core::pt_ss_k::all_response_collector;
%include "core/pt_ss_k_cell_model.h"

namespace shyft {
  namespace core {
    namespace pt_ss_k {
        %template(PTSSKCellAll)  cell<parameter, environment_t, state, state_collector, all_response_collector>;
        typedef cell<parameter, environment_t, state, state_collector, all_response_collector> PTSSKCellAll;
        %template(PTSSKCellOpt)     cell<parameter, environment_t, state, null_collector, discharge_collector>;
        typedef cell<parameter, environment_t, state, null_collector, discharge_collector> PTSSKCellOpt;
    }
  }
}

namespace shyft {
  namespace core {
    namespace model_calibration {
        %template(PTSSKOptimizer) optimizer<region_model<pt_ss_k::cell_discharge_response_t>, shyft::core::pt_ss_k::parameter, pts_t>;
    }
  }
}

namespace shyft {
  namespace core {
    %template(PTSSKOptModel) region_model<pt_ss_k::cell_discharge_response_t>;
    typedef region_model<pt_ss_k::cell_discharge_response_t> PTSSKOptModel;
    %template(PTSSKModel) region_model<pt_ss_k::cell_complete_response_t>;
    typedef region_model<pt_ss_k::cell_complete_response_t> PTSSKModel;
    %template(run_interpolation) PTSSKModel::run_interpolation<shyft::api::a_region_environment,shyft::core::interpolation_parameter>;
    %template(run_interpolation) PTSSKOptModel::run_interpolation<shyft::api::a_region_environment,shyft::core::interpolation_parameter>;
  }
}

namespace std {
   %template(PTSSKCellAllVector) vector< shyft::core::cell<shyft::core::pt_ss_k::parameter, shyft::core::environment_t, shyft::core::pt_ss_k::state, shyft::core::pt_ss_k::state_collector, shyft::core::pt_ss_k::all_response_collector> >;
    %template(PTSSKCellOptVector) vector< shyft::core::cell<shyft::core::pt_ss_k::parameter, shyft::core::environment_t, shyft::core::pt_ss_k::state, shyft::core::pt_ss_k::null_collector, shyft::core::pt_ss_k::discharge_collector> >;
    %template(PTSSKStateVector) vector<shyft::core::pt_ss_k::state>;
}

namespace shyft {
  namespace api {
    %template(PTSSKCellAllStatistics) basic_cell_statistics<shyft::core::cell<shyft::core::pt_ss_k::parameter, shyft::core::environment_t, shyft::core::pt_ss_k::state, shyft::core::pt_ss_k::state_collector, shyft::core::pt_ss_k::all_response_collector>>;
      %template(PTSSKCellSkaugenStateStatistics) skaugen_cell_state_statistics<shyft::core::cell<shyft::core::pt_ss_k::parameter, shyft::core::environment_t, shyft::core::pt_ss_k::state, shyft::core::pt_ss_k::state_collector, shyft::core::pt_ss_k::all_response_collector>>;
      %template(PTSSKCellSkaugenResponseStatistics) skaugen_cell_response_statistics<shyft::core::cell<shyft::core::pt_ss_k::parameter, shyft::core::environment_t, shyft::core::pt_ss_k::state, shyft::core::pt_ss_k::state_collector, shyft::core::pt_ss_k::all_response_collector>>;
      %template(PTSSKCellPriestleyTaylorResponseStatistics) priestley_taylor_cell_response_statistics<shyft::core::cell<shyft::core::pt_ss_k::parameter, shyft::core::environment_t, shyft::core::pt_ss_k::state, shyft::core::pt_ss_k::state_collector, shyft::core::pt_ss_k::all_response_collector>>;
      %template(PTSSKCellActualEvapotranspirationResponseStatistics) actual_evapotranspiration_cell_response_statistics<shyft::core::cell<shyft::core::pt_ss_k::parameter, shyft::core::environment_t, shyft::core::pt_ss_k::state, shyft::core::pt_ss_k::state_collector, shyft::core::pt_ss_k::all_response_collector>>;
        %template(PTSSKCellKirchnerStateStatistics) kirchner_cell_state_statistics<shyft::core::cell<shyft::core::pt_ss_k::parameter, shyft::core::environment_t, shyft::core::pt_ss_k::state, shyft::core::pt_ss_k::state_collector, shyft::core::pt_ss_k::all_response_collector>>;
      %template(PTSSKCellOptStatistics) basic_cell_statistics<shyft::core::cell<shyft::core::pt_ss_k::parameter, shyft::core::environment_t, shyft::core::pt_ss_k::state, shyft::core::pt_ss_k::null_collector, shyft::core::pt_ss_k::discharge_collector>>;
  }
}
%pythoncode %{
PTSSKModel.cell_t = PTSSKCellAll
PTSSKModel.statistics = property(lambda self: PTSSKCellAllStatistics(self.get_cells()))
PTSSKModel.skaugen_state = property(lambda self: PTSSKCellSkaugenStateStatistics(self.get_cells()))
PTSSKModel.skaugen_response = property(lambda self: PTSSKCellSkaugenResponseStatistics(self.get_cells()))
PTSSKModel.priestley_taylor_response = property(lambda self: PTSSKCellPriestleyTaylorResponseStatistics(self.get_cells()))
PTSSKModel.actual_evaptranspiration_response=property(lambda self: PTSSKCellActualEvapotranspirationResponseStatistics(self.get_cells()))
PTSSKModel.kirchner_state = property(lambda self: PTSSKCellKirchnerStateStatistics(self.get_cells()))
PTSSKOptModel.cell_t = PTSSKCellOpt
PTSSKOptModel.statistics = property(lambda self:PTSSKCellOptStatistics(self.get_cells()))
PTSSKCellAll.vector_t = PTSSKCellAllVector
PTSSKCellOpt.vector_t = PTSSKCellOptVector
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
