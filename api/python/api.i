/// Python interface for the shyft::core and shyft::api
///
/// Hints to help you survive:
///  swig.org have excellent doc for c++ and swig, the template section is needed reading.
///  special things to absolutely remember
///
///  %template:
///   * remember to always use fully qualified names in template parameters(stay out of trouble with difficult )
///   * it takes forever(several minutes) to regenerate swig (its an issue we work on) 


%module api

%feature("autodoc", "2");
%feature("kwargs");


//
// Include SWIG mappings for libraries that we use
// (std etc)

#define SWIG_FILE_WITH_INIT

//
// Now, the %header section needs to contain enough includes so that the c++ compiler
// can compile the generated wrapper.
//
// note that this section is not parsed by swig, but a help for the wrapper to compile
// once successfully processed by swig
//


%header %{

    #define SWIG_FILE_WITH_INIT
    #define SWIG_PYTHON_EXTRA_NATIVE_CONTAINERS
    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

    #include "core/utctime_utilities.h"

    #include "api/api.h"
    #include "core/geo_point.h"
    #include "core/geo_cell_data.h"
    #include "core/timeseries.h"


    #include "core/priestley_taylor.h"
    #include "core/actual_evapotranspiration.h"
    #include "core/gamma_snow.h"
    #include "core/hbv_snow.h"
    #include "core/skaugen.h"
    #include "core/kirchner.h"
    #include "core/precipitation_correction.h"

    #include "core/inverse_distance.h"
    #include "core/bayesian_kriging.h"

    #include "core/pt_gs_k.h"

    //#include "core/pt_hs_k_cell_model.h"
    #include "core/pt_gs_k_cell_model.h"
    #include "core/pt_ss_k_cell_model.h"

    //#include "core/model_calibration.h"


%}

//-- in this section, include all standard mappings available for swig

    %include "numpy.i"
    %include <windows.i>
    %include <std_string.i>
    %include <std_vector.i>
    %include <exception.i>
    %include <std_shared_ptr.i>


//-- in this section, declare all the shared pointers needs to be declared here,
//   -including their derivates used. (swig warns when missing)
// as a rule we let large objects in/out to be shared pointers.
//


%shared_ptr(std::vector<shyft::core::pt_gs_k::cell_discharge_response_t>)
%shared_ptr(std::vector<shyft::core::pt_gs_k::cell_complete_response_t>)

%shared_ptr(std::vector<shyft::core::pt_ss_k::cell_discharge_response_t>)
%shared_ptr(std::vector<shyft::core::pt_ss_k::cell_complete_response_t>)

%shared_ptr(std::vector<shyft::api::TemperatureSource>)
%shared_ptr(std::vector<shyft::api::PrecipitationSource>)
%shared_ptr(std::vector<shyft::api::RadiationSource>)
%shared_ptr(std::vector<shyft::api::WindSpeedSource>)
%shared_ptr(std::vector<shyft::api::RelHumSource>)

%shared_ptr(std::vector<shyft::core::pt_gs_k::state_t>)
%shared_ptr(std::vector<shyft::core::pt_ss_k::state_t>)

%shared_ptr( shyft::api::ITimeSeriesOfPoints)
%shared_ptr( shyft::api::GenericTs<shyft::timeseries::point_timeseries<shyft::timeseries::timeaxis>>)
%shared_ptr( shyft::api::GenericTs<shyft::timeseries::point_timeseries<shyft::timeseries::point_timeaxis>>)
%shared_ptr( shyft::api::GenericTs<shyft::timeseries::function_timeseries<shyft::timeseries::timeaxis,shyft::timeseries::sin_fx>>)
//%shared_ptr( shyft::core::pts_t )
%shared_ptr( shyft::timeseries::point_timeseries<shyft::timeseries::timeaxis> )
%shared_ptr(shyft::core::pt_gs_k::parameter)
%shared_ptr(shyft::core::pt_ss_k::parameter)

%shared_ptr(std::vector< shyft::core::cell<shyft::core::pt_gs_k::parameter, shyft::core::environment_t, shyft::core::pt_gs_k::state_t, shyft::core::pt_gs_k::state_collector, shyft::core::pt_gs_k::all_response_collector> > )
%shared_ptr(std::vector< shyft::core::cell<shyft::core::pt_gs_k::parameter, shyft::core::environment_t, shyft::core::pt_gs_k::state_t, shyft::core::pt_gs_k::null_collector, shyft::core::pt_gs_k::discharge_collector> > )

%shared_ptr(std::vector< shyft::core::cell<shyft::core::pt_ss_k::parameter_t, shyft::core::environment_t, shyft::core::pt_ss_k::state_t, shyft::core::pt_ss_k::state_collector, shyft::core::pt_ss_k::all_response_collector> > )
%shared_ptr(std::vector< shyft::core::cell<shyft::core::pt_ss_k::parameter_t, shyft::core::environment_t, shyft::core::pt_ss_k::state_t, shyft::core::pt_ss_k::null_collector, shyft::core::pt_ss_k::discharge_collector> > )



// -- Now we let SWIG parse and interpret the types in enki_api.h
// swig will do its best to reflect all types/methods exposed there into the python wrapper.
// Since we have include swig-mapping for std types, plus our declared shared_ptr, this is a walk in the park :-)
//


///
// In python, we need CamelCase for class-names.
// the rename(..) fixes that for all class-names.
//

%rename("%(camelcase)s",%$isclass) "";


///
// Then rename/extend methods for individual classes that we want to fine-tune:
//
    %rename(UtcPeriod) shyft::core::utcperiod;

//-- in this section, include core and api header files with content that swig should translate
//   hint&troubleshooting: use #ifndef SWIG around sections that are not supported
//
    //
    //-- basic types and tools used
    //
    %include "core/utctime_utilities.h"
    %typedef shyft::core::utctime utctime;
    %typedef shyft::core::utctimespan utctimespan;
    %typedef shyft::core::utctimespan timespan;

    %include "core/geo_point.h"
    %include "core/geo_cell_data.h"
    %include "core/timeseries.h"


    ///-- methods/algorithms
    ///-- individual methods/algorithms goes here
    /// note that we need to rename each namespace member like parameter/state/response/calculator since
    /// they are not supported by SWIG.
    /// troubleshooting: SWIG struggle a lot with constructors in combination with namespace and 'equal' concepts like state/parameter..
    ///  workaround: just drop the constructors, of ifndef SWIG around.

    %rename(PriestleyTaylorParameter)  shyft::core::priestley_taylor::parameter;
    %rename(PriestleyTaylorResponse)   shyft::core::priestley_taylor::response;
    %rename(PriestleyTaylorCalculator) shyft::core::priestley_taylor::calculator;
    %include "core/priestley_taylor.h"


    %rename(ActualEvapotranspirationParameter)  shyft::core::actual_evapotranspiration::parameter;
    %rename(ActualEvapotranspirationResponse)   shyft::core::actual_evapotranspiration::response;
    %rename(ActualEvapotranspirationCalculate_step) shyft::core::actual_evapotranspiration::calculate_step;
    %include "core/actual_evapotranspiration.h"


    %rename(GammaSnowParameter)  shyft::core::gamma_snow::parameter;
    %rename(GammaSnowState)      shyft::core::gamma_snow::state;
    %rename(GammaSnowResponse)   shyft::core::gamma_snow::response;
    %rename(GammaSnowCalculator) shyft::core::gamma_snow::calculator;
    %include "core/gamma_snow.h"

    %rename(SkaugenParameter)  shyft::core::skaugen::parameter;
    %rename(SkaugenState)      shyft::core::skaugen::state;
    %rename(SkaugenResponse)   shyft::core::skaugen::response;
    %include "core/skaugen.h"

    %rename(KirchnerParameter)  shyft::core::kirchner::parameter;
    %rename(KirchnerState)      shyft::core::kirchner::state;
    %rename(KirchnerResponse)   shyft::core::kirchner::response;
    %rename(KirchnerCalculator) shyft::core::kirchner::calculator;
    %include "core/kirchner.h"

    %rename(PrecipitationCorrectionParameter)  shyft::core::precipitation_correction::parameter;
    %rename(PrecipitationCorrectionCalculator) shyft::core::precipitation_correction::calculator;
    %include "core/precipitation_correction.h"

    //-- method stacks/assemblies
    %rename(PTGSKResponse)    shyft::core::pt_gs_k::response;
    %rename(PTGSKState)       shyft::core::pt_gs_k::state;
    %rename(PTGSKParameter)   shyft::core::pt_gs_k::parameter;
    %include "core/pt_gs_k.h"

    %rename(PTSSKResponse)  shyft::core::pt_ss_k::response;
    %rename(PTSSKState)     shyft::core::pt_ss_k::state;
    %rename(PTSSKParameter) shyft::core::pt_ss_k::parameter;
    %include "core/pt_ss_k.h"

    %include "core/cell_model.h"

    %rename(PTGSKDischargeCollector )  shyft::core::pt_gs_k::discharge_collector;
    %rename(PTGSKNullCollector)        shyft::core::pt_gs_k::null_collector;
    %rename(PTGSKStateCollector)       shyft::core::pt_gs_k::state_collector;
    %rename(PTGSKAllCollector)         shyft::core::pt_gs_k::all_response_collector;

    %rename(PTSSKDischargeCollector )  shyft::core::pt_ss_k::discharge_collector;
    %rename(PTSSKNullCollector)        shyft::core::pt_ss_k::null_collector;
    %rename(PTSSKStateCollector)       shyft::core::pt_ss_k::state_collector;
    %rename(PTSSKAllCollector)         shyft::core::pt_ss_k::all_response_collector;


    %include "core/pt_gs_k_cell_model.h"
    %include "core/pt_ss_k_cell_model.h"

    %rename(IDWParameter)  shyft::core::inverse_distance::parameter;
    %rename(IDWTemperatureParameter)  shyft::core::inverse_distance::temperature_parameter;
    %rename(IDWPrecipitationParameter)  shyft::core::inverse_distance::precipitation_parameter;
    %include "core/inverse_distance.h"



    %rename(BTKConstParameter)  shyft::core::bayesian_kriging::const_parameter;
    %rename(BTKParameter)       shyft::core::bayesian_kriging::parameter;
    %include "core/bayesian_kriging.h"


    %include "core/region_model.h"

    %include "core/model_calibration.h"


    %rename (PTGSKStateIo) shyft::api::ptgsk_state_io;
    %include "api.h"


namespace shyft {
    namespace core {
        namespace pt_gs_k {
            %template(PTGSKCellAll)  cell<parameter_t, environment_t, state_t, state_collector, all_response_collector>;
            typedef cell<parameter_t, environment_t, state_t, state_collector, all_response_collector> PTGSKCellAll;

            %template(PTGSKCellOpt)     cell<parameter_t, environment_t, state_t, null_collector, discharge_collector>;
            typedef cell<parameter_t, environment_t, state_t, null_collector, discharge_collector> PTGSKCellOpt;
        }

        namespace pt_ss_k {
            %template(PTSSKCellAll)  cell<parameter_t, environment_t, state_t, state_collector, all_response_collector>;
            typedef cell<parameter_t, environment_t, state_t, state_collector, all_response_collector> PTSSKCellAll;

            %template(PTSSKCellOpt)     cell<parameter_t, environment_t, state_t, null_collector, discharge_collector>;
            typedef cell<parameter_t, environment_t, state_t, null_collector, discharge_collector> PTSSKCellOpt;
        }

        namespace model_calibration {
            //  integrated in parameter :%template(PTGSKParameterAccessor) model_calibration::ptgsk_parameter_accessor<shyft::core::pt_gs_k::parameter>;

            %template(TargetSpecificationPts) target_specification<shyft::core::pts_t>;
         typedef ts_transform TsTransform;
         // to_average template method specialization
         %template(to_average) TsTransform::to_average<shyft::core::pts_t,shyft::api::ITimeSeriesOfPoints>;
         %template(to_average) TsTransform::to_average<shyft::core::pts_t,shyft::core::pts_t>;
            %template(PTGSKOptimizer) optimizer<region_model<pt_gs_k::cell_discharge_response_t>,shyft::core::pt_gs_k::parameter,pts_t>;
        }
        %template(CellEnvironment)              environment<timeaxis_t, pts_t, pts_t, pts_t, pts_t, pts_t>;
        %template(CellEnvironmentConstRHumWind) environment<timeaxis_t, pts_t, pts_t, pts_t, cts_t, cts_t>;

        ///-- region model
        %template(PTGSKOptModel) region_model<pt_gs_k::cell_discharge_response_t>;
        typedef region_model<pt_gs_k::cell_discharge_response_t> PTGSKOptModel;
        %template(PTGSKModel) region_model<pt_gs_k::cell_complete_response_t>;
        typedef region_model<pt_gs_k::cell_complete_response_t> PTGSKModel;
        %template(run_interpolation) PTGSKModel::run_interpolation<shyft::api::a_region_environment,shyft::core::interpolation_parameter>;
        %template(run_interpolation) PTGSKOptModel::run_interpolation<shyft::api::a_region_environment,shyft::core::interpolation_parameter>;

        %template(PTSSKOptModel) region_model<pt_ss_k::cell_discharge_response_t>;
        typedef region_model<pt_ss_k::cell_discharge_response_t> PTSSKOptModel;
        %template(PTSSKModel) region_model<pt_ss_k::cell_complete_response_t>;
        typedef region_model<pt_ss_k::cell_complete_response_t> PTSSKModel;
        %template(run_pt_ss_k_interpolation) PTSSKModel::run_interpolation<shyft::api::a_region_environment, shyft::core::interpolation_parameter>;
        %template(run_pt_ss_k_interpolation) PTSSKOptModel::run_interpolation<shyft::api::a_region_environment, shyft::core::interpolation_parameter>;
    }
}


// Expose the individual methods in the method stack
namespace shyft {
    namespace core {
        %template(SkaugenCalculator) skaugen::calculator<skaugen::parameter, skaugen::state, skaugen::response>;
        typedef skaugen::calculator<skaugen::parameter, skaugen::state, skaugen::response> SkaugenCalculator;
    }
}

// Since python does not support templates, we need to help swig with
// valid names for the template instances we use/expose.
// expand useful templates from std namespace here
// note: in the order of of appearance (easy for std lib)

namespace std {
    // basic types goes here
   %template(IntVector) vector<int>;
    %template(DoubleVector) vector<double>;
    %template(UtcTimeVector) vector<shyft::core::utctime>;
    %template(StringVector) vector<string>;

   // generic ITimeSeriesOfPoints_ reference vector (avoid copy ts)
    %template(TsVector) vector< shyft::api::ITimeSeriesOfPoints_ >;

   // geo located timeseries vectors, maybe we should rename it to GeoLocatedTemperature? etc.
    %template(TemperatureSourceVector) vector<shyft::api::TemperatureSource>;
    %template(PrecipitationSourceVector) vector<shyft::api::PrecipitationSource>;
    %template(RadiationSourceVector) vector<shyft::api::RadiationSource>;
    %template(RelHumSourceVector) vector<shyft::api::RelHumSource>;
    %template(WindSpeedSourceVector) vector<shyft::api::WindSpeedSource>;

   // then vectors of cells and state
   %template(PTGSKCellAllVector) vector< shyft::core::cell<shyft::core::pt_gs_k::parameter, shyft::core::environment_t, shyft::core::pt_gs_k::state_t, shyft::core::pt_gs_k::state_collector, shyft::core::pt_gs_k::all_response_collector> >;
    %template(PTGSKCellOptVector) vector< shyft::core::cell<shyft::core::pt_gs_k::parameter, shyft::core::environment_t, shyft::core::pt_gs_k::state_t, shyft::core::pt_gs_k::null_collector, shyft::core::pt_gs_k::discharge_collector> >;
    %template(PTGSKStateVector) vector<shyft::core::pt_gs_k::state_t >;

   %template(PTSSKCellAllVector) vector< shyft::core::cell<shyft::core::pt_ss_k::parameter_t, shyft::core::environment_t, shyft::core::pt_ss_k::state_t, shyft::core::pt_ss_k::state_collector, shyft::core::pt_ss_k::all_response_collector> >;
    %template(PTSSKCellOptVector) vector< shyft::core::cell<shyft::core::pt_ss_k::parameter_t, shyft::core::environment_t, shyft::core::pt_ss_k::state_t, shyft::core::pt_ss_k::null_collector, shyft::core::pt_ss_k::discharge_collector> >;
    %template(PTSSKStateVector) vector<shyft::core::pt_ss_k::state_t >;

   %template(TargetSpecificationVector) vector<shyft::core::model_calibration::target_specification<shyft::core::pts_t>>;

   // some extensions to ease conversion to/from numpy
   %extend vector<double> {
        static vector<double> FromNdArray(int DIM1,double *IN_ARRAY1) {
            return std::vector<double>(IN_ARRAY1,IN_ARRAY1+DIM1);
        }
    };
   %extend vector<int> {
        static vector<int> FromNdArray(int DIM1,int *IN_ARRAY1) {
            return std::vector<int>(IN_ARRAY1,IN_ARRAY1+DIM1);
        }
    };
    %extend vector<shyft::core::utctime> {
        static vector<shyft::core::utctime> FromNdArray(int DIM1,long long *IN_ARRAY1) {
            return std::vector<shyft::core::utctime>(IN_ARRAY1,IN_ARRAY1+DIM1);
        }
    }
}

namespace shyft {

    namespace timeseries {
        %template(TsFixed) point_timeseries<timeaxis>;
        %template(TsPoint) point_timeseries<point_timeaxis>;
      %template(TsSinFx) function_timeseries<timeaxis,sin_fx>;
      %template(AverageAccessorTsFixed) average_accessor<shyft::api::GenericTs<shyft::timeseries::point_timeseries<shyft::timeseries::timeaxis>>,shyft::timeseries::timeaxis>;
      %template(AverageAccessorTsPoint) average_accessor<shyft::api::GenericTs<shyft::timeseries::point_timeseries<shyft::timeseries::point_timeaxis>>,shyft::timeseries::timeaxis>;
        %template(AverageAccessorTs) average_accessor<shyft::api::ITimeSeriesOfPoints,shyft::timeseries::timeaxis>;

    }

    namespace api {
        typedef ITimeSeriesOfPoints ITimeSeriesDouble;
        %template(TsCoreFixed) GenericTs<point_timeseries<timeaxis>>;
        %template(TsCorePoint) GenericTs<point_timeseries<point_timeaxis>>;

        // PT_GS_K
        %template(PTGSKCellAllStatistics) basic_cell_statistics<shyft::core::cell<shyft::core::pt_gs_k::parameter, shyft::core::environment_t, shyft::core::pt_gs_k::state_t, shyft::core::pt_gs_k::state_collector, shyft::core::pt_gs_k::all_response_collector>>;
      %template(PTGSKCellGammaSnowStateStatistics) gamma_snow_cell_state_statistics<shyft::core::cell<shyft::core::pt_gs_k::parameter, shyft::core::environment_t, shyft::core::pt_gs_k::state_t, shyft::core::pt_gs_k::state_collector, shyft::core::pt_gs_k::all_response_collector>>;
      %template(PTGSKCellGammaSnowResponseStatistics) gamma_snow_cell_response_statistics<shyft::core::cell<shyft::core::pt_gs_k::parameter, shyft::core::environment_t, shyft::core::pt_gs_k::state_t, shyft::core::pt_gs_k::state_collector, shyft::core::pt_gs_k::all_response_collector>>;
      %template(PTGSKCellPriestleyTaylorResponseStatistics) priestley_taylor_cell_response_statistics<shyft::core::cell<shyft::core::pt_gs_k::parameter, shyft::core::environment_t, shyft::core::pt_gs_k::state_t, shyft::core::pt_gs_k::state_collector, shyft::core::pt_gs_k::all_response_collector>>;
      %template(PTGSKCellActualEvapotranspirationResponseStatistics) actual_evapotranspiration_cell_response_statistics<shyft::core::cell<shyft::core::pt_gs_k::parameter, shyft::core::environment_t, shyft::core::pt_gs_k::state_t, shyft::core::pt_gs_k::state_collector, shyft::core::pt_gs_k::all_response_collector>>;
        %template(PTGSKCellKirchnerStateStatistics) kirchner_cell_state_statistics<shyft::core::cell<shyft::core::pt_gs_k::parameter, shyft::core::environment_t, shyft::core::pt_gs_k::state_t, shyft::core::pt_gs_k::state_collector, shyft::core::pt_gs_k::all_response_collector>>;
      %template(PTGSKCellOptStatistics) basic_cell_statistics<shyft::core::cell<shyft::core::pt_gs_k::parameter, shyft::core::environment_t, shyft::core::pt_gs_k::state_t, shyft::core::pt_gs_k::null_collector, shyft::core::pt_gs_k::discharge_collector>>;

        // PT_SS_K
        %template(PTSSKCellAllStatistics) basic_cell_statistics<shyft::core::cell<shyft::core::pt_ss_k::parameter_t, shyft::core::environment_t, shyft::core::pt_ss_k::state_t, shyft::core::pt_ss_k::state_collector, shyft::core::pt_ss_k::all_response_collector>>;
      %template(PTSSKCellSnowStateStatistics) skaugen_cell_state_statistics<shyft::core::cell<shyft::core::pt_ss_k::parameter_t, shyft::core::environment_t, shyft::core::pt_ss_k::state_t, shyft::core::pt_ss_k::state_collector, shyft::core::pt_ss_k::all_response_collector>>;
      %template(PTSSKCellSnowResponseStatistics) skaugen_cell_response_statistics<shyft::core::cell<shyft::core::pt_ss_k::parameter_t, shyft::core::environment_t, shyft::core::pt_ss_k::state_t, shyft::core::pt_ss_k::state_collector, shyft::core::pt_ss_k::all_response_collector>>;
      %template(PTSSKCellPriestleyTaylorResponseStatistics) priestley_taylor_cell_response_statistics<shyft::core::cell<shyft::core::pt_ss_k::parameter_t, shyft::core::environment_t, shyft::core::pt_ss_k::state_t, shyft::core::pt_ss_k::state_collector, shyft::core::pt_ss_k::all_response_collector>>;
      %template(PTSSKCellActualEvapotranspirationResponseStatistics) actual_evapotranspiration_cell_response_statistics<shyft::core::cell<shyft::core::pt_ss_k::parameter_t, shyft::core::environment_t, shyft::core::pt_ss_k::state_t, shyft::core::pt_ss_k::state_collector, shyft::core::pt_ss_k::all_response_collector>>;
        %template(PTSSKCellKirchnerStateStatistics) kirchner_cell_state_statistics<shyft::core::cell<shyft::core::pt_ss_k::parameter_t, shyft::core::environment_t, shyft::core::pt_ss_k::state_t, shyft::core::pt_ss_k::state_collector, shyft::core::pt_ss_k::all_response_collector>>;
      %template(PTSSKCellOptStatistics) basic_cell_statistics<shyft::core::cell<shyft::core::pt_ss_k::parameter_t, shyft::core::environment_t, shyft::core::pt_ss_k::state_t, shyft::core::pt_ss_k::null_collector, shyft::core::pt_ss_k::discharge_collector>>;
    }
}




%extend shyft::core::geo_point {
    char *__str__() {
        static char temp[256];
        sprintf(temp, "[%g, %g, %g]", $self->x, $self->y, $self->z);
        return &temp[0];
    }
}


%extend shyft::timeseries::timeaxis {
%pythoncode %{
    def __str__(self):
        utc=Calendar()
        return "Timeaxis("+utc.to_string(self(0).start) + ", deltat=" + str(self.delta()) + "s, n= " + str(self.size())+")"

    def __len__(self):
        return self.size();

    def __iter__(self):
        self.counter = 0
        return self

    def next(self):
        if self.counter >= len(self):
            del self.counter
            raise StopIteration
        self.counter += 1
        return self(self.counter - 1)

%}
}
%extend shyft::core::utcperiod {
%pythoncode %{
    def __str__(self):
        return self.to_string()

%}

}

// to ease the python interface, we attach the type-specific species to the mode.s
// so that if we have a model type, we also have the
// .cell_t
// .base_statistics_t (with .catchment_temperature etc , + .catchment_discharge)
// [opt].<method>_statistics_t, <method> = gs, thus sca,swe, + 6 state params.
//
%pythoncode %{

PTGSKModel.cell_t = PTGSKCellAll
PTGSKModel.statistics=property(lambda self:PTGSKCellAllStatistics(self.get_cells()))
PTGSKModel.gamma_snow_state=property(lambda self:PTGSKCellGammaSnowStateStatistics(self.get_cells()))
PTGSKModel.gamma_snow_response=property(lambda self:PTGSKCellGammaSnowResponseStatistics(self.get_cells()))
PTGSKModel.priestley_taylor_response=property(lambda self:PTGSKCellPriestleyTaylorResponseStatistics(self.get_cells()))
PTGSKModel.actual_evaptranspiration_response=property(lambda self:PTGSKCellActualEvapotranspirationResponseStatistics(self.get_cells()))
PTGSKModel.kirchner_state=property(lambda self:PTGSKCellKirchnerStateStatistics(self.get_cells()))

PTGSKOptModel.cell_t = PTGSKCellOpt
PTGSKOptModel.statistics=property(lambda self:PTGSKCellOptStatistics(self.get_cells()))

PTGSKCellAll.vector_t=PTGSKCellAllVector
PTGSKCellOpt.vector_t=PTGSKCellOptVector

PTSSKModel.cell_t = PTSSKCellAll
PTSSKModel.statistics = property(lambda self: PTSSKCellAllStatistics(self.get_cells()))
PTSSKModel.snow_state = property(lambda self: PTSSKCellSnowStateStatistics(self.get_cells()))
PTSSKModel.snow_response = property(lambda self:PTSSKCellSnowResponseStatistics(self.get_cells()))
PTSSKModel.priestley_taylor_response = property(lambda self:PTSSKCellPriestleyTaylorResponseStatistics(self.get_cells()))
PTSSKModel.actual_evaptranspiration_response = property(lambda self:PTSSKCellActualEvapotranspirationResponseStatistics(self.get_cells()))
PTSSKModel.kirchner_state = property(lambda self:PTSSKCellKirchnerStateStatistics(self.get_cells()))
PTSSKOptModel.cell_t = PTSSKCellOpt
PTSSKOptModel.statistics = property(lambda self:PTSSKCellOptStatistics(self.get_cells()))
PTSSKCellAll.vector_t = PTSSKCellAllVector
PTSSKCellOpt.vector_t = PTSSKCellOptVector

TemperatureSource.vector_t = TemperatureSourceVector
PrecipitationSource.vector_t = PrecipitationSourceVector
RadiationSource.vector_t = RadiationSourceVector
RelHumSource.vector_t = RelHumSourceVector
WindSpeedSource.vector_t = WindSpeedSourceVector
%}
//
//
// Finally, exception handling mapping std::exception to python
//

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
