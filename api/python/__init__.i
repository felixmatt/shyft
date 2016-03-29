/// Python interface for the shyft::core and shyft::api
///
/// Hints to help you survive:
///  swig.org have excellent doc for c++ and swig, the template section is needed reading.
///  special things to absolutely remember
///
///  %template:
///   * remember to always use fully qualified names in template parameters(stay out of trouble with difficult )
///   * it takes forever(several minutes) to regenerate swig (its an issue we work on)


%module(package="shyft.api") __init__

%feature("autodoc", "2");
%feature("kwargs");
%feature("naturalvar");

%begin %{ // gcc win-compile needs this to avoid problems in cmath, fix: to include first
#include <cmath>
%}

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
    #include "core/core_pch.h"
    #include "core/utctime_utilities.h"
    #include "core/time_axis.h"
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
%}

//-- in this section, include all standard mappings available for swig
#define SWIG_SHARED_PTR_NAMESPACE std
    %include <std_shared_ptr.i>

    %include "numpy.i"
    %include <windows.i>
    %include <std_string.i>
    %include <std_vector.i>
    %include <std_map.i>
    %include <stl.i>
    %include <exception.i>


//-- in this section, declare all the shared pointers needs to be declared here,
//   -including their derivates used. (swig warns when missing)
// as a rule we let large objects in/out to be shared pointers.
//


%shared_ptr(std::vector<shyft::api::TemperatureSource>)
%shared_ptr(std::vector<shyft::api::PrecipitationSource>)
%shared_ptr(std::vector<shyft::api::RadiationSource>)
%shared_ptr(std::vector<shyft::api::WindSpeedSource>)
%shared_ptr(std::vector<shyft::api::RelHumSource>)

%shared_ptr( shyft::api::ITimeSeriesOfPoints)
%shared_ptr( shyft::api::GenericTs<shyft::timeseries::point_ts<shyft::time_axis::fixed_dt>>)
%shared_ptr( shyft::api::GenericTs<shyft::timeseries::point_ts<shyft::time_axis::point_dt>>)
%shared_ptr( shyft::api::GenericTs<shyft::timeseries::function_timeseries<shyft::time_axis::fixed_dt,shyft::timeseries::sin_fx>>)
//%shared_ptr( shyft::core::pts_t )
%shared_ptr( shyft::timeseries::point_ts<shyft::time_axis::fixed_dt> )
%shared_ptr( shyft::timeseries::point_ts<shyft::time_axis::point_dt> )

%shared_ptr( shyft::core::time_zone::tz_info<shyft::core::time_zone::tz_table> )

// -- Now we let SWIG parse and interpret the types in enki_api.h
// swig will do its best to reflect all types/methods exposed there into the python wrapper.
// Since we have include swig-mapping for std types, plus our declared shared_ptr, this is a walk in the park :-)
//

//
// Exception handling mapping std::exception to Python
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

    %rename(Timeaxis) shyft::time_axis::fixed_dt;
    %rename(PointTimeaxis) shyft::time_axis::point_dt;
    %rename(TimeaxisCalendarDt) shyft::time_axis::calendar_dt;
    %rename(TimeaxisCalendarDtP) shyft::time_axis::calendar_dt_p;
    %rename(TimeaxisPeriodList) shyft::time_axis::period_list;
    %include "core/time_axis.h"

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

    %rename(HbvSnowParameter)  shyft::core::hbv_snow::parameter;
    %rename(HbvSnowState)      shyft::core::hbv_snow::state;
    %rename(HbvSnowResponse)   shyft::core::hbv_snow::response;
    %rename(HbvSnowCalculator) shyft::core::hbv_snow::calculator;
    %include "core/hbv_snow.h"

    %rename(SkaugenParameter)  shyft::core::skaugen::parameter;
    %rename(SkaugenState)      shyft::core::skaugen::state;
    %rename(SkaugenResponse)   shyft::core::skaugen::response;
    %rename(SkaugenStatistics) shyft::core::skaugen::statistics;
    %rename(SkaugenMassBalanceError) shyft::core::skaugen::mass_balance_error;
    %include "core/skaugen.h"

    %rename(KirchnerParameter)  shyft::core::kirchner::parameter;
    %rename(KirchnerState)      shyft::core::kirchner::state;
    %rename(KirchnerResponse)   shyft::core::kirchner::response;
    %rename(KirchnerCalculator) shyft::core::kirchner::calculator;
    %include "core/kirchner.h"

    %rename(PrecipitationCorrectionParameter)  shyft::core::precipitation_correction::parameter;
    %rename(PrecipitationCorrectionCalculator) shyft::core::precipitation_correction::calculator;
    %include "core/precipitation_correction.h"

    %include "core/cell_model.h"

    %rename(IDWParameter)  shyft::core::inverse_distance::parameter;
    %rename(IDWTemperatureParameter)  shyft::core::inverse_distance::temperature_parameter;
    %rename(IDWPrecipitationParameter)  shyft::core::inverse_distance::precipitation_parameter;
    %include "core/inverse_distance.h"

    %rename(BTKConstParameter)  shyft::core::bayesian_kriging::const_parameter;
    %rename(BTKParameter)       shyft::core::bayesian_kriging::parameter;
    %include "core/bayesian_kriging.h"

    %include "core/region_model.h"

    %include "core/model_calibration.h"

    %include "api/api.h"

namespace shyft {
  namespace core {
     namespace time_zone {
         %template(TzTableInfo) tz_info<shyft::core::time_zone::tz_table>;
         //%template(NameToTzInfoMap) std::map<string,shared_ptr<shyft::core::time_zone::tz_info<shyft::core::time_zone::tz_table> >>;
     }
  }
}

namespace shyft {
  namespace core {
    namespace model_calibration {
        %template(TargetSpecificationPts) target_specification<shyft::core::pts_t>;
        typedef ts_transform TsTransform;
        %template(to_average) TsTransform::to_average<shyft::core::pts_t,shyft::api::ITimeSeriesOfPoints>;
        %template(to_average) TsTransform::to_average<shyft::core::pts_t,shyft::core::pts_t>;
    }
  }
}


namespace shyft {
  namespace core {
    %template(CellEnvironment)              environment<timeaxis_t, pts_t, pts_t, pts_t, pts_t, pts_t>;
    %template(CellEnvironmentConstRHumWind) environment<timeaxis_t, pts_t, pts_t, pts_t, cts_t, cts_t>;
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
    %template(TsFixed) point_ts<shyft::time_axis::fixed_dt>;
    %template(TsPoint) point_ts<shyft::time_axis::point_dt>;
    %template(TsSinFx) function_timeseries<shyft::time_axis::fixed_dt,sin_fx>;
    %template(AverageAccessorTsFixed) average_accessor<shyft::api::GenericTs<shyft::timeseries::point_ts<shyft::time_axis::fixed_dt>>,shyft::time_axis::fixed_dt>;
    %template(AverageAccessorTsPoint) average_accessor<shyft::api::GenericTs<shyft::timeseries::point_ts<shyft::time_axis::point_dt>>,shyft::time_axis::fixed_dt>;
    %template(AverageAccessorTs) average_accessor<shyft::api::ITimeSeriesOfPoints,shyft::time_axis::fixed_dt>;
  }

  namespace api {
    typedef ITimeSeriesOfPoints ITimeSeriesDouble;
    %template(TsCoreFixed) GenericTs<point_ts<shyft::time_axis::fixed_dt>>;
    %template(TsCorePoint) GenericTs<point_ts<shyft::time_axis::point_dt>>;
  }
}


%extend shyft::core::geo_point {
    char *__str__() {
        static char temp[256];
        sprintf(temp, "[%g, %g, %g]", $self->x, $self->y, $self->z);
        return &temp[0];
    }
}


%extend shyft::time_axis::fixed_dt {
%pythoncode %{
    def __str__(self):
        utc=Calendar()
        return "Timeaxis("+utc.to_string(self(0).start) + ", deltat=" + str(self.delta()) + "s, n= " + str(self.size())+")"

    def __len__(self):
        return self.size();

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):                 # Py3-style iterator interface
        return self.next()

    def next(self):
        if self.counter >= len(self):
            del self.counter
            raise StopIteration
        self.counter += 1
        return self(self.counter - 1)

    def __call__(self,ix):
        return self.period(ix)
%}
}

%extend shyft::time_axis::point_dt {
%pythoncode %{
    def __str__(self):
        return "PointTimeAxis("+str(self.total_period()) + ", n= " + str(self.size())+")"

    def __len__(self):
        return self.size();

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):                 # Py3-style iterator interface
        return self.next()

    def next(self):
        if self.counter >= len(self):
            del self.counter
            raise StopIteration
        self.counter += 1
        return self(self.counter - 1)

    def __call__(self,ix):
        return self.period(ix)
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

TemperatureSource.vector_t = TemperatureSourceVector
PrecipitationSource.vector_t = PrecipitationSourceVector
RadiationSource.vector_t = RadiationSourceVector
RelHumSource.vector_t = RelHumSourceVector
WindSpeedSource.vector_t = WindSpeedSourceVector
%}



%init %{
    import_array();
%}
