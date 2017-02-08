#include "boostpython_pch.h"

#include "py_convertible.h"

#include "core/utctime_utilities.h"
#include "core/time_axis.h"
#include "core/timeseries.h"
#include "api/api.h"
#include "core/model_calibration.h"

namespace expose {
    using namespace shyft;
    using namespace shyft::core;
    using namespace boost::python;
    using namespace std;

    struct TsTransform {
        shared_ptr<shyft::core::pts_t> to_average(utctime start, utctimespan dt, size_t n,const shyft::api::apoint_ts& src) {
            return shyft::model_calibration::ts_transform().to_average<shyft::core::pts_t,shyft::api::apoint_ts>(start,dt,n,src);
        }
        shared_ptr<shyft::core::pts_t> to_average(utctime start, utctimespan dt, size_t n,shared_ptr<shyft::api::apoint_ts> src) {
            return shyft::model_calibration::ts_transform().to_average<shyft::core::pts_t,shyft::api::apoint_ts>(start,dt,n,src);
        }

        shared_ptr<shyft::core::pts_t> to_average(utctime start, utctimespan dt, size_t n,const shyft::core::pts_t& src) {
            return shyft::model_calibration::ts_transform().to_average<shyft::core::pts_t,shyft::core::pts_t>(start,dt,n,src);
        }
        shared_ptr<shyft::core::pts_t> to_average(utctime start, utctimespan dt, size_t n,shared_ptr<shyft::core::pts_t> src) {
            return shyft::model_calibration::ts_transform().to_average<shyft::core::pts_t,shyft::core::pts_t>(start,dt,n,src);
        }
    };

    void target_specification() {
        enum_<model_calibration::target_spec_calc_type>("TargetSpecCalcType")
            .value("NASH_SUTCLIFFE",model_calibration::NASH_SUTCLIFFE)
            .value("KLING_GUPTA",model_calibration::KLING_GUPTA)
            .value("ABS_DIFF",model_calibration::ABS_DIFF)
            .export_values()
            ;
        enum_<model_calibration::target_property_type>("CatchmentPropertyType")
            .value("DISCHARGE", model_calibration::DISCHARGE)
            .value("SNOW_COVERED_AREA", model_calibration::SNOW_COVERED_AREA)
            .value("SNOW_WATER_EQUIVALENT", model_calibration::SNOW_WATER_EQUIVALENT)
            .value("ROUTED_DISCHARGE",model_calibration::ROUTED_DISCHARGE)
            .value("CELL_CHARGE",model_calibration::CELL_CHARGE)
            .export_values()
            ;
        typedef shyft::core::pts_t target_ts_t;
		
        typedef  model_calibration::target_specification<target_ts_t> TargetSpecificationPts;
		
        class_<TargetSpecificationPts>("TargetSpecificationPts",
            "To guide the model calibration, we have a goal-function that we try to minimize\n"
            "This class contains the needed specification of this goal-function so that we can\n"
            " 1. from simulations, collect time-series at catchment level for (DISCHARGE|SNOW_COVERED_AREA|SNOW_WATER_EQUIVALENT)\n"
            " 2. from observations, have a user-specified series the expression above should be equal to\n"
            " 3. use user specified kling-gupta factors to evaluate kind-of-difference between target and simulated\n"
            " 4. scale-factor to put a weight on this specific target-specification compared to other(we can have multiple)\n"
			" 5. a user specified id, uid, a string to identify the external source for calibration\n"
            "\n"
            )
            .def(init<const target_ts_t&,vector<int>,double,
                 optional<model_calibration::target_spec_calc_type,double,double,double,model_calibration::target_property_type,std::string>>(
                 args("ts","cids","scale_factor","calc_mode","s_r","s_a","s_b","catchment_property","uid"),"constructs a complete target specification using a TsFixed as target ts")
            )
            .def(init<const target_ts_t&, int, double,
                optional<model_calibration::target_spec_calc_type, double, double, double, std::string>>(
                    args("ts", "river_id", "scale_factor", "calc_mode", "s_r", "s_a", "s_b", "uid"), "constructs a complete target specification using a TsFixed as target ts")
            )
			/*Wanted! .def(init<shyft::api::apoint_ts, vector<int>, double,
				optional<model_calibration::target_spec_calc_type, double, double, double, model_calibration::catchment_property_type>>(
					args("ts", "cids", "scale_factor", "calc_mode", "s_r", "s_a", "s_b", "catchment_property"), "constructs a complete target specification, using a Timeseries as target ts")
			)*/
			.def_readwrite("scale_factor", &TargetSpecificationPts::scale_factor, "the scale factor to be used when considering multiple target_specifications")
            .def_readwrite("calc_mode",&TargetSpecificationPts::calc_mode,"*NASH_SUTCLIFFE, KLING_GUPTA")
            .def_readwrite("catchment_property",&TargetSpecificationPts::catchment_property,"*DISCHARGE,SNOW_COVERED_AREA, SNOW_WATER_EQUIVALENT")
            .def_readwrite("s_r",&TargetSpecificationPts::s_r,"KG-scalefactor for correlation")
            .def_readwrite("s_a",&TargetSpecificationPts::s_a,"KG-scalefactor for alpha (variance)")
            .def_readwrite("s_b",&TargetSpecificationPts::s_b,"KG-scalefactor for beta (bias)")
            .def_readwrite("ts", &TargetSpecificationPts::ts," target ts")
            .def_readwrite("river_id",&TargetSpecificationPts::river_id,"river identifier for routed discharge calibration")
			.def_readwrite("catchment_indexes",&TargetSpecificationPts::catchment_indexes,"catchment indexes, 'cids'")
		    .def_readwrite("uid",&TargetSpecificationPts::uid,"user specified identifier:string")
            ;


        typedef vector<TargetSpecificationPts> TargetSpecificationVector;
        class_<TargetSpecificationVector>("TargetSpecificationVector","A list of (weighted) target specifications to be used for model calibration")
            .def(vector_indexing_suite<TargetSpecificationVector>())
         ;

        shared_ptr<shyft::core::pts_t> (TsTransform::*m1)(utctime , utctimespan , size_t ,const shyft::api::apoint_ts& )=&TsTransform::to_average;
        shared_ptr<shyft::core::pts_t> (TsTransform::*m2)(utctime , utctimespan , size_t ,shared_ptr<shyft::api::apoint_ts> )=&TsTransform::to_average;
        shared_ptr<shyft::core::pts_t> (TsTransform::*m3)(utctime , utctimespan , size_t ,const shyft::core::pts_t&) = &TsTransform::to_average;
        shared_ptr<shyft::core::pts_t> (TsTransform::*m4)(utctime , utctimespan , size_t ,shared_ptr<shyft::core::pts_t> ) = &TsTransform::to_average;

        class_<TsTransform>("TsTransform",
                "transform the supplied time-series, f(t) interpreted according to its point_interpretation() policy\n"
                " into a new shyft core TsFixed time-series,\n"
                " that represents the true average for each of the n intervals of length dt, starting at start.\n"
                " the result ts will have the policy is set to POINT_AVERAGE_VALUE\n"
                " \note that the resulting ts is a fresh new ts, not connected to the source ts\n"
            )
            .def("to_average",m1,args("start","dt","n","src"),"")
            .def("to_average",m2,args("start","dt","n","src"),"")
            .def("to_average",m3,args("start","dt","n","src"),"")
            .def("to_average",m4,args("start","dt","n","src"),"")
            ;
        py_api::iterable_converter()
            .from_python< std::vector<TargetSpecificationPts> >()
        ;
    }
}
