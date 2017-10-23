#include "boostpython_pch.h"

#include "core/utctime_utilities.h"
#include "core/time_axis.h"
#include "core/time_series.h"
#include "api/api.h"
#include "core/model_calibration.h"

#include "py_convertible.h"

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
    typedef shyft::core::pts_t target_ts_t;

    typedef  model_calibration::target_specification<target_ts_t> TargetSpecificationPts;

    /** custom constructors needed for target-spec, to accept any type of ts
    * and at the same time represent the same efficient core-type at the
    * c++ impl. level
    * TODO: could we simply change the type of target_ts to apoint_ts and accept it as ok performance?
    * \ref boost::python make_constructor
    */
    struct target_specification_ext {

        static TargetSpecificationPts* create_default() {
            return new model_calibration::target_specification<target_ts_t>();
        }

        static TargetSpecificationPts* create_cids(
               const target_ts_t& ts,
               vector<int> cids,
               double scale_factor,
               model_calibration::target_spec_calc_type calc_mode = model_calibration::NASH_SUTCLIFFE,
               double s_r = 1.0,
               double s_a = 1.0,
               double s_b = 1.0,
               model_calibration::target_property_type catchment_property_ = model_calibration::DISCHARGE,
               std::string uid = "")
        {
            return  new model_calibration::target_specification<target_ts_t>(ts,cids,scale_factor,calc_mode,s_r,s_a,s_b,catchment_property_,uid);
        }
        static void ensure_time_axis_requirements(const shyft::api::gta_t& gta) {
            if(gta.gt != time_axis::generic_dt::FIXED)
                throw runtime_error("the supplied target specification ts time_axis must be a trivial fixed interval timeaxis");
        }
        static TargetSpecificationPts* acreate_cids(
               const shyft::api::apoint_ts& ats,
               vector<int> cids,
               double scale_factor,
               model_calibration::target_spec_calc_type calc_mode = model_calibration::NASH_SUTCLIFFE,
               double s_r = 1.0,
               double s_a = 1.0,
               double s_b = 1.0,
               model_calibration::target_property_type catchment_property_ = model_calibration::DISCHARGE,
               std::string uid = "")
        {
            ensure_time_axis_requirements(ats.time_axis());
            return  new model_calibration::target_specification<target_ts_t>(target_ts_t(ats.time_axis().f,ats.values(),ats.point_interpretation()),cids,scale_factor,calc_mode,s_r,s_a,s_b,catchment_property_,uid);
        }

        static TargetSpecificationPts* create_cids2(
               const target_ts_t& ts,
               vector<int> cids,
               double scale_factor,
               model_calibration::target_spec_calc_type calc_mode )
        {
            return  create_cids(ts,cids,scale_factor,calc_mode);
        }

        static TargetSpecificationPts* acreate_cids2(
               const shyft::api::apoint_ts& ats,
               vector<int> cids,
               double scale_factor,
               model_calibration::target_spec_calc_type calc_mode )
        {
            ensure_time_axis_requirements(ats.time_axis());
            return  create_cids(target_ts_t(ats.time_axis().f,ats.values(),ats.point_interpretation()),cids,scale_factor,calc_mode);
        }

        static TargetSpecificationPts* create_rid(
               const target_ts_t& ts,
               int river_id,
               double scale_factor,
               model_calibration::target_spec_calc_type calc_mode = model_calibration::NASH_SUTCLIFFE,
               double s_r = 1.0,
               double s_a = 1.0,
               double s_b = 1.0,
               std::string uid = "")
        {
            return  new model_calibration::target_specification<target_ts_t>(ts,river_id,scale_factor,calc_mode,s_r,s_a,s_b,uid);
        }
        static TargetSpecificationPts* acreate_rid(
               const shyft::api::apoint_ts& ats,
               int river_id,
               double scale_factor,
               model_calibration::target_spec_calc_type calc_mode = model_calibration::NASH_SUTCLIFFE,
               double s_r = 1.0,
               double s_a = 1.0,
               double s_b = 1.0,
               std::string uid = "")
        {
            ensure_time_axis_requirements(ats.time_axis());
            return  new model_calibration::target_specification<target_ts_t>(target_ts_t(ats.time_axis().f,ats.values(),ats.point_interpretation()),river_id,scale_factor,calc_mode,s_r,s_a,s_b,uid);
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
        using  pyarg=boost::python::arg;
        class_<TargetSpecificationPts>("TargetSpecificationPts",
            "To guide the model calibration, we have a goal-function that we try to minimize\n"
            "This class contains the needed specification of this goal-function so that we can\n"
            " 1. from simulations, collect time-series at catchment level for (DISCHARGE|SNOW_COVERED_AREA|SNOW_WATER_EQUIVALENT)\n"
            " 2. from observations, have a user-specified series the expression above should be equal to\n"
            " 3. use user specified kling-gupta factors to evaluate kind-of-difference between target and simulated\n"
            " 4. scale-factor to put a weight on this specific target-specification compared to other(we can have multiple)\n"
			" 5. a user specified id, uid, a string to identify the external source for calibration\n"
            "\n"
            ,no_init // required to make custom constructors
            )
//            .def(init<const target_ts_t&,vector<int>,double,
//                 optional<model_calibration::target_spec_calc_type,double,double,double,model_calibration::target_property_type,std::string>>(
//                 args("ts","cids","scale_factor","calc_mode","s_r","s_a","s_b","catchment_property","uid"),"constructs a complete target specification using a TsFixed as target ts")
//            )
//            .def(init<const target_ts_t&, int, double,
//                optional<model_calibration::target_spec_calc_type, double, double, double, std::string>>(
//                    args("ts", "river_id", "scale_factor", "calc_mode", "s_r", "s_a", "s_b", "uid"), "constructs a complete target specification using a TsFixed as target ts")
//            )
            .def("__init__",make_constructor(&target_specification_ext::create_default),
                 doc_intro("Construct an empty class")
                 )
            .def("__init__",make_constructor(&target_specification_ext::create_cids,
                default_call_policies(),
                (pyarg("ts"),pyarg("cids"),pyarg("scale_factor"),pyarg("calc_mode"),
                 pyarg("s_r"),pyarg("s_a"),pyarg("s_b"),pyarg("catchment_property"),pyarg("uid"))
                ),
                doc_intro("construct a target specification filled in with supplied parameters")
                doc_parameters()
                doc_parameter("ts","TsFixed","time-series containing the target time-series")
                doc_parameter("cids","IntVector","A list of catchment id's(cids) that together adds up into same as the target-ts")
                doc_parameter("scale_factor","float","the weight of this target-specification")
                doc_parameter("calc_mode","TargetSpecCalcType","specifies how to calculate the goal function, NS, KG, Abs method")
                doc_parameter("s_r","float","KG scalefactor for correlation")
                doc_parameter("s_a","float","KG scalefactor for alpha(variance)")
                doc_parameter("s_b","float","KG scalefactor for beta(bias)")
                doc_parameter("catchment_property","CatchmentPropertyType","what to extract from catchment(DISCHARGE|SNOW_COVERED_AREA|SNOW_WATER_EQUIVALENT|ROUTED_DISCHARGE|CELL_CHARGE)")
                doc_parameter("uid","str","user specified string/id to help integration efforts")
             )
            .def("__init__",make_constructor(&target_specification_ext::acreate_cids,
                default_call_policies(),
                (pyarg("ts"),pyarg("cids"),pyarg("scale_factor"),pyarg("calc_mode"),
                 pyarg("s_r"),pyarg("s_a"),pyarg("s_b"),pyarg("catchment_property"),pyarg("uid"))
                ),
                doc_intro("construct a target specification filled in with supplied parameters")
                doc_parameters()
                doc_parameter("ts","TimeSeries","time-series containing the target time-series, note that the time-axis of this ts must be a fixed-interval type")
                doc_parameter("cids","IntVector","A list of catchment id's(cids) that together adds up into same as the target-ts")
                doc_parameter("scale_factor","float","the weight of this target-specification")
                doc_parameter("calc_mode","TargetSpecCalcType","specifies how to calculate the goal function, NS, KG, Abs method")
                doc_parameter("s_r","float","KG scalefactor for correlation")
                doc_parameter("s_a","float","KG scalefactor for alpha(variance)")
                doc_parameter("s_b","float","KG scalefactor for beta(bias)")
                doc_parameter("catchment_property","CatchmentPropertyType","what to extract from catchment(DISCHARGE|SNOW_COVERED_AREA|SNOW_WATER_EQUIVALENT|ROUTED_DISCHARGE|CELL_CHARGE)")
                doc_parameter("uid","str","user specified string/id to help integration efforts")
                 )

            .def("__init__",make_constructor(&target_specification_ext::create_cids2,
                default_call_policies(),
                (pyarg("ts"),pyarg("cids"),pyarg("scale_factor"),pyarg("calc_mode"))
                ),
                doc_intro("construct a target specification filled in with supplied parameters")
                doc_parameters()
                doc_parameter("ts","TsFixed","time-series containing the target time-series")
                doc_parameter("cids","IntVector","A list of catchment id's(cids) that together adds up into same as the target-ts")
                doc_parameter("scale_factor","float","the weight of this target-specification")
                doc_parameter("calc_mode","TargetSpecCalcType","specifies how to calculate the goal function, NS, KG, Abs method")
             )
            .def("__init__",make_constructor(&target_specification_ext::acreate_cids2,
                default_call_policies(),
                (pyarg("ts"),pyarg("cids"),pyarg("scale_factor"),pyarg("calc_mode"))
                ),
                doc_intro("construct a target specification filled in with supplied parameters")
                doc_parameters()
                doc_parameter("ts","TimeSeries","time-series containing the target time-series, note the time-axis needs to be fixed_dt!")
                doc_parameter("cids","IntVector","A list of catchment id's(cids) that together adds up into same as the target-ts")
                doc_parameter("scale_factor","float","the weight of this target-specification")
                doc_parameter("calc_mode","TargetSpecCalcType","specifies how to calculate the goal function, NS, KG, Abs method")
             )


            .def("__init__",make_constructor(&target_specification_ext::create_rid,
                default_call_policies(),
                (pyarg("ts"),pyarg("rid"),pyarg("scale_factor"),pyarg("calc_mode"),
                 pyarg("s_r"),pyarg("s_a"),pyarg("s_b"),pyarg("uid"))
                ),
                doc_intro("construct a target specification filled in with supplied parameters")
                doc_parameters()
                doc_parameter("ts","TsFixed","time-series containing the target time-series")
                doc_parameter("rid","int","A river-id identifying the point of flow in the river-network")
                doc_parameter("scale_factor","float","the weight of this target-specification")
                doc_parameter("calc_mode","TargetSpecCalcType","specifies how to calculate the goal function, NS, KG, Abs method")
                doc_parameter("s_r","float","KG scalefactor for correlation")
                doc_parameter("s_a","float","KG scalefactor for alpha(variance)")
                doc_parameter("s_b","float","KG scalefactor for beta(bias)")
                doc_parameter("uid","str","user specified string/id to help integration efforts")
            )
            .def("__init__",make_constructor(&target_specification_ext::acreate_rid,
                default_call_policies(),
                (pyarg("ts"),pyarg("rid"),pyarg("scale_factor"),pyarg("calc_mode"),
                 pyarg("s_r"),pyarg("s_a"),pyarg("s_b"),pyarg("uid"))
                ),
                doc_intro("construct a target specification filled in with supplied parameters")
                doc_parameters()
                doc_parameter("ts","TimeSeries","time-series containing the target time-series, note time-axis required to be fixed-dt type")
                doc_parameter("rid","int","A river-id identifying the point of flow in the river-network")
                doc_parameter("scale_factor","float","the weight of this target-specification")
                doc_parameter("calc_mode","TargetSpecCalcType","specifies how to calculate the goal function, NS, KG, Abs method")
                doc_parameter("s_r","float","KG scalefactor for correlation")
                doc_parameter("s_a","float","KG scalefactor for alpha(variance)")
                doc_parameter("s_b","float","KG scalefactor for beta(bias)")
                doc_parameter("uid","str","user specified string/id to help integration efforts")
             )

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
			.def(init<const TargetSpecificationVector&>(args("clone"))
			)
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
