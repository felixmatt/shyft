
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/scope.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/return_value_policy.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <boost/python/operators.hpp>
#include <boost/python/overloads.hpp>
#include <boost/python/enum.hpp>
#include <boost/operators.hpp>


#include "core/utctime_utilities.h"
#include "core/time_axis.h"
#include "core/timeseries.h"
#include "api/api.h"


using namespace shyft;
using namespace shyft::core;
using namespace boost::python;
using namespace std;

#define DEF_STD_TS_STUFF() \
        .def("point_interpretation",&pts_t::point_interpretation,"returns the point interpretation policy")\
        .def("set_point_interpretation",&pts_t::set_point_interpretation,args("policy"),"set new policy")\
        .def("value",&pts_t::value,args("i"),"returns the value at the i'th time point")\
        .def("time",&pts_t::time,args("i"),"returns the time at the i'th point")\
        .def("get",&pts_t::get,args("t"),"returns the i'th Point ")\
        .def("set",&pts_t::set,args("i","v"),"set the i'th value")\
        .def("fill",&pts_t::fill,args("v"),"all values with v")\
        .def("scale_by",&pts_t::scale_by,args("v"),"scale all values by specified factor")\
        .def("size",&pts_t::size,"returns number of points")\
        .def("index_of",&pts_t::index_of,args("t"),"return the index of the intervall that contains t, or npos if not found")\
        .def("total_period",&pts_t::total_period,"returns the total period covered by the time-axis of this time-series")\
        .def("__call__",&pts_t::operator(),args("t"),"return the f(t) value for the time-series")


template <class TA>
static void def_point_ts(const char *ts_type_name,const char *doc) {
    typedef timeseries::point_ts<TA> pts_t;
    class_<pts_t,bases<>,shared_ptr<pts_t>,boost::noncopyable>(ts_type_name, doc)
        .def(init<const TA&,const vector<double>&,optional<timeseries::point_interpretation_policy>>(args("ta","v","policy"),"constructs a new timeseries from timeaxis and points"))
        .def(init<const TA&,double,optional<timeseries::point_interpretation_policy>>(args("ta","fill_value","policy"),"constructs a new timeseries from timeaxis and fill-value"))
        DEF_STD_TS_STUFF()
        ;
    register_ptr_to_python<shared_ptr<pts_t> >();
}
static void def_ITimeSeriesOfPoints() {
    typedef shyft::api::ITimeSeriesOfPoints pts_t;
    class_<pts_t,bases<>,shared_ptr<pts_t>,boost::noncopyable>("ITimeSeriesOfPoints", "Generic interface to time-series of points, any type",no_init)
        DEF_STD_TS_STUFF()
        ;
    register_ptr_to_python<shared_ptr<pts_t> >();
    typedef shared_ptr<pts_t> pts_t_;
    typedef vector<pts_t_> TsVector;
    class_<TsVector>("TsVector","A vector of refs to ITimeSeriesOfPoints")
        .def(vector_indexing_suite<TsVector>())
        ;
}
static void def_GenericTs() {
        //not needed, TsFactory does the job
}

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(point_ts_overloads     ,shyft::api::TsFactory::create_point_ts,4,5);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(time_point_ts_overloads,shyft::api::TsFactory::create_time_point_ts,3,4);

static void def_TsFactory() {
    class_<shyft::api::TsFactory>("TsFactory","TsFactory is used to create point time-series that exposes the ITimeSeriesOfPoint interface, using the internal ts-implementations")
        .def("create_point_ts",&shyft::api::TsFactory::create_point_ts,point_ts_overloads())//args("n","tStart","dt","values","interpretation"),"returns a new fixed interval ts from specified arguments")
        .def("create_time_point_ts",&shyft::api::TsFactory::create_time_point_ts,time_point_ts_overloads())//args("period","times","values","interpretation"),"return a point ts from specified arguments")
        ;
}
void def_timeseries() {


    enum_<timeseries::point_interpretation_policy>("point_interpretation_policy")
        .value("POINT_INSTANT_VALUE",timeseries::POINT_INSTANT_VALUE)
        .value("POINT_AVERAGE_VALUE",timeseries::POINT_AVERAGE_VALUE)
        .export_values()
        ;
    class_<timeseries::point> ("Point", "A timeseries point specifying utctime t and value v")
        .def(init<utctime,double>(args("t","v")))
        .def_readwrite("t",&timeseries::point::t)
        .def_readwrite("v",&timeseries::point::v)
        ;
    def_point_ts<time_axis::fixed_dt>("TsFixed","A time-series with a fixed delta t time-axis");
    def_point_ts<time_axis::point_dt>("TsPoint","A time-series with a variable delta time-axis");
    def_ITimeSeriesOfPoints();
    def_GenericTs();
    def_TsFactory();
}
