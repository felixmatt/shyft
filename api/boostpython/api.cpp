
//#include <boost/python.hpp>

#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/scope.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/overloads.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/handle.hpp>

//#include <numpy/npy_common.h>

#include "core/utctime_utilities.h"
#include "core/time_axis.h"

char const* version() {
   return "v1.0";
}


//typedef std::vector<utcperiod> UtcPeriodVector;
using namespace shyft::core;
using namespace boost::python;
using namespace std;



void def_Timeaxis() {
    using namespace shyft::time_axis;
    class_<fixed_dt>("Timeaxis","timeaxis doc")
        .def(init<utctime,utctimespan,long>(args("start","delta_t","n"),"creates a timeaxis with n intervals, fixed delta_t, starting at start"))
        //.def(init<npy_int64,npy_int64,npy_int64>(args("start","delta_t","n"),"creates a timeaxis with n intervals, fixed delta_t, starting at start"))
        .def("size",&fixed_dt::size,"returns number of intervals")
        .def_readonly("n",&fixed_dt::n,"number of periods")
        .def_readonly("start",&fixed_dt::t,"start of the time-axis")
        .def_readonly("delta_t",&fixed_dt::dt,"timespan of each interval")
        .def("total_period",&fixed_dt::total_period,"the period that covers the entire time-axis")
        .def("time",&fixed_dt::time,args("i"),"return the start of the i'th period of the time-axis")
        .def("period",&fixed_dt::period,args("i"),"return the i'th period of the time-axis")
        .def("index_of",&fixed_dt::index_of,args("t"),"return the index the time-axis period that contains t")
        .def("open_range_index_of",&fixed_dt::open_range_index_of,args("t"),"returns the index that contains t, or is before t")
        .def("full_range",&fixed_dt::full_range,"returns a timeaxis that covers [-oo..+oo> ").staticmethod("full_range")
        .def("null_range",&fixed_dt::null_range,"returns a null timeaxis").staticmethod("null_range");
}
void def_PointTimeaxis() {
    using namespace shyft::time_axis;
    class_<point_dt>("PointTimeaxis","timeaxis doc")
        .def(init<const vector<utctime>&,utctime>(args("time_points","t_end"),"creates a time-axis with n intervals using time-points plus the end-points"))
        .def(init<const vector<utctime>& >(args("time_points"),"create a time-axis supplying n+1 points to define n intervals"))
        .def("size",&point_dt::size,"returns number of intervals")
        .def_readonly("t",&point_dt::t,"timepoints except last")
        .def_readonly("t_end",&point_dt::t,"end of time-axis")
        //.def_readonly("delta_t",&point_dt::dt,"timespan of each interval")
        .def("total_period",&point_dt::total_period,"the period that covers the entire time-axis")
        .def("time",&point_dt::time,args("i"),"return the start of the i'th period of the time-axis")
        .def("period",&point_dt::period,args("i"),"return the i'th period of the time-axis")
        .def("index_of",&point_dt::index_of,args("t"),"return the index the time-axis period that contains t")
        .def("open_range_index_of",&point_dt::open_range_index_of,args("t"),"returns the index that contains t, or is before t")
        ;//.def("full_range",&point_dt::full_range,"returns a timeaxis that covers [-oo..+oo> ").staticmethod("full_range")
        //.def("null_range",&point_dt::null_range,"returns a null timeaxis").staticmethod("null_range");
}

void def_CalendarTimeaxis() {
    using namespace shyft::time_axis;
    class_<calendar_dt>("CalendarTimeaxis","timeaxis doc")
        .def(init<shared_ptr<const calendar>,utctime,utctimespan,long>(args("calendar","start","delta_t","n"),"creates a calendar timeaxis with n intervals, fixed calendar delta_t, starting at start"))
        .def("size",&calendar_dt::size,"returns number of intervals")
        .def_readonly("n",&calendar_dt::n,"number of periods")
        .def_readonly("start",&calendar_dt::t,"start of the time-axis")
        .def_readonly("delta_t",&calendar_dt::dt,"timespan of each interval")
        .def_readonly("calendar",&calendar_dt::cal,"calendar of the time-axis")
        .def("size",&calendar_dt::size,"returns number of intervals")
        .def("total_period",&calendar_dt::total_period,"the period that covers the entire time-axis")
        .def("time",&calendar_dt::time,args("i"),"return the start of the i'th period of the time-axis")
        .def("period",&calendar_dt::period,args("i"),"return the i'th period of the time-axis")
        .def("index_of",&calendar_dt::index_of,args("t"),"return the index the time-axis period that contains t")
        .def("open_range_index_of",&calendar_dt::open_range_index_of,args("t"),"returns the index that contains t, or is before t")
        ;//.def("full_range",&point_dt::full_range,"returns a timeaxis that covers [-oo..+oo> ").staticmethod("full_range")
        //.def("null_range",&point_dt::null_range,"returns a null timeaxis").staticmethod("null_range");
}


void def_api() {
    void def_GeoPoint();
    void def_GeoCellData();
    void def_utctime();
    void def_Calendar();
    void def_UtcPeriod();
    void def_vectors();
    def_utctime();
    def_Calendar();
    def_UtcPeriod();
    def_vectors();
    def_Timeaxis();
    def_PointTimeaxis();
    def_CalendarTimeaxis();
    def_GeoPoint();
    def_GeoCellData();
}

BOOST_PYTHON_MODULE(_api)
{

    scope().attr("__doc__")="SHyFT python api providing basic types";
    def("version", version);
    def_api();
}
