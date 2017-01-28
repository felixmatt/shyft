#include "boostpython_pch.h"


#include "core/utctime_utilities.h"
#include "core/time_axis.h"

namespace expose {
    namespace time_axis {
        using namespace shyft::core;
        using namespace boost::python;
        using namespace std;

        static void e_fixed_dt() {
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
        static void e_point_dt() {
            using namespace shyft::time_axis;
            class_<point_dt>("PointTimeaxis","timeaxis doc")
                .def(init<const vector<utctime>&,utctime>(args("time_points","t_end"),"creates a time-axis with n intervals using time-points plus the end-points"))
                .def(init<const vector<utctime>& >(args("time_points"),"create a time-axis supplying n+1 points to define n intervals"))
                .def("size",&point_dt::size,"returns number of intervals")
                .def_readonly("t",&point_dt::t,"timepoints except last")
                .def_readonly("t_end",&point_dt::t_end,"end of time-axis")
                //.def_readonly("delta_t",&point_dt::dt,"timespan of each interval")
                .def("total_period",&point_dt::total_period,"the period that covers the entire time-axis")
                .def("time",&point_dt::time,args("i"),"return the start of the i'th period of the time-axis")
                .def("period",&point_dt::period,args("i"),"return the i'th period of the time-axis")
                .def("index_of",&point_dt::index_of,args("t"),"return the index the time-axis period that contains t")
                .def("open_range_index_of",&point_dt::open_range_index_of,args("t"),"returns the index that contains t, or is before t")
                ;//.def("full_range",&point_dt::full_range,"returns a timeaxis that covers [-oo..+oo> ").staticmethod("full_range")
                //.def("null_range",&point_dt::null_range,"returns a null timeaxis").staticmethod("null_range");
        }

        static void e_calendar_dt() {
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
        static void e_generic_dt() {
            using namespace shyft::time_axis;

            enum_<generic_dt::generic_type>("TimeaxisType")
            .value("FIXED",generic_dt::generic_type::FIXED)
            .value("CALENDAR",generic_dt::generic_type::CALENDAR)
            .value("POINT",generic_dt::generic_type::POINT)
            .export_values()
            ;

            class_<generic_dt>("TimeAxis","A timeaxis that internally could be a (Fixed delta) TimeAxis, a CalendarTimeAxis or Point TimeAxis")
                .def(init<utctime,utctimespan,long>(args("start","delta_t","n"),"creates a timeaxis with n intervals, fixed delta_t, starting at start"))
                .def(init<shared_ptr<calendar>,utctime,utctimespan,long>(args("calendar","start","delta_t","n"),"creates a calendar timeaxis with n intervals, fixed calendar delta_t, starting at start"))
                .def(init<const vector<utctime>&,utctime>(args("time_points","t_end"),"creates a time-axis with n intervals using time-points plus the end-points"))
                .def(init<const vector<utctime>& >(args("time_points"),"create a time-axis supplying n+1 points to define n intervals"))
                .def(init<const calendar_dt&>(args("calendar_dt"),"creates a generic from a calendar timeaxis"))
                .def(init<const fixed_dt&>(args("fixed_dt"),"creates a generic from a calendar a fixed_dt timeaxis"))
                .def(init<const point_dt&>(args("point_dt"),"creates a generic from a calendar a point_dt timeaxis"))
                .def("size",&calendar_dt::size,"returns number of intervals")
                .def_readonly("timeaxis_type",&generic_dt::gt,"describes what time-axis representation type this is,e.g (fixed|calendar|point)_dt ")
                .def_readonly("fixed_dt",&generic_dt::f,"The fixed dt representation (if active)")
                .def_readonly("calendar_dt",&generic_dt::c,"The calendar dt representation(if active)")
                .def_readonly("point_dt",&generic_dt::p,"The point_dt representation(if active)")
                .def("size",&generic_dt::size,"returns number of intervals")
                .def("total_period",&generic_dt::total_period,"the period that covers the entire time-axis")
                .def("time",&generic_dt::time,args("i"),"return the start of the i'th period of the time-axis")
                .def("period",&generic_dt::period,args("i"),"return the i'th period of the time-axis")
                .def("index_of",&generic_dt::index_of,args("t"),"return the index the time-axis period that contains t")
                .def("open_range_index_of",&generic_dt::open_range_index_of,args("t"),"returns the index that contains t, or is before t")
                ;//.def("full_range",&point_dt::full_range,"returns a timeaxis that covers [-oo..+oo> ").staticmethod("full_range")
                //.def("null_range",&point_dt::null_range,"returns a null timeaxis").staticmethod("null_range");
        }

    }
    void api_time_axis() {
        time_axis::e_fixed_dt();
        time_axis::e_point_dt();
        time_axis::e_calendar_dt();
        time_axis::e_generic_dt();
    }

}
