
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

#include "core/utctime_utilities.h"

char const* version() {
   return "v1.0";
}


//typedef std::vector<utcperiod> UtcPeriodVector;
using namespace shyft::core;
using namespace boost::python;
using namespace std;

void def_std_vectors() {

    typedef std::vector<std::string> StringVector;
    typedef std::vector<int> IntVector;
    typedef std::vector<double> DoubleVector;
    typedef std::vector<utctime> UtcTimeVector;
    class_<StringVector>("StringVector","Is a list of strings")
        .def(vector_indexing_suite<StringVector>());
    class_<IntVector>("IntVector")
        .def(vector_indexing_suite<IntVector>());
    class_<UtcTimeVector>("UtcTimeVector")
        .def(vector_indexing_suite<UtcTimeVector>());
    class_<DoubleVector>("DoubleVector")
        .def(vector_indexing_suite<DoubleVector>());
}
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(calendar_time_overloads,calendar::time,1,6);

void def_Calendar() {
    def_std_vectors();
    std::string (shyft::core::calendar::*to_string_t)(shyft::core::utctime) const= &calendar::to_string;//selects correct ptr.
    std::string (calendar::*to_string_p)(utcperiod) const =&calendar::to_string;
    utctimespan (calendar::*diff_units)(utctime,utctime,utctimespan) const=&calendar::diff_units;
    utctime (calendar::*time_YMDhms)(YMDhms) const = &calendar::time;
    utctime (calendar::*time_6)(int,int,int,int,int,int) const = &calendar::time;

    class_<calendar,shared_ptr<calendar>>("Calendar",
        "Calendar deals with the concept of human calendar\n"
        " In SHyFT we practice the 'utctime-perimeter' principle,\n"
        "  * so the core is utc-time only \n"
        "  * we deal with time-zone and calendars at the interfaces/perimeters\n"
        "\n"
        " Please notice that although the calendar concept is complete\n"
        " we only implement features as needed in the core and interfaces\n"
        " Currently this includes most options, including olson time-zone handling\n"
        "Calendar functionality:\n"
        " -# Conversion between the calendar coordinates YMDhms and utctime, taking  any timezone and DST into account\n"
        " -# Calendar constants, utctimespan like values for Year,Month,Week,Day,Hour,Minute,Second\n"
        " -# Calendar arithmetic, like adding calendar units, e.g. day,month,year etc.\n"
        " -# Calendar arithmetic, like trim/truncate a utctime down to nearest timespan/calendar unit. eg. day\n"
        " -# Calendar arithmetic, like calculate difference in calendar units(e.g days) between two utctime points\n"
        " -# Calendar Timezone and DST handling\n"
        " -# Converting utctime to string and vice-versa\n"
        "\n"
        )
    .def(init<utctimespan>(args("tz-offset"),"creates a calendar with constant tz-offset"))
    .def(init<string>(args("olson tz-id"),"create a Calendar from Olson timezone id, eg. 'Europe/Oslo'" ))
    .def("to_string",to_string_t,args("utctime"),"convert time t to readable string taking current calendar properties, including timezone into account")
    .def("to_string",to_string_p,args("utcperiod"),"convert utcperiod p to readable string taking current calendar properties, including timezone into account")
    .def("region_id_list",&calendar::region_id_list,"Returns a list over predefined olson time-zone identifiers").staticmethod("region_id_list")
    .def("calendar_units",&calendar::calendar_units,args("t"),"returns YMDhms for specified t, in the time-zone as given by the calendar")
    .def("time",time_YMDhms,args("YMDhms"),"convert calendar coordinates into time using the calendar time-zone")

    .def("time",time_6,calendar_time_overloads("returns time accoring to calendar",args("Y,M,D,h,m,s")))//args("Y"),"returns time of Y.01.01 in calendar time-zone")

    .def("add",&calendar::add,args("t,delta_t,n"),
         "calendar semantic add\n"
         " conceptually this is similar to t + deltaT*n\n"
         " but with deltaT equal to calendar::DAY,WEEK,MONTH,YEAR\n"
         " and/or with dst enabled time-zone the variation of length due to dst\n"
         " or month/year length is taken into account\n"
         " e.g. add one day, and calendar have dst, could give 23,24 or 25 hours due to dst.\n"
         " similar for week or any other time steps.\n"
         )
    .def("diff_units",diff_units,args("t,delta_t,n"),
         "calculate the distance t1..t2 in specified units\n"
         " The function takes calendar semantics when deltaT is calendar::DAY,WEEK,MONTH,YEAR,\n"
         " and in addition also dst.\n"
         "e.g. the diff_units of calendar::DAY over summer->winter shift is 1, remainder is 0,\n"
         "even if the number of hours during those days are 23 and 25 summer and winter transition respectively\n"
         "returns: calendar semantics of (t2-t1)/deltaT, where deltaT could be calendar units DAY,WEEK,MONTH,YEAR"
         )
    .def("trim",&calendar::trim,args("t,delta_t"),"round down t to nearest calendar time-unit delta_t")
    .def_readonly("YEAR",&calendar::YEAR)
    .def_readonly("MONTH",&calendar::MONTH)
    .def_readonly("DAY",&calendar::DAY)
    .def_readonly("WEEK",&calendar::WEEK)
    .def_readonly("HOUR",&calendar::HOUR)
    .def_readonly("MINUTE",&calendar::MINUTE)
    .def_readonly("SECOND",&calendar::SECOND);


    class_<YMDhms>("YMDhms","Defines calendar coordinates as Year Month Day hour minute second")
    .def(init<int>(args("Y"),"creates coordinates specifiying Year, resulting in YMDhms(Y,1,1,0,0,0)"))
    .def(init<int,int>(args("Y,M"),"creates coordinates specifiying Year, resulting in YMDhms(Y,M,1,0,0,0)"))
    .def(init<int,int,int>(args("Y,M,D"),"creates coordinates specifiying Year, resulting in YMDhms(Y,M,D,0,0,0)"))
    .def(init<int,int,int,int>(args("Y,M,D,h"),"creates coordinates specifiying Year, resulting in YMDhms(Y,M,D,h,0,0)"))
    .def(init<int,int,int,int,int>(args("Y,M,D,h,m"),"creates coordinates specifiying Year, resulting in YMDhms(Y,M,D,h,m,0)"))
    .def(init<int,int,int,int,int,int>(args("Y,M,D,h,m,s"),"creates coordinates specifiying Year, resulting in YMDhms(Y,M,D,h,m,s)"))
    .def("is_valid",&YMDhms::is_valid,"returns true if YMDhms values are reasonable")
    .def("is_null",&YMDhms::is_null,"returns true if all values are 0, - the null definition")
    .def_readwrite("year",&YMDhms::year)
    .def_readwrite("month",&YMDhms::month)
    .def_readwrite("day",&YMDhms::day)
    .def_readwrite("hour",&YMDhms::hour)
    .def_readwrite("minute",&YMDhms::minute)
    .def_readwrite("second",&YMDhms::second)
    .def("max",&YMDhms::max,"returns the maximum representation").staticmethod("max")
    .def("min",&YMDhms::max,"returns the minimum representation").staticmethod("min");
}

void def_UtcPeriod() {
    bool (utcperiod::*contains_t)(utctime) const = &utcperiod::contains;
    bool (utcperiod::*contains_p)(const utcperiod&) const = &utcperiod::contains;
    class_<utcperiod>("UtcPeriod","UtcPeriod defines the open utctime range [start..end> \nwhere end is required to be equal or greater than start")
    .def(init<utctime,utctime>(args("start,end"),"Create utcperiod given start and end"))
    .def("valid",&utcperiod::valid,"returns true if start<=end otherwise false")
    .def("contains",contains_t,args("t"),"returns true if utctime t is contained in this utcperiod" )
    .def("contains",contains_p,args("p"),"returns true if utcperiod p is contained in this utcperiod" )
    .def("overlaps",&utcperiod::overlaps,args("p"), "returns true if period p overlaps this utcperiod" )
    .def("__str__",&utcperiod::to_string,"returns the str using time-zone utc to convert to readable time")
    .def("timespan",&utcperiod::timespan,"returns end-start, the timespan of the period")
    .def_readwrite("start",&utcperiod::start,"Defines the start of the period, inclusive")
    .def_readwrite("end",&utcperiod::end,"Defines the end of the period, not inclusive");
    def("intersection",&intersection,args("a,b"),"Returns the intersection of two utcperiods");
}
void def_utctime_utilities() {
    def("utctime_now",utctime_now,"returns utc-time now as seconds since 1970s");
    scope current;
    current.attr("max_utctime")= max_utctime;
    current.attr("min_utctime")= min_utctime;
    current.attr("no_utctime")=no_utctime;
    def_Calendar();
    def_UtcPeriod();
}


BOOST_PYTHON_MODULE(api)
{

    scope().attr("__doc__")="SHyFT python api providing basic types";
    def("version", version);
    def_utctime_utilities();
}
