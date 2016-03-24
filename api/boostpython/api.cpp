
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

using namespace shyft::core;
using namespace boost::python;
using namespace std;


void def_api() {
    void def_GeoPoint();
    void def_GeoCellData();
    void def_utctime();
    void def_Calendar();
    void def_UtcPeriod();
    void def_vectors();
    void def_Timeaxis();
    void def_PointTimeaxis();
    void def_CalendarTimeaxis();
    void def_timeseries();
    void def_target_specification();
    def_utctime();
    def_Calendar();
    def_UtcPeriod();
    def_vectors();
    def_Timeaxis();
    def_PointTimeaxis();
    def_CalendarTimeaxis();
    def_GeoPoint();
    def_GeoCellData();
    def_timeseries();
    def_target_specification();
}

BOOST_PYTHON_MODULE(_api)
{

    scope().attr("__doc__")="SHyFT python api providing basic types";
    def("version", version);
    def_api();
}
