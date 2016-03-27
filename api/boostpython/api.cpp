#include "boostpython_pch.h"


char const* version() {
   return "v1.0";
}
namespace expose {
    extern void api_geo_point();
    extern void api_geo_cell_data();
    extern void calendar_and_time();
    extern void vectors();
    extern void api_time_axis();
    extern void timeseries();
    extern void target_specification();
    extern void region_environment() ;
    extern void priestley_taylor();
    extern void actual_evapotranspiration();
    extern void gamma_snow();
    extern void kirchner();
    extern void precipitation_correction();
    extern void hbv_snow();
    extern void cell_environment();
    extern void interpolation();

    void api() {
        calendar_and_time();
        vectors();
        api_time_axis();
        api_geo_point();
        api_geo_cell_data();
        timeseries();
        target_specification();
        region_environment();
        priestley_taylor();
        actual_evapotranspiration();
        gamma_snow();
        kirchner();
        precipitation_correction();
        hbv_snow();
        cell_environment();
        interpolation();
    }
}

BOOST_PYTHON_MODULE(_api)
{
    boost::python::scope().attr("__doc__")="SHyFT python api providing basic types";
    boost::python::def("version", version);
    expose::api();
}
