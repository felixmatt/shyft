
//#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
//#include <boost/python/scope.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
//#include <boost/python/return_internal_reference.hpp>
//#include <boost/python/return_value_policy.hpp>
//#include <boost/python/copy_const_reference.hpp>
//#include <boost/python/operators.hpp>
//#include <boost/python/overloads.hpp>
#include <boost/python/enum.hpp>
//#include <boost/operators.hpp>
#include "py_convertible.h"

#include "core/core_pch.h"
#include "core/utctime_utilities.h"
#include "core/time_axis.h"
#include "core/timeseries.h"
#include "api/api.h"


//using namespace shyft;
//using namespace shyft::core;
//using namespace boost::python;
//using namespace std;
    namespace py=boost::python;
    namespace sc=shyft::core;
    namespace ts=shyft::timeseries;
    namespace ta=shyft::time_axis;
    namespace sa=shyft::api;

template<class T>
static void def_GeoPointSourceX(const char *py_name,const char *py_vector,const char *py_doc) {
    py::class_<T,py::bases<sa::GeoPointSource>>(py_name,py_doc)
        .def(py::init<py::optional<sc::geo_point,sa::ITimeSeriesOfPoints_ >>(py::args("midpoint","ts")))
        ;
    typedef std::vector<T> TSourceVector;
    py::class_<TSourceVector,py::bases<>,std::shared_ptr<TSourceVector> > (py_vector)
        .def(py::vector_indexing_suite<TSourceVector>())
        ;

}

static void def_GeoPointSource(void) {
    py::class_<sa::GeoPointSource>("GeoPointSource",
        "GeoPointSource contains common properties, functions\n"
        "for the point sources in SHyFT.\n"
        "Typically it contains a GeoPoint (3d position), plus a time-series\n"
        )
        .def(py::init<py::optional<sc::geo_point,sa::ITimeSeriesOfPoints_ >>(py::args("midpoint","ts")))
        .def_readwrite("mid_point",&sa::GeoPointSource::mid_point_)
        .def_readwrite("ts",&sa::GeoPointSource::ts)
        ;

    def_GeoPointSourceX<sa::TemperatureSource>("TemperatureSource","TemperatureSourceVector","geo located temperatures[deg Celcius]");
    def_GeoPointSourceX<sa::PrecipitationSource>("PrecipitationSource","PrecipitationSourceVector","geo located precipitation[mm/h]");
    def_GeoPointSourceX<sa::WindSpeedSource>("WindSpeedSource","WindSpeedSourceVector","geo located wind speeds[m/s]");
    def_GeoPointSourceX<sa::RelHumSource>("RelHumSource","RelHumSourceVector","geo located relative humidity[%rh], range 0..1");
    def_GeoPointSourceX<sa::RadiationSource>("RadiationSource","RadiationSourceVector","geo located radiation[W/m2]");

}
static void def_a_region_environment() {
    py::class_<sa::a_region_environment>("ARegionEnvironment","Contains all geo-located sources to be used by a SHyFT core model")
        .def_readwrite("temperature",&sa::a_region_environment::temperature)
        .def_readwrite("precipitation",&sa::a_region_environment::precipitation)
        .def_readwrite("radiation",&sa::a_region_environment::radiation)
        .def_readwrite("wind_speed",&sa::a_region_environment::wind_speed)
        .def_readwrite("rel_hum",&sa::a_region_environment::rel_hum)
        ;
}

void def_region_environment() {
    def_GeoPointSource();
    def_a_region_environment();
}
