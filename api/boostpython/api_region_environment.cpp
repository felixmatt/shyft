#include "boostpython_pch.h"
#include "py_convertible.h"


#include "core/utctime_utilities.h"
#include "core/time_axis.h"
#include "core/timeseries.h"
#include "api/api.h"

namespace expose {
    namespace py=boost::python;
    namespace sc=shyft::core;
    namespace ts=shyft::timeseries;
    namespace ta=shyft::time_axis;
    namespace sa=shyft::api;

    template<class T>
    static void GeoPointSourceX(const char *py_name,const char *py_vector,const char *py_doc) {
        py::class_<T,py::bases<sa::GeoPointSource>>(py_name,py_doc)
            .def(py::init<py::optional<sc::geo_point,sa::ITimeSeriesOfPoints_ >>(py::args("midpoint","ts")))
            ;
        typedef std::vector<T> TSourceVector;
        py::class_<TSourceVector,py::bases<>,std::shared_ptr<TSourceVector> > (py_vector)
            .def(py::vector_indexing_suite<TSourceVector>())
            ;
        py::register_ptr_to_python<std::shared_ptr<TSourceVector> >();
    }

    static void GeoPointSource(void) {
        py::class_<sa::GeoPointSource>("GeoPointSource",
            "GeoPointSource contains common properties, functions\n"
            "for the point sources in SHyFT.\n"
            "Typically it contains a GeoPoint (3d position), plus a time-series\n"
            )
            .def(py::init<py::optional<sc::geo_point,sa::ITimeSeriesOfPoints_ >>(py::args("midpoint","ts")))
            .def_readwrite("mid_point_",&sa::GeoPointSource::mid_point_,"reference to internal mid_point")
            .def("mid_point",&sa::GeoPointSource::mid_point,"returns a copy of mid_point")
            .def_readwrite("ts",&sa::GeoPointSource::ts)
            ;

        GeoPointSourceX<sa::TemperatureSource>("TemperatureSource","TemperatureSourceVector","geo located temperatures[deg Celcius]");
        GeoPointSourceX<sa::PrecipitationSource>("PrecipitationSource","PrecipitationSourceVector","geo located precipitation[mm/h]");
        GeoPointSourceX<sa::WindSpeedSource>("WindSpeedSource","WindSpeedSourceVector","geo located wind speeds[m/s]");
        GeoPointSourceX<sa::RelHumSource>("RelHumSource","RelHumSourceVector","geo located relative humidity[%rh], range 0..1");
        GeoPointSourceX<sa::RadiationSource>("RadiationSource","RadiationSourceVector","geo located radiation[W/m2]");
    }

    static void a_region_environment() {
        py::class_<sa::a_region_environment>("ARegionEnvironment","Contains all geo-located sources to be used by a SHyFT core model")
            .def_readwrite("temperature",&sa::a_region_environment::temperature)
            .def_readwrite("precipitation",&sa::a_region_environment::precipitation)
            .def_readwrite("radiation",&sa::a_region_environment::radiation)
            .def_readwrite("wind_speed",&sa::a_region_environment::wind_speed)
            .def_readwrite("rel_hum",&sa::a_region_environment::rel_hum)
            ;
    }

    void region_environment() {
        GeoPointSource();
        a_region_environment();
    }
}
