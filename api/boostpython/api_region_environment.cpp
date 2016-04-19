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
            .def(py::init<py::optional<sc::geo_point,sa::apoint_ts >>(py::args("midpoint","ts")))
            ;
        typedef std::vector<T> TSourceVector;
        py::class_<TSourceVector,py::bases<>,std::shared_ptr<TSourceVector> > (py_vector)
            .def(py::vector_indexing_suite<TSourceVector>())
            .def(py::init<const TSourceVector&>(py::args("src"),"clone src"))
            ;
        py::register_ptr_to_python<std::shared_ptr<TSourceVector> >();
    }

    static void GeoPointSource(void) {
        py::class_<sa::GeoPointSource>("GeoPointSource",
            "GeoPointSource contains common properties, functions\n"
            "for the point sources in SHyFT.\n"
            "Typically it contains a GeoPoint (3d position), plus a time-series\n"
            )
            .def(py::init<py::optional<sc::geo_point,sa::apoint_ts >>(py::args("midpoint","ts")))
            .def_readwrite("mid_point_",&sa::GeoPointSource::mid_point_,"reference to internal mid_point")
            .def("mid_point",&sa::GeoPointSource::mid_point,"returns a copy of mid_point")
            // does not work  yet:.def_readwrite("ts",&sa::GeoPointSource::ts)
            .add_property("ts",&sa::GeoPointSource::get_ts,&sa::GeoPointSource::set_ts)
            ;
        py::register_ptr_to_python<std::shared_ptr<sa::GeoPointSource>>();
        GeoPointSourceX<sa::TemperatureSource>("TemperatureSource","TemperatureSourceVector","geo located temperatures[deg Celcius]");
        GeoPointSourceX<sa::PrecipitationSource>("PrecipitationSource","PrecipitationSourceVector","geo located precipitation[mm/h]");
        GeoPointSourceX<sa::WindSpeedSource>("WindSpeedSource","WindSpeedSourceVector","geo located wind speeds[m/s]");
        GeoPointSourceX<sa::RelHumSource>("RelHumSource","RelHumSourceVector","geo located relative humidity[%rh], range 0..1");
        GeoPointSourceX<sa::RadiationSource>("RadiationSource","RadiationSourceVector","geo located radiation[W/m2]");
    }

    static void a_region_environment() {
        //SiH: Here I had trouble using def_readwrite(), the getter was not working as expected, the setter did the right thing
        //     work-around was to use add_property with explicit set_ get_ methods that returned  shared_ptr to vectors
        py::class_<sa::a_region_environment>("ARegionEnvironment","Contains all geo-located sources to be used by a SHyFT core model")
            .add_property("temperature",&sa::a_region_environment::get_temperature,&sa::a_region_environment::set_temperature)
            .add_property("precipitation",&sa::a_region_environment::get_precipitation,&sa::a_region_environment::set_precipitation)
            .add_property("wind_speed",&sa::a_region_environment::get_wind_speed,&sa::a_region_environment::set_wind_speed)
            .add_property("rel_hum",&sa::a_region_environment::get_rel_hum,&sa::a_region_environment::set_rel_hum)
            .add_property("radiation",&sa::a_region_environment::get_radiation,&sa::a_region_environment::set_radiation)
            ;
    }

    void region_environment() {
        GeoPointSource();
        a_region_environment();
    }
}
