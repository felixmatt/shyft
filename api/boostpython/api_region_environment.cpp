#include "boostpython_pch.h"
#include "py_convertible.h"


#include "core/utctime_utilities.h"
#include "core/time_axis.h"
#include "core/time_series.h"
#include "api/api.h"

namespace expose {
    namespace py=boost::python;
    namespace sc=shyft::core;
    namespace ts=shyft::time_series;
    namespace ta=shyft::time_axis;
    namespace sa=shyft::api;

    template <class S>
    static std::vector<double> geo_tsv_values(std::shared_ptr<std::vector<S>> const &geo_tsv, sc::utctime t) {
        std::vector<double> r;
        if (geo_tsv) {
            r.reserve(geo_tsv->size());
            for (auto const &gts : *geo_tsv)
                r.push_back(gts.ts(t));
        }
        return r;
    }

    template<class T>
    static std::vector<T> create_from_geo_and_tsv(const std::vector<sc::geo_point>& gpv, const sa::ats_vector& tsv) {
        if(gpv.size()!=tsv.size())
            throw std::runtime_error("list of geo-points and time-series must have equal length");
        std::vector<T> r;r.reserve(tsv.size());
        for(size_t i=0;i<tsv.size();++i)
            r.emplace_back(gpv[i],tsv[i]);
        return r;
    }

    template<class T>
    static void GeoPointSourceX(const char *py_name,const char *py_vector,const char *py_doc) {
        py::class_<T,py::bases<sa::GeoPointSource>>(py_name,py_doc)
            .def(py::init<const sc::geo_point&,const sa::apoint_ts &>((py::arg("midpoint"),py::arg("ts"))))
            ;
        typedef std::vector<T> TSourceVector;
        py::class_<TSourceVector,py::bases<>,std::shared_ptr<TSourceVector> > (py_vector)
            .def(py::vector_indexing_suite<TSourceVector>())
            .def(py::init<const TSourceVector&>(py::args("src"),"clone src"))
            .def("from_geo_and_ts_vector",&create_from_geo_and_tsv<T>,(py::arg("geo_points"), py::arg("tsv")),
                doc_intro("Create from a geo_points and corresponding ts-vectors")
                doc_parameters()
                doc_parameter("geo_points","GeoPointVector","the geo-points")
                doc_parameter("tsv","TsVector","the corresponding time-series located at corresponding geo-point")
                doc_returns("src_vector","SourceVector","a newly created geo-located vector of specified type")
            )
            .staticmethod("from_geo_and_ts_vector")
            ;

        py::def("compute_geo_ts_values_at_time", &geo_tsv_values<T>, py::args("geo_ts_vector", "t"),
            doc_intro("compute the ts-values of the GeoPointSourceVector type for the specified time t and return DoubleVector")
            doc_parameters()
            doc_parameter("geo_ts_vector", "GeoPointSourceVector", "Any kind of GeoPointSource vector")
            doc_parameter("t", "int", "timestamp in utc seconds since epoch")
            doc_returns("values", "DoubleValue", "List of extracted values at same size/position as the geo_ts_vector")
        );
    }

    static void GeoPointSource(void) {
        typedef std::vector<sa::GeoPointSource> GeoPointSourceVector;

        py::class_<sa::GeoPointSource>("GeoPointSource",
            "GeoPointSource contains common properties, functions\n"
            "for the point sources in SHyFT.\n"
            "Typically it contains a GeoPoint (3d position), plus a time-series\n"
            )
            .def(py::init<const sc::geo_point&,const sa::apoint_ts&>( (py::arg("midpoint"),py::arg("ts"))  ) )
            .def_readwrite("mid_point_",&sa::GeoPointSource::mid_point_,"reference to internal mid_point")
            .def("mid_point",&sa::GeoPointSource::mid_point,(py::arg("self")),"returns a copy of mid_point")
            // does not work  yet:.def_readwrite("ts",&sa::GeoPointSource::ts)
            .add_property("ts",&sa::GeoPointSource::get_ts,&sa::GeoPointSource::set_ts)
			.def_readwrite("uid",&sa::GeoPointSource::uid,"user specified identifier, string")
            ;

        py::class_<GeoPointSourceVector,py::bases<>,std::shared_ptr<GeoPointSourceVector> > ("GeoPointSourceVector")
            .def(py::vector_indexing_suite<GeoPointSourceVector>())
            .def(py::init<const GeoPointSourceVector&>(py::args("src"),"clone src"))
            ;
        py::def("compute_geo_ts_values_at_time", &geo_tsv_values<sa::GeoPointSource>, py::args("geo_ts_vector", "t"),
            doc_intro("compute the ts-values of the GeoPointSourceVector for the specified time t and return DoubleVector")
            doc_parameters()
            doc_parameter("geo_ts_vector","GeoPointSourceVector","Any kind of GeoPointSource vector")
            doc_parameter("t","int","timestamp in utc seconds since epoch")
            doc_returns("values","DoubleValue","List of extracted values at same size/position as the geo_ts_vector")
            );

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
