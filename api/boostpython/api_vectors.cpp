#include "boostpython_pch.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "numpy_boost_python.hpp"

#include "py_convertible.h"
#include "core/utctime_utilities.h"
#include "core/geo_point.h"
#include "core/geo_cell_data.h"
#include "api/time_series.h"
#include "api/api.h"

namespace expose {
    using namespace shyft::core;
    using namespace boost::python;
    using namespace std;
    namespace sa=shyft::api;
    namespace sc=shyft::core;
    namespace ts=shyft::time_series;
    namespace py=boost::python;

    static void* np_import() {
        import_array();
        return nullptr;
    }

    sa::ats_vector create_tsv_from_np(const sa::gta_t& ta, const numpy_boost<double,2>& a ,ts::ts_point_fx point_fx) {
        sa::ats_vector r;
        size_t n_ts = a.shape()[0];
        size_t n_pts = a.shape()[1];
        if(ta.size() != n_pts)
            throw std::runtime_error("time-axis should have same length as second dim in numpy array");
        r.reserve(n_ts);
        for(size_t i=0;i<n_ts;++i) {
            std::vector<double> v;v.reserve(n_pts);
            for(size_t j=0;j<n_pts;++j) v.emplace_back(a[i][j]);
            r.emplace_back(ta,v ,point_fx);
        }
        return r;
    }

    template <class S>
    vector<S> create_from_geo_tsv_from_np(const sa::gta_t& ta,const vector<sc::geo_point>&gpv ,const numpy_boost<double,2>& a ,ts::ts_point_fx point_fx) {
        vector<S> r;
        size_t n_ts = a.shape()[0];
        size_t n_pts = a.shape()[1];
        if(ta.size() != n_pts)
            throw std::runtime_error("time-axis should have same length as second dim in numpy array");
        if(n_ts != gpv.size())
            throw std::runtime_error("geo-point vector should have same size as first dim (n_ts) in numpy array");
        r.reserve(n_ts);
        for(size_t i=0;i<n_ts;++i) {
            std::vector<double> v;v.reserve(n_pts);
            for(size_t j=0;j<n_pts;++j) v.emplace_back(a[i][j]);
            r.emplace_back(gpv[i], sa::apoint_ts(ta,v ,point_fx));
        }
        return r;
    }

    template<class T>
    static vector<T> FromNdArray(const numpy_boost<T,1>& npv) {
        vector<T> r;r.reserve(npv.shape()[0]);
        for(size_t i=0;i<npv.shape()[0];++i) {
            r.push_back(npv[i]);
        }
        return r;
    }

    template<class T>
    static numpy_boost<T,1> ToNpArray(const vector<T>&v) {
        int dims[]={int(v.size())};
        numpy_boost<T,1> r(dims);
        for(size_t i=0;i<r.size();++i) {
            r[i]=v[i];
        }
        return r;
    }

    template <class T>
    static void expose_vector(const char *name) {
        typedef std::vector<T> XVector;

        class_<XVector>(name)
        .def(vector_indexing_suite<XVector>()) // meaning it get all it needs to appear as python list
        .def(init<const XVector&>(args("const_ref_v"))) // so we can copy construct
        .def("FromNdArray",FromNdArray<T>).staticmethod("FromNdArray") // BW compatible
        .def("from_numpy",FromNdArray<T>).staticmethod("from_numpy")// static construct from numpy TODO: fix __init__
        .def("to_numpy",ToNpArray<T>,"convert to numpy") // Ok, to make numpy 1-d arrays
        ;
        numpy_boost_python_register_type<T, 1>(); // register the numpy object so we can access it in C++
        py_api::iterable_converter().from_python<XVector>();
    }
    static void expose_str_vector(const char *name) {
        typedef std::vector<std::string> XVector;
        class_<XVector>(name)
            .def(vector_indexing_suite<XVector>()) // meaning it get all it needs to appear as python list
            .def(init<const XVector&>(args("const_ref_v"))) // so we can copy construct
            ;
        py_api::iterable_converter().from_python<XVector>();
    }

    typedef std::vector<shyft::core::geo_point> GeoPointVector;
    static GeoPointVector create_from_x_y_z_vectors(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double> z) {
        if(!(x.size()==y.size() && y.size()==z.size()))
            throw std::runtime_error("x,y,z vectors need to have same number of elements");
        GeoPointVector r(x.size());
        for(size_t i=0;i<x.size();++i)
            r.emplace_back(x[i],y[i],z[i]);
        return std::move(r);
    }

    static void expose_geo_point_vector() {

        class_<GeoPointVector>("GeoPointVector", "A vector, list, of GeoPoints")
            .def(vector_indexing_suite<GeoPointVector>())
            .def(init<const GeoPointVector&>(args("const_ref_v")))
            .def("create_from_x_y_z",create_from_x_y_z_vectors,args("x","y","z"),"Create a GeoPointVector from x,y and z DoubleVectors of equal length")
            .staticmethod("create_from_x_y_z")
            ;
    }
    static void expose_geo_cell_data_vector() {
        typedef std::vector<shyft::core::geo_cell_data> GeoCellDataVector;
        class_<GeoCellDataVector>("GeoCellDataVector", "A vector, list, of GeoCellData")
            .def(vector_indexing_suite<GeoCellDataVector>())
            .def(init<const GeoCellDataVector&>(args("const_ref_v")))
            ;
    }
    static void expose_ts_vector_create() {
        def("create_ts_vector_from_np_array",
            &create_tsv_from_np,(py::arg("time_axis"),py::arg("np_array"),py::arg("point_fx")),
            doc_intro("Create a TsVector from specified time_axis, 2-d np_array and point_fx.")
            doc_parameters()
            doc_parameter("time_axis","TimeAxis","time-axis that matches in length to 2nd dim of np_array")
            doc_parameter("np_array","np.ndarray","numpy array of dtype=np.float64, and shape(n_ts,n_points)")
            doc_parameter("point_fx","point interpretation", "one of POINT_AVERAGE_VALUE|POINT_INSTANT_VALUE")
            doc_returns("tsv","TsVector","a TsVector of length first np_array dim, n_ts, each with time-axis, values and point_fx")
        );

        def("create_temperature_source_vector_from_np_array",
            &create_from_geo_tsv_from_np<sa::TemperatureSource>,(py::arg("time_axis"),py::arg("geo_points"),py::arg("np_array"),py::arg("point_fx")),
            doc_intro("Create a TemperatureSourceVector from specified time_axis,geo_points, 2-d np_array and point_fx.")
            doc_parameters()
            doc_parameter("time_axis","TimeAxis","time-axis that matches in length to 2nd dim of np_array")
            doc_parameter("geo_points","GeoPointVector","the geo-positions for the time-series, should be of length n_ts")
            doc_parameter("np_array","np.ndarray","numpy array of dtype=np.float64, and shape(n_ts,n_points)")
            doc_parameter("point_fx","point interpretation", "one of POINT_AVERAGE_VALUE|POINT_INSTANT_VALUE")
            doc_returns("tsv","TemperatureSourceVector","a TemperatureSourceVector of length first np_array dim, n_ts, each with geo-point and time-series with time-axis, values and point_fx")
        );

        def("create_precipitation_source_vector_from_np_array",
            &create_from_geo_tsv_from_np<sa::PrecipitationSource>,(py::arg("time_axis"),py::arg("geo_points"),py::arg("np_array"),py::arg("point_fx")),
            doc_intro("Create a PrecipitationSourceVector from specified time_axis,geo_points, 2-d np_array and point_fx.")
            doc_parameters()
            doc_parameter("time_axis","TimeAxis","time-axis that matches in length to 2nd dim of np_array")
            doc_parameter("geo_points","GeoPointVector","the geo-positions for the time-series, should be of length n_ts")
            doc_parameter("np_array","np.ndarray","numpy array of dtype=np.float64, and shape(n_ts,n_points)")
            doc_parameter("point_fx","point interpretation", "one of POINT_AVERAGE_VALUE|POINT_INSTANT_VALUE")
            doc_returns("tsv","PrecipitationSourceVector","a PrecipitationSourceVector of length first np_array dim, n_ts, each with geo-point and time-series with time-axis, values and point_fx")
        );

        def("create_wind_speed_source_vector_from_np_array",
            &create_from_geo_tsv_from_np<sa::WindSpeedSource>,(py::arg("time_axis"),py::arg("geo_points"),py::arg("np_array"),py::arg("point_fx")),
            doc_intro("Create a WindSpeedSourceVector from specified time_axis,geo_points, 2-d np_array and point_fx.")
            doc_parameters()
            doc_parameter("time_axis","TimeAxis","time-axis that matches in length to 2nd dim of np_array")
            doc_parameter("geo_points","GeoPointVector","the geo-positions for the time-series, should be of length n_ts")
            doc_parameter("np_array","np.ndarray","numpy array of dtype=np.float64, and shape(n_ts,n_points)")
            doc_parameter("point_fx","point interpretation", "one of POINT_AVERAGE_VALUE|POINT_INSTANT_VALUE")
            doc_returns("tsv","WindSpeedSourceVector","a WindSpeedSourceVector of length first np_array dim, n_ts, each with geo-point and time-series with time-axis, values and point_fx")
        );

        def("create_rel_hum_source_vector_from_np_array",
            &create_from_geo_tsv_from_np<sa::RelHumSource>,(py::arg("time_axis"),py::arg("geo_points"),py::arg("np_array"),py::arg("point_fx")),
            doc_intro("Create a RelHumSourceVector from specified time_axis,geo_points, 2-d np_array and point_fx.")
            doc_parameters()
            doc_parameter("time_axis","TimeAxis","time-axis that matches in length to 2nd dim of np_array")
            doc_parameter("geo_points","GeoPointVector","the geo-positions for the time-series, should be of length n_ts")
            doc_parameter("np_array","np.ndarray","numpy array of dtype=np.float64, and shape(n_ts,n_points)")
            doc_parameter("point_fx","point interpretation", "one of POINT_AVERAGE_VALUE|POINT_INSTANT_VALUE")
            doc_returns("tsv","RelHumSourceVector","a RelHumSourceVector of length first np_array dim, n_ts, each with geo-point and time-series with time-axis, values and point_fx")
        );

        def("create_radiation_source_vector_from_np_array",
            &create_from_geo_tsv_from_np<sa::RadiationSource>,(py::arg("time_axis"),py::arg("geo_points"),py::arg("np_array"),py::arg("point_fx")),
            doc_intro("Create a RadiationSourceVector from specified time_axis,geo_points, 2-d np_array and point_fx.")
            doc_parameters()
            doc_parameter("time_axis","TimeAxis","time-axis that matches in length to 2nd dim of np_array")
            doc_parameter("geo_points","GeoPointVector","the geo-positions for the time-series, should be of length n_ts")
            doc_parameter("np_array","np.ndarray","numpy array of dtype=np.float64, and shape(n_ts,n_points)")
            doc_parameter("point_fx","point interpretation", "one of POINT_AVERAGE_VALUE|POINT_INSTANT_VALUE")
            doc_returns("tsv","RadiationSourceVector","a RadiationSourceVector of length first np_array dim, n_ts, each with geo-point and time-series with time-axis, values and point_fx")
        );
        numpy_boost_python_register_type<double, 2>();
    }

    void vectors() {
        np_import();
        expose_str_vector("StringVector");
        expose_vector<double>("DoubleVector");
        expose_vector<int>("IntVector");
        expose_vector<char>("ByteVector");
        expose_vector<utctime>("UtcTimeVector");
        expose_geo_point_vector();
        expose_geo_cell_data_vector();
        expose_ts_vector_create();
    }
}

