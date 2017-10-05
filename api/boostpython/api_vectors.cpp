#include "boostpython_pch.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "numpy_boost_python.hpp"

#include "py_convertible.h"
#include "core/utctime_utilities.h"
#include "core/geo_point.h"
#include "core/geo_cell_data.h"

namespace expose {
    using namespace shyft::core;
    using namespace boost::python;
    using namespace std;

    static void* np_import() {
        import_array();
        return nullptr;
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

    void vectors() {
        np_import();
        expose_str_vector("StringVector");
        expose_vector<double>("DoubleVector");
        expose_vector<int>("IntVector");
        expose_vector<char>("ByteVector");
        expose_vector<utctime>("UtcTimeVector");
        expose_geo_point_vector();
        expose_geo_cell_data_vector();
    }
}

