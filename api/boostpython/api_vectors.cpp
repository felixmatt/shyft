#include "boostpython_pch.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy_boost_python.hpp"
#include "py_convertible.h"
#include "core/utctime_utilities.h"

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
        py_api::iterable_converter() // and finally ensure that lists into constructor do work
            .from_python<XVector>()
        ;
    }

    void vectors() {
        np_import();
        expose_vector<std::string>("StringVector");
        expose_vector<double>("DoubleVector");
        expose_vector<int>("IntVector");
        expose_vector<utctime>("UtcTimeVector");
    }
}

