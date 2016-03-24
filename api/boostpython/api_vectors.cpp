#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/scope.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/overloads.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/handle.hpp>

#include <boost/python/numeric.hpp>
#include <boost/python/tuple.hpp>

//#include <numpy/npy_common.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy_boost_python.hpp"

#include "core/utctime_utilities.h"



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

void def_vectors() {
    typedef std::vector<std::string> StringVector;
    typedef std::vector<int> IntVector;
    typedef std::vector<double> DoubleVector;
    typedef std::vector<utctime> UtcTimeVector;
    np_import();

    numpy_boost_python_register_type<int, 1>();
    numpy_boost_python_register_type<utctime,1>();
    numpy_boost_python_register_type<double, 1>();

    class_<StringVector>("StringVector","Is a list of strings")
        .def(vector_indexing_suite<StringVector>())
        ;

    class_<IntVector>("IntVector")
        .def(vector_indexing_suite<IntVector>())
        .def("FromNdArray",FromNdArray<int>).staticmethod("FromNdArray")
        .def("from_numpy",FromNdArray<int>).staticmethod("from_numpy")
        .def("to_numpy",ToNpArray<int>,"convert IntVector to numpy").staticmethod("to_numpy")
        ;

    class_<UtcTimeVector>("UtcTimeVector")
        .def(vector_indexing_suite<UtcTimeVector>())
        .def("FromNdArray",FromNdArray<utctime>).staticmethod("FromNdArray")
        .def("from_numpy",FromNdArray<utctime>).staticmethod("from_numpy")
        .def("to_numpy",ToNpArray<utctime>,"convert UtcTimeVector to numpy").staticmethod("to_numpy")
        ;

    class_<DoubleVector>("DoubleVector")
        .def(vector_indexing_suite<DoubleVector>())
        .def("FromNdArray",FromNdArray<double>).staticmethod("FromNdArray")
        .def("from_numpy",FromNdArray<double>).staticmethod("from_numpy")
        .def("to_numpy",ToNpArray<double>,"convert DoubleVector to numpy").staticmethod("to_numpy")
        //.def("ToNpArray",ToNpArray<utctime>,"convert UtcTimeVector to numpy").staticmethod("ToNpArray")
        ;//.def("push_back",&vector<double>::push_back,args("v"),"adds another element at end of the vector");
}

