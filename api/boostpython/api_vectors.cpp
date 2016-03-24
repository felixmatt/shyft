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

namespace py_api {
    /// Thanks to stack overflow http://stackoverflow.com/questions/15842126/feeding-a-python-list-into-a-function-taking-in-a-vector-with-boost-python/15940413#15940413


    /// @brief Type that allows for registration of conversions from
    ///        python iterable types.
    struct iterable_converter
    {
      /// @note Registers converter from a python interable type to the
      ///       provided type.
      template <typename Container>
      iterable_converter&
      from_python()
      {
        boost::python::converter::registry::push_back(
          &iterable_converter::convertible,
          &iterable_converter::construct<Container>,
          boost::python::type_id<Container>());

        // Support chaining.
        return *this;
      }

      /// @brief Check if PyObject is iterable.
      static void* convertible(PyObject* object)
      {
        return PyObject_GetIter(object) ? object : NULL;
      }

      /// @brief Convert iterable PyObject to C++ container type.
      ///
      /// Container Concept requirements:
      ///
      ///   * Container::value_type is CopyConstructable.
      ///   * Container can be constructed and populated with two iterators.
      ///     I.e. Container(begin, end)
      template <typename Container>
      static void construct(
        PyObject* object,
        boost::python::converter::rvalue_from_python_stage1_data* data)
      {
        namespace python = boost::python;
        // Object is a borrowed reference, so create a handle indicting it is
        // borrowed for proper reference counting.
        python::handle<> handle(python::borrowed(object));

        // Obtain a handle to the memory block that the converter has allocated
        // for the C++ type.
        typedef python::converter::rvalue_from_python_storage<Container>
                                                                    storage_type;
        void* storage = reinterpret_cast<storage_type*>(data)->storage.bytes;

        typedef python::stl_input_iterator<typename Container::value_type>
                                                                        iterator;

        // Allocate the C++ type into the converter's memory block, and assign
        // its handle to the converter's convertible variable.  The C++
        // container is populated by passing the begin and end iterators of
        // the python object to the container's constructor.
        new (storage) Container(
          iterator(python::object(handle)), // begin
          iterator());                      // end
        data->convertible = storage;
      }
    };
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

    // Register interable conversions.
    py_api::iterable_converter()
        .from_python<std::vector<double> >()
        .from_python<std::vector<std::string> >()
        .from_python<std::vector<int> >()
        .from_python<std::vector<utctime> >()
    ;


    class_<StringVector>("StringVector","Is a list of strings")
        .def(vector_indexing_suite<StringVector>())
        ;

    class_<IntVector>("IntVector")
        .def(vector_indexing_suite<IntVector>())
        .def(init<const IntVector&>(args("const_ref_v")))
        .def("FromNdArray",FromNdArray<int>).staticmethod("FromNdArray")
        .def("from_numpy",FromNdArray<int>).staticmethod("from_numpy")
        .def("to_numpy",ToNpArray<int>,"convert IntVector to numpy").staticmethod("to_numpy")
        ;

    class_<UtcTimeVector>("UtcTimeVector")
        .def(vector_indexing_suite<UtcTimeVector>())
        .def(init<const UtcTimeVector&>(args("const_ref_v")))
        .def("FromNdArray",FromNdArray<utctime>).staticmethod("FromNdArray")
        .def("from_numpy",FromNdArray<utctime>).staticmethod("from_numpy")
        .def("to_numpy",ToNpArray<utctime>,"convert UtcTimeVector to numpy").staticmethod("to_numpy")
        ;

    class_<DoubleVector>("DoubleVector")
        .def(vector_indexing_suite<DoubleVector>())
        .def(init<const DoubleVector&>(args("const_ref_v")))
        .def("FromNdArray",FromNdArray<double>).staticmethod("FromNdArray")
        .def("from_numpy",FromNdArray<double>).staticmethod("from_numpy")
        .def("to_numpy",ToNpArray<double>,"convert DoubleVector to numpy").staticmethod("to_numpy")
        ;
}

