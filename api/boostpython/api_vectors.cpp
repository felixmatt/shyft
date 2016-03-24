#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/scope.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/overloads.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/handle.hpp>

//#include <numpy/npy_common.h>

#include "core/utctime_utilities.h"



using namespace shyft::core;
using namespace boost::python;
using namespace std;

void def_vectors() {
    typedef std::vector<std::string> StringVector;
    typedef std::vector<int> IntVector;
    typedef std::vector<double> DoubleVector;
    typedef std::vector<utctime> UtcTimeVector;

    class_<StringVector>("StringVector","Is a list of strings")
        .def(vector_indexing_suite<StringVector>())
        ;

    class_<IntVector>("IntVector")
        .def(vector_indexing_suite<IntVector>())
        ;

    class_<UtcTimeVector>("UtcTimeVector")
        .def(vector_indexing_suite<UtcTimeVector>())
        ;

    class_<DoubleVector>("DoubleVector")
        .def(vector_indexing_suite<DoubleVector>())
        ;//.def("push_back",&vector<double>::push_back,args("v"),"adds another element at end of the vector");
}

