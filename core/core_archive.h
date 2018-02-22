#pragma once

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#define core_arch_flags boost::archive::archive_flags::no_header

namespace shyft {
namespace core  {

using core_iarchive=boost::archive::binary_iarchive;
using core_oarchive=boost::archive::binary_oarchive;

// note that no_header is required when doing compatible-enough
// binary archives between the ms c++ compiler and gcc
// The difference is that ms c++ have 32bits ints (!)
//

// core_nvp is a stand-in for make_nvp (could resolve to one)
// that ensures that the serialization using core_nvp
// can enforce use of properly sized integers enough to
// provide serialization compatibility between ms c++ and gcc.
// ref. different int-sizes, the strategy is to
// forbid use of int which is different.

template <class T>
inline T& core_nvp(const char*,T&t) {
    return t;
}

// specialization that forbids the int type
// (still possible to trick the system, but a cheap fix that keeps the performance and boost library clean)
template <>
inline int & core_nvp<int>(const char*,int&t) {
    // would be better, but needs a more clever template (with a struct core_nvp etc.):
    // static_assert(false,"shyft serialization needs specific int type to work");
    throw std::runtime_error("shyft serialization needs specific int type to work"); // easy to spot the int source source runtime during testing
    return t;

}

#define x_arch(T) x_serialize_archive(T,core_oarchive,core_iarchive)

}
}
