#ifndef _shyft_core_serialize_h
#define _shyft_core_serialize_h
#pragma once

#if defined(_WINDOWS)
#pragma warning (disable : 4267)
#pragma warning (disable : 4244)
#pragma warning (disable : 4503)
#endif

#include <boost/serialization/access.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/level.hpp>
#include <boost/serialization/tracking.hpp>
#include <boost/serialization/is_bitwise_serializable.hpp>
/** in the header-files:
  * x_serialize_decl to be used as the *last*
  * line in the class/struct declaration
  */
#   define x_serialize_decl() \
      private:\
    friend class boost::serialization::access;\
    template<class Archive>\
    void serialize(Archive & ar, const unsigned int file_version)\

/** in the header-files:
 *  x_serialize_export_key to be used *outside*
 *  namespace at the *end* of the header file
 */
#   define x_serialize_export_key(T) BOOST_CLASS_EXPORT_KEY(T)
#   define x_serialize_export_key_nt(T) x_serialize_export_key(T);BOOST_CLASS_TRACKING(T, boost::serialization::track_never)
#   define x_serialize_binary(T) BOOST_IS_BITWISE_SERIALIZABLE(T);BOOST_CLASS_IMPLEMENTATION(T, boost::serialization::primitive_type)
/** in the library implementation files:
 *  x_serialize_implement to be use in the library
 *  implementation file, *after*
 *  template<class Archive>
 *     namespace::cls::serialize(Archive&ar,const int v) {} impl
 */
#   define x_serialize_implement(T)  BOOST_CLASS_EXPORT_IMPLEMENT(T)


/** in the library implementation files after x_serialize_implement
 */
#   define x_serialize_archive(T,AO,AI) \
    template void T::serialize( AO &,const unsigned int);\
    template void T::serialize( AI &,const unsigned int);



#endif
