///	Copyright 2012 Statkraft Energi A/S
///
///	This file is part of Shyft.
///
///	Shyft is free software: you can redistribute it and/or modify
///	it under the terms of the GNU Lesser General Public License as
///	published by the Free Software Foundation, either version 3 of
///	the License, or (at your option) any later version.
///
///	Shyft is distributed in the hope that it will be useful,
///	but WITHOUT ANY WARRANTY; without even the implied warranty of
///	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
///	GNU Lesser General Public License for more details.
///
///	You should have received a copy of the GNU Lesser General Public
///	License along with Shyft, usually located under the Shyft root
///	directory in two files named COPYING.txt and COPYING_LESSER.txt.
///	If not, see <http://www.gnu.org/licenses/>.
///
#ifndef _shyft_core_serialize_h
#define _shyft_core_serialize_h

#if defined(_WINDOWS)
#pragma warning (disable : 4267)
#pragma warning (disable : 4244)
#pragma warning (disable : 4503)
#endif

#include <boost/serialization/access.hpp>
#include <boost/serialization/export.hpp>

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
    template void T::serialize( boost::archive::AO &,const unsigned int);\
    template void T::serialize( boost::archive::AI &,const unsigned int);
#endif // _shyft_core_serialize_h



