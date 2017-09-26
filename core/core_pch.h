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

#if defined(_WINDOWS)
#pragma once

#pragma warning (disable : 4267)
#pragma warning (disable : 4244)
#pragma warning (disable : 4503)
#endif


#include <memory>
#include <limits>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <functional>
#include <iosfwd>
#include <thread>
#include <future>
#include <mutex>
#include <stdexcept>
#include <random>
#include <type_traits>
#include <time.h>
#include <cerrno>
#include <cstring>
#include <cstddef>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <regex>

#ifdef _WIN32
#include <io.h>
#else
#include <sys/io.h>
#define O_BINARY 0
#define O_SEQUENTIAL 0
#include <sys/stat.h>
#endif
#include <fcntl.h>

#include <boost/filesystem.hpp>
namespace fs=boost::filesystem; // it's a standard c++ 17
#include <boost/numeric/odeint.hpp>
#include <boost/range.hpp>
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/math/tools/minima.hpp>

//-- serialization support

/**
 serializiation implemented using boost,
  see reference: http://www.boost.org/doc/libs/1_62_0/libs/serialization/doc/
 */

#include <boost/serialization/serialization.hpp>
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


//--providing all needed lin-alg:
#include <armadillo>

//-- providing all needed optimization

#include <dlib/optimization.h>
#include <dlib/statistics.h>

