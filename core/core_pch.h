///	Copyright 2012 Statkraft Energi A/S
///
///	This file is part of ENKI.
///
///	ENKI is free software: you can redistribute it and/or modify
///	it under the terms of the GNU Lesser General Public License as
///	published by the Free Software Foundation, either version 3 of
///	the License, or (at your option) any later version.
///
///	ENKI is distributed in the hope that it will be useful,
///	but WITHOUT ANY WARRANTY; without even the implied warranty of
///	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
///	GNU Lesser General Public License for more details.
///
///	You should have received a copy of the GNU Lesser General Public
///	License along with ENKI, usually located under the ENKI root
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
#include <iostream>
#include <ostream>
#include <sstream>
#include <thread>
#include <future>
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

#ifndef _MSC_VER
// SiH: On ubuntu 15.04, with boost 1.59, gcc 4.9.2 this was needed
// to avoid undefined static_gcd_type
#include <boost/math_fwd.hpp>
typedef boost::math::static_gcd_type static_gcd_type;
#endif
#include <boost/numeric/odeint.hpp>
#include <boost/range.hpp>
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/math/tools/minima.hpp>

//-- serialization support
#include <boost/serialization/string.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/assume_abstract.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/access.hpp>

#include <armadillo>
