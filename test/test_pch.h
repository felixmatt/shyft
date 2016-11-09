// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#if defined(_WINDOWS)
#pragma once
#pragma warning (disable : 4267)
#pragma warning (disable : 4244)
#pragma warning (disable : 4503)
#endif
#define BOOST_CHRONO_VERSION 2
#define BOOST_CHRONO_PROVIDES_DATE_IO_FOR_SYSTEM_CLOCK_TIME_POINT 1


#include "core/core_pch.h"
#include <cxxtest/TestSuite.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// TODO: reference additional headers your program requires here
