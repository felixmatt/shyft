// test_pch.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#if defined(_WINDOWS)
#pragma once
#pragma warning (disable : 4267)
#pragma warning (disable : 4244)
#pragma warning (disable : 4503)
#endif
#ifdef SHYFT_NO_PCH
#include <chrono>
#include <iostream>

#else
#include "core/core_pch.h"
#endif // SHYFT_NO_PCH
#include <doctest/doctest.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define TS_ASSERT_EQUALS(a,b) FAST_CHECK_EQ((a),(b))
#define TS_ASSERT_DIFFERS(a,b) FAST_CHECK_NE((a),(b))
#define TS_ASSERT_DELTA(a,b,d) CHECK(std::abs((a)-(b))<=(d))
#define TS_ASSERT(a) FAST_CHECK_UNARY((a) == true)
#define TS_ASSERT_LESS_THAN(a,b) FAST_CHECK_LT((a),(b))
#define TS_WARN(msg) WARN_UNARY((msg)!=nullptr)
#define TS_TRACE(msg) WARN_UNARY((msg)!=nullptr)
#define TS_ASSERT_THROWS(e,et) CHECK_THROWS_AS((e),et)
#define TS_FAIL(msg) FAST_CHECK_UNARY((msg)!=nullptr)
#define TS_ASSERT_THROWS_ANYTHING(e) CHECK_THROWS((e))
// TODO: reference additional headers your program requires here
using timing = std::chrono::high_resolution_clock;
inline int  elapsed_ms(timing::time_point t0, timing::time_point t1) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
};
inline int  elapsed_us(timing::time_point t0, timing::time_point t1) {
	return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
};
