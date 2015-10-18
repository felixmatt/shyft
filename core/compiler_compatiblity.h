#pragma once

/// Windows portability stuff goes here
#ifdef _MSC_VER
#if _MSC_VER < 1800
namespace std {
	static inline bool isfinite(double x) {return _finite(x)!=0;}

	static inline double nan(const char*) {return std::numeric_limits<double>::quiet_NaN();}
}

#endif
#endif
