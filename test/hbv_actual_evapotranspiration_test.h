#pragma once
#include <cxxtest/TestSuite.h>

class hbv_actual_evapotranspiration_test : public CxxTest::TestSuite
{
public:
	void test_soil_moisture();
	void test_snow();
	void test_soil_moisture_threshold();
};

