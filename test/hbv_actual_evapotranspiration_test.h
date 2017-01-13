#pragma once

class hbv_actual_evapotranspiration_test : public CxxTest::TestSuite
{
public:
	void test_soil_moisture();
	void test_snow();
	void test_soil_moisture_threshold();
    void test_evap_from_non_snow_only();
};

