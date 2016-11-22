#pragma once

class actual_evapotranspiration_test: public CxxTest::TestSuite
{
  public:
    void test_water();
    void test_snow();
    void test_scale_factor();
};

