#pragma once
#include <cxxtest/TestSuite.h>

class hbv_snow_test: public CxxTest::TestSuite {
  public:
      void test_integral_calculations();
      void test_mass_balance_at_snowpack_reset();
      void test_mass_balance_at_snowpack_buildup();
      void test_mass_balance_rain_no_snow();
      void test_mass_balance_melt_no_precip();
};

/* vim: set filetype=cpp: */
