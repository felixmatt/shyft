#pragma once
#include <cxxtest/TestSuite.h>

class pt_gs_k_test: public CxxTest::TestSuite {
  public:
    void test_call_stack();
    void test_raster_call_stack();
    void test_mass_balance();
};
