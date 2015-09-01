#pragma once

#include <cxxtest/TestSuite.h>

class sceua_test: public CxxTest::TestSuite {
  public:
    void test_basic();
    void test_complex();
};
