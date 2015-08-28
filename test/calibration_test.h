#pragma once

#include <cxxtest/TestSuite.h>

class calibration_test: public CxxTest::TestSuite {
  public:
    void test_dummy();
    void test_simple();
    void test_nash_sutcliffe_goal_function();
    void test_kling_gupta_goal_function();
};

