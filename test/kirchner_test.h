#pragma once

class kirchner_test: public CxxTest::TestSuite {
  public:
    void test_single_solve();
    void test_solve_from_zero_q();
    void test_hard_case();
    void test_simple_average_loads();
    void test_composite_average_loads();
};

/* vim: set filetype=cpp: */
