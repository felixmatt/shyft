#pragma once
#include <cxxtest/TestSuite.h>

class kriging_test: public CxxTest::TestSuite {
    public:
    void test_covariance_calculation();
    void test_build_covariance_obs_to_obs_matrix();
    void test_build_covariance_obs_to_grid_matrix();
    void test_interpolation();
};

/* vim: set filetype=cpp: */

