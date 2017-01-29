#pragma once

class bayesian_kriging_test: public CxxTest::TestSuite
{
public:
    void test_covariance_calculation();
    void test_build_elevation_matrices();
    void test_build_covariance_matrices();
    void test_interpolation();
    void test_performance();
};

/* vim: set filetype=cpp: */
