#pragma once
#include <cxxtest/TestSuite.h>

#ifdef ENKI_HAS_GRF
class gaussian_random_field_test: public CxxTest::TestSuite {
  public:
    void test_calculate_anisotropy_distance();
    void test_gaussian_model();
    void test_spherical_model();
    void test_exponential_model();
    void test_matern_model();
    void test_calculate_gibbs_weights();
    void test_calculate_local_krig_weights();
    void test_gamma_transform();
    void test_gibbs_sampler();
    void test_random_fields();
};
#endif
