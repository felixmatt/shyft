#pragma once

class gamma_snow_test: public CxxTest::TestSuite {
  public:
    void test_reset_snow_pack_zero_storage();
    void test_reset_snow_pack_with_storage();
    void test_calculate_snow_state();
    void test_correct_lwc();
    void test_step();
    void test_warm_winter_effect();
    void test_output_independent_of_timestep();
    void test_forest_altitude_dependent_snow_cv();
};

/* vim: set filetype=cpp: */
