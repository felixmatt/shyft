#pragma once


class inverse_distance_test : public CxxTest::TestSuite {
public:
	void test_temperature_model();
	void test_temperature_model_default_gradient();
	void test_radiation_model();
	void test_precipitation_model();
	void test_one_source_one_dest_calculation();
	void test_two_sources_one_dest_calculation();
	void test_using_finite_sources_only();
	void test_eliminate_far_away_sources();
	void test_using_up_to_max_sources();
	void test_handling_different_sources_pr_timesteps();
	void test_performance();
	void test_temperature_gradient_model();
	void test_zscale_distance();
};

/* vim: set filetype=cpp: */
