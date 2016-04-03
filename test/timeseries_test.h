#pragma once

#include <cxxtest/TestSuite.h>

class timeseries_test: public CxxTest::TestSuite
{
public:
    void test_point_timeaxis();
    void test_timeaxis();
    void test_point_source_with_timeaxis();
    void test_point_source_scale_by_value();
    void test_hint_based_bsearch();

    void test_average_value_staircase();
    void test_average_value_linear_between_points();
    void test_TxFxSource();

    void test_point_timeseries_with_point_timeaxis();
    void test_time_series_difference();
    void test_ts_weighted_average();
	void test_sin_fx_ts();
	void test_binary_operator();
	void test_api_ts();

};


/* vim: set filetype=cpp: */
