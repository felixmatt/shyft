#pragma once
#include <cxxtest/TestSuite.h>


class kalman_test : public CxxTest::TestSuite {
public:
	void test_filter();
	void test_bias_predictor();
};
