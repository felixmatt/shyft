#pragma once


class kalman_test : public CxxTest::TestSuite {
public:
	void test_filter();
	void test_bias_predictor();
};
