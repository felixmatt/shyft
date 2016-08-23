#pragma once
#include <cxxtest/TestSuite.h>


class gridpp_test : public CxxTest::TestSuite {
public:
	void test_interpolate_sources_should_populate_grids();
	void test_main_workflow_should_populate_grids();
	void test_sih_workbench();
};
