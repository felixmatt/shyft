#pragma once

/** \brief CoreApiTest verifies how to assemble/orchestrate enki core and api
 * classes into useful functions, using mocks for simulation of environment series.
 *
 */

class api_test : public CxxTest::TestSuite {
  public:
    void test_ptgsk_state_io(void);
	void test_ptssk_state_io(void);

};
