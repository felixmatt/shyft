#pragma once


class serialization_test:public CxxTest::TestSuite
{
    public:
    void test_serialization();
    void test_api_ts_ref_binding();
    void test_serialization_performance() ;
};
