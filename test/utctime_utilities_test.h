#pragma once


class utctime_utilities_test: public CxxTest::TestSuite {
    public:
    void test_utctime();
    void test_calendar_timezone();
    void test_calendar_to_string();
    void test_calendar_day_of_year();
    void test_calendar_month();
	void test_calendar_day_of_week();
	void test_calendar_trim();
	void test_calendar_add_and_diff_units();
	void test_YMDhms_reasonable_calendar_coordinates();
};


