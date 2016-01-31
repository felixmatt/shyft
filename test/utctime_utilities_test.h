#pragma once


class utctime_utilities_test: public CxxTest::TestSuite {
    public:
    void test_utctime();
    void test_utcperiod();
    void test_calendar_timezone();
    void test_calendar_to_string();
    void test_calendar_day_of_year();
    void test_calendar_month();
	void test_calendar_day_of_week();
	void test_calendar_trim();
	void test_calendar_add_and_diff_units();
	void test_YMDhms_reasonable_calendar_coordinates();
	void test_add_over_dst_transitions();
	void test_tz_info_db();
	void test_calendar_trim_with_dst();
	void test_add_months();
};


