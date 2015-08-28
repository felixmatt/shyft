#include "test_pch.h"
#include "core/utctime_utilities.h"
#include "utctime_utilities_test.h"
using namespace std;
using namespace shyft;
using namespace shyft::core;

void utctime_utilities_test::test_utctime() {
    calendar c(0);
    YMDhms unixEra(1970,01,01,00,00,00);
    YMDhms y_null;

    TS_ASSERT(y_null.is_null());
    TS_ASSERT_EQUALS(0L,c.time(unixEra));
    YMDhms r=c.calendar_units(utctime(0L));
    TS_ASSERT_EQUALS(r,unixEra);
}

void utctime_utilities_test::test_calendar_trim() {
    // simple trim test
    calendar cet(deltahours(1));
    utctime t=cet.time(YMDhms(2012,3,8,12,16,44));
    YMDhms t_y  (2012, 1, 1, 0, 0, 0);
    YMDhms t_m  (2012, 3, 1, 0, 0, 0);
    YMDhms t_d  (2012, 3, 8, 0, 0, 0);
    YMDhms t_h  (2012, 3, 8,12, 0, 0);
    YMDhms t_15m(2012, 3, 8,12,15, 0);
    TS_ASSERT_EQUALS(cet.calendar_units( cet.trim(t,calendar::YEAR)),t_y);
    TS_ASSERT_EQUALS(cet.calendar_units( cet.trim(t,calendar::MONTH)),t_m);
    TS_ASSERT_EQUALS(cet.calendar_units( cet.trim(t,calendar::DAY)),t_d);
    TS_ASSERT_EQUALS(cet.calendar_units( cet.trim(t,calendar::HOUR)),t_h);
    TS_ASSERT_EQUALS(cet.calendar_units( cet.trim(t,deltaminutes(15))),t_15m);
}

void utctime_utilities_test::test_calendar_timezone() {
    YMDhms unixEra(1970,01,01,00,00,00);
    calendar cet(deltahours(1));
    TS_ASSERT_EQUALS(deltahours(-1),cet.time(unixEra));
}

void utctime_utilities_test::test_calendar_to_string() {
    calendar cet(deltahours(1));
    utctime t=cet.time(YMDhms(2012,3,8,12,16,44));
    string t_s= cet.to_string(t);
    TS_ASSERT_EQUALS(t_s,string("2012.03.08T12:16:44"));
}

void utctime_utilities_test::test_calendar_day_of_year() {
    calendar cet(deltahours(1));
    TS_ASSERT_EQUALS(1,cet.day_of_year(cet.time(YMDhms(2012,1,1,10,11,12))));
    TS_ASSERT_EQUALS(2,cet.day_of_year(cet.time(YMDhms(2012,1,2,0,0,0))));
    TS_ASSERT_EQUALS(366,cet.day_of_year(cet.time(YMDhms(2012,12,31,12,0,0))));


}
void utctime_utilities_test::test_calendar_month() {
    calendar cet(deltahours(1));
    TS_ASSERT_EQUALS( 1,cet.month(cet.time(YMDhms(2012,1,1,10,11,12))));
    TS_ASSERT_EQUALS( 2,cet.month(cet.time(YMDhms(2012,2,2,0,0,0))));
    TS_ASSERT_EQUALS(12,cet.month(cet.time(YMDhms(2012,12,31,23,59,59))));

}


void utctime_utilities_test::test_calendar_add_and_diff_units() {
	calendar cet(deltahours(1));
	int n_units = 3;
	utctimespan dts[] = { calendar::HOUR, calendar::DAY, calendar::WEEK };// MONTH YEAR is not supported/needed (yet)
	auto t1 = cet.time(YMDhms(2012, 1, 1));
	for (auto dt : dts) {
		auto t2 = cet.add(t1, dt, n_units);
		auto t3 = cet.add(t2, dt, -n_units);
		utctimespan rem;
		auto ndiff = cet.diff_units(t1, t2, dt, rem);//verify we get difference ok
		TS_ASSERT_EQUALS(ndiff, n_units);
		TS_ASSERT_EQUALS(t3, t1);// verify subtraction gives back original result
	}
}
void utctime_utilities_test::test_calendar_day_of_week() {
	calendar cet(deltahours(1));
	for (int i = 0; i < 7;++i)
		TS_ASSERT_EQUALS(i, cet.day_of_week(cet.time(YMDhms(2015, 8, 9+i, 10, 11, 12))));

}
