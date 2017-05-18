#include "test_pch.h"
#include "core/time_axis.h"

using namespace shyft;
using namespace shyft::core;
using namespace std;

/** \brief Utility function to verify one time-axis are conceptually equal to another */
template <class TA, class TB>
static bool test_if_equal( const TA& e, const TB& t ) {
    using namespace std;
    if( e.size() != t.size() )
        return false;
    if( e.total_period() != t.total_period() )
        return false;
    if( e.index_of( e.total_period().end ) != t.index_of( e.total_period().end ) )
        return false;
    if( e.open_range_index_of( e.total_period().end ) != t.open_range_index_of( e.total_period().end ) )
        return false;

    for( size_t i = 0; i < e.size(); ++i ) {
        if( e.time( i ) != t.time( i ) )
            return false;
        if( e.period( i ) != t.period( i ) )
            return false;
        if( e.index_of( e.time( i ) + deltaminutes( 30 ) ) != t.index_of( e.time( i ) + deltaminutes( 30 ) ) )
            return false;
        if( e.index_of( e.time( i ) - deltaminutes( 30 ) ) != t.index_of( e.time( i ) - deltaminutes( 30 ) ) )
            return false;
        if( e.open_range_index_of( e.time( i ) + deltaminutes( 30 ) ) != t.open_range_index_of( e.time( i ) + deltaminutes( 30 ) ) )
            return false;
        utctime tx = e.time( i ) - deltaminutes( 30 );
        size_t ei = e.open_range_index_of( tx );
        size_t ti = t.open_range_index_of( tx );
        TS_ASSERT_EQUALS( ei, ti );
        if( ei != ti )
            return false;
    }

	if (!equivalent_time_axis(e, t)) // verify e and t produces the same periods
		return false;
	// now create a time-axis different from e and t, just to verify that equivalent_time_axis states it's false
	time_axis::fixed_dt u(utctime(1234), deltahours(1), 21);

	return !equivalent_time_axis(u,e) && !equivalent_time_axis(u,t);
}
TEST_SUITE("time_axis") {
TEST_CASE("test_all") {
    // Verify that if all types of time-axis are setup up to have the same periods
    // they all have the same properties.
    // test-strategy: Have one fixed time-axis that the other should equal

    auto utc = make_shared<calendar>();
    utctime start = utc->time( YMDhms( 2016, 3, 8 ) );
    auto dt = deltahours( 3 );
    int  n = 9 * 3;
    time_axis::fixed_dt expected( start, dt, n ); // this is the simplest possible time axis
    //
    // STEP 0: verify that the expected time-axis is correct
    //
    TS_ASSERT_EQUALS( n, (int)expected.size() );
    TS_ASSERT_EQUALS( utcperiod( start, start + n * dt ), expected.total_period() );
    TS_ASSERT_EQUALS( string::npos, expected.index_of( start - 1 ) );
    TS_ASSERT_EQUALS( string::npos, expected.open_range_index_of( start - 1 ) );
    TS_ASSERT_EQUALS( string::npos, expected.index_of( start + n * dt ) );
    TS_ASSERT_EQUALS( n - 1,(int) expected.open_range_index_of( start + n * dt ) );
    for( int i = 0; i < n; ++i ) {
        TS_ASSERT_EQUALS( start + i * dt, expected.time( i ) );
        TS_ASSERT_EQUALS( utcperiod( start + i * dt, start + ( i + 1 )*dt ), expected.period( i ) );
        TS_ASSERT_EQUALS( i,(int) expected.index_of( start + i * dt ) );
        TS_ASSERT_EQUALS( i,(int) expected.index_of( start + i * dt + dt - 1 ) );
        TS_ASSERT_EQUALS( i, (int)expected.open_range_index_of( start + i * dt ) );
        TS_ASSERT_EQUALS( i,(int) expected.open_range_index_of( start + i * dt + dt - 1 ) );
    }
    //
    // STEP 1: construct all the other types of time-axis, with equal content, but represented differently
    //
    time_axis::calendar_dt c_dt( utc, start, dt, n );
    vector<utctime> tp;
    for( int i = 0; i < n; ++i )tp.push_back( start + i * dt );
    time_axis::point_dt p_dt( tp, start + n * dt );
    vector<utcperiod> sub_period;
    for( int i = 0; i < 3; ++i ) sub_period.emplace_back( i * dt, ( i + 1 )*dt );
    time_axis::calendar_dt_p c_dt_p( utc, start, 3 * dt, n / 3, sub_period );
    vector<utcperiod> periods;
    for( int i = 0; i < n; ++i ) periods.emplace_back( start + i * dt, start + ( i + 1 )*dt );
    time_axis::period_list ta_of_periods( periods );
    //
    // STEP 2: Verify that all the other types are equal to the now verified correct expected time_axis
    //
    TS_ASSERT( test_if_equal( expected, c_dt ) );
    TS_ASSERT( test_if_equal( expected, p_dt ) );
    TS_ASSERT( test_if_equal( expected, c_dt_p ) );
    TS_ASSERT( test_if_equal( expected, ta_of_periods ) );
    TS_ASSERT( test_if_equal( expected, time_axis::generic_dt( expected ) ) );
    TS_ASSERT( test_if_equal( expected, time_axis::generic_dt( p_dt ) ) );
    TS_ASSERT( test_if_equal( expected, time_axis::generic_dt( c_dt ) ) );
    //
    // STEP 3: Verify the time_axis::combine algorithm when equal time-axis are combined
    //
    TS_ASSERT( test_if_equal( expected, time_axis::combine( expected, expected ) ) );
    TS_ASSERT( test_if_equal( expected, time_axis::combine( c_dt, expected ) ) );
    TS_ASSERT( test_if_equal( expected, time_axis::combine( c_dt, p_dt ) ) );
    TS_ASSERT( test_if_equal( expected, time_axis::combine( c_dt, p_dt ) ) );
    TS_ASSERT( test_if_equal( expected, time_axis::combine( ta_of_periods, p_dt ) ) );
    TS_ASSERT( test_if_equal( expected, time_axis::combine( ta_of_periods, c_dt_p ) ) );

    //
    // STEP 4: Verify the time_axis::combine algorithm for non-overlapping timeaxis(should give null-timeaxis)
    //
    time_axis::fixed_dt f_dt_null = time_axis::fixed_dt::null_range();
    time_axis::point_dt p_dt_x( {start + n * dt, start + ( n + 1 )*dt}, start + ( n + 2 )*dt );
    TS_ASSERT( test_if_equal( f_dt_null, time_axis::combine( c_dt, p_dt_x ) ) );
    TS_ASSERT( test_if_equal( f_dt_null, time_axis::combine( expected, p_dt_x ) ) );
    TS_ASSERT( test_if_equal( f_dt_null, time_axis::combine( p_dt, p_dt_x ) ) );
    TS_ASSERT( test_if_equal( f_dt_null, time_axis::combine( ta_of_periods, f_dt_null ) ) );
    TS_ASSERT( test_if_equal( f_dt_null, time_axis::combine( p_dt_x, c_dt_p ) ) );


    //
    // STEP 5: Verify the time_axis::combine algorithm for overlapping time-axis
    //
    time_axis::fixed_dt overlap1( start + dt, dt, n );
    time_axis::fixed_dt expected_combine1( start + dt, dt, n - 1 );
    TS_ASSERT( test_if_equal( expected_combine1, time_axis::combine( expected, overlap1 ) ) );
    TS_ASSERT( test_if_equal( expected_combine1, time_axis::combine( c_dt, overlap1 ) ) );
    TS_ASSERT( test_if_equal( expected_combine1, time_axis::combine( p_dt, overlap1 ) ) );
    TS_ASSERT( test_if_equal( expected_combine1, time_axis::combine( overlap1, time_axis::generic_dt( c_dt ) ) ) );
    TS_ASSERT( test_if_equal( expected_combine1, time_axis::combine( overlap1, time_axis::generic_dt( p_dt ) ) ) );
    TS_ASSERT( test_if_equal( expected_combine1, time_axis::combine( overlap1, time_axis::generic_dt( expected ) ) ) );
    TS_ASSERT( test_if_equal( expected_combine1, time_axis::combine( overlap1, c_dt_p ) ) );
    TS_ASSERT( test_if_equal( expected_combine1, time_axis::combine( ta_of_periods, overlap1 ) ) );

    //
    // STEP 6: Verify the time_axis::combine algorithm for sparse time-axis period_list|calendar_dt_p
    //

    // create sparse time-axis samples
    vector<utcperiod> sparse_sub_period;
    for( int i = 0; i < 3; ++i )  sparse_sub_period.emplace_back( i * dt + deltahours( 1 ), ( i + 1 )*dt - deltahours( 1 ) );
    vector<utcperiod> sparse_period;
    for( int i = 0; i < n; ++i ) sparse_period.emplace_back( start + i * dt + deltahours( 1 ), start + ( i + 1 )*dt - deltahours( 1 ) );

    time_axis::calendar_dt_p sparse_c_dt_p( utc, start, 3 * dt, n / 3, sparse_sub_period );
    time_axis::period_list sparse_period_list( sparse_period );
    TS_ASSERT( test_if_equal( sparse_c_dt_p, sparse_period_list ) ); // they should be equal
    // now combine a sparse with a dense time-axis, the result should be equal to the sparse (given they cover same period)
    TS_ASSERT( test_if_equal( sparse_c_dt_p, time_axis::combine( expected, sparse_c_dt_p ) ) ); // combine to a dense should give the sparse result
    TS_ASSERT( test_if_equal( sparse_c_dt_p, time_axis::combine( sparse_c_dt_p, expected ) ) ); // combine to a dense should give the sparse result

    TS_ASSERT( test_if_equal( sparse_c_dt_p, time_axis::combine( expected, sparse_period_list ) ) ); // combine to a dense should give the sparse result
    TS_ASSERT( test_if_equal( sparse_c_dt_p, time_axis::combine( sparse_c_dt_p, sparse_period_list ) ) ); // combine to a dense should give the sparse result
    TS_ASSERT( test_if_equal( sparse_c_dt_p, time_axis::combine( c_dt_p, sparse_period_list ) ) ); // combine to a dense should give the sparse result
    //final tests: verify that if we combine two sparse time-axis, we get the distinct set of the periods.
    {
        time_axis::period_list ta1( {utcperiod( 1, 3 ), utcperiod( 5, 7 ), utcperiod( 9, 11 )} );
        time_axis::period_list ta2( {utcperiod( 0, 2 ), utcperiod( 4, 10 ), utcperiod( 11, 12 )} );
        time_axis::period_list exp( {utcperiod( 1, 2 ), utcperiod( 5, 7 ), utcperiod( 9, 10 )} );
        TS_ASSERT( test_if_equal( exp, time_axis::combine( ta1, ta2 ) ) );

    }

}


TEST_CASE("test_time_shift") {
    calendar utc;
    utctime t0=utc.time(2015,1,1);
    utctime t1=utc.time(2016,1,1);
    auto dt=deltahours(1);
    size_t n=24;
    time_axis::fixed_dt ta0(t0,dt,n);
    time_axis::fixed_dt ta1(time_shift(ta0,t1-t0));
    TS_ASSERT( test_if_equal( time_axis::fixed_dt(t1,dt, n), ta1 ) );

}
TEST_CASE("time_axis_map") {

	using namespace shyft;
	using namespace shyft::core;
	using namespace std;

	calendar utc;
	auto t0 = utc.time(2016, 1, 1);
	auto dt1 = deltahours(1);
	auto dt2 = deltahours(3);
	time_axis::fixed_dt src(t0, dt1, 24);
	time_axis::fixed_dt m(t0, dt2, 24 / 3);
	SUBCASE("simple reduction test") {
		auto ta_xf = time_axis::make_time_axis_map(src, m);
		for (size_t i = 0; i < m.size(); ++i) {
			FAST_CHECK_EQ(ta_xf.src_index(i), i * 3);
		}
		FAST_CHECK_EQ(ta_xf.src_index(300), std::string::npos);
	}
	SUBCASE("from coarse to fine") {
		auto ta_xf = time_axis::make_time_axis_map(m, src);
		for (size_t i = 0; i < src.size(); ++i) {
			FAST_CHECK_EQ(ta_xf.src_index(i), i / 3);
		}
	}
	SUBCASE("src offset the time t with one delta") {
		src.t += dt1;
		auto ta_xf = time_axis::make_time_axis_map(src, m);
		for (size_t i = 0; i < m.size(); ++i) {
			if (i == 0) {
				FAST_CHECK_EQ(ta_xf.src_index(i), string::npos);
			} else {
				FAST_CHECK_EQ(ta_xf.src_index(i), (i * 3) - 1);
			}
		}
	}
	SUBCASE("verify npos if entirely after") {
		src.t = m.total_period().end;
		auto ta_xf = time_axis::make_time_axis_map(src, m);
		for (size_t i = 0; i < m.size(); ++i) {
			FAST_CHECK_EQ(ta_xf.src_index(i), string::npos);
		}
	}
	SUBCASE("verify if src is entirely before") {
		src.t = m.t - src.dt*src.n * 10;
		auto ta_xf = time_axis::make_time_axis_map(src, m);
		for (size_t i = 0; i < m.size(); ++i) {
			FAST_CHECK_EQ(ta_xf.src_index(i), string::npos);
		}
	}
	time_axis::generic_dt src2(src);
	SUBCASE("simple reduction test") {
		auto ta_xf = time_axis::make_time_axis_map(src2, m);
		for (size_t i = 0; i < m.size(); ++i) {
			FAST_CHECK_EQ(ta_xf.src_index(i), i * 3);
		}
		FAST_CHECK_EQ(ta_xf.src_index(300), std::string::npos);
	}
	SUBCASE("from coarse to fine") {
		auto ta_xf = time_axis::make_time_axis_map(m, src2);
		for (size_t i = 0; i < src2.size(); ++i) {
			FAST_CHECK_EQ(ta_xf.src_index(i), i / 3);
		}
	}
	SUBCASE("src offset the time t with one delta") {
		src2.f.t += dt1;
		auto ta_xf = time_axis::make_time_axis_map(src2, m);
		for (size_t i = 0; i < m.size(); ++i) {
			if (i == 0) {
				FAST_CHECK_EQ(ta_xf.src_index(i), string::npos);
			} else {
				FAST_CHECK_EQ(ta_xf.src_index(i), (i * 3) - 1);
			}
		}
	}
	SUBCASE("verify npos if entirely after") {
		src2.f.t = m.total_period().end;
		auto ta_xf = time_axis::make_time_axis_map(src2, m);
		for (size_t i = 0; i < m.size(); ++i) {
			FAST_CHECK_EQ(ta_xf.src_index(i), string::npos);
		}
	}
	SUBCASE("verify if src is entirely before") {
		src2.f.t = m.t - src2.f.dt*src.n * 10;
		auto ta_xf = time_axis::make_time_axis_map(src2, m);
		for (size_t i = 0; i < m.size(); ++i) {
			FAST_CHECK_EQ(ta_xf.src_index(i), string::npos);
		}
	}

	//auto ix_map = tat.map(a, b);
	//FAST_CHECK_EQ(ix_map.size(), b.size());
}
}
