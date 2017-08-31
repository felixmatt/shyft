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
TEST_CASE("time_axis_extend") {

    namespace ta = shyft::time_axis;
    namespace core = shyft::core;

    core::calendar utc;

    SUBCASE("directly sequential fixed_dt") {
        core::utctime t0 = utc.time(2017, 1, 1);
        core::utctimespan dt = core::deltahours(1);
        size_t n = 512;

        ta::fixed_dt axis(t0, dt, 2*n);
        ta::fixed_dt ext(t0 + 2*n*dt, dt, 2*n);

        ta::generic_dt res = ta::extend(axis, ext, t0 + 2*n*dt);
        ta::fixed_dt expected(t0, dt, 4*n);

        FAST_REQUIRE_EQ(res.gt, ta::generic_dt::FIXED);
        FAST_CHECK_EQ(res.f, expected);
    }

    SUBCASE("fixed_dt with fixed_dt") {

        core::utctime t0 = utc.time(2017, 1, 1);
        core::utctimespan dt = deltahours(1);
        size_t n = 24u;

        ta::fixed_dt
            axis(t0, dt, n),
            empty = ta::fixed_dt::null_range();

        SUBCASE("empty time-axes") {
            SUBCASE("both empty") {
                ta::generic_dt res = ta::extend(empty, empty, empty.t + empty.dt*empty.n);
                FAST_REQUIRE_EQ(res.f, empty);
            }
            SUBCASE("last empty") {
                SUBCASE("split after") {
                    ta::generic_dt res = ta::extend(axis, empty, t0 + dt * n);
                    FAST_REQUIRE_EQ(res.f, axis);
                }
                SUBCASE("split inside") {
                    size_t split_after = n / 2;
                    ta::generic_dt res = ta::extend(axis, empty, t0 + dt * split_after);
                    FAST_REQUIRE_EQ(res.f, ta::fixed_dt(t0, dt, split_after));
                }
                SUBCASE("split before") {
                    ta::generic_dt res = ta::extend(axis, empty, t0 - 1);
                    FAST_REQUIRE_EQ(res.f, empty);
                }
            }
            SUBCASE("first empty") {
                SUBCASE("split after") {
                    ta::generic_dt res = ta::extend(empty, axis, t0 + dt * n);
                    FAST_REQUIRE_EQ(res.f, empty);
                }
                SUBCASE("split inside") {
                    size_t split_after = n / 2;
                    ta::generic_dt res = ta::extend(empty, axis, t0 + dt * split_after);
                    FAST_REQUIRE_EQ(res.f, ta::fixed_dt(t0 + dt * split_after, dt, n - split_after));
                }
                SUBCASE("split before") {
                    ta::generic_dt res = ta::extend(empty, axis, t0 - 1);
                    FAST_REQUIRE_EQ(res.f, axis);
                }
            }
        }
        SUBCASE("aligned") {
            SUBCASE("rhs fully before lhs") {  // branch 4
                ta::fixed_dt extension(t0 - 24u * dt, dt, 12u);
                ta::generic_dt res = ta::extend(axis, extension, t0 - 6 * dt);

                FAST_REQUIRE_EQ(res.gt, ta::generic_dt::generic_type::FIXED);
                FAST_CHECK_EQ(res.f, empty);
            }
            SUBCASE("rhs start before lhs and end inside") {  // branch 3
                ta::fixed_dt extension(t0 - 12u * dt, dt, n);
                ta::fixed_dt expected(t0, dt, 12u);

                ta::generic_dt res = ta::extend(axis, extension, t0);

                FAST_REQUIRE_EQ(res.gt, ta::generic_dt::generic_type::FIXED);
                FAST_CHECK_EQ(res.f, expected);
            }
            SUBCASE("rhs start before and end after lhs") {  // branch 1
                ta::fixed_dt extension(t0 - 12u * dt, dt, n + 24u);
                ta::fixed_dt expected(t0, dt, n + 12u);

                ta::generic_dt res = ta::extend(axis, extension, t0 + 12u * dt);

                FAST_REQUIRE_EQ(res.gt, ta::generic_dt::generic_type::FIXED);
                FAST_CHECK_EQ(res.f, expected);
            }
            SUBCASE("rhs matches exactly lhs") {  // branch 2
                ta::fixed_dt extension(t0, dt, n);
                ta::generic_dt res = ta::extend(axis, extension, t0 + n * dt);

                FAST_REQUIRE_EQ(res.gt, ta::generic_dt::generic_type::FIXED);
                FAST_CHECK_EQ(res.f, axis);
            }
            SUBCASE("rhs start inside lhs and end after") {  // branch 2
                ta::fixed_dt extension(t0 + (n / 2u)*dt, dt, n + 12u);
                ta::generic_dt res = ta::extend(axis, extension, t0 + (n / 2u + n + 12u)*dt);

                FAST_REQUIRE_EQ(res.gt, ta::generic_dt::generic_type::FIXED);
                FAST_CHECK_EQ(res.f, axis);
            }
            SUBCASE("rhs start and end inside lhs") {  // degenerate to point_dt
                ta::fixed_dt extension(t0 + 6u * dt, dt, 12u);
                ta::generic_dt res = ta::extend(axis, extension, t0 + 3u * dt);

                // construct time points
                std::vector<utctime> expected_points;
                for ( size_t i = 0u; i <= 3u; ++i )
                    expected_points.push_back(t0 + i * dt);
                for ( size_t i = 0u; i <= 12u; ++i )
                    expected_points.push_back(t0 + (i + 6u) * dt);

                FAST_REQUIRE_EQ(res.gt, ta::generic_dt::generic_type::POINT);
                FAST_CHECK_EQ(res.p, ta::point_dt(expected_points));
            }
            SUBCASE("rhs fully after lhs") {  // degenerate to point_dt
                ta::fixed_dt extension(t0 + (n + 2)*dt, dt, 12u);
                ta::generic_dt res = ta::extend(axis, extension, t0 + (n + 1) * dt);

                // construct time points
                std::vector<utctime> expected_points;
                for ( size_t i = 0u; i <= n; ++i )
                    expected_points.push_back(t0 + i * dt);
                for ( size_t i = 0u; i <= 12u; ++i )
                    expected_points.push_back(t0 + (i + n + 2) * dt);

                FAST_REQUIRE_EQ(res.gt, ta::generic_dt::generic_type::POINT);
                FAST_CHECK_EQ(res.p, ta::point_dt(expected_points));
            }
        }
        SUBCASE("unaligned") {
            SUBCASE("equal dt, unaligned boundaries") {
                utctime t0_ext = t0 + deltaminutes(30);

                ta::fixed_dt extension(t0_ext, dt, 2 * n);
                ta::generic_dt res = ta::extend(axis, extension, t0 + n*dt);

                // construct time points
                std::vector<utctime> expected_points;
                for ( utctime t = t0; t <= t0 + utctimespan(dt*n); t += dt )
                    expected_points.push_back(t);
                for ( utctime t = t0_ext + n*dt; t <= t0_ext + utctimespan(2 * n * dt); t += dt )
                    expected_points.push_back(t);

                FAST_REQUIRE_EQ(res.gt, ta::generic_dt::generic_type::POINT);
                FAST_CHECK_EQ(res.p, ta::point_dt(expected_points));
            }
            SUBCASE("unequal dt, aligned boundaries") {
                utctimespan dt_ext = deltaminutes(30);
                size_t n_ext = 4 * n;

                ta::fixed_dt extension(t0, dt_ext, n_ext);
                ta::generic_dt res = ta::extend(axis, extension, t0 + dt * n);

                // construct time points
                std::vector<utctime> expected_points;
                for ( utctime t = t0; t <= t0 + utctimespan(dt * n); t += dt )
                    expected_points.push_back(t);

                for ( utctime t = t0 + n * dt + dt_ext; t <= t0 + utctimespan(4 * n * dt_ext); t += dt_ext )
                    expected_points.push_back(t);

                FAST_REQUIRE_EQ(res.gt, ta::generic_dt::generic_type::POINT);
                FAST_CHECK_EQ(res.p, ta::point_dt(expected_points));
            }
        }
    }
    SUBCASE("calendar_dt with calendar_dt") {

        std::shared_ptr<core::calendar> cal = std::make_shared<core::calendar>(utc);
        core::utctime t0 = cal->time(2017, 4, 1);
        core::utctimespan dt = core::calendar::DAY;
        size_t n = 30;

        ta::calendar_dt
            axis(cal, t0, dt, n),
            empty = ta::calendar_dt::null_range();

        SUBCASE("empty time-axes") {
            SUBCASE("both empty") {
                ta::generic_dt res = ta::extend(empty, empty, 0);

                FAST_REQUIRE_EQ(res.gt, ta::generic_dt::CALENDAR);
                FAST_REQUIRE_EQ(res.c, empty);
            }
            SUBCASE("last empty") {
                SUBCASE("split after") {
                    ta::generic_dt res = ta::extend(axis, empty, cal->add(t0, dt, n + 1));

                    FAST_REQUIRE_EQ(res.gt, ta::generic_dt::CALENDAR);
                    FAST_REQUIRE_EQ(res.c, axis);
                }
                SUBCASE("split inside") {
                    size_t split_after = n / 2;  // 15
                    ta::calendar_dt expected(cal, t0, dt, split_after);
                    ta::generic_dt res = ta::extend(axis, empty, cal->add(t0, dt, split_after));

                    FAST_REQUIRE_EQ(res.gt, ta::generic_dt::CALENDAR);
                    FAST_REQUIRE_EQ(res.c, expected);
                }
                SUBCASE("split before") {
                    ta::generic_dt res = ta::extend(axis, empty, cal->add(t0, dt, -1));

                    FAST_REQUIRE_EQ(res.gt, ta::generic_dt::CALENDAR);
                    FAST_REQUIRE_EQ(res.c, empty);
                }
            }
            SUBCASE("first empty") {
                SUBCASE("split after") {
                    ta::generic_dt res = ta::extend(empty, axis, cal->add(t0, dt, n+1));

                    FAST_REQUIRE_EQ(res.gt, ta::generic_dt::CALENDAR);
                    FAST_REQUIRE_EQ(res.c, empty);
                }
                SUBCASE("split inside") {
                    size_t split_after = n / 2;  // 15

                    ta::calendar_dt expected(cal, cal->time(2017, 4, split_after + 1), dt, n - split_after);
                    ta::generic_dt res = ta::extend(empty, axis, cal->add(t0, dt, split_after));

                    FAST_REQUIRE_EQ(res.gt, ta::generic_dt::CALENDAR);
                    FAST_CHECK_EQ(res.c, expected);
                }
                SUBCASE("split before") {
                    ta::generic_dt res = ta::extend(empty, axis, cal->add(t0, dt, -1));

                    FAST_REQUIRE_EQ(res.gt, ta::generic_dt::CALENDAR);
                    FAST_REQUIRE_EQ(res.c, axis);
                }
            }
        }
        SUBCASE("aligned") {
            SUBCASE("rhs fully before lhs") {  // branch 4
                ta::calendar_dt extension(cal, cal->add(t0, dt, -15), dt, 10);
                ta::generic_dt res = ta::extend(axis, extension, cal->add(t0, dt, -5));

                FAST_REQUIRE_EQ(res.gt, ta::generic_dt::generic_type::CALENDAR);
                FAST_CHECK_EQ(res.c, empty);
            }
            SUBCASE("rhs start before lhs and end inside") {  // branch 3
                ta::calendar_dt extension(cal, cal->add(t0, dt, -15), dt, n);
                ta::calendar_dt expected(cal, t0, dt, 15u);

                ta::generic_dt res = ta::extend(axis, extension, t0);

                FAST_REQUIRE_EQ(res.gt, ta::generic_dt::generic_type::CALENDAR);
                FAST_CHECK_EQ(res.c, expected);
            }
            SUBCASE("rhs start before and end after lhs") {  // branch 1
                ta::calendar_dt extension(cal, cal->add(t0, dt, -10), dt, n+20);
                ta::calendar_dt expected(cal, t0, dt, n + 10u);

                ta::generic_dt res = ta::extend(axis, extension, cal->add(t0, dt, 15));

                FAST_REQUIRE_EQ(res.gt, ta::generic_dt::generic_type::CALENDAR);
                FAST_CHECK_EQ(res.c, expected);
            }
            SUBCASE("rhs matches exactly lhs") {  // branch 2
                ta::calendar_dt extension(cal, t0, dt, n);

                ta::generic_dt res = ta::extend(axis, extension, cal->add(t0, dt, n));

                FAST_REQUIRE_EQ(res.gt, ta::generic_dt::generic_type::CALENDAR);
                FAST_CHECK_EQ(res.c, axis);
            }
            SUBCASE("rhs start inside lhs and end after") {  // branch 2
                ta::calendar_dt extension(cal, cal->add(t0, dt, 15), dt, n);
                ta::generic_dt res = ta::extend(axis, extension, cal->add(t0, dt, n + 15));

                FAST_REQUIRE_EQ(res.gt, ta::generic_dt::generic_type::CALENDAR);
                FAST_CHECK_EQ(res.c, axis);
            }
            SUBCASE("rhs start and end inside lhs") {  // degenerate to point_dt
                ta::calendar_dt extension(cal, cal->add(t0, dt, 10), dt, n - 20);
                ta::generic_dt res = ta::extend(axis, extension, cal->add(t0, dt, 5));

                // construct time points
                std::vector<utctime> expected_points;
                for ( size_t i = 0u; i <= 5u; ++i )
                    expected_points.push_back(cal->add(t0, dt, i));
                for ( size_t i = 0u; i <= 10u; ++i )
                    expected_points.push_back(cal->add(t0, dt, 10 + i));

                FAST_REQUIRE_EQ(res.gt, ta::generic_dt::generic_type::POINT);
                FAST_CHECK_EQ(res.p, ta::point_dt(expected_points));
            }
            SUBCASE("rhs fully after lhs") {  // degenerate to point_dt
                ta::calendar_dt extension(cal, cal->add(t0, dt, n+10), dt, n);
                ta::generic_dt res = ta::extend(axis, extension, cal->add(t0, dt, n+5));

                // construct time points
                std::vector<utctime> expected_points;
                for ( size_t i = 0u; i <= n; ++i )
                    expected_points.push_back(cal->add(t0, dt, i));
                for ( size_t i = 0u; i <= n; ++i )
                    expected_points.push_back(cal->add(t0, dt, n + 10 + i));

                FAST_REQUIRE_EQ(res.gt, ta::generic_dt::generic_type::POINT);
                FAST_CHECK_EQ(res.p, ta::point_dt(expected_points));
            }
        }
        SUBCASE("unaligned") {
            SUBCASE("equal dt, unaligned boundaries") {
                utctime t0_ext = cal->add(t0, core::calendar::HOUR, 12);

                ta::calendar_dt extension(cal, t0_ext, dt, 2 * n);
                ta::generic_dt res = ta::extend(axis, extension, cal->add(t0, dt, n));

                // construct time points
                std::vector<utctime> expected_points;
                for ( size_t i = 0u; i <= n; ++i )
                    expected_points.push_back(cal->add(t0, dt, i));
                for ( size_t i = 0u; i <= n; ++i )
                    expected_points.push_back(cal->add(t0_ext, dt, n + i));

                FAST_REQUIRE_EQ(res.gt, ta::generic_dt::generic_type::POINT);
                FAST_CHECK_EQ(res.p, ta::point_dt(expected_points));
            }
            SUBCASE("unequal dt, aligned boundaries") {
                core::utctimespan dt_ext = core::calendar::HOUR;
                size_t n_ext = 2 * 24 * n;

                ta::calendar_dt extension(cal, t0, dt_ext, n_ext);
                ta::generic_dt res = ta::extend(axis, extension, cal->add(t0, dt, n));

                // construct time points
                std::vector<utctime> expected_points;
                for ( size_t i = 0u; i <= n; ++i )
                    expected_points.push_back(cal->add(t0, dt, i));
                //
                core::utctime end = cal->add(t0, dt, n);
                for ( size_t i = 1u; i <= n_ext / 2; ++i )
                    expected_points.push_back(cal->add(end, dt_ext, i));

                FAST_REQUIRE_EQ(res.gt, ta::generic_dt::generic_type::POINT);
                FAST_CHECK_EQ(res.p, ta::point_dt(expected_points));
            }
            SUBCASE("different calendars") {
                std::shared_ptr<core::calendar> other_cal = std::make_shared<core::calendar>(2 * core::calendar::HOUR);

                ta::calendar_dt extension(other_cal, t0, dt, n);
                ta::generic_dt res = ta::extend(axis, extension, cal->add(t0, dt, n / 2));

                // construct time points
                std::vector<utctime> expected_points;
                for ( size_t i = 0u; i <= n; ++i )
                    expected_points.push_back(cal->add(t0, dt, i));

                FAST_REQUIRE_EQ(res.gt, ta::generic_dt::generic_type::POINT);
                FAST_CHECK_EQ(res.p, ta::point_dt(expected_points));
            }
        }
    }
    SUBCASE("continuous with different continuous") {

        std::shared_ptr<core::calendar> utc_ptr = std::make_shared<core::calendar>(utc);

        core::utctime t0 = utc.time(2017, 1, 1);
        core::utctimespan dt_30m = 30 * core::calendar::MINUTE;
        core::utctimespan dt_h = core::calendar::HOUR;
        size_t n = 50;

        SUBCASE("empty cases") {

            ta::fixed_dt empty_fdt = ta::fixed_dt::null_range();
            ta::point_dt empty_pdt = ta::point_dt::null_range();
            ta::calendar_dt non_empty(utc_ptr, t0, dt_h, n);

            SUBCASE("empty with empty") {
                ta::generic_dt res = ta::extend(empty_fdt, empty_pdt, 0);
                FAST_REQUIRE_EQ(res.gt, ta::generic_dt::POINT);
                FAST_CHECK_EQ(res.p.size(), 0);
            }
            SUBCASE("empty with non-empty (split_at == middle of non_empty)") {
                ta::calendar_dt expected(utc_ptr, utc.add(t0, dt_h, n/2), dt_h, n/2);

                ta::generic_dt res = ta::extend(empty_fdt, non_empty, utc.add(t0, dt_h, n/2));

                FAST_REQUIRE_EQ(res.gt, ta::generic_dt::CALENDAR);
                FAST_CHECK_EQ(res.c, expected);
            }
            SUBCASE("non-empty with empty (split_at == non_empty.end)") {
                ta::generic_dt res = ta::extend(non_empty, empty_pdt, utc.add(t0, dt_h, n));

                FAST_REQUIRE_EQ(res.gt, ta::generic_dt::CALENDAR);
                FAST_CHECK_EQ(res.c, non_empty);
            }
        }
        SUBCASE("non-empty cases") {
            SUBCASE("fully before (split between)") {
                ta::fixed_dt ax_fdt(t0, dt_h, n);
                ta::calendar_dt ext_cdt(utc_ptr,
                    utc.add(t0, dt_h, -2*((long)n)),
                    dt_h, n);

                ta::generic_dt res = ta::extend(ax_fdt, ext_cdt, utc.add(t0, dt_h, -(long)n/2));

                FAST_REQUIRE_EQ(res.gt, ta::generic_dt::POINT);
                FAST_CHECK_EQ(res.p, ta::point_dt::null_range());
            }
            SUBCASE("overlapping starting before, ending inside (split at axis start)") {
                std::vector<core::utctime> ext_points;
                core::utctime t0_ext = utc.add(t0, dt_30m, -(long)n/2);
                for ( size_t i = 0; i <= n; ++i ) {
                    ext_points.push_back(utc.add(t0_ext, dt_30m, i));
                }

                ta::calendar_dt ax_cdt(utc_ptr, t0, dt_h, n);
                ta::point_dt ext_pdt(ext_points);

                ta::generic_dt res = ta::extend(ax_cdt, ext_pdt, t0);

                std::vector<core::utctime> exp_points;
                for ( size_t i = 0; i <= n/2; ++i ) {
                    exp_points.push_back(utc.add(t0, dt_30m, i));
                }
                ta::point_dt expected_pdt(exp_points);

                FAST_REQUIRE_EQ(res.gt, ta::generic_dt::POINT);
                FAST_CHECK_EQ(res.p, expected_pdt);
            }
            SUBCASE("overlapping starting before, ending after (split at middle of axis)") {
                std::vector<core::utctime> ax_points;
                for ( size_t i = 0; i <= n; ++i ) {
                    ax_points.push_back(t0 + i * dt_h);
                }

                ta::point_dt ax_pdt(ax_points);
                ta::fixed_dt ext_fdt(t0 - n * dt_h / 2, dt_h, 2 * n);

                ta::generic_dt res = extend(ax_pdt, ext_fdt, t0 + n * dt_h / 2);

                std::vector<core::utctime> exp_points;
                for ( size_t i = 0; i <= 3 * n / 2; ++i ) {
                    exp_points.push_back(utc.add(t0, dt_h, i));
                }
                ta::point_dt expected_pdt(exp_points);

                FAST_REQUIRE_EQ(res.gt, ta::generic_dt::POINT);
                FAST_CHECK_EQ(res.p, expected_pdt);
            }
            SUBCASE("overlapping exactly (split in the middle)") {
                std::vector<core::utctime> ext_points;
                for ( size_t i = 0; i <= 2*n; ++i ) {
                    ext_points.push_back(t0 + i * dt_h / 2);
                }

                ta::fixed_dt ax_fdt(t0, dt_h, n);
                ta::point_dt ext_pdt(ext_points);

                ta::generic_dt res = ta::extend(ax_fdt, ext_pdt, t0 + n * dt_h / 2);

                std::vector<core::utctime> exp_points;
                for ( size_t i = 0; i <= n / 2; ++i ) {
                    exp_points.push_back(t0 + i * dt_h);
                }
                for ( size_t i = 1; i <= n; ++i ) {
                    exp_points.push_back((t0 + n * dt_h / 2) + i * dt_h / 2);
                }
                ta::point_dt expected_pdt(exp_points);

                FAST_REQUIRE_EQ(res.gt, ta::generic_dt::POINT);
                FAST_CHECK_EQ(res.p, expected_pdt);
            }
            SUBCASE("overlapping fully inside (split mid between axis start and extend start)") {
                ta::calendar_dt ax_cdt(utc_ptr, t0, dt_h, n);
                ta::fixed_dt ext_fdt(utc.add(t0, dt_h, n / 5), dt_h, 3 * n / 5);

                ta::generic_dt res = ta::extend(ax_cdt, ext_fdt, utc.add(t0, dt_h, n / 10));

                std::vector<core::utctime> exp_points;
                for ( size_t i = 0; i <= n / 10; ++i ) {
                    exp_points.push_back(utc.add(t0, dt_h, i));
                }
                for ( size_t i = 0; i <= 3 * n / 5; ++i ) {
                    exp_points.push_back(utc.add(t0, dt_h, n / 5) + i * dt_h);
                }
                ta::point_dt expected(exp_points);

                FAST_REQUIRE_EQ(res.gt, ta::generic_dt::POINT);
                FAST_CHECK_EQ(res.p, expected);
            }
            SUBCASE("overlapping starting inside, ending after (split at end of extend)") {
                std::vector<core::utctime> ax_points;
                for ( size_t i = 0; i <= n; ++i ) {
                    ax_points.push_back(utc.add(t0, dt_h, i));
                }

                ta::point_dt ax_pdt(ax_points);
                ta::calendar_dt ext_cdt(utc_ptr, utc.add(t0, dt_h, n / 2), dt_h, n);

                ta::generic_dt res = ta::extend(ax_pdt, ext_cdt, utc.add(t0, dt_h, 3 * n / 2));

                FAST_REQUIRE_EQ(res.gt, ta::generic_dt::POINT);
                FAST_CHECK_EQ(res.p, ax_pdt);
            }
            SUBCASE("fully after (split before axis)") {
                std::vector<core::utctime> ax_points;
                for ( size_t i = 0; i <= n; ++i ) {
                    ax_points.push_back(t0 + i*dt_h);
                }

                std::vector<core::utctime> ext_points;
                for ( size_t i = 0; i <= n; ++i ) {
                    ext_points.push_back((t0 + 2 * n * dt_h) + i * dt_30m);
                }

                ta::point_dt ax_pdt(ax_points);
                ta::point_dt ext_pdt(ext_points);

                ta::generic_dt res = ta::extend(ax_pdt, ext_pdt, t0 - n * dt_h);

                FAST_REQUIRE_EQ(res.gt, ta::generic_dt::POINT);
                FAST_CHECK_EQ(res.p, ext_pdt);
            }
        }
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
