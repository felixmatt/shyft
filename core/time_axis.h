#pragma once
#include "utctime_utilities.h"
namespace shyft {

    /**\brief The time-axis concept is an important component of time-series.
     *time_axis contains all the definitions of time_axis including the combine algorithms.
     *
     *
     *   definition: time-axis is an ordered sequence of non-overlapping periods
     *
     *   notice:
     *      a) the most usual time-axis is the fixed_dt time-axis
     *         in the SHyFT core, this is the one we use(could even need a highspeed-nocompromise version)
     *
     *      b) continuous/dense time-axis: there are no holes in total_period
     *                               types: fixed_dt, calendar_dt, point_dt
     *      c) sparse time-axis: there are holes within the span of total_period
     *                          types: period_list and calendar_dt_p
     */
    namespace time_axis {

        using namespace std;
        using namespace shyft::core;
        /**\brief  time_axis-traits we can use to generate more efficient code
         *
         * For the time_axis::combine family of functions, we use continuous<bool> to
         * select the most efficient combine algorithm.
         */
        template <bool T>
        struct continuous {
            static const bool value = false;
        };
        template<>
        struct continuous<true> {
            static const bool value = true;
        };


        /**\brief a simple regular time-axis, starting at t, with n consecutive periods of fixed length dt
         *
         *  In the shyft::core this is the most useful&fast variant
         */
        struct fixed_dt: continuous<true> {
            utctime t;
            utctimespan dt;
            size_t n;
            fixed_dt( utctime start=no_utctime, utctimespan deltat=0, size_t n_periods=0 ) : t( start ), dt( deltat ), n( n_periods ) {}
            utctimespan delta() const {return dt;}//BW compat
            utctime start() const {return t;} //BW compat
            size_t size() const {return n;}

            utcperiod total_period() const {
                return n == 0 ?
                       utcperiod( min_utctime, min_utctime ) :  // maybe just a non-valid period?
                       utcperiod( t, t + n * dt );
            }

            utctime time( size_t i ) const {
                if( i < n ) return t + i * dt;
                throw std::out_of_range( "fixed_dt.time(i)" );
            }

            utcperiod period( size_t i ) const {
                if( i < n ) return utcperiod( t + i * dt, t + ( i + 1 ) * dt );
                throw std::out_of_range( "fixed_dt.period(i)" );
            }

            size_t index_of( utctime tx ) const {
                if( tx < t ) return std::string::npos;
                size_t r = ( tx - t ) / dt;
                if( r < n ) return r;
                return std::string::npos;
            }
            size_t open_range_index_of( utctime tx, size_t ix_hint=std::string::npos ) const {return n > 0 && ( tx >= t + utctimespan( n * dt ) ) ? n - 1 : index_of( tx );}
            static fixed_dt full_range() {return fixed_dt( min_utctime, max_utctime, 2 );}  //Hmm.
            static fixed_dt null_range() {return fixed_dt( 0, 0, 0 );}
            x_serialize_decl();
        };

        /** A variant of time_axis that adheres to calendar periods, possibly including DST handling
         *  e.g.: a calendar day might be 23,24 or 25 hour long in a DST calendar.
         *  If delta-t is less or equal to one hour, it's close to as efficient as time_axis
         */
        struct calendar_dt :continuous<true> {
            static constexpr utctimespan dt_h = 3600;
            shared_ptr<const calendar> cal;
            utctime t;
            utctimespan dt;
            size_t n;

            calendar_dt() : t( no_utctime ), dt( 0 ), n( 0 ) {}
            calendar_dt( const shared_ptr<const calendar>& cal, utctime t, utctimespan dt, size_t n ) : cal( cal ), t( t ), dt( dt ), n( n ) {}
            calendar_dt(const calendar_dt&c):cal(c.cal),t(c.t),dt(c.dt),n(c.n) {}
            calendar_dt(calendar_dt &&c):cal(std::move(c.cal)),t(c.t),dt(c.dt),n(c.n){}
            calendar_dt& operator=(calendar_dt&&c) {
                cal=std::move(c.cal);
                t=c.t;
                dt=c.dt;
                n=c.n;
                return *this;
            }
            calendar_dt& operator=(const calendar_dt &x) {
                if(this != &x) {
                    cal=x.cal;
                    t=x.t;
                    dt=x.dt;
                    n=x.n;
                }
                return *this;
            }

            size_t size() const {return n;}

            utcperiod total_period() const {
                return n == 0 ?
                       utcperiod( min_utctime, min_utctime ) :  // maybe just a non-valid period?
                       utcperiod( t, dt <= dt_h ? t + n*dt : cal->add( t, dt,long(n) ) );
            }

            utctime time( size_t i ) const {
                if( i < n ) return dt <= dt_h ? t + i * dt : cal->add( t, dt, long(i) );
                throw out_of_range( "calendar_dt.time(i)" );
            }

            utcperiod period( size_t i ) const {
                if( i < n ) return dt <= dt_h ? utcperiod( t + i * dt, t + ( i + 1 ) * dt ) : utcperiod( cal->add( t, dt, long( i) ), cal->add( t, dt, long(i + 1) ) );
                throw out_of_range( "calendar_dt.period(i)" );
            }

            size_t index_of( utctime tx ) const {
                auto p = total_period();
                if( !p.contains( tx ) )
                    return string::npos;
                return dt <= dt_h ?
                       ( size_t )( ( tx - t ) / dt ) :
                       ( size_t ) cal->diff_units( t, tx, dt );
            }
            size_t open_range_index_of( utctime tx, size_t ix_hint = std::string::npos) const {return tx >= total_period().end && n > 0 ? n - 1 : index_of( tx );}
            x_serialize_decl();
        };

        /** \brief point_dt is the most generic dense time-axis.
        *
        * The representation of time-axis, are n time points + end-point,
        * where interval(i) utcperiod(t[i],t[i+1])
        *          except for the last
        *                   utcperiod(t[i],te)
        *
        * very flexible, but inefficient in space and time
        * to minimize the problem the, .index_of() provides
        *  'ix_hint' to hint about the last location used
        *    then if specified, search +- 10 positions to see if we get a hit
        *   otherwise just a binary-search for the correct index.
        *  TODO: maybe even do some smarter partitioning, guessing equidistance on average between points.
        *        then guess the area (+-10), if within range, search there ?
        */
        struct point_dt:continuous<true>{
            vector<utctime> t;
            utctime  t_end;// need one extra, after t.back(), to give the last period!
            point_dt() {}
            point_dt( const vector<utctime>& t, utctime t_end ) : t( t ), t_end( t_end ) {
                //TODO: throw if t.back()>= t_end
                // consider t_end==no_utctime , t_end=t.back()+tick.
                if(t.size()==0 || t.back()>=t_end )
                    throw runtime_error("time_axis::point_dt() illegal initialization parameters");
            }
            point_dt(const vector<utctime>& all_points):t(all_points){
                if(t.size()<2)
                    throw runtime_error("time_axis::point_dt() needs at least two time-points");
				t_end = t.back();
				t.pop_back();

            }
			point_dt(vector<utctime> && all_points):t(move(all_points)) {
				if (t.size() < 2)
					throw runtime_error("time_axis::point_dt() needs at least two time-points");
				t_end = t.back();
				t.pop_back();
			}
            // ms seems to need explicit move etc.
            point_dt(const point_dt&c) : t(c.t), t_end(c.t_end) {}
            point_dt(point_dt &&c) :t(std::move(c.t)), t_end(c.t_end) {}
            point_dt& operator=(point_dt&&c) {
                t = std::move(c.t);
                t_end = c.t_end;
                return *this;
            }
            point_dt& operator=(const point_dt &x) {
                if (this != &x) {
                    t = x.t;
                    t_end = x.t_end;
                }
                return *this;
            }
            size_t size() const {return t.size();}

            utcperiod total_period() const {
                return t.size() == 0 ?
                       utcperiod( min_utctime, min_utctime ) :  // maybe just a non-valid period?
                       utcperiod( t[0], t_end );
            }

            utctime time( size_t i ) const {
                if( i < t.size() ) return t[i];
                throw std::out_of_range( "point_dt.time(i)" );
            }

            utcperiod period( size_t i ) const {
                if( i < t.size() )  return  utcperiod( t[i], i + 1 < t.size() ? t[i + 1] : t_end );
                throw std::out_of_range( "point_dt.period(i)" );
            }

            size_t index_of( utctime tx, size_t ix_hint = std::string::npos ) const {
                if( t.size() == 0 || tx < t[0] || tx >= t_end ) return std::string::npos;
                if( tx >= t.back() ) return t.size() - 1;

                if( ix_hint != std::string::npos && ix_hint < t.size() ) {
                    if( t[ix_hint] == tx ) return ix_hint;
                    const size_t max_directional_search = 10; // just  a wild guess
                    if( t[ix_hint] < tx ) {
                        size_t j = 0;
                        while( t[ix_hint] < tx && ++j < max_directional_search && ix_hint < t.size() ) {
                            ix_hint++;
                        }
                        if( t[ix_hint] >= tx || ix_hint == t.size() )  // we startet below p.start, so we got one to far(or at end), so revert back one step
                            return ix_hint - 1;
                        // give up and fall through to binary-search
                    } else {
                        size_t j = 0;
                        while( t[ix_hint] > tx && ++j < max_directional_search && ix_hint > 0 ) {
                            --ix_hint;
                        }
                        if( t[ix_hint] > tx && ix_hint > 0 )  // if we are still not before p.start, and i is >0, there is a hope to find better index, otherwise we are at/before start
                            ; // bad luck searching downward, need to use binary search.
                        else
                            return ix_hint;
                    }
                }

                auto r = lower_bound( t.cbegin(), t.cend(), tx,[]( utctime pt, utctime val ) { return pt <= val; } );
                return static_cast<size_t>( r - t.cbegin() ) - 1;
            }
            size_t open_range_index_of( utctime tx, size_t ix_hint = std::string::npos) const {return size() > 0 && tx >= t_end ? size() - 1 : index_of( tx,ix_hint );}

            static point_dt null_range() {
                return point_dt{};
            }
            x_serialize_decl();
        };

        /** \brief a generic (not sparse) time interval time-axis.
         *
         * This is a static dispatch generic time-axis for all dense time-axis.
         * It's merely utilizing the three other types to do the implementation.
         * It's useful when combining time-axis, and we would like to keep
         * the internal rep. to the most efficient as determined at runtime.
         */
        struct generic_dt:continuous<true> {
            //--possible implementation types:
            enum generic_type { FIXED = 0, CALENDAR = 1, POINT = 2};
            generic_type gt;
            fixed_dt f;
            calendar_dt c;
            point_dt p;
            //---------------
            generic_dt(): gt( FIXED ) {}
            // provide convinience constructors, to directly create the wanted time-axis, regardless underlying rep.
            generic_dt( utctime t0,utctimespan dt,size_t n):gt(FIXED),f(t0,dt,n) {}
            generic_dt( const shared_ptr<const calendar>& cal, utctime t, utctimespan dt, size_t n ) : gt(CALENDAR),c(cal,t,dt,n) {}
            generic_dt( const vector<utctime>& t, utctime t_end ):gt(POINT),p(t,t_end) {}
            generic_dt( const vector<utctime>& all_points):gt(POINT),p(all_points){}
            // --
            generic_dt( const fixed_dt&f ): gt( FIXED ), f( f ) {}
            generic_dt( const calendar_dt &c ): gt( CALENDAR ), c( c ) {}
            generic_dt( const point_dt& p ): gt( POINT ), p( p ) {}
            // -- need move,ct etc for msc++
            // ms seems to need explicit move etc.
            generic_dt(const generic_dt&cc) : gt(cc.gt),f(cc.f),c(cc.c),p(cc.p) {}
            generic_dt(generic_dt &&cc) :gt(cc.gt),f(std::move(cc.f)), c(std::move(cc.c)), p(std::move(cc.p)) {}
            generic_dt& operator=(generic_dt&&cc) {
                gt = cc.gt;
                f = std::move(cc.f);
                c = std::move(cc.c);
                p = std::move(cc.p);
                return *this;
            }
            generic_dt& operator=(const generic_dt &x) {
                if (this != &x) {
                    gt = x.gt;
                    f = x.f;
                    c = x.c;
                    p = x.p;
                }
                return *this;
            }

            //--
            bool is_fixed_dt() const {return gt != POINT;}

            size_t size() const          {switch( gt ) {default: case FIXED: return f.size(); case CALENDAR: return c.size(); case POINT: return p.size();}}
            utcperiod total_period() const  {switch( gt ) {default: case FIXED: return f.total_period(); case CALENDAR: return c.total_period(); case POINT: return p.total_period();}}
            utcperiod period( size_t i ) const {switch( gt ) {default: case FIXED: return f.period( i ); case CALENDAR: return c.period( i ); case POINT: return p.period( i );}}
            utctime     time( size_t i ) const {switch( gt ) {default: case FIXED: return f.time( i ); case CALENDAR: return c.time( i ); case POINT: return p.time( i );}}
            size_t index_of( utctime t ) const {switch( gt ) {default: case FIXED: return f.index_of( t ); case CALENDAR: return c.index_of( t ); case POINT: return p.index_of( t );}}
            size_t open_range_index_of( utctime t, size_t ix_hint = std::string::npos) const {switch( gt ) {default: case FIXED: return f.open_range_index_of( t ); case CALENDAR: return c.open_range_index_of( t ); case POINT: return p.open_range_index_of( t,ix_hint );}}
            x_serialize_decl();
        };

        /** create a new time-shifted dt time-axis */
        inline fixed_dt time_shift(const fixed_dt &src, utctimespan dt) {
            return fixed_dt(src.t+dt,src.dt,src.n);
        }

        /** create a new time-shifted dt time-axis */
        inline calendar_dt time_shift(const calendar_dt& src,utctimespan dt) {
            calendar_dt r(src);
            r.t+=dt;
            return r;
        }

        /** create a new time-shifted dt time-axis */
        inline point_dt time_shift(const point_dt& src, utctimespan dt) {
            point_dt r(src);
            for(auto& t: r.t) t+=dt; // potential cost, we could consider other approaches with refs..
            r.t_end+=dt;
            return r;
        }

        /** create a new time-shifted dt time-axis */
        inline generic_dt time_shift(const generic_dt&src, utctimespan dt) {
            if(src.gt==generic_dt::FIXED) return generic_dt(time_shift(src.f,dt));
            if(src.gt==generic_dt::CALENDAR) return generic_dt(time_shift(src.c,dt));
            return generic_dt(time_shift(src.p,dt));
        }


        /** \brief Yet another variant of calendar time-axis.
         *
         * This one is similar to calendar_dt, except, each main-period given by
         * calendar_dt have sub-periods.
         * E.g. you would like to have a weekly time-axis that represents all working-hours
         * etc.
         * If the sub-period(s) specified are null, then this time_axis is equal to calendar_dt.
         * \note this is a non-continuous| sparse time-axis concept
         */
        struct calendar_dt_p:continuous<false> {
            calendar_dt cta;
            vector<utcperiod> p;// sub-periods within each cta.period, using cta.period.start as t0
            calendar_dt_p(){}
            calendar_dt_p( const shared_ptr<const calendar>& cal, utctime t, utctimespan dt, size_t n, vector<utcperiod> p )
                : cta( cal, t, dt, n ), p( move( p ) ) {
                // TODO: validate p, each p[i] non-overlapping and within ~ dt
                // possibly throw if invalid period, but
                // this could be tricky, because the 'gross period' will vary over the timeaxis,
                // so worst case we would have to run through all gross-periods and verify that
                // sub-periods fits-within each gross period.

            }
            calendar_dt_p(const calendar_dt_p&c):cta(c.cta),p(c.p) {}
            calendar_dt_p(calendar_dt_p &&c):cta(std::move(c.cta)),p(std::move(c.p)){}
            calendar_dt_p& operator=(calendar_dt_p&&c) {
                cta=std::move(c.cta);
                p=std::move(c.p);
                return *this;
            }
            calendar_dt_p& operator=(const calendar_dt_p &x) {
                if(this != &x) {
                    cta=x.cta;
                    p=x.p;
                }
                return *this;
            }

            size_t size() const { return cta.size() * ( p.size() ? p.size() : 1 );}

            utcperiod total_period() const {
                if( p.size() == 0 || cta.size() == 0 )
                    return cta.total_period();
                return utcperiod( period( 0 ).start, period( size() - 1 ).end );
            }

            utctime time( size_t i ) const { return period( i ).start;}

            utcperiod period( size_t i ) const {
                if( i < size() ) {
                    size_t main_ix = i / p.size();
                    size_t p_ix = i - main_ix * p.size();
                    auto pi = cta.period( main_ix );
                    return p.size() ?
                           utcperiod( cta.cal->add( pi.start, p[p_ix].start, 1 ), cta.cal->add( pi.start, p[p_ix].end, 1 ) ) :
                           pi;
                }
                throw out_of_range( "calendar_dt_p.period(i)" );
            }

            size_t index_of( utctime tx, bool period_containment = true ) const {
                auto tp = total_period();
                if( !tp.contains( tx ) )
                    return string::npos;
                size_t main_ix = cta.index_of( tx );
                if( p.size() == 0 )
                    return main_ix;

                utctime t0 = cta.time( main_ix );
                for( size_t i = 0; i < p.size(); ++i ) {  //important: p is assumed to have size like 5, 7 (workdays etc).
                    utcperiod pi( cta.cal->add( t0, p[i].start, 1 ), cta.cal->add( t0, p[i].end, 1 ) );
                    if( pi.contains( tx ) )
                        return p.size() * main_ix + i;
                    if( !period_containment ) {  // if just searching for the closest period back in time, then:
                        if( pi.start > tx )  // the previous period is before tx, then we know we have a hit
                            return p.size() * main_ix + ( i > 0 || main_ix > 0 ? i - 1 : 0 ); // it might be a hit just before first sub-period as well. Kind of special case(not consistent,but useful)
                        if( i + 1 == p.size() )  // the current period is the last one, then we also know we have a hit
                            return p.size() * main_ix + i;

                    }
                }

                return string::npos;
            }
            size_t open_range_index_of( utctime tx , size_t ix_hint = std::string::npos) const {
                return size() > 0 && tx >= total_period().end ? size() - 1 : index_of( tx, false );
            }
        };




        /**\brief The "by definition" ultimate time-axis using a list of periods
         *
         * an ordered sequence of non-overlapping periods
         *
         */
        struct period_list:continuous<false>{
            vector<utcperiod> p;
            period_list() {}
            period_list( const vector<utcperiod>& p ) : p( p ) {
                //TODO: high cost, consider doing this in DEBUG only!
                for(size_t i=1;i<p.size();++i) {
                    if(p[i].start<p[i-1].end)
                        throw runtime_error("period_list:periods should be ordered and non-overlapping");
                }
            }
            template< class TA>
            static period_list convert( const TA& ta ) {
                period_list r;
                r.p.reserve( ta.size() );
                for( size_t i = 0; i < ta.size(); ++i )
                    r.p.push_back( ta.period( i ) );
                return r;
            }
            size_t size() const {return p.size();}

            utcperiod total_period() const {
                return p.size() == 0 ?
                       utcperiod( min_utctime, min_utctime ) :  // maybe just a non-valid period?
                       utcperiod( p.front().start, p.back().end );
            }

            utctime time( size_t i ) const {
                if( i < p.size() ) return p[i].start;
                throw std::out_of_range( "period_list.time(i)" );
            }

            utcperiod period( size_t i ) const {
                if( i < p.size() )  return  p[i];;
                throw std::out_of_range( "period_list.period(i)" );
            }

            size_t index_of( utctime tx, size_t ix_hint = std::string::npos, bool period_containment = true ) const {
                if( p.size() == 0 || tx < p.front().start || tx >= p.back().end ) return std::string::npos;

                if( ix_hint != string::npos && ix_hint < p.size() ) {
                    if( ( period_containment && p[ix_hint].contains( tx ) ) ||
                            ( !period_containment && ( p[ix_hint].start >= tx && ( ix_hint + 1 < p.size() ? tx < p[ix_hint + 1].start : true ) ) ) ) return ix_hint;
                    const size_t max_directional_search = 10; // just  a wild guess
                    if( p[ix_hint].end < tx ) {
                        size_t j = 0;
                        while( p[ix_hint].end < tx && ++j < max_directional_search && ix_hint < p.size() ) {
                            ix_hint++;
                        }
                        if( ix_hint == p.size() )
                            return string::npos;
                        if( p[ix_hint].contains( tx ) )   // ok this is it.
                            return ix_hint;
                        else if( !period_containment && ix_hint + 1 < p.size() && tx >= p[ix_hint].start && tx < p[ix_hint + 1].start )
                            return ix_hint;// this is the closest period <= to tx
                        // give up and fall through to binary-search
                    } else {
                        size_t j = 0;
                        while( p[ix_hint].start > tx && ++j < max_directional_search && ix_hint > 0 ) {
                            --ix_hint;
                        }
                        if( p[ix_hint].contains( tx ) )
                            return ix_hint;
                        else if( !period_containment && ix_hint + 1 < p.size() && tx >= p[ix_hint].start && tx < p[ix_hint + 1].start )
                            return ix_hint;// this is the closest period <= to tx

                        // give up, binary search.
                    }
                }

                auto r = lower_bound( p.cbegin(), p.cend(), tx,
                []( utcperiod pt, utctime val ) { return pt.start <= val; } );
                size_t ix = static_cast<size_t>( r - p.cbegin() ) - 1;
                if( ix == string::npos || p[ix].contains( tx ) )
                    return ix; //cover most cases, including period containment
                if( !period_containment && tx >= p[ix].start )
                    return ix;// ok this is the closest period that matches
                return string::npos;// no match to period,what so ever
            }
            size_t open_range_index_of( utctime tx, size_t ix_hint = std::string::npos) const {
                return size() > 0 && tx >= total_period().end ? size() - 1 : index_of( tx, ix_hint, false );
            }

            static period_list null_range() {
                return period_list();
            }
        };

        /** \brief fast&efficient combine for two fixed_dt time-axis */
        inline fixed_dt combine( const fixed_dt& a, const fixed_dt& b )  {
            // 0. check if they overlap (todo: this could hide dt-errors checked for later)
            utcperiod pa = a.total_period();
            utcperiod pb = b.total_period();
            if( !pa.overlaps( pb ) || a.size() == 0 || b.size() == 0 )
                return fixed_dt::null_range();
            if( a.dt == b.dt ) {
                if( a.t == b.t && a.n == b.n ) return a;
                utctime t0 = max( pa.start, pb.start );
                return fixed_dt( t0, a.dt, ( min( pa.end, pb.end ) - t0 ) / a.dt );
            } if( a.dt > b.dt ) {
                if( ( a.dt % b.dt ) != 0 ) throw std::runtime_error( "combine(fixed_dt a,b) needs dt to align" );
                utctime t0 = max( pa.start, pb.start );
                return fixed_dt( t0, b.dt, ( min( pa.end, pb.end ) - t0 ) / b.dt );
            } else {
                if( ( b.dt % a.dt ) != 0 )
                    throw std::runtime_error( "combine(fixed_dt a,b) needs dt to align" );
                utctime t0 = max( pa.start, pb.start );
                return fixed_dt( t0, a.dt, ( min( pa.end, pb.end ) - t0 ) / a.dt );
            }
        }

        /** \brief combine continuous (time-axis,time-axis) template
         * for combining any continuous time-axis with another continuous time-axis
         * \note this could have potentially linear cost of n-points
         */
        template<class TA, class TB>
        inline generic_dt combine( const TA& a, const TB & b, typename enable_if < TA::continuous::value && TB::continuous::value >::type* x = 0 ) {
            utcperiod pa = a.total_period();
            utcperiod pb = b.total_period();
            if( !pa.overlaps( pb ) || a.size() == 0 || b.size() == 0 )
                return generic_dt( point_dt::null_range() );
            if( pa == pb && a.size() == b.size() ) {  //possibly exact equal ?
                bool all_equal = true;
                for( size_t i = 0; i < a.size(); ++i ) {
                    if( a.period( i ) != b.period( i ) ) {
                        all_equal = false; break;
                    }
                }
                if( all_equal )
                    return generic_dt( a );
            }
            // the hard way merge points in the intersection of periods
            utctime t0 = std::max( pa.start, pb.start );
            utctime te = std::min( pa.end, pb.end );
            size_t ia = a.open_range_index_of( t0 );// first possible candidate from a
            size_t ib = b.open_range_index_of( t0 );// first possible candidate from b
            size_t ea = 1 + a.open_range_index_of( te );// one past last possible candidate from a
            size_t eb = 1 + b.open_range_index_of( te );// one past last possible candidate from b
            point_dt r;// result generic type for dense time-axis
            r.t.reserve( ( ea - ia ) + ( eb - ib ) );  //assume worst case here, avoid realloc
            r.t_end = te;// last point set

            while( ia < ea && ib < eb ) {
                utctime ta = a.time( ia );
                utctime tb = b.time( ib );

                if( ta == tb ) {
                    r.t.push_back( ta ); ++ia; ++ib;  // common point,push&incr. both
                } else if( ta < tb ) {
                    r.t.push_back( ta ); ++ia;  // a contribution only, incr. a
                } else {
                    r.t.push_back( tb ); ++ib;  // b contribution only, incr. b
                }
            }
            // a or b (or both) are empty for time-points, we need to fill up remaining < te
            if( ia < ea ) {  // more to fill in from a ?
                while( ia < ea ) {
                    auto t_i = a.time( ia++ );
                    if( t_i < te ) r.t.push_back( t_i );
                }
            } else { // more to fill in from b ?
                while( ib < eb ) {
                    auto t_i = b.time( ib++ );
                    if( t_i < te ) r.t.push_back( t_i );
                }
            }

            if( r.t.back() == r.t_end )  // make sure we leave t_end as the last point.
                r.t.pop_back();
            return generic_dt( r );
        }

        /**\brief TODO write the combine sparse time-axis algorithm
         *  for calendar_p_dt and period-list.
         * period_list+period_list -> period_list
         *
         */
        template<class TA, class TB>
        inline period_list combine( const TA& a, const TB & b, typename enable_if < !TA::continuous::value || !TB::continuous::value >::type* x = 0 ) {
            utcperiod pa = a.total_period();
            utcperiod pb = b.total_period();
            if( !pa.overlaps( pb ) || a.size() == 0 || b.size() == 0 )
                return period_list::null_range();
            if( pa == pb && a.size() == b.size() ) {  //possibly exact equal ?
                bool all_equal = true;
                for( size_t i = 0; i < a.size(); ++i ) {
                    if( a.period( i ) != b.period( i ) ) {
                        all_equal = false; break;
                    }
                }
                if( all_equal )
                    return period_list::convert( a );
            }

            // the hard way merge points in the intersection of periods
            utctime t0 = std::max( pa.start, pb.start );
            utctime te = std::min( pa.end, pb.end );
            utcperiod tp( t0, te );
            size_t ia = a.open_range_index_of( t0 );  //notice these have to be non-nan!
            size_t ib = b.open_range_index_of( t0 );
            if( ia == string::npos || ib == string::npos )
                throw runtime_error( "period_list::combine algorithmic error npos not expected here" );
            size_t ea = 1 + a.open_range_index_of( te );
            size_t eb = 1 + b.open_range_index_of( te );
            period_list r;
            r.p.reserve( ( ea - ia ) + ( eb - ib ) );  //assume worst case here, avoid realloc

            while( ia < ea && ib < eb ) {  // while both do have contributions
                utcperiod p_ia = intersection( a.period( ia ), tp );
                utcperiod p_ib = intersection( b.period( ib ), tp );
                if( p_ia.timespan() == 0 ) {++ia; continue;}  // no contribution from a, skip to next a
                if( p_ib.timespan() == 0 ) {++ib; continue;}  // no contribution from b, skip to next b
                utcperiod p_i = intersection( p_ia, p_ib );  // compute the intersection
                if( p_i.timespan() == 0 ) {  // no overlap|intersection
                    if( p_ia.start < p_ib.start ) ++ia;  //advance the left-most interval
                    else if( p_ib.start < p_ia.start ) ++ib;
                    else {++ia; ++ib;} //TODO: should not be possible ? since it's not overlapping start cant be equal(except one is empty period)
                    continue;
                }

                if( p_ia == p_ib ) {
                    r.p.push_back( p_ia ); ++ia; ++ib;  // common period, push&incr. both
                } else if( p_ib.contains( p_ia ) )  {
                    r.p.push_back( p_ia ); ++ia;  // a contribution only, incr. a
                } else if( p_ia.contains( p_ib ) ) {
                    r.p.push_back( p_ib ); ++ib;  // b contribution only, incr. b
                } else { //ok, not equal, and neither a or b contains the other
                    if( p_ia.start < p_ib.start ) {
                        r.p.push_back( p_i ); ++ia;  // a contribution only, incr. a
                    } if( p_ib.start < p_ia.start ) {
                        r.p.push_back( p_i ); ++ib;  // b contribution only, incr. b
                    }
                }
            }
            return r;
        }


        /** \brief time-axis combine type deduction system for combine algorithm
         *
         * The goal here is to deduce the fastest possible representation type of
         * two time-axis to combine.
         */
        template <typename T_A, typename T_B, typename C = void >
        struct combine_type { // generic fallback to period_list type, very general, but expensive
            typedef period_list type;
        };

        /** specialization for fixed_dt at max speed */
        template<>
        struct combine_type<fixed_dt, fixed_dt, void> {typedef fixed_dt type;};

        /** specialization for all continuous time_axis types */
        template<typename T_A, typename T_B> // then take care of all the continuous type of time-axis, they all goes into generic_dt type
        struct combine_type < T_A, T_B, typename enable_if < T_A::continuous::value && T_B::continuous::value >::type > {typedef generic_dt type;};
    }
}
//--serialization support
x_serialize_export_key(shyft::time_axis::fixed_dt);
x_serialize_export_key(shyft::time_axis::calendar_dt);
x_serialize_export_key(shyft::time_axis::point_dt);
x_serialize_export_key(shyft::time_axis::generic_dt);

