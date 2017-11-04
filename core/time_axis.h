#pragma once

#include <string>
#include <stdexcept>
#include <vector>
#include <memory>
#include <utility>
#include <stdexcept>
#include <type_traits>
#include <algorithm>
#include "core_pch.h"
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

		/** generic test if two different time-axis types resembles the same conceptual time-axis
		*
		* Given time-axis are of differnt types (point, versus period versus fixed_dt etc.)
		* just compare and see if they produces the same number of periods, and that each
		* period is equal.
		*/
		template<class A, class B>
		bool equivalent_time_axis(const A& a, const B& b) {
			if (a.size() != b.size())
				return false;
			for (size_t i = 0;i < a.size();++i) { if (a.period(i) != b.period(i)) return false; }
			return true;
		}
		/** Specialization of the equivalent_time_axis given that they are of the same type.
		*  In this case we forward the comparison to the type it self relying on the
		*  fact that the time_axis it self knows how to fastest figure out if it's equal.
		*/
		template <class A>
		bool equivalent_time_axis(const A& a, const A&b) { return a == b; }

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
			bool operator==(const fixed_dt& other) const {return t==other.t && dt== other.dt && n==other.n;}
			bool operator!=(const fixed_dt& other) const { return !this->operator==(other); }
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
                if( tx < t || dt == 0 ) return std::string::npos;
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
        struct calendar_dt : continuous<true> {

            static constexpr utctimespan dt_h = 3600;

            shared_ptr<calendar> cal;
            utctime t;
            utctimespan dt;
            size_t n;

            shared_ptr<calendar> get_calendar() const {
                return cal;
            }

            calendar_dt()
                : t( no_utctime ),
                dt( 0 ),
                n( 0 ) { }
            calendar_dt(const shared_ptr< calendar> & cal,
                utctime t,
                utctimespan dt,
                size_t n)
                : cal( cal ),
                t( t ),
                dt( dt ),
                n( n ) { }
            calendar_dt(const calendar_dt & c)
                : cal( c.cal ),
                t( c.t ),
                dt( c.dt ),
                n( c.n ) { }
            calendar_dt(calendar_dt && c)
                : cal( std::move(c.cal) ),
                t( c.t ),
                dt( c.dt ),
                n( c.n ) { }

            calendar_dt & operator=(calendar_dt && c) {
                cal = std::move(c.cal);
                t = c.t;
                dt = c.dt;
                n = c.n;
                return *this;
            }
            calendar_dt & operator=(const calendar_dt & x) {
                if ( this != &x ) {
                    cal = x.cal;
                    t = x.t;
                    dt = x.dt;
                    n = x.n;
                }
                return *this;
            }
            /** equality, notice that calendar is equal if they refer to exactly same calendar pointer */
            bool operator==(const calendar_dt & other) const {
                return (cal.get() == other.cal.get() || cal->tz_info->name()== other.cal->tz_info->name())
                    && t == other.t
                    && dt == other.dt
                    && n == other.n;
            }
            bool operator!=(const calendar_dt & other) const {
                return !this->operator == (other);
            }

            size_t size() const {
                return n;
            }

            utcperiod total_period() const {
                return n == 0
                    ? utcperiod(min_utctime, min_utctime)  // maybe just a non-valid period?
                    : utcperiod(t, dt <= dt_h ? t + n*dt : cal->add(t, dt, long(n)));
            }

            utctime time(size_t i) const {
                if ( i < n ) {
                    return dt <= dt_h
                        ? t + i * dt
                        : cal->add(t, dt, long(i));
                }
                throw out_of_range("calendar_dt.time(i)");
            }

            utcperiod period(size_t i) const {
                if ( i < n ) {
                    return dt <= dt_h
                        ? utcperiod(t + i * dt, t + (i + 1) * dt)
                        : utcperiod(cal->add(t, dt, static_cast<long>(i)), cal->add(t, dt, static_cast<long>(i + 1)));
                }
                throw out_of_range("calendar_dt.period(i)");
            }

            size_t index_of(utctime tx) const {
                auto p = total_period();
                if ( !p.contains(tx) )
                    return std::string::npos;  // why string...? Introduce a static constant + check similar classes
                return dt <= dt_h
                    ? static_cast<size_t>((tx - t) / dt)
                    : static_cast<size_t>(cal->diff_units(t, tx, dt));
            }

            size_t open_range_index_of(utctime tx, size_t ix_hint = std::string::npos) const {
                return tx >= total_period().end && n > 0
                    ? n - 1
                    : index_of( tx );
            }

            static calendar_dt null_range() {
                return calendar_dt();
            }

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
            point_dt()
                : t( vector<utctime>{} ),
                  t_end( no_utctime ) {}
            point_dt( const vector<utctime>& t, utctime t_end ) : t( t ), t_end( t_end ) {
                //TODO: throw if t.back()>= t_end
                // consider t_end==no_utctime , t_end=t.back()+tick.
                if(t.size()==0 || t.back()>=t_end )
                    throw runtime_error("time_axis::point_dt() illegal initialization parameters");
            }
            explicit point_dt(const vector<utctime>& all_points):t(all_points){
                if(t.size()<2)
                    throw runtime_error("time_axis::point_dt() needs at least two time-points");
				t_end = t.back();
				t.pop_back();

            }
			explicit point_dt(vector<utctime> && all_points):t(move(all_points)) {
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
            bool operator==(const point_dt &other)const {return t == other.t && t_end == other.t_end;}
			bool operator!=(const point_dt& other) const { return !this->operator==(other); }
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
                return point_dt();
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
        struct generic_dt : continuous<true> {

            /** \brief Possible time-axis types.
             */
            enum generic_type:int8_t {
                FIXED = 0,     /**< Represents storage of fixed_dt. */
                CALENDAR = 1,  /**< Represents storage of calendar_dt. */
                POINT = 2      /**< Represents storage of point_dt. */
            };

            generic_type gt;

            fixed_dt f;
            calendar_dt c;
            point_dt p;

            generic_dt() : gt( FIXED ) { }
            // provide convinience constructors, to directly create the wanted time-axis, regardless underlying rep.
            generic_dt(utctime t0, utctimespan dt, size_t n)
                : gt(FIXED),
                  f(t0, dt, n) { }
            generic_dt(const shared_ptr<calendar> & cal, utctime t, utctimespan dt, size_t n)
                : gt(CALENDAR),
                  c(cal, t, dt, n) { }
            generic_dt(const vector<utctime> & t, utctime t_end)
                : gt(POINT),
                  p(t, t_end) { }
            explicit generic_dt(const vector<utctime> & all_points)
                : gt(POINT),
                  p(all_points) { }

            explicit generic_dt(const fixed_dt&f)
                : gt(FIXED),
                  f(f) { }
            explicit generic_dt(const calendar_dt &c)
                : gt(CALENDAR),
                  c(c) { }
            explicit generic_dt(const point_dt& p)
                : gt(POINT),
                  p(p) { }

            // -- need move,ct etc for msc++
            // ms seems to need explicit move etc.
            generic_dt(const generic_dt&cc)
                : gt(cc.gt),
                  f(cc.f),
                  c(cc.c),
                  p(cc.p) { }
            generic_dt(generic_dt &&cc)
                : gt(cc.gt),
                  f(std::move(cc.f)),
                  c(std::move(cc.c)),
                  p(std::move(cc.p)) { }
            generic_dt& operator=(generic_dt&&cc) {
                gt = cc.gt;
                f = std::move(cc.f);
                c = std::move(cc.c);
                p = std::move(cc.p);
                return *this;
            }
            generic_dt& operator=(const generic_dt &x) {
                if ( this != &x ) {
                    gt = x.gt;
                    f = x.f;
                    c = x.c;
                    p = x.p;
                }
                return *this;
            }
            bool operator==(const generic_dt& other) const {
                if ( gt != other.gt ) {// they are represented differently:
                    switch ( gt ) {
                    default:
                    case FIXED:    return equivalent_time_axis(f, other);
                    case CALENDAR: return equivalent_time_axis(c, other);
                    case POINT:    return equivalent_time_axis(p, other);
                    }
                } // else they have same-representation, use equality directly
                switch ( gt ) {
                default:
                case FIXED:    return f == other.f;
                case CALENDAR: return c == other.c;
                case POINT:    return p == other.p;
                }
            }
			bool operator!=(const generic_dt& other) const {
                return ! this->operator==(other);
            }

            bool is_fixed_dt() const {
                return gt != POINT;
            }

            size_t size() const {
                switch( gt ) {
                default:
                case FIXED:    return f.size();
                case CALENDAR: return c.size();
                case POINT:    return p.size();
                }
            }
            utcperiod total_period() const {
                switch ( gt ) {
                default:
                case FIXED:    return f.total_period();
                case CALENDAR: return c.total_period();
                case POINT:    return p.total_period();
                }
            }
            utcperiod period(size_t i) const {
                switch( gt ) {
                default:
                case FIXED:    return f.period(i);
                case CALENDAR: return c.period(i);
                case POINT:    return p.period(i);
                }
            }
            utctime time(size_t i) const {
                switch( gt ) {
                default:
                case FIXED:    return f.time(i);
                case CALENDAR: return c.time(i);
                case POINT:    return p.time(i);
                }
            }
            size_t index_of(utctime t, size_t ix_hint=std::string::npos) const {
                switch( gt ) {
                default:
                case FIXED:    return f.index_of(t);
                case CALENDAR: return c.index_of(t);
                case POINT:    return p.index_of(t, ix_hint);
                }
            }
            size_t open_range_index_of(utctime t, size_t ix_hint = std::string::npos) const {
                switch ( gt ) {
                default:
                case FIXED:    return f.open_range_index_of(t);
                case CALENDAR: return c.open_range_index_of(t);
                case POINT:    return p.open_range_index_of(t, ix_hint);
                }
            }

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
            calendar_dt_p( const shared_ptr< calendar>& cal, utctime t, utctimespan dt, size_t n, const vector<utcperiod>& p )
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
			bool operator==(const calendar_dt_p& other)const { return p==other.p && cta== other.cta;}
			bool operator!=(const calendar_dt_p& other) const { return !this->operator==(other); }
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
            explicit period_list( const vector<utcperiod>& p ) : p( p ) {
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
            bool operator==(const period_list& other) const {return p == other.p;}
			bool operator!=(const period_list& other) const { return !this->operator==(other); }
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

        // don't leak from the compilation unit
        namespace {

            /** \brief Helper handling special actions for different time-axes.
            *
            * Specialized on the different continuous time axes: fixed_dt, calendar_dt, point_dt, and generic_dt.
            */
            template<class T> struct extend_helper;

            template<>
            struct extend_helper<fixed_dt> {
                /** \brief Wrap the supplied time-axis as a generic_dt.
                *
                * \warning Use with caution, there is not bounds checking done!
                *
                * \param base   Time-axis.
                * \param skip   Number of intervals to skip from the start.
                * \param steps  Number of intervals to include. Counted from the first included interval.
                */
                static generic_dt as_generic(const fixed_dt & base, size_t skip, size_t steps) {
                    return generic_dt(fixed_dt(base.t + skip*base.dt, base.dt, steps));
                }
            };
            template<>
            struct extend_helper<calendar_dt> {
                /** \brief Wrap the supplied time-axis as a generic_dt.
                *
                * \warning Use with caution, there is not bounds checking done!
                *
                * \param base   Time-axis.
                * \param skip   Number of intervals to skip from the start.
                * \param steps  Number of intervals to include. Counted from the first included interval.
                */
                static generic_dt as_generic(const calendar_dt & base, size_t skip, size_t steps) {
                    return generic_dt(calendar_dt(base.cal, base.cal->add(base.t, base.dt, skip), base.dt, steps));
                }
            };
            template<>
            struct extend_helper<point_dt> {
                /** \brief Wrap the supplied time-axis as a generic_dt.
                *
                * \warning Use with caution, there is not bounds checking done!
                *
                * \param base   Time-axis.
                * \param skip   Number of intervals to skip from the start.
                * \param steps  Number of intervals to include. Counted from the first included interval.
                */
                static generic_dt as_generic(const point_dt & base, size_t skip, size_t steps) {
                    auto it_begin = base.t.cbegin(); std::advance(it_begin, skip);
                    auto it_end = it_begin;          std::advance(it_end, steps);

                    utctime end_time = base.t_end;
                    if ( it_end != base.t.cend() ) {
                        end_time = base.t[skip + steps + 1];
                    }

                    return generic_dt(point_dt(std::vector<core::utctime>(it_begin, it_end), end_time));
                }
            };
            template<>
            struct extend_helper<generic_dt> {
                /** \brief Wrap the supplied time-axis as a generic_dt.
                *
                * \warning Use with caution, there is not bounds checking done!
                *
                * \param base   Time-axis.
                * \param skip   Number of intervals to skip from the start.
                * \param steps  Number of intervals to include. Counted from the first included interval.
                */
                static generic_dt as_generic(const generic_dt & base, size_t skip, size_t steps) {
                    switch ( base.gt ) {
                    case generic_dt::FIXED:    return extend_helper<fixed_dt>::as_generic(base.f, skip, steps);
                    case generic_dt::CALENDAR: return extend_helper<calendar_dt>::as_generic(base.c, skip, steps);
                    case generic_dt::POINT:    return extend_helper<point_dt>::as_generic(base.p, skip, steps);
                    }
                }
            };

        }

        /** \brief Extend time-axis `a` with time-axis `b`.
         *
         *
         *
         * Values are only added after time-axis `a`, never inside.
         *
         * \param a  Time-axis to extend.
         * \param b  Time-axis to extend.
         * \param split_at  Time-point to split between `a` and `b`.
         *   If at a interval boundary in `a` the interval is _not_ included.
         *   If inside a interval in `a` the interval is included.
         *   If at a interval boundary in `b` the interval is included.
         *   If inside a interval in `b` the interval is _not_ included.
         */
        inline generic_dt extend(const fixed_dt & a, const fixed_dt & b, const utctime split_at) {
            const utcperiod pa = a.total_period();
            const utcperiod pb = b.total_period();

            {
                const size_t asz = a.size();
                const size_t bsz = b.size();

                // trivial cases
                if ( asz == 0 || bsz == 0 ) {
                    // - both empty -> return empty range
                    if ( asz == 0 && bsz == 0 ) {
                        return generic_dt(fixed_dt::null_range());
                    }
                    // - one empty -> return non-empty sliced at split_at
                    else if ( asz == 0 ) {
                        size_t split_index = b.index_of(split_at);
                        if ( split_index == std::string::npos ) {
                            if ( split_at < pb.start ) {
                                return generic_dt(b);
                            } else {
                                return generic_dt(fixed_dt::null_range());
                            }
                        } else {
                            utcperiod split_p = b.period(split_index);
                            return generic_dt(fixed_dt(
                                split_p.start,
                                b.dt, bsz - split_index));
                        }
                    } else {
                        size_t split_index = a.index_of(split_at);
                        if ( split_index == std::string::npos ) {
                            if ( split_at < pa.start ) {
                                return generic_dt(fixed_dt::null_range());
                            } else {
                                return generic_dt(a);
                            }
                        } else {
                            return generic_dt(fixed_dt(
                                pa.start,
                                a.dt, split_index));
                        }
                    }
                }
            }

            // sliced spans for a and b
            const utcperiod sa(  // span a
                    pa.start,
                    min(max(pa.start + ((split_at - pa.start) / a.dt) * a.dt, pa.start), pa.end) );
            const utcperiod sb(  // span b
                    max(min(pb.start + ((split_at - pb.start) / b.dt) * b.dt, pb.end), pb.start),
                    pb.end );

            // aligned and consecutive
            if (
                a.dt == b.dt && pa.start == pb.start + ((pa.start - pb.start) / b.dt)*b.dt  // aligned
                && (sa.start == sa.end || sb.start == sb.end || sa.end == sb.start)  // consecutive
            ) {
                if ( sa.start != sa.end ) {  // non-empty
                    if ( sb.start != sb.end ) {  // non-empty
                        return generic_dt(fixed_dt(sa.start, a.dt, (sb.end - sa.start) / a.dt));
                    } else {
                        return generic_dt(fixed_dt(sa.start, a.dt, (sa.end - sa.start) / a.dt));
                    }
                } else {
                    if ( sb.start != sb.end ) {  // non-empty
                        return generic_dt(fixed_dt(sb.start, a.dt, (sb.end - sb.start) / a.dt));
                    } else {
                        return generic_dt(fixed_dt::null_range());
                    }
                }
            // unaligned or non-consecutive
            } else {
                std::vector<utctime> points;
                points.reserve(
                    (sa.end - sa.start) / a.dt + (sa.start != sa.end ? 1 : 0)
                    + (sb.end - sb.start) / b.dt + (sb.start != sb.end ? 1 : 0));

                // add a
                if ( sa.start != sa.end ) {
                    for ( utctime t = sa.start; t <= sa.end; t += a.dt )
                        points.push_back(t);
                }

                // add b
                if ( sb.start != sb.end ) {
                    // the first interval may overlap for unaligned time-axes
                    if ( sa.start != sa.end && sb.start > sa.end )
                        points.push_back(sb.start);
                    for ( utctime t = sb.start+b.dt; t <= sb.end; t += b.dt )
                        points.push_back(t);
                }

                // finalize
                if ( points.size() >= 2 ) {
                    return generic_dt(point_dt(std::move(points)));
                } else {
                    return generic_dt(point_dt::null_range());
                }
            }
        }

        inline generic_dt extend(const calendar_dt & a, const calendar_dt & b, const utctime split_at) {
            const utcperiod pa = a.total_period();
            const utcperiod pb = b.total_period();

            {
                const size_t asz = a.size();
                const size_t bsz = b.size();

                // trivial cases
                if ( asz == 0 || bsz == 0 ) {
                    // - both empty -> return empty range
                    if ( asz == 0 && bsz == 0 ) {
                        return generic_dt(calendar_dt::null_range());
                    }
                    // - one empty -> return non-empty sliced at split_at
                    else if ( asz == 0 ) {
                        size_t split_index = b.index_of(split_at);
                        if ( split_index == std::string::npos ) {
                            if ( split_at < pb.start ) {
                                return generic_dt(b);
                            } else {
                                return generic_dt(calendar_dt::null_range());
                            }
                        } else {
                            utcperiod split_p = b.period(split_index);
                            return generic_dt(calendar_dt(
                                b.get_calendar(),
                                split_p.start,
                                b.dt, bsz - split_index));
                        }
                    } else {
                        size_t split_index = a.index_of(split_at);
                        if ( split_index == std::string::npos ) {
                            if ( split_at < pa.start ) {
                                return generic_dt(calendar_dt::null_range());
                            } else {
                                return generic_dt(a);
                            }
                        } else {
                            return generic_dt(calendar_dt(
                                a.get_calendar(),
                                pa.start,
                                a.dt, split_index));
                        }
                    }
                }
            }

            // sliced spans for a and b
            size_t idx = a.index_of(split_at);
            const size_t split_a_idx = idx != std::string::npos ? idx : (split_at < pa.start ? 0 : a.size());
            //
            idx = b.index_of(split_at);
            const size_t split_b_idx = idx != std::string::npos ? idx : (split_at < pb.start ? 0 : b.size() - 1);

            // split interval
            const utcperiod span_a(
                pa.start,
                split_at < pa.end ? a.period(split_a_idx).start : pa.end);
            const utcperiod span_b(
                split_at < pb.end ? b.period(split_b_idx).start : pb.end,
                pb.end);

            if ( span_a.start == span_a.end && span_b.start == span_b.end ) {
                return generic_dt(calendar_dt::null_range());
            }

            // equivalent calendars, aligned dt, and consecutive
            if ( a.cal->tz_info->name() == b.cal->tz_info->name()
                && a.dt == b.dt
                && (span_a.start == span_a.end || span_b.start == span_b.end || span_a.end == span_b.start)
            ) {

                // determine aligned offset
                utctimespan remainder;
                size_t n = static_cast<size_t>(a.cal->diff_units(pa.start, pb.end, a.dt, remainder));

                // no offset
                if ( remainder == 0 ) {
                    if ( span_a.start != span_a.end ) {  // non-empty
                        if ( span_b.start != span_b.end ) {  // non-empty
                            return generic_dt(calendar_dt(a.get_calendar(), span_a.start, a.dt, n));
                        } else {
                            return generic_dt(calendar_dt(a.get_calendar(), span_a.start, a.dt, split_a_idx));
                        }
                    } else {
                        if ( span_b.start != span_b.end ) {  // non-empty
                            return generic_dt(calendar_dt(a.get_calendar(), span_b.start, a.dt, b.size() - split_b_idx));
                        //} else {
                            //return generic_dt(calendar_dt::null_range());
                        }
                    }
                }
            }

            // ELSE unaligned or non-consecutive

            std::vector<utctime> points;
            points.reserve(
                (span_a.start != span_a.end ? split_a_idx + 1 : 0)
                 + (span_b.start != span_b.end && split_at < pb.end ? b.size() - split_b_idx : 0)
                 + (span_a.start != span_b.end && span_a.end < span_b.start ? 1 : 0));

            // add a
            if ( split_a_idx > 0 ) {
                for ( size_t i = 0; i <= split_a_idx; ++i ) {
                    points.push_back(a.cal->add(pa.start, a.dt, i));
                }
            }

            // add b
            if ( span_b.start != span_b.end ) {
                // the first interval may overlap for unaligned time-axes
                if ( span_a.start == span_a.end || span_b.start > span_a.end ) {
                    points.push_back(b.cal->add(pb.start, b.dt, split_b_idx));
                }
                const size_t bsz = b.size();
                for ( size_t i = split_b_idx + 1; i <= bsz; ++i ) {
                    points.push_back(b.cal->add(pb.start, b.dt, i));
                }
            }

            // finalize
            if ( points.size() >= 2 ) {
                return generic_dt(point_dt(std::move(points)));
            } else {
                return generic_dt(point_dt::null_range());
            }
        }

		template<class TA, class TB>
		inline auto extend(const TA & a, const TB & b, const utctime split_at)
			-> typename std::enable_if<TA::continuous::value && TB::continuous::value, generic_dt>::type  // SFINAE
		{
			namespace core = shyft::core;

			const size_t a_sz = a.size(),
				b_sz = b.size();
			const core::utcperiod pa = a.total_period(),
				pb = b.total_period();

			// determine number of intervals to use
			const size_t a_idx = a.index_of(split_at),
				a_end_idx = a_idx != std::string::npos  // split index not after a?
				? a_idx : (a_sz == 0 || split_at < pa.start ? 0 : a_sz);
			// -----
			const size_t b_idx = b.index_of(split_at),
				b_start_idx = b_idx != std::string::npos  // split index not before b?
				? b_idx : (b_sz == 0 || split_at < pb.start ? 0 : b_sz);

			// one empty?
			if (a_end_idx == 0 || b_start_idx == b_sz) {
				if (a_end_idx == 0 && b_start_idx == b_sz) {
					return std::move(generic_dt(point_dt::null_range()));
				}
				// b empty? (remember then a can't be)
				else if (b_start_idx == b_sz) {
					if (a_end_idx == 0) {
						return generic_dt(a);
					} else {
						return extend_helper<TA>::as_generic(a, 0, a_end_idx);
					}
				} else {
					if (b_start_idx == 0) {
						return generic_dt(b);
					} else {
						return extend_helper<TB>::as_generic(b, b_start_idx, b_sz - b_start_idx);
					}
				}
			}

			std::vector<utctime> points;

			// any a points to use?
			if (a_sz > 0 && split_at >= a.period(0).end) {
				for (size_t i = 0; i < a_end_idx; ++i) {
					points.push_back(a.period(i).start);
				}
				points.push_back(a.period(a_end_idx - 1).end);
			}

			// any b points to use?
			if (b_sz > 0 && pa.start < pb.end && split_at < pb.end) {
				if (
					pa.start == pa.end      // a is empty
					|| pb.start > pa.end    // OR b starts after end of a
					|| split_at > pa.end    // OR split is after end of a
					|| pb.start > split_at  // OR the start of b is after the split
					) {
					// then push the first point of b (otherwise it is included as the last from a)
					points.push_back(b.period(b_start_idx).start);
				}
				for (size_t i = b_start_idx + 1; i < b_sz; ++i) {
					points.push_back(b.period(i).start);
				}
				points.push_back(b.period(b_sz - 1).end);
			}

			// finalize
			if (points.size() >= 2) {
				return generic_dt(point_dt(std::move(points)));
			} else {
				return generic_dt(point_dt::null_range());
			}
		}

		template<class TA, class TB>
		inline auto extend(const TA & a, const TB & b, const utctime split_at)
			-> typename std::enable_if<!TA::continuous::value || !TB::continuous::value, generic_dt>::type  // SFINAE
		{
			throw std::runtime_error("extension of/with discontinuous time-axis not supported");
		}

        inline generic_dt extend(const generic_dt & a, const generic_dt & b, const utctime split_at) {
            if ( a.gt == generic_dt::FIXED && b.gt == generic_dt::FIXED ) {
                return extend(a.f, b.f, split_at);
            } else if ( a.gt == generic_dt::CALENDAR && b.gt == generic_dt::CALENDAR ) {
                return extend(a.c, b.c, split_at);
            } else {
                if ( a.gt == generic_dt::FIXED ) {
                    if ( b.gt == generic_dt::CALENDAR ) {
                        return extend(a.f, b.c, split_at);
                    } else {  // point
                        return extend(a.f, b.p, split_at);
                    }
                } else if ( a.gt == generic_dt::CALENDAR ) {
                    if ( b.gt == generic_dt::FIXED ) {
                        return extend(a.c, b.f, split_at);
                    } else {  // point
                        return extend(a.c, b.p, split_at);
                    }
                } else {
                    if ( b.gt == generic_dt::FIXED ) {
                        return extend(a.p, b.f, split_at);
                    } else if ( b.gt == generic_dt::CALENDAR ) {
                        return extend(a.p, b.c, split_at);
                    } else {  // point
                        return extend(a.p, b.p, split_at);
                    }
                }
            }
        }

 

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
        /**ensure generic_dt optimizes fixed-interval cases */
        inline generic_dt combine( const generic_dt& a, const generic_dt & b ) {
            switch(a.gt) {
            case generic_dt::FIXED:switch(b.gt) {
                case generic_dt::FIXED: return generic_dt{combine(a.f,b.f)};//fast
                case generic_dt::CALENDAR: return combine(a.f,b.c);
                case generic_dt::POINT: return combine(a.f,b.p);
                }
            break;
            case generic_dt::CALENDAR:switch(b.gt) {
                case generic_dt::FIXED:return combine(a.c,b.f);
                case generic_dt::CALENDAR:return combine(a.c,b.c);
                case generic_dt::POINT:return combine(a.c,b.p);
                }

            case generic_dt::POINT:switch(b.gt) {
                case generic_dt::FIXED:return combine(a.p,b.f);
                case generic_dt::CALENDAR:return combine(a.p,b.c);
                case generic_dt::POINT:return combine(a.p,b.p);
                }
            }
            return generic_dt{};//never reached
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

		//-- fixup missing index-hint for some time-axis-types
		template<class TA  > inline size_t ta_index_of(TA const&ta, utctime t, size_t ix_hint) {	return ta.index_of(t);}
		template<> inline size_t ta_index_of(time_axis::point_dt const&ta, utctime t, size_t ix_hint) { return ta.index_of(t, ix_hint);}
		template<> inline size_t ta_index_of(time_axis::generic_dt const&ta, utctime t, size_t ix_hint) {	return ta.index_of(t, ix_hint);	}

		/** \brief time_axis_transform finds index mapping from source to map-time-axis
		*
		* Given a source time-axis src, for each
		* start of the map time-axis interval, find
		* the right-most index of src (the one equal to or to the left of map time-axis period)
		* and provide those through the .source_index() method.
		*
		*/
		template<class TA1, class TA2>
		struct time_axis_map {
			TA1 const src;
			size_t src_ix = string::npos;
			TA2 const m;
			time_axis_map(TA1 const&src, TA2 const&m)
				:src(src), m(m) {}

			inline size_t src_index(size_t i) {
				if (i > m.size())
					return string::npos;
				src_ix = ta_index_of(src, m.time(i), src_ix);
				return src_ix;
			}
		};


		/** \brief specialize for fixed_dt time-axis, that can be done really fast
		*/
		template<>
		struct time_axis_map<time_axis::fixed_dt, time_axis::fixed_dt> {
			time_axis::fixed_dt src;
			time_axis::fixed_dt m;

			time_axis_map(time_axis::fixed_dt const& src, time_axis::fixed_dt const&m) :src(src), m(m) {}
			inline size_t src_index(size_t im) const {
				auto r = utctime((utctime(im)*m.dt + m.t - src.t) / src.dt);// (mis)using utctime as signed int64 here
				if (r < 0 || r >= (utctime)src.n)
					return std::string::npos;
				return size_t(r);
			}
		};

		/** \brief auto-deduce a time-axis transform adapted to the time-axis that we have
		*/
		template<class TA1, class TA2>
		inline auto make_time_axis_map(TA1 const&src, TA2 const&m) {
			return time_axis_map<TA1, TA2>(src, m);
		}

		/* The section below contains merge functionality for
		 * time-axis, but to be used in the context of time-series
		 * the 'merge(ta a,ta b)' operation
		 * require the time-axis to be compatible
		 * and the .total_period() should overlap, or extend.
		 * It's important that time-axis and values are merged using
		 * same info/algorithm.
 		 */

		/** helper class to keep time-series/axis merge info
		*  for time-axis a (priority) and b (fillin/extend)
		*/
		struct merge_info {
			size_t b_n{ 0 };///< copy n- first from b before a
			size_t a_i{ string::npos };///< extend a with  b[a_i]..a_n after a
			size_t a_n{ 0 };///< number of elements to extend after a is at the end.
			size_t size() const { return b_n + a_n; }
			utctime t_end{ no_utctime };///< the t_end, relevant for point_dt time-axis
		};

        /**returns true if the period a and be union can be one continuous period*/
		inline bool continuous_merge(const utcperiod& a, const utcperiod& b) {
			return !(a.end < b.start || b.end < a.start);
		}

        /**return true if a calendars are reference equal or have same name */
		inline bool equal_calendars(const shared_ptr<calendar>&a, const shared_ptr<calendar>&b) {
			return a.get() == b.get() || (a->tz_info->name() == b->tz_info->name());
		}

		/** return true if fixed time-axis a and b can be merged into one time-axis */
		inline bool can_merge(const fixed_dt&a, const fixed_dt&b) {
			return a.dt == b.dt && a.dt != 0 && a.n > 0 && b.n > 0 && continuous_merge(a.total_period(), b.total_period());
		}

		/** return true if calendar time-axis a and b can be merged into one time-axis */
		inline bool can_merge(const calendar_dt& a, const calendar_dt& b) {
			return a.dt == b.dt && a.dt != 0 && a.n > 0 && b.n > 0
				&& equal_calendars(a.cal, b.cal)
				&& continuous_merge(a.total_period(), b.total_period());
		}

		/** return true if point time-axis a and b can be merged into one time-axis */
		inline bool can_merge(const point_dt &a, const point_dt& b) {
			return continuous_merge(a.total_period(), b.total_period());
		}

        /** return true if generic time-axis a and b can be merged into one time-axis */
		inline bool can_merge(const generic_dt& a, const generic_dt& b) {
			switch (a.gt) {
			case generic_dt::FIXED: return can_merge(a.f, b.f);
			case generic_dt::CALENDAR: return can_merge(a.c, b.c);
			case generic_dt::POINT: return can_merge(a.p, b.p);
			}
			throw runtime_error("unsupported time-axis in can_merge");
		}

		/**computes the merge-info for two time-axis
		*
		* to enable easy and consistent time-series merge
		* operations.
		*/
		template <class TA, enable_if_t< TA::continuous::value,int> =0 >// enable if time-axis
		inline merge_info compute_merge_info(const TA& a, const TA&b) {
			const auto a_p = a.total_period();
			const auto b_p = b.total_period();
			if (!continuous_merge(a_p, b_p)) throw runtime_error(string("attempt to merge disjoint non-overlapping time-axis"));
			merge_info r;
			if (a_p.start > b_p.start) { // a starts after b, so b contribute before a starts
				r.b_n = b.index_of(a_p.start - 1)+1;
			}
			if (a_p.end < b_p.end) { // a ends before b ends, so b extends the result
				r.a_i = b.index_of(a_p.end); // check if b.time(i) is >= a_p.end, if not increment i.
				if (b.time(r.a_i) < a_p.end)
					++r.a_i;
				r.a_n = b.size() - r.a_i;
				r.t_end = b_p.end;
			} else { // a ends after b
				r.t_end = a_p.end;
			}
			return r;
		}

		/** merge time-axis a and b into one.
		*  require a.dt equal to b.dt
		*         and that the two axis covers a contiguous period
		*/
		inline fixed_dt merge(const fixed_dt& a, const fixed_dt& b, const merge_info& m) {
			const auto a_p = a.total_period();
			const auto b_p = b.total_period();
			utcperiod p{ min(a_p.start,b_p.start), max(a_p.end, b_p.end) };
			return fixed_dt{ p.start,a.dt, a.size() + m.size() };
		}

		/** merge time-axis a and b into one.
		*  require a.dt equal to b.dt,
		*         and same calendar (tz-id)
		*         and that the two axis covers a contiguous period
		*/
		inline calendar_dt merge(const calendar_dt& a, const calendar_dt& b, const merge_info& m) {
			const auto a_p = a.total_period();
			const auto b_p = b.total_period();
			utcperiod p{ min(a_p.start,b_p.start), max(a_p.end, b_p.end) };
			return calendar_dt{ a.cal, p.start,a.dt,a.size() + m.size() };
		}

		/** merge value-vector from two time-series a and b using merge_info */
		template <class T>
		vector<T> merge(const vector<T> &a, const vector<T> &b, const merge_info& m) {
			auto n = m.size() + a.size();
			vector<T> r; r.reserve(n);
			if (m.b_n) copy(begin(b), begin(b) + m.b_n, back_inserter(r));
			copy(begin(a), end(a), back_inserter(r));
			if (m.a_n) copy(begin(b) + m.a_i, begin(b) + m.a_i + m.a_n, back_inserter(r));
			return r;
		}

		/** merge time-axis a and b into one using merge_info.
		*  require a validated merg_info for a and b
		*  \return a new point_dt where the points are
		*           all points of a, plus points of b not covered by a
		*/
		inline point_dt merge(const point_dt& a, const point_dt& b, const merge_info& m) {
			return point_dt{ merge(a.t,b.t,m),m.t_end };
		}

		inline generic_dt merge(const generic_dt& a, const generic_dt& b, const merge_info& m) {
			switch (a.gt) {
			case generic_dt::FIXED:return generic_dt(merge(a.f, b.f, compute_merge_info(a.f, b.f)));
			case generic_dt::CALENDAR:return generic_dt(merge(a.c, b.c, compute_merge_info(a.c, b.c)));
			case generic_dt::POINT:return generic_dt(merge(a.c, b.c, compute_merge_info(a.c, b.c)));
			}
			throw runtime_error("merge(generic_dt..): unsupported time-axis type");
		}

		/** simple template that merges two equally typed time-series
		 *
		 *  \returns the merged time-series, or throws if not compatible
		 */
		template<class TA,enable_if_t< TA::continuous::value,int> =0>
		TA merge(const TA& a, const TA& b) {
			if (!can_merge(a, b)) throw runtime_error("can not merge time-axis, not compatible or disjoint total_period");
			return merge(a, b, compute_merge_info(a, b));
		}
    }
}
//--serialization support
x_serialize_export_key(shyft::time_axis::fixed_dt);
x_serialize_export_key(shyft::time_axis::calendar_dt);
x_serialize_export_key(shyft::time_axis::point_dt);
x_serialize_export_key(shyft::time_axis::generic_dt);

