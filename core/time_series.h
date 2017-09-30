#pragma once
#ifdef SHYFT_NO_PCH
#include <cstdint>
#include <cmath>
#include <string>
#include <stdexcept>
#include <vector>
#include <memory>
#include <utility>
#include <stdexcept>
#include <type_traits>
#include <algorithm>
#include <sstream>
#include "core_pch.h"
#endif // SHYFT_NO_PCH

#include "compiler_compatiblity.h"
#include "utctime_utilities.h"
#include "time_axis.h"
#include "glacier_melt.h" // to get the glacier melt function
#include "unit_conversion.h"

#ifndef M_PI
# define M_PI           3.14159265358979323846  /* pi */
#endif

namespace shyft{
    const double nan = std::numeric_limits<double>::quiet_NaN();

    /** The time_series namespace contains all needed concepts related
    * to representing and handling time-series concepts efficiently.
    *
    * concepts:
    *  -#: point, -a time,value pair,utctime,double representing points on a time-series, f(t).
    *  -#: timeaxis, -(TA)the fundation for the timeline, represented by the timeaxis class, a non-overlapping set of periods ordered ascending.
    *  -#: point_ts,(S) - provide a source of points, that can be interpreted by an accessor as a f(t)
    *  -#: accessor, -(A) average_accessor and direct_accessor, provides transformation from f(t) to some provide time-axis
    */
    namespace time_series {

        using namespace shyft::core;
        using namespace std;
        const double EPS=1e-12; ///< used some places for comparison to equality, \ref point

        /** \brief simply a point identified by utctime t and value v */
        struct point {
            utctime t;
            double v;
            point(utctime t=0, double v=0.0) : t(t), v(v) { /* Do nothing */}
        };

        /** \brief point a and b are considered equal if same time t and value-diff less than EPS
        */
        inline bool operator==(const point &a,const point &b)  {return (a.t==b.t) && std::fabs(a.v-b.v)< EPS;}

        /** \brief Enumerates how points are mapped to f(t)
         *
         * If there is a point_ts, this determines how we would draw f(t).
         * Average values are typically staircase-start-of step, a constant over the interval
         * for which the average is computed.
         * State-in-time values are typically POINT_INSTANT_VALUES; and we could as an approximation
         * draw a straight line between the points.
         */
        enum ts_point_fx:int8_t {
            POINT_INSTANT_VALUE, ///< the point value represents the value at the specific time (or centered around that time),typically linear accessor
            POINT_AVERAGE_VALUE///< the point value represents the average of the interval, typically stair-case start of step accessor

        };

        inline ts_point_fx result_policy(ts_point_fx a, ts_point_fx b) {
            return a==ts_point_fx::POINT_INSTANT_VALUE || b==ts_point_fx::POINT_INSTANT_VALUE?ts_point_fx::POINT_INSTANT_VALUE:ts_point_fx::POINT_AVERAGE_VALUE;
        }

        //--- to avoid duplicating algorithms in classes where the stored references are
        // either by shared_ptr, or by value, we use template function:
        //  to make shared_ptr<T> appear equal to T
        template<typename T> struct is_shared_ptr {static const bool value=false;};
        template<typename T> struct is_shared_ptr<shared_ptr<T>> {static const bool value=true;};

        /** is_ts<T> is a place holder to support specific dispatch for time-series types
         * default is false
         */
        template<class T> struct is_ts {static const bool value=false;};

        /** needs_bind<T> is a default place holder to support specific dispatch for
         * time-series or expressions that might have symbolic references.
         * We use this to deduce compile-time if an expression needs a bind
         * cycle before being evaluated. That is; that the ref_ts get their values
         * set by a bind-operation.
         * By default, this is true for all types, except point_ts that we know do not need bind
         * - all other need to turn it by specializing needs_bind similar as for point_ts
         * \see ref_ts
         */
        template<class T> struct needs_bind {static const bool value=true;};

		/** Resolves compiletime to dispatch runtime calls to needs_bind where supported.
		 * Additionally allows for querying if a type supports needs_bind.
		 */
		template <
			class T, typename = void
		> struct needs_bind_dispatcher : std::false_type
		{
			static bool needs_bind(const T& t) {
				return false;
			}
		};

		template <class T> struct needs_bind_dispatcher<
			T, std::enable_if_t<std::is_same<decltype(std::declval<T>().needs_bind()), bool>::value>
		> : std::true_type
		{
			static bool needs_bind(const T& t) {
				return t.needs_bind();
			}
		};

		template<class T> bool e_needs_bind(T const& t) { return needs_bind_dispatcher<T>::needs_bind(t); }// run-time check on state, default false

        /** \brief d_ref function to d_ref object or shared_ptr
         *
         * The T d_ref(T) template to return a ref or const ref to T, if T is shared_ptr<T> or just T
         */
        template<class U> const U& d_ref(const std::shared_ptr<U>& p) { return *p; }
        template<class U> const U& d_ref(const U& u) { return u; }
        template<class U> U& d_ref(std::shared_ptr<U>& p) { return *p; }
        template<class U> U& d_ref(U& u) { return u; }


        ///< \brief d_ref_t template to rip out T of shared_ptr<T> or T if T specified
        template <class T, class P = void >
        struct d_ref_t { };

        template <class T >
        struct d_ref_t<T,typename enable_if<is_shared_ptr<T>::value>::type> {
            typedef typename T::element_type type;
        };
        template <class T >
        struct d_ref_t<T,typename enable_if<!is_shared_ptr<T>::value>::type > {
            typedef T type;
        };


        /** \brief hint_based search to eliminate binary-search in irregular time-point series.
         *
         *  utilizing the fact that most access are periods, sequential, so the average_xxx functions
         *  can utilize this, and keep the returned index as a hint for the next request for average_xxx
         *  value.
         * \tparam S a point ts source, must have .get(i) ->point, and .size(), and .index_of(t)->size_t
         * \param source a point ts source as described above
         * \param p utcperiod for which we search a start point <= p.start
         * \param i the start-of-search hint, could be -1, then ts.index_of(p.start) is used to figure out the ix.
         * \return lowerbound index or npos if not found
         * \note We should specialize this for sources with computed time-axis to improve speed
         */
        template<class S>
        size_t hint_based_search(const S& source, const utcperiod& p, size_t i) {
            const size_t n = source.size();
            if (n == 0)
                return std::string::npos;
            if (i != std::string::npos && i<n) { // hint-based search logic goes here:
                const size_t max_directional_search = 5; // +-5 just a guess for what is a reasonable try upward/downward
                auto ti = source.get(i).t;
                if (ti == p.start) { // direct hit and extreme luck ?
                    return i; // just use it !
                } else if (ti < p.start) { // do a local search upward to see if we find the spot we search for
                    if (i == n - 1) return i;// unless we are at the end (n-1), try to search upward
                    size_t i_max = std::min(i + max_directional_search, n);
                    while (++i < i_max) {
                        ti = source.get(i).t;
                        if (ti < p.start )
                            continue;
                        return  ti > p.start? i - 1 : i;// we either got one to far, or direct-hit
                    }
                    return (i < n) ? source.index_of(p.start):n-1; // either local search failed->bsearch etc., or we are at the end -> n-1
                } else if (ti > p.start) { // do a local search downwards from last index, maybe we are lucky
                    if (i == 0) // if we are at the beginning, just return npos (no left-bound index found)
                        return 0;//std::string::npos;
                    size_t i_min =  (i - std::min(i, max_directional_search));
                    do {
                        ti = source.get(--i).t;//notice that i> 0 before we start due to if(i==0) above(needed due to unsigned i!)
                        if ( ti > p.start)
                            continue;
                        return i; // we found the lower left bound (either exact, or less p.start)
                    } while (i > i_min);
                    return i>0? source.index_of(p.start): std::string::npos; // i is >0, there is a hope to find the index using index_of, otherwise, no left lower bound
                }
            }
            return source.index_of(p.start);// no hint given, just use binary search to establish the start.
        }


        /** \brief accumulate_value provides a projection/interpretation
        * of the values of a pointsource on to a time-axis as provided.
        * This includes interpolation and true average, linear between points
        * and nan-handling semantics.
        * In addition the Accessor allows fast sequential access to these values
        * using clever caching of the last position used in the underlying
        * point source. The time axis and point source can be of any type
        * as listed above as long as the basic type requirement as described below
        * are satisfied.
        * \tparam S point source, must provide:
        *  -# .size() const               -> number of points in the source
        *  -# .index_of(utctime tx) const -> return lower bound index or -1 for the supplied tx
        *  -# .get(size_t i) const        -> return the i'th point  (t,v)
        * \param source of type S
        * \param p         the period [start,end) on time-axis, the range where we will accumulate/integrate the f(t)
        * \param last_idx  position of the last time point used on the source, updated after each call.
        * \param tsum      the sum of time under non-nan areas of the curve
        * \param linear    interpret points as linear between, if set to false, use stair-case start of step def
        * \return the area under the non-nan areas of the curve, specified by tsum reference-parameter
        */
        template <class S>
        double accumulate_value(const S& source, const utcperiod& p, size_t& last_idx, utctimespan& tsum,bool linear = true) {
            const size_t n = source.size();
            if (n == 0) // early exit if possible
                return shyft::nan;
            size_t i = hint_based_search(source, p, last_idx);  // use last_idx as hint, allowing sequential periodic average to execute at high speed(no binary searches)

            if (i == std::string::npos) { // this might be a case
                last_idx = 0;// we update the hint to the left-most possible index
                return shyft::nan; // and is equivalent to no points, or all points after requested period.
            }
            point l;// Left point
            bool l_finite = false;

            double area = 0.0;  // Integrated area over the non-nan parts of the time-axis
            tsum = 0; // length of non-nan f(t) time-axis

            while (true) { //there are two exit criteria: no more points, or we pass the period end.
                if (!l_finite) {//search for 'point' anchor phase
                    l = source.get(i++);
                    l_finite = std::isfinite(l.v);
                    if (i == n) { // exit condition
                        if (l_finite && l.t < p.end) {//give contribution
                            utctimespan dt = p.end - l.t;
                            tsum += dt;
                            area += dt* l.v; // extrapolate value flat
                        }
                        break;//done
                    }
                    if (l.t >= p.end) {//also exit condition, if the point time is after period we search, then no anchor to be found
                        break;//done
                    }
                } else { // got point anchor l, search for right point/end
                    point r = source.get(i++);// r is guaranteed > p.start due to hint-based search
                    bool r_finite = std::isfinite(r.v);
                    utcperiod px(std::max(l.t, p.start), std::min(r.t, p.end));
                    utctimespan dt = px.timespan();
                    tsum += dt;

                    // now add area contribution for l..r
                    if (linear && r_finite) {
                        double a = (r.v - l.v) / (r.t - l.t);
                        double b = r.v - a*r.t;
                        //area += dt* 0.5*( a*px.start+b + a*px.end+b);
                        area += dt * (0.5*a*(px.start + px.end) + b);
                    } else { // flat contribution from l  max(l.t,p.start) until max time of r.t|p.end
                        area += l.v*dt;
                    }
                    if (i == n) { // exit condition: final value in sequence, then we need to finish, but
                        if (r_finite && r.t < p.end) {// add area contribution from r and out to the end of p
                            dt = p.end - r.t;
                            tsum += dt;
                            area += dt* r.v; // extrapolate value flat
                        }
                        break;//done
                    }
                    if (r.t >= p.end)
                        break;//also exit condition, done
                    l_finite = r_finite;
                    l = r;
                }

            }
            last_idx = i - 1;
            return tsum ? area : shyft::nan;
        }

       /** \brief average_value provides a projection/interpretation
         * of the values of a pointsource on to a time-axis as provided.
         * This includes interpolation and true average, linear between points
         * and nan-handling semantics.
         * In addition the Accessor allows fast sequential access to these values
         * using clever caching of the last position used in the underlying
         * point source. The time axis and point source can be of any type
         * as listed above as long as the basic type requirement as described below
         * are satisfied.
         * \tparam S point source, must provide:
         *  -# .size() const               -> number of points in the source
         *  -# .index_of(utctime tx) const -> return lower bound index or -1 for the supplied tx
         *  -# .get(size_t i) const        -> return the i'th point  (t,v)
         * \param source of type S
         * \param p the period [start,end) on time-axis
         * \param last_idx in/out, position of the last time point used on the source, updated after each call.
         * \param linear how to interpret the points, if true, use linear between points specification
         * \return double, the value at the as true average of the specified period
         */
        template <class S>
        inline double average_value(const S& source, const utcperiod& p, size_t& last_idx,bool linear=true) {
            utctimespan tsum = 0;
            double area = accumulate_value(source, p, last_idx, tsum, linear);// just forward the call to the accumulate function
            return tsum>0?area/tsum:shyft::nan;
        }



        /**\brief point time-series, pts, defined by
         * its
         * templated time-axis, ta
         * the values corresponding periods in time-axis (same size)
         * the ts_point_fx that determine how to compute the
         * f(t) on each interval of the time-axis (linear or stair-case)
         * and
         * value of the i'th interval of the time-series.
         */
        template <class TA>
        struct point_ts {
            typedef TA ta_t;
            TA ta;
            const TA& time_axis() const {return ta;}
            vector<double> v;
            ts_point_fx fx_policy;

            ts_point_fx point_interpretation() const { return fx_policy; }
            void set_point_interpretation(ts_point_fx point_interpretation) { fx_policy=point_interpretation;}

            point_ts():fx_policy(ts_point_fx::POINT_INSTANT_VALUE){}
            point_ts(const TA& ta, double fill_value,ts_point_fx fx_policy=POINT_INSTANT_VALUE):ta(ta),v(ta.size(),fill_value),fx_policy(fx_policy) {}
            point_ts(const TA& ta,const vector<double>&vx,ts_point_fx fx_policy=POINT_INSTANT_VALUE):ta(ta),v(vx),fx_policy(fx_policy) {
                if(ta.size() != v.size())
                    throw runtime_error("point_ts: time-axis size is different from value-size");
            }
            //TODO: move/cp constructors needed ?
            //TODO should we provide/hide v ?
            // TA ta, ta is expected to provide 'time_axis' functions as needed
            // so we do not re-pack and provide functions like .size(), .index_of etc.
            /**\brief the function value f(t) at time t, fx_policy taken into account */
            double operator()(utctime t) const {
                size_t i = ta.index_of(t);
                if(i == string::npos) return nan;
                if( fx_policy==ts_point_fx::POINT_INSTANT_VALUE && i+1<ta.size() && isfinite(v[i+1])) {
                    utctime t1=ta.time(i);
                    utctime t2=ta.time(i+1);
                    double f= double(t2-t)/double(t2-t1);
                    return v[i]*f + (1.0-f)*v[i+1];
                }
                return v[i]; // just keep current value flat to +oo or nan
            }

            /**\brief i'th value of the value,
             */
            double value(size_t i) const  { return v[i]; }
            // BW compatiblity ?
            size_t size() const { return ta.size();}
            size_t index_of(utctime t) const {return ta.index_of(t);}
            utcperiod total_period() const {return ta.total_period();}
            utctime time(size_t i ) const {return ta.time(i);}

            // to help average_value method for now!
            point get(size_t i) const {return point(ta.time(i),value(i));}

            // Additional write/modify interface to operate directly on the values in the time-series
            void set(size_t i,double x) {v[i]=x;}
            void add(size_t i, double value) { v[i] += value; }
            void add(const point_ts<TA>& other) {
                std::transform(begin(v), end(v), other.v.cbegin(), begin(v), std::plus<double>());
            }
            void add_scale(const point_ts<TA>&other,double scale) {
                std::transform(begin(v), end(v), other.v.cbegin(), begin(v), [scale](double a, double b) {return a + b*scale; });
            }
            void fill(double value) { std::fill(begin(v), end(v), value); }
            void fill_range(double value, int start_step, int n_steps) { if (n_steps == 0)fill(value); else std::fill(begin(v) + start_step, begin(v) + start_step + n_steps, value); }
            void scale_by(double value) { std::for_each(begin(v), end(v), [value](double&v){v *= value; }); }
            x_serialize_decl();
        };

        /** \brief time_shift ts do a time-shift dt on the supplied ts
         *
         * The values are exactly the same as the supplied ts argument to the constructor
         * but the time-axis is shifted utctimespan dt to the left.
         * e.g.: t_new = t_original + dt
         *
         *       lets say you have a time-series 'a'  with time-axis covering 2015
         *       and you want to time-shift so that you have a time- series 'b'data for 2016,
         *       then you could do this to get 'b':
         *
         *           utc = calendar() // utc calendar
         *           dt  = utc.time(2016,1,1) - utc.time(2015,1,1)
         *            b  = timeshift_ts(a, dt)
         *
         */
        template<class Ts>
        struct time_shift_ts {
            typedef typename Ts::ta_t ta_t;
            Ts ts;
            // need to have a time-shifted time-axis here:
            // TA, ta.timeshift(dt) -> a clone of ta...
            ta_t ta;
            ts_point_fx fx_policy=POINT_AVERAGE_VALUE; // inherited from ts
            utctimespan dt=0;// despite ta time-axis, we need it
            bool bound=false;
            ts_point_fx point_interpretation() const { return fx_policy; }
            void set_point_interpretation(ts_point_fx point_interpretation) { fx_policy=point_interpretation;}

            //-- default stuff, ct/copy etc goes here
            time_shift_ts()=default;

            //-- useful ct goes here
            template<class A_>
            time_shift_ts(A_ && ts,utctimespan dt)
                :ts(std::forward<A_>(ts)),dt(dt) {
                // we have to wait with time-axis until we know underlying stuff are ready:
                if( !(needs_bind< typename d_ref_t<Ts>::type>::value && e_needs_bind(ts))) {
                    do_bind();//if possible do it now
                }
            }

            void do_bind() {
                if(!bound) {
                    fx_policy = d_ref(ts).point_interpretation();
                    ta = time_axis::time_shift(d_ref(ts).time_axis(),dt);
                    bound=true;
                }
            }
            const ta_t& time_axis() const { return ta;}

            point get(size_t i) const {return point(ta.time(i),ts.value(i));}

            // BW compatiblity ?
            size_t size() const { return ta.size();}
            size_t index_of(utctime t) const {return ta.index_of(t);}
            utcperiod total_period() const {return ta.total_period();}
            utctime time(size_t i ) const {return ta.time(i);}

            //--
            double value(size_t i) const { return ts.value(i);}
            double operator()(utctime t) const { return ts(t-dt);} ///< just here we needed the dt
            x_serialize_decl();
        };

        /**\brief average_ts, average time-series
         *
         * Represents a ts that for
         * the specified time-axis returns the true-average of the
         * underlying specified TS ts.
         *
         */
        template<class TS,class TA>
        struct average_ts {
            typedef TA ta_t;
            TA ta;
            TS ts;
            ts_point_fx fx_policy=POINT_AVERAGE_VALUE;
            const TA& time_axis() const {return ta;}
            ts_point_fx point_interpretation() const { return fx_policy; }
            void set_point_interpretation(ts_point_fx point_interpretation) { fx_policy=point_interpretation;}
            average_ts()=default; // allow default construct
            average_ts(const TS&ts,const TA& ta)
            :ta(ta),ts(ts) {

            } // because true-average of periods is per def. POINT_AVERAGE_VALUE
            // to help average_value method for now!
            point get(size_t i) const {return point(ta.time(i),value(i));}
            size_t size() const { return ta.size();}
            size_t index_of(utctime t) const {return ta.index_of(t);}
            //--
            double value(size_t i) const {
                if(i >= ta.size())
                    return nan;
                size_t ix_hint=(i*d_ref(ts).ta.size())/ta.size();// assume almost fixed delta-t.
                //TODO: make specialized pr. time-axis average_value, since average of fixed_dt is trivial compared to other ta.
                return average_value(ts,ta.period(i),ix_hint,d_ref(ts).fx_policy == ts_point_fx::POINT_INSTANT_VALUE);// also note: average of non-nan areas !
            }
            double operator()(utctime t) const {
                size_t i=ta.index_of(t);
                if( i==string::npos)
                    return nan;
                return value(i);
            }
            x_serialize_decl();
        };

        /**\brief accumulate_ts, accumulate time-series
         *
         * Represents a ts that for
         * the specified time-axis the accumulated sum
         * of the underlying specified TS ts.
         * The i'th value in the time-axis is computed
         * as the sum of the previous true-averages.
         * The ts_point_fx is POINT_INSTANT_VALUE
         * definition:
         * The value of t0=time_axis(0) is zero.
         * The value of t1= time_axis(1) is defined as
         * integral_of f(t) dt from t0 to t1, skipping nan-areas.
         *
         */
        template<class TS, class TA>
        struct accumulate_ts {
            typedef TA ta_t;
            TA ta;
            TS ts;
            ts_point_fx fx_policy=POINT_INSTANT_VALUE;
            const TA& time_axis() const { return ta; }
            ts_point_fx point_interpretation() const { return fx_policy; }
            void set_point_interpretation(ts_point_fx point_interpretation) { fx_policy=point_interpretation;}

            accumulate_ts()=default;
            accumulate_ts(const TS&ts, const TA& ta)
                :ta(ta), ts(ts)
                 {
            } // because accumulate represents the integral of the distance from t0 to t, valid at t

            point get(size_t i) const { return point(ta.time(i), value(i)); }
            size_t size() const { return ta.size(); }
            size_t index_of(utctime t) const { return ta.index_of(t); }
            utcperiod total_period()const {return ta.total_period();}
            //--
            double value(size_t i) const {
                if (i >= ta.size())
                    return nan;
                if (i == 0)
                    return 0.0;
                size_t ix_hint = 0;// we have to start at the beginning
                utcperiod accumulate_period(ta.time(0), ta.time(i));
                utctimespan tsum;
                return accumulate_value(ts, accumulate_period, ix_hint,tsum, d_ref(ts).fx_policy == ts_point_fx::POINT_INSTANT_VALUE);// also note: average of non-nan areas !
            }
            double operator()(utctime t) const {
                size_t i = ta.index_of(t);
                if (i == string::npos || ta.size()==0)
                    return nan;
                if (t == ta.time(0))
                    return 0.0; // by definition
                utctimespan tsum;
                size_t ix_hint = 0;
                return accumulate_value(ts, utcperiod(ta.time(0),t), ix_hint,tsum, d_ref(ts).fx_policy == ts_point_fx::POINT_INSTANT_VALUE);// also note: average of non-nan areas !;
            }
            x_serialize_decl();
        };

        /**\brief A simple profile description defined by start-time and a equidistance by delta-t value-profile.
         *
         * The profile_description is used to define the profile by a vector of double-values.
         * It's first use is in conjunction with creating a time-series with repeated profile
         * that play a role when creating daily bias-profiles for forecast predictions.
         * The class provides t0,dt, ,size() and operator()(i) to access profile values.
         *  -thus we can use other forms of profile-descriptor, not tied to providing the points
         * \sa profile_accessor
         * \sa profile_ts
         */
        struct profile_description {
            utctime t0;
            utctimespan dt;
            std::vector<double> profile;

            profile_description(utctime t0, utctimespan dt, const std::vector<double>& profile) :
                t0(t0), dt(dt), profile(profile) {}
            profile_description() {}

            size_t size() const { return profile.size(); }
            utctimespan duration() const { return dt * size(); }
            void reset_start(utctime ta0) {
                auto offset = (t0 - ta0) / duration();
                t0 -= offset * duration();
            }
            double operator()(size_t i) const {
                if (i < profile.size())
                    return profile[i];
                return nan;
            }
            x_serialize_decl();
        };

        /** \brief profile accessor that enables use of average_value etc.
         *
         * The profile accessor provide a 'point-source' compatible interface signatures
         * for use by the average_value-functions.
         * It does so by repeating the profile-description values over the complete specified time-axis.
         * Each time-point within that time-axis can be mapped to an interval/index of the original
         * profile so that it 'appears' as the profile is repeated the number of times needed to cover
         * the time-axis.
         *
         * Currently we also delegate the 'hard-work' of the op(t) and value(i'th period) here so
         * that the periodic_ts, -using profile_accessor is at a minimun.
         *
         * \note that the profile can start anytime before or equal to the time-series time-axis.
         * \remark .. that we should add profile_description as template arg
         */
        template<class TA>
        struct profile_accessor {
            TA ta; ///< The time-axis that we want to map/repeat the profile on to
            profile_description profile;///<< the profile description, we use .t0, dt, duration(),.size(),op(). reset_start_time()
            ts_point_fx fx_policy;

            profile_accessor( const profile_description& pd, const TA& ta, ts_point_fx fx_policy ) :  ta(ta),profile(pd),fx_policy(fx_policy) {
                profile.reset_start(ta.time(0));
            }
            profile_accessor() {}
            /** map utctime t to the index of the profile value array/function */
            utctimespan map_index(utctime t) const { return ((t - profile.t0) / profile.dt) % profile.size(); }
            /** map utctime t to the profile pattern section,
             *  the profile is repeated n-section times to cover the complete time-axis
             */
            utctimespan section_index(utctime t) const { return (t - profile.t0) / profile.duration(); }

            /** returns the value at time t, taking the point_interpretation policy
             * into account and provide the linear interpolated value between two points
             * in the profile if so specified.
             * If linear between points, and last point is nan, first point in interval-value
             * is returned (flatten out f(t)).
             * If point to the left is nan, then nan is returned.
             */
            double value(utctime t) const {
                int i = map_index(t);
                if (fx_policy == ts_point_fx::POINT_AVERAGE_VALUE)
                    return profile(i);
                //-- interpolate between time(i)    .. t  .. time(i+1)
                //                       profile(i)          profile((i+1) % nt)
                double p1 = profile(i);
                double p2 = profile((i+1) % profile.size());
                if (!isfinite(p1))
                    return nan;
                if (!isfinite(p2))
                    return p1; // keep value until nan

                int s = section_index(t);
                auto ti = profile.t0 + s * profile.duration() + i * profile.dt;
                double w1 = (t - ti) / profile.dt;
                return p1*(1.0-w1) + p2*w1;
            }
            double value(size_t i) const {
                auto p = ta.period(i);
                size_t ix = index_of(p.start); // the perfect hint, matches exactly needed ix
                return average_value(*this, p, ix, ts_point_fx::POINT_INSTANT_VALUE == fx_policy);
            }
            // provided functions to the average_value<..> function
            size_t size() const { return profile.size() * (1 + ta.total_period().timespan() / profile.duration()); }
            point get(size_t i) const { return point(profile.t0 + i*profile.dt, profile(i % profile.size())); }
            size_t index_of(utctime t) const { return map_index(t) + profile.size()*section_index(t); }
            x_serialize_decl();

        };

        /**\brief periodic_ts, periodic pattern time-series
         *
         * Represents a ts that for the specified time-axis returns:
         * - either the instant value of a periodic profile
         * - or the average interpolated between two values of the periodic profile
         */
        template<class TA>
        struct periodic_ts {
            TA ta;
            profile_accessor<TA> pa;
            ts_point_fx fx_policy=POINT_AVERAGE_VALUE;
            const TA& time_axis() const {return ta;}
            ts_point_fx point_interpretation() const { return fx_policy; }
            void set_point_interpretation(ts_point_fx point_interpretation) { fx_policy=point_interpretation;}

            template <class PD>
            periodic_ts(const PD& pd, const TA& ta, ts_point_fx policy = ts_point_fx::POINT_AVERAGE_VALUE) :
                ta(ta), pa(pd, ta,policy), fx_policy(policy) {}
            periodic_ts(const vector<double>& pattern, utctimespan dt, const TA& ta) :
                periodic_ts(profile_description(ta.time(0), dt, pattern), ta) {}
            periodic_ts(const vector<double>& pattern, utctimespan dt,utctime pattern_t0, const TA& ta) :
                periodic_ts(profile_description(pattern_t0, dt, pattern), ta) {
            }
            periodic_ts() =default;
            double operator() (utctime t) const { return pa.value(t); }
            size_t size() const { return ta.size(); }
            utcperiod total_period() const {return ta.total_period();}
            size_t index_of(utctime t) const { return ta.index_of(t); }
            double value(size_t i) const { return pa.value(i); }
            std::vector<double> values() const {
                std::vector<double> v;
                v.reserve(ta.size());
                for (size_t i=0; i<ta.size(); ++i)
                    v.emplace_back(value(i));
                return v;
            }
            x_serialize_decl();
        };

        /**\brief glacier melt ts
         *
         * Using supplied temperature and snow covered area[m2] time-series
         * computes the glacier melt in units of [m3/s] using the
         * the following supplied parameters:
         *  -# dtf:day temperature factor (dtf),
         *  -# glacier_area_m2: glacier area in [m2] units
         *
         * \tparam TS a time-series type
         * \note that both temperature and snow covered area (sca) TS is of same type
         * \ref shyft::core::glacier_melt::step function
         */
        template<class TS_A,class TS_B=TS_A>
        struct glacier_melt_ts {
            typedef typename d_ref_t<TS_A>::type::ta_t ta_t;
            TS_A temperature;
            TS_B sca_m2;
            double glacier_area_m2=0.0;
            double dtf=0.0;
            ts_point_fx fx_policy=POINT_AVERAGE_VALUE;
            const ta_t& time_axis() const { return d_ref(temperature).time_axis(); }
            ts_point_fx point_interpretation() const { return fx_policy; }
            void set_point_interpretation(ts_point_fx point_interpretation) { fx_policy=point_interpretation;}

            /** construct a glacier_melt_ts
             * \param temperature in [deg.C]
             * \param sca_m2 snow covered area [m2]
             * \param glacier_area_m2 [m2]
             * \param dtf degree timestep factor [mm/day/deg.C]; lit. values for Norway: 5.5 - 6.4 in Hock, R. (2003), J. Hydrol., 282, 104-115.
             */
            template<class A_,class B_>
            glacier_melt_ts(A_&& temperature, B_&& sca_m2, double glacier_area_m2, double dtf)
                :temperature(forward<A_>(temperature)), sca_m2(forward<B_>(sca_m2)),glacier_area_m2(glacier_area_m2),dtf(dtf)
                {
            }
            // std. ct etc
            glacier_melt_ts()=default;
            // ts property definitions
            point get(size_t i) const { return point(time_axis().time(i), value(i)); }
            size_t size() const { return time_axis().size(); }
            size_t index_of(utctime t) const { return time_axis().index_of(t); }
            utcperiod total_period() const {return time_axis().total_period();}
            //--
            double value(size_t i) const {
                if (i >= time_axis().size())
                    return nan;
                utcperiod p=time_axis().period(i);
                double t_i = d_ref(temperature).value(i);
                size_t ix_hint=i;// assume same indexing of sca and temperature
                double sca_m2_i= average_value(d_ref(sca_m2),p,ix_hint,d_ref(sca_m2).point_interpretation()==ts_point_fx::POINT_INSTANT_VALUE);
                return shyft::core::glacier_melt::step(dtf, t_i,sca_m2_i, glacier_area_m2);
            }
            double operator()(utctime t) const {
                size_t i = index_of(t);
                if (i == string::npos)
                    return nan;
                return value(i);
            }
            x_serialize_decl();
        };

        /**\brief a symbolic reference ts
         *
         * A ref_ts is a time-series that have a ref, a uid, to some underlying
         * time-series data store.
         * It can be bound, or unbound.
         * If it's bound, then it behaves like the underlying time-series ts class, typically
         * a point_ts<ta> time-series, with a time-axis and some values associated.
         * If it's unbound, then any attempt to use it will cause runtime_error exception
         * to be thrown.
         *
         * The purpose of the ref_ts is to provide means of (unbound) time-series expressions
         * to be transferred to a node, where the binding process can be performed, and the resulting
         * values of the expression can be computed.
         * The result can then be transferred back to the origin (client) for further processing/presentation.
         *
         * This allow for execution of ts-expressions that reduces data
         *  to be executed at nodes close to data.
         *
         * Note that the mechanism is also useful, even for local evaluation of expressions, since it allow
         * analysis and binding to happen before the final evaluation of the values of the expression.
         *
         */
        template <class TS>
        struct ref_ts {
            typedef typename TS::ta_t ta_t;
            string ref;///< reference to time-series supporting storage
            shared_ptr<TS> ts;
            TS& bts() {
                if(ts==nullptr)
                    throw runtime_error("unbound access to ref_ts attempted");
                return *ts;
            }
            const TS& bts() const {
                if(ts==nullptr)
                    throw runtime_error("unbound access to ref_ts attempted");
                return *ts;
            }
            ts_point_fx point_interpretation() const {
                return bts().point_interpretation();
            }
            void set_point_interpretation(ts_point_fx point_interpretation) {
                bts().set_point_interpretation(point_interpretation);
            }
            // std. ct/dt etc.
            ref_ts() = default;

            void set_ts(shared_ptr<TS>const &tsn) {
                ts = tsn;
            }
            // useful constructors goes here:
            explicit ref_ts(const string& sym_ref) :ref(sym_ref) {}//, fx_policy(POINT_AVERAGE_VALUE) {}
            const ta_t& time_axis() const {return bts().time_axis();}
            /**\brief the function value f(t) at time t, fx_policy taken into account */
            double operator()(utctime t) const {
                return bts()(t);
            }

            /**\brief the i'th value of ts
             */
            double value(size_t i) const  {
                return bts().value(i);
            }
            // BW compatiblity ?
            size_t size() const { return bts().size();}
            size_t index_of(utctime t) const {return bts().index_of(t);}
            utcperiod total_period() const {return bts().total_period();}
            utctime time(size_t i ) const {return bts().time(i);}

            // to help average_value method for now!
            point get(size_t i) const {return bts().get(i);}

            // Additional write/modify interface to operate directly on the values in the time-series
            void set(size_t i,double x) {bts().set(i,x);}
            void add(size_t i, double value) { bts().add(i,value); }
            void add(const point_ts<ta_t>& other) {
                bts().add(other);
            }
            void add_scale(const point_ts<ta_t>&other,double scale) {
                bts().add_scale(other,scale);
            }
            void fill(double value) { bts().fill(value); }
            void fill_range(double value, int start_step, int n_steps) {
                bts().fill_range(value,start_step,n_steps);
            }
            void scale_by(double value) {
                bts().scale_by(value);
            }
            x_serialize_decl();
        };


        /** The convolve_policy determines how the convolve_w_ts functions deals with the
        *  initial boundary condition, i.e. when trying to convolve with values before
        *  the first value, value(0)
        * \sa convolve_w_ts
        */
        enum convolve_policy {
            USE_FIRST, ///< ts.value(0) is used for all values before value(0): 'mass preserving'
            USE_ZERO, ///< fill in zero for all values before value(0):shape preserving
            USE_NAN ///< nan filled in for the first length of the filter
        };
        /** \brief convolve_w convolves a time-series with weights w
        *
        * The resulting time-series value(i) is the result of convolution (ts*w)|w.size()
        *  value(i) => ts(i-k)*w(k), k is [0..w.size()>
        *
        * the convolve_policy determines how to resolve start-condition-zone
        * where i < w.size(), \ref convolve_policy
        *
        * \tparam Ts any time-series
        *  maybe \tparam W any container with .size() and [i] operator
        */
        template<class Ts>
        struct convolve_w_ts {
            typedef typename Ts::ta_t ta_t;
            typedef std::vector<double> W;
            Ts ts;
            ts_point_fx fx_policy=POINT_AVERAGE_VALUE;
            W w;
            convolve_policy policy = convolve_policy::USE_FIRST;
			bool bound = false;
            //-- default stuff, ct/copy etc goes here
            convolve_w_ts() = default;

            //-- useful ct goes here
            template<class A_, class W_>
            convolve_w_ts(A_ && tsx, W_ && w, convolve_policy policy = convolve_policy::USE_FIRST)
                :ts(std::forward<A_>(tsx)),
                w(std::forward<W_>(w)),
                policy(policy) {
                //if( !(needs_bind<typename d_ref_t<Ts>::type>::value && e_needs_bind(ts))) {
				if ( ! e_needs_bind(d_ref(ts)) ) {
					local_do_bind();
				}
            }
			void do_bind() {
				ts.do_bind();
				local_do_bind();
			}
			bool needs_bind() const {
				return ! bound;
			}

			void local_do_bind() {
				if ( ! bound ) {
					fx_policy=d_ref(ts).point_interpretation();
					bound = true;
				}
			}

            const ta_t& time_axis() const { return ts.time_axis(); }
            ts_point_fx point_interpretation() const { return fx_policy; }
            void set_point_interpretation(ts_point_fx point_interpretation) { fx_policy = point_interpretation; }

            point get(size_t i) const { return point(ts.time(i), value(i)); }

            size_t size() const { return ts.size(); }
            size_t index_of(utctime t) const { return ts.index_of(t); }
            utcperiod total_period() const { return ts.total_period(); }
            utctime time(size_t i) const { return ts.time(i); }

            //--
            double value(size_t i) const {
                double v = 0.0;
                for (size_t j = 0;j<w.size();++j)
                    v += j <= i ?
                    w[j] * ts.value(i - j)
                    :
                    (policy == convolve_policy::USE_FIRST ? w[j] * ts.value(0) :
                    (policy == convolve_policy::USE_ZERO ? 0.0 : shyft::nan)); // or nan ? policy based ?
                return  v;
            }
            double operator()(utctime t) const {
                return value(ts.index_of(t));
            }
            x_serialize_decl();
        };

        /** \brief a sum_ts computes the sum ts  of vector<ts>
        *
        * Enforces all ts do have at least same number of elements in time-dimension
        * Require number of ts >0
        * computes .value(i) so that it is the sum of all tsv.value(i)
        * time-axis equal to tsv[0].time_axis
        * time-axis should be equal, notice that only .size() is equal is ensured in the constructor
        *
        *
        * \note The current approach only works for ts of same type, enforced by the compiler
        *  The runtime enforces that each time-axis are equal as well, and throws exception
        *  if not (in the non-default constructor), - if user later modifies the tsv
        *  it could break this assumption.
        *  Point interpretation is default POINT_AVERAGE_VALUE, you can override it in ct.
        *  Currently ts-vector as expressions is not properly supported (they need to be bound before passed to ct!)
        */
        template<class T>
        struct uniform_sum_ts {
            typedef typename T::ta_t ta_t;
        private:
            std::vector<T> tsv; ///< need this private to ensure consistency after creation
        public:
            ts_point_fx fx_policy=POINT_AVERAGE_VALUE; // inherited from ts
            uniform_sum_ts() = default;


            //-- useful ct goes here
            template<class A_>
            uniform_sum_ts(A_ && tsv,ts_point_fx fx_policy=POINT_AVERAGE_VALUE)
                :tsv(std::forward<A_>(tsv)),
                fx_policy(fx_policy) {
                if (tsv.size() == 0)
                    throw std::runtime_error("vector<ts> size should be > 0");
                for (size_t i = 1;i < tsv.size();++i)
                    if (!(tsv[i].time_axis() == tsv[i - 1].time_axis())) // todo: a more extensive test needed, @ to high cost.. so  maybe provide a ta, and do true-average ?
                        throw std::runtime_error("vector<ts> timeaxis should be aligned in sizes and numbers");
            }

            const ta_t& time_axis() const { return tsv[0].time_axis(); }
            ts_point_fx point_interpretation() const { return fx_policy; }
            void set_point_interpretation(ts_point_fx point_interpretation) { fx_policy = point_interpretation; }

            point get(size_t i) const { return point(tsv[0].time(i), value(i)); }

            size_t size() const { return tsv.size() ? tsv[0].size() : 0; }
            size_t index_of(utctime t) const { return tsv.size() ? tsv[0].index_of(t) : std::string::npos; }
            utcperiod total_period() const { return tsv.size() ? tsv[0].total_period() : utcperiod(); }
            utctime time(size_t i) const { if (tsv.size()) return tsv[0].time(i); throw std::runtime_error("uniform_sum_ts:empty sum"); }

            //--
            double value(size_t i) const {
                if (tsv.size() == 0 || i == std::string::npos) return shyft::nan;
                double v = tsv[0].value(i);
                for (size_t j = 1;j < tsv.size();++j) v += tsv[j].value(i);
                return  v;
            }
            double operator()(utctime t) const {
                return value(index_of(t));
            }
            std::vector<double> values() const {
                std::vector<double> r;r.reserve(size());
                for (size_t i = 0;i < size();++i) r.push_back(value(i));
                return r;
            }
        };

        template <class A>
        constexpr void dbind_ts(A&) {}

        /** \brief Basic math operators
         *
         * Here we take a very direct approach, just create a bin_op object for
         * each bin_op :
         *   -# specialize for double vs. ts
         *   -# use perfect forwarding of the arguments, using d_ref to keep shared_ptr vs. object the same
         *
         */
        template<class A, class B, class O, class TA>
        struct bin_op {
            typedef TA ta_t;
            O op;
            A lhs;
            B rhs;
            TA ta;
            bool bind_done = false;
            ts_point_fx fx_policy=POINT_AVERAGE_VALUE;
            bin_op() = default;

            template<class A_,class B_>
            bin_op(A_&& lhsx,O op,B_&& rhsx):op(op),lhs(forward<A_>(lhsx)),rhs(forward<B_>(rhsx)) {
                //-- if lhs and rhs allows us (i.e) !needs_bind, we do the binding now
                if(  !(needs_bind<typename d_ref_t<A>::type >::value && e_needs_bind(d_ref(lhs)))
                   &&!(needs_bind<typename d_ref_t<B>::type >::value && e_needs_bind(d_ref(rhs))) )
                   do_bind();
            }

            void ensure_bound() const {
                if(!bind_done)
                    throw runtime_error("bin_op: access to not yet bound attempted");
            }
            void do_bind()  {
                if(!bind_done) {
                    dbind_ts(d_ref(lhs));
                    dbind_ts(d_ref(rhs));
                    ta = time_axis::combine(d_ref(lhs).time_axis(), d_ref(rhs).time_axis());
                    fx_policy = result_policy(d_ref(lhs).point_interpretation(), d_ref(rhs).point_interpretation());
                    bind_done = true;
                }
            }

            const TA& time_axis() const {
				ensure_bound();
                return ta;
            }
            ts_point_fx point_interpretation() const {
				ensure_bound();
                return fx_policy;
            }
            void set_point_interpretation(ts_point_fx point_interpretation) {
                fx_policy=point_interpretation;
            }
            inline double value_at(utctime t) const {
                return op(d_ref(lhs)(t),d_ref(rhs)(t));
            }
            double operator()(utctime t) const {
                if(!time_axis().total_period().contains(t))
                    return nan;
                return value_at(t);
            }
            double value(size_t i) const {
                if(i==string::npos || i>=time_axis().size() )
                    return nan;
                return value_at(ta.time(i));
            }
            size_t size() const {
				ensure_bound();
                return ta.size();
            }
            x_serialize_decl();
       };

        template<class A, class B, class O, class TA>
        void dbind_ts(bin_op<A,B,O,TA>&&ts) {
            std::forward<bin_op<A,B,O,TA>>(ts).do_bind();
        }

        /** specialize for double bin_op ts */
        template<class B,class O,class TA>
        struct bin_op<double,B,O,TA> {
            typedef TA ta_t;
            double lhs;
            B rhs;
            O op;
            TA ta;
            bool bind_done = false;
            ts_point_fx fx_policy = POINT_AVERAGE_VALUE;
            bin_op() = default;

            template<class A_,class B_>
            bin_op(A_&& lhsx,O op,B_&& rhsx):lhs(forward<A_>(lhsx)),rhs(forward<B_>(rhsx)),op(op) {
                //-- if lhs and rhs allows us (i.e) !needs_bind, we do the binding now
                if( !(needs_bind<typename d_ref_t<B>::type >::value && e_needs_bind(d_ref(rhs))) )
                   do_bind();

            }

            void do_bind()  {
                if (!bind_done) {
                    dbind_ts(d_ref(rhs));
                    ta = d_ref(rhs).time_axis();
                    fx_policy = d_ref(rhs).point_interpretation();
                    bind_done = true;
                }
            }

            void ensure_bound() const {
                if(!bind_done)
                    throw runtime_error("bin_op: access to not yet bound attempted");
            }

            const TA& time_axis() const {ensure_bound();return ta;}
            ts_point_fx point_interpretation() const {
				ensure_bound();
                return fx_policy;
            }
            void set_point_interpretation(ts_point_fx point_interpretation) {
				ensure_bound();//ensure it's done
                fx_policy=point_interpretation;
            }

            double operator()(utctime t) const {return op(lhs,d_ref(rhs)(t));}
            double value(size_t i) const {return op(lhs,d_ref(rhs).value(i));}
            size_t size() const {
				ensure_bound();
                return ta.size();
            }
            x_serialize_decl();
        };

        /** specialize for ts bin_op double */
        template<class A,class O,class TA>
        struct bin_op<A,double,O,TA> {
            typedef TA ta_t;
            A lhs;
            double rhs;
            O op;
            TA ta;
            bool bind_done = false;
            ts_point_fx fx_policy = POINT_AVERAGE_VALUE;
            bin_op() = default;

            template<class A_,class B_>
            bin_op(A_&& lhsx,O op,B_&& rhsx):lhs(forward<A_>(lhsx)),rhs(forward<B_>(rhsx)),op(op) {
                //-- if hs allows us (i.e) !needs_bind, we do the binding now
                if( !(needs_bind<typename d_ref_t<A>::type >::value && e_needs_bind(d_ref(lhs))) )
                   do_bind();
            }

            void do_bind()  {
                if (!bind_done) {
                    dbind_ts(d_ref(lhs));
                    ta = d_ref(lhs).time_axis();
                    fx_policy = d_ref(lhs).point_interpretation();
                    bind_done = true;
                }
            }
            void ensure_bound() const {
                if(!bind_done)
                    throw runtime_error("bin_op: access to not yet bound attempted");
            }

            const TA& time_axis() const {ensure_bound();return ta;}
            ts_point_fx point_interpretation() const {
				ensure_bound();
                return fx_policy;
            }
            void set_point_interpretation(ts_point_fx point_interpretation) {
				ensure_bound();//ensure it's done
                fx_policy=point_interpretation;
            }
            double operator()(utctime t) const {return op(d_ref(lhs)(t),rhs);}
            double value(size_t i) const {return op(d_ref(lhs).value(i),rhs);}
            size_t size() const {
				ensure_bound();
                return ta.size();
            }
            x_serialize_decl();
        };


        /** \brief op_axis is about deriving the time-axis of a result
         *
         * When doing binary-ts operations, we need to deduce at run-time what will be the
         *  most efficient time-axis type of the binary result
         * We use time_axis::combine_type<..> if we have two time-series
         * otherwise we specialize for double binop ts, and ts binop double.
         * \sa shyft::time_axis::combine_type
         */
        template<typename L,typename R>
        struct op_axis {
            typedef typename time_axis::combine_type< typename d_ref_t<L>::type::ta_t, typename d_ref_t<R>::type::ta_t>::type type;
        };

        template<typename R>
        struct op_axis<double,R> {
            typedef typename d_ref_t<R>::type::ta_t type;
        };

        template<typename L>
        struct op_axis<L,double> {
            typedef typename d_ref_t<L>::type::ta_t type;
        };

		class rating_curve_segment {

		public:
			double lower;  ///< Lower level for this segment, considered valid indefinitly after.
			double a;  ///< Rating-curve parameter.
			double b;  ///< Rating-curve parameter.
			double c;  ///< Rating-curve parameter.

		public:  // con/de-struction, copy & move
			rating_curve_segment()
				: lower{ 0. }, a{ 0. }, b{ 0. }, c{ 0. } { }
			rating_curve_segment(double lower, double a, double b, double c)
				: lower{ lower }, a{ a }, b{ b }, c{ c } { }
			~rating_curve_segment() = default;
			// -----
			rating_curve_segment(const rating_curve_segment &) = default;
			rating_curve_segment & operator=(const rating_curve_segment &) = default;
			// -----
			rating_curve_segment(rating_curve_segment &&) = default;
			rating_curve_segment & operator=(rating_curve_segment &&) = default;

		public:  // api
			/** Return whether the segment is valid for the level.
			 * Validity is based on whether level is less than lower.
			 */
			bool valid(double level) const {
				return lower <= level;
			}
			// -----
			/** Compute the flow from a water level.
			 * Does _not_ check if `h` is valid according to `lower`.
			 */
			double flow(double level) const {
				return a * std::pow(level - b, c);
			}
			/** Compute the flow for a list of water levels.
			* Does _not_ check if the water levels are valid according to `lower`.
			*/
			std::vector<double> flow(
				const std::vector<double> & levels,
				std::size_t i0 = 0u,
				std::size_t iN = std::numeric_limits<std::size_t>::max()
			) const {
				std::vector<double> flow;
				flow.reserve(levels.size());
				for (std::size_t i = i0, idx_end = std::min(levels.size(), iN); i < idx_end; ++i) {
					flow.emplace_back(a * std::pow(levels[i] - b, c));
				}
				return flow;
			}
			// -----
			operator std::string() {
				std::ostringstream ret{ "rating_curve_segment{ ", std::ios_base::ate };
				ret << "lower=" << lower << " a=" << a << " b=" << b << " c=" << c << " }";
				return ret.str();
			}
			// -----
			/** Compare two rating-curve segments according to their lower value.
			 */
			bool operator< (rating_curve_segment & other) const {
				return this->lower < other.lower;
			}
			/** Compare a rating-curve segment to a value interpreted as a level value.
			*/
			bool operator< (double value) const {
				return this->lower < value;
			}

			x_serialize_decl();
		};

		class rating_curve_function {
			std::vector<rating_curve_segment> segments;  // invariant: This is kept sorted ascending on el.lower!

		public:
			rating_curve_function() = default;
			rating_curve_function(const std::vector<rating_curve_segment> & segment_vector, bool sorted = false)
				: segments{ segment_vector }
			{
				if ( ! sorted )
					std::sort(segments.begin(), segments.end());
			}
			rating_curve_function(std::vector<rating_curve_segment> && segment_vector, bool sorted = false)
				: segments{ std::move(segment_vector) }
			{
				if ( ! sorted)
					std::sort(segments.begin(), segments.end());
			}
			template <class InputIt>
			rating_curve_function(InputIt first, InputIt last, bool sorted = false)
				: segments{ first, last }
			{
				if ( ! sorted)
					std::sort(segments.begin(), segments.end());
			}
			~rating_curve_function() = default;
			// -----
			rating_curve_function(const rating_curve_function &) = default;
			rating_curve_function & operator=(const rating_curve_function &) = default;
			// -----
			rating_curve_function(rating_curve_function &&) = default;
			rating_curve_function & operator=(rating_curve_function &&) = default;

		public:  // api
			decltype(segments)::const_iterator cbegin() const {
				return segments.cbegin();
			}
			decltype(segments)::const_iterator cend() const {
				return segments.cend();
			}
			std::size_t size() const {
				return segments.size();
			}
			// -----
			operator std::string() {
				std::ostringstream ret{ "rating_curve_function{", std::ios_base::ate };
				for ( auto & it : segments )
					ret << " " << static_cast<std::string>(it) << ",";
				ret << " }";
				return ret.str();
			}
			// -----
			void add_segment(double lower, double a, double b, double c) {
				add_segment(rating_curve_segment(lower, a, b, c));
			}
			void add_segment(rating_curve_segment && seg) {
				segments.emplace(
					std::upper_bound(segments.begin(), segments.end(), seg),
					std::forward<rating_curve_segment>(seg) );
			}
			void add_segment(const rating_curve_segment & seg) {
				segments.emplace(std::upper_bound(segments.begin(), segments.end(), seg), seg );
			}
			// -----
			/** Compute the flow at a specific level.
			 */
			double flow(const double level) const {
				if ( segments.size() == 0 )
					throw std::runtime_error("no rating-curve segments");

				// assume segments is sorted ascending on el.lower
				auto it = std::lower_bound(segments.cbegin(), segments.cend(), level);
				if ( it != segments.cend() && level == it->lower ) {
					return it->flow(level);
				} else if ( it != segments.cbegin() ) {  // not at begining? -> compute with prev curve
					return (it - 1)->flow(level);
				} else {  // before first segment -> no flow
					return nan;
				}
			}
			/** Compute the flow for all values in a vector.
			 */
			vector<double> flow(const vector<double> & levels) const {
				return flow(levels.cbegin(), levels.cend());
			}
			/** Compute the flow for all values from a iterator.
			*/
			template <typename InputIt>
			vector<double> flow(InputIt first, InputIt last) const {
				if ( segments.size() == 0 )
					throw std::runtime_error("no rating-curve segments");

				std::size_t count = std::distance(first, last);

				if ( count == 0 )
					return std::vector<double>{};

				std::vector<double> flow;
				flow.reserve(count);
				for ( auto lvl = first; lvl != last; ++lvl ) {
					auto it = std::lower_bound(segments.cbegin(), segments.cend(), *lvl);
					if ( it != segments.cend() && *lvl == it->lower ) {
						flow.emplace_back(it->flow(*lvl));
					} else if ( it != segments.cbegin() ) {  // not at begining? -> compute with prev curve
						flow.emplace_back((*(it - 1)).flow(*lvl));
					} else {  // before first segment -> no flow
						flow.emplace_back(nan);
					}
				}

				return flow;
			}

			x_serialize_decl();
		};

		class rating_curve_parameters {
			std::map<utctime, rating_curve_function> curves;

		public:  // con/de-struction
			rating_curve_parameters() = default;
			/** Instanciate a new rating-curve parameter block from
			* a iterator yielding `std::pair< [const] utctime, rating_curve_function >`.
			*/
			template <typename InputIt>
			rating_curve_parameters(InputIt first, InputIt last)
				: curves{ first, last } { }
			explicit rating_curve_parameters(const std::vector<std::pair<utctime, rating_curve_function>> & curves)
				: rating_curve_parameters{ curves.cbegin(), curves.cend() } { }
			// -----
			~rating_curve_parameters() = default;
			// -----
			rating_curve_parameters(const rating_curve_parameters &) = default;
			rating_curve_parameters & operator= (const rating_curve_parameters &) = default;
			// -----
			rating_curve_parameters(rating_curve_parameters &&) = default;
			rating_curve_parameters & operator= (rating_curve_parameters &&) = default;

		public:  // api
			decltype(curves)::const_iterator cbegin() const {
				return curves.cbegin();
			}
			decltype(curves)::const_iterator cend() const {
				return curves.cend();
			}			// -----
			operator std::string() {
				std::ostringstream ret{ "rating_curve_parameters{", std::ios_base::ate };
				for ( auto & it : curves )
					ret << " " << it.first << ": [ " << static_cast<std::string>(it.second) << " ],";
				ret << " }";
				return ret.str();
			}
			// -----
			void add_curve(utctime t, rating_curve_function && rcf) {
				curves.emplace(t, std::forward<rating_curve_function>(rcf));
			}
			void add_curve(utctime t, const rating_curve_function & rcf) {
				curves.emplace(t, rcf);
			}
			// -----
			/** Apply the rating-curve pack at a specific time. */
			double flow(utctime t, double level) const {
				using curve_vt = decltype(curves)::value_type;
				auto it = std::lower_bound(
					curves.cbegin(), curves.cend(), t,
					[](curve_vt lhs, utctime rhs) -> bool { return lhs.first < rhs; }
				);
				if ( it == curves.cbegin() && it->first > t ) {
					return shyft::nan;
				} else if ( it == curves.cend() || it->first > t ) {
					// last curve valid indefinitly
					it--;
				}
				return it->second.flow(level);
			}
			/** Apply the rating-curve pack on a time-series. */
			template <typename TA>
			std::vector<double> flow(const TA & ta) const {
				auto it = curves.cbegin();
				if ( it == curves.cend() || it->first >= ta.total_period().end ) {  // no curves...
					return std::vector<double>(ta.size(), shyft::nan);
				}
				std::vector<double> flow;
				flow.reserve(ta.size());

				// determine start and pad with nan
				std::size_t i;
				if ( it->first > ta.time(0u) ) {
					i = ta.index_of(it->first);
					for ( std::size_t j = 0u; j < i; ++j ) {
						flow.emplace_back(shyft::nan);
					}
				} else {
					i = 0u;
				}

				auto it_next = it;  // peeking iterator
				it_next++;
				for ( std::size_t dim = ta.size(); i < dim; ++i ) {
					utctime t = ta.time(i);
					double val = ta.value(i);

					if ( it_next != curves.cend() && it_next->first <= t ) {
						// advance both iterators
						it++; it_next++;
					}

					flow.emplace_back(it->second.flow(val));
				}

				return flow;
			}

			x_serialize_decl();
		};

		template <class TS>
		class rating_curve_ts {

		public:  // type api
			using ts_t = TS;
			using ta_t = typename TS::ta_t;

		public:  // data
			ts_t level_ts;
			rating_curve_parameters rc_param;
			// -----
			ts_point_fx fx_policy = ts_point_fx::POINT_INSTANT_VALUE;
			// -----
			bool bound = false;

		public:  // con/de-struction, copy & move
			rating_curve_ts() = default;
			rating_curve_ts(ts_t && ts, rating_curve_parameters && rc,
							ts_point_fx fx_policy = ts_point_fx::POINT_INSTANT_VALUE)
				: level_ts{ std::forward<ts_t>(ts) },
				  rc_param{ std::forward<rating_curve_parameters>(rc) },
				  fx_policy{ fx_policy }
			{
				if( ! e_needs_bind(d_ref(level_ts)) )
					local_do_bind();
			}
			rating_curve_ts(const ts_t & ts, const rating_curve_parameters & rc,
							ts_point_fx fx_policy = ts_point_fx::POINT_INSTANT_VALUE)
				: level_ts{ ts }, rc_param{ rc },
				  fx_policy{ fx_policy }
			{
				if( ! e_needs_bind(d_ref(level_ts)) ) {
					local_do_bind();
				}
			}
			// -----
			~rating_curve_ts() = default;
			// -----
			rating_curve_ts(const rating_curve_ts &) = default;
			rating_curve_ts & operator= (const rating_curve_ts &) = default;
			// -----
			rating_curve_ts(rating_curve_ts &&) = default;
			rating_curve_ts & operator= (rating_curve_ts &&) = default;

		public:  // usage
			bool needs_bind() const {
				return ! bound;
			}
			void do_bind() {
				if ( !bound ) {
					level_ts.do_bind();  // bind water levels
					local_do_bind();
				}
			}
			void local_do_bind() {
				fx_policy = d_ref(level_ts).point_interpretation();
				bound = true;
			}
			void ensure_bound() const {
				if ( ! bound ) {
					throw runtime_error("rating_curve_ts: access to not yet bound attempted");
				}
			}
			// -----
			ts_point_fx point_interpretation() const {
				return fx_policy;
			}
			void set_point_interpretation(ts_point_fx policy) {
				fx_policy = policy;
			}

		public:  // api
			std::size_t size() const {
				return level_ts.size();
			}
			utcperiod total_period() const {
				return level_ts.total_period();
			}
			const ta_t & time_axis() const {
				return level_ts.time_axis();
			}
			// -----
			std::size_t index_of(utctime t) const {
				return level_ts.index_of(t);
			}
			double operator()(utctime t) const {
				ensure_bound();
				return rc_param.flow(t, level_ts(t));
			}
			// -----
			utctime time(std::size_t i) const {
				return level_ts.time(i);
			}
			double value(std::size_t i) const {
				ensure_bound();
				return rc_param.flow(time(i), level_ts.value(i));
			}

			x_serialize_decl();
		};

		/** The template is_ts<T> is used to enable operator overloading +-/  etc to time-series only.
		* otherwise the operators will interfere with other libraries doing the same.
		*/
		template<class T> struct is_ts<point_ts<T>> {static const bool value=true;};
		template<class T> struct is_ts<shared_ptr<point_ts<T>>> {static const bool value=true;};
		template<class T> struct is_ts<time_shift_ts<T>> {static const bool value=true;};
		template<class T> struct is_ts<shared_ptr<time_shift_ts<T>>> {static const bool value=true;};
		template<class T> struct is_ts<uniform_sum_ts<T>> { static const bool value = true; };
		template<class T> struct is_ts<shared_ptr<uniform_sum_ts<T>>> { static const bool value = true; };
		template<class TS> struct is_ts<convolve_w_ts<TS>> { static const bool value = true; };
		template<class TS> struct is_ts<shared_ptr<convolve_w_ts<TS>>> { static const bool value = true; };
		// This is to allow this ts to participate in ts-math expressions
		template<class T> struct is_ts<glacier_melt_ts<T>> {static const bool value=true;};
		template<class T> struct is_ts<shared_ptr<glacier_melt_ts<T>>> {static const bool value=true;};
		template<class TS> struct is_ts<rating_curve_ts<TS>> { static const bool value = true; };

		template<class TS,class TA> struct is_ts<average_ts<TS,TA>> {static const bool value=true;};
		template<class TS,class TA> struct is_ts<shared_ptr<average_ts<TS,TA>>> {static const bool value=true;};
		template<class A, class B, class O, class TA> struct is_ts< bin_op<A,B,O,TA> > {static const bool value=true;};

		template<class T> struct is_ts<ref_ts<T>> {static const bool value=true;};
		template<class T> struct is_ts<shared_ptr<ref_ts<T>>> {static const bool value=true;};

        /** time_shift function, to ease syntax and usability */
        template<class Ts>
        time_shift_ts<typename std::decay<Ts>::type > time_shift( Ts &&ts, utctimespan dt) {return time_shift_ts< typename std::decay<Ts>::type >(std::forward<Ts>(ts),dt);}

        struct op_max {
            double operator()(const double&a,const double&b) const {return max(a,b);}
        };
        struct op_min {
            double operator()(const double&a,const double&b) const {return min(a,b);}
        };

        template <class A,class B, typename =
                    enable_if_t<
                        (is_ts<A>::value && (is_floating_point<B>::value || is_ts<B>::value))
                      ||(is_ts<B>::value && (is_floating_point<A>::value || is_ts<A>::value))
                    >
                  >

        auto operator+ (const A& lhs, const B& rhs) {
            return bin_op<A,B,plus<double>,typename op_axis<A,B>::type> (lhs,plus<double>(),rhs);
        }

        template <class A,class B,typename =
                    enable_if_t<
                        (is_ts<A>::value && (is_floating_point<B>::value || is_ts<B>::value))
                      ||(is_ts<B>::value && (is_floating_point<A>::value || is_ts<A>::value))
                    >
                  >
        auto operator- (const A& lhs, const B& rhs) {
            return bin_op<A,B,minus<double>,typename op_axis<A,B>::type> (lhs,minus<double>(),rhs);
        }

        /** unary minus implemented as -1.0* ts */
        template <class A,typename = enable_if_t< is_ts<A>::value >>
        auto operator- (const A& lhs) {
            return bin_op<A,double,multiplies<double>,typename op_axis<A,double>::type> (lhs,multiplies<double>(),-1.0);
        }

        template <class A,class B,typename =
                    enable_if_t<
                        (is_ts<A>::value && (is_floating_point<B>::value || is_ts<B>::value))
                      ||(is_ts<B>::value && (is_floating_point<A>::value || is_ts<A>::value))
                    >
                  >
        auto operator* (const A& lhs, const B& rhs) {
            return bin_op<A,B,multiplies<double>,typename op_axis<A,B>::type> (lhs,multiplies<double>(),rhs);
        }

        template <class A,class B,typename =
                    enable_if_t<
                        (is_ts<A>::value && (is_floating_point<B>::value || is_ts<B>::value))
                      ||(is_ts<B>::value && (is_floating_point<A>::value || is_ts<A>::value))
                    >
                  >
        auto operator/ (const A& lhs, const B& rhs) {
            return bin_op<A,B,divides<double>,typename op_axis<A,B>::type> (lhs,divides<double>(),rhs);
        }

        template <class A,class B,typename =
                    enable_if_t<
                        (is_ts<A>::value && (is_floating_point<B>::value || is_ts<B>::value))
                      ||(is_ts<B>::value && (is_floating_point<A>::value || is_ts<A>::value))
                    >
                  >
        auto max(const A& lhs, const B& rhs) {
            return bin_op<A,B,op_max,typename op_axis<A,B>::type> (lhs,op_max(),rhs);
        }

        template <class A,class B,typename =
                    enable_if_t<
                        (is_ts<A>::value && (is_floating_point<B>::value || is_ts<B>::value))
                      ||(is_ts<B>::value && (is_floating_point<A>::value || is_ts<A>::value))
                    >
                  >
        auto min(const A& lhs, const B& rhs) {
            return bin_op<A,B,op_min,typename op_axis<A,B>::type> (lhs,op_min(),rhs);
        }

        /** \brief A constant_source, just return the same constant value for all points
         * \tparam TA time-axis
         */
        template<typename TA>
        class constant_timeseries {
            TA time_axis;
            double cvalue;
          public:
            constant_timeseries(){cvalue=0.0;}
            constant_timeseries(const TA& time_axis, double value) : time_axis(time_axis), cvalue(value) {}
            // Source read interface
            point get(size_t i) const { return point(time_axis.time(i), cvalue); }
            utcperiod total_period() const { return time_axis.total_period(); }
            size_t index_of(const utctime tx) const { return time_axis.index_of(tx); }
            size_t size() const { return time_axis.size(); }
            // Accessor interface
            double value(size_t i) const { return cvalue; }
            void fill(double v) { cvalue = v; }
            void fill_range(double v, int start_step, int n_steps) { cvalue = v; }
            ts_point_fx point_interpretation() const {return POINT_AVERAGE_VALUE;}
            void set_point_interpretation(ts_point_fx point_interpretation) {}///<ignored
        };

        // The policy of how the accessors should handle data points after the
        // last point that is given in the source
        enum class extension_policy {
            USE_DEFAULT, ///< Use the 'native' behaviour of the accessor, whatever that may be
            USE_ZERO, ///< fill in zero for all values after end of source length
            USE_NAN ///< nan filled in for all values after end of source length
        };

        /** \brief average_accessor provides a projection/interpretation
         * of the values of a point source on to a time-axis as provided
         * with semantics as needed/practical for this project.
         *
         * This includes interpolation and true average,
         * \note point interpretation:
         *  We try to interpret the points as f(t),
         *   using either
         *    stair-case (start of step)
         *   or
         *    linear between points
         *  interpretations.
         *
         * \note stair-case is ideal for timeseries where each period on the timeaxis
         * contains a value that represents the average value for that timestep.
         *  E.g. average discharge, precipitation etc.
         *
         * \note linear-between points is ideal/better when the time-value points represents
         * an instantaneous value, observed at(or around) that particular timepoints
         *  and where the signal is assumed to follow 'natural' changes.
         *  eg. temperature: observed 12 degC at 8am, 16 degC at 10am, then a linear approximation
         *      is used to calculate f(t) in that interval
         *
         * \note nan-handling semantics:
         *   when a nan point is introduced at to,
         *   then f(t) is Nan from that point until next non-nan point.
         *
         * \note Computation of true average
         *    true-average = 1/(t1-t0) * integral (t0..t1) f(t) dt
         *
         *   Only non-nan parts of the time-axis contributes to the area, integrated time distance.
         *   e.g.: if half the interval have nan values, and remaining is 10.0, then true average is 10.0 (not 5.0).
         *
         *   TODO: maybe this approach fits for some cases, but not all? Then introduce parameters to get what we want ?
         *
         *
         * In addition the accessor allows fast sequential access to these values
         * using clever caching of the last position used in the underlying
         * point source. The time axis and point source can be of any type
         * as listed above as long as the basic type requirement as described below
         * are satisfied.
         * This is typically the case for observed time-series point sources, where values could
         * be missing, bad, or just irregularly sampled.
         *
         * \note Notice that the S and T objects are const referenced and not owned by Accessor,
         * the intention is that accessor is low-cost/light-weight class, to be created
         * and used in tight(closed) local (thread)scopes.
         * The variables it wrapped should have a lifetime longer than accessor.
         * TODO: consider to leave the lifetime-stuff to the user, by spec types..
         *
         * \tparam S point source, must provide:
         *  -# .size() const               -> number of points in the source
         *  -# .index_of(utctime tx) const -> return lower bound index or -1 for the supplied tx
         *  -# .get(size_t i) const        -> return the i'th point  (t,v)
         *  -# .point_fx const -> point_interpretation(POINT_INSTANT_VALUE|POINT_AVERAGE_VALUE)
         * \tparam TA time axis, must provide:
         *  -# operator()(size_t i) const --> return the i'th period of timeaxis
         *  -#
         * \return Accessor that provides:
         * -# value(size_t i) const -> double, the value at the i'th interval of the supplied time-axis
         */

        template <class S, class TA>
        class average_accessor {
          private:
            static const size_t npos = -1;  // msc doesn't allow this std::basic_string::npos;
            mutable size_t last_idx=-1;
            mutable size_t q_idx=-1;// last queried index
            mutable double q_value=nan;// outcome of
            const TA& time_axis;
            const S& source;
            std::shared_ptr<S> source_ref;// to keep ref.counting if ct with a shared-ptr. source will have a const ref to *ref
            bool linear_between_points=false;
            extension_policy ext_policy = extension_policy::USE_DEFAULT;
          public:
            average_accessor(const S& source, const TA& time_axis, extension_policy policy=extension_policy::USE_DEFAULT)
              : last_idx(0), q_idx(npos), q_value(0.0), time_axis(time_axis), source(source),
                linear_between_points(source.point_interpretation() == POINT_INSTANT_VALUE), ext_policy(policy){ /* Do nothing */ }
            average_accessor(const std::shared_ptr<S>& source,const TA& time_axis, extension_policy policy=extension_policy::USE_DEFAULT)// also support shared ptr. access
              : last_idx(0),q_idx(npos),q_value(0.0),time_axis(time_axis),source(*source),
                source_ref(source),linear_between_points(source->point_interpretation() == POINT_INSTANT_VALUE),
                ext_policy(policy) {}


            double value(const size_t i) const {
                if(i == q_idx)
                    return q_value;// 1.level cache, asking for same value n-times, have cost of 1.
                if (ext_policy == extension_policy::USE_NAN && time_axis.time(i) >= source.total_period().end) {
                    q_idx = i;
                    q_value = nan;
                } else if (ext_policy == extension_policy::USE_ZERO && time_axis.time(i) >= source.total_period().end) {
                    q_idx = i;
                    q_value = 0;
                } else {
                    q_value = average_value(source, time_axis.period(q_idx=i), last_idx,linear_between_points);
                }

                return q_value;
            }

            size_t size() const { return time_axis.size(); }
        };

        /**\brief provides the accumulated value accessor over the supplied time-axis
         *
         * Given sequential access, this accessor tries to be smart using previous
         * accumulated value plus the new delta to be efficient computing the
         * accumulated series from another kind of point source.
         */
        template <class S, class TA>
        class accumulate_accessor {
          private:
            static const size_t npos = -1;  // msc doesn't allow this std::basic_string::npos;
            mutable size_t last_idx=-1;
            mutable size_t q_idx=-1;// last queried index
            mutable double q_value=nan;// outcome of

            const TA& time_axis;
            const S& source;
            std::shared_ptr<S> source_ref;// to keep ref.counting if ct with a shared-ptr. source will have a const ref to *ref
            extension_policy ext_policy = extension_policy::USE_DEFAULT;
          public:
            accumulate_accessor(const S& source, const TA& time_axis, extension_policy policy=extension_policy::USE_DEFAULT)
                : last_idx(0), q_idx(npos), q_value(0.0), time_axis(time_axis), source(source), ext_policy(policy) { /* Do nothing */
            }
            accumulate_accessor(const std::shared_ptr<S>& source, const TA& time_axis, extension_policy policy=extension_policy::USE_DEFAULT)// also support shared ptr. access
                : last_idx(0), q_idx(npos), q_value(0.0), time_axis(time_axis), source(*source), source_ref(source) {
            }

            double value(const size_t i) const {
                if (i == 0)
                    return 0.0;// as defined by now, (but if ts is nan, then nan could be correct value!)
                if (i == q_idx)
                    return q_value;// 1.level cache, asking for same value n-times, have cost of 1.
                if (ext_policy == extension_policy::USE_NAN && time_axis.time(i) >= source.total_period().end) {
                    q_value = nan;
                } else {
                    utctimespan tsum = 0;
                    auto t_end = time_axis.time(i);
                    if( ext_policy == extension_policy::USE_ZERO && t_end >= source.total_period().end)
                        t_end=source.total_period().end; // clip to end (effect of use zero at extension
                    if (i > q_idx && q_idx != npos) { // utilize the fact that we already have computed the sum up to q_idx
                        q_value += accumulate_value(source, utcperiod(time_axis.time(q_idx), t_end), last_idx, tsum, source.point_interpretation() == POINT_INSTANT_VALUE);
                    } else { // just have to do the heavy work, calculate the entire sum again.
                        q_value = accumulate_value(source, utcperiod(time_axis.time(0), t_end), last_idx, tsum, source.point_interpretation() == POINT_INSTANT_VALUE);
                    }
                }
                q_idx = i;
                return q_value;
            }

            size_t size() const { return time_axis.size(); }
        };

        /** \brief The direct_accessor template is a fast accessor
         *
         * that utilizes prior knowledge about the source ts, e.g. fixed regualar intervals,
         * where each values represents the average/representative value for the interval.
         * This is typical for our internal timeseries results, and we need to utilize that.
         *
         * \note life-time warning: The supplied constructor parameters are const ref.
         *
         * \tparam S the point_source that need to relate to the supplied timeaxis T
         * \tparam TA the timeaxis that need to match up with the S
         * \sa average_accessor that provides transformation of unknown source to the provide timeaxis
         */
        template <class S, class TA>
        class direct_accessor {
          private:
            const TA& time_axis;
            const S& source;
          public:
            direct_accessor(const S& source, const TA& time_axis)
              : time_axis(time_axis), source(source) {
                  if (source.size() != time_axis.size())
                      throw std::runtime_error("Cannot use time axis to access source: Dimensions don't match.");
              }

            /** \brief Return value at pos, and check that the time axis agrees on the time.
             */
            double value(const size_t i) const {
                point pt = source.get(i);
                if (pt.t != time_axis.time(i))
                    throw std::runtime_error("Time axis and source are not aligned.");
                return pt.v;
            }

            size_t size() const { return time_axis.size(); }
        };

        /** \brief Specialization of the direct_accessor template for commonly used point_source<TA>
         *
         * \note life-time warning: the provided point_source is kept by const ref.
         * \sa direct_accessor
         * \tparam TA the time-axis
         */
        template <class TA>
        class direct_accessor<point_ts<TA>, TA> {
          private:
            const point_ts<TA>& source; //< \note this is a reference to the supplied point_source, so please be aware of life-time
          public:
            direct_accessor(const point_ts<TA>& source, const TA& ta) : source(source) { }

            /** \brief Return value at pos without check since the source has its own timeaxis
             */
            double value(const size_t i) const {
                return source.get(i).v;
            }
            size_t size() const { return source.size(); }
        };


        /** \brief Specialization of direct_accessor for a constant_source
         *
         * utilizes the fact that the values is always the same for the constant.
         * \tparam TA time-axis
         */
        template <class TA>
        class direct_accessor<constant_timeseries<TA>, TA> {
          private:
            const double _value;
          public:
            direct_accessor(const constant_timeseries<TA>& source, const TA& ta) : _value(source.value(0)) {}

            double value(const size_t i) const { return _value; }
        };

        /**specialization of hint-based search for all time-axis that merely can calculate the index from time t
         */
        template<>
        inline size_t hint_based_search<point_ts<time_axis::fixed_dt>>(const point_ts<time_axis::fixed_dt>& source, const utcperiod& p, size_t i) {
            return source.ta.open_range_index_of(p.start);
        }
        template<>
        inline size_t hint_based_search<point_ts<time_axis::calendar_dt>>(const point_ts<time_axis::calendar_dt>& source, const utcperiod& p, size_t i) {
            return source.ta.open_range_index_of(p.start);
        }
        template<>
        inline size_t hint_based_search<point_ts<time_axis::point_dt>>(const point_ts<time_axis::point_dt>& source, const utcperiod& p, size_t i) {
            return source.ta.open_range_index_of(p.start,i);
        }
        template<>
        inline size_t hint_based_search<point_ts<time_axis::generic_dt>>(const point_ts<time_axis::generic_dt>& source, const utcperiod& p, size_t i) {
            return source.ta.open_range_index_of(p.start,i);
        }
        /** \brief Discrete l2 norm of input time series treated as a vector: (sqrt(sum(x_i)))
         *
         * \note only used for debug/printout during calibration
         * \tparam A the accessor type that implement:
         * -#: .size() number of elements the accessor can provide
         * -#: .value(i) value of the i'th element in the accessor
         */
        template <class A>
        double l2_norm(A& ts) {
            double squared_sum = 0.0;
            for (size_t i = 0; i < ts.size(); ++i) {
                double tmp = ts.value(i);
                squared_sum += tmp*tmp;
            }
            return sqrt(squared_sum);
        }

        /**\brief Nash Sutcliffe model effiency coefficient based goal function
        * <a ref href=http://en.wikipedia.org/wiki/Nash%E2%80%93Sutcliffe_model_efficiency_coefficient">NS coeffecient</a>
        * \note throws runtime exception if supplied arguments differs in .size() or .size()==0
        * \note if obs. is a constant, we get 1/0
        * \tparam TSA1 a ts accessor for the observed ts ( support .size() and double .value(i))
        * \tparam TSA2 a ts accessor for the observed ts ( support .size() and double .value(i))
        * \param observed_ts contains the observed values for the model
        * \param model_ts contains the (simulated) model output values
        * \return 1- E, given E=n.s,  i.e. 0 is best performance > 0 .. +oo is less good performance.
        */
        template<class TSA1, class TSA2>
        double nash_sutcliffe_goal_function(const TSA1& observed_ts, const TSA2& model_ts) {
            if (observed_ts.size() != model_ts.size() || observed_ts.size() == 0)
                throw runtime_error("nash_sutcliffe needs equal sized ts accessors with elements >1");
            double sum_of_obs_measured_diff2 = 0;
            double obs_avg = 0;
            for (size_t i = 0; i < observed_ts.size(); ++i) {
                double diff_i = observed_ts.value(i) - model_ts.value(i);
                sum_of_obs_measured_diff2 += diff_i*diff_i;
                obs_avg += observed_ts.value(i);
            }
            obs_avg /= double(observed_ts.size());
            double sum_of_obs_obs_mean_diff2 = 0;
            for (size_t i = 0; i < observed_ts.size(); ++i) {
                double diff_i = observed_ts.value(i) - obs_avg;
                sum_of_obs_obs_mean_diff2 += diff_i*diff_i;
            }
            return sum_of_obs_measured_diff2 / sum_of_obs_obs_mean_diff2;
        }

        /** \brief KLING-GUPTA Journal of Hydrology 377
         *              (2009) 80–91, page 83,
         *                     formula (10), where shorthands are
         *                     a=alpha, b=betha, q =sigma, u=my, s=simulated, o=observed
         *
         * \tparam  running_stat_calculator template class like dlib::running_scalar_covariance<double>
         *          that supports .add(x), hten mean_x|_y stddev_x|_y,correlation
         * \tparam TSA1 time-series accessor that supports .size(), .value(i)
         * \tparam TSA2 time-series accessor that supports .size(), .value(i)
         * \param observed_ts the time-series that is the target, observed true value
         * \param model_ts the time-series that is the model simulated/calculated ts
         * \param s_r the kling gupta scale r factor (weight the correlation of goal function)
         * \param s_a the kling gupta scale a factor (weight the relative average of the goal function)
         * \param s_b the kling gupta scale b factor (weight the relative standard deviation of the goal function)
         * \return EDs= 1-KGEs, that have a minimum at zero
         *
         */

        template<class running_stat_calculator,class TSA1, class TSA2>
        double kling_gupta_goal_function(const TSA1& observed_ts, const TSA2& model_ts,double s_r,double s_a,double s_b) {
            running_stat_calculator rs;
            for (size_t i = 0; i < observed_ts.size(); ++i) {
                double tv = observed_ts.value(i);
                double dv = model_ts.value(i);
                if (isfinite(tv) && isfinite(dv))
                    rs.add(tv, dv);
            }
            double qo = rs.mean_x();
            double qs = rs.mean_y();
            double us = rs.stddev_y();
            double uo = rs.stddev_x();
            double r = rs.correlation();
            double a = qs / qo;
            double b = us / uo;
            if(!isfinite(a)) a = 1.0;//could happen in theory if qo is zero
            if(!isfinite(b)) b = 1.0;// could happen if uo is zero
            // We use EDs to scale, and KGEs = (1-EDs) with max at 1.0, we use 1-KGEs to get minimum as 0 for minbobyqa
            double eds2 = (s_r != 0.0 ? std::pow(s_r*(r - 1), 2) : 0.0) + (s_a != 0.0 ? std::pow(s_a*(a - 1), 2) : 0.0) + (s_b != 0.0 ? std::pow(s_b*(b - 1), 2) : 0.0);
            return /*EDs=*/ sqrt( eds2);
        }
        template<class TSA1, class TSA2>
        double abs_diff_sum_goal_function(const TSA1& observed_ts, const TSA2& model_ts) {
            double abs_diff_sum = 0.0;
            for (size_t i = 0; i < observed_ts.size(); ++i) {
                double tv = observed_ts.value(i);
                double dv = model_ts.value(i);
                if (isfinite(tv) && isfinite(dv))
                    abs_diff_sum +=std::fabs(tv-dv);
            }
            return abs_diff_sum;
        }


        /**\brief partition_by convert a time-series into a vector of time-shifted partitions of ts with a common time-reference
        *
        * The partitions are simply specified by calendar, delta_t (could be symbolic, like YEAR:MONTH:DAY) and n.
        * To make yearly partitions, just pass calendar::YEAR as dt.
        * The t-parameter set the start-time point in the source-time-series, like 1930.09.01
        * The t0-parameter set the common start-time of the new partitions
        *
        * The typical usage will be to use this function to partition years into a vector with e.g.
        * 80 years, where we can do statistics, percentiles to compare and see the different effects of
        * yearly season variations.
        * Note that the function is more general, allowing any periodic partition, like daily, weekly, monthly etc.
        * to study any pattern or statistics that might be periodic by the partition pattern.
        *
        * For exposure to python, additional preparation of the partitions could be useful
        * like .average( timeaxis(t0,deltahours(1),365*24)) to make all with an equal-sized time-axis
        *
        * \tparam rts_t return type time-series, equal to the return-type of the time_shift_func()
        * \tparam time_shift_func a callable type, that accepts ts_t and utctimespan as input and return rts_t
        * \tparam ts_t time-series type that goes into the partition algorithm
        * \param ts of type ts_t
        * \param cal  specifies the calendar to be used for possible calendar and time-zone semantic operations
        * \param t specifies the time-point to start, e.g. 1930.09.01
        * \param dt specifies the calendar-semantic length of the partitions, e.g. calendar::YEAR|MONTH|DAY|WEEK
        * \param n number of partitions, e.g. if you would have 80 yearly partitions, set n=80
        * \param t0 the common time-reference for the partitions, e.g. 2016.09.01 for 80 yearly partitions 1930.09.01 to 2010.09.01
        * \param make_time_shift_fx a callable that accepts const ts_t& and utctimespan and returns a time-shifted ts of type rts_t
        *
        * \return the partition vector, std::vector<rts_t> of size n, where each partition ts have its start-value at t0
        *
        * \note t0 must align with multiple delta-t from t, e.g. if t is 1930.09.1, then t0 must have a pattern like YYYY.09.01
        * \throw runtime_error if t0 is not aligned with t, see note above.
        *
        */
        template <class rts_t, class time_shift_func, class ts_t >
        std::vector<rts_t> partition_by(const ts_t& ts, const calendar&cal, utctime t, utctimespan dt, size_t n, utctime t0, time_shift_func && make_time_shift_fx) {
            utctimespan rem;
            cal.diff_units(t, t0, dt, rem);
            if (rem != utctimespan(0))
                throw std::runtime_error("t0 must align with a complete calendar multiple dt from t");
            std::vector<rts_t> r;r.reserve(n);
            for (size_t i = 0;i<n;++i)
                r.emplace_back(make_time_shift_fx(ts, t0 - cal.add(t, dt, i)));
            return std::move(r);
        }

        /**bind ref_ts
         * default impl. does nothing (nothing to bind)
         */
        template <class Ts, class Fbind>
        void bind_ref_ts(Ts& ts,Fbind && f_bind ) {
        }

        template<class Ts,class Fbind>
        void bind_ref_ts( ref_ts<Ts>& ts,Fbind&& f_bind) {
            f_bind(ts);
        }
        template<class A, class B, class O, class TA,class Fbind>
        void bind_ref_ts(bin_op<A,B,O,TA>& ts,Fbind&& f_bind) {
            bind_ref_ts(d_ref(ts.lhs),f_bind);
            bind_ref_ts(d_ref(ts.rhs),f_bind);
            ts.do_bind();
        }
        template<class B, class O, class TA,class Fbind>
        void bind_ref_ts(bin_op<double,B,O,TA>& ts,Fbind&& f_bind) {
            bind_ref_ts(d_ref(ts.rhs),f_bind);
            ts.do_bind();
        }
        template<class A, class O, class TA,class Fbind>
        void bind_ref_ts(bin_op<A,double,O,TA>& ts,Fbind&& f_bind) {
            bind_ref_ts(d_ref(ts.lhs),f_bind);
            ts.do_bind();
        }
        template <class Ts,class Fbind>
        void bind_ref_ts(time_shift_ts<Ts>&time_shift,Fbind&& f_bind) {
            bind_ref_ts(d_ref(time_shift.ts,f_bind));
            time_shift.do_bind();
        }
        template <class Ts,class Ta,class Fbind>
        void bind_ref_ts(const average_ts<Ts,Ta>& avg, Fbind&& f_bind) {
            bind_ref_ts(d_ref(avg.ts),f_bind);
        }
        template <class Ts,class Ta,class Fbind>
        void bind_ref_ts(accumulate_ts<Ts,Ta>& acc,Fbind&& f_bind) {
            bind_ref_ts(d_ref(acc.ts),f_bind);
        }
        template <class TS_A,class TS_B,class Fbind>
        void bind_ref_ts(glacier_melt_ts<TS_A,TS_B>& glacier_melt,Fbind&& f_bind) {
            bind_ref_ts(d_ref(glacier_melt.temperature),f_bind);
            bind_ref_ts(d_ref(glacier_melt.sca_m2),f_bind);
        }

        /** e_needs_bind(A const&ts) specializations
         *  recall that e_needs_bind(..) is called at run-time to
         *  determine if a ts needs bind before use of any evaluation method.
         *
         *  point_ts<ta> is known to not need bind (so we have to override needs_bind->false
         *   the default value is true.
         *  all others that can be a part of an expression
         *  are candidates, and needs to be visited.
         * this includes
         *  ref_ts : first and foremost, this is the *one* that is the root cause for bind-stuff
         *  bin_op : obvious, the rhs and lhs could contain a  ref_ts
         *  average_ts .. and family : less obvious, but they refer a ts like bin-op
         */

        template<class Ta>
        struct needs_bind<point_ts<Ta>> {static bool const value=false;};

        // the ref_ts conditionally needs a bind, depending on if it has a ts or not
        template<class Ts>
        bool e_needs_bind( ref_ts<Ts> const& rts) {
            return rts.ts==nullptr;
        }

        // the bin_op conditionally needs a bind, depending on their siblings(lhs,rhs)
        template<class A, class B, class O, class TA>
        bool e_needs_bind( bin_op<A,B,O,TA> const &e) { return !e.bind_done;}
        template<class A, class B, class O, class TA>
        bool e_needs_bind( bin_op<A,B,O,TA> &&e) { return !e.bind_done;}

    } // timeseries
} // shyft
//-- serialization support
x_serialize_export_key(shyft::time_series::point_ts<shyft::time_axis::fixed_dt>);
x_serialize_export_key(shyft::time_series::point_ts<shyft::time_axis::calendar_dt>);
x_serialize_export_key(shyft::time_series::point_ts<shyft::time_axis::point_dt>);
x_serialize_export_key(shyft::time_series::point_ts<shyft::time_axis::generic_dt>);
x_serialize_export_key(shyft::time_series::convolve_w_ts<shyft::time_series::point_ts<shyft::time_axis::fixed_dt>>);
x_serialize_export_key(shyft::time_series::convolve_w_ts<shyft::time_series::point_ts<shyft::time_axis::generic_dt>>);


