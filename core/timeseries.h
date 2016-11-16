#pragma once
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

    /** The timeseries namespace contains all needed concepts related
    * to representing and handling time-series concepts efficiently.
    *
    * concepts:
    *  -#: point, -a time,value pair,utctime,double representing points on a time-series, f(t).
    *  -#: timeaxis, -(TA)the fundation for the timeline, represented by the timeaxis class, a non-overlapping set of periods ordered ascending.
    *  -#: point_ts,(S) - provide a source of points, that can be interpreted by an accessor as a f(t)
    *  -#: accessor, -(A) average_accessor and direct_accessor, provides transformation from f(t) to some provide time-axis
    */
    namespace timeseries {

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
        inline bool operator==(const point &a,const point &b)  {return (a.t==b.t) && fabs(a.v-b.v)< EPS;}

        typedef shyft::time_axis::fixed_dt timeaxis;
        typedef time_axis::point_dt point_timeaxis;

        /** \brief Enumerates how points are mapped to f(t)
         *
         * If there is a point_ts, this determines how we would draw f(t).
         * Average values are typically staircase-start-of step, a constant over the interval
         * for which the average is computed.
         * State-in-time values are typically POINT_INSTANT_VALUES; and we could as an approximation
         * draw a straight line between the points.
         */
        enum point_interpretation_policy {
            POINT_INSTANT_VALUE, ///< the point value represents the value at the specific time (or centered around that time),typically linear accessor
            POINT_AVERAGE_VALUE///< the point value represents the average of the interval, typically stair-case start of step accessor

        };
        typedef point_interpretation_policy fx_policy_t;
        //typedef point_interpretation_policy point_interpretation_policy;// BW compatible

        inline point_interpretation_policy result_policy(point_interpretation_policy a, point_interpretation_policy b) {
            return a==point_interpretation_policy::POINT_INSTANT_VALUE || b==point_interpretation_policy::POINT_INSTANT_VALUE?point_interpretation_policy::POINT_INSTANT_VALUE:point_interpretation_policy::POINT_AVERAGE_VALUE;
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
         * the point_interpretation_policy that determine how to compute the
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
            point_interpretation_policy fx_policy;

            point_interpretation_policy point_interpretation() const { return fx_policy; }
            void set_point_interpretation(point_interpretation_policy point_interpretation) { fx_policy=point_interpretation;}

            point_ts():fx_policy(point_interpretation_policy::POINT_INSTANT_VALUE){}
            point_ts(const TA& ta, double fill_value,point_interpretation_policy fx_policy=POINT_INSTANT_VALUE):ta(ta),v(ta.size(),fill_value),fx_policy(fx_policy) {}
            point_ts(const TA& ta,const vector<double>&vx,point_interpretation_policy fx_policy=POINT_INSTANT_VALUE):ta(ta),v(vx),fx_policy(fx_policy) {
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
                if( fx_policy==point_interpretation_policy::POINT_INSTANT_VALUE && i+1<ta.size() && isfinite(v[i+1])) {
                    utctime t1=ta.time(i);
                    utctime t2=ta.time(i+1);
                    double f= double(t2-t)/double(t2-t1);
                    return v[i]*f + (1.0-f)*v[i+1];
                }
                return v[i]; // just keep current value flat to +oo or nan
            }

            /**\brief value of the i'th interval fx_policy taken into account,
             * Hmm. is that policy ever useful in this context ?
             */
            double value(size_t i) const  {
                //if( fx_policy==point_interpretation_policy::POINT_INSTANT_VALUE && i+1<ta.size() && isfinite(v[i+1]))
                //    return 0.5*(v[i] + v[i+1]); // average of the value, linear between points(is that useful ?)
                return v[i];
            }
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
            point_interpretation_policy fx_policy; // inherited from ts
            utctimespan dt;// despite ta time-axis, we need it
            point_interpretation_policy point_interpretation() const { return fx_policy; }
            void set_point_interpretation(point_interpretation_policy point_interpretation) { fx_policy=point_interpretation;}

            //-- default stuff, ct/copy etc goes here
            time_shift_ts():fx_policy(POINT_AVERAGE_VALUE),dt(0) {}
            time_shift_ts(const time_shift_ts& c):ts(c.ts),ta(c.ta),fx_policy(c.fx_policy),dt(c.dt) {}
            time_shift_ts(time_shift_ts&&c):ts(std::move(c.ts)),ta(std::move(c.ta)),fx_policy(c.fx_policy),dt(c.dt) {}
            time_shift_ts& operator=(const time_shift_ts& o) {
                if(this != &o) {
                    ts=o.ts;
                    ta=o.ta;
                    fx_policy=o.fx_policy;
                    dt=o.dt;
                }
                return *this;
            }

            time_shift_ts& operator=(time_shift_ts&& o) {
                ts=std::move(o.ts);
                ta=std::move(o.ta);
                fx_policy=o.fx_policy;
                dt=o.dt;
                return *this;
            }

            //-- useful ct goes here
            template<class A_>
            time_shift_ts(A_ && ts,utctimespan dt)
                :ts(std::forward<A_>(ts)),
                 ta(time_axis::time_shift(ts.time_axis(),dt)),
                 fx_policy(ts.fx_policy),
                 dt(dt) {}

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
            point_interpretation_policy fx_policy;
            const TA& time_axis() const {return ta;}
            point_interpretation_policy point_interpretation() const { return fx_policy; }
            void set_point_interpretation(point_interpretation_policy point_interpretation) { fx_policy=point_interpretation;}
            average_ts(){} // allow default construct
            average_ts(const TS&ts,const TA& ta)
            :ta(ta),ts(ts)
            ,fx_policy(point_interpretation_policy::POINT_AVERAGE_VALUE) {} // because true-average of periods is per def. POINT_AVERAGE_VALUE
            // to help average_value method for now!
            point get(size_t i) const {return point(ta.time(i),ts.value(i));}
            size_t size() const { return ta.size();}
            size_t index_of(utctime t) const {return ta.index_of(t);}
            //--
            double value(size_t i) const {
                if(i >= ta.size())
                    return nan;
                size_t ix_hint=(i*d_ref(ts).ta.size())/ta.size();// assume almost fixed delta-t.
                //TODO: make specialized pr. time-axis average_value, since average of fixed_dt is trivial compared to other ta.
                return average_value(*this,ta.period(i),ix_hint,d_ref(ts).fx_policy == point_interpretation_policy::POINT_INSTANT_VALUE);// also note: average of non-nan areas !
            }
            double operator()(utctime t) const {
                size_t i=ta.index_of(t);
                if( i==string::npos)
                    return nan;
                return value(i);
            }
        };

		/**\brief accumulate_ts, accumulate time-series
		 *
		 * Represents a ts that for
		 * the specified time-axis the accumulated sum
		 * of the underlying specified TS ts.
		 * The i'th value in the time-axis is computed
		 * as the sum of the previous true-averages.
		 * The point_interpretation_policy is POINT_INSTANT_VALUE
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
			point_interpretation_policy fx_policy;
			const TA& time_axis() const { return ta; }
            point_interpretation_policy point_interpretation() const { return fx_policy; }
            void set_point_interpretation(point_interpretation_policy point_interpretation) { fx_policy=point_interpretation;}

            accumulate_ts():fx_policy(point_interpretation_policy::POINT_INSTANT_VALUE){} // support default construct
			accumulate_ts(const TS&ts, const TA& ta)
				:ta(ta), ts(ts)
				, fx_policy(point_interpretation_policy::POINT_INSTANT_VALUE) {
			} // because accumulate represents the integral of the distance from t0 to t, valid at t

			point get(size_t i) const { return point(ta.time(i), ts.value(i)); }
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
				return accumulate_value(*this, accumulate_period, ix_hint,tsum, d_ref(ts).fx_policy == point_interpretation_policy::POINT_INSTANT_VALUE);// also note: average of non-nan areas !
			}
			double operator()(utctime t) const {
				size_t i = ta.index_of(t);
				if (i == string::npos)
					return nan;
				if (t == ta.time(0))
					return 0.0; // by definition
				utctimespan tsum;
				size_t ix_hint = 0;
				return accumulate_value(*this, utcperiod(ta.time(0),t), ix_hint,tsum, d_ref(ts).fx_policy == point_interpretation_policy::POINT_INSTANT_VALUE);// also note: average of non-nan areas !;
			}
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
			point_interpretation_policy fx_policy;

			profile_accessor( const profile_description& pd, const TA& ta, point_interpretation_policy fx_policy ) :  ta(ta),profile(pd),fx_policy(fx_policy) {
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
				if (fx_policy == point_interpretation_policy::POINT_AVERAGE_VALUE)
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
				return average_value(*this, p, ix, point_interpretation_policy::POINT_INSTANT_VALUE == fx_policy);
			}
			// provided functions to the average_value<..> function
			size_t size() const { return profile.size() * (1 + ta.total_period().timespan() / profile.duration()); }
			point get(size_t i) const { return point(profile.t0 + i*profile.dt, profile(i % profile.size())); }
			size_t index_of(utctime t) const { return map_index(t) + profile.size()*section_index(t); }
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
			point_interpretation_policy fx_policy;
			const TA& time_axis() const {return ta;}
            point_interpretation_policy point_interpretation() const { return fx_policy; }
            void set_point_interpretation(point_interpretation_policy point_interpretation) { fx_policy=point_interpretation;}

            template <class PD>
			periodic_ts(const PD& pd, const TA& ta, point_interpretation_policy policy = point_interpretation_policy::POINT_AVERAGE_VALUE) :
				ta(ta), pa(pd, ta,policy), fx_policy(policy) {}
			periodic_ts(const vector<double>& pattern, utctimespan dt, const TA& ta) :
				periodic_ts(profile_description(ta.time(0), dt, pattern), ta) {}
			periodic_ts(const vector<double>& pattern, utctimespan dt,utctime pattern_t0, const TA& ta) :
				periodic_ts(profile_description(pattern_t0, dt, pattern), ta) {
			}
			periodic_ts() {}
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
				return std::move(v);
			}
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
			double glacier_area_m2;
			double dtf;
			point_interpretation_policy fx_policy;
			const ta_t& time_axis() const { return d_ref(temperature).time_axis(); }
            point_interpretation_policy point_interpretation() const { return fx_policy; }
            void set_point_interpretation(point_interpretation_policy point_interpretation) { fx_policy=point_interpretation;}

			/** construct a glacier_melt_ts
			 * \param temperature in [deg.C]
			 * \param sca_m2 snow covered area [m2]
			 * \param glacier_area_m2 [m2]
			 * \param dtf degree timestep factor [mm/day/deg.C]; lit. values for Norway: 5.5 - 6.4 in Hock, R. (2003), J. Hydrol., 282, 104-115.
			 */
            template<class A_,class B_>
			glacier_melt_ts(A_&& temperature, B_&& sca_m2, double glacier_area_m2, double dtf)
				:temperature(forward<A_>(temperature)), sca_m2(forward<B_>(sca_m2)),glacier_area_m2(glacier_area_m2),dtf(dtf)
				, fx_policy(fx_policy_t::POINT_AVERAGE_VALUE) {
			}
			// std. ct etc
            glacier_melt_ts(){dtf=0.0;glacier_area_m2=0.0;fx_policy=fx_policy_t::POINT_AVERAGE_VALUE;}
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
				double sca_m2_i= average_value(d_ref(sca_m2),p,ix_hint,d_ref(sca_m2).point_interpretation()==fx_policy_t::POINT_INSTANT_VALUE);
				return shyft::core::glacier_melt::step(dtf, t_i,sca_m2_i, glacier_area_m2);
			}
			double operator()(utctime t) const {
				size_t i = index_of(t);
				if (i == string::npos)
					return nan;
                return value(i);
			}
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
            fx_policy_t fx_policy;
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
            point_interpretation_policy point_interpretation() const {
                return bts().point_interpretation();
            }
            void set_point_interpretation(point_interpretation_policy point_interpretation) {
                bts().set_point_interpretation(point_interpretation);
            }
            // std. ct/dt etc.
            ref_ts(){}
            ref_ts(const ref_ts &c):ref(c.ref),fx_policy(c.fx_policy),ts(c.ts) {}
            ref_ts(ref_ts&&c):ref(std::move(c.ref)),fx_policy(c.fx_policy),ts(std::move(c.ts)) {}
            ref_ts& operator=(const ref_ts& c) {
                if(this != &c ) {
                    ref=c.ref;
                    fx_policy=c.fx_policy;
                    ts=c.ts;
                }
                return *this;
            }
            ref_ts& operator=(ref_ts&& c) {
                ref=std::move(c.ref);
                fx_policy=c.fx_policy;
                ts=std::move(c.ts);
                return *this;
            }

            // useful constructors goes here:
            ref_ts(string sym_ref):ref(sym_ref) {}
            const ta_t& time_axis() const {return bts().time_axis();}
            /**\brief the function value f(t) at time t, fx_policy taken into account */
            double operator()(utctime t) const {
                return bts()(t);
            }

            /**\brief value of the i'th interval fx_policy taken into account,
             * Hmm. is that policy ever useful in this context ?
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

        };




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
            bin_op(){}
            point_interpretation_policy fx_policy;
            void deferred_bind() const {
                if(ta.size()==0) {
                    ((bin_op*)this)->ta=time_axis::combine(d_ref(lhs).time_axis(),d_ref(rhs).time_axis());
                    ((bin_op*)this)->fx_policy=result_policy(d_ref(lhs).fx_policy,d_ref(rhs).fx_policy);
                }
            }
            const TA& time_axis() const {
                deferred_bind();
                return ta;
            }
            point_interpretation_policy point_interpretation() const {
                deferred_bind();
                return fx_policy;
            }
            void set_point_interpretation(point_interpretation_policy point_interpretation) {
                fx_policy=point_interpretation;
            }


            template<class A_,class B_>
            bin_op(A_&& lhsx,O op,B_&& rhsx):op(op),lhs(forward<A_>(lhsx)),rhs(forward<B_>(rhsx)) {
                //ta=time_axis::combine(d_ref(lhs).time_axis(),d_ref(rhs).time_axis());
                //fx_policy = result_policy(d_ref(lhs).fx_policy,d_ref(rhs).fx_policy);
            }
            double operator()(utctime t) const {
                if(!time_axis().total_period().contains(t))
                    return nan;
                return op(d_ref(lhs)(t),d_ref(rhs)(t));
            }
            double value(size_t i) const {
                if(i==string::npos || i>=time_axis().size() )
                    return nan;
                if(fx_policy==point_interpretation_policy::POINT_AVERAGE_VALUE)
                    return (*this)(ta.time(i));
                utcperiod p=ta.period(i);
                double v0= (*this)(p.start);
                double v1= (*this)(p.end);
                if(isfinite(v1)) return 0.5*(v0 + v1);
                return v0;
            }
            size_t size() const {
                deferred_bind();
                return ta.size();
            }
        };

        /** specialize for double bin_op ts */
        template<class B,class O,class TA>
        struct bin_op<double,B,O,TA> {
            typedef TA ta_t;
            double lhs;
            B rhs;
            O op;
            TA ta;
            point_interpretation_policy fx_policy;
            bin_op(){}
            void deferred_bind() const {
                if(ta.size()==0) {
                    ((bin_op*)this)->ta=d_ref(rhs).time_axis();
                    ((bin_op*)this)->fx_policy=d_ref(rhs).fx_policy;
                }
            }
            const TA& time_axis() const {deferred_bind();return ta;}
            point_interpretation_policy point_interpretation() const {
                deferred_bind();
                return fx_policy;
            }
            void set_point_interpretation(point_interpretation_policy point_interpretation) {
                deferred_bind();//ensure it's done
                fx_policy=point_interpretation;
            }
            template<class A_,class B_>
            bin_op(A_&& lhsx,O op,B_&& rhsx):lhs(forward<A_>(lhsx)),rhs(forward<B_>(rhsx)),op(op) {
                //ta=d_ref(rhs).time_axis();
                //fx_policy = d_ref(rhs).fx_policy;
            }

            double operator()(utctime t) const {return op(lhs,d_ref(rhs)(t));}
            double value(size_t i) const {return op(lhs,d_ref(rhs).value(i));}
            size_t size() const {
                deferred_bind();
                return ta.size();
            }
        };

        /** specialize for ts bin_op double */
        template<class A,class O,class TA>
        struct bin_op<A,double,O,TA> {
            typedef TA ta_t;
            A lhs;
            double rhs;
            O op;
            TA ta;
            point_interpretation_policy fx_policy;
            bin_op(){}
            void deferred_bind() const {
                if(ta.size()==0) {
                    ((bin_op*)this)->ta=d_ref(lhs).time_axis();
                    ((bin_op*)this)->fx_policy=d_ref(lhs).fx_policy;
                }
            }
            const TA& time_axis() const {deferred_bind();return ta;}
            point_interpretation_policy point_interpretation() const {
                deferred_bind();
                return fx_policy;
            }
            void set_point_interpretation(point_interpretation_policy point_interpretation) {
                deferred_bind();//ensure it's done
                fx_policy=point_interpretation;
            }
            template<class A_,class B_>
            bin_op(A_&& lhsx,O op,B_&& rhsx):lhs(forward<A_>(lhsx)),rhs(forward<B_>(rhsx)),op(op) {
                //ta=d_ref(lhs).time_axis();
                //fx_policy = d_ref(lhs).fx_policy;
            }
            double operator()(utctime t) const {return op(d_ref(lhs)(t),rhs);}
            double value(size_t i) const {return op(d_ref(lhs).value(i),rhs);}
            size_t size() const {
                deferred_bind();
                return ta.size();
            }
        };


        /** \brief op_axis is about deriving the time-axis of a result
         *
         * When doing binary-ts operations, we need to deduce at runtime what will be the
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

        /** The template is_ts<T> is used to enable operator overloading +-/  etc to time-series only.
         * otherwise the operators will interfere with other libraries doing the same.
         */
        template<class T> struct is_ts<point_ts<T>> {static const bool value=true;};
        template<class T> struct is_ts<shared_ptr<point_ts<T>>> {static const bool value=true;};
        template<class T> struct is_ts<time_shift_ts<T>> {static const bool value=true;};
        template<class T> struct is_ts<shared_ptr<time_shift_ts<T>>> {static const bool value=true;};
		// This is to allow this ts to participate in ts-math expressions
		template<class T> struct is_ts<glacier_melt_ts<T>> {static const bool value=true;};
        template<class T> struct is_ts<shared_ptr<glacier_melt_ts<T>>> {static const bool value=true;};

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



        /** \brief A point source with a time_axis, and a function Fx(utctime t) that
         *  gives the value at utctime t, where t is from timeaxis(i).t
         *  suitable for functional type of time series
         * \tparam TA Time axis type that supports:
         *  -# .index_of(tx) const --> -1(npos) or lowerbound index
         *  -# .size() const       --> number of periods on time-axis
         *  -#  op() (i) const     --> utcperiod of the i'th interval
         *
         * \tparam F functor object that supports
         *  -# op()(utctime t) const --> double, F(t)
         */
        template<typename TA,typename F>
        class function_timeseries {
            TA ta;
            F fx;
            point_interpretation_policy point_fx=POINT_INSTANT_VALUE;
          public:

                          /** point intepretation: how we should map points to f(t) */
            point_interpretation_policy point_interpretation() const {return point_fx;}
            void set_point_interpretation(point_interpretation_policy point_interpretation) {point_fx=point_interpretation;}

            function_timeseries(const TA& ta, const F& f,point_interpretation_policy point_interpretation=POINT_INSTANT_VALUE):ta(ta),fx(f) {}
            const TA& time_axis() const {return ta;}

            // Source read interface:
            point get(size_t i) const {
                utctime t = ta.time(i);
                return point(t, (*this)(t));
            } // maybe verify v.size(), vs. time_axis size ?
            size_t size() const {return ta.size();}
            size_t index_of(const utctime tx) const {return ta.index_of(tx);}

            // Specific methods
            double operator()(utctime t) const {return fx(t);}
        };

		//For test&fun, a sinus function
		// r= _y0 + _a * sin( _w * (t-t0) )
		// r > _yMax: r=_yMax
		// r < _yMin: r=_Ymin
		// _w = fullPeriod/2*M_PI
		//  a sinewave for 24 hours, fullPeriod=24*3600
		class sin_fx {
			double y0;
			double y_min;
			double y_max;
			double a;
			double w;
			utctime t0;
		public:
			sin_fx(double y0, double y_min, double y_max, double amplitude, utctime t0, utctimespan full_period)
				: y0(y0), y_min(y_min), y_max(y_max), a(amplitude), w(2.0*M_PI / double(full_period)), t0(t0) {}

			double operator()(utctime t) const {
				double r = y0 + a*sin(w*(t - t0));
				if (r < y_min) r = y_min;
				if (r > y_max) r = y_max;
				return r;
			}
		};

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
            size_t index_of(const utctime tx) const { return time_axis.index_of(tx); }
            size_t size() const { return time_axis.size(); }
            // Accessor interface
            double value(size_t i) const { return cvalue; }
            void fill(double v) { cvalue = v; }
            void fill_range(double v, int start_step, int n_steps) { cvalue = v; }
            point_interpretation_policy point_interpretation() const {return POINT_AVERAGE_VALUE;}
            void set_point_interpretation(point_interpretation_policy point_interpretation) {}///<ignored
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
            mutable size_t last_idx;
            mutable size_t q_idx;// last queried index
            mutable double q_value;// outcome of
            const TA& time_axis;
            const S& source;
            std::shared_ptr<S> source_ref;// to keep ref.counting if ct with a shared-ptr. source will have a const ref to *ref
            bool linear_between_points;
          public:
            average_accessor(const S& source, const TA& time_axis)
              : last_idx(0), q_idx(npos), q_value(0.0), time_axis(time_axis), source(source),
                linear_between_points(source.point_interpretation() == POINT_INSTANT_VALUE){ /* Do nothing */ }
            average_accessor(std::shared_ptr<S> source,const TA& time_axis)// also support shared ptr. access
              : last_idx(0),q_idx(npos),q_value(0.0),time_axis(time_axis),source(*source),
                source_ref(source),linear_between_points(source->point_interpretation() == POINT_INSTANT_VALUE) {}

            size_t get_last_index() const { return last_idx; }  // TODO: Testing utility, remove later.

            double value(const size_t i) const {
                if(i == q_idx)
                    return q_value;// 1.level cache, asking for same value n-times, have cost of 1.
                q_value = average_value(source, time_axis.period(q_idx=i), last_idx,linear_between_points);
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
			mutable size_t last_idx;
			mutable size_t q_idx;// last queried index
			mutable double q_value;// outcome of
			const TA& time_axis;
			const S& source;
			std::shared_ptr<S> source_ref;// to keep ref.counting if ct with a shared-ptr. source will have a const ref to *ref
		public:
			accumulate_accessor(const S& source, const TA& time_axis)
				: last_idx(0), q_idx(npos), q_value(0.0), time_axis(time_axis), source(source) { /* Do nothing */
			}
			accumulate_accessor(std::shared_ptr<S> source, const TA& time_axis)// also support shared ptr. access
				: last_idx(0), q_idx(npos), q_value(0.0), time_axis(time_axis), source(*source), source_ref(source) {
			}

			size_t get_last_index() const { return last_idx; }  // TODO: Testing utility, remove later.

			double value(const size_t i) const {
				if (i == 0)
					return 0.0;// as defined by now, (but if ts is nan, then nan could be correct value!)
				if (i == q_idx)
					return q_value;// 1.level cache, asking for same value n-times, have cost of 1.
				utctimespan tsum = 0;
				if (i > q_idx && q_idx != npos) { // utilize the fact that we already have computed the sum up to q_idx
					q_value += accumulate_value(source, utcperiod(time_axis.time(q_idx), time_axis.time(i)), last_idx, tsum, source.point_interpretation() == POINT_INSTANT_VALUE);
				} else { // just have to do the heavy work, calculate the entire sum again.
					q_value = accumulate_value(source, utcperiod(time_axis.time(0), time_axis.time(i)), last_idx, tsum, source.point_interpretation() == POINT_INSTANT_VALUE);
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
            return /*EDs=*/ sqrt(std::pow(s_r*(r - 1), 2) + std::pow(s_a*(a - 1), 2) + std::pow(s_b*(b - 1), 2));
		}

        /// http://en.wikipedia.org/wiki/Percentile NIST definitions, we use R7, R and excel seems more natural..
        /// http://www.itl.nist.gov/div898/handbook/prc/section2/prc262.htm
        /// calculate percentile using full sort.. works nice for a larger set of percentiles.
        inline vector<double> calculate_percentiles_excel_method_full_sort(vector<double>& samples, const vector<int>& percentiles) {
            vector<double> result; result.reserve(percentiles.size());
            const int n_samples = (int)samples.size();
            const double silent_nan = std::numeric_limits<double>::quiet_NaN();
            if (n_samples == 0) {
                for (size_t i = 0; i < percentiles.size(); ++i)
                    result.emplace_back(silent_nan);
            } else {
                //TODO: filter out Nans
                sort(begin(samples), end(samples));
                for (auto i : percentiles) {
                    // use NIST definition for percentile
                    if (i < 0) { // hack: negative value,  aka. the mean value..
                        double sum = 0; int n = 0;
                        for (auto x : samples) {
                            if (std::isfinite(x)) { sum += x; ++n; }
                        }
                        result.emplace_back(n > 0 ? sum / n : silent_nan);
                    } else {

                        const double eps = 1e-30;
                        // use Hyndman and fam R7 definition, excel, R, and python
                        double nd = 1.0 + (n_samples - 1)*double(i) / 100.0;
                        int  n = int(nd);
                        double delta = nd - n;
                        --n;//0 based index
                        if (n <= 0 && delta <= eps) result.emplace_back(samples.front());
                        else if (n >= n_samples) result.emplace_back(samples.back());
                        else {

                            if (delta < eps) { //direct hit on the index, use just one.
                                result.emplace_back(samples[n]);
                            } else { // in-between two samples, use positional weight
                                auto lower = samples[n];
                                if (n < n_samples - 1)
                                    n++;
                                auto upper = samples[n];
                                result.emplace_back(lower + (delta)*(upper - lower));
                            }
                        }
                    }
                }
            }
            return result;
        }

       /** \brief calculate specified percentiles for supplied list of time-series over the specified time-axis

		Percentiles for a set of timeseries, over a time-axis
		we would like to :
		percentiles_timeseries = calculate_percentiles(ts-id-list,time-axis,percentiles={0,25,50,100})
		done like this
		1..m ts-id, time-axis( start,dt, n),
		read 1..m ts into memory
		percentiles specified is np
		result is percentiles_timeseries
		accessor "accumulate" on time-axis to dt, using stair-case or linear between points
		create result vector[1..n] (the time-axis dimension)
		where each element is vector[1..np] (for each timestep, we get the percentiles
		for each timestep_i in timeaxis
		for each tsa_i: accessors(timeaxis,ts)
		samplevector[timestep_i].emplace_back( tsa_i(time_step_i) )
		percentiles_timeseries[timestep_i]= calculate_percentiles(..)

		\return percentiles_timeseries

		*/
		template <class ts_t,class ta_t>
		inline std::vector< point_ts<ta_t> > calculate_percentiles(const ta_t& ta, const std::vector<ts_t>& ts_list, const std::vector<int>& percentiles,size_t min_t_steps=1000) {
            std::vector<point_ts<ta_t>> result;

			for (size_t r = 0; r < percentiles.size(); ++r) // pre-init the result ts that we are going to fill up
				result.emplace_back(ta, 0.0);

			auto partition_calc=[&result,&ts_list,&ta,&percentiles](size_t i0,size_t n) {
                std::vector < average_accessor<ts_t, ta_t>> tsa_list; tsa_list.reserve(ts_list.size());
                for (const auto& ts : ts_list) // initialize the ts accessors to we can accumulate to time-axis ta e.g.(hour->day)
                    tsa_list.emplace_back(ts, ta);

                std::vector<double> samples(tsa_list.size(), 0.0);

                for (size_t t = i0; t < i0+n; ++t) {//each time step t in the timeaxis, here we could do parallell partition
                    for (size_t i = 0; i < tsa_list.size(); ++i) // get samples from all the "tsa"
                        samples[i] = tsa_list[i].value(t);
                    // possible with pipe-line to percentile calc here !
                    std::vector<double> percentiles_at_t(calculate_percentiles_excel_method_full_sort(samples, percentiles));
                    for (size_t p = 0; p < result.size(); ++p)
                        result[p].set(t, percentiles_at_t[p]);
                }

			};
			if(ta.size()<min_t_steps) {
                partition_calc(0,ta.size());
			} else {
                vector<future<void>> calcs;
                //size_t n_partitions= 1+ ta.size()/min_t_steps;
                for(size_t p=0;p<ta.size(); ) {
                    size_t np = p+ min_t_steps<= ta.size()?min_t_steps:ta.size()-p;
                    calcs.push_back(std::async(std::launch::async,partition_calc,p,np));
                    p+=np;
                }
                for(auto &f:calcs)
                    f.get();

			}

			return std::move(result);
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
        }
        template<class B, class O, class TA,class Fbind>
        void bind_ref_ts(bin_op<double,B,O,TA>& ts,Fbind&& f_bind) {
            //bind_ref_ts(d_ref(ts.lhs),f_bind);
            bind_ref_ts(d_ref(ts.rhs),f_bind);
        }
        template<class A, class O, class TA,class Fbind>
        void bind_ref_ts(bin_op<A,double,O,TA>& ts,Fbind&& f_bind) {
            bind_ref_ts(d_ref(ts.lhs),f_bind);
            //bind_ref_ts(d_ref(ts.rhs),f_bind);
        }
        template <class Ts,class Fbind>
        void bind_ref_ts(time_shift_ts<Ts>&time_shift,Fbind&& f_bind) {
            bind_ref_ts(d_ref(time_shift.ts,f_bind));
        }
        template <class Ts,class Ta,class Fbind>
        void bind_ref_ts(average_ts<Ts,Ta> avg, Fbind&& f_bind) {
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
    } // timeseries
} // shyft
