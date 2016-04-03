#pragma once
#include "compiler_compatiblity.h"
#include "utctime_utilities.h"
#include "time_axis.h"
#include <stdexcept>
#include <limits>
#include <algorithm>
#include <iostream>
#include <vector>
#include <type_traits>
#include <stdexcept>
#include <memory>
#ifndef M_PI
# define M_PI           3.14159265358979323846  /* pi */
#endif
/** \file
* Contains the minimal concepts for the time-series and point source functionality needed in shyft
*/
namespace shyft{

    const double nan = std::numeric_limits<double>::quiet_NaN();
    /** \namespace The timeseries namespace contains all needed concepts related
    * to representing and handling time-series concepts efficiently.
    *
    * concepts:
    *  -#: point, -a time,value pair,utctime,double representing points on a time-series, f(t).
    *  -#: timeaxis, -(TA)the fundation for the timeline, represented by the timeaxis class, a non-overlapping set of periods ordered ascending.
    *  -#: point_source,(S) - provide a source of points, that can be interpreted by an accessor as a f(t)
    *  -#: accessor, -(A) average_accessor and direct_accessor, provides transformation from f(t) to some provide time-axis
    */
    namespace timeseries {

        using namespace shyft::core;
        using namespace std;
        const double EPS=1e-12;

		/** \brief simply a point identified by utctime t and value v */
        struct point {
            utctime t;
            double v;
            point(utctime t=0, double v=0.0) : t(t), v(v) { /* Do nothing */}
#ifndef SWIG
            friend std::ostream& operator<<(std::ostream& os, const point& pt) {
                os << calendar().to_string(pt.t) << ", " << pt.v;
                return os;
            }
#endif
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
        //typedef point_interpretation_policy point_interpretation_policy;// BW compatible

        inline point_interpretation_policy result_policy(point_interpretation_policy a, point_interpretation_policy b) {
            return a==point_interpretation_policy::POINT_INSTANT_VALUE || b==point_interpretation_policy::POINT_INSTANT_VALUE?point_interpretation_policy::POINT_INSTANT_VALUE:point_interpretation_policy::POINT_AVERAGE_VALUE;
        }
        #ifndef SWIG
        //--- to avoid duplicating algorithms in classes where the stored references are
        // either by shared_ptr, or by value, we use template function:
        //  to make shared_ptr<T> appear equal to T
        template<typename T> struct is_shared_ptr {static const bool value=false;};
        template<typename T> struct is_shared_ptr<shared_ptr<T>> {static const bool value=true;};



        /** \brief d_ref function to d_ref object or shared_ptr
         *
         * The T d_ref(T) template to return a ref or const ref to T, if T is shared_ptr<T> or just T
         */
        template<class U> const U& d_ref(const std::shared_ptr<U>& p) { return *p; }
        template<class U> const U& d_ref(const U& u) { return u; }


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
        #endif
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
             * Hmm. is that policy every useful in this context ?
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
			void scale_by(double value) { std::for_each(begin(v), end(v), [value](double&v){v *= value; }); }

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


        #ifndef SWIG

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
            point_interpretation_policy fx_policy;
            const TA& time_axis() const {return ta;}

            template<class A_,class B_>
            bin_op(A_&& lhsx,O op,B_&& rhsx):op(op),lhs(forward<A_>(lhsx)),rhs(forward<B_>(rhsx)) {
                ta=time_axis::combine(d_ref(lhs).time_axis(),d_ref(rhs).time_axis());
                fx_policy = result_policy(d_ref(lhs).fx_policy,d_ref(rhs).fx_policy);
            }
            double operator()(utctime t) const {
                if(!ta.total_period().contains(t))
                    return nan;
                return op(d_ref(lhs)(t),d_ref(rhs)(t));
            }
            double value(size_t i) const {
                if(i==string::npos || i>=ta.size() )
                    return nan;
                if(fx_policy==point_interpretation_policy::POINT_AVERAGE_VALUE)
                    return (*this)(ta.time(i));
                utcperiod p=ta.period(i);
                double v0= (*this)(p.start);
                double v1= (*this)(p.end);
                if(isfinite(v1)) return 0.5*(v0 + v1);
                return v0;
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
            const TA& time_axis() const {return ta;}
            template<class A_,class B_>
            bin_op(A_&& lhsx,O op,B_&& rhsx):lhs(forward<A_>(lhsx)),rhs(forward<B_>(rhsx)),op(op) {
                ta=d_ref(rhs).time_axis();
                fx_policy = d_ref(rhs).fx_policy;
            }

            double operator()(utctime t) const {return op(lhs,d_ref(rhs)(t));}
            double value(size_t i) const {return op(lhs,d_ref(rhs).value(i));}
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
            const TA& time_axis() const {return ta;}
            template<class A_,class B_>
            bin_op(A_&& lhsx,O op,B_&& rhsx):lhs(forward<A_>(lhsx)),rhs(forward<B_>(rhsx)),op(op) {
                ta=d_ref(lhs).time_axis();
                fx_policy = d_ref(lhs).fx_policy;
            }
            double operator()(utctime t) const {return op(d_ref(lhs)(t),rhs);}
            double value(size_t i) const {return op(d_ref(lhs).value(i),rhs);}
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
        template<class T> struct is_ts {static const bool value=false;};
        template<class T> struct is_ts<point_ts<T>> {static const bool value=true;};
        template<class T> struct is_ts<shared_ptr<point_ts<T>>> {static const bool value=true;};

        template<class TS,class TA> struct is_ts<average_ts<TS,TA>> {static const bool value=true;};
        template<class TS,class TA> struct is_ts<shared_ptr<average_ts<TS,TA>>> {static const bool value=true;};
        template<class A, class B, class O, class TA> struct is_ts< bin_op<A,B,O,TA> > {static const bool value=true;};

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


        #endif

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
            point_interpretation_policy point_interpretation() const {return POINT_AVERAGE_VALUE;}
            void set_point_interpretation(point_interpretation_policy point_interpretation) {}///<ignored
        };




        /** \brief hint_based search to eliminate binary-search in irregular time-point series.
         *  utilizing the fact that most access are periods, sequential, so the average_xxx functions
         *  can utilize this, and keep the returned index as a hint for the next request for average_xxx
         *  value.
         * \tparam S a point ts source, must have .get(i) ->point, and .size(), and .index_of(t)->size_t
         * \param source a point ts source as described above
         * \param p utcperiod for which we search a start point <= p.start
         * \param i the start-of-search hint, could be -1, then ts.index_of(p.start) is used to figure out the ix.
         */
        template<class S>
        size_t hint_based_search(const S& source,const utcperiod& p, size_t i) {
            if (source.size() == 0)
                return std::string::npos;
            const size_t n=source.size();
            if (i != std::string::npos && i <n ) { // hint-based search logic goes here:
                const size_t max_directional_search=5;// +-5 just a guess for what is a reasonable try upward/downward
                auto ti = source.get(i).t;
                if (ti == p.start) {// direct hit and extreme luck ?
                    ;// just use it !
                } else if(ti < p.start) { // do a local search upward to see if we find the spot we search for
                    size_t j=0;
                    while(ti < p.start && ++j <max_directional_search && i<n) {
                        ti=source.get(i++).t;
                    }
                    if(ti>=p.start || i==n ) // we startet below p.start, so we got one to far(or at end), so revert back one step
                        --i;
                    else //if( ti !=p.start) // bad luck, local bounded search fail,
                        i=source.index_of(p.start);// cost passed to binary search
                } else if(ti>p.start) { // do a local search downwards from last index, maybe we are lucky
                    size_t j=0;
                    while(ti>p.start && ++j <max_directional_search && i >0) {
                        ti=source.get(--i).t;
                    }
                    if(ti>p.start && i>0) // if we are still not before p.start, and i is >0, there is a hope to find better index, otherwise we are at/before start
                        i=source.index_of(p.start); // bad luck searching downward, need to use binary search.
                }
            } else // no hint given, just use binary search to establish the start.
                i =  source.index_of(p.start);
            return i;
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
         * \param last_idx, in/out, position of the last time point used on the source, updated after each call.
         * \return double, the value at the i'th interval of the supplied time-axis
         */
        template <class S>
        double average_value(const S& source, const utcperiod& p, size_t& last_idx,bool linear=true) {
            const size_t n=source.size();
            if (n == 0) // early exit if possible
                return shyft::nan;
            size_t i=hint_based_search(source,p,last_idx);  // use last_idx as hint, allowing sequential periodic average to execute at high speed(no binary searches)

            if(i==std::string::npos) // this might be a case
                return shyft::nan; // and is equivalent to no points, or all points after requested period.

            point l;// Left point
            bool l_finite=false;

            double area = 0.0;  // Integrated area over the non-nan parts of the time-axis
            utctimespan tsum=0; // length of non-nan f(t) time-axis

            while(true) { //there are two exit criteria: no more points, or we pass the period end.
                if(!l_finite) {//search for 'point' anchor phase
                    l=source.get(i++);
                    l_finite=std::isfinite(l.v);
                    if(i==n) { // exit condition
                        if(l_finite && l.t < p.end ) {//give contribution
                            utctimespan dt= p.end-l.t;
                            tsum += dt;
                            area += dt* l.v; // extrapolate value flat
                        }
                        break;//done
                    }
					if (l.t >= p.end) {//also exit condition, if the point time is after period we search, then no anchor to be found
						break;//done
					}
                } else { // got point anchor l, search for right point/end
                    point r=source.get(i++);// r is guaranteed > p.start due to hint-based search
                    bool r_finite=std::isfinite(r.v);
                    utcperiod px(std::max(l.t,p.start),std::min(r.t,p.end));
                    utctimespan dt= px.timespan();
                    tsum += dt;

                    // now add area contribution for l..r
                    if(linear && r_finite) {
                        double a= (r.v-l.v)/(r.t-l.t);
                        double b=  r.v - a*r.t;
                        //area += dt* 0.5*( a*px.start+b + a*px.end+b);
                        area += dt * ( 0.5*a*(px.start+px.end)+ b);
                    } else { // flat contribution from l  max(l.t,p.start) until max time of r.t|p.end
                        area += l.v*dt;
                    }
                    if(i==n) { // exit condition: final value in sequence, then we need to finish, but
                        if(r_finite && r.t < p.end) {// add area contribution from r and out to the end of p
                            dt= p.end-r.t;
                            tsum += dt;
                            area += dt* r.v; // extrapolate value flat
                        }
                        break;//done
                    }
                    if(r.t >= p.end)
                        break;//also exit condition, done
                    l_finite=r_finite;
                    l=r;
                }

            }
            last_idx=i-1;
            return tsum?area/tsum:shyft::nan;
        }


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
          public:
            average_accessor(const S& source, const TA& time_axis)
              : last_idx(0), q_idx(npos), q_value(0.0), time_axis(time_axis), source(source) { /* Do nothing */ }
            average_accessor(std::shared_ptr<S> source,const TA& time_axis)// also support shared ptr. access
              : last_idx(0),q_idx(npos),q_value(0.0),time_axis(time_axis),source(*source),source_ref(source) {}

            size_t get_last_index() const { return last_idx; }  // TODO: Testing utility, remove later.

            double value(const size_t i) const {
                if(i == q_idx)
                    return q_value;// 1.level cache, asking for same value n-times, have cost of 1.
                q_value = average_value(source, time_axis.period(q_idx=i), last_idx,source.point_interpretation()==POINT_INSTANT_VALUE);
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
		* \ref http://en.wikipedia.org/wiki/Nash%E2%80%93Sutcliffe_model_efficiency_coefficient
		* \note throws runtime exception if supplied arguments differs in .size() or .size()==0
		* \note if obs. is a constant, we get 1/0
		* \tparam TSA1 a ts accessor for the observed ts ( support .size() and double .value(i))
		* \tparam TSA2 a ts accessor for the observed ts ( support .size() and double .value(i))
		* \param observed_ts contains the observed values for the model
		* \param model_ts contains the (simulated) model output values
		* \returns 1- E, given E=n.s,  i.e. 0 is best performance > 0 .. +oo is less good performance.
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

        /** \brief \ref KLING-GUPTA Journal of Hydrology 377
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

    } // timeseries
} // shyft
