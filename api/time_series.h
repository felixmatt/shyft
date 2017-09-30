#pragma once
#ifdef SHYFT_NO_PCH
#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include <utility>
#include <map>

#include "core/core_pch.h"
#endif // SHYFT_NO_PCH

#include "core/utctime_utilities.h"
#include "core/time_axis.h"
#include "core/time_series.h"
#include "core/time_series_statistics.h"
namespace shyft {
    namespace api {
        using namespace shyft::core;
        using namespace shyft::time_series;
        /**
            time-series math to be exposed to python

            This provide functionality like

            a = TsFactory.create_ts(..)
            b = TsFactory.create_ts(..)
            c = a + 3*b
            d = max(c,0.0)

            implementation strategy

            provide a type apoint_ts, that always appears as a time-series.

            It could however represent either a
              point_ts<time_axis:generic_dt>
            or an expression
              like a abin_op_ts( lhs,op,rhs)

            Then we provide operators:
            apoint_ts bin_op a_point_ts
            and
            double bin_op a_points_ts
         */

        /** \brief generic time-axis

            Using the time_axis library and concept directly.
            This time-axis is generic, but currently only dense-types
            fixed_dt (fastest)
            calendar_dt( quite fast, but with calendar semantics)
            point_dt ( a point at start of each interval, plus the end point of the last interval, could give performance hits in some scenarios)

         */
        typedef shyft::time_axis::generic_dt gta_t;
        typedef shyft::time_series::point_ts<gta_t> gts_t;
        typedef shyft::time_series::point_ts<time_axis::fixed_dt> rts_t;
        /** \brief A virtual abstract interface (thus the prefix i) base for point_ts
         *
         * There are three defining properties of a time-series:
         *
         * 1. The \ref time_axis provided by the time_axis() method
         *    or direct time-axis information as provided by the
         *    total_period(),size(), time(i), index_of(t) methods
         *
         * 2. The values, provided by values(),
         *    or any of the value(i),value_at(utctime t) methods.
         *
         * 3. The \ref ts_point_fx
         *    that determines how the points should be projected to f(t)
         *
         */
        struct ipoint_ts {
            typedef gta_t ta_t;// time-axis type
            ipoint_ts() {} // ease boost serialization
            virtual ~ipoint_ts(){}

            virtual ts_point_fx point_interpretation() const =0;
            virtual void set_point_interpretation(ts_point_fx point_interpretation) =0;

            /** \return Returns the effective time-axis of the timeseries
             */
            virtual const gta_t& time_axis() const =0;

            /** \return the total period of time_axis(), same as time_axis().total_period()
             */
            virtual utcperiod total_period() const=0;

            /** \return the index_of(utctime t), using time_axis().index_of(t) \ref time_axis::fixed_dt
             */
            virtual size_t index_of(utctime t) const=0; ///< we might want index_of(t,ix-hint..)

            /** \return number of points that descr. y=f(t) on t ::= period
             */
            virtual size_t size() const=0;

            /** \return time_axis().time(i), the start of the i'th period in the time-axis
             */
            virtual utctime time(size_t i) const=0;

            /** \return the value at the i'th period of the time-axis
             */
            virtual double value(size_t i) const=0;

            /** \return the f(t) at the specified time t, \note if t outside total_period() nan is returned
             */
            virtual double value_at(utctime t) const =0;

            /** \return the values, one for each of the time_axis() intervals as a vector
             */
            virtual std::vector<double> values() const =0;

            /** for internal computation and expression trees, we need to know *if*
             * there are unbound symbolic ts in the chain of this ts
             * We know that a point-ts never do not need a binding, but
             * all others are expressions of ts, and could potentially
             * needs a bind (if it has an unbound symbolic ts in it's siblings)
             */
            virtual bool needs_bind() const =0;

            /** propagate do_bind to expression tree siblings(if any)
             * then do any needed binding stuff needed for the specific class impl.
             */
            virtual void do_bind()=0;

            // to be removed:
            point get(size_t i) const {return point(time(i),value(i));}
            x_serialize_decl();
        };
        struct average_ts;//fwd api
        struct accumulate_ts;//fwd api
        struct integral_ts;// fwd api
        struct time_shift_ts;// fwd api
        struct aglacier_melt_ts;// fwd api
        struct aref_ts;// fwd api
        struct ts_bind_info;
        struct ats_vector;//fwd
        struct abs_ts;//fwd

		/** \brief Enumerates fill policies for time-axis extension.
		 */
		enum extend_ts_fill_policy {
			EPF_NAN,   /**< Fill any gap between the time-axes with NaN. */
			EPF_LAST,  /**< At a gap, keep the last time-axis value through a gap. */
			EPF_FILL,  /**< Fill any gap between the time-axes with a given value. */
		};

		/** \brief Enumerates split policies for time-axis extension.
		 */
		enum extend_ts_split_policy {
			EPS_LHS_LAST,   /**< Split at the last value of the lhs ts. */
			EPS_RHS_FIRST,  /**< Split at the first value of the rhs ts. */
			EPS_VALUE,      /**< Split at a given time-value. */
		};

        /** \brief  apoint_ts, a value-type conceptual ts.
         *
         *  This is the class that we expose to python, with operations, expressions etc.
         *  and that we build all the exposed semantics on.
         *  It holds a shared_ptr to some ipoint_ts, that could be a concrete point timeseries or
         *  an expression.
         *
         */
        struct apoint_ts {
            /** a ref to the real implementation, could be a concrete point ts, or an expression */
            std::shared_ptr<ipoint_ts> ts;// consider unique pointer instead,possibly public, to ease transparency in python

           typedef gta_t ta_t;///< this is the generic time-axis type for apoint_ts, needed by timeseries namespace templates
           friend struct average_ts;
           friend struct integral_ts;
           friend struct time_shift_ts;
           friend struct accumulate_ts;
           friend struct aglacier_melt_ts;
           friend struct abs_ts;
            // constructors that we want to expose
            // like

            // these are for the python exposure
            apoint_ts(const time_axis::fixed_dt& ta,double fill_value,ts_point_fx point_fx=POINT_INSTANT_VALUE);
            apoint_ts(const time_axis::fixed_dt& ta,const std::vector<double>& values,ts_point_fx point_fx=POINT_INSTANT_VALUE);

            apoint_ts(const time_axis::point_dt& ta,double fill_value,ts_point_fx point_fx=POINT_INSTANT_VALUE);
            apoint_ts(const time_axis::point_dt& ta,const std::vector<double>& values,ts_point_fx point_fx=POINT_INSTANT_VALUE);
            explicit apoint_ts(const rts_t & rts);// ct for result-ts at cell-level that we want to wrap.
            apoint_ts(const vector<double>& pattern, utctimespan dt, const time_axis::generic_dt& ta);
            apoint_ts(const vector<double>& pattern, utctimespan dt, utctime pattern_t0,const time_axis::generic_dt& ta);
            // these are the one we need.
            apoint_ts(const gta_t& ta,double fill_value,ts_point_fx point_fx=POINT_INSTANT_VALUE);
            apoint_ts(const gta_t& ta,const std::vector<double>& values,ts_point_fx point_fx=POINT_INSTANT_VALUE);
            apoint_ts(const gta_t& ta,std::vector<double>&& values,ts_point_fx point_fx=POINT_INSTANT_VALUE);

            apoint_ts(gta_t&& ta,std::vector<double>&& values,ts_point_fx point_fx=POINT_INSTANT_VALUE);
            apoint_ts(gta_t&& ta,double fill_value,ts_point_fx point_fx=POINT_INSTANT_VALUE);
            explicit apoint_ts(const std::shared_ptr<ipoint_ts>& c):ts(c) {}

            explicit apoint_ts(std::string ref_ts_id);
            apoint_ts(std::string ref_ts_id, const apoint_ts& x);
            // some more exotic stuff like average_ts

            apoint_ts()=default;
            bool needs_bind() const {
                return sts()->needs_bind();
            }
            string id() const;///< if the ts is aref_ts return it's id (or maybe better url?)
            void do_bind() {
                sts()->do_bind();
            }
            std::shared_ptr<ipoint_ts> const& sts() const {
                if(!ts)
                    throw runtime_error("TimeSeries is empty");
                return ts;
            }
            std::shared_ptr<ipoint_ts> & sts() {
                if(!ts)
                    throw runtime_error("TimeSeries is empty");
                return ts;
            }
            /** support operator! bool  to let an empty ts evaluate to */
            bool operator !() const { // can't expose it as op, due to math promotion
                return !(  ts && ts->size() > 0);
            }
            /**\brief Easy to compare for equality, but tricky if performance needed */
            bool operator==(const apoint_ts& other) const;

            // interface we want to expose
            // the standard ipoint-ts stuff:
            ts_point_fx point_interpretation() const {return ts->point_interpretation();}
            void set_point_interpretation(ts_point_fx point_interpretation) { sts()->set_point_interpretation(point_interpretation); };
            const gta_t& time_axis() const { return sts()->time_axis();};
            utcperiod total_period() const {return ts?ts->total_period():utcperiod();};   ///< Returns period that covers points, given
            size_t index_of(utctime t) const {return ts?ts->index_of(t):std::string::npos;};
            size_t open_range_index_of(utctime t, size_t ix_hint = std::string::npos) const {
                return ts ? ts->time_axis().open_range_index_of(t, ix_hint):std::string::npos; }
            size_t size() const {return ts?ts->size():0;};        ///< number of points that descr. y=f(t) on t ::= period
            utctime time(size_t i) const {return sts()->time(i);};///< get the i'th time point
            double value(size_t i) const {return sts()->value(i);};///< get the i'th value
            double operator()(utctime t) const  {return sts()->value_at(t);};
            std::vector<double> values() const {return ts?ts->values():std::vector<double>();}

            //-- then some useful functions/properties
            apoint_ts extend( const apoint_ts & ts,
                extend_ts_split_policy split_policy, extend_ts_fill_policy fill_policy,
                utctime split_at, double fill_value ) const;
            apoint_ts average(const gta_t& ta) const;
            apoint_ts integral(gta_t const &ta) const;
            apoint_ts accumulate(const gta_t& ta) const;
            apoint_ts time_shift(utctimespan dt) const;
            apoint_ts max(double a) const;
            apoint_ts min(double a) const;
            apoint_ts max(const apoint_ts& other) const;
            apoint_ts min(const apoint_ts& other) const;
            static apoint_ts max(const apoint_ts& a, const apoint_ts& b);
            static apoint_ts min(const apoint_ts& a, const apoint_ts& b);
            ats_vector partition_by(const calendar& cal, utctime t, utctimespan partition_interval, size_t n_partitions, utctime common_t0) const;
            apoint_ts convolve_w(const std::vector<double>& w, shyft::time_series::convolve_policy conv_policy) const;
            apoint_ts abs() const;
			apoint_ts rating_curve(const rating_curve_parameters & rc_param) const;

            //-- in case the underlying ipoint_ts is a gpoint_ts (concrete points)
            //   we would like these to be working (exception if it's not possible,i.e. an expression)
            point get(size_t i) const {return point(time(i),value(i));}
            void set(size_t i, double x) ;
            void fill(double x) ;
            void scale_by(double x) ;

            /** given that this ts is a bind-able ts (aref_ts)
             * and that bts is a gpoint_ts, make
             * a *copy* of gpoint_ts and use it as representation
             * for the values of this ts
             * \parameter bts time-series of type point that will be applied to this ts.
             * \throw runtime_error if any of preconditions is not true.
             */
            void bind(const apoint_ts& bts);

            /** recursive search through the expression that this ts represents,
             *  and return a list of bind_ts_info that can be used to
             *  inspect and possibly 'bind' to values \ref bind.
             * \return a vector of ts_bind_info
             */
            std::vector<ts_bind_info> find_ts_bind_info() const;

            std::string serialize() const;
            static apoint_ts deserialize(const std::string&ss);
            std::vector<char> serialize_to_bytes() const;
            static apoint_ts deserialize_from_bytes(const std::vector<char>&ss);
            x_serialize_decl();
        };

        /** ts_bind_info gives information about the timeseries and it's binding
        * represented by encoded string reference
        * Given that you have a concrete ts,
        * you can bind that the bind_info.ts
        * using bind_info.ts.bind().
        */
        struct ts_bind_info {
            ts_bind_info(const std::string& id, const apoint_ts&ts) :reference(id), ts(ts) {}
            ts_bind_info() {}
            bool operator==(const ts_bind_info& o) const { return reference == o.reference; }
            std::string reference;
            apoint_ts ts;
        };

        /** \brief gpoint_ts a generic concrete point_ts, a terminal, not an expression
         *
         * The gpoint_ts is typically provided by repositories, that reads time-series
         * from some provider, like database, file(netcdf etc), providing a set of values
         * that are aligned to the specified time-axis.
         *
         */
        struct gpoint_ts:ipoint_ts {
            gts_t rep;
            // To create gpoint_ts, we use const ref, move ct wherever possible:
            explicit gpoint_ts(gts_t&&x):rep(std::move(x)){};
            explicit gpoint_ts(const gts_t&x):rep(x){}
            // note (we would normally use ct template here, but we are aiming at exposing to python)
            gpoint_ts(const gta_t&ta,double fill_value,ts_point_fx point_fx=POINT_INSTANT_VALUE):rep(ta,fill_value,point_fx){}
            gpoint_ts(const gta_t&ta,const std::vector<double>& v,ts_point_fx point_fx=POINT_INSTANT_VALUE):rep(ta,v,point_fx) {}
            gpoint_ts(gta_t&&ta,double fill_value,ts_point_fx point_fx=POINT_INSTANT_VALUE):rep(std::move(ta),fill_value,point_fx){}
            gpoint_ts(gta_t&&ta,std::vector<double>&& v,ts_point_fx point_fx=POINT_INSTANT_VALUE):rep(std::move(ta),std::move(v),point_fx) {}
            gpoint_ts(const gta_t& ta,std::vector<double>&& v,ts_point_fx point_fx=POINT_INSTANT_VALUE):rep(ta,std::move(v),point_fx) {}

            // now for the gpoint_ts it self, constructors incl. move
            gpoint_ts() = default; // default for serialization conv
            // implement ipoint_ts contract:
            virtual ts_point_fx point_interpretation() const {return rep.point_interpretation();}
            virtual void set_point_interpretation(ts_point_fx point_interpretation) {rep.set_point_interpretation(point_interpretation);}
            virtual const gta_t& time_axis() const {return rep.time_axis();}
            virtual utcperiod total_period() const {return rep.total_period();}
            virtual size_t index_of(utctime t) const {return rep.index_of(t);}
            virtual size_t size() const {return rep.size();}
            virtual utctime time(size_t i) const {return rep.time(i);};
            virtual double value(size_t i) const {return rep.v[i];}
            virtual double value_at(utctime t) const {return rep(t);}
            virtual std::vector<double> values() const {return rep.v;}
            // implement some extra functions to manipulate the points
            void set(size_t i, double x) {rep.set(i,x);}
            void fill(double x) {rep.fill(x);}
            void scale_by(double x) {rep.scale_by(x);}
            virtual bool needs_bind() const { return false;}
            virtual void do_bind()  {}
            x_serialize_decl();
        };

        struct aref_ts:ipoint_ts {
            typedef shyft::time_series::ref_ts<gts_t> ref_ts_t;
            ref_ts_t rep;
            explicit aref_ts(const string& sym_ref):rep(sym_ref) {}
            aref_ts() = default; // default for serialization conv
            // implement ipoint_ts contract:
            virtual ts_point_fx point_interpretation() const {return rep.point_interpretation();}
            virtual void set_point_interpretation(ts_point_fx point_interpretation) {rep.set_point_interpretation(point_interpretation);}
            virtual const gta_t& time_axis() const {return rep.time_axis();}
            virtual utcperiod total_period() const {return rep.total_period();}
            virtual size_t index_of(utctime t) const {return rep.index_of(t);}
            virtual size_t size() const {return rep.size();}
            virtual utctime time(size_t i) const {return rep.time(i);};
            virtual double value(size_t i) const {return rep.value(i);}
            virtual double value_at(utctime t) const {return rep(t);}
            virtual std::vector<double> values() const {return rep.bts().v;}
            // implement some extra functions to manipulate the points
            void set(size_t i, double x) {rep.set(i,x);}
            void fill(double x) {rep.fill(x);}
            void scale_by(double x) {rep.scale_by(x);}
            virtual bool needs_bind() const { return shyft::time_series::e_needs_bind(rep);}
            virtual void do_bind()  {}
            x_serialize_decl();
       };

        /** \brief The average_ts is used for providing ts average values over a time-axis
         *
         * Given a any ts, concrete, or an expression, provide the true average values on the
         * intervals as provided by the specified time-axis.
         *
         * true average for each period in the time-axis is defined as:
         *
         *   integral of f(t) dt from t0 to t1 / (t1-t0)
         *
         * using the f(t) interpretation of the supplied ts (linear or stair case).
         *
         * The \ref ts_point_fx is always POINT_AVERAGE_VALUE for the result ts.
         *
         * \note if a nan-value intervals are excluded from the integral and time-computations.
         *       E.g. let's say half the interval is nan, then the true average is computed for
         *       the other half of the interval.
         *
         */
        struct average_ts:ipoint_ts {
            gta_t ta;
            std::shared_ptr<ipoint_ts> ts;
            // useful constructors
            average_ts(gta_t&& ta,const apoint_ts& ats):ta(std::move(ta)),ts(ats.ts) {}
            average_ts(gta_t&& ta,apoint_ts&& ats):ta(std::move(ta)),ts(std::move(ats.ts)) {}
            average_ts(const gta_t& ta,apoint_ts&& ats):ta(ta),ts(std::move(ats.ts)) {}
            average_ts(const gta_t& ta,const apoint_ts& ats):ta(ta),ts(ats.ts) {}
            average_ts(const gta_t& ta,const std::shared_ptr<ipoint_ts> &ts ):ta(ta),ts(ts){}
            average_ts(gta_t&& ta,const std::shared_ptr<ipoint_ts> &ts ):ta(std::move(ta)),ts(ts){}
            // std copy ct and assign
            average_ts()=default;
            // implement ipoint_ts contract:
            virtual ts_point_fx point_interpretation() const {return ts_point_fx::POINT_AVERAGE_VALUE;}
            virtual void set_point_interpretation(ts_point_fx point_interpretation) {;}
            virtual const gta_t& time_axis() const {return ta;}
            virtual utcperiod total_period() const {return ta.total_period();}
            virtual size_t index_of(utctime t) const {return ta.index_of(t);}
            virtual size_t size() const {return ta.size();}
            virtual utctime time(size_t i) const {return ta.time(i);};
            virtual double value(size_t i) const {
                #ifdef _DEBUG
                if(i>ta.size())
                    return nan;
                #endif
                size_t ix_hint=(i*ts->size())/ta.size();// assume almost fixed delta-t.
                return average_value(*ts,ta.period(i),ix_hint,ts->point_interpretation() == ts_point_fx::POINT_INSTANT_VALUE);
            }
            virtual double value_at(utctime t) const {
                // return true average at t
                if(!ta.total_period().contains(t))
                    return nan;
                return value(index_of(t));
            }
			virtual std::vector<double> values() const;
            virtual bool needs_bind() const { return ts->needs_bind();}
            virtual void do_bind() {ts->do_bind();}
            x_serialize_decl();

        };

        /** \brief The integral_ts is used for providing ts integral values over a time-axis
        *
        * Given a any ts, concrete, or an expression, provide the 'true integral' values on the
        * intervals as provided by the specified time-axis.
        *
        * true inegral for each period in the time-axis is defined as:
        *
        *   integral of f(t) dt from t0 to t1
        *
        * using the f(t) interpretation of the supplied ts (linear or stair case).
        *
        * The \ref ts_point_fx is always POINT_AVERAGE_VALUE for the result ts.
        *
        * \note if a nan-value intervals are excluded from the integral and time-computations.
        *       E.g. let's say half the interval is nan, then the true integral is computed for
        *       the other half of the interval.
        *
        */
        struct integral_ts :ipoint_ts {
            gta_t ta;
            std::shared_ptr<ipoint_ts> ts;
            // useful constructors
            integral_ts(gta_t&& ta, const apoint_ts& ats) :ta(std::move(ta)), ts(ats.ts) {}
            integral_ts(gta_t&& ta, apoint_ts&& ats) :ta(std::move(ta)), ts(std::move(ats.ts)) {}
            integral_ts(const gta_t& ta, apoint_ts&& ats) :ta(ta), ts(std::move(ats.ts)) {}
            integral_ts(const gta_t& ta, const apoint_ts& ats) :ta(ta), ts(ats.ts) {}
            integral_ts(const gta_t& ta, const std::shared_ptr<ipoint_ts> &ts) :ta(ta), ts(ts) {}
            integral_ts(gta_t&& ta, const std::shared_ptr<ipoint_ts> &ts) :ta(std::move(ta)), ts(ts) {}
            // std copy ct and assign
            integral_ts()=default;
            // implement ipoint_ts contract:
            virtual ts_point_fx point_interpretation() const { return ts_point_fx::POINT_AVERAGE_VALUE; }
            virtual void set_point_interpretation(ts_point_fx point_interpretation) { ; }
            virtual const gta_t& time_axis() const { return ta; }
            virtual utcperiod total_period() const { return ta.total_period(); }
            virtual size_t index_of(utctime t) const { return ta.index_of(t); }
            virtual size_t size() const { return ta.size(); }
            virtual utctime time(size_t i) const { return ta.time(i); };
            virtual double value(size_t i) const {
                if (i>ta.size())
                    return nan;
                size_t ix_hint = (i*ts->size()) / ta.size();// assume almost fixed delta-t.
                utctimespan tsum = 0;
                return accumulate_value(*ts, ta.period(i), ix_hint,tsum, ts->point_interpretation() == ts_point_fx::POINT_INSTANT_VALUE);
            }
            virtual double value_at(utctime t) const {
                // return true average at t
                if (!ta.total_period().contains(t))
                    return nan;
                return value(index_of(t));
            }
            virtual std::vector<double> values() const ;
            virtual bool needs_bind() const { return ts->needs_bind();}
            virtual void do_bind() {ts->do_bind();}
            x_serialize_decl();

        };

        /** \brief The accumulate_ts is used for providing accumulated(integrated) ts values over a time-axis
        *
        * Given a any ts, concrete, or an expression, provide the true accumulated values,
        * defined as area under non-nan values of the f(t) curve,
        * on the intervals points as provided by the specified time-axis.
        *
        * The value at the i'th point of the time-axis is given by:
        *
        *   integral of f(t) dt from t0 to ti ,
        *
        *   where t0 is time_axis.period(0).start, and ti=time_axis.period(i).start
        *
        * using the f(t) interpretation of the supplied ts (linear or stair case).
        *
        * \note The value at t=t0 is 0.0 (by definition)
        * \note The value of t outside ta.total_period() is nan
        *
        * The \ref ts_point_fx is always POINT_INSTANT_VALUE for the result ts.
        *
        * \note if a nan-value intervals are excluded from the integral and time-computations.
        *       E.g. let's say half the interval is nan, then the true average is computed for
        *       the other half of the interval.
        *
        */
        struct accumulate_ts :ipoint_ts {
            gta_t ta;
            std::shared_ptr<ipoint_ts> ts;
            // useful constructors
            accumulate_ts(gta_t&& ta, const apoint_ts& ats) :ta(std::move(ta)), ts(ats.ts) {}
            accumulate_ts(gta_t&& ta, apoint_ts&& ats) :ta(std::move(ta)), ts(std::move(ats.ts)) {}
            accumulate_ts(const gta_t& ta, apoint_ts&& ats) :ta(ta), ts(std::move(ats.ts)) {}
            accumulate_ts(const gta_t& ta, const apoint_ts& ats) :ta(ta), ts(ats.ts) {}
            accumulate_ts(const gta_t& ta, const std::shared_ptr<ipoint_ts> &ts) :ta(ta), ts(ts) {}
            accumulate_ts(gta_t&& ta, const std::shared_ptr<ipoint_ts> &ts) :ta(std::move(ta)), ts(ts) {}
            // std copy ct and assign
            accumulate_ts()=default;
            // implement ipoint_ts contract:
            virtual ts_point_fx point_interpretation() const { return ts_point_fx::POINT_INSTANT_VALUE; }
            virtual void set_point_interpretation(ts_point_fx point_interpretation) { ; }// we could throw here..
            virtual const gta_t& time_axis() const { return ta; }
            virtual utcperiod total_period() const { return ta.total_period(); }
            virtual size_t index_of(utctime t) const { return ta.index_of(t); }
            virtual size_t size() const { return ta.size(); }
            virtual utctime time(size_t i) const { return ta.time(i); };
            virtual double value(size_t i) const {
                if (i>ta.size())
                    return nan;
                if (i == 0)// by definition,0.0 at i=0
                    return 0.0;
                size_t ix_hint = 0;// assume almost fixed delta-t.
                utctimespan tsum;
                return accumulate_value(*ts, utcperiod(ta.time(0), ta.time(i)), ix_hint, tsum, ts->point_interpretation() == ts_point_fx::POINT_INSTANT_VALUE);
            }
            virtual double value_at(utctime t) const {
                // return true accumulated value at t
                if (!ta.total_period().contains(t))
                    return nan;
                if (t == ta.time(0))
                    return 0.0; // by definition
                utctimespan tsum;
                size_t ix_hint = 0;
                return accumulate_value(*this, utcperiod(ta.time(0), t), ix_hint, tsum, ts->point_interpretation() == ts_point_fx::POINT_INSTANT_VALUE);// also note: average of non-nan areas !;
            }
            virtual std::vector<double> values() const {
                std::vector<double> r;r.reserve(ta.size());
                accumulate_accessor<ipoint_ts, gta_t> accumulate(*ts, ta);// use accessor, that
                for (size_t i = 0;i<ta.size();++i) {                      // given sequential access
                    r.push_back(accumulate.value(i));                     // reuses acc.computation
                }
                return r;
            }
            virtual bool needs_bind() const { return ts->needs_bind();}
            virtual void do_bind() {ts->do_bind();}
            // to help the average function, return the i'th point of the underlying timeseries
            //point get(size_t i) const {return point(ts->time(i),ts->value(i));}
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
         * \note If the ts given at constructor time is an unbound ts or expression,
         *       then .do_bind() needs to be called before any call to
         *       value or time-axis function calls.
         *
         */
        struct time_shift_ts:ipoint_ts {
            std::shared_ptr<ipoint_ts> ts;
            gta_t ta;
            utctimespan dt=0;// despite ta time-axis, we need it

            time_shift_ts()=default;

            //-- useful ct goes here
            time_shift_ts(const apoint_ts& ats,utctimespan adt)
                :ts(ats.ts),dt(adt) {
                if(!ts->needs_bind())
                    local_do_bind();

            }
            time_shift_ts(apoint_ts&& ats, utctimespan adt)
                :ts(std::move(ats.ts)),dt(adt) {
                if(!ts->needs_bind())
					local_do_bind();
            }
            time_shift_ts(const std::shared_ptr<ipoint_ts> &ts, utctime adt )
                :ts(ts),dt(adt){
                if(!ts->needs_bind())
					local_do_bind();
            }
            void local_do_bind() {
                if(ta.size()==0) {//TODO: introduce bound flag, and use that, using the ta.size() is a problem if ta *is* size 0.
                    ta= time_axis::time_shift(ts->time_axis(),dt);
                }
            }
            // implement ipoint_ts contract:
            virtual ts_point_fx point_interpretation() const {return ts->point_interpretation();}
            virtual void set_point_interpretation(ts_point_fx point_interpretation) {ts->set_point_interpretation(point_interpretation);}
            virtual const gta_t& time_axis() const {return ta;}
            virtual utcperiod total_period() const {return ta.total_period();}
            virtual size_t index_of(utctime t) const {return ta.index_of(t);}
            virtual size_t size() const {return ta.size();}
            virtual utctime time(size_t i) const {return ta.time(i);};
            virtual double value(size_t i) const {return ts->value(i);}
            virtual double value_at(utctime t) const {return ts->value_at(t-dt);}
            virtual std::vector<double> values() const {return ts->values();}
            virtual bool needs_bind() const { return ts->needs_bind();}
            virtual void do_bind() {ts->do_bind();local_do_bind();}
            x_serialize_decl();

        };

        /** \brief abs_ts as  abs(ts)
        *
        * The time-axis as source, values are abs of source
        *
        *
        */
        struct abs_ts :ipoint_ts {
            std::shared_ptr<ipoint_ts> ts;
            gta_t ta;

            abs_ts() = default;

            //-- useful ct goes here
            explicit abs_ts(const apoint_ts& ats)
                :ts(ats.ts) {
                if (!ts->needs_bind())
					local_do_bind();

            }
            explicit abs_ts(apoint_ts&& ats)
                :ts(std::move(ats.ts)){
                if (!ts->needs_bind())
					local_do_bind();
            }
            explicit abs_ts(const std::shared_ptr<ipoint_ts> &ts)
                :ts(ts) {
                if (!ts->needs_bind())
					local_do_bind();
            }
            void local_do_bind() {
                if (ta.size() == 0) {//TODO: introduce bound flag, and use that, using the ta.size() is a problem if ta *is* size 0.
                    ta = ts->time_axis();
                }
            }
            // implement ipoint_ts contract:
            virtual ts_point_fx point_interpretation() const { return ts->point_interpretation(); }
            virtual void set_point_interpretation(ts_point_fx point_interpretation) { ts->set_point_interpretation(point_interpretation); }
            virtual const gta_t& time_axis() const { return ta; }
            virtual utcperiod total_period() const { return ta.total_period(); }
            virtual size_t index_of(utctime t) const { return ts->index_of(t); }
            virtual size_t size() const { return ta.size(); }
            virtual utctime time(size_t i) const { return ta.time(i); };
            virtual double value(size_t i) const { return abs(ts->value(i)); }
            virtual double value_at(utctime t) const { return abs(ts->value_at(t)); }
            virtual std::vector<double> values() const {
                auto vv=ts->values();
                for (auto &v : vv) v = abs(v);
                return vv;
            }
            virtual bool needs_bind() const { return ts->needs_bind(); }
            virtual void do_bind() { ts->do_bind(); local_do_bind(); }
            x_serialize_decl();

        };

        /** \brief periodic_ts is used for providing ts periodic values over a time-axis
        *
        */
        struct periodic_ts : ipoint_ts {
            typedef shyft::time_series::periodic_ts<gta_t> pts_t;
            pts_t ts;

            periodic_ts(const vector<double>& pattern, utctimespan dt, const gta_t& ta) : ts(pattern, dt, ta) {}
            periodic_ts(const vector<double>& pattern, utctimespan dt, utctime pattern_t0,const gta_t& ta) : ts(pattern, dt,pattern_t0,ta) {}
            periodic_ts(const periodic_ts& c) : ts(c.ts) {}
            periodic_ts(periodic_ts&& c) : ts(move(c.ts)) {}
            periodic_ts& operator=(const periodic_ts& c) {
                if (this != &c) {
                    ts = c.ts;
                }
                return *this;
            }
            periodic_ts& operator=(periodic_ts&& c) {
                ts = move(c.ts);
                return *this;
            }
            periodic_ts()=default;
            // implement ipoint_ts contract
            virtual ts_point_fx point_interpretation() const { return ts_point_fx::POINT_AVERAGE_VALUE; }
            virtual void set_point_interpretation(ts_point_fx) { ; }
            virtual const gta_t& time_axis() const { return ts.ta; }
            virtual utcperiod total_period() const { return ts.ta.total_period(); }
            virtual size_t index_of(utctime t) const { return ts.index_of(t); }
            virtual size_t size() const { return ts.ta.size(); }
            virtual utctime time(size_t i) const { return ts.ta.time(i); }
            virtual double value(size_t i) const { return ts.value(i); }
            virtual double value_at(utctime t) const { return value(index_of(t)); }
            virtual vector<double> values() const { return ts.values(); }
            virtual bool needs_bind() const { return false;}// this is a terminal node, no bind needed
            virtual void do_bind()  {}
            x_serialize_decl();
        };

        /** \brief convolve_w is used for providing a convolution by weights ts
        *
        * The convolve_w_ts is particularly useful for implementing routing and model
        * time-delays and shape-of hydro-response.
        *
        */
        struct convolve_w_ts : ipoint_ts {
            typedef vector<double> weights_t;
            typedef shyft::time_series::convolve_w_ts<apoint_ts> cnv_ts_t;
            cnv_ts_t ts_impl;

            convolve_w_ts(const apoint_ts& ats, const weights_t& w, convolve_policy conv_policy) :ts_impl(ats, w, conv_policy) {}
            convolve_w_ts(apoint_ts&& ats, const weights_t& w, convolve_policy conv_policy) :ts_impl(move(ats), w, conv_policy) {}
            // hmm: convolve_w_ts(const std::shared_ptr<ipoint_ts> &ats,const weights_t& w,convolve_policy conv_policy ):ts(ats),ts_impl(*ts,w,conv_policy) {}

            // std.ct
            convolve_w_ts() =default;

            // implement ipoint_ts contract
            virtual ts_point_fx point_interpretation() const { return ts_impl.point_interpretation(); }
            virtual void set_point_interpretation(ts_point_fx) { throw std::runtime_error("not implemented"); }
            virtual const gta_t& time_axis() const { return ts_impl.time_axis(); }
            virtual utcperiod total_period() const { return ts_impl.total_period(); }
            virtual size_t index_of(utctime t) const { return ts_impl.index_of(t); }
            virtual size_t size() const { return ts_impl.size(); }
            virtual utctime time(size_t i) const { return ts_impl.time(i); }
            virtual double value(size_t i) const { return ts_impl.value(i); }
            virtual double value_at(utctime t) const { return value(index_of(t)); }
            virtual vector<double> values() const {
                vector<double> r;r.reserve(size());
                for (size_t i = 0;i<size();++i)
                    r.push_back(ts_impl.value(i));
                return r;
            }
            virtual bool needs_bind() const { return ts_impl.needs_bind();}
            virtual void do_bind() {ts_impl.do_bind();}
            x_serialize_decl();
        };

        /** \brief Extend for ts.extend(ts).
         */
        struct extend_ts : ipoint_ts {

            apoint_ts lhs;
            apoint_ts rhs;
            extend_ts_split_policy ets_split_p = EPS_LHS_LAST;
            utctime split_at;
            extend_ts_fill_policy ets_fill_p = EPF_NAN;
            double fill_value;

            gta_t ta;
            ts_point_fx fx_policy = POINT_AVERAGE_VALUE;  // how f(t) are mapped to t

            bool bound = false;

            ts_point_fx point_interpretation() const {
                return fx_policy;
            }

            void set_point_interpretation(ts_point_fx x) {
                fx_policy = x;
            }

            void local_do_bind() {
                if (!bound) {
                    fx_policy = result_policy(lhs.point_interpretation(), rhs.point_interpretation());
                    ta = time_axis::extend(lhs.time_axis(), rhs.time_axis(), get_split_at());
                    bound = true;
                }
            }

            extend_ts() = default;
            extend_ts(
                const apoint_ts & lhs, const apoint_ts & rhs,
                extend_ts_split_policy split_policy, extend_ts_fill_policy fill_policy,
                utctime split_at, double fill_value
            )
                : lhs( lhs ), rhs( rhs ),
                  ets_split_p(split_policy ), split_at( split_at ),
                  ets_fill_p(fill_policy ), fill_value( fill_value ) {
                if (!needs_bind())
					local_do_bind();
            }

            utctime get_split_at() const {
                switch ( this->ets_split_p ) {
                default:
                case EPS_LHS_LAST:  return this->lhs.total_period().end;
                case EPS_RHS_FIRST: return this->rhs.total_period().start;
                case EPS_VALUE:     return this->split_at;
                }
            }

            void bind_check() const {
                if (!bound)
                    throw runtime_error("attempting to use unbound timeseries, context abin_op_ts");
            }
            virtual utcperiod total_period() const {
                return time_axis().total_period();
            }
            const gta_t& time_axis() const {
                bind_check();
                return ta;
            };// combine lhs,rhs

            size_t index_of(utctime t) const {
                return time_axis().index_of(t);
            };

            size_t size() const {
                return time_axis().size();
            };// use the combined ta.size();

            utctime time(size_t i) const {
                return time_axis().time(i);
            }; // return combined ta.time(i)

            /** Get ta value at time. */
            double value_at(utctime t) const;

            /** Get ts value at point no. */
            double value(size_t i) const;

            /** Collect all values for the extended ts. */
            std::vector<double> values() const;

            bool needs_bind() const {
                return lhs.needs_bind() || rhs.needs_bind();
            }
            virtual void do_bind() {
                lhs.do_bind();
                rhs.do_bind();
				local_do_bind();
            }

            x_serialize_decl();

        };

		struct rating_curve_ts : ipoint_ts {

			using rct_t = shyft::time_series::rating_curve_ts<apoint_ts>;
			using rc_param_t = shyft::time_series::rating_curve_parameters;

			rct_t ts;

			rating_curve_ts() = default;
			rating_curve_ts(apoint_ts && ts, rc_param_t && rcp)
				: ts{ std::forward<apoint_ts>(ts), std::forward<rc_param_t>(rcp) } { }
			rating_curve_ts(const apoint_ts & ts, const rc_param_t & rcp)
				: ts{ ts, rcp } { }
			// -----
			virtual ~rating_curve_ts() = default;
			// -----
			rating_curve_ts(const rating_curve_ts &) = default;
			rating_curve_ts & operator= (const rating_curve_ts &) = default;
			// -----
			rating_curve_ts(rating_curve_ts &&) = default;
			rating_curve_ts & operator= (rating_curve_ts &&) = default;

			virtual bool needs_bind() const { return ts.needs_bind(); }
			virtual void do_bind() { ts.do_bind(); }
			// -----
			virtual ts_point_fx point_interpretation() const { return ts.point_interpretation(); }
			virtual void set_point_interpretation(ts_point_fx policy) { return ts.set_point_interpretation(policy); }
			// -----
			virtual std::size_t size() const { return ts.size(); }
			virtual utcperiod total_period() const { return ts.total_period(); }
			virtual const gta_t & time_axis() const { return ts.time_axis(); }
			// -----
			virtual std::size_t index_of(utctime t) const { return ts.index_of(t); }
			virtual double value_at(utctime t) const { return ts(t); }
			// -----
			virtual utctime time(std::size_t i) const { return ts.time(i); }
			virtual double value(std::size_t i) const { return ts.value(i); }
			// -----
			virtual std::vector<double> values() const {
				std::size_t dim = size();
				std::vector<double> ret; ret.reserve(dim);
				for ( std::size_t i = 0u; i < dim; ++i ) { ret.emplace_back(value(i)); }
				return ret;
			}

			x_serialize_decl();

		};

        /** The iop_t represent the basic 'binary' operation,
         *   a stateless function that takes two doubles and returns the binary operation.
         *   E.g.: a+b
         *   The iop_t is used as the operation element of the abin_op_ts class
         */
        enum iop_t {
            OP_NONE,OP_ADD,OP_SUB,OP_DIV,OP_MUL,OP_MIN,OP_MAX
        };

        /** do_bind helps to defer the computation cost of the
         * expression bin-op variants until its actually used.
         * this is also needed when having ts_refs| unbound symbolic time-series references
         * that we would like to serialize and pass over to another server for execution.
         *
         * By inspecting the time-series in the construction phase (bin_op etc.)
         * we try to take the preparation for computation as early as possible,
         * so, only when there is a symbolic time-series reference, there will be
         * a do_bind() that will take action *after* the
         * symbolic time-series has been prepared with real values (bound).
         *
         */

        /** \brief The binary operation for type ts op ts
         *
         * The binary operation is lazy, and only keep the reference to the two operands
         * that are of the \ref apoint_ts type.
         * The operation is of \ref iop_t, and details for plus minus divide multiply etc is in
         * the implementation file.
         *
         * As per definition a this class implements the \ref ipoint_ts interface,
         * and the time-axis of type \ref gta_t is currently computed in the constructor.
         * This could take some cpu if the time-axis is of type point_dt, so we could
         * consider working some more on the internal algorithms to avoid this.
         *
         * The \ref ts_point_fx is computed based on rhs,lhs. But can be overridden
         * by the user.
         *
         */
        struct abin_op_ts:ipoint_ts {

            apoint_ts lhs;
            iop_t op=iop_t::OP_NONE;
            apoint_ts rhs;
            gta_t ta;
            ts_point_fx fx_policy=POINT_AVERAGE_VALUE;
            bool bound=false;

            ts_point_fx point_interpretation() const {return fx_policy;}
            void set_point_interpretation(ts_point_fx x) {fx_policy=x;}

            void local_do_bind() {
                if(!bound) {
                    fx_policy=result_policy(lhs.point_interpretation(),rhs.point_interpretation());
                    ta=time_axis::combine(lhs.time_axis(),rhs.time_axis());
                    bound=true;
                }
            }

            abin_op_ts()=default;
            abin_op_ts(const apoint_ts &lhs,iop_t op,const apoint_ts& rhs)
                :lhs(lhs),op(op),rhs(rhs) {
                if( !needs_bind() )
					local_do_bind();
            }
            void bind_check() const {if(!bound) throw runtime_error("attempting to use unbound timeseries, context abin_op_ts");}
            virtual utcperiod total_period() const {return time_axis().total_period();}
            const gta_t& time_axis() const {bind_check(); return ta;};// combine lhs,rhs
            size_t index_of(utctime t) const{return time_axis().index_of(t);};
            size_t size() const {return time_axis().size();};// use the combined ta.size();
            utctime time( size_t i) const {return time_axis().time(i);}; // return combined ta.time(i)
            double value_at(utctime t) const ;
            double value(size_t i) const;// return op( lhs(t), rhs(t)) ..
            std::vector<double> values() const;
            bool needs_bind() const {return lhs.needs_bind() || rhs.needs_bind(); }
            virtual void do_bind() {lhs.do_bind();rhs.do_bind();local_do_bind();}
            x_serialize_decl();

        };

        /** \brief  binary operation for type ts op double
         *
         * The resulting time-axis and point interpretation policy is equal to the ts.
         */
        struct abin_op_scalar_ts:ipoint_ts {

              double lhs;
              iop_t op=iop_t::OP_NONE;
              apoint_ts rhs;
              gta_t ta;
              ts_point_fx fx_policy=POINT_AVERAGE_VALUE;
              bool bound=false;


              ts_point_fx point_interpretation() const {return fx_policy;}
              void set_point_interpretation(ts_point_fx x) {fx_policy=x;}

              void local_do_bind()  {
                  if(!bound) {
                      ta=rhs.time_axis();
                      fx_policy= rhs.point_interpretation();
                      bound=true;
                  }
              }
              void bind_check() const {if(!bound) throw runtime_error("attempting to use unbound timeseries, context abin_op_scalar");}
              abin_op_scalar_ts()=default;

              abin_op_scalar_ts(double lhs,iop_t op,const apoint_ts& rhs)
              :lhs(lhs),op(op),rhs(rhs) {
                  if(!needs_bind())
					  local_do_bind();
              }

              virtual utcperiod total_period() const {return time_axis().total_period();}
              const gta_t& time_axis() const {bind_check();return ta;};// combine lhs,rhs
              size_t index_of(utctime t) const{return time_axis().index_of(t);};
              size_t size() const {return time_axis().size();};
              utctime time( size_t i) const {return time_axis().time(i);};
              double value_at(utctime t) const ;
              double value(size_t i) const ;
              std::vector<double> values() const ;
              bool needs_bind() const {return rhs.needs_bind(); }
              virtual void do_bind() {rhs.do_bind();local_do_bind();}
              x_serialize_decl();
        };

        /** \brief  binary operation for type ts op double
         *
         * The resulting time-axis and point interpretation policy is equal to the ts.
         */
        struct abin_op_ts_scalar:ipoint_ts {
              apoint_ts lhs;
              iop_t op=iop_t::OP_NONE;
              double rhs;
              gta_t ta;
              bool bound=false;
              ts_point_fx fx_policy=POINT_AVERAGE_VALUE;
              ts_point_fx point_interpretation() const {return fx_policy;}
              void set_point_interpretation(ts_point_fx x) {fx_policy=x;}
              void local_do_bind()  {
                  if(!bound) {
                      ta=lhs.time_axis();
                      fx_policy= lhs.point_interpretation();
                      bound=true;
                  }
              }
              void bind_check() const {if(!bound) throw runtime_error("attempting to use unbound timeseries, context abin_op_ts_scalar");}
              abin_op_ts_scalar()=default;

              abin_op_ts_scalar(const apoint_ts &lhs,iop_t op,double rhs)
              :lhs(lhs),op(op),rhs(rhs) {
                  if(!needs_bind())
					  local_do_bind();
              }

              virtual utcperiod total_period() const {return time_axis().total_period();}
              const gta_t& time_axis() const {bind_check();return ta;};
              size_t index_of(utctime t) const{return time_axis().index_of(t);};
              size_t size() const {return time_axis().size();};
              utctime time( size_t i) const {return time_axis().time(i);};
              double value_at(utctime t) const;
              double value(size_t i) const;
              std::vector<double> values() const;
              bool needs_bind() const {return lhs.needs_bind(); }
              virtual void do_bind() {lhs.do_bind();local_do_bind();}
              x_serialize_decl();

        };

        // add operators and functions to the apoint_ts class, of all variants that we want to expose
        apoint_ts average(const apoint_ts& ts,const gta_t& ta/*fx-type */) ;
        apoint_ts average(apoint_ts&& ts,const gta_t& ta) ;

        apoint_ts integral(const apoint_ts& ts, const gta_t& ta/*fx-type */);
        apoint_ts integral(apoint_ts&& ts, const gta_t& ta);

        apoint_ts accumulate(const apoint_ts& ts, const gta_t& ta/*fx-type */);
        apoint_ts accumulate(apoint_ts&& ts, const gta_t& ta);

        apoint_ts create_glacier_melt_ts_m3s(const apoint_ts & temp,const apoint_ts& sca_m2,double glacier_area_m2,double dtf);

        double nash_sutcliffe(const apoint_ts& observation_ts, const apoint_ts& model_ts, const gta_t &ta);

        double kling_gupta(const apoint_ts& observation_ts, const apoint_ts&  model_ts, const gta_t& ta, double s_r, double s_a, double s_b);

        apoint_ts create_periodic_pattern_ts(const vector<double>& pattern, utctimespan dt,utctime t0, const gta_t& ta);

        apoint_ts operator+(const apoint_ts& lhs,const apoint_ts& rhs) ;
        apoint_ts operator+(const apoint_ts& lhs,double           rhs) ;
        apoint_ts operator+(double           lhs,const apoint_ts& rhs) ;

        apoint_ts operator-(const apoint_ts& lhs,const apoint_ts& rhs) ;
        apoint_ts operator-(const apoint_ts& lhs,double           rhs) ;
        apoint_ts operator-(double           lhs,const apoint_ts& rhs) ;
        apoint_ts operator-(const apoint_ts& rhs) ;

        apoint_ts operator/(const apoint_ts& lhs,const apoint_ts& rhs) ;
        apoint_ts operator/(const apoint_ts& lhs,double           rhs) ;
        apoint_ts operator/(double           lhs,const apoint_ts& rhs) ;

        apoint_ts operator*(const apoint_ts& lhs,const apoint_ts& rhs) ;
        apoint_ts operator*(const apoint_ts& lhs,double           rhs) ;
        apoint_ts operator*(double           lhs,const apoint_ts& rhs) ;


        apoint_ts max(const apoint_ts& lhs,const apoint_ts& rhs) ;
        apoint_ts max(const apoint_ts& lhs,double           rhs) ;
        apoint_ts max(double           lhs,const apoint_ts& rhs) ;

        apoint_ts min(const apoint_ts& lhs,const apoint_ts& rhs) ;
        apoint_ts min(const apoint_ts& lhs,double           rhs) ;
        apoint_ts min(double           lhs,const apoint_ts& rhs) ;

        ///< percentiles, need to include several forms of time_axis for python
        std::vector<apoint_ts> percentiles(const std::vector<apoint_ts>& ts_list,const gta_t & ta,const vector<int>& percentiles);
        std::vector<apoint_ts> percentiles(const std::vector<apoint_ts>& ts_list,const time_axis::fixed_dt & ta,const vector<int>& percentiles);

        ///< time_shift i.e. same ts values, but time-axis is time-axis + dt
        apoint_ts time_shift(const apoint_ts &ts, utctimespan dt);

        apoint_ts extend(
            const apoint_ts & lhs_ts, const apoint_ts & rhs_ts,
            extend_ts_split_policy split_policy, extend_ts_fill_policy fill_policy,
            utctime split_at, double fill_value );

        /** Given a vector of expressions, deflate(evaluate) the expressions and return the
         * equivalent concrete point-time-series of the expressions in the
         * preferred destination type Ts
         * Useful for the dtss,
         * evaluates the expressions in parallell
         */
        template <class Ts,class TsV>
        std::vector<Ts>
        deflate_ts_vector(TsV const&tsv1) {
            std::vector<Ts> tsv2(tsv1.size());

            auto deflate_range=[&tsv1,&tsv2](size_t i0,size_t n) {
                for(size_t i=i0;i<i0+n;++i)
                    tsv2[i]= Ts(tsv1[i].time_axis(),tsv1[i].values(),tsv1[i].point_interpretation());
            };
            auto n_threads =  thread::hardware_concurrency();
            if(n_threads <2) n_threads=4;// hard coded minimum
            std::vector<std::future<void>> calcs;
            size_t ps= 1 + tsv1.size()/n_threads;
            for (size_t p = 0;p < tsv1.size(); ) {
                size_t np = p + ps <= tsv1.size() ? ps : tsv1.size() - p;
                calcs.push_back(std::async(std::launch::async, deflate_range, p, np));
                p += np;
            }
            for (auto &f : calcs) f.get();
            return tsv2;
        }

        /** \brief ats_vector represents a list of time-series, support math-operations.
         *
         * Supports handling and math operations for vectors of time-series.
         * Especially convinient in python due to compact notation and speed.
         */
        typedef vector<apoint_ts> ats_vec;
        struct ats_vector:ats_vec  {  // inheritance from vector, to get most parts for free
            // constructor stuff that needs to be complete for boost::python
            ats_vector()=default;
            ats_vector(ats_vec const&c):ats_vec(c) {}
            ats_vector(ats_vec&& c):ats_vec(std::move(c)) {}
            ats_vector(ats_vector const&c):ats_vec(c) {}
            explicit ats_vector(size_t sz):ats_vec(sz) {}
            ats_vector(ats_vector&&c):ats_vec(move(c)) {}
            ats_vector& operator=(ats_vector const&c) {
                if(this !=&c) {
                    ats_vec::operator=(c);
                }
                return *this;
            }
            ats_vector& operator=(ats_vector&&c) {
                ats_vec::operator=(c);
                return *this;
            }
            //-- minimal iterator support in order to expose it as vector
            ats_vector(ats_vec::iterator b,ats_vec::iterator e):ats_vec(b,e) {}
            auto begin() {return ats_vec::begin();}
            auto begin() const {return ats_vec::begin();}
            auto end() {return ats_vec::end();}
            auto end() const {return ats_vec::end();}
            void reserve(size_t x) {ats_vec::reserve(x);}
            apoint_ts& operator()(size_t i) {return *(begin()+i);}
            apoint_ts const & operator()(size_t i) const {return *(begin()+i);}

            vector<double> values_at_time(utctime t) const {
                std::vector<double> r;r.reserve(size());
                for (auto const &ts : *this ) r.push_back(ts(t));
                return r;
            }
            ats_vector percentiles(gta_t const &ta,vector<int> const& percentile_list) const {
                ats_vector r;r.reserve(percentile_list.size());
                auto rp= shyft::time_series::calculate_percentiles(ta,deflate_ts_vector<gts_t>(*this),percentile_list);
                for(auto&ts:rp) r.emplace_back(ta,std::move(ts.v),POINT_AVERAGE_VALUE);
                return r;
            }
            ats_vector percentiles_f(time_axis::fixed_dt const&ta,vector<int> const& percentile_list) const {
                return percentiles(gta_t(ta),percentile_list);
            }
            ats_vector slice(vector<int>const& slice_spec) const {
                if(slice_spec.size()==0) {
                    return ats_vector(*this);// just a clone of this
                } else {
                    ats_vector r;for(auto ix:slice_spec) r.push_back(begin()[ix]);
                    return r;
                }
            }

            ats_vector extend_ts(
                apoint_ts const & ta,
                extend_ts_split_policy split_policy, extend_ts_fill_policy fill_policy,
                utctime split_at, double fill_value
            ) const {
                ats_vector r; r.reserve(this->size());
                for ( auto const & ts : *this )
                    r.push_back(ts.extend(ta, split_policy, fill_policy, split_at, fill_value));
                return r;
            }
            ats_vector extend_vec(
                ats_vector const & ts_vec,
                extend_ts_split_policy split_policy, extend_ts_fill_policy fill_policy,
                utctime split_at, double fill_value
            ) const {
                if ( this->size() != ts_vec.size() ) throw std::runtime_error("vector size mismatch, must be of the same size");
                ats_vector r; r.reserve(this->size());
                auto lhs_it = this->cbegin(); auto rhs_it = ts_vec.cbegin();
                while ( lhs_it != this->cend() ) {
                    r.push_back(lhs_it->extend(*rhs_it, split_policy, fill_policy, split_at, fill_value));
                    lhs_it++; rhs_it++;
                }
                return r;
            }

            ats_vector abs() const {
                ats_vector r; r.reserve(size()); for (auto const &ts : *this) r.push_back(ts.abs()); return r;
            }
            ats_vector average(gta_t const&ta) const {
                ats_vector r;r.reserve(size());for(auto const &ts:*this) r.push_back(ts.average(ta)); return r;
            }
            ats_vector integral(gta_t const&ta) const {
                ats_vector r;r.reserve(size());for(auto const &ts:*this) r.push_back(ts.integral(ta)); return r;
            }
            ats_vector accumulate(gta_t const&ta) const {
                ats_vector r;r.reserve(size());for(auto const &ts:*this) r.push_back(ts.accumulate(ta)); return r;
            }
            ats_vector time_shift(utctimespan delta_t) const {
                ats_vector r;r.reserve(size());for(auto const &ts:*this) r.push_back(ts.time_shift(delta_t)); return r;
            }
            ats_vector min(double x) const {
                ats_vector r;r.reserve(size());for (auto const &ts : *this) r.push_back(ts.min(x)); return r;
            }
            ats_vector max(double x) const {
                ats_vector r;r.reserve(size());for (auto const &ts : *this) r.push_back(ts.max(x)); return r;
            }
            ats_vector min(apoint_ts const& x) const {
                ats_vector r;r.reserve(size());for (auto const &ts : *this) r.push_back(ts.min(x)); return r;
            }
            ats_vector max(apoint_ts const& x) const {
                ats_vector r;r.reserve(size());for (auto const &ts : *this) r.push_back(ts.max(x)); return r;
            }
            ats_vector min(ats_vector const& x) const;
            ats_vector max(ats_vector const& x) const;

            apoint_ts forecast_merge(utctimespan lead_time,utctimespan fc_interval) const;
            ats_vector average_slice(utctimespan t0_offset,utctimespan dt, int n) const ;
            double nash_sutcliffe(apoint_ts const &obs,utctimespan t0_offset,utctimespan dt, int n)const ;
            x_serialize_decl();
        };
        // quantile-mapping
        ats_vector quantile_map_forecast(vector<ats_vector> const & forecast_set,vector<double> const& set_weights,ats_vector const& historical_data,gta_t const&time_axis,utctime interpolation_start, utctime interpolation_end=no_utctime, bool interpolated_quantiles=false);
        // multiply operators
        ats_vector operator*(ats_vector const &a,double b);
        ats_vector operator*(double a,ats_vector const &b);
        ats_vector operator*(ats_vector const &a,ats_vector const& b);
        ats_vector operator*(ats_vector::value_type const &a,ats_vector const& b);
        ats_vector operator*(ats_vector const& b,ats_vector::value_type const &a);


        // divide operators
        ats_vector operator/(ats_vector const &a,double b);
        ats_vector operator/(double a,ats_vector const &b);
        ats_vector operator/(ats_vector const &a,ats_vector const& b);
        ats_vector operator/(ats_vector::value_type const &a,ats_vector const& b);
        ats_vector operator/(ats_vector const& b,ats_vector::value_type const &a);

        // add operators
        ats_vector operator+(ats_vector const &a,double b);
        ats_vector operator+(double a,ats_vector const &b);
        ats_vector operator+(ats_vector const &a,ats_vector const& b);
        ats_vector operator+(ats_vector::value_type const &a,ats_vector const& b);
        ats_vector operator+(ats_vector const& b,ats_vector::value_type const &a);

        // sub operators
        ats_vector operator-(const ats_vector& a);

        ats_vector operator-(ats_vector const &a,double b);
        ats_vector operator-(double a,ats_vector const &b);
        ats_vector operator-(ats_vector const &a,ats_vector const& b);
        ats_vector operator-(ats_vector::value_type const &a,ats_vector const& b);
        ats_vector operator-(ats_vector const& b,ats_vector::value_type const &a);

        // max-min func overloads (2x!)
        ats_vector min(ats_vector const &a, double b);
        ats_vector min(double b, ats_vector const &a);
        ats_vector min(ats_vector const &a, apoint_ts const& b);
        ats_vector min(apoint_ts const &b, ats_vector const& a);
        ats_vector min(ats_vector const &a, ats_vector const &b);

        ats_vector max(ats_vector const &a, double b);
        ats_vector max(double b, ats_vector const &a);
        ats_vector max(ats_vector const &a, apoint_ts const & b);
        ats_vector max(apoint_ts const &b, ats_vector const &a);
        ats_vector max(ats_vector const &a, ats_vector const & b);
    }
    namespace time_series {
        template<>
        inline size_t hint_based_search<api::apoint_ts>(const api::apoint_ts& source, const utcperiod& p, size_t i) {
            return source.open_range_index_of(p.start, i);
        }
		template<>
		inline size_t hint_based_search<api::ipoint_ts>(const api::ipoint_ts& source, const utcperiod& p, size_t i) {
			return source.time_axis().open_range_index_of(p.start, i);
		}
	}
}
//-- serialization support
x_serialize_export_key(shyft::api::ipoint_ts);
x_serialize_export_key(shyft::api::gpoint_ts);
x_serialize_export_key(shyft::api::average_ts);
x_serialize_export_key(shyft::api::integral_ts);
x_serialize_export_key(shyft::api::accumulate_ts);
x_serialize_export_key(shyft::api::time_shift_ts);
x_serialize_export_key(shyft::api::periodic_ts);
x_serialize_export_key(shyft::api::extend_ts);
x_serialize_export_key(shyft::api::abin_op_scalar_ts);
x_serialize_export_key(shyft::api::abin_op_ts);
x_serialize_export_key(shyft::api::abin_op_ts_scalar);
x_serialize_export_key(shyft::api::aref_ts);
x_serialize_export_key(shyft::time_series::convolve_w_ts<shyft::api::apoint_ts>); // oops need this from core
x_serialize_export_key(shyft::api::convolve_w_ts);
x_serialize_export_key(shyft::time_series::rating_curve_ts<shyft::api::apoint_ts>);
x_serialize_export_key(shyft::api::rating_curve_ts);
x_serialize_export_key(shyft::api::apoint_ts);
x_serialize_export_key(shyft::api::ats_vector);
x_serialize_export_key(shyft::api::abs_ts);
