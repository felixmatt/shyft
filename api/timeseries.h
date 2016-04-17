#pragma once

#include "core/core_pch.h"
#include "core/utctime_utilities.h"
#include "core/time_axis.h"
#include "core/timeseries.h"

namespace shyft {
    namespace api {
        using namespace shyft::core;
        using namespace shyft::timeseries;
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
            typedef shyft::timeseries::point_ts<gta_t> gts_t;

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
             * 3. The \ref point_interpretation_policy
             *    that determines how the points should be projected to f(t)
             *
             */
            struct  ipoint_ts {
                virtual ~ipoint_ts(){}

                virtual point_interpretation_policy point_interpretation() const =0;
                virtual void set_point_interpretation(point_interpretation_policy point_interpretation) =0;


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

                // to be removed:
                point get(size_t i) const {return point(time(i),value(i));}

            };
            struct average_ts;//fwd api
            struct time_shift_ts;// fwd api
            /** \brief  apoint_ts, a value-type conceptual ts.
             *
             *  This is the class that we expose to python, with operations, expressions etc.
             *  and that we build all the exposed semantics on.
             *  It holds a shared_ptr to some ipoint_ts, that could be a concrete point timeseries or
             *  an expression.
             *
             */
            class apoint_ts {
                /** a ref to the real implementation, could be a concrete point ts, or an expression */
                std::shared_ptr<ipoint_ts> ts;// consider unique pointer instead,possibly public, to ease transparency in python

               public:
                   typedef gta_t ta_t;///< this is the generic time-axis type for apoint_ts, needed by timeseries namespace templates
				   friend struct average_ts;
				   friend struct time_shift_ts;
                // constructors that we want to expose
                // like

                // these are for the python exposure
                apoint_ts(const time_axis::fixed_dt& ta,double fill_value,point_interpretation_policy point_fx=POINT_INSTANT_VALUE);
                apoint_ts(const time_axis::fixed_dt& ta,const std::vector<double>& values,point_interpretation_policy point_fx=POINT_INSTANT_VALUE);

                apoint_ts(const time_axis::point_dt& ta,double fill_value,point_interpretation_policy point_fx=POINT_INSTANT_VALUE);
                apoint_ts(const time_axis::point_dt& ta,const std::vector<double>& values,point_interpretation_policy point_fx=POINT_INSTANT_VALUE);

                // these are the one we need.
                apoint_ts(const gta_t& ta,double fill_value,point_interpretation_policy point_fx=POINT_INSTANT_VALUE);
                apoint_ts(const gta_t& ta,const std::vector<double>& values,point_interpretation_policy point_fx=POINT_INSTANT_VALUE);
                apoint_ts(const gta_t& ta,std::vector<double>&& values,point_interpretation_policy point_fx=POINT_INSTANT_VALUE);

                apoint_ts(gta_t&& ta,std::vector<double>&& values,point_interpretation_policy point_fx=POINT_INSTANT_VALUE);
                apoint_ts(gta_t&& ta,double fill_value,point_interpretation_policy point_fx=POINT_INSTANT_VALUE);
                apoint_ts(const std::shared_ptr<ipoint_ts>& c):ts(c) {}
                // some more exotic stuff like average_ts


                // std ct/= stuff, that might be ommitted if c++ do the right thing.
                apoint_ts(){}
                apoint_ts(const apoint_ts&c):ts(c.ts){}
                apoint_ts(apoint_ts&& c):ts(std::move(c.ts)){}
                apoint_ts& operator=(const apoint_ts& c) {
                    if( this != &c )
                        ts=c.ts;
                    return *this;
                }
                apoint_ts& operator=(apoint_ts&& c) {
                    ts=std::move(c.ts);
                    return *this;
                }

                /**\brief Easy to compare for equality, but tricky if performance needed */
                bool operator==(const apoint_ts& other) const;

                // interface we want to expose
                // the standard ipoint-ts stuff:
                point_interpretation_policy point_interpretation() const {return ts->point_interpretation();}
                void set_point_interpretation(point_interpretation_policy point_interpretation) { ts->set_point_interpretation(point_interpretation); };
                const gta_t& time_axis() const { return ts->time_axis();};
                utcperiod total_period() const {return ts->total_period();};   ///< Returns period that covers points, given
                size_t index_of(utctime t) const {return ts->index_of(t);};
                size_t size() const {return ts->size();};        ///< number of points that descr. y=f(t) on t ::= period
                utctime time(size_t i) const {return ts->time(i);};///< get the i'th time point
                double value(size_t i) const {return ts->value(i);};///< get the i'th value
                double operator()(utctime t) const  {return ts->value_at(t);};
                std::vector<double> values() const {return ts->values();}

                //-- then some useful functions/properties
                apoint_ts average(const gta_t &ta) const;
                apoint_ts max(double a) const;
                apoint_ts min(double a) const;
                apoint_ts max(const apoint_ts& other) const;
                apoint_ts min(const apoint_ts& other) const;
                static apoint_ts max(const apoint_ts &a, const apoint_ts&b);
                static apoint_ts min(const apoint_ts &a, const apoint_ts&b);

                //-- in case the underlying ipoint_ts is a gpoint_ts (concrete points)
                //   we would like these to be working (exception if it's not possible,i.e. an expression)
                point get(size_t i) const {return point(time(i),value(i));}
                void set(size_t i, double x) ;
                void fill(double x) ;
                void scale_by(double x) ;
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
                // note (we would normally use ct template here, but we are aiming at exposing to python)
                gpoint_ts(const gta_t&ta,double fill_value,point_interpretation_policy point_fx=POINT_INSTANT_VALUE):rep(ta,fill_value,point_fx){}
                gpoint_ts(const gta_t&ta,const std::vector<double>& v,point_interpretation_policy point_fx=POINT_INSTANT_VALUE):rep(ta,v,point_fx) {}
                gpoint_ts(gta_t&&ta,double fill_value,point_interpretation_policy point_fx=POINT_INSTANT_VALUE):rep(std::move(ta),fill_value,point_fx){}
                gpoint_ts(gta_t&&ta,std::vector<double>&& v,point_interpretation_policy point_fx=POINT_INSTANT_VALUE):rep(std::move(ta),std::move(v),point_fx) {}
                gpoint_ts(const gta_t& ta,std::vector<double>&& v,point_interpretation_policy point_fx=POINT_INSTANT_VALUE):rep(ta,std::move(v),point_fx) {}

                // now for the gpoint_ts it self, constructors incl. move
                gpoint_ts(const gpoint_ts& c):rep(c.rep){}
                gpoint_ts(gts_t&& c):rep(std::move(c)){}
                gpoint_ts& operator=(const gpoint_ts&c) {
                    if(this != &c)
                        rep=c.rep;
                    return *this;
                }
                gpoint_ts& operator=(gpoint_ts&& c) {
                    rep=std::move(c.rep);
                    return *this;
                }

                // implement ipoint_ts contract:
                virtual point_interpretation_policy point_interpretation() const {return rep.point_interpretation();}
                virtual void set_point_interpretation(point_interpretation_policy point_interpretation) {rep.set_point_interpretation(point_interpretation);}
                virtual const gta_t& time_axis() const {return rep.time_axis();}
                virtual utcperiod total_period() const {return rep.total_period();}
                virtual size_t index_of(utctime t) const {return rep.index_of(t);}
                virtual size_t size() const {return rep.size();}
                virtual utctime time(size_t i) const {return rep.time(i);};
                virtual double value(size_t i) const {return rep.value(i);}
                virtual double value_at(utctime t) const {return rep(t);}
                virtual std::vector<double> values() const {return rep.v;}
                // implement some extra functions to manipulate the points
                void set(size_t i, double x) {rep.set(i,x);}
                void fill(double x) {rep.fill(x);}
                void scale_by(double x) {rep.scale_by(x);}
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
             * The \ref point_interpretation_policy is always POINT_AVERAGE_VALUE for the result ts.
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
                average_ts(const average_ts &c):ta(c.ta),ts(c.ts) {}
                average_ts(average_ts&&c):ta(std::move(ta)),ts(std::move(c.ts)) {}
                average_ts& operator=(const average_ts&c) {
                    if( this != &c) {
                        ta=c.ta;
                        ts=c.ts;
                    }
                    return *this;
                }
                average_ts& operator=(average_ts&& c) {
                    ta=std::move(c.ta);
                    ts=std::move(c.ts);
                    return *this;
                }
                                // implement ipoint_ts contract:
                virtual point_interpretation_policy point_interpretation() const {return point_interpretation_policy::POINT_AVERAGE_VALUE;}
                virtual void set_point_interpretation(point_interpretation_policy point_interpretation) {;}
                virtual const gta_t& time_axis() const {return ta;}
                virtual utcperiod total_period() const {return ta.total_period();}
                virtual size_t index_of(utctime t) const {return ta.index_of(t);}
                virtual size_t size() const {return ta.size();}
                virtual utctime time(size_t i) const {return ta.time(i);};
                virtual double value(size_t i) const {
                    if(i>ta.size())
                        return nan;
                    size_t ix_hint=(i*ts->size())/ta.size();// assume almost fixed delta-t.
                    return average_value(*ts,ta.period(i),ix_hint,ts->point_interpretation() == point_interpretation_policy::POINT_INSTANT_VALUE);
                }
                virtual double value_at(utctime t) const {
                    // return true average at t
                    if(!ta.total_period().contains(t))
                        return nan;
                    return value(index_of(t));
                }
                virtual std::vector<double> values() const {
                    std::vector<double> r;r.reserve(ta.size());
                    size_t ix_hint=ts->index_of(ta.time(0));
                    bool linear_interpretation=ts->point_interpretation() == point_interpretation_policy::POINT_INSTANT_VALUE;
                    for(size_t i=0;i<ta.size();++i) {
                        r.push_back(average_value(*ts,ta.period(i),ix_hint,linear_interpretation));
                    }
                    return std::move(r);//needed ?
                }
                // to help the average function, return the i'th point of the underlying timeseries
                //point get(size_t i) const {return point(ts->time(i),ts->value(i));}

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
            struct time_shift_ts:ipoint_ts {
                std::shared_ptr<ipoint_ts> ts;
                gta_t ta;
                utctimespan dt;// despite ta time-axis, we need it

                //-- default stuff, ct/copy etc goes here
                time_shift_ts():dt(0) {}
                time_shift_ts(const time_shift_ts& c):ts(c.ts),ta(c.ta),dt(c.dt) {}
                time_shift_ts(time_shift_ts&&c):ts(std::move(c.ts)),ta(std::move(c.ta)),dt(c.dt) {}
                time_shift_ts& operator=(const time_shift_ts& o) {
                    if(this != &o) {
                        ts=o.ts;
                        ta=o.ta;
                        //fx_policy=o.fx_policy;
                        dt=o.dt;
                    }
                    return *this;
                }

                time_shift_ts& operator=(time_shift_ts&& o) {
                    ts=std::move(o.ts);
                    ta=std::move(o.ta);
                    //fx_policy=o.fx_policy;
                    dt=o.dt;
                    return *this;
                }

                //-- useful ct goes here
                time_shift_ts(const apoint_ts& ats,utctimespan dt):ts(ats.ts),ta(time_axis::time_shift(ats.time_axis(),dt)),dt(dt) {}
                time_shift_ts(apoint_ts&& ats, utctimespan dt):ts(std::move(ats.ts)),ta(time_axis::time_shift(ats.time_axis(),dt)),dt(dt) {}
                time_shift_ts(const std::shared_ptr<ipoint_ts> &ts, utctime dt ):ts(ts),ta(time_axis::time_shift(ts->time_axis(),dt)),dt(dt){}

                // implement ipoint_ts contract:
                virtual point_interpretation_policy point_interpretation() const {return ts->point_interpretation();}
                virtual void set_point_interpretation(point_interpretation_policy point_interpretation) {ts->set_point_interpretation(point_interpretation);}
                virtual const gta_t& time_axis() const {return ta;}
                virtual utcperiod total_period() const {return ta.total_period();}
                virtual size_t index_of(utctime t) const {return ta.index_of(t);}
                virtual size_t size() const {return ta.size();}
                virtual utctime time(size_t i) const {return ta.time(i);};
                virtual double value(size_t i) const {return ts->value(i);}
                virtual double value_at(utctime t) const {return ts->value_at(t-dt);}
                virtual std::vector<double> values() const {return ts->values();}

            };




            /** The iop_t represent the basic 'binary' operation,
             *   a stateless function that takes two doubles and returns the binary operation.
             *   E.g.: a+b
             *   The iop_t is used as the operation element of the abin_op_ts class
             */
            typedef double (*iop_t)(double a,double b);


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
             * The \ref point_interpretation_policy is computed based on rhs,lhs. But can be overridden
             * by the user.
             *
             */
            struct abin_op_ts:ipoint_ts {

                  apoint_ts lhs;
                  iop_t op;
                  apoint_ts rhs;
                  gta_t ta;
                  point_interpretation_policy fx_policy;
                  point_interpretation_policy point_interpretation() const {return fx_policy;}
                  void set_point_interpretation(point_interpretation_policy x) {fx_policy=x;}

                  abin_op_ts(const apoint_ts &lhs,iop_t op,const apoint_ts& rhs)
                  :lhs(lhs),op(op),rhs(rhs) {
                      ta=time_axis::combine(lhs.time_axis(),rhs.time_axis());
                      fx_policy= result_policy(lhs.point_interpretation(),rhs.point_interpretation());
                  }
                  abin_op_ts(const abin_op_ts& c)
                    :lhs(c.lhs),op(c.op),rhs(c.rhs),ta(c.ta),fx_policy(c.fx_policy) {
                  }
                  abin_op_ts(abin_op_ts&& c)
                    :lhs(std::move(c.lhs)),op(c.op),rhs(std::move(c.rhs)),ta(std::move(c.ta)),fx_policy(c.fx_policy)
                     {
                  }

                  abin_op_ts& operator=(const abin_op_ts& c) {
                    if( this != & c) {
                        lhs = c.lhs;
                        op = c.op;
                        rhs = c.rhs;
                        ta = c.ta;
                        fx_policy = c.fx_policy;
                    }
                    return *this;
                  }

                  abin_op_ts& operator=(abin_op_ts&& c) {
                    if( this != & c) {
                        lhs = std::move(c.lhs);
                        op = c.op;
                        rhs = std::move(c.rhs);
                        ta  = std::move(c.ta);
                        fx_policy = c.fx_policy;
                    }
                    return *this;
                  }

                  virtual utcperiod total_period() const {return ta.total_period();}
                  const gta_t& time_axis() const {return ta;};// combine lhs,rhs
                  size_t index_of(utctime t) const{return ta.index_of(t);};
                  size_t size() const {return ta.size();};// use the combined ta.size();
                  utctime time( size_t i) const {return ta.time(i);}; // reeturn combined ta.time(i)
                  double value_at(utctime t) const ;
                  double value(size_t i) const;// return op( lhs(t), rhs(t)) ..
                  std::vector<double> values() const;

            };

            /** \brief  binary operation for type ts op double
             *
             * The resulting time-axis and point interpretation policy is equal to the ts.
             */
            struct abin_op_scalar_ts:ipoint_ts {
                  double lhs;
                  iop_t op;
                  apoint_ts rhs;
                  gta_t ta;
                  point_interpretation_policy fx_policy;
                  point_interpretation_policy point_interpretation() const {return fx_policy;}
                  void set_point_interpretation(point_interpretation_policy x) {fx_policy=x;}

                  abin_op_scalar_ts(double lhs,iop_t op,const apoint_ts& rhs)
                  :lhs(lhs),op(op),rhs(rhs) {
                      ta=rhs.time_axis();
                      fx_policy= rhs.point_interpretation();
                  }
                  abin_op_scalar_ts(const abin_op_scalar_ts& c)
                    :lhs(c.lhs),op(c.op),rhs(c.rhs),ta(c.ta),fx_policy(c.fx_policy) {
                  }
                  abin_op_scalar_ts(abin_op_scalar_ts&& c)
                    :lhs(c.lhs),op(c.op),rhs(std::move(c.rhs)),ta(std::move(c.ta)),fx_policy(c.fx_policy)
                     {
                  }

                  abin_op_scalar_ts& operator=(const abin_op_scalar_ts& c) {
                    if( this != & c) {
                        lhs = c.lhs;
                        op = c.op;
                        rhs = c.rhs;
                        ta = c.ta;
                        fx_policy = c.fx_policy;
                    }
                    return *this;
                  }

                  abin_op_scalar_ts& operator=(abin_op_scalar_ts&& c) {
                    if( this != & c) {
                        lhs = c.lhs;
                        op = c.op;
                        rhs = std::move(c.rhs);
                        ta  = std::move(c.ta);
                        fx_policy = c.fx_policy;
                    }
                    return *this;
                  }

                  virtual utcperiod total_period() const {return ta.total_period();}
                  const gta_t& time_axis() const {return ta;};// combine lhs,rhs
                  size_t index_of(utctime t) const{return ta.index_of(t);};
                  size_t size() const {return ta.size();};
                  utctime time( size_t i) const {return ta.time(i);};
                  double value_at(utctime t) const {return op(lhs,rhs(t));}
                  double value(size_t i) const {return op(lhs,rhs.value(i));}
                  std::vector<double> values() const {
                      std::vector<double> r(rhs.values());
                      for(auto& v:r)
                        v=op(lhs,v);
                      return r;
                  }

            };

            /** \brief  binary operation for type ts op double
             *
             * The resulting time-axis and point interpretation policy is equal to the ts.
             */
            struct abin_op_ts_scalar:ipoint_ts {
                  apoint_ts lhs;
                  iop_t op;
                  double rhs;
                  gta_t ta;
                  point_interpretation_policy fx_policy;
                  point_interpretation_policy point_interpretation() const {return fx_policy;}
                  void set_point_interpretation(point_interpretation_policy x) {fx_policy=x;}

                  abin_op_ts_scalar(const apoint_ts &lhs,iop_t op,double rhs)
                  :lhs(lhs),op(op),rhs(rhs) {
                      ta=lhs.time_axis();
                      fx_policy= lhs.point_interpretation();
                  }
                  abin_op_ts_scalar(const abin_op_ts_scalar& c)
                    :lhs(c.lhs),op(c.op),rhs(c.rhs),ta(c.ta),fx_policy(c.fx_policy) {
                  }
                  abin_op_ts_scalar(abin_op_ts_scalar&& c)
                    :lhs(std::move(c.lhs)),op(c.op),rhs(c.rhs),ta(std::move(c.ta)),fx_policy(c.fx_policy)
                     {
                  }

                  abin_op_ts_scalar& operator=(const abin_op_ts_scalar& c) {
                    if( this != & c) {
                        lhs = c.lhs;
                        op = c.op;
                        rhs = c.rhs;
                        ta = c.ta;
                        fx_policy = c.fx_policy;
                    }
                    return *this;
                  }

                  abin_op_ts_scalar& operator=(abin_op_ts_scalar&& c) {
                    if( this != & c) {
                        lhs = std::move(c.lhs);
                        op = c.op;
                        rhs = c.rhs;
                        ta  = std::move(c.ta);
                        fx_policy = c.fx_policy;
                    }
                    return *this;
                  }

                  virtual utcperiod total_period() const {return ta.total_period();}
                  const gta_t& time_axis() const {return ta;};
                  size_t index_of(utctime t) const{return ta.index_of(t);};
                  size_t size() const {return ta.size();};
                  utctime time( size_t i) const {return ta.time(i);};
                  double value_at(utctime t) const {return op(lhs(t),rhs);}
                  double value(size_t i) const {return op(lhs.value(i),rhs);}
                  std::vector<double> values() const {
                      std::vector<double> r(lhs.values());
                      for(auto& v:r)
                        v=op(rhs,v);
                      return r;
                  }

            };

            // add operators and functions to the apoint_ts class, of all variants that we want to expose
            apoint_ts average(const apoint_ts& ts,const gta_t& ta/*fx-type */) ;
            apoint_ts average(apoint_ts&& ts,const gta_t& ta) ;


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
    }
}
