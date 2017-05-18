#include "core/core_pch.h"
#include "time_series.h"

#include <dlib/statistics.h>

namespace shyft{
    namespace api {
        static inline double do_op(double a,iop_t op,double b) {
            switch(op) {
            case iop_t::OP_ADD:return a+b;
            case iop_t::OP_SUB:return a-b;
            case iop_t::OP_DIV:return a/b;
            case iop_t::OP_MUL:return a*b;
            case iop_t::OP_MAX:return std::max(a,b);
            case iop_t::OP_MIN:return std::min(a,b);
            case iop_t::OP_NONE:break;// just fall to exception
            }
            throw std::runtime_error("unsupported shyft::api::iop_t");
        }
       // add operators and functions to the apoint_ts class, of all variants that we want to expose
        apoint_ts average(const apoint_ts& ts,const gta_t& ta/*fx-type */)  { return apoint_ts(std::make_shared<average_ts>(ta,ts));}
        apoint_ts average(apoint_ts&& ts,const gta_t& ta)  { return apoint_ts(std::make_shared<average_ts>(ta,std::move(ts)));}
        apoint_ts integral(const apoint_ts& ts, const gta_t& ta/*fx-type */) { return apoint_ts(std::make_shared<integral_ts>(ta, ts)); }
        apoint_ts integral(apoint_ts&& ts, const gta_t& ta) { return apoint_ts(std::make_shared<integral_ts>(ta, std::move(ts))); }

		apoint_ts accumulate(const apoint_ts& ts, const gta_t& ta/*fx-type */) { return apoint_ts(std::make_shared<accumulate_ts>(ta, ts)); }
		apoint_ts accumulate(apoint_ts&& ts, const gta_t& ta) { return apoint_ts(std::make_shared<accumulate_ts>(ta, std::move(ts))); }

		apoint_ts create_periodic_pattern_ts(const vector<double>& pattern, utctimespan dt, utctime pattern_t0,const gta_t& ta) { return apoint_ts(make_shared<periodic_ts>(pattern, dt, pattern_t0, ta)); }

        apoint_ts operator+(const apoint_ts& lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_ts       >( lhs,iop_t::OP_ADD,rhs )); }
        apoint_ts operator+(const apoint_ts& lhs,double           rhs) {return apoint_ts(std::make_shared<abin_op_ts_scalar>( lhs,iop_t::OP_ADD,rhs )); }
        apoint_ts operator+(double           lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_scalar_ts>( lhs,iop_t::OP_ADD,rhs )); }

        apoint_ts operator-(const apoint_ts& lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_ts       >( lhs,iop_t::OP_SUB,rhs )); }
        apoint_ts operator-(const apoint_ts& lhs,double           rhs) {return apoint_ts(std::make_shared<abin_op_ts_scalar>( lhs,iop_t::OP_SUB,rhs )); }
        apoint_ts operator-(double           lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_scalar_ts>( lhs,iop_t::OP_SUB,rhs )); }
        apoint_ts operator-(const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_scalar_ts>( -1.0,iop_t::OP_MUL,rhs )); }

        apoint_ts operator/(const apoint_ts& lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_ts       >( lhs,iop_t::OP_DIV,rhs )); }
        apoint_ts operator/(const apoint_ts& lhs,double           rhs) {return apoint_ts(std::make_shared<abin_op_ts_scalar>( lhs,iop_t::OP_DIV,rhs )); }
        apoint_ts operator/(double           lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_scalar_ts>( lhs,iop_t::OP_DIV,rhs )); }

        apoint_ts operator*(const apoint_ts& lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_ts       >( lhs,iop_t::OP_MUL,rhs )); }
        apoint_ts operator*(const apoint_ts& lhs,double           rhs) {return apoint_ts(std::make_shared<abin_op_ts_scalar>( lhs,iop_t::OP_MUL,rhs )); }
        apoint_ts operator*(double           lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_scalar_ts>( lhs,iop_t::OP_MUL,rhs )); }


        apoint_ts max(const apoint_ts& lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_ts       >( lhs,iop_t::OP_MAX,rhs ));}
        apoint_ts max(const apoint_ts& lhs,double           rhs) {return apoint_ts(std::make_shared<abin_op_ts_scalar>( lhs,iop_t::OP_MAX,rhs ));}
        apoint_ts max(double           lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_scalar_ts>( lhs,iop_t::OP_MAX,rhs ));}

        apoint_ts min(const apoint_ts& lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_ts>( lhs, iop_t::OP_MIN, rhs ));}
        apoint_ts min(const apoint_ts& lhs,double           rhs) {return apoint_ts(std::make_shared<abin_op_ts_scalar>( lhs,iop_t::OP_MIN,rhs ));}
        apoint_ts min(double           lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_scalar_ts>( lhs,iop_t::OP_MIN,rhs ));}

        double abin_op_ts::value_at(utctime t) const {
            if(!time_axis().total_period().contains(t))
                return nan;
            return do_op( lhs(t),op,rhs(t) );// this might cost a 2xbin-search if not the underlying ts have smart incremental search (at the cost of thread safety)
        }

        std::vector<double> abin_op_ts::values() const {
            std::vector<double> r;r.reserve(time_axis().size());
            for(size_t i=0;i<time_axis().size();++i) {
                r.push_back(value(i));//TODO: improve speed using accessors with ix-hint for lhs/rhs stepwise traversal
            }
            return r;
        }

        // implement popular ct for apoint_ts to make it easy to expose & use
        apoint_ts::apoint_ts(const time_axis::generic_dt& ta,double fill_value,ts_point_fx point_fx)
            :ts(std::make_shared<gpoint_ts>(ta,fill_value,point_fx)) {
        }
        apoint_ts::apoint_ts(const time_axis::generic_dt& ta,const std::vector<double>& values,ts_point_fx point_fx)
            :ts(std::make_shared<gpoint_ts>(ta,values,point_fx)) {
        }
        apoint_ts::apoint_ts(const time_axis::generic_dt& ta,std::vector<double>&& values,ts_point_fx point_fx)
            :ts(std::make_shared<gpoint_ts>(ta,std::move(values),point_fx)) {
        }

        apoint_ts::apoint_ts(std::string ref_ts_id)
             :ts(std::make_shared<aref_ts>(ref_ts_id)) {
        }
        void apoint_ts::bind(const apoint_ts& bts) {
            if(!dynamic_cast<aref_ts*>(ts.get()))
                throw runtime_error("this time-series is not bindable");
            if(!dynamic_cast<gpoint_ts*>(bts.ts.get()))
                throw runtime_error("the supplied argument time-series must be a point ts");
            dynamic_cast<aref_ts*>(ts.get())->rep.set_ts( make_shared<gts_t>( dynamic_cast<gpoint_ts*>(bts.ts.get())->rep ));
        }

        // and python needs these:
        apoint_ts::apoint_ts(const time_axis::fixed_dt& ta,double fill_value,ts_point_fx point_fx)
            :apoint_ts(time_axis::generic_dt(ta),fill_value,point_fx) {
        }
        apoint_ts::apoint_ts(const time_axis::fixed_dt& ta,const std::vector<double>& values,ts_point_fx point_fx)
            :apoint_ts(time_axis::generic_dt(ta),values,point_fx) {
        }

        // and python needs these:
        apoint_ts::apoint_ts(const time_axis::point_dt& ta,double fill_value,ts_point_fx point_fx)
            :apoint_ts(time_axis::generic_dt(ta),fill_value,point_fx) {
        }
        apoint_ts::apoint_ts(const time_axis::point_dt& ta,const std::vector<double>& values,ts_point_fx point_fx)
            :apoint_ts(time_axis::generic_dt(ta),values,point_fx) {
        }
		apoint_ts::apoint_ts(const rts_t &rts):
			apoint_ts(time_axis::generic_dt(rts.ta),rts.v,rts.point_interpretation())
		{}

		apoint_ts::apoint_ts(const vector<double>& pattern, utctimespan dt, const time_axis::generic_dt& ta) :
			apoint_ts(make_shared<periodic_ts>(pattern, dt, ta)) {}
		apoint_ts::apoint_ts(const vector<double>& pattern, utctimespan dt, utctime pattern_t0,const time_axis::generic_dt& ta) :
			apoint_ts(make_shared<periodic_ts>(pattern, dt,pattern_t0,ta)) {}

        apoint_ts::apoint_ts(time_axis::generic_dt&& ta,std::vector<double>&& values,ts_point_fx point_fx)
            :ts(std::make_shared<gpoint_ts>(std::move(ta),std::move(values),point_fx)) {
        }
        apoint_ts::apoint_ts(time_axis::generic_dt&& ta,double fill_value,ts_point_fx point_fx)
            :ts(std::make_shared<gpoint_ts>(std::move(ta),fill_value,point_fx))
            {
        }
        apoint_ts apoint_ts::average(const gta_t &ta) const {
            return shyft::api::average(*this,ta);
        }
        apoint_ts apoint_ts::integral(const gta_t &ta) const {
            return shyft::api::integral(*this, ta);
        }
		apoint_ts apoint_ts::accumulate(const gta_t &ta) const {
			return shyft::api::accumulate(*this, ta);
		}
		apoint_ts apoint_ts::time_shift(utctimespan dt) const {
			return shyft::api::time_shift(*this, dt);
		}

        /** recursive function to dig out bind_info */
        static void find_ts_bind_info(const std::shared_ptr<shyft::api::ipoint_ts>&its, std::vector<ts_bind_info>&r) {
            using namespace shyft;
            if (its == nullptr)
                return;
            if (dynamic_cast<const api::aref_ts*>(its.get())) {
                auto rts = dynamic_cast<const api::aref_ts*>(its.get());
                if (rts)
                    r.push_back(api::ts_bind_info( rts->rep.ref,api::apoint_ts(its)));
                else
                    ;// maybe throw ?
            } else if (dynamic_cast<const api::average_ts*>(its.get())) {
                find_ts_bind_info(dynamic_cast<const api::average_ts*>(its.get())->ts, r);
            } else if (dynamic_cast<const api::integral_ts*>(its.get())) {
                find_ts_bind_info(dynamic_cast<const api::integral_ts*>(its.get())->ts, r);
            } else if(dynamic_cast<const api::accumulate_ts*>(its.get())) {
                find_ts_bind_info(dynamic_cast<const api::accumulate_ts*>(its.get())->ts, r);
            } else if (dynamic_cast<const api::time_shift_ts*>(its.get())) {
                find_ts_bind_info(dynamic_cast<const api::time_shift_ts*>(its.get())->ts, r);
            } else if (dynamic_cast<const api::abin_op_ts*>(its.get())) {
                auto bin_op = dynamic_cast<const api::abin_op_ts*>(its.get());
                find_ts_bind_info(bin_op->lhs.ts, r);
                find_ts_bind_info(bin_op->rhs.ts, r);
            } else if (dynamic_cast<const api::abin_op_scalar_ts*>(its.get())) {
                auto bin_op = dynamic_cast<const api::abin_op_scalar_ts*>(its.get());
                find_ts_bind_info(bin_op->rhs.ts, r);
            } else if (dynamic_cast<const api::abin_op_ts_scalar*>(its.get())) {
                auto bin_op = dynamic_cast<const api::abin_op_ts_scalar*>(its.get());
                find_ts_bind_info(bin_op->lhs.ts, r);
            }
        }

        std::vector<ts_bind_info> apoint_ts::find_ts_bind_info() const {
            std::vector<ts_bind_info> r;
            shyft::api::find_ts_bind_info(ts, r);
            return r;
        }

		std::vector<apoint_ts> apoint_ts::partition_by(const calendar& cal, utctime t, utctimespan partition_interval, size_t n_partitions, utctime common_t0) const {
			// some very rudimentary argument checks:
			if (n_partitions < 1)
				throw std::runtime_error("n_partitions should be > 0");
			if (partition_interval <= 0)
				throw std::runtime_error("partition_interval should be > 0, typically Calendar::YEAR|MONTH|WEEK|DAY");
			auto mk_raw_time_shift = [](const apoint_ts& ts, utctimespan dt)->apoint_ts {
				return apoint_ts(std::make_shared<shyft::api::time_shift_ts>(ts, dt));
			};
			return shyft::time_series::partition_by<apoint_ts>(*this, cal, t,partition_interval, n_partitions, common_t0, mk_raw_time_shift);
		}

        void apoint_ts::set(size_t i, double x) {
            gpoint_ts *gpts=dynamic_cast<gpoint_ts*>(ts.get());
            if(!gpts)
                throw std::runtime_error("apoint_ts::set(i,x) only allowed for ts of non-expression types");
            gpts->set(i,x);
        }
        void apoint_ts::fill(double x) {
            gpoint_ts *gpts=dynamic_cast<gpoint_ts*>(ts.get());
            if(!gpts)
                throw std::runtime_error("apoint_ts::fill(x) only allowed for ts of non-expression types");
            gpts->fill(x);
        }
        void apoint_ts::scale_by(double x) {
            gpoint_ts *gpts=dynamic_cast<gpoint_ts*>(ts.get());
            if(!gpts)
                throw std::runtime_error("apoint_ts::scale_by(x) only allowed for ts of non-expression types");
            gpts->scale_by(x);
        }

        bool apoint_ts::operator==(const apoint_ts& other) const {
            if(ts.get() == other.ts.get()) // equal by reference
                return true;
            // the time-consuming part, equal by value
            // SiH: I am not sure if this is ever useful, nevertheless it's doable
            //      given that it's ok with the zero-limit(not generally applicable!)

            const double zero_limit=1e-9;// in hydrology, this is usually a small number
            if(ts->size() != other.ts->size())
                return false;
            for(size_t i=0;i<ts->size();++i) {
                if( ts->time_axis().period(i) != other.ts->time_axis().period(i))
                    return false;
                if( fabs(ts->value(i) - other.ts->value(i)) >zero_limit)
                    return false;
            }
            return true;
        }

        apoint_ts apoint_ts::max(double a) const {return shyft::api::max(*this,a);}
        apoint_ts apoint_ts::min(double a) const {return shyft::api::min(*this,a);}

        apoint_ts apoint_ts::max(const apoint_ts& other) const {return shyft::api::max(*this,other);}
        apoint_ts apoint_ts::min(const apoint_ts& other) const {return shyft::api::min(*this,other);}

        apoint_ts apoint_ts::max(const apoint_ts &a, const apoint_ts&b){return shyft::api::max(a,b);}
        apoint_ts apoint_ts::min(const apoint_ts &a, const apoint_ts&b){return shyft::api::min(a,b);}
        apoint_ts apoint_ts::convolve_w(const std::vector<double> &w, shyft::time_series::convolve_policy conv_policy) const {
            return apoint_ts(std::make_shared<shyft::api::convolve_w_ts>(*this, w, conv_policy));
        }


        std::vector<apoint_ts> percentiles(const std::vector<apoint_ts>& tsv1,const gta_t& ta, const vector<int>& percentile_list) {
            std::vector<apoint_ts> r;r.reserve(percentile_list.size());
            auto rp= shyft::time_series::calculate_percentiles(ta,deflate_ts_vector<gts_t>(tsv1),percentile_list);
            for(auto&ts:rp) r.emplace_back(ta,std::move(ts.v),POINT_AVERAGE_VALUE);
            return r;
        }

        std::vector<apoint_ts> percentiles(const std::vector<apoint_ts>& ts_list,const time_axis::fixed_dt& ta, const vector<int>& percentile_list) {
            return percentiles(ts_list,time_axis::generic_dt(ta),percentile_list);
        }

        double abin_op_ts::value(size_t i) const {
            if(i==std::string::npos || i>=time_axis().size() )
                return nan;
            return value_at(time_axis().time(i));
        }
        double abin_op_scalar_ts::value_at(utctime t) const {
            bind_check();
            return do_op(lhs,op,rhs(t));
        }
        double abin_op_scalar_ts::value(size_t i) const {
            bind_check();
            return do_op(lhs,op,rhs.value(i));
        }

        std::vector<double> abin_op_scalar_ts::values() const {
          bind_check();
          std::vector<double> r(rhs.values());
          for(auto& v:r)
            v=do_op(lhs,op,v);
          return r;
        }

        double abin_op_ts_scalar::value_at(utctime t) const {
            bind_check();
            return do_op(lhs(t),op,rhs);
        }
        double abin_op_ts_scalar::value(size_t i) const {
            bind_check();
            return do_op(lhs.value(i),op,rhs);
        }
        std::vector<double> abin_op_ts_scalar::values() const {
          bind_check();
          std::vector<double> r(lhs.values());
          for(auto& v:r)
            v=do_op(v,op,rhs);
          return r;
        }

        apoint_ts time_shift(const apoint_ts& ts, utctimespan dt) {
            return apoint_ts( std::make_shared<shyft::api::time_shift_ts>(ts,dt));
        }

		double nash_sutcliffe(const apoint_ts& observation_ts, const apoint_ts& model_ts, const gta_t &ta) {
			average_accessor<apoint_ts, gta_t> o(observation_ts, ta);
			average_accessor<apoint_ts, gta_t> m(model_ts, ta);
			return 1.0 - shyft::time_series::nash_sutcliffe_goal_function(o, m);
		}

		double kling_gupta( const apoint_ts & observation_ts,  const apoint_ts &model_ts, const gta_t & ta, double s_r, double s_a, double s_b) {
			average_accessor<apoint_ts, gta_t> o(observation_ts, ta);
			average_accessor<apoint_ts, gta_t> m(model_ts, ta);
			return 1.0 - shyft::time_series::kling_gupta_goal_function<dlib::running_scalar_covariance<double>>(o, m, s_r, s_a, s_b);
		}
		// glacier_melt_ts as apoint_ts with it's internal being a glacier_melt_ts
        struct aglacier_melt_ts:ipoint_ts {
            glacier_melt_ts<std::shared_ptr<ipoint_ts>> gm;
            //-- default stuff, ct/copy etc goes here
            aglacier_melt_ts() =default;

            //-- useful ct goes here
            aglacier_melt_ts(const apoint_ts& temp,const apoint_ts& sca_m2, double glacier_area_m2,double dtf):
                gm(temp.ts,sca_m2.ts,glacier_area_m2,dtf)
                {
                }
            //aglacier_melt_ts(apoint_ts&& ats, utctimespan dt):ts(std::move(ats.ts)),ta(time_axis::time_shift(ats.time_axis(),dt)),dt(dt) {}
            //aglacier_melt_ts(const std::shared_ptr<ipoint_ts> &ts, utctime dt ):ts(ts),ta(time_axis::time_shift(ts->time_axis(),dt)),dt(dt){}

            // implement ipoint_ts contract:
            virtual ts_point_fx point_interpretation() const {return gm.fx_policy;}
            virtual void set_point_interpretation(ts_point_fx point_interpretation) {gm.fx_policy=point_interpretation;}
            virtual const gta_t& time_axis() const {return gm.time_axis();}
            virtual utcperiod total_period() const {return gm.time_axis().total_period();}
            virtual size_t index_of(utctime t) const {return gm.time_axis().index_of(t);}
            virtual size_t size() const {return gm.time_axis().size();}
            virtual utctime time(size_t i) const {return gm.time_axis().time(i);};
            virtual double value(size_t i) const {return gm.value(i);}
            virtual double value_at(utctime t) const {return gm(t);}
            virtual std::vector<double> values() const {
                std::vector<double> r;r.reserve(size());
                for(size_t i=0;i<size();++i) r.push_back(value(i));
                return r;
            }
            virtual bool needs_bind() const {return gm.temperature->needs_bind() || gm.sca_m2->needs_bind();}
            virtual void do_bind() {gm.temperature->do_bind();gm.sca_m2->do_bind();}

        };
        apoint_ts create_glacier_melt_ts_m3s(const apoint_ts & temp,const apoint_ts& sca_m2,double glacier_area_m2,double dtf) {
            return apoint_ts(make_shared<aglacier_melt_ts>(temp,sca_m2,glacier_area_m2,dtf));
        }



        std::vector<char> apoint_ts::serialize_to_bytes() const {
            auto ss = serialize();
            return std::vector<char>(std::begin(ss), std::end(ss));
        }
        apoint_ts apoint_ts::deserialize_from_bytes(const std::vector<char>&ss) {
            return deserialize(std::string(ss.data(), ss.size()));
        }

    }
}

