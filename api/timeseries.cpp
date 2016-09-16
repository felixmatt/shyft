#include "timeseries.h"

namespace shyft{
    namespace api {
        static double iop_add(double a,double b) {return a + b;}
        static double iop_sub(double a,double b) {return a - b;}
        static double iop_div(double a,double b) {return a/b;}
        static double iop_mul(double a,double b) {return a*b;}
        static double iop_min(double a,double b) {return std::min(a,b);}
        static double iop_max(double a,double b) {return std::max(a,b);}

       // add operators and functions to the apoint_ts class, of all variants that we want to expose
        apoint_ts average(const apoint_ts& ts,const gta_t& ta/*fx-type */)  { return apoint_ts(std::make_shared<average_ts>(ta,ts));}
        apoint_ts average(apoint_ts&& ts,const gta_t& ta)  { return apoint_ts(std::make_shared<average_ts>(ta,std::move(ts)));}

		apoint_ts accumulate(const apoint_ts& ts, const gta_t& ta/*fx-type */) { return apoint_ts(std::make_shared<accumulate_ts>(ta, ts)); }
		apoint_ts accumulate(apoint_ts&& ts, const gta_t& ta) { return apoint_ts(std::make_shared<accumulate_ts>(ta, std::move(ts))); }

		apoint_ts periodic(const vector<double>& pattern, utctimespan dt, const gta_t& ta) { return apoint_ts(make_shared<periodic_ts>(pattern, dt, ta)); }

        apoint_ts operator+(const apoint_ts& lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_ts       >( lhs,iop_add,rhs )); }
        apoint_ts operator+(const apoint_ts& lhs,double           rhs) {return apoint_ts(std::make_shared<abin_op_ts_scalar>( lhs,iop_add,rhs )); }
        apoint_ts operator+(double           lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_scalar_ts>( lhs,iop_add,rhs )); }

        apoint_ts operator-(const apoint_ts& lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_ts       >( lhs,iop_sub,rhs )); }
        apoint_ts operator-(const apoint_ts& lhs,double           rhs) {return apoint_ts(std::make_shared<abin_op_ts_scalar>( lhs,iop_sub,rhs )); }
        apoint_ts operator-(double           lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_scalar_ts>( lhs,iop_sub,rhs )); }
        apoint_ts operator-(const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_scalar_ts>( -1.0,iop_mul,rhs )); }

        apoint_ts operator/(const apoint_ts& lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_ts       >( lhs,iop_div,rhs )); }
        apoint_ts operator/(const apoint_ts& lhs,double           rhs) {return apoint_ts(std::make_shared<abin_op_ts_scalar>( lhs,iop_div,rhs )); }
        apoint_ts operator/(double           lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_scalar_ts>( lhs,iop_div,rhs )); }

        apoint_ts operator*(const apoint_ts& lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_ts       >( lhs,iop_mul,rhs )); }
        apoint_ts operator*(const apoint_ts& lhs,double           rhs) {return apoint_ts(std::make_shared<abin_op_ts_scalar>( lhs,iop_mul,rhs )); }
        apoint_ts operator*(double           lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_scalar_ts>( lhs,iop_mul,rhs )); }


        apoint_ts max(const apoint_ts& lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_ts       >( lhs,iop_max,rhs ));}
        apoint_ts max(const apoint_ts& lhs,double           rhs) {return apoint_ts(std::make_shared<abin_op_ts_scalar>( lhs,iop_max,rhs ));}
        apoint_ts max(double           lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_scalar_ts>( lhs,iop_max,rhs ));}

        apoint_ts min(const apoint_ts& lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_ts>( lhs,iop_min,rhs ));}
        apoint_ts min(const apoint_ts& lhs,double           rhs) {return apoint_ts(std::make_shared<abin_op_ts_scalar>( lhs,iop_min,rhs ));}
        apoint_ts min(double           lhs,const apoint_ts& rhs) {return apoint_ts(std::make_shared<abin_op_scalar_ts>( lhs,iop_min,rhs ));}

        double abin_op_ts::value_at(utctime t) const {
            if(!ta.total_period().contains(t))
                return nan;
            return op( lhs(t),rhs(t) );// this might cost a 2xbin-search if not the underlying ts have smart incremental search (at the cost of thread safety)
        }

        std::vector<double> abin_op_ts::values() const {
            std::vector<double> r;r.reserve(ta.size());
            for(size_t i=0;i<ta.size();++i) {
                r.push_back(value(i));//TODO: improve speed using accessors with ix-hint for lhs/rhs stepwise traversal
            }
            return std::move(r);
        }

        // implement popular ct for apoint_ts to make it easy to expose & use
        apoint_ts::apoint_ts(const time_axis::generic_dt& ta,double fill_value,point_interpretation_policy point_fx)
            :ts(std::make_shared<gpoint_ts>(ta,fill_value,point_fx)) {
        }
        apoint_ts::apoint_ts(const time_axis::generic_dt& ta,const std::vector<double>& values,point_interpretation_policy point_fx)
            :ts(std::make_shared<gpoint_ts>(ta,values,point_fx)) {
        }
        apoint_ts::apoint_ts(const time_axis::generic_dt& ta,std::vector<double>&& values,point_interpretation_policy point_fx)
            :ts(std::make_shared<gpoint_ts>(ta,std::move(values),point_fx)) {
        }

        // and python needs these:
        apoint_ts::apoint_ts(const time_axis::fixed_dt& ta,double fill_value,point_interpretation_policy point_fx)
            :apoint_ts(time_axis::generic_dt(ta),fill_value,point_fx) {
        }
        apoint_ts::apoint_ts(const time_axis::fixed_dt& ta,const std::vector<double>& values,point_interpretation_policy point_fx)
            :apoint_ts(time_axis::generic_dt(ta),values,point_fx) {
        }

        // and python needs these:
        apoint_ts::apoint_ts(const time_axis::point_dt& ta,double fill_value,point_interpretation_policy point_fx)
            :apoint_ts(time_axis::generic_dt(ta),fill_value,point_fx) {
        }
        apoint_ts::apoint_ts(const time_axis::point_dt& ta,const std::vector<double>& values,point_interpretation_policy point_fx)
            :apoint_ts(time_axis::generic_dt(ta),values,point_fx) {
        }
		apoint_ts::apoint_ts(const rts_t &rts):
			apoint_ts(time_axis::generic_dt(rts.ta),rts.v,rts.point_interpretation())
		{}

		apoint_ts::apoint_ts(const vector<double>& pattern, utctimespan dt, const time_axis::generic_dt& ta) :
			apoint_ts(make_shared<periodic_ts>(pattern, dt, ta)) {}


        apoint_ts::apoint_ts(time_axis::generic_dt&& ta,std::vector<double>&& values,point_interpretation_policy point_fx)
            :ts(std::make_shared<gpoint_ts>(std::move(ta),std::move(values),point_fx)) {
        }
        apoint_ts::apoint_ts(time_axis::generic_dt&& ta,double fill_value,point_interpretation_policy point_fx)
            :ts(std::make_shared<gpoint_ts>(std::move(ta),fill_value,point_fx))
            {
        }
        apoint_ts apoint_ts::average(const gta_t &ta) const {
            return shyft::api::average(*this,ta);
        }
		apoint_ts apoint_ts::accumulate(const gta_t &ta) const {
			return shyft::api::accumulate(*this, ta);
		}
		apoint_ts apoint_ts::time_shift(utctimespan dt) const {
			return shyft::api::time_shift(*this, dt);
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

        std::vector<apoint_ts> percentiles(const std::vector<apoint_ts>& ts_list,const gta_t& ta, const vector<int>& percentile_list) {
            std::vector<apoint_ts> r;r.reserve(percentile_list.size());
            //-- use core calc here:
            auto rp= shyft::timeseries::calculate_percentiles(ta,ts_list,percentile_list);
            for(auto&ts:rp) r.emplace_back(ta,std::move(ts.v),POINT_AVERAGE_VALUE);
            return r;
        }
        std::vector<apoint_ts> percentiles(const std::vector<apoint_ts>& ts_list,const time_axis::fixed_dt& ta, const vector<int>& percentile_list) {
            return percentiles(ts_list,time_axis::generic_dt(ta),percentile_list);
        }

        double abin_op_ts::value(size_t i) const {
            if(i==std::string::npos || i>=ta.size() )
                return nan;
            return value_at(ta.time(i));
            // Hmm! here we have to define what value(i) means
            // .. we could define it as
            //     the value at the beginning of the i'th period
            //
            //   then we define value(t) as
            //    as the point interpretation of f(t)..
            //
            //if(fx_policy==point_interpretation_policy::POINT_AVERAGE_VALUE)
            //    return value_at(ta.time(i));
            //utcperiod p=ta.period(i);
            //double v0= value_at(p.start);
            //double v1= value_at(p.end);
            //if(isfinite(v1)) return 0.5*(v0 + v1);
            //return v0;
        }
        apoint_ts time_shift(const apoint_ts& ts, utctimespan dt) {
            return apoint_ts( std::make_shared<shyft::api::time_shift_ts>(ts,dt));
        }

    }
}
