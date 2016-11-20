#pragma once

#include "timeseries.h"


namespace boost {
    namespace serialization {


        /* time & calendar serialization */
        template <class Archive>
        void serialize(Archive & ar, shyft::core::utcperiod &o, const unsigned int version) {
            ar
                &  make_nvp("start", o.start)
                &  make_nvp("end", o.end)
                ;
        }

        template <class Archive>
        void serialize(Archive & ar, shyft::core::time_zone::tz_table &o, const unsigned int version) {
            ar
                &  make_nvp("start_year", o.start_year)
                &  make_nvp("tz_nae", o.tz_name)
                &  make_nvp("dst", o.dst)
                &  make_nvp("dt", o.dt)
                ;
        }

        template <class Archive>
        void serialize(Archive & ar, shyft::core::time_zone::tz_info_t &o, const unsigned int version) {
            ar
                &  make_nvp("base_tz", o.base_tz)
                &  make_nvp("tz_table", o.tz)
                ;
        }

        template <class Archive>
        void serialize(Archive & ar, shyft::core::calendar &o, const unsigned int version) {
            ar
                & make_nvp("tz_info", o.tz_info)
                ;
        }


        /* time-axis serialization (the most popular ones) */
        template <class Archive>
        void serialize(Archive & ar, shyft::time_axis::fixed_dt &o, const unsigned int version) {
            ar
                &  make_nvp("t", o.t)
                &  make_nvp("dt", o.dt)
                &  make_nvp("n", o.n)
                ;
        }

        template <class Archive>
        void serialize(Archive & ar, shyft::time_axis::calendar_dt &o, const unsigned int version) {
            ar
                &  make_nvp("calendar", o.cal)
                &  make_nvp("t", o.t)
                &  make_nvp("dt", o.dt)
                &  make_nvp("n", o.n)
                ;
        }


        template <class Archive>
        void serialize(Archive & ar, shyft::time_axis::point_dt &o, const unsigned int version) {
            ar
                & make_nvp("t", o.t)
                & make_nvp("t_end", o.t_end)
                ;
        }


        template <class Archive>
        void serialize(Archive & ar, shyft::time_axis::generic_dt &o, const unsigned int version) {
            ar
                & make_nvp("gt", o.gt)
                ;
            if (o.gt == shyft::time_axis::generic_dt::FIXED) ar & make_nvp("f", o.f);
            else if (o.gt == shyft::time_axis::generic_dt::CALENDAR) ar & make_nvp("c", o.c);
            else ar & make_nvp("p", o.p);

        }


        /* core time-series serialization */

        template <class Archive, class TA>
        void serialize(Archive & ar, shyft::timeseries::point_ts<TA> &o, const unsigned int version) {
            ar
                & make_nvp("time_axis", o.ta)
                & make_nvp("fx_policy", o.fx_policy)
                & make_nvp("values", o.v)
                ;
        }

        template <class Archive, class TS>
        void serialize(Archive & ar, shyft::timeseries::ref_ts<TS> &o, const unsigned int version) {
            ar
                & make_nvp("ref", o.ref)
                & make_nvp("fx_policy", o.fx_policy)
                & make_nvp("ts", o.ts)
                ;
        }


        template <class Archive, class Ts>
        void serialize(Archive & ar, shyft::timeseries::time_shift_ts<Ts> &o, const unsigned int version) {
            ar
                & make_nvp("ts", o.ts)
                & make_nvp("ta", o.ta)
                & make_nvp("fx_policy", o.fx_policy)
                & make_nvp("dt", o.dt)
                ;
        }

        template <class Archive, class Ts, class Ta>
        void serialize(Archive & ar, shyft::timeseries::average_ts<Ts, Ta> &o, const unsigned int version) {
            ar
                & make_nvp("ts", o.ts)
                & make_nvp("ta", o.ta)
                & make_nvp("fx_policy", o.fx_policy)
                ;
        }

        template <class Archive, class Ts, class Ta>
        void serialize(Archive & ar, shyft::timeseries::accumulate_ts<Ts, Ta> &o, const unsigned int version) {
            ar
                & make_nvp("ts", o.ts)
                & make_nvp("ta", o.ta)
                & make_nvp("fx_policy", o.fx_policy)
                ;
        }

        template <class Archive>
        void serialize(Archive & ar, shyft::timeseries::profile_description &o, const unsigned int version) {
            ar
                & make_nvp("t0", o.t0)
                & make_nvp("dt", o.dt)
                & make_nvp("profile", o.profile)
                ;
        }

        template <class Archive, class TA>
        void serialize(Archive & ar, shyft::timeseries::profile_accessor<TA> &o, const unsigned int version) {
            ar
                & make_nvp("ta", o.ta)
                & make_nvp("profile", o.profile)
                & make_nvp("fx_policy", o.fx_policy)
                ;
        }

        template <class Archive, class TA>
        void serialize(Archive & ar, shyft::timeseries::periodic_ts<TA> &o, const unsigned int version) {
            ar
                & make_nvp("ta", o.ta)
                & make_nvp("pa", o.pa)
                & make_nvp("fx_policy", o.fx_policy)
                ;
        }

        template <class Archive, class TS_A, class TS_B>
        void serialize(Archive & ar, shyft::timeseries::glacier_melt_ts<TS_A, TS_B> &o, const unsigned int version) {
            ar
                & make_nvp("temperature", o.temperature)
                & make_nvp("sca_m2", o.sca_m2)
                & make_nvp("glacier_area_m2", o.glacier_area_m2)
                & make_nvp("dtf", o.dtf)
                & make_nvp("fx_policy", o.fx_policy)
                ;
        }

        template<class Archive, class A, class B, class O, class TA>
        void serialize(Archive & ar, shyft::timeseries::bin_op<A, B, O, TA> &o, const unsigned int version) {
            ar
                //& make_nvp("op",o.op) // not needed yet, needed when op starts to carry data
                & make_nvp("lhs", o.lhs)
                & make_nvp("rhs", o.rhs)
                & make_nvp("ta", o.ta)
                & make_nvp("fx_policy", o.fx_policy)
                ;
        }

        /* api time-series serialization (dyn-dispatch) */
        template <class Archive>
        void serialize(Archive & ar, shyft::api::ipoint_ts&, const unsigned) {
        }

        template<class Archive>
        void serialize(Archive & ar, shyft::api::gpoint_ts &o, const unsigned int version) {
            void_cast_register<shyft::api::gpoint_ts, shyft::api::ipoint_ts>();
            base_object<shyft::api::ipoint_ts>(o);
            ar
                & make_nvp("rep", o.rep)
                ;
        }


        template<class Archive>
        void serialize(Archive & ar, shyft::api::aref_ts &o, const unsigned int version) {
            void_cast_register<shyft::api::aref_ts, shyft::api::ipoint_ts>();
            base_object<shyft::api::ipoint_ts>(o);
            ar
                & make_nvp("rep", o.rep)
                ;
        }

        template<class Archive>
        void serialize(Archive & ar, shyft::api::average_ts &o, const unsigned int version) {
            void_cast_register<shyft::api::average_ts, shyft::api::ipoint_ts>();
            base_object<shyft::api::ipoint_ts>(o);
            ar
                & make_nvp("ta", o.ta)
                & make_nvp("ts", o.ts)
                ;
        }

        template<class Archive>
        void serialize(Archive & ar, shyft::api::accumulate_ts &o, const unsigned int version) {
            void_cast_register<shyft::api::accumulate_ts, shyft::api::ipoint_ts>();
            base_object<shyft::api::ipoint_ts>(o);
            ar
                & make_nvp("ta", o.ta)
                & make_nvp("ts", o.ts)
                ;
        }

        template<class Archive>
        void serialize(Archive & ar, shyft::api::time_shift_ts &o, const unsigned int version) {
            void_cast_register<shyft::api::time_shift_ts, shyft::api::ipoint_ts>();
            base_object<shyft::api::ipoint_ts>(o);
            ar
                & make_nvp("ta", o.ta)
                & make_nvp("ts", o.ts)
                & make_nvp("dt", o.dt)
                ;
        }

        template<class Archive>
        void serialize(Archive & ar, shyft::api::periodic_ts &o, const unsigned int version) {
            void_cast_register<shyft::api::periodic_ts, shyft::api::ipoint_ts>();
            base_object<shyft::api::ipoint_ts>(o);
            ar
                & make_nvp("ts", o.ts)
                ;
        }

        template<class Archive>
        void serialize(Archive & ar, shyft::api::abin_op_ts &o, const unsigned int version) {
            void_cast_register<shyft::api::abin_op_ts, shyft::api::ipoint_ts>();
            base_object<shyft::api::ipoint_ts>(o);
            ar
                & make_nvp("lhs", o.lhs)
                & make_nvp("op", o.op)
                & make_nvp("rhs", o.rhs)
                & make_nvp("ta", o.ta)
                & make_nvp("fx_policy", o.fx_policy)
                ;
        }

        template<class Archive>
        void serialize(Archive & ar, shyft::api::abin_op_scalar_ts &o, const unsigned int version) {
            void_cast_register<shyft::api::abin_op_scalar_ts, shyft::api::ipoint_ts>();
            base_object<shyft::api::ipoint_ts>(o);
            ar
                & make_nvp("lhs", o.lhs)
                & make_nvp("op", o.op)
                & make_nvp("rhs", o.rhs)
                & make_nvp("ta", o.ta)
                & make_nvp("fx_policy", o.fx_policy)
                ;
        }

        template<class Archive>
        void serialize(Archive & ar, shyft::api::abin_op_ts_scalar &o, const unsigned int version) {
            void_cast_register<shyft::api::abin_op_ts_scalar, shyft::api::ipoint_ts>();
            base_object<shyft::api::ipoint_ts>(o);
            ar
                & make_nvp("lhs", o.lhs)
                & make_nvp("op", o.op)
                & make_nvp("rhs", o.rhs)
                & make_nvp("ta", o.ta)
                & make_nvp("fx_policy", o.fx_policy)
                ;
        }


        template<class Archive>
        void serialize(Archive & ar, shyft::api::apoint_ts &o, const unsigned int version) {
            ar
                & make_nvp("ts", o.ts)
                ;
        }
    }
}

