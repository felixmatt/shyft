#include "core_pch.h"

//
// 1. first include std stuff and the headers for
// files with serializeation support
//

#include "utctime_utilities.h"
#include "time_axis.h"
#include "timeseries.h"
#include "geo_cell_data.h"

// then include stuff you need like vector,shared, base_obj,nvp etc.

#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>

//
// 2. Then implement each class serialization support
//

using namespace boost::serialization;

//-- utctime_utilities.h

template<class Archive>
void shyft::core::utcperiod::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("start", start)
    & make_nvp("end",end);
    ;
}

template<class Archive>
void shyft::core::time_zone::tz_table::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("start_year", start_year)
    & make_nvp("tz_name",tz_name)
    & make_nvp("dst",dst)
    & make_nvp("dt",dt)
    ;
}

template<class Archive>
void shyft::core::time_zone::tz_info_t::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("base_tz", base_tz)
    & make_nvp("tz",tz);
    ;
}

template <class Archive>
void shyft::core::calendar::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("tz_info", tz_info)
    ;
}

//-- time_axis.h

template <class Archive>
void shyft::time_axis::fixed_dt::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("t", t)
    & make_nvp("dt",dt)
    & make_nvp("n",n)
    ;
}

template <class Archive>
void shyft::time_axis::calendar_dt::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("cal", cal)
    & make_nvp("t", t)
    & make_nvp("dt",dt)
    & make_nvp("n",n)
    ;
}

template <class Archive>
void shyft::time_axis::point_dt::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("t", t)
    & make_nvp("dt",t_end)
    ;
}

template <class Archive>
void shyft::time_axis::generic_dt::serialize(Archive & ar,const unsigned int version) {
    ar
    & make_nvp("gt", gt)
    ;
    if (gt == shyft::time_axis::generic_dt::FIXED)
        ar & make_nvp("f", f);
    else if (gt == shyft::time_axis::generic_dt::CALENDAR)
        ar & make_nvp("c", c);
    else
        ar & make_nvp("p", p);
}

//-- time-series serialization
template <class Ta>
template <class Archive>
void shyft::timeseries::point_ts<Ta>::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("time_axis", ta)
    & make_nvp("fx_policy", fx_policy)
    & make_nvp("values", v)
    ;
}
template <class TS>
template <class Archive>
void shyft::timeseries::ref_ts<TS>::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("ref", ref)
    & make_nvp("fx_policy", fx_policy)
    & make_nvp("ts", ts)
    ;
}

template <class Ts>
template <class Archive>
void shyft::timeseries::time_shift_ts<Ts>::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("ts", ts)
    & make_nvp("ta", ta)
    & make_nvp("fx_policy", fx_policy)
    & make_nvp("dt", dt)
    ;
}
template <class Ts, class Ta>
template <class Archive>
void shyft::timeseries::average_ts<Ts, Ta>::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("ts", ts)
    & make_nvp("ta", ta)
    & make_nvp("fx_policy", fx_policy)
    ;
}

template <class Ts, class Ta>
template <class Archive>
void shyft::timeseries::accumulate_ts<Ts, Ta>::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("ts", ts)
    & make_nvp("ta", ta)
    & make_nvp("fx_policy", fx_policy)
    ;
}

template <class Archive>
void shyft::timeseries::profile_description::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("t0", t0)
    & make_nvp("dt", dt)
    & make_nvp("profile", profile)
    ;
}
template <class TA>
template <class Archive>
void shyft::timeseries::profile_accessor<TA>::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("ta", ta)
    & make_nvp("profile", profile)
    & make_nvp("fx_policy", fx_policy)
    ;
}

template <class TA>
template <class Archive>
void shyft::timeseries::periodic_ts<TA>::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("ta", ta)
    & make_nvp("pa", pa)
    & make_nvp("fx_policy", fx_policy)
    ;
}
template <class TS_A, class TS_B>
template <class Archive>
void shyft::timeseries::glacier_melt_ts<TS_A, TS_B>::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("temperature", temperature)
    & make_nvp("sca_m2", sca_m2)
    & make_nvp("glacier_area_m2", glacier_area_m2)
    & make_nvp("dtf", dtf)
    & make_nvp("fx_policy", fx_policy)
    ;
}

template <class Ts>
template <class Archive>
void shyft::timeseries::convolve_w_ts<Ts>::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("ts", ts)
    & make_nvp("fx_policy", fx_policy)
    & make_nvp("w", w)
    & make_nvp("convolve_policy",policy)
    ;
}

template <class A, class B, class O, class TA>
template<class Archive>
void shyft::timeseries::bin_op<A, B, O, TA>::serialize(Archive & ar, const unsigned int version) {
    ar
    //& make_nvp("op",o.op) // not needed yet, needed when op starts to carry data
    & make_nvp("lhs", lhs)
    & make_nvp("rhs", rhs)
    & make_nvp("ta", ta)
    & make_nvp("fx_policy", fx_policy)
    ;
}

//-- basic geo stuff
template <class Archive>
void shyft::core::geo_point::serialize(Archive& ar, const unsigned int version) {
    ar
    & make_nvp("x",x)
    & make_nvp("y",y)
    & make_nvp("z",z)
    ;
}

template <class Archive>
void shyft::core::land_type_fractions::serialize(Archive& ar, const unsigned int version) {
    ar
    & make_nvp("glacier_",glacier_)
    & make_nvp("lake_",lake_)
    & make_nvp("reservoir_",reservoir_)
    & make_nvp("forest_",forest_)
    ;
}
template <class Archive>
void shyft::core::routing_info::serialize(Archive& ar, const unsigned int version) {
    ar
    & make_nvp("id", id)
    & make_nvp("distance", distance)
    ;
}

template <class Archive>
void shyft::core::geo_cell_data::serialize(Archive& ar, const unsigned int version) {
    ar
    & make_nvp("mid_point_",mid_point_)
    & make_nvp("area_m2",area_m2)
    & make_nvp("catchment_id_",catchment_id_)
    & make_nvp("radiation_slope_factor_",radiation_slope_factor_)
    & make_nvp("fractions",fractions)
    & make_nvp("routing",routing)
    ;
}

//-- export geo stuff
x_serialize_implement(shyft::core::geo_point);
x_serialize_implement(shyft::core::land_type_fractions);
x_serialize_implement(shyft::core::routing_info);
x_serialize_implement(shyft::core::geo_cell_data);

//-- export utctime_utilities
x_serialize_implement(shyft::core::utcperiod);
x_serialize_implement(shyft::core::time_zone::tz_info_t);
x_serialize_implement(shyft::core::time_zone::tz_table);
x_serialize_implement(shyft::core::calendar);

//-- export time-axis
x_serialize_implement(shyft::time_axis::fixed_dt);
x_serialize_implement(shyft::time_axis::calendar_dt);
x_serialize_implement(shyft::time_axis::point_dt);
x_serialize_implement(shyft::time_axis::generic_dt);

//-- export core time-series (except binary-ops)

x_serialize_implement(shyft::timeseries::point_ts<shyft::time_axis::fixed_dt>);
x_serialize_implement(shyft::timeseries::point_ts<shyft::time_axis::calendar_dt>);
x_serialize_implement(shyft::timeseries::point_ts<shyft::time_axis::point_dt>);
x_serialize_implement(shyft::timeseries::point_ts<shyft::time_axis::generic_dt>);

x_serialize_implement(shyft::timeseries::ref_ts<shyft::timeseries::point_ts<shyft::time_axis::fixed_dt>>);
x_serialize_implement(shyft::timeseries::ref_ts<shyft::timeseries::point_ts<shyft::time_axis::calendar_dt>>);
x_serialize_implement(shyft::timeseries::ref_ts<shyft::timeseries::point_ts<shyft::time_axis::point_dt>>);
x_serialize_implement(shyft::timeseries::ref_ts<shyft::timeseries::point_ts<shyft::time_axis::generic_dt>>);

x_serialize_implement(shyft::timeseries::profile_description);
x_serialize_implement(shyft::timeseries::profile_accessor<shyft::time_axis::fixed_dt>);
x_serialize_implement(shyft::timeseries::profile_accessor<shyft::time_axis::calendar_dt>);
x_serialize_implement(shyft::timeseries::profile_accessor<shyft::time_axis::point_dt>);
x_serialize_implement(shyft::timeseries::profile_accessor<shyft::time_axis::generic_dt>);

x_serialize_implement(shyft::timeseries::convolve_w_ts<shyft::timeseries::point_ts<shyft::time_axis::fixed_dt>>);
x_serialize_implement(shyft::timeseries::convolve_w_ts<shyft::timeseries::point_ts<shyft::time_axis::generic_dt>>);


x_serialize_implement(shyft::timeseries::periodic_ts<shyft::time_axis::fixed_dt>);
x_serialize_implement(shyft::timeseries::periodic_ts<shyft::time_axis::calendar_dt>);
x_serialize_implement(shyft::timeseries::periodic_ts<shyft::time_axis::point_dt>);
x_serialize_implement(shyft::timeseries::periodic_ts<shyft::time_axis::generic_dt>);

//
// 4. Then include the archive supported
//
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

// repeat template instance for each archive class
#define x_arch(T) x_serialize_archive(T,binary_oarchive,binary_iarchive)
x_arch(shyft::core::utcperiod);
x_arch(shyft::core::time_zone::tz_info_t);
x_arch(shyft::core::time_zone::tz_table);
x_arch(shyft::core::calendar);

x_arch(shyft::time_axis::fixed_dt);
x_arch(shyft::time_axis::calendar_dt);
x_arch(shyft::time_axis::point_dt);
x_arch(shyft::time_axis::generic_dt);

x_arch(shyft::timeseries::point_ts<shyft::time_axis::fixed_dt>);
x_arch(shyft::timeseries::point_ts<shyft::time_axis::calendar_dt>);
x_arch(shyft::timeseries::point_ts<shyft::time_axis::point_dt>);
x_arch(shyft::timeseries::point_ts<shyft::time_axis::generic_dt>);

x_arch(shyft::timeseries::convolve_w_ts<shyft::timeseries::point_ts<shyft::time_axis::fixed_dt>>);
x_arch(shyft::timeseries::convolve_w_ts<shyft::timeseries::point_ts<shyft::time_axis::generic_dt>>);

x_arch(shyft::timeseries::ref_ts<shyft::timeseries::point_ts<shyft::time_axis::fixed_dt>>);
x_arch(shyft::timeseries::ref_ts<shyft::timeseries::point_ts<shyft::time_axis::calendar_dt>>);
x_arch(shyft::timeseries::ref_ts<shyft::timeseries::point_ts<shyft::time_axis::point_dt>>);
x_arch(shyft::timeseries::ref_ts<shyft::timeseries::point_ts<shyft::time_axis::generic_dt>>);

x_arch(shyft::timeseries::profile_description);
x_arch(shyft::timeseries::profile_accessor<shyft::time_axis::fixed_dt>);
x_arch(shyft::timeseries::profile_accessor<shyft::time_axis::calendar_dt>);
x_arch(shyft::timeseries::profile_accessor<shyft::time_axis::point_dt>);
x_arch(shyft::timeseries::profile_accessor<shyft::time_axis::generic_dt>);
x_arch(shyft::timeseries::periodic_ts<shyft::time_axis::fixed_dt>);
x_arch(shyft::timeseries::periodic_ts<shyft::time_axis::calendar_dt>);
x_arch(shyft::timeseries::periodic_ts<shyft::time_axis::point_dt>);
x_arch(shyft::timeseries::periodic_ts<shyft::time_axis::generic_dt>);

x_arch(shyft::core::geo_point);
x_arch(shyft::core::land_type_fractions);
x_arch(shyft::core::routing_info);
x_arch(shyft::core::geo_cell_data);
