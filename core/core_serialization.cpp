#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wsign-compare"
#endif

#include "core_pch.h"

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>

//
// 1. first include std stuff and the headers for
// files with serializeation support
//

#include "utctime_utilities.h"
#include "time_axis.h"
#include "time_series.h"
#include "geo_cell_data.h"
#include "hbv_snow.h"
#include "hbv_soil.h"
#include "hbv_tank.h"
#include "gamma_snow.h"
#include "kirchner.h"
#include "skaugen.h"
#include "hbv_stack.h"
#include "pt_gs_k.h"
#include "pt_ss_k.h"
#include "pt_hs_k.h"
#include "time_series_info.h"
#include "predictions.h"
// then include stuff you need like vector,shared, base_obj,nvp etc.



#include <dlib/serialize.h>

//
// 2. Then implement each class serialization support
//

using namespace boost::serialization;

namespace shyft {
    namespace dtss {
        // later relocate to core/dtss.cpp when created
        std::string shyft_prefix{"shyft://"};
    }
}
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
void shyft::time_series::point_ts<Ta>::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("time_axis", ta)
    & make_nvp("fx_policy", fx_policy)
    & make_nvp("values", v)
    ;
}
template <class TS>
template <class Archive>
void shyft::time_series::ref_ts<TS>::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("ref", ref)
    & make_nvp("ts", ts)
    ;
}

template <class Ts>
template <class Archive>
void shyft::time_series::time_shift_ts<Ts>::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("ts", ts)
    & make_nvp("ta", ta)
    & make_nvp("fx_policy", fx_policy)
    & make_nvp("dt", dt)
    & make_nvp("bound",bound)
    ;
}
template <class Ts, class Ta>
template <class Archive>
void shyft::time_series::average_ts<Ts, Ta>::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("ts", ts)
    & make_nvp("ta", ta)
    & make_nvp("fx_policy", fx_policy)
    ;
}

template <class Ts, class Ta>
template <class Archive>
void shyft::time_series::accumulate_ts<Ts, Ta>::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("ts", ts)
    & make_nvp("ta", ta)
    & make_nvp("fx_policy", fx_policy)
    ;
}

template <class Archive>
void shyft::time_series::profile_description::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("t0", t0)
    & make_nvp("dt", dt)
    & make_nvp("profile", profile)
    ;
}
template <class TA>
template <class Archive>
void shyft::time_series::profile_accessor<TA>::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("ta", ta)
    & make_nvp("profile", profile)
    & make_nvp("fx_policy", fx_policy)
    ;
}

template <class TA>
template <class Archive>
void shyft::time_series::periodic_ts<TA>::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("ta", ta)
    & make_nvp("pa", pa)
    & make_nvp("fx_policy", fx_policy)
    ;
}
template <class TS_A, class TS_B>
template <class Archive>
void shyft::time_series::glacier_melt_ts<TS_A, TS_B>::serialize(Archive & ar, const unsigned int version) {
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
void shyft::time_series::convolve_w_ts<Ts>::serialize(Archive & ar, const unsigned int version) {
	bool b = bound;
    ar
		& make_nvp("ts", ts)
		& make_nvp("fx_policy", fx_policy)
		& make_nvp("w", w)
		& make_nvp("convolve_policy",policy)
		& make_nvp("bound", b)
    ;
	bound = b;
}

template <class A, class B, class O, class TA>
template<class Archive>
void shyft::time_series::bin_op<A, B, O, TA>::serialize(Archive & ar, const unsigned int version) {
    bool bd=bind_done;
    ar
    //& make_nvp("op",o.op) // not needed yet, needed when op starts to carry data
    & make_nvp("lhs", lhs)
    & make_nvp("rhs", rhs)
    & make_nvp("ta", ta)
    & make_nvp("fx_policy", fx_policy)
    & make_nvp("bind_done",bind_done)
    ;
    bind_done = bd;
}

template<class Archive>
void shyft::time_series::rating_curve_segment::serialize(Archive & ar, const unsigned int version) {
	ar
		& make_nvp("lower", lower)
		& make_nvp("a", a)
		& make_nvp("b", b)
		& make_nvp("c", c)
		;
}

template<class Archive>
void shyft::time_series::rating_curve_function::serialize(Archive & ar, const unsigned int version) {
	ar
		& make_nvp("segments", segments)
		;
}

template<class Archive>
void shyft::time_series::rating_curve_parameters::serialize(Archive & ar, const unsigned int version) {
	ar
		& make_nvp("curves", curves)
		;
}

template <class TS>
template<class Archive>
void shyft::time_series::rating_curve_ts<TS>::serialize(Archive & ar, const unsigned int version) {
	bool bd = bound;
	ar
		& make_nvp("level_ts", level_ts)
		& make_nvp("rc_param", rc_param)
		& make_nvp("fx_policy", fx_policy)
		& make_nvp("bound", bd)
		;
	bound = bd;
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
//-- state serialization
template <class Archive>
void shyft::core::hbv_snow::state::serialize(Archive & ar, const unsigned int file_version) {
    ar
    & make_nvp("swe",swe)
    & make_nvp("sca",sca)
    ;
}
template <class Archive>
void shyft::core::hbv_soil::state::serialize(Archive & ar, const unsigned int file_version) {
    ar
    & make_nvp("sm",sm)
    ;
}
template <class Archive>
void shyft::core::hbv_tank::state::serialize(Archive & ar, const unsigned int file_version) {
    ar
    & make_nvp("uz",uz)
    & make_nvp("lz",lz)
    ;
}
template <class Archive>
void shyft::core::kirchner::state::serialize(Archive & ar, const unsigned int file_version) {
    ar
        & make_nvp("q",q)
        ;
}
template <class Archive>
void shyft::core::gamma_snow::state::serialize(Archive & ar, const unsigned int file_version) {
    ar
        & make_nvp("albedo", albedo)
        & make_nvp("lwc", lwc)
        & make_nvp("surface_heat",surface_heat )
        & make_nvp("alpha", alpha)
        & make_nvp("sdc_melt_mean",sdc_melt_mean )
        & make_nvp("acc_melt",acc_melt )
        & make_nvp("iso_pot_energy", iso_pot_energy)
        & make_nvp("temp_swe",temp_swe )
        ;
}
template <class Archive>
void shyft::core::skaugen::state::serialize(Archive & ar, const unsigned int file_version) {
    ar
        & make_nvp("nu",nu)
        & make_nvp("alpha",alpha)
        & make_nvp("sca",sca)
        & make_nvp("swe",swe)
        & make_nvp("free_water",free_water)
        & make_nvp("residual",residual)
        & make_nvp("num_units",num_units)
        ;
}
template <class Archive>
void shyft::core::hbv_stack::state::serialize(Archive & ar, const unsigned int file_version) {
    ar
        & make_nvp("snow",snow)
        & make_nvp("soil",soil)
        & make_nvp("tank",tank)
        ;
}
template <class Archive>
void shyft::core::pt_gs_k::state::serialize(Archive & ar, const unsigned int file_version) {
    ar
        & make_nvp("gs",gs)
        & make_nvp("kirchner",kirchner)
        ;
}
template <class Archive>
void shyft::core::pt_ss_k::state::serialize(Archive & ar, const unsigned int file_version) {
    ar
        & make_nvp("snow",snow)
        & make_nvp("kirchner", kirchner)
        ;
}
template <class Archive>
void shyft::core::pt_hs_k::state::serialize(Archive & ar, const unsigned int file_version) {
    ar
        & make_nvp("snow",snow)
        & make_nvp("kirchner", kirchner)
        ;
}

template <class Archive>
void shyft::dtss::ts_info::serialize(Archive& ar, const unsigned int file_version) {
    ar
        & make_nvp("name", name)
        & make_nvp("point_fx", point_fx)
        & make_nvp("delta_t",delta_t)
        & make_nvp("olson_tz_id",olson_tz_id)
        & make_nvp("data_period",data_period)
        & make_nvp("created",created)
        & make_nvp("modified",modified)
        ;
}

//-- predictor serialization
template < typename A, typename K >
inline void serialize_helper(A & ar, const dlib::krls<K> & krls) {

    using namespace dlib;

    std::ostringstream ostream{ std::ios_base::out | std::ios_base::binary };
    serialize(krls, ostream);
    auto blob = ostream.str();
    ar & make_nvp("krls_data", blob);
}

template < typename A, typename K >
inline void deserialize_helper(A & ar, dlib::krls<K> & krls) {

    using namespace dlib;

    std::string tmp;
    ar & make_nvp("krls_data", tmp);
    std::istringstream istream{ tmp, std::ios_base::in | std::ios_base::binary };
    deserialize(krls, istream);
}

template <class Archive>
void shyft::prediction::krls_rbf_predictor::serialize(Archive& ar, const unsigned int file_version) {
    ar & make_nvp("_dt", _dt) &make_nvp("train_point_fx",train_point_fx);

    if ( Archive::is_saving::value ) {
        serialize_helper(ar, this->_krls);
    } else {
        deserialize_helper(ar, this->_krls);
    }
}

//-- export dtss stuff
x_serialize_implement(shyft::dtss::ts_info);

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

x_serialize_implement(shyft::time_series::point_ts<shyft::time_axis::fixed_dt>);
x_serialize_implement(shyft::time_series::point_ts<shyft::time_axis::calendar_dt>);
x_serialize_implement(shyft::time_series::point_ts<shyft::time_axis::point_dt>);
x_serialize_implement(shyft::time_series::point_ts<shyft::time_axis::generic_dt>);

x_serialize_implement(shyft::time_series::ref_ts<shyft::time_series::point_ts<shyft::time_axis::fixed_dt>>);
x_serialize_implement(shyft::time_series::ref_ts<shyft::time_series::point_ts<shyft::time_axis::calendar_dt>>);
x_serialize_implement(shyft::time_series::ref_ts<shyft::time_series::point_ts<shyft::time_axis::point_dt>>);
x_serialize_implement(shyft::time_series::ref_ts<shyft::time_series::point_ts<shyft::time_axis::generic_dt>>);

x_serialize_implement(shyft::time_series::profile_description);
x_serialize_implement(shyft::time_series::profile_accessor<shyft::time_axis::fixed_dt>);
x_serialize_implement(shyft::time_series::profile_accessor<shyft::time_axis::calendar_dt>);
x_serialize_implement(shyft::time_series::profile_accessor<shyft::time_axis::point_dt>);
x_serialize_implement(shyft::time_series::profile_accessor<shyft::time_axis::generic_dt>);

x_serialize_implement(shyft::time_series::convolve_w_ts<shyft::time_series::point_ts<shyft::time_axis::fixed_dt>>);
x_serialize_implement(shyft::time_series::convolve_w_ts<shyft::time_series::point_ts<shyft::time_axis::generic_dt>>);


x_serialize_implement(shyft::time_series::periodic_ts<shyft::time_axis::fixed_dt>);
x_serialize_implement(shyft::time_series::periodic_ts<shyft::time_axis::calendar_dt>);
x_serialize_implement(shyft::time_series::periodic_ts<shyft::time_axis::point_dt>);
x_serialize_implement(shyft::time_series::periodic_ts<shyft::time_axis::generic_dt>);

x_serialize_implement(shyft::time_series::rating_curve_segment);
x_serialize_implement(shyft::time_series::rating_curve_function);
x_serialize_implement(shyft::time_series::rating_curve_parameters);
x_serialize_implement(shyft::time_series::rating_curve_ts<shyft::time_series::point_ts<shyft::time_axis::fixed_dt>>);
x_serialize_implement(shyft::time_series::rating_curve_ts<shyft::time_series::point_ts<shyft::time_axis::calendar_dt>>);
x_serialize_implement(shyft::time_series::rating_curve_ts<shyft::time_series::point_ts<shyft::time_axis::point_dt>>);
x_serialize_implement(shyft::time_series::rating_curve_ts<shyft::time_series::point_ts<shyft::time_axis::generic_dt>>);

//-- export method and method-stack state
x_serialize_implement(shyft::core::hbv_snow::state);
x_serialize_implement(shyft::core::hbv_soil::state);
x_serialize_implement(shyft::core::hbv_tank::state);
x_serialize_implement(shyft::core::gamma_snow::state);
x_serialize_implement(shyft::core::skaugen::state);
x_serialize_implement(shyft::core::kirchner::state);

x_serialize_implement(shyft::core::pt_gs_k::state);
x_serialize_implement(shyft::core::pt_hs_k::state);
x_serialize_implement(shyft::core::pt_ss_k::state);
x_serialize_implement(shyft::core::hbv_stack::state);

//-- export predictors
x_serialize_implement(shyft::prediction::krls_rbf_predictor);

//
// 4. Then include the archive supported
//
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

// repeat template instance for each archive class
#define x_arch(T) x_serialize_archive(T,binary_oarchive,binary_iarchive)

x_arch(shyft::dtss::ts_info);

x_arch(shyft::core::utcperiod);
x_arch(shyft::core::time_zone::tz_info_t);
x_arch(shyft::core::time_zone::tz_table);
x_arch(shyft::core::calendar);

x_arch(shyft::time_axis::fixed_dt);
x_arch(shyft::time_axis::calendar_dt);
x_arch(shyft::time_axis::point_dt);
x_arch(shyft::time_axis::generic_dt);

x_arch(shyft::time_series::point_ts<shyft::time_axis::fixed_dt>);
x_arch(shyft::time_series::point_ts<shyft::time_axis::calendar_dt>);
x_arch(shyft::time_series::point_ts<shyft::time_axis::point_dt>);
x_arch(shyft::time_series::point_ts<shyft::time_axis::generic_dt>);

x_arch(shyft::time_series::convolve_w_ts<shyft::time_series::point_ts<shyft::time_axis::fixed_dt>>);
x_arch(shyft::time_series::convolve_w_ts<shyft::time_series::point_ts<shyft::time_axis::generic_dt>>);

x_arch(shyft::time_series::ref_ts<shyft::time_series::point_ts<shyft::time_axis::fixed_dt>>);
x_arch(shyft::time_series::ref_ts<shyft::time_series::point_ts<shyft::time_axis::calendar_dt>>);
x_arch(shyft::time_series::ref_ts<shyft::time_series::point_ts<shyft::time_axis::point_dt>>);
x_arch(shyft::time_series::ref_ts<shyft::time_series::point_ts<shyft::time_axis::generic_dt>>);

x_arch(shyft::time_series::profile_description);
x_arch(shyft::time_series::profile_accessor<shyft::time_axis::fixed_dt>);
x_arch(shyft::time_series::profile_accessor<shyft::time_axis::calendar_dt>);
x_arch(shyft::time_series::profile_accessor<shyft::time_axis::point_dt>);
x_arch(shyft::time_series::profile_accessor<shyft::time_axis::generic_dt>);
x_arch(shyft::time_series::periodic_ts<shyft::time_axis::fixed_dt>);
x_arch(shyft::time_series::periodic_ts<shyft::time_axis::calendar_dt>);
x_arch(shyft::time_series::periodic_ts<shyft::time_axis::point_dt>);
x_arch(shyft::time_series::periodic_ts<shyft::time_axis::generic_dt>);

x_arch(shyft::time_series::rating_curve_segment);
x_arch(shyft::time_series::rating_curve_function);
x_arch(shyft::time_series::rating_curve_parameters);
x_arch(shyft::time_series::rating_curve_ts<shyft::time_series::point_ts<shyft::time_axis::fixed_dt>>);
x_arch(shyft::time_series::rating_curve_ts<shyft::time_series::point_ts<shyft::time_axis::calendar_dt>>);
x_arch(shyft::time_series::rating_curve_ts<shyft::time_series::point_ts<shyft::time_axis::point_dt>>);
x_arch(shyft::time_series::rating_curve_ts<shyft::time_series::point_ts<shyft::time_axis::generic_dt>>);

x_arch(shyft::core::geo_point);
x_arch(shyft::core::land_type_fractions);
x_arch(shyft::core::routing_info);
x_arch(shyft::core::geo_cell_data);

x_arch(shyft::core::hbv_snow::state);
x_arch(shyft::core::hbv_soil::state);
x_arch(shyft::core::hbv_tank::state);
x_arch(shyft::core::gamma_snow::state);
x_arch(shyft::core::skaugen::state);
x_arch(shyft::core::kirchner::state);

x_arch(shyft::core::pt_gs_k::state);
x_arch(shyft::core::pt_hs_k::state);
x_arch(shyft::core::pt_ss_k::state);
x_arch(shyft::core::hbv_stack::state);

x_arch(shyft::prediction::krls_rbf_predictor);
