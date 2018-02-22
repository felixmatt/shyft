#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wsign-compare"
#endif

#include "core_serialization.h"
#include "core_archive.h"

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/base_object.hpp>

//
// 1. first include std stuff and the headers for
// files with serializeation support
//

#include "utctime_utilities.h"
#include "time_axis.h"
#include "time_series.h"
#include "time_series_dd.h"
#include "time_series_info.h"
#include "predictions.h"
#include "dtss_cache.h"

#include <dlib/serialize.h>

//
// 2. Then implement each class serialization support
//

using namespace boost::serialization;
using namespace shyft::core;


//-- time-series serialization
template <class Ta>
template <class Archive>
void shyft::time_series::point_ts<Ta>::serialize(Archive & ar, const unsigned int version) {
	ar
		& core_nvp("time_axis", ta)
		& core_nvp("fx_policy", fx_policy)
		& core_nvp("values", v)
		;
}
template <class TS>
template <class Archive>
void shyft::time_series::ref_ts<TS>::serialize(Archive & ar, const unsigned int version) {
	ar
		& core_nvp("ref", ref)
		& core_nvp("ts", ts)
		;
}

template <class Ts>
template <class Archive>
void shyft::time_series::time_shift_ts<Ts>::serialize(Archive & ar, const unsigned int version) {
	ar
		& core_nvp("ts", ts)
		& core_nvp("ta", ta)
		& core_nvp("fx_policy", fx_policy)
		& core_nvp("dt", dt)
		& core_nvp("bound", bound)
		;
}
template <class Ts, class Ta>
template <class Archive>
void shyft::time_series::average_ts<Ts, Ta>::serialize(Archive & ar, const unsigned int version) {
	ar
		& core_nvp("ts", ts)
		& core_nvp("ta", ta)
		& core_nvp("fx_policy", fx_policy)
		;
}

template <class Ts, class Ta>
template <class Archive>
void shyft::time_series::accumulate_ts<Ts, Ta>::serialize(Archive & ar, const unsigned int version) {
	ar
		& core_nvp("ts", ts)
		& core_nvp("ta", ta)
		& core_nvp("fx_policy", fx_policy)
		;
}

template <class Archive>
void shyft::time_series::profile_description::serialize(Archive & ar, const unsigned int version) {
	ar
		& core_nvp("t0", t0)
		& core_nvp("dt", dt)
		& core_nvp("profile", profile)
		;
}
template <class TA>
template <class Archive>
void shyft::time_series::profile_accessor<TA>::serialize(Archive & ar, const unsigned int version) {
	ar
		& core_nvp("ta", ta)
		& core_nvp("profile", profile)
		& core_nvp("fx_policy", fx_policy)
		;
}

template <class TA>
template <class Archive>
void shyft::time_series::periodic_ts<TA>::serialize(Archive & ar, const unsigned int version) {
	ar
		& core_nvp("ta", ta)
		& core_nvp("pa", pa)
		& core_nvp("fx_policy", fx_policy)
		;
}
template <class TS_A, class TS_B>
template <class Archive>
void shyft::time_series::glacier_melt_ts<TS_A, TS_B>::serialize(Archive & ar, const unsigned int version) {
	ar
		& core_nvp("temperature", temperature)
		& core_nvp("sca_m2", sca_m2)
		& core_nvp("glacier_area_m2", glacier_area_m2)
		& core_nvp("dtf", dtf)
		& core_nvp("fx_policy", fx_policy)
		;
}

template <class Ts>
template <class Archive>
void shyft::time_series::convolve_w_ts<Ts>::serialize(Archive & ar, const unsigned int version) {
	bool b = bound;
	ar
		& core_nvp("ts", ts)
		& core_nvp("fx_policy", fx_policy)
		& core_nvp("w", w)
		& core_nvp("convolve_policy", policy)
		& core_nvp("bound", b)
		;
	bound = b;
}

template <class A, class B, class O, class TA>
template<class Archive>
void shyft::time_series::bin_op<A, B, O, TA>::serialize(Archive & ar, const unsigned int version) {
	bool bd = bind_done;
	ar
		//& core_nvp("op",o.op) // not needed yet, needed when op starts to carry data
		& core_nvp("lhs", lhs)
		& core_nvp("rhs", rhs)
		& core_nvp("ta", ta)
		& core_nvp("fx_policy", fx_policy)
		& core_nvp("bind_done", bind_done)
		;
	bind_done = bd;
}

template<class Archive>
void shyft::time_series::rating_curve_segment::serialize(Archive & ar, const unsigned int version) {
	ar
		& core_nvp("lower", lower)
		& core_nvp("a", a)
		& core_nvp("b", b)
		& core_nvp("c", c)
		;
}

template<class Archive>
void shyft::time_series::rating_curve_function::serialize(Archive & ar, const unsigned int version) {
	ar
		& core_nvp("segments", segments)
		;
}

template<class Archive>
void shyft::time_series::rating_curve_parameters::serialize(Archive & ar, const unsigned int version) {
	ar
		& core_nvp("curves", curves)
		;
}

template <class TS>
template<class Archive>
void shyft::time_series::rating_curve_ts<TS>::serialize(Archive & ar, const unsigned int version) {
	bool bd = bound;
	ar
		& core_nvp("level_ts", level_ts)
		& core_nvp("rc_param", rc_param)
		& core_nvp("fx_policy", fx_policy)
		& core_nvp("bound", bd)
		;
	bound = bd;
}

template<class Archive>
void shyft::time_series::ice_packing_parameters::serialize(Archive & ar, const unsigned int version) {
    ar
        & core_nvp("window", window)
        & core_nvp("threshold_temp", threshold_temp)
        ;
}

template <class TS>
template<class Archive>
void shyft::time_series::ice_packing_ts<TS>::serialize(Archive & ar, const unsigned int version) {
    bool bd = bound;
    ar
        & core_nvp("temp_ts", temp_ts)
        & core_nvp("ip_param", ip_param)
        & core_nvp("ipt_policy", ipt_policy)
        & core_nvp("fx_policy", fx_policy)
        & core_nvp("bound", bd)
        ;
    bound = bd;
}

template <class Archive>
void shyft::dtss::ts_info::serialize(Archive& ar, const unsigned int file_version) {
	ar
		& core_nvp("name", name)
		& core_nvp("point_fx", point_fx)
		& core_nvp("delta_t", delta_t)
		& core_nvp("olson_tz_id", olson_tz_id)
		& core_nvp("data_period", data_period)
		& core_nvp("created", created)
		& core_nvp("modified", modified)
		;
}

//-- predictor serialization
template < typename A, typename K >
inline void serialize_helper(A & ar, const dlib::krls<K> & krls) {

	using namespace dlib;

	std::ostringstream ostream{ std::ios_base::out | std::ios_base::binary };
	serialize(krls, ostream);
	auto blob = ostream.str();
	ar & core_nvp("krls_data", blob);
}

template < typename A, typename K >
inline void deserialize_helper(A & ar, dlib::krls<K> & krls) {

	using namespace dlib;

	std::string tmp;
	ar & core_nvp("krls_data", tmp);
	std::istringstream istream{ tmp, std::ios_base::in | std::ios_base::binary };
	deserialize(krls, istream);
}

template <class Archive>
void shyft::prediction::krls_rbf_predictor::serialize(Archive& ar, const unsigned int file_version) {
	ar & core_nvp("_dt", _dt) &core_nvp("train_point_fx", train_point_fx);

	if (Archive::is_saving::value) {
		serialize_helper(ar, this->_krls);
	} else {
		deserialize_helper(ar, this->_krls);
	}
}
template <class Arcive>
void shyft::dtss::cache_stats::serialize(Arcive& ar, const unsigned int file_version) {
	ar
		& core_nvp("hits", hits)
		& core_nvp("misses", misses)
		& core_nvp("coverage_misses", coverage_misses)
		& core_nvp("id_count", id_count)
		& core_nvp("point_count", point_count)
		& core_nvp("fragment_count", fragment_count)
		;
}

/* api time-series serialization (dyn-dispatch) */

template <class Archive>
void shyft::time_series::dd::ipoint_ts::serialize(Archive & ar, const unsigned) {
}

template<class Archive>
void shyft::time_series::dd::gpoint_ts::serialize(Archive & ar, const unsigned int version) {
	ar
		& core_nvp("ipoint_ts", base_object<shyft::time_series::dd::ipoint_ts>(*this))
		& core_nvp("rep", rep)
		;
}


template<class Archive>
void shyft::time_series::dd::aref_ts::serialize(Archive & ar, const unsigned int version) {
	ar
		& core_nvp("ipoint_ts", base_object<shyft::time_series::dd::ipoint_ts>(*this))
		& core_nvp("id", id)
		& core_nvp("rep", rep)

		;
}

template<class Archive>
void shyft::time_series::dd::average_ts::serialize(Archive & ar, const unsigned int version) {
	ar
		& core_nvp("ipoint_ts", base_object<shyft::time_series::dd::ipoint_ts>(*this))
		& core_nvp("ta", ta)
		& core_nvp("ts", ts)
		;
}

template<class Archive>
void shyft::time_series::dd::integral_ts::serialize(Archive & ar, const unsigned int version) {
	ar
		& core_nvp("ipoint_ts", base_object<shyft::time_series::dd::ipoint_ts>(*this))
		& core_nvp("ta", ta)
		& core_nvp("ts", ts)
		;
}

template<class Archive>
void shyft::time_series::dd::accumulate_ts::serialize(Archive & ar, const unsigned int version) {
	ar
		& core_nvp("ipoint_ts", base_object<shyft::time_series::dd::ipoint_ts>(*this))
		& core_nvp("ta", ta)
		& core_nvp("ts", ts)
		;
}

template<class Archive>
void shyft::time_series::dd::time_shift_ts::serialize(Archive & ar, const unsigned int version) {
	ar
		& core_nvp("ipoint_ts", base_object<shyft::time_series::dd::ipoint_ts>(*this))
		& core_nvp("ta", ta)
		& core_nvp("ts", ts)
		& core_nvp("dt", dt)
		;
}

template<class Archive>
void shyft::time_series::dd::periodic_ts::serialize(Archive & ar, const unsigned int version) {
	ar
		& core_nvp("ipoint_ts", base_object<shyft::time_series::dd::ipoint_ts>(*this))
		& core_nvp("ts", ts)
		;
}

template<class Archive>
void shyft::time_series::dd::convolve_w_ts::serialize(Archive & ar, const unsigned int version) {
	ar
		& core_nvp("ipoint_ts", base_object<shyft::time_series::dd::ipoint_ts>(*this))
		& core_nvp("ts_impl", ts_impl)
		;
}

template<class Archive>
void shyft::time_series::dd::rating_curve_ts::serialize(Archive & ar, const unsigned int version) {
	ar
		& core_nvp("ipoint_ts", base_object<shyft::time_series::dd::ipoint_ts>(*this))
		& core_nvp("ts", ts)
		;
}

template<>
template<class Archive>
void shyft::time_series::rating_curve_ts<shyft::time_series::dd::apoint_ts>::serialize(Archive & ar, const unsigned int version) {
	bool bd = bound;
	ar
		& core_nvp("level_ts", level_ts)
		& core_nvp("rc_param", rc_param)
		& core_nvp("fx_policy", fx_policy)
		& core_nvp("bound", bd)
		;
	bound = bd;
}

template<class Archive>
void shyft::time_series::dd::ice_packing_ts::serialize(Archive & ar, const unsigned int version) {
    ar
        & core_nvp("ipoint_ts", base_object<shyft::time_series::dd::ipoint_ts>(*this))
        & core_nvp("ts", ts)
        ;
}

template<>
template<class Archive>
void shyft::time_series::ice_packing_ts<shyft::time_series::dd::apoint_ts>::serialize(Archive & ar, const unsigned int version) {
    bool bd = bound;
    ar
        & core_nvp("temp_ts", temp_ts)
        & core_nvp("ip_param", ip_param)
        & core_nvp("ipt_policy", ipt_policy)
        & core_nvp("fx_policy", fx_policy)
        & core_nvp("bound", bd)
        ;
    bound = bd;
}

template<class Archive>
void shyft::time_series::dd::ice_packing_recession_parameters::serialize(Archive & ar, const unsigned int version) {
    ar
        & core_nvp("alpha", alpha)
        & core_nvp("recession_minimum", recession_minimum)
        ;
}

template<class Archive>
void shyft::time_series::dd::ice_packing_recession_ts::serialize(Archive & ar, const unsigned int version) {
    bool bd = bound;
    ar
        & core_nvp("ipoint_ts", base_object<shyft::time_series::dd::ipoint_ts>(*this))
        & core_nvp("flow_ts", flow_ts)
        & core_nvp("ice_packing_ts", ice_packing_ts)
        & core_nvp("ipr_param", ipr_param)
        & core_nvp("fx_policy", fx_policy)
        & core_nvp("bound", bd)
        ;
    bound = bd;
}

template<class Archive>
void shyft::time_series::dd::krls_interpolation_ts::serialize(Archive & ar, const unsigned int version) {
	bool bd = bound;
	ar
		& core_nvp("ipoint_ts", base_object<shyft::time_series::dd::ipoint_ts>(*this))
		& core_nvp("ts", ts)
		& core_nvp("predictor", predictor)
		& core_nvp("bound", bd)
		;
	bound = bd;
}

template<class Archive>
void shyft::time_series::dd::ats_vector::serialize(Archive& ar, const unsigned int version) {
	ar
		& core_nvp("ats_vec", base_object<shyft::time_series::dd::ats_vec>(*this))
		;
}

// kind of special, mix core and api, hmm!
template<>
template <class Archive>
void shyft::time_series::convolve_w_ts<shyft::time_series::dd::apoint_ts>::serialize(Archive & ar, const unsigned int version) {
	ar
		& core_nvp("ts", ts)
		& core_nvp("fx_policy", fx_policy)
		& core_nvp("w", w)
		& core_nvp("convolve_policy", policy)
		& core_nvp("bound", bound)
		;
}

template<class Archive>
void shyft::time_series::dd::extend_ts::serialize(Archive & ar, const unsigned int version) {
	ar
		& core_nvp("ipoint_ts", base_object<shyft::time_series::dd::ipoint_ts>(*this))
		& core_nvp("lhs", lhs)
		& core_nvp("rhs", rhs)
		& core_nvp("split_at", split_at)
		& core_nvp("ets_split_p", ets_split_p)
		& core_nvp("fill_value", fill_value)
		& core_nvp("ets_fill_p", ets_fill_p)
		& core_nvp("ta", ta)
		& core_nvp("fx_policy", fx_policy)
		& core_nvp("bound", bound)
		;
}

template<class Archive>
void shyft::time_series::dd::abin_op_ts::serialize(Archive & ar, const unsigned int version) {
	ar
		& core_nvp("ipoint_ts", base_object<shyft::time_series::dd::ipoint_ts>(*this))
		& core_nvp("lhs", lhs)
		& core_nvp("op", op)
		& core_nvp("rhs", rhs)
		& core_nvp("ta", ta)
		& core_nvp("fx_policy", fx_policy)
		& core_nvp("bound", bound)
		;
}

template<class Archive>
void shyft::time_series::dd::abin_op_scalar_ts::serialize(Archive & ar, const unsigned int version) {
	ar
		& core_nvp("ipoint_ts", base_object<shyft::time_series::dd::ipoint_ts>(*this))
		& core_nvp("lhs", lhs)
		& core_nvp("op", op)
		& core_nvp("rhs", rhs)
		& core_nvp("ta", ta)
		& core_nvp("fx_policy", fx_policy)
		& core_nvp("bound", bound)
		;
}

template<class Archive>
void shyft::time_series::dd::abin_op_ts_scalar::serialize(Archive & ar, const unsigned int version) {
	ar
		& core_nvp("ipoint_ts", base_object<shyft::time_series::dd::ipoint_ts>(*this))
		& core_nvp("lhs", lhs)
		& core_nvp("op", op)
		& core_nvp("rhs", rhs)
		& core_nvp("ta", ta)
		& core_nvp("fx_policy", fx_policy)
		& core_nvp("bound", bound)
		;
}

template<class Archive>
void shyft::time_series::dd::abs_ts::serialize(Archive & ar, const unsigned int version) {
	ar
		& core_nvp("ipoint_ts", base_object<shyft::time_series::dd::ipoint_ts>(*this))
		& core_nvp("ts", ts)
		;
}

template<class Archive>
void shyft::time_series::dd::qac_ts::serialize(Archive & ar, const unsigned int version) {
	ar
		& core_nvp("ipoint_ts", base_object<shyft::time_series::dd::ipoint_ts>(*this))
		& core_nvp("ts", ts)
		& core_nvp("cts", cts)
		& core_nvp("p", p)
		;
}
#if 0
template<class Archive>
void shyft::time_series::dd::qac_parameter::serialize(Archive & ar, const unsigned int version) {
	ar
		& core_nvp("min_x", min_x)
		& core_nvp("max_x", max_x)
		& core_nvp("max_timespan", max_timespan)
		;
}
#endif

template<class Archive>
void shyft::time_series::dd::apoint_ts::serialize(Archive & ar, const unsigned int version) {
	ar
		& core_nvp("ts", ts)
		;
}


//-- export dtss stuff
x_serialize_implement(shyft::dtss::ts_info);
x_serialize_implement(shyft::dtss::cache_stats);

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

x_serialize_implement(shyft::time_series::ice_packing_parameters);
x_serialize_implement(shyft::time_series::ice_packing_ts<shyft::time_series::point_ts<shyft::time_axis::fixed_dt>>);
x_serialize_implement(shyft::time_series::ice_packing_ts<shyft::time_series::point_ts<shyft::time_axis::calendar_dt>>);
x_serialize_implement(shyft::time_series::ice_packing_ts<shyft::time_series::point_ts<shyft::time_axis::point_dt>>);
x_serialize_implement(shyft::time_series::ice_packing_ts<shyft::time_series::point_ts<shyft::time_axis::generic_dt>>);

x_serialize_implement(shyft::time_series::dd::ipoint_ts);
x_serialize_implement(shyft::time_series::dd::gpoint_ts);
x_serialize_implement(shyft::time_series::dd::aref_ts);
x_serialize_implement(shyft::time_series::dd::average_ts);
x_serialize_implement(shyft::time_series::dd::integral_ts);
x_serialize_implement(shyft::time_series::dd::accumulate_ts);
x_serialize_implement(shyft::time_series::dd::abs_ts);
x_serialize_implement(shyft::time_series::dd::time_shift_ts);
x_serialize_implement(shyft::time_series::dd::periodic_ts);
x_serialize_implement(shyft::time_series::convolve_w_ts<shyft::time_series::dd::apoint_ts>);
x_serialize_implement(shyft::time_series::dd::convolve_w_ts);
x_serialize_implement(shyft::time_series::dd::rating_curve_ts);
x_serialize_implement(shyft::time_series::rating_curve_ts<shyft::time_series::dd::apoint_ts>);
x_serialize_implement(shyft::time_series::dd::ice_packing_ts);
x_serialize_implement(shyft::time_series::ice_packing_ts<shyft::time_series::dd::apoint_ts>);
x_serialize_implement(shyft::time_series::dd::ice_packing_recession_parameters);
x_serialize_implement(shyft::time_series::dd::ice_packing_recession_ts);
x_serialize_implement(shyft::time_series::dd::extend_ts);
x_serialize_implement(shyft::time_series::dd::abin_op_scalar_ts);
x_serialize_implement(shyft::time_series::dd::abin_op_ts);
x_serialize_implement(shyft::time_series::dd::abin_op_ts_scalar);
x_serialize_implement(shyft::time_series::dd::apoint_ts);
x_serialize_implement(shyft::time_series::dd::krls_interpolation_ts);

x_serialize_implement(shyft::time_series::dd::ats_vector);
x_serialize_implement(shyft::time_series::dd::qac_ts);


//-- export predictors
x_serialize_implement(shyft::prediction::krls_rbf_predictor);

//
// 4. Then include the archive supported
//

x_arch(shyft::dtss::ts_info);

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

x_arch(shyft::time_series::ice_packing_parameters);
x_arch(shyft::time_series::ice_packing_ts<shyft::time_series::point_ts<shyft::time_axis::fixed_dt>>);
x_arch(shyft::time_series::ice_packing_ts<shyft::time_series::point_ts<shyft::time_axis::calendar_dt>>);
x_arch(shyft::time_series::ice_packing_ts<shyft::time_series::point_ts<shyft::time_axis::point_dt>>);
x_arch(shyft::time_series::ice_packing_ts<shyft::time_series::point_ts<shyft::time_axis::generic_dt>>);

x_arch(shyft::prediction::krls_rbf_predictor);
x_arch(shyft::dtss::cache_stats);

x_arch(shyft::time_series::dd::ipoint_ts);
x_arch(shyft::time_series::dd::gpoint_ts);
x_arch(shyft::time_series::dd::aref_ts);
x_arch(shyft::time_series::dd::average_ts);
x_arch(shyft::time_series::dd::integral_ts);
x_arch(shyft::time_series::dd::accumulate_ts);
x_arch(shyft::time_series::dd::abs_ts);
x_arch(shyft::time_series::dd::time_shift_ts);
x_arch(shyft::time_series::dd::periodic_ts);
x_arch(shyft::time_series::convolve_w_ts<shyft::time_series::dd::apoint_ts>);
x_arch(shyft::time_series::dd::convolve_w_ts);
x_arch(shyft::time_series::dd::rating_curve_ts);
x_arch(shyft::time_series::rating_curve_ts<shyft::time_series::dd::apoint_ts>);
x_arch(shyft::time_series::dd::ice_packing_ts);
x_arch(shyft::time_series::ice_packing_ts<shyft::time_series::dd::apoint_ts>);
x_arch(shyft::time_series::dd::ice_packing_recession_parameters);
x_arch(shyft::time_series::dd::ice_packing_recession_ts);
x_arch(shyft::time_series::dd::extend_ts);
x_arch(shyft::time_series::dd::abin_op_scalar_ts);
x_arch(shyft::time_series::dd::abin_op_ts);
x_arch(shyft::time_series::dd::abin_op_ts_scalar);
x_arch(shyft::time_series::dd::apoint_ts);
x_arch(shyft::time_series::dd::krls_interpolation_ts);
x_arch(shyft::time_series::dd::ats_vector);
x_arch(shyft::time_series::dd::qac_ts);
//binary x_arch(shyft::time_series::dd::qac_parameter);

std::string shyft::time_series::dd::apoint_ts::serialize() const {
	using namespace std;
	std::ostringstream xmls;
	core_oarchive oa(xmls, core_arch_flags);
	oa << core_nvp("ats", *this);
	xmls.flush();
	return xmls.str();
}
shyft::time_series::dd::apoint_ts shyft::time_series::dd::apoint_ts::deserialize(const std::string&str_bin) {
	istringstream xmli(str_bin);
	core_iarchive ia(xmli, core_arch_flags);
	shyft::time_series::dd::apoint_ts ats;
	ia >> core_nvp("ats", ats);
	return ats;
}
