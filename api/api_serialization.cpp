#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wsign-compare"
#endif
#include "api_pch.h"

/**
 serializiation implemented using boost,
  see reference: http://www.boost.org/doc/libs/1_62_0/libs/serialization/doc/
 */
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
//    files with serializeation support
//

#include "core/utctime_utilities.h"
#include "core/time_axis.h"
#include "core/time_series.h"

#include "time_series.h"
#include "api_state.h"
// then include stuff you need like vector,shared, base_obj,nvp etc.

//
// 2. Then implement each class serialization support
//

using namespace boost::serialization;

/* api time-series serialization (dyn-dispatch) */

template <class Archive>
void shyft::api::ipoint_ts::serialize(Archive & ar, const unsigned) {
}

template<class Archive>
void shyft::api::gpoint_ts::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("ipoint_ts",base_object<shyft::api::ipoint_ts>(*this))
    & make_nvp("rep", rep)
    ;
}


template<class Archive>
void shyft::api::aref_ts::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("ipoint_ts",base_object<shyft::api::ipoint_ts>(*this))
    & make_nvp("id", id)
    & make_nvp("rep",rep)

    ;
}

template<class Archive>
void shyft::api::average_ts::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("ipoint_ts",base_object<shyft::api::ipoint_ts>(*this))
    & make_nvp("ta", ta)
    & make_nvp("ts", ts)
    ;
}

template<class Archive>
void shyft::api::integral_ts::serialize(Archive & ar, const unsigned int version) {
    ar
        & make_nvp("ipoint_ts", base_object<shyft::api::ipoint_ts>(*this))
        & make_nvp("ta", ta)
        & make_nvp("ts", ts)
        ;
}

template<class Archive>
void shyft::api::accumulate_ts::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("ipoint_ts",base_object<shyft::api::ipoint_ts>(*this))
    & make_nvp("ta", ta)
    & make_nvp("ts", ts)
    ;
}

template<class Archive>
void shyft::api::time_shift_ts::serialize(Archive & ar,const unsigned int version) {
    ar
    & make_nvp("ipoint_ts",base_object<shyft::api::ipoint_ts>(*this))
    & make_nvp("ta", ta)
    & make_nvp("ts", ts)
    & make_nvp("dt", dt)
    ;
}

template<class Archive>
void shyft::api::periodic_ts::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("ipoint_ts",base_object<shyft::api::ipoint_ts>(*this))
    & make_nvp("ts", ts)
    ;
}

template<class Archive>
void shyft::api::convolve_w_ts::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("ipoint_ts", base_object<shyft::api::ipoint_ts>(*this))
    & make_nvp("ts_impl", ts_impl)
    ;
}

template<class Archive>
void shyft::api::rating_curve_ts::serialize(Archive & ar, const unsigned int version) {
	ar
		& make_nvp("ipoint_ts", base_object<shyft::api::ipoint_ts>(*this))
		& make_nvp("ts", ts)
		;
}

template<>
template<class Archive>
void shyft::time_series::rating_curve_ts<shyft::api::apoint_ts>::serialize(Archive & ar, const unsigned int version) {
	bool bd = bound;
	ar
		& make_nvp("level_ts", level_ts)
		& make_nvp("rc_param", rc_param)
		& make_nvp("fx_policy", fx_policy)
		& make_nvp("bound", bd)
		;
	bound = bd;
}

template<class Archive>
void shyft::api::krls_interpolation_ts::serialize(Archive & ar, const unsigned int version) {
    bool bd = bound;
    ar
        & make_nvp("ipoint_ts", base_object<shyft::api::ipoint_ts>(*this))
        & make_nvp("ts", ts)
        & make_nvp("predictor", predictor)
        & make_nvp("bound", bd)
        ;
    bound = bd;
}

template<class Archive>
void shyft::api::ats_vector::serialize(Archive& ar, const unsigned int version) {
    ar
    & make_nvp("ats_vec",base_object<shyft::api::ats_vec>(*this))
    ;
}

// kind of special, mix core and api, hmm!
template<>
template <class Archive>
void shyft::time_series::convolve_w_ts<shyft::api::apoint_ts>::serialize(Archive & ar, const unsigned int version) {
    ar
		& make_nvp("ts", ts)
		& make_nvp("fx_policy", fx_policy)
		& make_nvp("w", w)
		& make_nvp("convolve_policy", policy)
		& make_nvp("bound", bound)
    ;
}

template<class Archive>
void shyft::api::extend_ts::serialize(Archive & ar, const unsigned int version) {
    ar
        & make_nvp("ipoint_ts",base_object<shyft::api::ipoint_ts>(*this))
        & make_nvp("lhs", lhs)
        & make_nvp("rhs", rhs)
        & make_nvp("split_at", split_at)
        & make_nvp("ets_split_p", ets_split_p)
        & make_nvp("fill_value", fill_value)
        & make_nvp("ets_fill_p", ets_fill_p)
        & make_nvp("ta", ta)
        & make_nvp("fx_policy", fx_policy)
        & make_nvp("bound",bound)
        ;
}

template<class Archive>
void shyft::api::abin_op_ts::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("ipoint_ts",base_object<shyft::api::ipoint_ts>(*this))
    & make_nvp("lhs", lhs)
    & make_nvp("op", op)
    & make_nvp("rhs", rhs)
    & make_nvp("ta", ta)
    & make_nvp("fx_policy", fx_policy)
    & make_nvp("bound",bound)
    ;
}

template<class Archive>
void shyft::api::abin_op_scalar_ts::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("ipoint_ts",base_object<shyft::api::ipoint_ts>(*this))
    & make_nvp("lhs", lhs)
    & make_nvp("op", op)
    & make_nvp("rhs", rhs)
    & make_nvp("ta", ta)
    & make_nvp("fx_policy", fx_policy)
    & make_nvp("bound",bound)
    ;
}

template<class Archive>
void shyft::api::abin_op_ts_scalar::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("ipoint_ts",base_object<shyft::api::ipoint_ts>(*this))
    & make_nvp("lhs", lhs)
    & make_nvp("op", op)
    & make_nvp("rhs", rhs)
    & make_nvp("ta", ta)
    & make_nvp("fx_policy", fx_policy)
    & make_nvp("bound",bound)
    ;
}

template<class Archive>
void shyft::api::abs_ts::serialize(Archive & ar, const unsigned int version) {
    ar
        & make_nvp("ipoint_ts", base_object<shyft::api::ipoint_ts>(*this))
        & make_nvp("ts", ts)
        ;
}

template<class Archive>
void shyft::api::qac_ts::serialize(Archive & ar, const unsigned int version) {
    ar
        & make_nvp("ipoint_ts", base_object<shyft::api::ipoint_ts>(*this))
        & make_nvp("ts", ts)
        & make_nvp("cts",cts)
        & make_nvp("p",p)
        ;
}

template<class Archive>
void shyft::api::qac_parameter::serialize(Archive & ar, const unsigned int version) {
    ar
        & make_nvp("min_x",min_x)
        & make_nvp("max_x",max_x)
        & make_nvp("max_timespan",max_timespan)
        ;
}


template<class Archive>
void shyft::api::apoint_ts::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("ts", ts)
    ;
}

template<class Archive>
void shyft::api::cell_state_id::serialize(Archive & ar, const unsigned int file_version) {
    ar
    & make_nvp("cid", cid)
    & make_nvp("x", x)
    & make_nvp("y", y)
    & make_nvp("a", area)
    ;
}
template <class CS>
template <class Archive>
void shyft::api::cell_state_with_id<CS>::serialize(Archive&ar, const unsigned int file_version) {
    ar
    & make_nvp("id", id)
    & make_nvp("state", state)
    ;
}
//
// 3. force impl. of pointers etc.
//
x_serialize_implement(shyft::api::ipoint_ts);
x_serialize_implement(shyft::api::gpoint_ts);
x_serialize_implement(shyft::api::aref_ts);
x_serialize_implement(shyft::api::average_ts);
x_serialize_implement(shyft::api::integral_ts);
x_serialize_implement(shyft::api::accumulate_ts);
x_serialize_implement(shyft::api::abs_ts);
x_serialize_implement(shyft::api::time_shift_ts);
x_serialize_implement(shyft::api::periodic_ts);
x_serialize_implement(shyft::time_series::convolve_w_ts<shyft::api::apoint_ts>);
x_serialize_implement(shyft::api::convolve_w_ts);
x_serialize_implement(shyft::api::rating_curve_ts);
x_serialize_implement(shyft::time_series::rating_curve_ts<shyft::api::apoint_ts>);
x_serialize_implement(shyft::api::extend_ts);
x_serialize_implement(shyft::api::abin_op_scalar_ts);
x_serialize_implement(shyft::api::abin_op_ts);
x_serialize_implement(shyft::api::abin_op_ts_scalar);
x_serialize_implement(shyft::api::apoint_ts);
x_serialize_implement(shyft::api::krls_interpolation_ts);
x_serialize_implement(shyft::api::cell_state_id);
x_serialize_implement(shyft::api::cell_state_with_id<shyft::core::hbv_stack::state>);
x_serialize_implement(shyft::api::cell_state_with_id<shyft::core::pt_gs_k::state>);
x_serialize_implement(shyft::api::cell_state_with_id<shyft::core::pt_ss_k::state>);
x_serialize_implement(shyft::api::cell_state_with_id<shyft::core::pt_hs_k::state>);
x_serialize_implement(shyft::api::ats_vector);
x_serialize_implement(shyft::api::qac_ts);
x_serialize_implement(shyft::api::qac_parameter);

//
// 4. Then include the archive supported
//
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

// repeat template instance for each archive class
#define x_arch(T) x_serialize_archive(T,binary_oarchive,binary_iarchive)

x_arch(shyft::api::ipoint_ts);
x_arch(shyft::api::gpoint_ts);
x_arch(shyft::api::aref_ts);
x_arch(shyft::api::average_ts);
x_arch(shyft::api::integral_ts);
x_arch(shyft::api::accumulate_ts);
x_arch(shyft::api::abs_ts);
x_arch(shyft::api::time_shift_ts);
x_arch(shyft::api::periodic_ts);
x_arch(shyft::time_series::convolve_w_ts<shyft::api::apoint_ts>);
x_arch(shyft::api::convolve_w_ts);
x_arch(shyft::api::rating_curve_ts);
x_arch(shyft::time_series::rating_curve_ts<shyft::api::apoint_ts>);
x_arch(shyft::api::extend_ts);
x_arch(shyft::api::abin_op_scalar_ts);
x_arch(shyft::api::abin_op_ts);
x_arch(shyft::api::abin_op_ts_scalar);
x_arch(shyft::api::apoint_ts);
x_arch(shyft::api::krls_interpolation_ts);
x_arch(shyft::api::cell_state_id);
x_arch(shyft::api::cell_state_with_id<shyft::core::hbv_stack::state>);
x_arch(shyft::api::cell_state_with_id<shyft::core::pt_gs_k::state>);
x_arch(shyft::api::cell_state_with_id<shyft::core::pt_ss_k::state>);
x_arch(shyft::api::cell_state_with_id<shyft::core::pt_hs_k::state>);
x_arch(shyft::api::ats_vector);
x_arch(shyft::api::qac_ts);
x_arch(shyft::api::qac_parameter);

std::string shyft::api::apoint_ts::serialize() const {
    using namespace std;
    std::ostringstream xmls;
    boost::archive::binary_oarchive oa(xmls);
    oa << BOOST_SERIALIZATION_NVP(*this);
    xmls.flush();
    return xmls.str();
}
shyft::api::apoint_ts shyft::api::apoint_ts::deserialize(const std::string&str_bin) {
    istringstream xmli(str_bin);
    boost::archive::binary_iarchive ia(xmli);
    shyft::api::apoint_ts ats;
    ia >> BOOST_SERIALIZATION_NVP(ats);
    return ats;
}

namespace shyft {
    namespace api {
        //-serialization of state to byte-array in python support
        template <class CS>
        std::vector<char> serialize_to_bytes(const std::shared_ptr<std::vector<CS>>& states) {
            using namespace std;
            std::ostringstream xmls;
            boost::archive::binary_oarchive oa(xmls);
            oa << BOOST_SERIALIZATION_NVP(states);
            xmls.flush();
            auto s = xmls.str();
            return std::vector<char>(s.begin(), s.end());
        }
        template std::vector<char> serialize_to_bytes(const std::shared_ptr<std::vector<cell_state_with_id<shyft::core::hbv_stack::state>>>& states);// { return serialize_to_bytes_impl(states); }
        template std::vector<char> serialize_to_bytes(const std::shared_ptr<std::vector<cell_state_with_id<shyft::core::pt_gs_k::state>>>& states);// { return serialize_to_bytes_impl(states); }
        template std::vector<char> serialize_to_bytes(const std::shared_ptr<std::vector<cell_state_with_id<shyft::core::pt_ss_k::state>>>& states);// { return serialize_to_bytes_impl(states); }
        template std::vector<char> serialize_to_bytes(const std::shared_ptr<std::vector<cell_state_with_id<shyft::core::pt_hs_k::state>>>& states);// { return serialize_to_bytes_impl(states); }

        template <class CS>
        void deserialize_from_bytes(const std::vector<char>& bytes, std::shared_ptr<std::vector<CS>>&states) {
            using namespace std;
            string str_bin(bytes.begin(), bytes.end());
            istringstream xmli(str_bin);
            boost::archive::binary_iarchive ia(xmli);
            ia >> BOOST_SERIALIZATION_NVP(states);
        }
        template void deserialize_from_bytes(const std::vector<char>& bytes, std::shared_ptr<std::vector<cell_state_with_id<shyft::core::hbv_stack::state>>>&states);// { deserialize_from_bytes_impl(bytes, states); }
        template void deserialize_from_bytes(const std::vector<char>& bytes, std::shared_ptr<std::vector<cell_state_with_id<shyft::core::pt_gs_k::state>>>&states);// { deserialize_from_bytes_impl(bytes, states); }
        template void deserialize_from_bytes(const std::vector<char>& bytes, std::shared_ptr<std::vector<cell_state_with_id<shyft::core::pt_hs_k::state>>>&states);// { deserialize_from_bytes_impl(bytes, states); }
        template void deserialize_from_bytes(const std::vector<char>& bytes, std::shared_ptr<std::vector<cell_state_with_id<shyft::core::pt_ss_k::state>>>&states);// { deserialize_from_bytes_impl(bytes, states); }
    }
}
