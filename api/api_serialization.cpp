#include "core/core_pch.h"

//
// 1. first include std stuff and the headers for
//    files with serializeation support
//

#include "core/utctime_utilities.h"
#include "core/time_axis.h"
#include "core/timeseries.h"

#include "timeseries.h"

// then include stuff you need like vector,shared, base_obj,nvp etc.

#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>

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
    & make_nvp("rep", rep)
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
void shyft::api::abin_op_ts::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("ipoint_ts",base_object<shyft::api::ipoint_ts>(*this))
    & make_nvp("lhs", lhs)
    & make_nvp("op", op)
    & make_nvp("rhs", rhs)
    & make_nvp("ta", ta)
    & make_nvp("fx_policy", fx_policy)
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
    ;
}


template<class Archive>
void shyft::api::apoint_ts::serialize(Archive & ar, const unsigned int version) {
    ar
    & make_nvp("ts", ts)
    ;
}

//
// 3. force impl. of pointers etc.
//
x_serialize_implement(shyft::api::ipoint_ts);
x_serialize_implement(shyft::api::gpoint_ts);
x_serialize_implement(shyft::api::aref_ts);
x_serialize_implement(shyft::api::average_ts);
x_serialize_implement(shyft::api::accumulate_ts);
x_serialize_implement(shyft::api::time_shift_ts);
x_serialize_implement(shyft::api::periodic_ts);
x_serialize_implement(shyft::api::abin_op_scalar_ts);
x_serialize_implement(shyft::api::abin_op_ts);
x_serialize_implement(shyft::api::abin_op_ts_scalar);
x_serialize_implement(shyft::api::apoint_ts);

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
x_arch(shyft::api::accumulate_ts);
x_arch(shyft::api::time_shift_ts);
x_arch(shyft::api::periodic_ts);
x_arch(shyft::api::abin_op_scalar_ts);
x_arch(shyft::api::abin_op_ts);
x_arch(shyft::api::abin_op_ts_scalar);
x_arch(shyft::api::apoint_ts);

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

