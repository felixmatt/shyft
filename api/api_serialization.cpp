#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wsign-compare"
#endif
#include "api_pch.h"

/**
 serializiation implemented using boost,
  see reference: http://www.boost.org/doc/libs/1_62_0/libs/serialization/doc/
 */
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>

//
// 1. first include std stuff and the headers for
//    files with serializeation support
//
#include "core/core_serialization.h"
#include "core/core_archive.h"
#include "api_state.h"
// then include stuff you need like vector,shared, base_obj,nvp etc.

//
// 2. Then implement each class serialization support
//

using namespace boost::serialization;
using namespace shyft::core;



template<class Archive>
void shyft::api::cell_state_id::serialize(Archive & ar, const unsigned int file_version) {
    ar
    & core_nvp("cid", cid)
    & core_nvp("x", x)
    & core_nvp("y", y)
    & core_nvp("a", area)
    ;
}
template <class CS>
template <class Archive>
void shyft::api::cell_state_with_id<CS>::serialize(Archive&ar, const unsigned int file_version) {
    ar
    & core_nvp("id", id)
    & core_nvp("state", state)
    ;
}
//
// 3. force impl. of pointers etc.
//
x_serialize_implement(shyft::api::cell_state_id);
x_serialize_implement(shyft::api::cell_state_with_id<shyft::core::hbv_stack::state>);
x_serialize_implement(shyft::api::cell_state_with_id<shyft::core::pt_gs_k::state>);
x_serialize_implement(shyft::api::cell_state_with_id<shyft::core::pt_ss_k::state>);
x_serialize_implement(shyft::api::cell_state_with_id<shyft::core::pt_hs_k::state>);
x_serialize_implement(shyft::api::cell_state_with_id<shyft::core::pt_hps_k::state>);
//
// 4. Then include the archive supported
//

x_arch(shyft::api::cell_state_id);
x_arch(shyft::api::cell_state_with_id<shyft::core::hbv_stack::state>);
x_arch(shyft::api::cell_state_with_id<shyft::core::pt_gs_k::state>);
x_arch(shyft::api::cell_state_with_id<shyft::core::pt_ss_k::state>);
x_arch(shyft::api::cell_state_with_id<shyft::core::pt_hs_k::state>);
x_arch(shyft::api::cell_state_with_id<shyft::core::pt_hps_k::state>);

namespace shyft {
    namespace api {
        //-serialization of state to byte-array in python support
        template <class CS>
        std::vector<char> serialize_to_bytes(const std::shared_ptr<std::vector<CS>>& states) {
            using namespace std;
            std::ostringstream xmls;
            core_oarchive oa(xmls,core_arch_flags);
            oa << core_nvp("states",states);
            xmls.flush();
            auto s = xmls.str();
            return std::vector<char>(s.begin(), s.end());
        }
        template std::vector<char> serialize_to_bytes(const std::shared_ptr<std::vector<cell_state_with_id<shyft::core::hbv_stack::state>>>& states);// { return serialize_to_bytes_impl(states); }
        template std::vector<char> serialize_to_bytes(const std::shared_ptr<std::vector<cell_state_with_id<shyft::core::pt_gs_k::state>>>& states);// { return serialize_to_bytes_impl(states); }
        template std::vector<char> serialize_to_bytes(const std::shared_ptr<std::vector<cell_state_with_id<shyft::core::pt_ss_k::state>>>& states);// { return serialize_to_bytes_impl(states); }
        template std::vector<char> serialize_to_bytes(const std::shared_ptr<std::vector<cell_state_with_id<shyft::core::pt_hs_k::state>>>& states);// { return serialize_to_bytes_impl(states); }
        template std::vector<char> serialize_to_bytes(const std::shared_ptr<std::vector<cell_state_with_id<shyft::core::pt_hps_k::state>>>& states);// { return serialize_to_bytes_impl(states); }

        template <class CS>
        void deserialize_from_bytes(const std::vector<char>& bytes, std::shared_ptr<std::vector<CS>>&states) {
            using namespace std;
            string str_bin(bytes.begin(), bytes.end());
            istringstream xmli(str_bin);
            core_iarchive ia(xmli,core_arch_flags);
            ia >> core_nvp("states",states);
        }
        template void deserialize_from_bytes(const std::vector<char>& bytes, std::shared_ptr<std::vector<cell_state_with_id<shyft::core::hbv_stack::state>>>&states);// { deserialize_from_bytes_impl(bytes, states); }
        template void deserialize_from_bytes(const std::vector<char>& bytes, std::shared_ptr<std::vector<cell_state_with_id<shyft::core::pt_gs_k::state>>>&states);// { deserialize_from_bytes_impl(bytes, states); }
        template void deserialize_from_bytes(const std::vector<char>& bytes, std::shared_ptr<std::vector<cell_state_with_id<shyft::core::pt_hs_k::state>>>&states);// { deserialize_from_bytes_impl(bytes, states); }
        template void deserialize_from_bytes(const std::vector<char>& bytes, std::shared_ptr<std::vector<cell_state_with_id<shyft::core::pt_ss_k::state>>>&states);// { deserialize_from_bytes_impl(bytes, states); }
        template void deserialize_from_bytes(const std::vector<char>& bytes, std::shared_ptr<std::vector<cell_state_with_id<shyft::core::pt_hps_k::state>>>&states);// { deserialize_from_bytes_impl(bytes, states); }
    }
}
