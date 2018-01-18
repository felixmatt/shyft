#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <memory>
#include <utility>
#include <functional>
#include <cstring>
#include <regex>



#include "core_serialization.h"

#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>


#include <dlib/server.h>
#include <dlib/iosockstream.h>
#include <dlib/logger.h>
#include <dlib/misc_api.h>


#include "core/time_series_dd.h"
#include "time_series_info.h"
#include "utctime_utilities.h"
#include "dtss_cache.h"
#include "dtss_url.h"
#include "dtss_msg.h"
#include "dtss_db.h"

namespace shyft {
namespace dtss {

using shyft::core::utctime;
using shyft::core::utcperiod;
using shyft::core::utctimespan;
using shyft::core::no_utctime;
using shyft::core::calendar;
using shyft::core::deltahours;

using gta_t = shyft::time_axis::generic_dt;
using gts_t = shyft::time_series::point_ts<gta_t>;

using shyft::time_series::dd::apoint_ts;
using shyft::time_series::dd::gpoint_ts;
using shyft::time_series::dd::gts_t;
using shyft::time_series::dd::aref_ts;

// ========================================

using ts_vector_t = shyft::time_series::dd::ats_vector;
using ts_info_vector_t = std::vector<ts_info>;
using id_vector_t = std::vector<std::string>;
using read_call_back_t = std::function<ts_vector_t(const id_vector_t& ts_ids, utcperiod p)>;
using store_call_back_t = std::function<void(const ts_vector_t&)>;
using find_call_back_t = std::function<ts_info_vector_t(std::string search_expression)>;


/** \brief A dtss server with time-series server-side functions
 *
 * The dtss server listens on a port, receives messages, interpret them
 * and ship the response back to the client.
 *
 * Callbacks are provided for extending/delegating find/read_ts/store_ts,
 * as well as internal implementation of storing time-series
 * using plain binary files stored in containers(directory).
 *
 * Time-series are named with url's, and all request involving 'shyft://'
 * like
 *   shyft://<container>/<local_ts_name>
 * resolves to the internal implementation.
 *
 * TODO: container and thread-safety, given user changes the containers after
 *       the server is started.
 *
 */
struct server : dlib::server_iostream {
    using ts_cache_t = cache<apoint_ts_frag,apoint_ts>;
    // callbacks for extensions
    read_call_back_t bind_ts_cb; ///< called to read non shyft:// unbound ts
    find_call_back_t find_ts_cb; ///< called for all non shyft:// find operations
    store_call_back_t store_ts_cb;///< called for all non shyft:// store operations
    // shyft-internal implementation
    std::unordered_map<std::string, ts_db> container;///< mapping of internal shyft <container> -> ts_db
    ts_cache_t ts_cache{1000000};// default 1 mill ts in cache
    bool cache_all_reads{false};
    // constructors

    server()=default;
    server(server&&)=delete;
    server(const server&) =delete;
    server& operator=(const server&)=delete;
    server& operator=(server&&)=delete;

    template <class CB>
    explicit server(CB&& cb):bind_ts_cb(std::forward<CB>(cb)) {
    }

    template <class RCB,class FCB>
    server(RCB&& rcb, FCB && fcb ) :
        bind_ts_cb(std::forward<RCB>(rcb)),
        find_ts_cb(std::forward<FCB>(fcb)) {
    }

    template <class RCB,class FCB,class SCB>
    server(RCB&& rcb, FCB && fcb ,SCB&& scb) :
        bind_ts_cb(std::forward<RCB>(rcb)),
        find_ts_cb(std::forward<FCB>(fcb)),
        store_ts_cb(std::forward<SCB>(scb)) {
    }

    ~server() =default;

    //-- container management
    void add_container(const std::string &container_name,const std::string& root_dir) {
        container[container_name]=ts_db(root_dir); // TODO: This is not thread-safe(so needs to be done before starting)
    }

    const ts_db& internal(const std::string& container_name) const {
        auto f=container.find(container_name);
        if(f == end(container))
            throw runtime_error(std::string("Failed to find shyft container:")+container_name);
        return f->second;
    }

	//-- expose cache functions

    void add_to_cache(id_vector_t&ids, ts_vector_t& tss) { ts_cache.add(ids,tss);}
    void remove_from_cache(id_vector_t &ids) { ts_cache.remove(ids);}
    cache_stats get_cache_stats() { return ts_cache.get_cache_stats();}
    void clear_cache_stats() { ts_cache.clear_cache_stats();}
    void flush_cache() { return ts_cache.flush();}
    void set_cache_size(std::size_t max_size) { ts_cache.set_capacity(max_size);}
    void set_auto_cache(bool active) { cache_all_reads=active;}
    std::size_t get_cache_size() const {return ts_cache.get_capacity();}

    ts_info_vector_t do_find_ts(const std::string& search_expression);

    std::string extract_url(const apoint_ts&ats) const {
        auto rts = dynamic_pointer_cast<aref_ts>(ats.ts);
        if(rts)
            return rts->id;
        throw runtime_error("dtss store.extract_url:supplied type must be of type ref_ts");
    }

    void do_cache_update_on_write(const ts_vector_t&tsv);

    void do_store_ts(const ts_vector_t & tsv, bool overwrite_on_write, bool cache_on_write);

    void do_merge_store_ts(const ts_vector_t & tsv, bool cache_on_write);
    /** \brief Read the time-series from providers for specified period
    *
    * \param ts_ids identifiers, url form, where shyft://.. is specially filtered
    * \param p the period to read
    * \param use_ts_cached_read allow reading results from already existing cached results
    * \param update_ts_cache when reading, also update the ts-cache with the results
    * \return read ts-vector in the order of the ts_ids
    */
    ts_vector_t do_read(const id_vector_t& ts_ids,utcperiod p,bool use_ts_cached_read,bool update_ts_cache);
    void do_bind_ts(utcperiod bind_period, ts_vector_t& atsv,bool use_ts_cached_read,bool update_ts_cache);
    ts_vector_t do_evaluate_ts_vector(utcperiod bind_period, ts_vector_t& atsv,bool use_ts_cached_read,bool update_ts_cache);
    ts_vector_t do_evaluate_percentiles(utcperiod bind_period, ts_vector_t& atsv, gta_t const&ta,std::vector<int64_t> const& percentile_spec,bool use_ts_cached_read,bool update_ts_cache);

    // ref. dlib, all connection calls are directed here
    void on_connect(
        std::istream& in,
        std::ostream& out,
        const std::string& foreign_ip,
        const std::string& local_ip,
        unsigned short foreign_port,
        unsigned short local_port,
        dlib::uint64 connection_id
    );
};


} // shyft::dtss
} // shyft
