#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <memory>
#include <utility>
#include <functional>
#include <cstring>
#include <regex>


#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>


#include <dlib/server.h>
#include <dlib/iosockstream.h>
#include <dlib/logger.h>
#include <dlib/misc_api.h>


#include "api/time_series.h"
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

// TODO: Remove API dependency from core...
using shyft::api::apoint_ts;
using shyft::api::gpoint_ts;
using shyft::api::gts_t;
using shyft::api::aref_ts;

// ========================================

using ts_vector_t = shyft::api::ats_vector;
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
    std::map<std::string, ts_db> container;///< mapping of internal shyft <container> -> ts_db
    ts_cache_t ts_cache{1000000};// default 1 mill ts in cache
    bool cache_all_reads{false};
    // constructors

    server()=default;

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

    ts_info_vector_t do_find_ts(const std::string& search_expression) {
        // 1. filter shyft://<container>/
        auto c=extract_shyft_url_container(search_expression);
        if(c.size()) {
            return internal(c).find(search_expression.substr(shyft_prefix.size()+c.size()+1));
        } else if (find_ts_cb) {
            return find_ts_cb(search_expression);
        } else {
            return ts_info_vector_t();
        }
    }

    std::string extract_url(const apoint_ts&ats) const {
        auto rts = dynamic_pointer_cast<aref_ts>(ats.ts);
        if(rts)
            return rts->id;
        throw runtime_error("dtss store.extract_url:supplied type must be of type ref_ts");
    }

    void do_cache_update_on_write(const ts_vector_t&tsv) {
        for (std::size_t i = 0; i<tsv.size(); ++i) {
            auto rts = dynamic_pointer_cast<aref_ts>(tsv[i].ts);
            ts_cache.add(rts->id, apoint_ts(rts->rep));
        }
    }

    void do_store_ts(const ts_vector_t & tsv, bool overwrite_on_write, bool cache_on_write) {
        if(tsv.size()==0) return;
        // 1. filter out all shyft://<container>/<ts-path> elements
        //    and route these to the internal storage controller (threaded)
        //    std::map<std::string, ts_db> shyft_internal;
        //
        std::vector<std::size_t> other;
        other.reserve(tsv.size());
        for(std::size_t i=0;i<tsv.size();++i) {
            auto rts = dynamic_pointer_cast<aref_ts>(tsv[i].ts);
            if(!rts) throw runtime_error("dtss store: require ts with url-references");
            auto c= extract_shyft_url_container(rts->id);
            if(c.size()) {
                internal(c).save(
                    rts->id.substr(shyft_prefix.size()+c.size()+1),  // path
                    rts->core_ts(),  // ts to save
                    overwrite_on_write  // should do overwrite instead of merge
                );
                if ( cache_on_write ) { // ok, this ends up in a copy, and lock for each item(can be optimized if many)
                    ts_cache.add(rts->id, apoint_ts(rts->rep));
                }
            } else {
                other.push_back(i); // keep idx of those we have not saved
            }
        }

        // 2. for all non shyft:// forward those to the
        //    store_ts_cb
        if(store_ts_cb && other.size()) {
            if(other.size()==tsv.size()) { //avoid copy/move if possible
                store_ts_cb(tsv);
                if (cache_on_write) do_cache_update_on_write(tsv);
            } else { // have to do a copy to new vector
                ts_vector_t r;
                for(auto i:other) r.push_back(tsv[i]);
                store_ts_cb(r);
                if (cache_on_write) do_cache_update_on_write(r);
            }
        }
    }

    ts_vector_t do_read(const id_vector_t& ts_ids,utcperiod p) {
        if(ts_ids.size()==0) return ts_vector_t{};
        // 0. filter out cached ts
        auto cc = ts_cache.get(ts_ids,p);
        ts_vector_t r(ts_ids.size());
        std::vector<std::size_t> other;other.reserve(ts_ids.size());
        // 1. filter out shyft://
        //    if all shyft: return internal read
        for(std::size_t i=0;i<ts_ids.size();++i) {
            if(cc.find(ts_ids[i])==cc.end()) {
                auto c=extract_shyft_url_container(ts_ids[i]);
                if(c.size()) {
                    r[i]=apoint_ts(make_shared<gpoint_ts>(internal(c).read(ts_ids[i].substr(shyft_prefix.size()+c.size()+1),p)));
                    if(cache_all_reads) ts_cache.add(ts_ids[i],r[i]);
                } else
                    other.push_back(i);
            } else {
                r[i]=cc[ts_ids[i]];
            }
        }
        // 2. if other/more than shyft
        //    get all those
        if(other.size()) {
            if(!bind_ts_cb)
                throw std::runtime_error("dtss: read-request to external ts, without external handler");
            if(other.size()==ts_ids.size()) {// only other series, just return result
                auto rts= bind_ts_cb(ts_ids,p);
                if(cache_all_reads) ts_cache.add(ts_ids,rts);
                return rts;
            }
            std::vector<std::string> o_ts_ids;o_ts_ids.reserve(other.size());
            for(auto i:other) o_ts_ids.push_back(ts_ids[i]);
            auto o=bind_ts_cb(o_ts_ids,p);
            if(cache_all_reads) ts_cache.add(o_ts_ids,o);
            // if 1 and 2, merge into one ordered result vector
            //
            for(std::size_t i=0;i<o.size();++i)
                r[other[i]]=o[i];
        }
        return r;
    }

    void
    do_bind_ts(utcperiod bind_period, ts_vector_t& atsv)  {
        std::map<std::string, std::vector<api::ts_bind_info>> ts_bind_map;
        std::vector<std::string> ts_id_list;
        // step 1: bind not yet bound time-series ( ts with only symbol, needs to be resolved using bind_cb)
        for (auto& ats : atsv) {
            auto ts_refs = ats.find_ts_bind_info();
            for (const auto& bi : ts_refs) {
                if (ts_bind_map.find(bi.reference) == ts_bind_map.end()) { // maintain unique set
                    ts_id_list.push_back(bi.reference);
                    ts_bind_map[bi.reference] = std::vector<api::ts_bind_info>();
                }
                ts_bind_map[bi.reference].push_back(bi);
            }
        }

        // step 2: (optional) bind_ts callback should resolve symbol time-series with content
        if (ts_bind_map.size()) {
            auto bts = do_read(ts_id_list, bind_period);
            if (bts.size() != ts_id_list.size())
                throw std::runtime_error(std::string("failed to bind all of ") + std::to_string(bts.size()) + std::string(" ts"));

            for ( std::size_t i = 0; i < ts_id_list.size(); ++i ) {
                for ( auto & bi : ts_bind_map[ts_id_list[i]] )
                    bi.ts.bind(bts[i]);
            }
        }
        // step 3: after the symbolic ts are read and bound, we iterate over the
        //         expression tree and calls .do_bind() so that
        //         the new information is taken into account and the expression tree are
        //         ready for evaluate with everything const so threading is safe.
        for (auto& ats : atsv)
            ats.do_bind();
    }

    ts_vector_t
    do_evaluate_ts_vector(utcperiod bind_period, ts_vector_t& atsv) {
        do_bind_ts(bind_period, atsv);
        return ts_vector_t(api::deflate_ts_vector<apoint_ts>(atsv));
    }

    ts_vector_t
    do_evaluate_percentiles(utcperiod bind_period, ts_vector_t& atsv, api::gta_t const&ta,std::vector<int> const& percentile_spec) {
        do_bind_ts(bind_period, atsv);
        return api::percentiles(atsv, ta, percentile_spec);// we can assume the result is trivial to serialize
    }

    void on_connect(
        std::istream& in,
        std::ostream& out,
        const std::string& foreign_ip,
        const std::string& local_ip,
        unsigned short foreign_port,
        unsigned short local_port,
        dlib::uint64 connection_id
    ) {
        while (in.peek() != EOF) {
            auto msg_type= msg::read_type(in);
            try {
                switch (msg_type) { // currently switch, later maybe table[msg_type]=msg_handler
                case message_type::EVALUATE_TS_VECTOR: {
                    utcperiod bind_period;
                    ts_vector_t rtsv; {
                        boost::archive::binary_iarchive ia(in);
                        ia>>bind_period>>rtsv;
                    }

                    auto result=do_evaluate_ts_vector(bind_period, rtsv); {// first get result
                        msg::write_type(message_type::EVALUATE_TS_VECTOR,out);// then send
                        boost::archive::binary_oarchive oa(out);
                        oa<<result;
                    }
                } break;
                case message_type::EVALUATE_TS_VECTOR_PERCENTILES: {
                    utcperiod bind_period;
                    ts_vector_t rtsv;
                    std::vector<int> percentile_spec;
                    api::gta_t ta; {
                        boost::archive::binary_iarchive ia(in);
                        ia >> bind_period >> rtsv>>ta>>percentile_spec;
                    }

                    auto result = do_evaluate_percentiles(bind_period, rtsv,ta,percentile_spec);{
                        msg::write_type(message_type::EVALUATE_TS_VECTOR_PERCENTILES, out);
                        boost::archive::binary_oarchive oa(out);
                        oa << result;
                    }
                } break;
                case message_type::FIND_TS: {
                    std::string search_expression; {
                        search_expression = msg::read_string(in);// >> search_expression;
                    }
                    auto find_result = do_find_ts(search_expression); {
                        msg::write_type(message_type::FIND_TS, out);
                        boost::archive::binary_oarchive oa(out);
                        oa << find_result;
                    }
                } break;
                case message_type::STORE_TS: {
                    ts_vector_t rtsv;
                    bool overwrite_on_write{ true };
                    bool cache_on_write{ false };
                    {
                        boost::archive::binary_iarchive ia(in);
                        ia >> rtsv >> overwrite_on_write >> cache_on_write;
                    }
                    do_store_ts(rtsv, overwrite_on_write, cache_on_write);
                    {
                        msg::write_type(message_type::STORE_TS, out);
                    }
                } break;
                default:
                    throw std::runtime_error(std::string("Got unknown message type:") + std::to_string((int)msg_type));
                }
            } catch (std::exception const& e) {
                msg::send_exception(e,out);
            }
        }
    }
};


} // shyft::dtss
} // shyft
