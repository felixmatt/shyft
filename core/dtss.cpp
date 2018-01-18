#include <boost/functional/hash.hpp>
#include "dtss.h"
#include "core_serialization.h"
#include "expression_serialization.h"
#include "core_archive.h"
namespace shyft {
namespace dtss {

using std::vector;
using std::unordered_map;
using std::string;
using std::runtime_error;
using std::size_t;

using shyft::core::core_iarchive;
using shyft::core::core_oarchive;
using shyft::time_series::dd::ts_bind_info;
using shyft::time_series::dd::deflate_ts_vector;
using shyft::time_series::dd::expression_decompressor;
using shyft::time_series::dd::compressed_ts_expression;

struct utcperiod_hasher {
    size_t operator()(const utcperiod&k) const {
        using boost::hash_value;
        using boost::hash_combine;
        size_t seed=0;
        hash_combine(seed,hash_value(k.start));
        hash_combine(seed,hash_value(k.end));
        return seed;
    }
};

std::string shyft_prefix{ "shyft://" };

ts_info_vector_t server::do_find_ts(const string& search_expression) {
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


void server::do_cache_update_on_write(const ts_vector_t&tsv) {
    for (size_t i = 0; i<tsv.size(); ++i) {
        auto rts = dynamic_pointer_cast<aref_ts>(tsv[i].ts);
        ts_cache.add(rts->id, apoint_ts(rts->rep));
    }
}


void server::do_store_ts(const ts_vector_t & tsv, bool overwrite_on_write, bool cache_on_write) {
    if(tsv.size()==0) return;
    // 1. filter out all shyft://<container>/<ts-path> elements
    //    and route these to the internal storage controller (threaded)
    //    map<string, ts_db> shyft_internal;
    //
    vector<size_t> other;
    other.reserve(tsv.size());
    for(size_t i=0;i<tsv.size();++i) {
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

void server::do_merge_store_ts(const ts_vector_t& tsv,bool cache_on_write) {
    if(tsv.size()==0) return;
    //
    // 0. check&prepare the read time-series in tsv for the specified period of each ts
    // (we optimize a little bit grouping on common period, and reading in batches with equal periods)
    //
    id_vector_t ts_ids;ts_ids.reserve(tsv.size());
    unordered_map<utcperiod,id_vector_t,utcperiod_hasher> read_map;
    unordered_map<string,apoint_ts> id_map;
    for(size_t i=0;i<tsv.size();++i) {
        auto rts = dynamic_pointer_cast<aref_ts>(tsv[i].ts);
        if(!rts) throw runtime_error("dtss store merge: require ts with url-references");
        // sanity check
        if(id_map.find(rts->id)!= end(id_map))
            throw runtime_error("dtss store merge requires distinct set of ids, first duplicate found:"+rts->id);
        id_map[rts->id]=apoint_ts(rts->rep);
        // then just build up map[period] = list of time-series to read
        auto rp=rts->rep->total_period();
        if(read_map.find(rp) != end(read_map)) {
            read_map[rp].push_back(rts->id);
        } else {
            read_map[rp]=id_vector_t{rts->id};
        }
    }
    //
    // .1 do the read-merge for each common period, append to final minimal write list
    //
    ts_vector_t tsv_store;tsv_store.reserve(tsv.size());
    for(auto rr=read_map.begin();rr!=read_map.end();++rr) {
        auto read_ts= do_read(rr->second,rr->first,cache_on_write,cache_on_write);
        // read_ts is in the order of the ts-id-list rr->second
        for(size_t i=0;i<read_ts.size();++i) {
            auto ts_id =rr->second[i];
            read_ts[i].merge_points(id_map[ts_id]);
            tsv_store.push_back(apoint_ts(ts_id,read_ts[i]));
        }
    }
    
    // 
    // (2) finally write the merged result back to whatever store is there
    //
    do_store_ts(tsv_store,false,cache_on_write);
    
}

ts_vector_t server::do_read(const id_vector_t& ts_ids,utcperiod p,bool use_ts_cached_read,bool update_ts_cache) {
    if(ts_ids.size()==0) return ts_vector_t{};
    bool cache_read_results=update_ts_cache || cache_all_reads;
    // 0. filter out ts we can get from cache, given we are allowed to use cache
    unordered_map<string,apoint_ts> cc;
    if(use_ts_cached_read)
        cc = ts_cache.get(ts_ids,p);
    ts_vector_t r(ts_ids.size());
    vector<size_t> other;
    if (cc.size() == ts_ids.size()) { // if we got all from cache, just go ahead and map in the results
        for(size_t i=0;i<ts_ids.size();++i)
            r[i] = cc[ts_ids[i]];
    } else {
        // 1. filter out shyft://
        //    if all shyft: return internal read
        other.reserve(ts_ids.size()); // only reserve space when needed
        for (size_t i = 0; i < ts_ids.size(); ++i) {
            if (cc.find(ts_ids[i]) == cc.end()) {
                auto c = extract_shyft_url_container(ts_ids[i]);
                if (c.size()) {
                    r[i] = apoint_ts(make_shared<gpoint_ts>(internal(c).read(ts_ids[i].substr(shyft_prefix.size() + c.size() + 1), p)));
                    if (cache_read_results) ts_cache.add(ts_ids[i], r[i]);
                } else
                    other.push_back(i);
            } else {
                r[i] = cc[ts_ids[i]];
            }
        }
    }
    // 2. if other/more than shyft
    //    get all those
    if(other.size()) {
        if(!bind_ts_cb)
            throw runtime_error("dtss: read-request to external ts, without external handler");
        if(other.size()==ts_ids.size()) {// only other series, just return result
            auto rts= bind_ts_cb(ts_ids,p);
            if(cache_read_results) ts_cache.add(ts_ids,rts);
            return rts;
        }
        vector<string> o_ts_ids;o_ts_ids.reserve(other.size());
        for(auto i:other) o_ts_ids.push_back(ts_ids[i]);
        auto o=bind_ts_cb(o_ts_ids,p);
        if(cache_read_results) ts_cache.add(o_ts_ids,o);
        // if both shyft&cached plus other, merge into one ordered result vector
        //
        for(size_t i=0;i<o.size();++i)
            r[other[i]]=o[i];
    }
    return r;
}

void
server::do_bind_ts(utcperiod bind_period, ts_vector_t& atsv,bool use_ts_cached_read,bool update_ts_cache)  {
    unordered_map<string, vector<ts_bind_info>> ts_bind_map;
    vector<string> ts_id_list;
    // step 1: bind not yet bound time-series ( ts with only symbol, needs to be resolved using bind_cb)
    for (auto& ats : atsv) {
        auto ts_refs = ats.find_ts_bind_info();
        for (const auto& bi : ts_refs) {
            if (ts_bind_map.find(bi.reference) == ts_bind_map.end()) { // maintain unique set
                ts_id_list.push_back(bi.reference);
                ts_bind_map[bi.reference] = vector<ts_bind_info>();
            }
            ts_bind_map[bi.reference].push_back(bi);
        }
    }

    // step 2: (optional) bind_ts callback should resolve symbol time-series with content
    if (ts_bind_map.size()) {
        auto bts = do_read(ts_id_list, bind_period,use_ts_cached_read,update_ts_cache);
        if (bts.size() != ts_id_list.size())
            throw runtime_error(string("failed to bind all of ") + std::to_string(bts.size()) + string(" ts"));

        for ( size_t i = 0; i < ts_id_list.size(); ++i ) {
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
server::do_evaluate_ts_vector(utcperiod bind_period, ts_vector_t& atsv,bool use_ts_cached_read,bool update_ts_cache) {
    do_bind_ts(bind_period, atsv,use_ts_cached_read,update_ts_cache);
    return ts_vector_t{deflate_ts_vector<apoint_ts>(atsv)};
}

ts_vector_t
server::do_evaluate_percentiles(utcperiod bind_period, ts_vector_t& atsv, gta_t const&ta, vector<int64_t> const& percentile_spec,bool use_ts_cached_read,bool update_ts_cache) {
    do_bind_ts(bind_period, atsv,use_ts_cached_read,update_ts_cache);
    vector<int> p_spec;for(const auto p:percentile_spec) p_spec.push_back(int(p));// convert
    return percentiles(atsv, ta, p_spec);// we can assume the result is trivial to serialize
}

void server::on_connect(
    std::istream& in,
    std::ostream& out,
    const string& foreign_ip,
    const string& local_ip,
    unsigned short foreign_port,
    unsigned short local_port,
    dlib::uint64 connection_id
    ) {
    while (in.peek() != EOF) {
        auto msg_type= msg::read_type(in);
        try { // scoping the binary-archive could be ok, since it forces destruction time (considerable) to taken immediately, reduce memory foot-print early
              //  at the cost of early& fast response. I leave the commented scopes in there for now, and aim for fastest response-time
            switch (msg_type) { // currently switch, later maybe table[msg_type]=msg_handler
            case message_type::EVALUATE_TS_VECTOR:
            case message_type::EVALUATE_EXPRESSION:{
                utcperiod bind_period;bool use_ts_cached_read,update_ts_cache;
                ts_vector_t rtsv;
                core_iarchive ia(in,core_arch_flags);
                ia>>bind_period;
                if(msg_type==message_type::EVALUATE_EXPRESSION) {
                    compressed_ts_expression c_expr;
                    ia>>c_expr;
                    rtsv=expression_decompressor::decompress(c_expr);
                } else {
                    ia>>rtsv;
                }
                ia>>use_ts_cached_read>>update_ts_cache;
                auto result=do_evaluate_ts_vector(bind_period, rtsv,use_ts_cached_read,update_ts_cache);//first get result
                msg::write_type(message_type::EVALUATE_TS_VECTOR,out);// then send
                core_oarchive oa(out,core_arch_flags);
                oa<<result;

            } break;
            case message_type::EVALUATE_EXPRESSION_PERCENTILES:
            case message_type::EVALUATE_TS_VECTOR_PERCENTILES: {
                utcperiod bind_period;bool use_ts_cached_read,update_ts_cache;
                ts_vector_t rtsv;
                vector<int64_t> percentile_spec;
                gta_t ta;
                core_iarchive ia(in,core_arch_flags);
                ia >> bind_period;
                if(msg_type==message_type::EVALUATE_EXPRESSION_PERCENTILES) {
                    compressed_ts_expression c_expr;
                    ia>>c_expr;
                    rtsv=expression_decompressor::decompress(c_expr);
                } else {
                    ia>>rtsv;
                }

                ia>>ta>>percentile_spec>>use_ts_cached_read>>update_ts_cache;

                auto result = do_evaluate_percentiles(bind_period, rtsv,ta,percentile_spec,use_ts_cached_read,update_ts_cache);//{
                msg::write_type(message_type::EVALUATE_TS_VECTOR_PERCENTILES, out);
                core_oarchive oa(out,core_arch_flags);
                oa << result;
            } break;
            case message_type::FIND_TS: {
                string search_expression; //{
                search_expression = msg::read_string(in);// >> search_expression;
                auto find_result = do_find_ts(search_expression);
                msg::write_type(message_type::FIND_TS, out);
                core_oarchive oa(out,core_arch_flags);
                oa << find_result;
            } break;
            case message_type::STORE_TS: {
                ts_vector_t rtsv;
                bool overwrite_on_write{ true };
                bool cache_on_write{ false };
                core_iarchive ia(in,core_arch_flags);
                ia >> rtsv >> overwrite_on_write >> cache_on_write;
                do_store_ts(rtsv, overwrite_on_write, cache_on_write);
                msg::write_type(message_type::STORE_TS, out);
            } break;
            case message_type::MERGE_STORE_TS: {
                ts_vector_t rtsv;
                bool cache_on_write{ false };
                core_iarchive ia(in,core_arch_flags);
                ia >> rtsv >> cache_on_write;
                do_merge_store_ts(rtsv, cache_on_write);
                msg::write_type(message_type::MERGE_STORE_TS, out);
            } break;
            case message_type::CACHE_FLUSH: {
                flush_cache();
                clear_cache_stats();
                msg::write_type(message_type::CACHE_FLUSH,out);
            } break;
            case message_type::CACHE_STATS: {
                auto cs = get_cache_stats();
                msg::write_type(message_type::CACHE_STATS,out);
                core_oarchive oa(out,core_arch_flags);
                oa<<cs;
            } break;
            default:
                throw runtime_error(string("Server got unknown message type:") + std::to_string((int)msg_type));
            }
        } catch (std::exception const& e) {
            msg::send_exception(e,out);
        }
    }
}


}
}
