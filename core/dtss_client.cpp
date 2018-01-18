#include <map>
#include <algorithm>
#include <memory>
#include <utility>
#include <functional>
#include <cstring>
#include <regex>
#include <future>
#include <utility>

#include "dtss_client.h"
#include "dtss_url.h"
#include "dtss_msg.h"

#include "core_serialization.h"
#include "core_archive.h"
#include "expression_serialization.h"

#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>

namespace shyft {
namespace dtss {
using std::runtime_error;
using std::vector;
using std::string;
using std::int64_t;
using std::exception;
using std::future;
using std::min;

using shyft::core::core_iarchive;
using shyft::core::core_oarchive;
using shyft::time_series::dd::apoint_ts;
using shyft::time_series::dd::aref_ts;
using shyft::time_series::dd::gta_t;
using shyft::time_series::dd::expression_compressor;
using shyft::time_series::statistics_property;

void srv_connection::open(int timeout_ms) {
    io->open(host_port,max(timeout_ms,this->timeout_ms));
}
void srv_connection::close(int timeout_ms) {
    io->close(max(timeout_ms,this->timeout_ms));
}
void srv_connection::reopen(int timeout_ms) {
    io->open(host_port,max(timeout_ms,this->timeout_ms));
}


//--helper-class to enable autoconnect/close
struct scoped_connect {
    client& c;
    scoped_connect (client& c):c(c){
        bool rethrow=false;
        runtime_error rt_re("");
        if (c.auto_connect) {
            for(auto&sc:c.srv_con) {
                try {
                    sc.open();
                } catch(const runtime_error&re) {
                    rt_re=re;
                    rethrow=true;
                }
            }
            if(rethrow)
                throw rt_re;
        }
    }
    ~scoped_connect() noexcept(false) {
        if(c.auto_connect) {
            bool rethrow=false;
            runtime_error rt_re("");
            for(auto&sc:c.srv_con) {
                try {
                    sc.close();
                } catch(const exception& re) {
                    rt_re=runtime_error(re.what());
                    rethrow=true;
                }
            }
            if(rethrow)
                throw rt_re;
        }
    }
    scoped_connect (const scoped_connect&) = delete;
    scoped_connect ( scoped_connect&&) = delete;
    scoped_connect& operator=(const scoped_connect&) = delete;
    scoped_connect()=delete;
    scoped_connect& operator=(scoped_connect&&)=delete;
};

client::client ( const string& host_port, bool auto_connect, int timeout_ms )
    :auto_connect(auto_connect)
{
    srv_con.push_back(srv_connection{make_unique<dlib::iosockstream>(),host_port,timeout_ms});
    if(!auto_connect)
        srv_con[0].open();
}

client::client(const vector<string>& host_ports,bool auto_connect,int timeout_ms):auto_connect(auto_connect) {
    if(host_ports.size()==0)
        throw runtime_error("host_ports must contain at least one element");
    for(const auto &hp:host_ports) {
        srv_con.push_back(srv_connection{make_unique<dlib::iosockstream>(),hp,timeout_ms});
    }
    if(!auto_connect)
        reopen(timeout_ms);
}

void client::reopen(int timeout_ms) {
    for(auto&sc:srv_con)
        sc.reopen(timeout_ms);
}

void client::close(int timeout_ms) {
    bool rethrow = false;
    runtime_error rt_re("");
    for(auto&sc:srv_con) {
        try {
            sc.close(timeout_ms);
        } catch (const exception &re) { // ensure we try to close all
            rt_re=runtime_error(re.what());
            rethrow=true;
        }
    }
    if(rethrow)
        throw rt_re;
}

vector<apoint_ts>
client::percentiles(ts_vector_t const& tsv, utcperiod p, gta_t const&ta, const vector<int64_t>& percentile_spec,bool use_ts_cached_read,bool update_ts_cache) {
    if (tsv.size() == 0)
        throw runtime_error("percentiles requires a source ts-vector with more than 0 time-series");
    if (percentile_spec.size() == 0)
        throw std::runtime_error("percentile function require more than 0 percentiles specified");
    if (!p.valid())
        throw std::runtime_error("percentiles require a valid period-specification");
    if (ta.size() == 0)
        throw std::runtime_error("percentile function require a time-axis with more than 0 steps");
    scoped_connect ac(*this);

    if(srv_con.size()==1) {
        dlib::iosockstream& io = *(srv_con[0].io);
        msg::write_type(compress_expressions?message_type::EVALUATE_EXPRESSION_PERCENTILES:message_type::EVALUATE_TS_VECTOR_PERCENTILES, io);
        core_oarchive oa(io,core_arch_flags);
        oa << p;
        if (compress_expressions) {
            oa<< expression_compressor::compress(tsv);
        } else {
            oa<< tsv;
        }
        oa<< ta << percentile_spec<<use_ts_cached_read<<update_ts_cache;
        auto response_type = msg::read_type(io);
        if (response_type == message_type::SERVER_EXCEPTION) {
            auto re = msg::read_exception(io);
            throw re;
        } else if (response_type == message_type::EVALUATE_TS_VECTOR_PERCENTILES) {
            ts_vector_t r;
            core_iarchive ia(io,core_arch_flags);
            ia >> r;
            return r;
        }
        throw std::runtime_error(std::string("Got unexpected response:") + std::to_string((int)response_type));
    } else {
        vector<int> p_spec;
        bool can_do_server_side_average=true; // in case we are searching for min-max extreme, we can not do server-side average
        for(size_t i=0;i<percentile_spec.size();++i) {
            p_spec.push_back(int(percentile_spec[i]));
            if(percentile_spec[i] == statistics_property::MIN_EXTREME || percentile_spec[i]==statistics_property::MAX_EXTREME)
                can_do_server_side_average=false;
        }
        auto atsv = evaluate(can_do_server_side_average?tsv.average(ta):tsv,p,use_ts_cached_read,update_ts_cache); // get the result we can do percentiles on
        return shyft::time_series::dd::percentiles(atsv, ta, p_spec);
    }
}

std::vector<apoint_ts>
client::evaluate(ts_vector_t const& tsv, utcperiod p,bool use_ts_cached_read,bool update_ts_cache) {
    if (tsv.size() == 0)
        throw std::runtime_error("evaluate requires a source ts-vector with more than 0 time-series");
    if (!p.valid())
        throw std::runtime_error("percentiles require a valid period-specification");
    scoped_connect ac(*this);
    // local lambda to ensure one definition of communication with the server
    auto eval_io = [this] (dlib::iosockstream&io,const ts_vector_t& tsv,const utcperiod& p,bool use_ts_cached_read,bool update_ts_cache) {
            msg::write_type(compress_expressions?message_type::EVALUATE_EXPRESSION:message_type::EVALUATE_TS_VECTOR, io); {
            core_oarchive oa(io,core_arch_flags);
            oa << p ;
            if(compress_expressions) { // notice that we stream out all in once here
                // .. just in case the destruction of the compressed expr take time (it could..)
                oa<< expression_compressor::compress(tsv)<<use_ts_cached_read<<update_ts_cache;
            } else {
                oa<< tsv<<use_ts_cached_read<<update_ts_cache;
            }
        }
        auto response_type = msg::read_type(io);
        if (response_type == message_type::SERVER_EXCEPTION) {
            auto re = msg::read_exception(io);
            throw re;
        } else if (response_type == message_type::EVALUATE_TS_VECTOR) {
            ts_vector_t r; {
                core_iarchive ia(io,core_arch_flags);
                ia >> r;
            }
            return r;
        }
        throw std::runtime_error(std::string("Got unexpected response:") + std::to_string((int)response_type));
    };

    if(srv_con.size()==1 || tsv.size() == 1) { // one server, or just one ts, do it easy
        dlib::iosockstream& io = *(srv_con[0].io);
        return eval_io(io,tsv,p,use_ts_cached_read,update_ts_cache);
    } else {
        ts_vector_t rt(tsv.size()); // make place for the result contributions from threads
        // lamda to eval partition on server
        auto eval_partition= [&rt,&tsv,&eval_io,p,use_ts_cached_read,update_ts_cache]
            (dlib::iosockstream& io,size_t i0, size_t n) {
                ts_vector_t ptsv;ptsv.reserve(tsv.size());
                for(size_t i=i0;i<i0+n;++i) ptsv.push_back(tsv[i]);
                auto pt = eval_io(io,ptsv,p,use_ts_cached_read,update_ts_cache);
                for(size_t i=0;i<pt.size();++i)
                    rt[i0+i]=pt[i];
        };
        size_t partition_size = 1+tsv.size()/srv_con.size();// if tsv.size() < srv_con.size() ->
        vector<future<void>> calcs;
        for(size_t i=0;i<srv_con.size();++i) {
            size_t i0= i*partition_size;
            size_t n = min(partition_size,tsv.size()-i0);
            calcs.push_back(std::async(
                            std::launch::async,
                            [this,i,i0,n,&eval_partition] () {
                                eval_partition (*(srv_con[i].io),i0,n);
                            }
                           )
                  );
        }
        for (auto &f : calcs)
            f.get();

        return rt;
    }
}

void
client::store_ts(const ts_vector_t &tsv, bool overwrite_on_write, bool cache_on_write) {
    if (tsv.size() == 0)
        return; //trivial and considered valid case
                // verify that each member of tsv is a gpoint_ts
    for (auto const &ats : tsv) {
        auto rts = dynamic_cast<aref_ts*>(ats.ts.get());
        if (!rts) throw std::runtime_error(std::string("attempt to store a null ts"));
        if (rts->needs_bind()) throw std::runtime_error(std::string("attempt to store unbound ts:") + rts->id);
    }
    scoped_connect ac(*this);
    dlib::iosockstream& io = *(srv_con[0].io);
    msg::write_type(message_type::STORE_TS, io);
    {
        core_oarchive oa(io,core_arch_flags);
        oa << tsv << overwrite_on_write << cache_on_write;
    }
    auto response_type = msg::read_type(io);
    if (response_type == message_type::SERVER_EXCEPTION) {
        auto re = msg::read_exception(io);
        throw re;
    } else if (response_type == message_type::STORE_TS) {
        return;
    }
    throw std::runtime_error(std::string("Got unexpected response:") + std::to_string((int)response_type));
}

void
client::merge_store_ts(const ts_vector_t &tsv,bool cache_on_write) {
    if (tsv.size() == 0)
        return; //trivial and considered valid case
                // verify that each member of tsv is a gpoint_ts
    for (auto const &ats : tsv) {
        auto rts = dynamic_cast<aref_ts*>(ats.ts.get());
        if (!rts) throw std::runtime_error(std::string("attempt to store a null ts"));
        if (rts->needs_bind()) throw std::runtime_error(std::string("attempt to store unbound ts:") + rts->id);
    }
    scoped_connect ac(*this);
    dlib::iosockstream& io = *(srv_con[0].io);
    msg::write_type(message_type::MERGE_STORE_TS, io);
    {
        core_oarchive oa(io,core_arch_flags);
        oa << tsv << cache_on_write;
    }
    auto response_type = msg::read_type(io);
    if (response_type == message_type::SERVER_EXCEPTION) {
        auto re = msg::read_exception(io);
        throw re;
    } else if (response_type == message_type::MERGE_STORE_TS) {
        return;
    }
    throw std::runtime_error(std::string("Got unexpected response:") + std::to_string((int)response_type));
}

ts_info_vector_t
client::find(const std::string& search_expression) {
    scoped_connect ac(*this);
    auto& io = *(srv_con[0].io);
    msg::write_type(message_type::FIND_TS, io);
    {
        msg::write_string(search_expression, io);
    }
    auto response_type = msg::read_type(io);
    if (response_type == message_type::SERVER_EXCEPTION) {
        auto re = msg::read_exception(io);
        throw re;
    } else if (response_type == message_type::FIND_TS) {
        ts_info_vector_t r;
        {
            core_iarchive ia(io,core_arch_flags);
            ia >> r;
        }
        return r;
    }
    throw std::runtime_error(std::string("Got unexpected response:") + std::to_string((int)response_type));
}

void
client::cache_flush() {
    scoped_connect ac(*this);
    for(auto& sc:srv_con) {
        auto& io = *(sc.io);
        msg::write_type(message_type::CACHE_FLUSH, io);
        auto response_type = msg::read_type(io);
        if (response_type == message_type::SERVER_EXCEPTION) {
            auto re = msg::read_exception(io);
            throw re;
        } else if (response_type!=message_type::CACHE_FLUSH) {
            throw std::runtime_error(std::string("Got unexpected response:") + std::to_string((int)response_type));
        }
    }
}

cache_stats
client::get_cache_stats() {
    scoped_connect ac(*this);
    cache_stats s;
    for(auto& sc:srv_con) {
        auto& io = *(sc.io);
        msg::write_type(message_type::CACHE_STATS, io);
        auto response_type = msg::read_type(io);
        if (response_type==message_type::CACHE_STATS) {
            cache_stats r;
            core_iarchive oa(io,core_arch_flags);
            oa>>r;
            s= s+r;
        } else if (response_type == message_type::SERVER_EXCEPTION) {
            auto re = msg::read_exception(io);
            throw re;
        } else {
            throw std::runtime_error(std::string("Got unexpected response:") + std::to_string((int)response_type));
        }
    }
    return s;
}

}
}
