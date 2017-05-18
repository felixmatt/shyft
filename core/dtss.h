#pragma once
#include "api/time_series.h"
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/base_object.hpp>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

namespace shyft {
    namespace dtss {
        enum message_type {
            SERVER_EXCEPTION,
            EVALUATE_TS_VECTOR,
            EVALUATE_TS_VECTOR_PERCENTILES,
            // EVALUATE_TS_VECTOR_HISTOGRAM //-- tsv,period,ta,bin_min,bin_max -> ts_vector[n_bins]
        };

        namespace msg {
            template <class T>
            message_type read_type(T&in) {
                int32_t mtype;
                in.read((char*)&mtype,sizeof(mtype));
                return (message_type) mtype;
            }

            template <class T>
            void write_type(message_type mt,T&out) {
                int32_t mtype=(int32_t)mt;
                out.write((char const *)&mtype,sizeof(mtype));
            }



            template <class T>
            void write_string(std::string const&s,T& out) {
                int32_t sz=s.size();
                out.write((char const*)&sz,sizeof(sz));
                out.write(s.data(),sz);
            }

            template<class T>
            void write_exception(std::exception const & e,T& out) {
                int32_t sz=strlen(e.what());
                out.write((char const*)&sz,sizeof(sz));
                out.write(e.what(),sz);
            }

            template<class T>
            void send_exception(std::exception const & e,T& out) {
                write_type(message_type::SERVER_EXCEPTION,out);
                int32_t sz=strlen(e.what());
                out.write((char const*)&sz,sizeof(sz));
                out.write(e.what(),sz);
            }

            template<class T>
            std::runtime_error read_exception(T& in) {
                int32_t sz;
                in.read((char*)&sz,sizeof(sz));
                std::string msg(sz,'\0');
                in.read((char*)msg.data(),sz);
                return std::runtime_error(msg);
            }

        }

        typedef std::vector<api::apoint_ts> ts_vector_t;
        typedef std::vector<std::string> id_vector_t;
        typedef std::function< ts_vector_t (id_vector_t const& ts_ids,core::utcperiod p)> call_back_t;

        struct server : dlib::server_iostream {
            call_back_t bind_ts_cb;

            template <class CB>
            server(CB&& cb):bind_ts_cb(std::forward<CB>(cb)) {
            }
            ~server() {}

            void
            do_bind_ts(core::utcperiod bind_period, ts_vector_t& atsv) const {
                std::map<std::string, std::vector<api::ts_bind_info>> ts_bind_map;
                std::vector<std::string> ts_id_list;
                // step 1: bind not yet bound time-series ( ts with only symbol, needs to be resolved using bind_cb)
                //std::cout<<"bind 1\n"<<std::endl;
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
                //std::cout<<"bind 2\n"<<std::endl;

                // step 2: (optional) bind_ts callback should resolve symbol time-series with content
                if (ts_bind_map.size()) {
                    auto bts = bind_ts_cb(ts_id_list, bind_period);
                    if (bts.size() != ts_id_list.size())
                        throw std::runtime_error(std::string("failed to bind all of ") + std::to_string(bts.size()) + std::string(" ts"));

                    for (size_t i = 0;i < ts_id_list.size();++i) {
                        for (auto &bi : ts_bind_map[ts_id_list[i]])
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
            do_evaluate_ts_vector(core::utcperiod bind_period, ts_vector_t& atsv) {
                do_bind_ts(bind_period, atsv);
                return api::deflate_ts_vector<api::apoint_ts>(atsv);
            }

            ts_vector_t
            do_evaluate_percentiles(core::utcperiod bind_period, ts_vector_t& atsv, api::gta_t const&ta,std::vector<int> const& percentile_spec) {
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
                //std::cout<<"on conn:"<<foreign_ip<<":"<<foreign_port<<", local_port ="<<local_port<<std::endl;
                while (in.peek() != EOF) {
                    auto msg_type= msg::read_type(in);
                    try {
                        switch (msg_type) { // currently switch, later maybe table[msg_type]=msg_handler
                            case EVALUATE_TS_VECTOR: {
                                core::utcperiod bind_period;
                                ts_vector_t rtsv;
                                {
                                    boost::archive::binary_iarchive ia(in);
                                    ia>>bind_period>>rtsv;
                                }

                                auto result=do_evaluate_ts_vector(bind_period, rtsv);// first get result
                                {
                                    msg::write_type(message_type::EVALUATE_TS_VECTOR,out);// then send
                                    boost::archive::binary_oarchive oa(out);
                                    oa<<result;
                                }
                            } break;
                            case EVALUATE_TS_VECTOR_PERCENTILES: {
                                core::utcperiod bind_period;
                                ts_vector_t rtsv;
                                std::vector<int> percentile_spec;
                                api::gta_t ta;
                                {
                                    boost::archive::binary_iarchive ia(in);
                                    ia >> bind_period >> rtsv>>ta>>percentile_spec;
                                }

                                auto result = do_evaluate_percentiles(bind_period, rtsv,ta,percentile_spec);
                                {
                                    msg::write_type(message_type::EVALUATE_TS_VECTOR_PERCENTILES, out);
                                    boost::archive::binary_oarchive oa(out);
                                    oa << result;
                                }
                            } break;
                            default:
                                throw std::runtime_error(std::string("Got unknown message type:") + std::to_string((int)msg_type));
                        }
                    } catch (std::exception const& e) {
                        msg::send_exception(e,out);
                    }
                }
                //std::cout<<"of conn:"<<foreign_ip<<":"<<foreign_port<<", local_port ="<<local_port<<std::endl;

            }
        };

        struct client {
            dlib::iosockstream io;
            std::string host_port;
            client(std::string host_port):io(host_port),host_port(host_port) {}

            void reopen() {
                io.close();
                io.open(host_port);
            }
            void close(int timeout_ms=1000) {io.close(timeout_ms);}

            std::vector<api::apoint_ts>
            percentiles(ts_vector_t const& tsv, core::utcperiod p,api::gta_t const&ta,std::vector<int> percentile_spec) {
                msg::write_type(message_type::EVALUATE_TS_VECTOR_PERCENTILES, io);
                {
                    boost::archive::binary_oarchive oa(io);
                    oa << p << tsv<<ta<<percentile_spec;
                }
                auto response_type = msg::read_type(io);
                if (response_type == message_type::SERVER_EXCEPTION) {
                    auto re = msg::read_exception(io);
                    throw re;
                } else if (response_type == message_type::EVALUATE_TS_VECTOR_PERCENTILES) {
                    ts_vector_t r;
                    {
                        boost::archive::binary_iarchive ia(io);
                        ia >> r;
                    }
                    return r;
                }
                throw std::runtime_error(std::string("Got unexpected response:") + std::to_string((int)response_type));
            }

            std::vector<api::apoint_ts>
            evaluate(ts_vector_t const& tsv, core::utcperiod p) {
                msg::write_type(message_type::EVALUATE_TS_VECTOR,io);
                {
                    boost::archive::binary_oarchive oa(io);
                    oa<<p<<tsv;
                }
                auto response_type= msg::read_type(io);
                if(response_type==message_type::SERVER_EXCEPTION) {
                    auto re= msg::read_exception(io);
                    throw re;
                } else if(response_type==message_type::EVALUATE_TS_VECTOR) {
                    ts_vector_t r;
                    {
                        boost::archive::binary_iarchive ia(io);
                        ia>>r;
                    }
                    return r;
                }
                throw std::runtime_error(std::string("Got unexpected response:")+std::to_string((int)response_type));
            }

        };

        inline std::vector<api::apoint_ts> dtss_evaluate(std::string host_port, ts_vector_t const& tsv, core::utcperiod p,int timeout_ms=10000) {
            return client(host_port).evaluate(tsv,p);
        }
        inline std::vector<api::apoint_ts> dtss_percentiles(std::string host_port, ts_vector_t const& tsv, core::utcperiod p,api::gta_t const&ta,std::vector<int> percentile_spec,int timeout_ms=10000) {
            return client(host_port).percentiles(tsv,p,ta,percentile_spec);
        }


    }
}
