#include "boostpython_pch.h"
#include <boost/python/docstring_options.hpp>

//-- for serialization:
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

//-- notice that boost serialization require us to
//   include shared_ptr/vector .. etc.. wherever it's needed

#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/vector.hpp>

//-- for the server
#include <dlib/server.h>
#include <dlib/iosockstream.h>

//#include "core/timeseries.h"
#include "api/timeseries.h"


namespace shyft {
    namespace dtss {

        template<class T>
        static api::apoint_ts read_ts(T& in) {
            int sz;
            in.read((char*)&sz, sizeof(sz));
            std::vector<char> blob(sz, 0);
            in.read((char*)blob.data(), sz);
            return api::apoint_ts::deserialize_from_bytes(blob);
        }

        template <class T>
        static void  write_ts(const api::apoint_ts& ats, T& out) {
            auto blob = ats.serialize_to_bytes();
            int sz = blob.size();
            out.write((const char*)&sz, sizeof(sz));
            out.write((const char*)blob.data(), sz);
        }

        template <class TSV,class T>
        static void write_ts_vector(TSV &&ats, T & out) {
            int sz = ats.size();
            out.write((const char*)&sz, sizeof(sz));
            for (const auto & ts : ats)
                write_ts(ts, out);
        }

        template<class T>
        static std::vector<api::apoint_ts> read_ts_vector(T& in) {
            int sz;
            in.read((char*)&sz, sizeof(sz));
            std::vector<api::apoint_ts> r;
            r.reserve(sz);
            for (int i = 0;i < sz;++i)
                r.push_back(read_ts(in));
            return r;
        }
        enum dtss_message {
            EVALUATE_TS_VECTOR,
            //CACHE
            //FIND
            //etc. etc.
        };



        class dtss_server : public dlib::server_iostream {
        public:
            template<class TSV>
            std::vector<api::apoint_ts> do_evaluate_ts_vector(core::utcperiod bind_period, TSV&& atsv) {
                //-- just for the testing create dummy-ts here.
                // later: collect all bind_info.ref, collect by match into list, then invoke handler,
                //        then invoke default handler for the remaining not matching a bind-handlers (like a regexpr ?)
                //
                core::calendar utc;
                time_axis::generic_dt ta(bind_period.start, core::deltahours(1), bind_period.timespan()/api::deltahours(1));
                api::apoint_ts dummy_ts(ta, 1.0, timeseries::POINT_AVERAGE_VALUE);
                for (auto& ats : atsv) {
                    auto ts_refs = ats.find_ts_bind_info();
                    // read all tsr here, then:
                    for (auto&bind_info : ts_refs) {
                        bind_info.ts.bind(dummy_ts);
                    }
                }
                //-- evaluate, when all binding is done (vectorized calc.
                std::vector<api::apoint_ts> evaluated_tsv;
                for (auto &ats : atsv)
                    evaluated_tsv.emplace_back(ats.time_axis(), ats.values(), ats.point_interpretation());
                return evaluated_tsv;
            }

            static int msg_count ;

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
                    int msg_type;
                    in.read((char*)&msg_type, sizeof(msg_type));
                    msg_count++;
                    switch ((dtss_message)msg_type) {
                        case EVALUATE_TS_VECTOR: {
                            core::utcperiod bind_period;
                            in.read((char*)&bind_period, sizeof(bind_period));
                            write_ts_vector(do_evaluate_ts_vector(bind_period, read_ts_vector(in)),out);
                        } break;
                        default:
                            throw std::runtime_error(std::string("Got unknown message type:") + std::to_string(msg_type));
                    }
                }
            }
        };
        int dtss::dtss_server::msg_count = 0;
        std::vector<api::apoint_ts> dtss_evaluate(std::string host_port,const std::vector<api::apoint_ts>& tsv, core::utcperiod p) {
            dlib::iosockstream io(host_port);
            int msg_type = EVALUATE_TS_VECTOR;
            io.write((const char*) &msg_type, sizeof(msg_type));
            io.write((const char*)&p, sizeof(p));
            write_ts_vector(tsv, io);
            return read_ts_vector(io);
        }
    }
}
namespace expose {
    //using namespace shyft::core;
    using namespace boost::python;

    static void dtss_messages() {

    }
    static void dtss_server() {
        typedef shyft::dtss::dtss_server DtsServer;
        //bases<>,std::shared_ptr<DtsServer>
        class_<DtsServer, boost::noncopyable >("DtsServer")
            .def("set_listening_port", &DtsServer::set_listening_port, args("port_no"), "tbd")
            .def("start_async",&DtsServer::start_async)
            .def("set_max_connections",&DtsServer::set_max_connections,args("max_connect"),"tbd")
            .def("get_max_connections",&DtsServer::get_max_connections,"tbd")
            .def("clear",&DtsServer::clear,"stop serving connections")
            .def("is_running",&DtsServer::is_running,"true if server is listening and running")
            .def("get_listening_port",&DtsServer::get_listening_port,"returns the port number it's listening at")
            ;

    }
    static void dtss_client() {
        def("dtss_evaluate", shyft::dtss::dtss_evaluate, args("host_port","ts_vector","utcperiod"),
            "tbd"
            );

    }

    void dtss() {
        dtss_messages();
        dtss_server();
        dtss_client();
    }
}
