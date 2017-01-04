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

// from https://www.codevate.com/blog/7-concurrency-with-embedded-python-in-a-multi-threaded-c-application
namespace boost {
    namespace python {

        struct release_gil_policy {
            // Ownership of this argument tuple will ultimately be adopted by
            // the caller.
            template <class ArgumentPackage>
            static bool precall(ArgumentPackage const&) {
                // Release GIL and save PyThreadState for this thread here

                return true;
            }

            // Pass the result through
            template <class ArgumentPackage>
            static PyObject* postcall(ArgumentPackage const&, PyObject* result) {
                // Reacquire GIL using PyThreadState for this thread here

                return result;
            }

            typedef default_result_converter result_converter;
            typedef PyObject* argument_package;

            template <class Sig>
            struct extract_return_type : mpl::front<Sig> {
            };

        private:
            // Retain pointer to PyThreadState on a per-thread basis here

        };
    }
}

struct scoped_gil_release {
    scoped_gil_release() noexcept {
        py_thread_state = PyEval_SaveThread();
    }
    ~scoped_gil_release() noexcept {
        PyEval_RestoreThread(py_thread_state);
    }
    scoped_gil_release(const scoped_gil_release&) = delete;
    scoped_gil_release(scoped_gil_release&&) = delete;
    scoped_gil_release& operator=(const scoped_gil_release&) = delete;
private:
    PyThreadState * py_thread_state;
};

struct scoped_gil_aquire {
    scoped_gil_aquire() noexcept {
        py_state = PyGILState_Ensure();
    }
    ~scoped_gil_aquire() noexcept {
        PyGILState_Release(py_state);
    }
    scoped_gil_aquire(const scoped_gil_aquire&) = delete;
    scoped_gil_aquire(scoped_gil_aquire&&) = delete;
    scoped_gil_aquire& operator=(const scoped_gil_aquire&) = delete;
private:
    PyGILState_STATE   py_state;
};

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



        struct dtss_server : dlib::server_iostream {
            boost::python::object cb;
            dtss_server() {
                if (!PyEval_ThreadsInitialized()) {
                    std::cout << "InitThreads needed\n";
                    PyEval_InitThreads();// ensure threads-is enabled
                }
            }
            ~dtss_server() {
                std::cout << "~dtss()\n";
                cb = boost::python::object();
            }
            template<class TSV>
            std::vector<api::apoint_ts> do_evaluate_ts_vector(core::utcperiod bind_period, TSV&& atsv) {
                std::map<std::string,std::vector<api::ts_bind_info>> ts_bind_map;
                std::vector<std::string> ts_id_list;
                for (auto& ats : atsv) {
                    auto ts_refs = ats.find_ts_bind_info();
                    for(const auto& bi:ts_refs) {
                        if (ts_bind_map.find(bi.reference) == ts_bind_map.end()) { // maintain unique set
                            ts_id_list.push_back( bi.reference );
                            ts_bind_map[bi.reference] = std::vector<api::ts_bind_info>();
                        }
                        ts_bind_map[bi.reference].push_back(bi);
                    }
                }
                auto bts=fire_cb(ts_id_list,bind_period);
                if(bts.size()!=ts_id_list.size())
                    throw std::runtime_error(std::string("failed to bind all of ")+std::to_string(bts.size())+std::string(" ts"));
                for(size_t i=0;i<ts_id_list.size();++i) {
                    try {
                        //std::cout<<"bind "<<i<<": "<<ts_id_list[i]<<":"<<bts[i].size()<<"\n";
                        for(auto &bi:ts_bind_map[ts_id_list[i]])
                            bi.ts.bind(bts[i]);
                    } catch(const std::runtime_error&re) {
                        std::cout<<"failed to bind "<<ts_id_list[i]<<re.what()<<"\n";
                    }
                }
                //-- evaluate, when all binding is done (vectorized calc.
                std::vector<api::apoint_ts> evaluated_tsv;
                int i=0;
                for (auto &ats : atsv) {
                    try {
                        evaluated_tsv.emplace_back(ats.time_axis(), ats.values(), ats.point_interpretation());
                    } catch(const std::runtime_error&re) {
                        std::cout<<"failed to evalutate ts:"<<i<<"::"<<re.what()<<"\n";
                    }
                    i++;
                }
                return evaluated_tsv;
            }

            static int msg_count ;

            std::vector<api::apoint_ts> fire_cb(std::vector<std::string>ts_ids,core::utcperiod p) {
                std::vector<api::apoint_ts> r;
                if (cb.ptr()!=Py_None) {
                    scoped_gil_aquire gil;
                    r = boost::python::call<std::vector<api::apoint_ts>>(cb.ptr(), ts_ids, p);
                }
                return r;
            }
            void process_messages(int msec) {
                scoped_gil_release gil;
                if(!is_running()) start_async();
                std::this_thread::sleep_for(std::chrono::milliseconds(msec));
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
            scoped_gil_release gil;
            dlib::iosockstream io(host_port);
            int msg_type = EVALUATE_TS_VECTOR;
            io.write((const char*) &msg_type, sizeof(msg_type));
            io.write((const char*)&p, sizeof(p));
            write_ts_vector(tsv, io);
            return read_ts_vector(io);
        }
    }
}
// experiment with python doc standard macro helpers
#define doc_intro(intro) intro ## "\n"
#define doc_parameters() "\nParameters\n----------\n"
#define doc_parameter(name_str,type_str,descr_str) name_str ## " : " ## type_str ## "\n\t" ## descr_str ## "\n"
#define doc_paramcont(doc_str) "\t" ## doc_str ## "\n"
#define doc_returns(name_str,type_str,descr_str) "\nReturns\n-------\n" ## name_str ## " : " ## type_str ## "\n"
#define doc_notes() "\nNotes\n-----\n"
#define doc_note(note_str) note_str ## "\n"
#define doc_see_also(ref) "\nSee Also\n--------\n"##ref ## "\n"
#define doc_ind(doc_str) "\t" ## doc_str

namespace expose {
    //using namespace shyft::core;
    using namespace boost::python;

    static void dtss_messages() {

    }
    static void dtss_server() {
        typedef shyft::dtss::dtss_server DtsServer;
        //bases<>,std::shared_ptr<DtsServer>
        class_<DtsServer, boost::noncopyable >("DtsServer",
            doc_intro("A distributed time-series server object")
            doc_intro("Capable of processing time-series messages and responding accordingly")
            doc_intro("The user can setup callback to python to handle unbound symbolic time-series references")
            doc_intro("- that typically involve reading time-series from a service or storage for the specified period")
            doc_intro("The server object will then compute the resulting time-series vector,")
            doc_intro("and respond back to clients with the results")
            doc_see_also("shyft.api.dtss_evalutate(port_host,ts_array,utc_period)")
            )
            .def("set_listening_port", &DtsServer::set_listening_port, args("port_no"), 
                doc_intro("set the listening port for the service")
                doc_parameters()
                doc_parameter("port_no","int","a valid and available tcp-ip port number to listen on.")
                doc_paramcont("typically it could be 20000 (avoid using official reserved numbers)")
                doc_returns("nothing","None","")
            )
            .def("start_async",&DtsServer::start_async,
                doc_intro("start server listening in background, and processing messages")
                doc_see_also("set_listening_port(port_no),is_running,cb,process_messages(msec)")
                doc_notes()
                doc_note("you should have setup up the callback, cb before calling start_async")
                doc_note("Also notice that processing will acquire the GIL\n -so you need to release the GIL to allow for processing messages")
                doc_see_also("process_messages(msec)")
            )
            .def("set_max_connections",&DtsServer::set_max_connections,args("max_connect"),
                doc_intro("limits simultaneous connections to the server (it's multithreaded!)")
                doc_parameters()
                doc_parameter("max_connect","int","maximum number of connections before denying more connections")
                doc_see_also("get_max_connections()")
            )
            .def("get_max_connections",&DtsServer::get_max_connections,"tbd")
            .def("clear",&DtsServer::clear,
                doc_intro("stop serving connections, gracefully.")
                doc_see_also("cb, process_messages(msec),start_async()")
            )
            .def("is_running",&DtsServer::is_running,
                doc_intro("true if server is listening and running")
                doc_see_also("start_async(),process_messages(msec)")
            )
            .def("get_listening_port",&DtsServer::get_listening_port,"returns the port number it's listening at")
            .def_readwrite("cb",&DtsServer::cb,
                doc_intro("callback for binding unresolved time-series references to concrete time-series.")
                doc_intro("Called *if* the incoming messages contains unbound time-series.")
                doc_intro("The signature of the callback function should be TsVector cb(StringVector,utcperiod)")
                doc_intro("\nExamples\n--------\n")
                doc_intro(
                    "from shyft import api as sa\n\n"
                    "def resolve_and_read_ts(ts_ids,read_period):\n"
                    "    print('ts_ids:', len(ts_ids), ', read period=', str(read_period))\n"
                    "    ta = sa.Timeaxis2(read_period.start, sa.deltahours(1), read_period.timespan()//sa.deltahours(1))\n"
                    "    x_value = 1.0\n"
                    "    r = sa.TsVector()\n"
                    "    for ts_id in ts_ids :\n"
                    "        r.append(sa.Timeseries(ta, fill_value = x_value))\n"
                    "        x_value = x_value + 1\n"
                    "    return r\n"
                    "# and then bind the function to the callback\n"
                    "dtss=sa.DtsServer()\n"
                    "dtss.cb=resolve_and_read_ts\n"
                    "dtss.set_listening_port(20000)\n"
                    "dtss.process_messages(60000)\n"
                )
            )
            .def("fire_cb",&DtsServer::fire_cb,args("msg","rp"),"testing fire from c++")
            .def("process_messages",&DtsServer::process_messages,args("msec"),
                doc_intro("wait and process messages for specified number of msec before returning")
                doc_intro("the dtss-server is started if not already running")
                doc_parameters()
                doc_parameter("msec","int","number of millisecond to process messages")
                doc_notes()
                doc_note("this method releases GIL so that callbacks are not blocked when the\n" 
                    "dtss-threads perform the callback ")
                doc_see_also("cb,start_async(),is_running,clear()")
            )
            //.add_static_property("msg_count",
            //                     make_getter(&DtsServer::msg_count),
            //                     make_setter(&DtsServer::msg_count),"total number of requests")
            ;

    }
    static void dtss_client() {
        def("dtss_evaluate", shyft::dtss::dtss_evaluate, args("host_port","ts_vector","utcperiod"),
            doc_intro("Evaluates the expressions in the ts_vector for the specified utcperiod.")
            doc_intro("If the expression includes unbound symbolic references to time-series,")
            doc_intro("these time-series will be passed to the binding service callback")
            doc_intro("on the serverside.")
            doc_parameters()
            doc_parameter("host_port","string", "a string of the format 'host:portnumber', e.g. 'localhost:20000'")
            doc_parameter("ts_vector","TsVector","a list of time-series (expressions), including unresolved symbolic references")
            doc_parameter("utcperiod","UtcPeriod","the period that the binding service should read from the backing ts-store/ts-service")
            doc_returns("tsvector","TsVector","an evaluated list of point timeseries in the same order as the input list")
            doc_see_also("DtsServer,DtsServer.cb")
            );

    }

    void dtss() {
        dtss_messages();
        dtss_server();
        dtss_client();
    }
}
