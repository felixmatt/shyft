#include "boostpython_pch.h"

#include "api/time_series.h"
#include "core/dtss.h"
#include "core/dtss_client.h"

// also consider policy: from https://www.codevate.com/blog/7-concurrency-with-embedded-python-in-a-multi-threaded-c-application

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

        struct py_server : server {
            boost::python::object cb;///< callback for the read function
            boost::python::object fcb;///< callback for the find function
            boost::python::object scb;///< callback for the store function

            py_server():server(
                [=](id_vector_t const &ts_ids,core::utcperiod p){return this->fire_cb(ts_ids,p); },
                [=](std::string search_expression) {return this->find_cb(search_expression); },
                [=](const ts_vector_t& tsv) {  this->store_cb(tsv);}
            ) {
                if (!PyEval_ThreadsInitialized()) {
                    PyEval_InitThreads();// ensure threads-is enabled
                }
            }
            ~py_server() {
                cb = boost::python::object();
                fcb = boost::python::object();
				scb = boost::python::object();
            }

            void handle_pyerror() {
                // from SO: https://stackoverflow.com/questions/1418015/how-to-get-python-exception-text
                using namespace boost::python;
                using namespace boost;
                std::string msg{"unspecified error"};
                if(PyErr_Occurred()) {
                    PyObject *exc,*val,*tb;
                    object formatted_list, formatted;
                    PyErr_Fetch(&exc,&val,&tb);
                    handle<> hexc(exc),hval(allow_null(val)),htb(allow_null(tb));
                    object traceback(import("traceback"));
                    if (!tb) {
                        object format_exception_only{ traceback.attr("format_exception_only") };
                        formatted_list = format_exception_only(hexc,hval);
                    } else {
                        object format_exception{traceback.attr("format_exception")};
                        if (format_exception) {
                            try {
                                formatted_list = format_exception(hexc, hval, htb);
                            } catch (...) { // any error here, and we bail out, no crash please
                                msg = "not able to extract exception info";
                            }
                        } else
                            msg="not able to extract exception info";
                    }
                    if (formatted_list) {
                        formatted = str("\n").join(formatted_list);
                        msg = extract<std::string>(formatted);
                    }
                }
                handle_exception();
                PyErr_Clear();
                throw std::runtime_error(msg);
            }

            static int msg_count ;
            ts_info_vector_t find_cb(std::string search_expression) {
                ts_info_vector_t r;
                if (fcb.ptr() != Py_None) {
                    scoped_gil_aquire gil;
                    try {
                        r = boost::python::call<ts_info_vector_t>(fcb.ptr(), search_expression);
                    } catch  (const boost::python::error_already_set&) {
                        handle_pyerror();
                    }
                }
                return r;
            }
            int store_cb(const ts_vector_t&tsv) {
                int r{ 0 };
                if (scb.ptr() != Py_None) {
                    scoped_gil_aquire gil;
                    try {
                         boost::python::call<void>(scb.ptr(), tsv);
                    } catch  (const boost::python::error_already_set&) {
                        handle_pyerror();
                    }
                }
                return r;
            }
            ts_vector_t fire_cb(id_vector_t const &ts_ids,core::utcperiod p) {
                api::ats_vector r;
                if (cb.ptr()!=Py_None) {
                    scoped_gil_aquire gil;
                    try {
                        r = boost::python::call<ts_vector_t>(cb.ptr(), ts_ids, p);
                    } catch  (const boost::python::error_already_set&) {
                        handle_pyerror();
                    }
                } else {
                    // for testing, just fill in constant values.
                    api::gta_t ta(p.start, core::deltahours(1), p.timespan() / core::deltahours(1));
                    for (size_t i = 0;i < ts_ids.size();++i)
                        r.push_back(api::apoint_ts(ta, double(i), time_series::ts_point_fx::POINT_AVERAGE_VALUE));
                }
                return r;
            }
            void process_messages(int msec) {
                scoped_gil_release gil;
                if(!is_running()) start_async();
                std::this_thread::sleep_for(std::chrono::milliseconds(msec));
            }
        };
        int py_server::msg_count = 0;
        // need to wrap core client to unlock gil during processing
        struct py_client {
            client impl;
            explicit py_client(const std::string& host_port):impl(host_port) {}
            ~py_client() {
                //std::cout << "~py_client" << std::endl;
            }
            py_client(py_client const&) = delete;
            py_client(py_client &&) = delete;
            py_client& operator=(py_client const&o) = delete;

            void close(int timeout_ms=1000) {
                scoped_gil_release gil;
                impl.close(timeout_ms);
            }
            ts_vector_t percentiles(const ts_vector_t & tsv, core::utcperiod p,const api::gta_t &ta,const std::vector<int>& percentile_spec) {
                scoped_gil_release gil;
                return impl.percentiles(tsv,p,ta,percentile_spec);
            }
            ts_vector_t evaluate(const ts_vector_t& tsv, core::utcperiod p) {
                scoped_gil_release gil;
                return ts_vector_t(impl.evaluate(tsv,p));
            }
            ts_info_vector_t find(const std::string& search_expression) {
                scoped_gil_release gil;
                return impl.find(search_expression);
            }
            void store_ts(const ts_vector_t&tsv, bool overwrite_on_write, bool cache_on_write) {
                scoped_gil_release gil;
                impl.store_ts(tsv, overwrite_on_write, cache_on_write);
            }

        };
    }
}


namespace expose {

    using namespace boost::python;
	namespace py = boost::python;
    void dtss_finalize() {
#ifdef _WIN32
        WSACleanup();
#endif
    }
    static void dtss_messages() {
        def("dtss_finalize", dtss_finalize, "dlib socket and timer cleanup before exit python(automatically called once at module exit)");

        typedef shyft::dtss::ts_info TsInfo;
        class_<TsInfo>("TsInfo",
            doc_intro("Gives some information from the backend ts data-store")
            doc_intro("about the stored time-series, that could be useful in some contexts")
			,init<>(py::arg("self"))
            )
            .def(init<std::string, shyft::time_series::ts_point_fx, shyft::core::utctimespan, std::string, shyft::core::utcperiod, shyft::core::utctime, shyft::core::utctime>(
                (py::arg("self"),py::arg("name"), py::arg("point_fx"), py::arg("delta_t"), py::arg("olson_tz_id"), py::arg("data_period"), py::arg("created"), py::arg("modified")),
                doc_intro("construct a TsInfo with all values specified")
                )
            )
            .def_readwrite("name", &TsInfo::name,
                doc_intro("the unique name")
            )
            .def_readwrite("point_fx", &TsInfo::point_fx,
                doc_intro("how to interpret the points, instant value, or average over period")
            )
            .def_readwrite("delta_t", &TsInfo::delta_t, 
                doc_intro("time-axis steps, in seconds, 0 if irregular time-steps")
            )
            .def_readwrite("olson_tz_id", &TsInfo::olson_tz_id,

                doc_intro("empty or time-axis calendar for calendar,t0,delta_t type time-axis")
            )
            .def_readwrite("data_period", &TsInfo::data_period,
                doc_intro("the period for data-stored, if applicable")
            )
            .def_readwrite("created", &TsInfo::created, 
                doc_intro("when time-series was created, seconds 1970s utc")
            )
            .def_readwrite("modified", &TsInfo::modified,
                doc_intro("when time-series was last modified, seconds 1970 utc")
            )
            .def(self == self)
            .def(self != self)
            ;

        typedef std::vector<TsInfo> TsInfoVector;
        class_<TsInfoVector>("TsInfoVector",
            doc_intro("A vector/list of TsInfo")
			,init<>(py::arg("self"))
            )
            .def(vector_indexing_suite<TsInfoVector>())
            .def(init<const TsInfoVector&>( (py::arg("self"),py::arg("clone_me"))))
            ;

    }
    static void dtss_server() {
        typedef shyft::dtss::py_server DtsServer;
        class_<DtsServer, boost::noncopyable >("DtsServer",
            doc_intro("A distributed time-series server object")
            doc_intro("Capable of processing time-series messages and responding accordingly")
            doc_intro("The user can setup callback to python to handle unbound symbolic time-series references")
            doc_intro("- that typically involve reading time-series from a service or storage for the specified period")
            doc_intro("The server object will then compute the resulting time-series vector,")
            doc_intro("and respond back to clients with the results")
			doc_intro("multi-node considerations:")
			doc_intro("    1. firewall/routing: ensure that the port you are using are open for ip-traffic")
			doc_intro("    2. currently we only support heterogenous client/server types(e.g. windows-windows, linux-linux)")
            doc_see_also("shyft.api.DtsClient"),
			init<>(args("self"))
            )
            .def("set_listening_port", &DtsServer::set_listening_port, args("self","port_no"),
                doc_intro("set the listening port for the service")
                doc_parameters()
                doc_parameter("port_no","int","a valid and available tcp-ip port number to listen on.")
                doc_paramcont("typically it could be 20000 (avoid using official reserved numbers)")
                doc_returns("nothing","None","")
            )
            .def("start_async",&DtsServer::start_async,(py::arg("self")),
                doc_intro("start server listening in background, and processing messages")
                doc_see_also("set_listening_port(port_no),is_running,cb,process_messages(msec)")
                doc_notes()
                doc_note("you should have setup up the callback, cb before calling start_async")
                doc_note("Also notice that processing will acquire the GIL\n -so you need to release the GIL to allow for processing messages")
                doc_see_also("process_messages(msec)")
            )
            .def("set_max_connections",&DtsServer::set_max_connections,(py::arg("self"),py::arg("max_connect")),
                doc_intro("limits simultaneous connections to the server (it's multithreaded, and uses on thread pr. connect)")
                doc_parameters()
                doc_parameter("max_connect","int","maximum number of connections before denying more connections")
                doc_see_also("get_max_connections()")
            )
            .def("get_max_connections",&DtsServer::get_max_connections, (py::arg("self")),
				doc_intro("returns the maximum number of connections to be served concurrently"))
            .def("clear",&DtsServer::clear, (py::arg("self")),
                doc_intro("stop serving connections, gracefully.")
                doc_see_also("cb, process_messages(msec),start_async()")
            )
            .def("is_running",&DtsServer::is_running, (py::arg("self")),
                doc_intro("true if server is listening and running")
                doc_see_also("start_async(),process_messages(msec)")
            )
            .def("get_listening_port",&DtsServer::get_listening_port, (py::arg("self")),
				"returns the port number it's listening at for serving incoming request"
			)
            .def_readwrite("cb",&DtsServer::cb,
                doc_intro("callback for binding unresolved time-series references to concrete time-series.")
                doc_intro("Called *if* the incoming messages contains unbound time-series.")
                doc_intro("The signature of the callback function should be TsVector cb(StringVector,utcperiod)")
                doc_intro("\nExamples\n--------\n")
                doc_intro(
                    "from shyft import api as sa\n\n"
                    "def resolve_and_read_ts(ts_ids,read_period):\n"
                    "    print('ts_ids:', len(ts_ids), ', read period=', str(read_period))\n"
                    "    ta = sa.TimeAxis(read_period.start, sa.deltahours(1), read_period.timespan()//sa.deltahours(1))\n"
                    "    x_value = 1.0\n"
                    "    r = sa.TsVector()\n"
                    "    for ts_id in ts_ids :\n"
                    "        r.append(sa.TimeSeries(ta, fill_value = x_value))\n"
                    "        x_value = x_value + 1\n"
                    "    return r\n"
                    "# and then bind the function to the callback\n"
                    "dtss=sa.DtsServer()\n"
                    "dtss.cb=resolve_and_read_ts\n"
                    "dtss.set_listening_port(20000)\n"
                    "dtss.process_messages(60000)\n"
                )
            )
            .def_readwrite("find_cb", &DtsServer::fcb,
                doc_intro("callback for finding time-series using a search-expression.")
                doc_intro("Called everytime the .find() method is called.")
                doc_intro("The signature of the callback function should be fcb(search_expr: str)->TsInfoVector")
                doc_intro("\nExamples\n--------\n")
                doc_intro(
                    "from shyft import api as sa\n\n"
                    "def find_ts(search_expr: str)->sa.TsInfoVector:\n"
                    "    print('find:',search_expr)\n"
                    "    r = sa.TsInfoVector()\n"
                    "    tsi = sa.TsInfo()\n"
                    "    tsi.name = 'some_test'\n"
                    "    r.append(tsi)\n"
                    "    return r\n"
                    "# and then bind the function to the callback\n"
                    "dtss=sa.DtsServer()\n"
                    "dtss.find_cb=find_ts\n"
                    "dtss.set_listening_port(20000)\n"
                    "# more code to invoce .find etc.\n"
                )
            )
            .def_readwrite("store_ts_cb",&DtsServer::scb,
                doc_intro("callback for storing time-series.")
                doc_intro("Called everytime the .store_ts() method is called and non-shyft urls are passed.")
                doc_intro("The signature of the callback function should be scb(tsv: TsVector)->None")
                doc_intro("\nExamples\n--------\n")
                doc_intro(
                    "from shyft import api as sa\n\n"
                    "def store_ts(tsv:sa.TsVector)->None:\n"
                    "    print('store:',len(tsv))\n"
                    "    # each member is a bound ref_ts with an url\n"
                    "    # extract the url, decode and store\n"
                    "    #\n"
                    "    #\n"
                    "    return\n"
                    "# and then bind the function to the callback\n"
                    "dtss=sa.DtsServer()\n"
                    "dtss.store_ts_cb=store_ts\n"
                    "dtss.set_listening_port(20000)\n"
                    "# more code to invoce .store_ts etc.\n"
                )
            )

            .def("fire_cb",&DtsServer::fire_cb,(py::arg("self"),py::arg("msg"),py::arg("rp")),"testing fire cb from c++")
            .def("process_messages",&DtsServer::process_messages,(py::arg("self"),py::arg("msec")),
                doc_intro("wait and process messages for specified number of msec before returning")
                doc_intro("the dtss-server is started if not already running")
                doc_parameters()
                doc_parameter("msec","int","number of millisecond to process messages")
                doc_notes()
                doc_note("this method releases GIL so that callbacks are not blocked when the\n"
                    "dtss-threads perform the callback ")
                doc_see_also("cb,start_async(),is_running,clear()")
            )
            .def("set_container",&DtsServer::add_container,(py::arg("self"),py::arg("name"),py::arg("root_dir")),
                 doc_intro("set ( or replaces) an internal shyft store container to the dtss-server.")
                 doc_intro("All ts-urls with shyft://<container>/ will resolve")
                 doc_intro("to this internal time-series storage for find/read/store operations")
                 doc_parameters()
                 doc_parameter("name","str","Name of the container as pr. url definition above")
                 doc_parameter("root_dir","str","A valid directory root for the container")
                 doc_notes()
                 doc_note("currently this call should only be used when the server is not processing messages\n"
                          "- before starting, or after stopping listening operations\n"
                          )

            )
            .def("set_auto_cache",&DtsServer::set_auto_cache,(py::arg("self"),py::arg("active")),
                doc_intro("set auto caching all reads active or passive.")
                doc_intro("Default is off, and caching must be done through")
                doc_intro("explicit calls to .cache(ts_ids,ts_vector)")
                doc_parameters()
                doc_parameter("active","bool","if set True, all reads will be put into cache")
            )
            .def("cache",&DtsServer::add_to_cache,(py::arg("self"),py::arg("ts_ids"),py::arg("ts_vector")),
                doc_intro("add/update specified ts_ids with corresponding ts to cache")
                doc_intro("please notice that there is no validation of the tds_ids, they")
                doc_intro("are threated identifiers,not verified against any existing containers etc.")
                doc_intro("Requests that follows, will use the cached item as long as it satisfies")
                doc_intro("the identifier and the coverage period requested")

                doc_parameters()
                doc_parameter("ts_ids","StringVector","a list of time-series ids")
                doc_parameter("ts_vector","TsVector","a list of corresponding time-series")
            )
            .def("flush_cache",&DtsServer::remove_from_cache,(py::arg("self"),py::arg("ts_ids")),
                doc_intro("flushes the *specified* ts_ids from cache")
                doc_intro("Has only effect for ts-ids that are in cache, non-existing items are ignored")
                doc_parameters()
                doc_parameter("ts_ids","StringVector","a list of time-series ids to flush out")
            )
            .def("flush_cache_all",&DtsServer::flush_cache,(py::arg("self")),
                doc_intro("flushes all items out of cache (cache_stats remain un-touched)")
            )
            .add_property("cache_stats",&DtsServer::get_cache_stats,
                doc_intro("return the current cache statistics")
            )
            .def("clear_cache_stats",&DtsServer::clear_cache_stats,(py::arg("self")),
                doc_intro("clear accumulated cache_stats")
            )
            .add_property("cache_max_items",&DtsServer::get_cache_size,&DtsServer::set_cache_size,
                doc_intro("cache_max_items is the maximum number of time-series identities that are")
                doc_intro("kept in memory. Elements exceeding this capacity is elided using the least-recently-used")
                doc_intro("algorithm. Notice that assigning a lower value than the existing value will also flush out")
                doc_intro("time-series from cache in the least recently used order.")
            )
            ;

    }
    static void dtss_client() {
        typedef shyft::dtss::py_client  DtsClient;
		class_<DtsClient, boost::noncopyable >("DtsClient",
			doc_intro("The client part of the DtsServer")
			doc_intro("Capable of processing time-series messages and responding accordingly")
			doc_intro("The user can setup callback to python to handle unbound symbolic time-series references")
			doc_intro("- that typically involve reading time-series from a service or storage for the specified period")
			doc_intro("The server object will then compute the resulting time-series vector,")
			doc_intro("and respond back to clients with the results")
			doc_see_also("DtsServer"), no_init
			)
			.def(init<std::string>((py::arg("self"), py::arg("host_port")),
				doc_intro("constructs a dts-client with the specifed host_port parameter")
				doc_intro("a connection is immediately done to the server at specified port.")
				doc_intro("If no such connection can be made, it raises a RuntimeError.")
				doc_parameter("host_port", "string", "a string of the format 'host:portnumber', e.g. 'localhost:20000'")
				)
			)
			.def("close", &DtsClient::close, (py::arg("self"), py::arg("timeout_ms") = 1000),
                doc_intro("close the connection")
            )
            .def("percentiles",&DtsClient::percentiles, (py::arg("self"),py::arg("ts_vector"), py::arg("utcperiod"), py::arg("time_axis"), py::arg("percentile_list")),
                doc_intro("Evaluates the expressions in the ts_vector for the specified utcperiod.")
                doc_intro("If the expression includes unbound symbolic references to time-series,")
                doc_intro("these time-series will be passed to the binding service callback")
                doc_intro("on the serverside.")
                doc_parameters()
                doc_parameter("ts_vector","TsVector","a list of time-series (expressions), including unresolved symbolic references")
                doc_parameter("utcperiod","UtcPeriod","the period that the binding service should read from the backing ts-store/ts-service")
                doc_parameter("time_axis","TimeAxis","the time_axis for the percentiles, e.g. a weekly time_axis")
                doc_parameter("percentile_list","IntVector","a list of percentiles, where -1 means true average, 25=25percentile etc")
                doc_returns("tsvector","TsVector","an evaluated list of percentile time-series in the same order as the percentile input list")
                doc_see_also(".evaluate(), DtsServer")
            )
            .def("evaluate", &DtsClient::evaluate, (py::arg("self"),py::arg("ts_vector"), py::arg("utcperiod") ),
                doc_intro("Evaluates the expressions in the ts_vector for the specified utcperiod.")
                doc_intro("If the expression includes unbound symbolic references to time-series,")
                doc_intro("these time-series will be passed to the binding service callback")
                doc_intro("on the serverside.")
                doc_parameters()
                doc_parameter("ts_vector","TsVector","a list of time-series (expressions), including unresolved symbolic references")
                doc_parameter("utcperiod","UtcPeriod","the period that the binding service should read from the backing ts-store/ts-service")
                doc_returns("tsvector","TsVector","an evaluated list of point time-series in the same order as the input list")
                doc_see_also(".percentiles(),DtsServer")
            )
            .def("find",&DtsClient::find,(py::arg("self"),py::arg("search_expression")),
                doc_intro("find ts information that matches the search-expression")
                doc_parameters()
                doc_parameter("search_expression","str","search-expression, to be interpreted by the back-end tss server (usually by callback to python)")
                doc_returns("ts_info_vector","TsInfoVector","The search result, as vector of TsInfo objects")
                doc_see_also("TsInfo,TsInfoVector")
            )
            .def("store_ts", &DtsClient::store_ts,
                (   py::arg("self"),
                    py::arg("tsv"),
                    py::arg("overwrite_on_write") = true,
                    py::arg("cache_on_write") = false
                ),
                doc_intro("Store the time-series in the ts-vector in the dtss backend.")
                doc_intro("The current internal shyft-container implementation overwrite the data by default, but can")
                doc_intro("update existing data if called with `overwrite_on_write = False`.")
				doc_intro("The intention is that the backend should overwrite the time-period they cover with")
                doc_intro("the new time-points unless asked explisitly to update it.")
                doc_intro("The time-series should be created like this, with url and a concrete point-ts:")
                doc_intro("")
                doc_intro(">>>   a=sa.TimeSeries(ts_url,ts_points)")
                doc_intro(">>>   tsv.append(a)")
                doc_parameters()
                doc_parameter("tsv","TsVector","ts-vector with time-series, url-reference and values to be stored at dtss server")
                doc_parameter("overwrite_on_write", "bool",
                    "When False the backend updates existing data matching the time-series to be saved instead of overwriting.\n"
                    "    Defaults to True.")
                doc_parameter("cache_on_write", "bool", "if set True, the written contents is also put into the read-cache of the dtss, defaults to False")
                doc_returns("None","","")
                doc_see_also("TsVector")
            )
            ;

    }
    void dtss_cache_stats() {
        using CacheStats = shyft::dtss::cache_stats;
        class_<CacheStats>("CacheStats",
            doc_intro("Cache statistics for the DtsServer."),
			init<>(py::arg("self"))
            )
            .def_readwrite("hits", &CacheStats::hits,
                doc_intro("number of hits by time-series id")
            )
            .def_readwrite("misses", &CacheStats::misses,
                doc_intro("number of misses by time-series id")
            )
            .def_readwrite("coverage_misses", &CacheStats::coverage_misses,
                doc_intro("number of misses where we did find the time-series id, but the period coverage was insufficient")
            )
            .def_readwrite("id_count", &CacheStats::id_count,

                doc_intro("number of unique time-series identities in cache")
            )
            .def_readwrite("point_count", &CacheStats::point_count,
                doc_intro("total number of time-series points in the cache")
            )
            .def_readwrite("fragment_count", &CacheStats::fragment_count,
                doc_intro("number of time-series fragments in the cache, (greater or equal to id_count)")
            )
            ;

    }

    void dtss() {
        dtss_messages();
        dtss_server();
        dtss_client();
        dtss_cache_stats();
    }
}
