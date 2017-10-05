#include "test_pch.h"

#include "core/dtss.h"
#include "core/utctime_utilities.h"
#include "core/time_axis.h"
#include "api/time_series.h"
#include <regex>
#include <boost/filesystem.hpp>
#include <cstdint>
#ifdef _WIN32
#include <io.h>
#else
#include <sys/io.h>
#define O_BINARY 0
#define O_SEQUENTIAL 0
#include <sys/stat.h>
#endif
#include <fcntl.h>

namespace  fs=boost::filesystem;

using namespace std;
using namespace shyft;
using namespace shyft::core;

api::apoint_ts mk_expression(utctime t, utctimespan dt, int n) {

    std::vector<double> x; x.reserve(n);
    for (int i = 0; i < n; ++i)
        x.push_back(-double(n) / 2.0 + i);
    api::apoint_ts aa(api::gta_t(t, dt, n), x);
    auto a = aa*3.0 + aa;
    return a;
}
namespace shyft {
    namespace dtss {
        using namespace shyft::api;

    }
}
dlib::logger dlog("dlib.log");

TEST_SUITE("dtss") {

TEST_CASE("dlib_server_basics") {
    dlog.set_level(dlib::LALL);
    dlib::set_all_logging_output_streams(std::cout);
    dlib::set_all_logging_levels(dlib::LALL);
    dlog << dlib::LINFO << "Starting dtss test";
    using namespace shyft::dtss;
    try {
        // Tell the server to begin accepting connections.
        calendar utc;
        auto t = utc.time(2016, 1, 1);
        auto dt = deltahours(1);
        auto dt24 = deltahours(24);
        int n = 240;
        int n24 = n / 24;
        shyft::time_axis::fixed_dt ta(t, dt, n);
        api::gta_t ta24(t, dt24, n24);
        bool throw_exception = false;
        read_call_back_t cb = [ta, &throw_exception](id_vector_t ts_ids, core::utcperiod p)
            ->ts_vector_t {
            ts_vector_t r; r.reserve(ts_ids.size());
            double fv = 1.0;
            for (size_t i = 0; i < ts_ids.size(); ++i)
                r.emplace_back(ta, fv += 1.0);
            if (throw_exception) {
                dlog << dlib::LINFO << "Throw from inside dtss executes!";
                throw std::runtime_error("test exception");
            }
            return r;
        };
        std::vector<std::string> ts_names = {
            string("a.prod.mw"),
            string("b.prod.mw")
        };
        find_call_back_t fcb = [&ts_names](std::string search_expression)
            ->ts_info_vector_t {
            ts_info_vector_t r;
            dlog << dlib::LINFO << "find-callback with search-string:" << search_expression;
            std::regex re(search_expression);
            auto match_end = std::sregex_iterator();
            for (auto const&tsn : ts_names) {
                if (std::sregex_iterator(tsn.begin(), tsn.end(), re)!= match_end) {
                    ts_info tsi; tsi.name = tsn;
                    r.push_back(tsi);
                }
            }
            return r;
        };

        server our_server(cb,fcb);

        // set up the server object we have made
        our_server.set_listening_ip("127.0.0.1");
        int port_no = 20000;
        our_server.set_listening_port(port_no);
        our_server.start_async();
        //while(our_server.is_running()&& our_server.get_listening_port()==0) //because dlib do not guarantee that listening port is set
        //    std::this_thread::sleep_for(std::chrono::milliseconds(3)); // upon return, so we have to wait until it's done

        //int port_no=our_server.get_listening_port();
        {
            string host_port = string("localhost:") + to_string(port_no);
            dlog << dlib::LINFO << "sending an expression ts to " << host_port ;
            std::vector<api::apoint_ts> tsl;
            for (size_t kb = 4;kb < 16;kb += 2)
                tsl.push_back(mk_expression(t, dt, kb * 1000)*api::apoint_ts(string("netcdf://group/path/ts") + std::to_string(kb)));
            client dtss(host_port);
            auto ts_b = dtss.evaluate(tsl, ta.total_period());
            dlog << dlib::LINFO << "Got vector back, size= " << ts_b.size();
            for (const auto& ts : ts_b)
                dlog << dlib::LINFO << "ts.size()" << ts.size();
            dlog << dlib::LINFO << "testing 2 time:";
            FAST_REQUIRE_UNARY(our_server.is_running());
            dtss.evaluate(tsl, ta.period(0));
            dlog << dlib::LINFO << "done second test";
            // test search functions
            dlog << dlib::LINFO << "test .find function";
            auto found_ts = dtss.find(string("a.*"));
            FAST_REQUIRE_EQ(found_ts.size(), 1);
            FAST_CHECK_EQ(found_ts[0].name, ts_names[0]);
            dlog << dlib::LINFO << "test .find function done";

            throw_exception = true;// verify server-side exception gets back here.
            TS_ASSERT_THROWS_ANYTHING(dtss.evaluate(tsl, ta.period(0)));
            dlog<<dlib::LINFO << "exceptions done,testing ordinary evaluate after exception";
            throw_exception = false;// verify server-side exception gets back here.
            dtss.evaluate(tsl, ta.period(0)); // verify no exception here, should still work ok
            FAST_REQUIRE_UNARY(our_server.is_running());
            dlog << dlib::LINFO << "ok, -now testing percentiles";
            std::vector<int> percentile_spec{ 0,25,50,-1,75,100 };
            auto percentiles = dtss.percentiles(tsl, ta.total_period(), ta24, percentile_spec);
            FAST_CHECK_EQ(percentiles.size(), percentile_spec.size());
            FAST_CHECK_EQ(percentiles[0].size(), ta24.size());
            dlog << dlib::LINFO << "done with percentiles, stopping localhost server";
            dtss.close();
            our_server.clear();
            dlog << dlib::LINFO << "done";
        }
    } catch (exception& e) {
        cout << e.what() << endl;
    }
    dlog << dlib::LINFO << "done";
}
TEST_CASE("dlib_server_performance") {
    dlog.set_level(dlib::LALL);
    dlib::set_all_logging_output_streams(std::cout);
    dlib::set_all_logging_levels(dlib::LALL);
    dlog << dlib::LINFO << "Starting dtss test";
    using namespace shyft::dtss;
    try {
        // Tell the server to begin accepting connections.
        calendar utc;
        auto t = utc.time(2016, 1, 1);
        auto dt = deltahours(1);
        auto dt24 = deltahours(24);
        int n = 24 * 365 * 5;// 5years of hourly data
        int n24 = n / 24;
		int n_ts = 83;
        shyft::time_axis::fixed_dt ta(t, dt, n);
        api::gta_t ta24(t, dt24, n24);
        bool throw_exception = false;
		ts_vector_t from_disk; from_disk.reserve(n_ts);
		double fv = 1.0;
		for (int i = 0; i < n_ts; ++i)
			from_disk.emplace_back(ta, fv += 1.0,shyft::time_series::ts_point_fx::POINT_AVERAGE_VALUE);

        read_call_back_t cb = [&from_disk, &throw_exception](id_vector_t ts_ids, core::utcperiod p)
            ->ts_vector_t {
            if (throw_exception) {
                dlog << dlib::LINFO << "Throw from inside dtss executes!";
                throw std::runtime_error("test exception");
            }
            return from_disk;
        };
        server our_server(cb);

        // set up the server object we have made
        our_server.set_listening_ip("127.0.0.1");
        int port_no = 20000;
        our_server.set_listening_port(port_no);
        our_server.start_async();
        size_t n_threads = 1;

        vector<future<void>> clients;
        for (size_t i = 0;i < n_threads;++i) {
            clients.emplace_back(
                    async(launch::async, [port_no,ta,ta24,i,n_ts]()         /** thread this */ {
                    string host_port = string("localhost:") + to_string(port_no);
                    dlog << dlib::LINFO << "sending an expression ts to " << host_port;
                    std::vector<api::apoint_ts> tsl;
					for (int x = 1; x <= n_ts; ++x) {// just make a  very thin request, that get loads of data back
#if 0
						auto ts_expr = api::apoint_ts(string("netcdf://group/path/ts_") + std::to_string(x));
						tsl.push_back(ts_expr);
#else
						auto ts_expr = 10.0 + 3.0*api::apoint_ts(string("netcdf://group/path/ts_") + std::to_string(x));
						if (x > 1) {
							ts_expr = ts_expr - 3.0*api::apoint_ts(string("netcdf://group/path/ts_") + std::to_string(x - 1));
						}
						tsl.push_back(ts_expr.average(ta));
#endif
					}

                    client dtss(host_port);
                    auto t0 = timing::now();
                    size_t eval_count = 0;
                    int test_duration_ms = 5000;
                    int kilo_points= tsl.size()*ta.size()/1000;
                    while (elapsed_ms(t0, timing::now()) < test_duration_ms) {
                        // burn cpu server side, save time on serialization
                        std::vector<int> percentile_spec{ -1 };
                        auto percentiles = dtss.percentiles(tsl, ta.total_period(), ta24, percentile_spec);
                        //slower due to serialization:
                        //auto ts_b = dtss.evaluate(tsl, ta.total_period());
                        ++eval_count;
                    }
                    auto total_ms = double(elapsed_ms(t0, timing::now()));
                    dlog << dlib::LINFO << "Done testing " << i << ": #= " << eval_count
                    << " e/s = " << 1000 * double(eval_count) / total_ms << " [e/s]\n"
                    << " netw throughput = "<< kilo_points*eval_count*8/(total_ms)<< " Mb/sec";
                    dtss.close();
                }

                )
            );
        }
        for (auto &f : clients) f.get();
        our_server.clear();
        dlog << dlib::LINFO << "done";

    } catch (exception& e) {
        cout << e.what() << endl;
    }
    dlog << dlib::LINFO << "done";
}
TEST_CASE("dtss_store_basics") {
        using namespace shyft::dtss;
        using namespace shyft::api;
        auto utc=make_shared<calendar>();
        auto osl=make_shared<calendar>("Europe/Oslo");

        auto t = utc->time(2016, 1, 1);
        auto dt = deltahours(1);
        int n = 24 * 365 * 2;//24*365*5;

		// construct time-axis that we want to test.
        time_axis::fixed_dt fta(t, dt, n);
        time_axis::calendar_dt cta1(utc,t,dt,n);
        time_axis::calendar_dt cta2(osl,t,dt,n);
        vector<utctime> tp;for(std::size_t i=0;i<fta.size();++i)tp.push_back(fta.time(i));
        time_axis::point_dt pta(tp,fta.total_period().end);
        auto tmpdir = (fs::temp_directory_path()/"ts.db.test");
        ts_db db(tmpdir.string());
        SUBCASE("store_fixed_dt") {
            gts_t o(fta,10.0,time_series::ts_point_fx::POINT_AVERAGE_VALUE);
            string fn("measurements/tssf.db");  // verify we can have path-parts
            db.save(fn,o);
            auto r=db.read(fn,utcperiod{});
            FAST_CHECK_EQ(o.point_interpretation(),r.point_interpretation());
            FAST_CHECK_EQ(o.time_axis(),r.time_axis());
            auto fr = db.find(string("measurements/.*\\.db")); // should match our ts.
            FAST_CHECK_EQ(fr.size(), 1 );
            db.remove(fn);
            fr = db.find(string("measurements/.*\\.db")); // should match our ts.
            FAST_CHECK_EQ(fr.size(),0);
        }

        SUBCASE("store_calendar_utc_dt") {
            gts_t o(cta1,10.0,time_series::ts_point_fx::POINT_AVERAGE_VALUE);
            string fn("tssf.db");
            db.save(fn,o);
            auto r=db.read(fn,utcperiod{});
            FAST_CHECK_EQ(o.point_interpretation(),r.point_interpretation());
            FAST_CHECK_EQ(o.time_axis(),r.time_axis());
            auto i = db.get_ts_info(fn);
            FAST_CHECK_EQ(i.name,fn);
            FAST_CHECK_EQ(i.data_period,o.total_period());
            FAST_CHECK_EQ(i.point_fx, o.point_interpretation());
            FAST_CHECK_LE( i.modified, utctime_now());
            auto fr = db.find(string(".ss.\\.db")); // should match our ts.
            FAST_CHECK_EQ(fr.size(), 1 );

            db.remove(fn);
        }
        SUBCASE("store_calendar_osl_dt") {
            gts_t o(cta2,10.0,time_series::ts_point_fx::POINT_AVERAGE_VALUE);
            string fn("tssf.db");
            db.save(fn,o);
            auto r=db.read(fn,utcperiod{});
            FAST_CHECK_EQ(o.point_interpretation(),r.point_interpretation());
            FAST_CHECK_EQ(o.time_axis(),r.time_axis());
            db.remove(fn);
        }
        SUBCASE("store_point_dt") {
            gts_t o(pta,10.0,time_series::ts_point_fx::POINT_INSTANT_VALUE);
            string fn("tssf.db");
            db.save(fn,o);
            auto r=db.read(fn,utcperiod{});
            FAST_CHECK_EQ(o.point_interpretation(),r.point_interpretation());
            FAST_CHECK_EQ(o.time_axis(),r.time_axis());
            auto i = db.get_ts_info(fn);
            FAST_CHECK_EQ(i.name,fn);
            FAST_CHECK_EQ(i.data_period,o.total_period());
            FAST_CHECK_EQ(i.point_fx, o.point_interpretation());
            FAST_CHECK_LE( i.modified, utctime_now());
            db.remove(fn);
        }

        SUBCASE("dtss_db_speed") {
			int n_ts = 1200;
			vector<gts_t> tsv; tsv.reserve(n_ts);
            double fv = 1.0;
            for (int i = 0; i < n_ts; ++i)
                tsv.emplace_back(fta, fv += 1.0,shyft::time_series::ts_point_fx::POINT_AVERAGE_VALUE);
            FAST_CHECK_EQ(n_ts,tsv.size());

            auto t0 = timing::now();
            for(std::size_t i=0;i<tsv.size();++i) {
                std::string fn("ts."+std::to_string(i)+".db");
                db.save(fn,tsv[i]);
            }
            auto t1 = timing::now();
            vector<gts_t> rv;
            for(std::size_t i=0;i<tsv.size();++i) {
                std::string fn("ts."+std::to_string(i)+".db");
                rv.push_back(db.read(fn,utcperiod{}));
            }
            auto t2= timing::now();
            auto w_mb_s= n_ts*n/double(elapsed_ms(t0,t1))/1000.0;
            auto r_mb_s= n_ts*n/double(elapsed_ms(t1,t2))/1000.0;
			// on windows(before workaround): ~ 6 mpts/sec write, 162 mpts/sec read (slow close->workaround with thread?)
			// on linux: ~ 120 mpts/sec write, 180 mpts/sec read
            std::cout<<"write pts/s = "<<w_mb_s<<", read pts/s = "<<r_mb_s<<" pts = "<<n_ts*n<<", roundtrip ms="<< double(elapsed_ms(t0,t2)) <<"\n";
			//std::cout << "open_ms:" << db.t_open << ", write_ms:" << db.t_write << ", t_close_ms:" << db.t_close << std::endl;
            FAST_CHECK_EQ(rv.size(),tsv.size());
            //fs::remove_all("*.db");
        }
#ifdef _WIN32
        this_thread::sleep_for(chrono::duration<int, std::milli>(1000));
        try {
            fs::remove_all(tmpdir);
        }
        catch (...) {
            std::cout << "Failed to remove " << tmpdir << "\n";
        }
#else
        fs::remove_all(tmpdir);
#endif

}
TEST_CASE("shyft_url") {
    using namespace shyft::dtss;
    FAST_CHECK_EQ(shyft_url("abc","123"),string("shyft://abc/123"));
    FAST_CHECK_EQ(extract_shyft_url_container("shyft://abc/something/else"),string("abc"));
    FAST_CHECK_EQ(extract_shyft_url_container("grugge"),string{});
}
TEST_CASE("dtss_store") { /*
    This test simply create and host a dtss on port 20000,
    then uses shyft:// prefix to test
    all internal operations that involve mapping to the
    shyft ts-db-store.
    */
    using namespace shyft::dtss;
    using namespace shyft::api;
    using time_series::ts_point_fx;

    auto utc=make_shared<calendar>();
    auto t = utc->time(2016, 1, 1);
    auto dt = deltahours(1);
    int n = 24 * 365 * 2;//24*365*5;

    // make dtss server
    auto tmpdir = fs::temp_directory_path()/"shyft.c.test";
    server our_server{};
    string tc{"tc"};
    our_server.add_container(tc,tmpdir.string());
    our_server.set_listening_ip("127.0.0.1");
    int port_no = 20000;
    string host_port = string("localhost:") + to_string(port_no);

    our_server.set_listening_port(port_no);
    our_server.start_async();
    // make corresponding client that we will use for the test.
    client dtss(host_port);
    SUBCASE("save_find_read") {
        size_t n_ts=100;
        time_axis::fixed_dt fta(t, dt, n);

        ts_vector_t tsv;
        for(size_t i=0;i<n_ts;++i) {
            tsv.push_back(
                apoint_ts{
                    shyft_url(tc,to_string(i)),
                    apoint_ts{fta,i*10.0,ts_point_fx::POINT_AVERAGE_VALUE}
                }
            );
        }
        auto f0 = dtss.find(shyft_url(tc,".*"));
        FAST_CHECK_EQ(f0.size(),0);// expect zero to start with
        auto t0=timing::now();
        dtss.store_ts(tsv);
        auto t1=timing::now();
        auto f1 = dtss.find(shyft_url(tc,".*"));
        FAST_CHECK_EQ(f1.size(),tsv.size());
        ts_vector_t ev;
        for(size_t i=0;i<tsv.size();++i)
            ev.push_back(3.0*apoint_ts(shyft_url(tc,to_string(i))));
        auto t2 = timing::now();
        auto er= dtss.evaluate(ev,fta.total_period());
        auto t3 = timing::now();
        FAST_CHECK_EQ(er.size(),ev.size());
        std::cout<<"store mpts/s "<<double(n_ts*n)/(double(elapsed_ms(t0,t1))/1000.0)/1e6<<"\n";
        std::cout<<"evalr mpts/s "<<double(n_ts*n)/(double(elapsed_ms(t2,t3))/1000.0)/1e6<<"\n";
    }

    our_server.clear();
#ifdef _WIN32
    this_thread::sleep_for(chrono::duration<int,std::milli>(1000));
    try {
        fs::remove_all(tmpdir);
    }
    catch (...) {
        std::cout << "Failed to remove " << tmpdir << "\n";
    }
#else
    fs::remove_all(tmpdir);
#endif


}
}
