#include "test_pch.h"

#include "core/dtss.h"
#include "core/dtss_cache.h"

#include "core/utctime_utilities.h"
#include "core/time_axis.h"
#include "core/time_series_merge.h"
#include "api/time_series.h"

#ifdef SHYFT_NO_PCH
#include <future>
#include <mutex>
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
#include <armadillo>
#endif // SHYFT_PCH

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

dlib::logger dlog("dlib.log");

#define TEST_SECTION(x)

TEST_SUITE("dtss") {

TEST_CASE("dtss_lru_cache") {
	using shyft::dtss::lru_cache;
	using std::map;
	using std::list;
	using std::vector;
	using std::string;
	using std::back_inserter;
	using shyft::api::apoint_ts;
	using shyft::api::gta_t;
	const auto stair_case=shyft::time_series::POINT_AVERAGE_VALUE;
	lru_cache<string, apoint_ts, map > c(2);

	apoint_ts r;
	vector<string> mru;
	gta_t ta(0, 1, 10);

	TEST_SECTION("empty_cache") {
		FAST_CHECK_UNARY_FALSE(c.try_get_item("a", r));
	}
	TEST_SECTION("add_one_item") {
		c.add_item("a", apoint_ts(ta, 1.0, stair_case));
		FAST_CHECK_UNARY(c.try_get_item("a", r));
		FAST_CHECK_EQ(ta.size(), r.time_axis().size());
		FAST_CHECK_UNARY_FALSE(c.try_get_item("b", r));
	}
	TEST_SECTION("add_second_item") {
		c.add_item("b", apoint_ts(ta, 2.0, stair_case));
		FAST_CHECK_UNARY(c.try_get_item("a", r));
		FAST_CHECK_UNARY(c.try_get_item("b", r));
		c.get_mru_keys(back_inserter(mru));
		FAST_CHECK_EQ(string("b"), mru[0]);
		FAST_CHECK_EQ(string("a"), mru[1]);
	}
	TEST_SECTION("mru_item_in_front") {
		c.try_get_item("a", r);
		mru.clear(); c.get_mru_keys(back_inserter(mru));
		FAST_CHECK_EQ(string("a"), mru[0]);
		FAST_CHECK_EQ(string("b"), mru[1]);
	}
	TEST_SECTION("excessive_lru_item_evicted_when_adding") {
		c.add_item("c", apoint_ts(ta, 3.0, stair_case));
		FAST_CHECK_UNARY_FALSE(c.try_get_item("b", r));
		FAST_CHECK_UNARY(c.try_get_item("c", r));
		FAST_CHECK_UNARY(c.try_get_item("a", r));
	}
	TEST_SECTION("remove_item") {
		c.remove_item("a");
		FAST_CHECK_UNARY_FALSE(c.try_get_item("a", r));
	}
	TEST_SECTION("ensure_items_added_are_first") {
		c.add_item("d", apoint_ts(ta, 4.0, stair_case));
		mru.clear(); c.get_mru_keys(back_inserter(mru));
		FAST_CHECK_EQ(string("d"), mru[0]);
		FAST_CHECK_EQ(string("c"), mru[1]);
	}
	TEST_SECTION("update_existing") {
		c.try_get_item("c", r);//just to ensure "c" is in first position
		c.add_item("d", apoint_ts(ta, 4.2, stair_case)); //update "d"
		c.try_get_item("d", r);
		FAST_CHECK_GT(r.value(0), 4.1);
		mru.clear(); c.get_mru_keys(back_inserter(mru));
		FAST_CHECK_EQ(string("d"), mru[0]);
		FAST_CHECK_EQ(string("c"), mru[1]);
	}
}
TEST_CASE("dtss_ts_cache") {
    using std::vector;
    using std::string;
    using shyft::core::utctime;
    using shyft::core::deltahours;
    using shyft::dtss::cache_stats;
    using shyft::api::apoint_ts;
    using shyft::api::gta_t;
    using shyft::dtss::apoint_ts_frag;
    using dtss_cache=shyft::dtss::cache<apoint_ts_frag,apoint_ts>;
    const auto stair_case=shyft::time_series::POINT_AVERAGE_VALUE;
    size_t max_ids=10;

    dtss_cache c(max_ids);

    apoint_ts x;
    utcperiod p{0,10};
    FAST_CHECK_EQ(false,c.try_get("a",p,x));

    utctime t0{5};
    utctime t1{10};

    utctimespan dt{1};
    size_t n{3};
    apoint_ts ts_a{gta_t{t0,dt,n},1.0,stair_case};
    apoint_ts ts_a2{gta_t{t1,dt,n},1.0,stair_case};

    c.add("a",ts_a);
    FAST_REQUIRE_EQ(true,c.try_get("a",utcperiod{t0,t0+dt},x));
    FAST_CHECK_EQ(1.0,x.value(0));

    c.add("a",ts_a2);// test we can add twice, a fragment (replace)

    FAST_REQUIRE_EQ(false,c.try_get("a",utcperiod{t0,t0+ 10*dt},x));

    auto s=c.get_cache_stats();
    FAST_CHECK_EQ(s.hits,2);
    FAST_CHECK_EQ(s.misses,1);
    FAST_CHECK_EQ(s.coverage_misses,1);
    FAST_CHECK_EQ(s.point_count,6);
    FAST_CHECK_EQ(s.fragment_count,2);
    FAST_CHECK_EQ(s.id_count,1);

    c.clear_cache_stats();
    c.flush();
    s = c.get_cache_stats();
    FAST_CHECK_EQ(s.hits, 0);
    FAST_CHECK_EQ(s.misses, 0);
    FAST_CHECK_EQ(s.coverage_misses, 0);
    FAST_CHECK_EQ(s.point_count, 0);
    FAST_CHECK_EQ(s.fragment_count, 0);
    FAST_CHECK_EQ(s.id_count, 0);

    c.add("a", ts_a2);

    c.remove("a");
    FAST_REQUIRE_EQ(false,c.try_get("a",utcperiod{t0,t0+dt},x));

    //-- test vector operations
    // arrange n_ts
    vector<string> ids;
    vector<apoint_ts> tss;
    size_t n_ts = 3;
    gta_t mta{ t0,dt,n };
    for (size_t i = 0; i<n_ts; ++i) {
        ids.push_back(to_string(i));
        tss.emplace_back(mta, double(i), stair_case);
    }

    c.add(ids, tss); // add a vector of ids|tss

    auto mts = c.get(ids, mta.total_period());// get a vector of  ids back as map[id]=ts
    FAST_REQUIRE_EQ(n_ts, mts.size());
    for (size_t i = 0; i<n_ts; ++i) {
        FAST_REQUIRE_UNARY(mts.find(ids[i])!=mts.end());
        FAST_CHECK_EQ(mts[ids[i]].value(0), double(i)); // just check one value unique for ts.
    }

    auto ids2 = ids; ids2.push_back("not there");// ask for something that's not there
    auto mts2 = c.get(ids2, mta.total_period());
    FAST_REQUIRE_EQ(n_ts, mts.size());
    for (size_t i = 0; i<n_ts; ++i) {
        FAST_REQUIRE_UNARY(mts2.find(ids[i])!=mts.end());
        FAST_CHECK_EQ(mts2[ids[i]].value(0), double(i)); // just check one value unique for ts.
    }

    c.remove(ids2); // remove by vector (even with elem not there)
    s = c.get_cache_stats();
    FAST_CHECK_EQ(s.point_count, 0);
    FAST_CHECK_EQ(s.fragment_count, 0);
    FAST_CHECK_EQ(s.id_count, 0);

}
TEST_CASE("dtss_mini_frag") {
    using std::vector;
    using std::string;
    using std::min;
    using std::max;
    using shyft::core::utcperiod;
    using shyft::core::utctime;
    using shyft::dtss::mini_frag;
    using shyft::time_axis::continuous_merge;

    struct tst_frag {
        utcperiod p;
        int id{0};
        tst_frag(utctime f,utctime u):p(f,u){}
        tst_frag(utctime f,utctime u,int id):p(f,u),id(id){}
        utcperiod total_period() const {return p;}
        size_t size() const {return size_t(p.timespan());}
        tst_frag merge(const tst_frag&o) const {
            if(!continuous_merge(p,o.p))
                throw runtime_error("Wrong merge op attempted");
            return tst_frag{min(o.p.start,p.start),max(o.p.end,p.end)};
        }
    };

    mini_frag<tst_frag> m;

    FAST_CHECK_EQ(m.count_fragments(),0);
    FAST_CHECK_EQ(m.estimate_size(),0);
    FAST_CHECK_EQ(m.get_ix(utcperiod(0,10)),string::npos);

    m.add(tst_frag(5,10));// add first
    FAST_CHECK_EQ(m.count_fragments(),1);
    FAST_CHECK_EQ(m.estimate_size(),5);
    FAST_CHECK_EQ(m.get_ix(utcperiod{5,10}),0);

    m.add(tst_frag(4,5));// add merge element
    FAST_CHECK_EQ(m.count_fragments(),1);
    FAST_CHECK_EQ(m.estimate_size(),6);
    FAST_CHECK_EQ(m.get_ix(utcperiod{4,6}),0);

    m.add(tst_frag(1,3));// add before first
    FAST_CHECK_EQ(m.count_fragments(),2);
    FAST_CHECK_EQ(m.estimate_size(),8);
    FAST_CHECK_EQ(m.get_ix(utcperiod{4,6}),1);
    FAST_CHECK_EQ(m.get_ix(utcperiod{1,2}),0);

    m.add(tst_frag(3,4));// add a piece that merge [0] [1]
    FAST_CHECK_EQ(m.count_fragments(),1);
    FAST_CHECK_EQ(m.estimate_size(),9);
    FAST_CHECK_EQ(m.get_ix(utcperiod{4,6}),0);

    m.add(tst_frag{11,12}); // append a frag at the end
    FAST_CHECK_EQ(m.count_fragments(),2);
    FAST_CHECK_EQ(m.estimate_size(),10);
    FAST_CHECK_EQ(m.get_ix(utcperiod{4,6}),0);
    FAST_CHECK_EQ(m.get_ix(utcperiod{11,12}),1);
    FAST_CHECK_EQ(m.get_ix(utcperiod{11,13}),string::npos);
    FAST_CHECK_EQ(m.get_ix(utcperiod{1,12}),string::npos);

    m.add(tst_frag{13,15}); // append a frag at the end
    FAST_CHECK_EQ(m.count_fragments(),3);
    FAST_CHECK_EQ(m.estimate_size(),12);
    FAST_CHECK_EQ(m.get_ix(utcperiod{13,14}),2);
    FAST_CHECK_EQ(m.get_ix(utcperiod{11,12}),1);
    FAST_CHECK_EQ(m.get_ix(utcperiod{5,6}),0);

    m.add(tst_frag{11,14});// append a frag that melts [1]..[2] into one
    FAST_CHECK_EQ(m.count_fragments(),2);
    FAST_CHECK_EQ(m.get_ix(utcperiod{11,15}),1);

    m.add(tst_frag{0,20}); // append a frag that melts all into one
    FAST_CHECK_EQ(m.count_fragments(),1);
    FAST_CHECK_EQ(m.estimate_size(),20);

    m.add(tst_frag{0,20,1}); // append a marked frag that is exactly equal
    FAST_CHECK_EQ(m.count_fragments(),1);
    FAST_CHECK_EQ(m.estimate_size(),20);
    FAST_CHECK_EQ(m.get_ix(utcperiod{0,10}),0);
    FAST_CHECK_EQ(m.get_by_ix(0).id,1);// verify we got the marked in place

    m.add(tst_frag{21,23});// two frags
    m.add(tst_frag{25,27});// three
    m.add(tst_frag{0,23});// exactly cover second frag
    FAST_CHECK_EQ(m.count_fragments(),2);

    m.add(tst_frag{1,24});
    FAST_CHECK_EQ(m.count_fragments(),2);

    m.add(tst_frag{2,27});// parts p1, p2.end
    FAST_CHECK_EQ(m.count_fragments(),1);

    m.add(tst_frag{-1,27});
    FAST_CHECK_EQ(m.count_fragments(),1);

}

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
        for (int i = 0; i<10; ++i) {
            this_thread::sleep_for(chrono::duration<int, std::milli>(1000));
            try {
                fs::remove_all(tmpdir);
                break;
            }
            catch (...) {
                std::cout <<"Try #"<<i+1<< ":Failed to remove " << tmpdir << "\n";
            }
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
        time_axis::generic_dt gta{t,dt*24,size_t(n/24)};
        const auto stair_case=ts_point_fx::POINT_AVERAGE_VALUE;
        ts_vector_t tsv;
        vector<point_ts<time_axis::fixed_dt>> ftsv;

        for(size_t i=0;i<n_ts;++i) {
            tsv.emplace_back(
                    shyft_url(tc,to_string(i)),
                    apoint_ts{fta,i*10.0,stair_case}
            );
            ftsv.emplace_back(fta,i*10.0,stair_case);
        }
        auto f0 = dtss.find(shyft_url(tc,".*"));
        FAST_CHECK_EQ(f0.size(),0);// expect zero to start with
        auto t0=timing::now();
        dtss.store_ts(tsv,false);
        auto t1=timing::now();
        auto f1 = dtss.find(shyft_url(tc,".*"));
        FAST_CHECK_EQ(f1.size(),tsv.size());
        ts_vector_t ev;
        for(size_t i=0;i<tsv.size();++i)
            ev.push_back(
                         3.0*apoint_ts(shyft_url(tc,to_string(i)))
                           //+ apoint_ts(shyft_url(tc,to_string(i>0?i-1:i)))
                         );
        // activate auto-cache, to prepare for next
        our_server.set_auto_cache(true);
        vector<int> pc{10,50,90};
        auto t2 = timing::now();
        //auto er= dtss.percentiles(ev,fta.total_period(),gta,pc);//uncached read
        auto er= dtss.evaluate(ev,fta.total_period());//uncached read
        auto t3 = timing::now();
        //auto ec = dtss.percentiles(ev, fta.total_period(),gta,pc);
        auto ec = dtss.evaluate(ev, fta.total_period());
        auto t4 = timing::now();// cached read.
        //-- establish benchmark
        vector<vector<double>> bmr;bmr.reserve(n_ts);
        for(const auto &ts:ftsv) {
            auto calc= 3.0* ts;
            vector<double> r;r.reserve(calc.size());
            for(size_t i=0;i<calc.size();++i)
                r.emplace_back(calc.value(i));
            bmr.emplace_back(move(r));
        }
        auto t5 =timing::now();
        FAST_CHECK_EQ(bmr.size(),n_ts);
//        FAST_CHECK_EQ(er.size(),pc.size());
//        FAST_CHECK_EQ(ec.size(), pc.size());
        FAST_CHECK_EQ(er.size(),ev.size());
        FAST_CHECK_EQ(ec.size(), ev.size());
        std::cout<<"store mpts/s "<<double(n_ts*n)/(double(elapsed_ms(t0,t1))/1000.0)/1e6<<"\n";
        std::cout<<"evalr mpts/s "<<double(n_ts*n)/(double(elapsed_ms(t2,t3))/1000.0)/1e6<<"\n";
        std::cout<<"evalc mpts/s "<<double(n_ts*n)/(double(elapsed_ms(t3,t4))/1000.0)/1e6<<"\n";
        std::cout<<"bench mpts/s "<<double(n_ts*n)/(double(elapsed_ms(t4,t5))/1000.0)/1e6<<"\t time :"<<double(elapsed_ms(t4,t5))<<"\n";
        auto cs = our_server.get_cache_stats();
        std::cout<<"cache stats(hits,misses,cover_misses,id_count,frag_count,point_count):\n "<<cs.hits<<","<<cs.misses<<","<<cs.coverage_misses<<","<<cs.id_count<<","<<cs.fragment_count<<","<<cs.point_count<<")\n";
    }

    our_server.clear();
#ifdef _WIN32
    for (int i = 0; i<10; ++i) {
        this_thread::sleep_for(chrono::duration<int, std::milli>(1000));
        try {
            fs::remove_all(tmpdir);
            break;
        }
        catch (...) {
            std::cout<<"Try #"<<i+1 << ": Failed to remove " << tmpdir << "\n";
        }
    }
#else
    fs::remove_all(tmpdir);
#endif


}
TEST_CASE("dtss_baseline") {
    using namespace shyft::dtss;
    using namespace shyft::api;
    using time_series::ts_point_fx;
    using std::cout;
    auto utc=make_shared<calendar>();
    auto t = utc->time(2016, 1, 1);
    auto dt = deltahours(1);
    const int n = 24 * 365 * 5/3;//24*365*5;

    vector<point_ts<time_axis::fixed_dt>> ftsv;
    const size_t n_ts=100*83;
    arma::mat a_mat(n,n_ts);
    time_axis::fixed_dt fta(t, dt, n);
    //time_axis::generic_dt gta{t,dt*24,size_t(n/24)};
    const auto stair_case=ts_point_fx::POINT_AVERAGE_VALUE;
    ts_vector_t tsv;
    for(size_t i=0;i<n_ts;++i) {
        tsv.emplace_back(to_string(i),apoint_ts(fta,i*10.0,stair_case));
        ftsv.emplace_back(fta,i*10.0,stair_case);
        for(size_t t =0;t<n;++t)
            a_mat(t,i) = i*10.0;
    }
    tsv = 3.0*tsv;


    //-- establish benchmark core-ts
    auto t0 = timing::now();

    vector<vector<double>> bmr;bmr.reserve(n_ts);
    for(const auto &ts:ftsv) {
        auto calc= 3.0* ts;
        vector<double> r;r.reserve(calc.size());
        for(size_t i=0;i<calc.size();++i)
            r.emplace_back(calc.value(i));
        bmr.emplace_back(move(r));
    }
    auto t1 =timing::now();

    //-- establish benchmark armadillo
    vector<vector<double>> amr;amr.reserve(n_ts);
    auto a_res= (a_mat*3.0).eval();
    for(size_t i=0;i<n_ts;++i) {
        amr.emplace_back(arma::conv_to<vector<double>>::from(a_res.col(i)) );
    }
    auto t2 = timing::now();

    //-- establish timing for apoint_ts eval.
    vector<vector<double>> xmr;xmr.reserve(n_ts);
    //auto xtsv = deflate_ts_vector<point_ts<time_axis::generic_dt>>(tsv);
    for(const auto &ts:tsv) {
        xmr.emplace_back(move(ts.values()));
    }
    auto t3 = timing::now();

    FAST_CHECK_EQ(bmr.size(),n_ts);
    FAST_CHECK_EQ(amr.size(),n_ts);
    FAST_CHECK_EQ(xmr.size(),n_ts);

    cout<<"core-ts base-line n_ts= "<<n_ts<<", n="<<n<<", time="<<double(elapsed_us(t0,t1))/1000.0<<"ms ->"
    << double(n*n_ts)/(elapsed_us(t0,t1)/1e6)/1e6<<" mops/s \n";

    cout<<"api -ts base-line n_ts= "<<n_ts<<", n="<<n<<", time="<<double(elapsed_us(t2,t3))/1000.0<<"ms ->"
    << double(n*n_ts)/(elapsed_us(t2,t3)/1e6)/1e6<<" mops/s \n";

    cout<<"armavec base-line n_ts= "<<n_ts<<", n="<<n<<", time="<<double(elapsed_us(t1,t2))/1000.0<<"ms ->"
    << double(n*n_ts)/(elapsed_us(t1,t2)/1e6)/1e6<<" mops/s \n";

}

TEST_CASE("dtss_ltm") {
    // this is basically just for performance study of
    // for api type of ts-expressions,
    using namespace shyft::dtss;
    using namespace shyft::api;
    using time_series::ts_point_fx;
    using std::cout;
    auto utc=make_shared<calendar>();
    auto t = utc->time(2016, 1, 1);
    auto dt = deltahours(1);
    const int n = 24 * 365 * 5/3;//24*365*5;

    const size_t n_scn=83;
    const size_t n_obj =10;
    const size_t n_ts=n_obj*2*n_scn;

    vector<point_ts<time_axis::fixed_dt>> ftsv;
    arma::mat a_mat(n,n_ts);
    time_axis::fixed_dt fta(t, dt, n);
    time_axis::generic_dt gta{t,dt*24,size_t(n/24)};
    const auto stair_case=ts_point_fx::POINT_AVERAGE_VALUE;
    map<string,apoint_ts> rtsv;
    ts_vector_t stsv;
    for(size_t i=0;i<n_ts;++i) {
        rtsv[to_string(i)] = apoint_ts(fta,i*10.0,stair_case);
        stsv.emplace_back(to_string(i));
        ftsv.emplace_back(fta,i*10.0,stair_case);
        for(size_t t =0;t<n;++t)
            a_mat(t,i) = i*10.0;
    }
    ts_vector_t tsv;
    for(size_t i =0; i<n_scn;++i) {
        apoint_ts sum;
        for(size_t j=0;j<n_obj;++j) {
            size_t p = i*(2*n_obj) + (2*j);
            size_t c = p+1;
            apoint_ts eff=3.0*(stsv[p]-stsv[c]);
            if(j==0)
                sum = eff;
            else
                sum = sum + eff;
        }
        tsv.emplace_back(sum);
    }
    tsv = 1000.0*(tsv.average(gta));


    //-- establish compute binding time
    auto t0 = timing::now();
    size_t bind_count{0};
    for(auto&sts:tsv) {
        auto ts_refs=sts.find_ts_bind_info();
        for(auto& bi:ts_refs) {
            bi.ts.bind(rtsv[bi.reference]);
            bind_count++;
        }
    }
    for(auto&sts:tsv)
        sts.do_bind();

    auto t1 =timing::now();
    //-- establish timing for apoint_ts eval.
    auto xmr = deflate_ts_vector<point_ts<time_axis::generic_dt>>(tsv);
    auto t2 = timing::now();

    FAST_CHECK_EQ(xmr.size(),n_scn);
    FAST_CHECK_EQ(bind_count,n_ts);
    cout<<"bind phase n_ts= "<<n_ts<<", n="<<n<<", time="<<double(elapsed_us(t0,t1))/1000.0<<"ms\n";
    cout<<"eval phase n_ts= "<<n_ts<<", n="<<n<<", time="<<double(elapsed_us(t1,t2))/1000.0<<"ms ->"
    << double(n*n_ts*(2+1))/(elapsed_us(t1,t2)/1e6)/1e6<<" mops/s \n";

}
}
