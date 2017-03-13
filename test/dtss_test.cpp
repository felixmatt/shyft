#include "test_pch.h"

#include <dlib/server.h>
#include <dlib/iosockstream.h>
#include "core/dtss.h"

#include "core/utctime_utilities.h"
#include "core/time_axis.h"
#include "api/timeseries.h"

using namespace std;
using namespace shyft;
using namespace shyft::core;

api::apoint_ts mk_expression(utctime t, utctimespan dt, int n) {

    std::vector<double> x;x.reserve(n);
    for (int i = 0;i < n;++i)
        x.push_back(-double(n) / 2.0 + i);
    api::apoint_ts aa(api::gta_t(t, dt, n), x);
    auto a = aa*3.0 + aa;
    return a;
}
#include <dlib/logger.h>
#include <dlib/misc_api.h>
dlib::logger dlog("dlib.log");

TEST_SUITE("serialization");

TEST_CASE("test_dlib_server") {
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
        call_back_t cb = [ta, &throw_exception](id_vector_t ts_ids, core::utcperiod p)
            ->ts_vector_t {
            ts_vector_t r;r.reserve(ts_ids.size());
            double fv = 1.0;
            for (size_t i = 0;i < ts_ids.size();++i)
                r.emplace_back(ta, fv += 1.0);
            if (throw_exception) {
                dlog << dlib::LINFO << "Throw from inside dtss executes!";
                throw std::runtime_error("test exception");
            }
            return r;
        };
        server our_server(cb);

        // set up the server object we have made
        our_server.set_listening_ip("127.0.0.1");
        int port_no = 20000;
        our_server.set_listening_port(port_no);
        our_server.start_async();
        //while(our_server.is_running()&& our_server.get_listening_port()==0)
        //    std::this_thread::sleep_for(std::chrono::milliseconds(3));

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