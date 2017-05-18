#include "test_pch.h"

#include "core/utctime_utilities.h"
#include "core/time_axis.h"
#include "api/time_series.h"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

//-- notice that boost serialization require us to
//   include shared_ptr/vector .. etc.. wherever it's needed

#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/vector.hpp>


using namespace std;
using namespace shyft;
using namespace shyft::core;

template <class T>
static T serialize_loop(const T& o) {
    ostringstream xmls;
    boost::archive::binary_oarchive oa(xmls);
    oa << BOOST_SERIALIZATION_NVP(o);
    xmls.flush();
    string ss=xmls.str();
    istringstream xmli(ss);
    boost::archive::binary_iarchive ia(xmli);
    T o2;
    ia>>BOOST_SERIALIZATION_NVP(o2);
    return o2;
}



template<class TA>
static bool is_equal(const time_series::point_ts<TA>& a,const time_series::point_ts<TA>&b) {
    if(a.size()!=b.size())
        return false;
    if(a.time_axis().total_period()!=b.time_axis().total_period())
        return false;
    if(a.fx_policy!= b.fx_policy)
        return false;
    const double eps=1e-12;
    for(size_t i=0;i<a.size();++i)
        if(fabs(a.v[i]-b.v[i])> eps)
            return false;
    return true;
}


template<class Ts>
static bool is_equal(const Ts& a,const Ts &b) {
    if(a.size()!=b.size())
        return false;
    if(a.time_axis().total_period()!=b.time_axis().total_period())
        return false;
    if(a.point_interpretation()!= b.point_interpretation())
        return false;
    const double eps=1e-12;
    for(size_t i=0;i<a.size();++i)
        if(fabs(a.value(i)-b.value(i))> eps)
            return false;
    return true;
}

TEST_SUITE("serialization") {
TEST_CASE("test_serialization") {
    // testing serialization in the order of appearance/dependency

    //
    // 1. time & calendar
    //

    calendar utc;
    utcperiod p_1(utc.time(2016,1,1),utc.time(2017,1,1));
    auto p_2 = serialize_loop(p_1);
    TS_ASSERT_EQUALS(p_1,p_2);
    time_zone::tz_info_t tzi_1;
    tzi_1.base_tz=deltahours(1);
    tzi_1.tz.tz_name="ANY+01:00";
    tzi_1.tz.dst.emplace_back(utc.time(2016,3,1),utc.time(2016,10,1));
    tzi_1.tz.dt.push_back(deltahours(3600));

    auto tzi_2 = serialize_loop(tzi_1);
    TS_ASSERT_EQUALS(tzi_1.base_tz,tzi_2.base_tz);
    TS_ASSERT_EQUALS(tzi_1.tz.tz_name,tzi_2.tz.tz_name);
    TS_ASSERT_EQUALS(tzi_1.tz.dst,tzi_2.tz.dst);
    TS_ASSERT_EQUALS(tzi_1.tz.dt,tzi_2.tz.dt);

    auto osl=make_shared<calendar>("Europe/Oslo");
    auto osl2 = serialize_loop(osl);

    TS_ASSERT_EQUALS(osl->tz_info->base_tz,osl2->tz_info->base_tz);
    TS_ASSERT_EQUALS(osl->tz_info->tz.tz_name,osl2->tz_info->tz.tz_name);


    //
    // 2. time-axis
    //
    time_axis::fixed_dt ta(utc.time(2016,1,1),deltahours(1),24);
    auto ta2 = serialize_loop(ta);
    TS_ASSERT_EQUALS(ta.t,ta2.t);
    TS_ASSERT_EQUALS(ta.dt,ta2.dt);
    TS_ASSERT_EQUALS(ta.n,ta2.n);

    time_axis::calendar_dt tac(osl,osl->time(2016,7,1),deltahours(1),24);

    auto tac2 = serialize_loop(tac);
    TS_ASSERT_EQUALS(tac.size(),tac2.size());
    TS_ASSERT_EQUALS(tac.total_period(),tac2.total_period());

    time_axis::generic_dt tag(ta);
    auto tag2 = serialize_loop(tag);
    TS_ASSERT_EQUALS(tag.gt,tag2.gt);
    TS_ASSERT_EQUALS(tag.size(),tag2.size());
    TS_ASSERT_EQUALS(tag.total_period(),tag2.total_period());
    //
    // 3. time-series
    //

    time_series::point_ts<time_axis::fixed_dt> ts(ta,1.0,time_series::ts_point_fx::POINT_AVERAGE_VALUE);
    auto ts2 = serialize_loop(ts);
    TS_ASSERT(is_equal(ts,ts2));
    time_axis::point_dt tap(vector<utctime>{0,3600},3600*2);
    auto tsp=make_shared<time_series::point_ts<time_axis::point_dt>>(tap,2.0,time_series::ts_point_fx::POINT_INSTANT_VALUE);
    auto tsp2 = serialize_loop(tsp);
    TS_ASSERT(is_equal(*tsp,*tsp2));

    time_series::periodic_ts<decltype(ta)> tspp(vector<double>{1.0,10.0,2.0,3.0},deltahours(1),utc.time(2016,1,1),ta);
    auto tspp2=serialize_loop(tspp);
    TS_ASSERT(is_equal(tspp,tspp2));

#if 0
    time_series::time_shift_ts<decltype(ts)> tsts(ts,deltahours(3600));
    auto tsts2 = serialize_loop(tsts);
    TS_ASSERT(is_equal(tsts,tsts2));

    time_series::average_ts<decltype(ts),decltype(ta) > tsavg(ts,ta);
    auto tsavg2=serialize_loop(tsavg);
    TS_ASSERT(is_equal(tsts,tsts2));

    time_series::accumulate_ts<decltype(ts),decltype(ta) > tsacc(ts,ta);
    auto tsacc2=serialize_loop(tsacc);
    TS_ASSERT(is_equal(tsacc,tsacc2));




    time_series::glacier_melt_ts<decltype(ts)> tsgm(ts,ts,1000.0,6.2);
    auto tsgm2=serialize_loop(tsgm);
    TS_ASSERT(is_equal(tsgm,tsgm2));

    auto c = max(ts+ts*2.0,(1.0 + tsp)*tsavg);
    auto c2 = serialize_loop(c);
    TS_ASSERT(is_equal(c,c2));
#endif
    //-- api time-series

    api::gpoint_ts gts(tag,10.0);
    auto gts2=serialize_loop(gts);
    TS_ASSERT(is_equal(gts,gts2));


    auto igts=make_shared<api::gpoint_ts>(tag,2.5);
    auto igts2 = serialize_loop(igts);
    TS_ASSERT(is_equal(*igts,*igts2));
    shared_ptr<api::ipoint_ts> iigts=igts;
    api::apoint_ts iagts(iigts);
    string xiagts=iagts.serialize();
    auto iagts2 = serialize_loop(iagts);
    TS_ASSERT(is_equal(iagts,iagts2));
    api::apoint_ts agts(tag,20.0);
    auto agts2 = serialize_loop(agts);
    TS_ASSERT(is_equal(agts,agts2));

    api::average_ts gtsavg(tag,agts);
    auto gtsavg2 = serialize_loop(gtsavg);
    TS_ASSERT(is_equal(gtsavg,gtsavg2));

    api::accumulate_ts gtsacc(tag,agts);
    auto gtsacc2 = serialize_loop(gtsacc);
    TS_ASSERT(is_equal(gtsacc,gtsacc2));
    api::time_shift_ts atsts(igts,deltahours(24));
    auto atsts2= serialize_loop(atsts);
    TS_ASSERT(is_equal(atsts,atsts2));

    api::periodic_ts apts(vector<double>{1.0,10.0,5.0,2.0},deltahours(1),tag);
    auto apts2= serialize_loop(apts);
    TS_ASSERT(is_equal(apts,apts2));

    api::aref_ts arts("netcdf://file.nc");
    arts.rep.ts=make_shared<api::gts_t>(tag,1.0,time_series::ts_point_fx::POINT_AVERAGE_VALUE);
    auto arts2=serialize_loop(arts);
    TS_ASSERT_EQUALS(arts.rep.ref,arts2.rep.ref);
    TS_ASSERT(is_equal(arts,arts2));

    auto aexpr = (agts*2.0 + agts/4.0 + 12)/agts;
    auto aexpr2 = serialize_loop(aexpr);
    TS_ASSERT(is_equal(aexpr,aexpr2));


    // verify vector stuff.
    vector<api::apoint_ts> tsv;
    tsv.push_back(agts);
    tsv.push_back(3.0*agts+agts);
    tsv.push_back(10.0*agts+ 1.0/agts);
    auto tsv2 = serialize_loop(tsv);

    TS_ASSERT_EQUALS(tsv.size(), tsv2.size());
    for (size_t i = 0;i < tsv.size();++i)
        TS_ASSERT(is_equal(tsv[i], tsv2[i]));


}



TEST_CASE("test_api_ts_ref_binding") {

    calendar utc;
    time_axis::generic_dt ta(utc.time(2016,1,1),deltahours(1),24);
    api::apoint_ts a(ta,1.0);
    api::apoint_ts b(ta,2.0);
    string s_c="fm::/nordic_main/xyz";
    api::apoint_ts c(s_c);
    string s_d="netcdf://arome_2016_01_01T00:00/UTM32/E12.123/N64.222";
    api::apoint_ts d(s_d);
    auto f = 3.0*a*(b+(c*d)*4);
    auto tsr=f.find_ts_bind_info();

    auto xmls_unbound = f.serialize();

    TS_ASSERT_EQUALS(tsr.size(),2u);
    CHECK_THROWS_AS(f.value(0), runtime_error);
    // -now bind the variables
    api::apoint_ts b_c(ta,5.0);
    api::apoint_ts b_d(ta,3.0);

    for (auto&bind_info : tsr) {
        if (bind_info.reference == s_c)
            bind_info.ts.bind(b_c);
        else if (bind_info.reference == s_d)
            bind_info.ts.bind(b_d);
        else
            TS_FAIL("ref not found");
    }
    f.do_bind();
    // then retry evaluate
    try {
        double v0=f.value(0);
        TS_ASSERT_DELTA(v0,3.0*1.0*(2.0+(5.0*3.0)*4),1e-9);
    } catch (const runtime_error&) {
        TS_FAIL("Sorry, still not bound values");
    }
    auto a_f = api::apoint_ts::deserialize(xmls_unbound);
    auto unbound_ts = a_f.find_ts_bind_info();
    for (auto&bind_info : unbound_ts) {
        if (bind_info.reference == s_c)
            bind_info.ts.bind(b_c);
        else if (bind_info.reference == s_d)
            bind_info.ts.bind(b_d);
        else
            TS_FAIL("ref not found");
    }
    a_f.do_bind();
    TS_ASSERT_DELTA(f.value(0), a_f.value(0), 1e-9);
}

TEST_CASE("test_serialization_performance") {
    //ostringstream os;
    //os.seekp(1000);
    //os.put('\a');
    //string oss = os.str();
    //FAST_CHECK_EQ(oss.size(), 1001);
    bool verbose = getenv("SHYFT_VERBOSE") ? true : false;
    //
    // 1. create one large ts, do loop it.
    //
    for (size_t n_threads = 1;n_threads < (verbose?6:1);++n_threads) {
        calendar utc;
        size_t n = 10 * 1000 * 1000;// gives 80 Mb memory
        vector<double> x(n, 0.0);
        vector<api::apoint_ts> av;
        for (size_t i = 0;i < n_threads;++i)
            av.emplace_back(api::gta_t(utc.time(2016, 1, 1), deltahours(1), n), x);

        //auto a = aa*3.0 + aa;
        //
        // 2. serialize it
        //
        clock_t t0 = clock();
        // -multi-thread this to n threads:
        vector<future<void>> calcs1;
        for (size_t i = 0;i < n_threads;++i) {
            calcs1.emplace_back(
                async(launch::async, [&av, i]() {
                auto xmls = av[i].serialize();
            }
                )
            );
        }
        for (auto &f : calcs1) f.get();

        auto ms = (clock() - t0)*1000.0 / double(CLOCKS_PER_SEC);

        if (verbose)cout << "\nserialization took " << ms << "ms\n";
        TS_ASSERT_LESS_THAN(ms, 1200.0); // i7 ~ 10 ms
        auto xmls = av[0].serialize();
        vector<string> xmlsv;
        for (size_t i = 0;i < n_threads;++i)
            xmlsv.push_back(xmls);

        auto b = api::apoint_ts::deserialize(xmls);
        t0 = clock();
        vector<future<void>> calcs2;
        for (size_t i = 0;i < n_threads;++i) {
            calcs2.emplace_back(
                async(launch::async, [&xmlsv, i]() {
                auto b = api::apoint_ts::deserialize(xmlsv[i]);
            }
                )
            );
        }
        for (auto &f : calcs2) f.get();
        ms = (clock() - t0)*1000.0 / double(CLOCKS_PER_SEC);
        TS_ASSERT_LESS_THAN(ms, 1200.0);// i7 ~ 10 ms
        if (verbose) cout << "de-serialization took " << ms
            << "ms\n\tsize:" << n_threads*xmls.size()
            << " bytes \n\t number of doubles is " << n_threads *double(b.size()) / 1e6 << "mill 8byte size\n"
            << "performance:" << n_threads *double(b.size())*8 / 1e6 / (ms / 1000.0) << " [MB/s]\n";
    }
}

TEST_CASE("test_serialization_memcpy_performance") {

    bool verbose = getenv("SHYFT_VERBOSE") ? true : false;
    //
    // 1. create one large ts, do loop it.
    //
    for (size_t n_threads = 1;n_threads < (verbose ? 6 : 1);++n_threads) {
        calendar utc;
        size_t n = 10 * 1000 * 1000;// gives 80 Mb memory
        vector<double> x(n, 0.0);
        vector<vector<double>> av;
        for (size_t i = 0;i < n_threads;++i)
            av.emplace_back(x);


        clock_t t0 = clock();
        // -multi-thread this to n threads:
        mutex c_mx;
        condition_variable cv;
        vector<future<void>> calcs1;
        size_t c = 0;
        for (size_t i = 0;i < n_threads;++i) {
            calcs1.emplace_back(
                async(launch::async, [&av, i,n,&c,&c_mx,&cv]() {
                    double *y= new double[n];
                    memcpy(y, av[i].data(), n * sizeof(double));
                    //copy(av[i].begin(),av[i].end(),back_inserter(y));
                    {
                        unique_lock<mutex> sl(c_mx);
                        c++;
                        cv.notify_all();
                    }

                    delete y;
                }
                )
            );
        }
        {
            unique_lock<mutex> m_lck(c_mx);
            while (c != n_threads)cv.wait(m_lck);
        }
        auto ms = (clock() - t0)*1000.0 / double(CLOCKS_PER_SEC);
        size_t mcpy_size = 8 * n_threads*n;
        if (verbose) cout << "memcpy-serialization took " << ms
            << "ms\n\tsize:" << mcpy_size
            << " bytes \n\t number of doubles is " << mcpy_size / 1e6/8 << "mill 8byte size\n"
            << "performance:" << mcpy_size / 1e6 / (ms / 1000.0) << " [MB/s]\n";
        for (auto &f : calcs1) f.get();

    }
}
}
