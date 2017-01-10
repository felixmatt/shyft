#include "test_pch.h"
#include "serialization_test.h"

#include "core/utctime_utilities.h"
#include "core/time_axis.h"
#include "api/timeseries.h"

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
static bool is_equal(const timeseries::point_ts<TA>& a,const timeseries::point_ts<TA>&b) {
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


void serialization_test::test_serialization() {
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

    timeseries::point_ts<time_axis::fixed_dt> ts(ta,1.0,timeseries::fx_policy_t::POINT_AVERAGE_VALUE);
    auto ts2 = serialize_loop(ts);
    TS_ASSERT(is_equal(ts,ts2));
    time_axis::point_dt tap(vector<utctime>{0,3600},3600*2);
    auto tsp=make_shared<timeseries::point_ts<time_axis::point_dt>>(tap,2.0,timeseries::fx_policy_t::POINT_INSTANT_VALUE);
    auto tsp2 = serialize_loop(tsp);
    TS_ASSERT(is_equal(*tsp,*tsp2));

    timeseries::periodic_ts<decltype(ta)> tspp(vector<double>{1.0,10.0,2.0,3.0},deltahours(1),utc.time(2016,1,1),ta);
    auto tspp2=serialize_loop(tspp);
    TS_ASSERT(is_equal(tspp,tspp2));

#if 0
    timeseries::time_shift_ts<decltype(ts)> tsts(ts,deltahours(3600));
    auto tsts2 = serialize_loop(tsts);
    TS_ASSERT(is_equal(tsts,tsts2));

    timeseries::average_ts<decltype(ts),decltype(ta) > tsavg(ts,ta);
    auto tsavg2=serialize_loop(tsavg);
    TS_ASSERT(is_equal(tsts,tsts2));

    timeseries::accumulate_ts<decltype(ts),decltype(ta) > tsacc(ts,ta);
    auto tsacc2=serialize_loop(tsacc);
    TS_ASSERT(is_equal(tsacc,tsacc2));




    timeseries::glacier_melt_ts<decltype(ts)> tsgm(ts,ts,1000.0,6.2);
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
    arts.rep.ts=make_shared<api::gts_t>(tag,1.0,timeseries::fx_policy_t::POINT_AVERAGE_VALUE);
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



void serialization_test::test_api_ts_ref_binding() {

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

    TS_ASSERT_EQUALS(tsr.size(),2);
    try {
        f.value(0);
        TS_FAIL("Expected runtine_error here");

    } catch (const runtime_error&) {
        ;//OK!
    }
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
    TS_ASSERT_DELTA(f.value(0), a_f.value(0), 1e-9);
}

void serialization_test::test_serialization_performance() {
    bool verbose = getenv("SHYFT_VERBOSE") ? true : false;
    //
    // 1. create one large ts, do loop it.
    //
    calendar utc;
    size_t n = 10*1000*1000;// gives 80 Mb memory
    vector<double> x(n,0.0);//x.reserve(n);
    //for (size_t i = 0;i < n;++i)
    //    x.push_back(-double(n)/2.0 + i);
    api::apoint_ts aa(api::gta_t(utc.time(2016, 1, 1), deltahours(1), n), x);
    auto a = aa*3.0 + aa;
    //
    // 2. serialize it
    //
    clock_t t0 = clock();
    auto xmls = a.serialize();
    auto ms = (clock() - t0)*1000.0 / double(CLOCKS_PER_SEC);
    if(verbose)cout << "\nserialization took " << ms << "ms\n";
    TS_ASSERT_LESS_THAN(ms, 200.0); // i7 ~ 10 ms
    t0 = clock();
    auto b = api::apoint_ts::deserialize(xmls);
    ms = (clock() - t0)*1000.0 / double(CLOCKS_PER_SEC);
    TS_ASSERT_LESS_THAN(ms, 200.0);// i7 ~ 10 ms
    if(verbose) cout  << "de-serialization took " << ms << "ms\n\tsize:"<<xmls.size()<<" bytes \n\t number of doubles is "<<b.size()<<"\n";
    //TS_ASSERT(is_equal(a, b));
}

#include <dlib/server.h>
#include <dlib/iosockstream.h>

using namespace dlib;
using namespace std;


template<class T>
static api::apoint_ts read_ts(T& in) {
    int sz;
    in.read((char*)&sz,sizeof(sz));
    std::vector<char> blob(sz,0);
    in.read((char*)blob.data(),sz);
    return api::apoint_ts::deserialize_from_bytes(blob);
}

template <class T>
static void  write_ts( const api::apoint_ts& ats,T& out) {
    auto blob= ats.serialize_to_bytes();
    int sz=blob.size();
    out.write((const char*)&sz,sizeof(sz));
    out.write((const char*)blob.data(),sz);
}

template <class T>
static void write_ts_vector(const std::vector<api::apoint_ts> &ats,T & out) {
    int sz=ats.size();
    out.write((const char*)&sz,sizeof(sz));
    for(const auto & ts:ats)
        write_ts(ts,out);
}

template<class T>
static std::vector<api::apoint_ts> read_ts_vector(T& in) {
    int sz;
    in.read((char*)&sz,sizeof(sz));
    std::vector<api::apoint_ts> r;
    r.reserve(sz);
    for(int i=0;i<sz;++i)
        r.push_back(read_ts(in));
    return r;
}

class shyft_server : public server_iostream {

    void on_connect  (
        std::istream& in,
        std::ostream& out,
        const std::string& foreign_ip,
        const std::string& local_ip,
        unsigned short foreign_port,
        unsigned short local_port,
        uint64 connection_id
    ) {
        // The details of the connection are contained in the last few arguments to
        // on_connect().  For more information, see the documentation for the
        // server_iostream.  However, the main arguments of interest are the two streams.
        // Here we also print the IP address of the remote machine.
        cout << "Got a connection from " << foreign_ip << endl;

        // Loop until we hit the end of the stream.  This happens when the connection
        // terminates.
        core::calendar utc;
        time_axis::generic_dt ta(utc.time(2016,1,1),core::deltahours(1),365*24);
        api::apoint_ts dummy_ts(ta,1.0,timeseries::POINT_AVERAGE_VALUE);
        while (in.peek() != EOF) {
            auto atsv= read_ts_vector(in);
            // find stuff to bind, read and bind, then:
            for(auto& ats:atsv) {
                auto ts_refs=ats.find_ts_bind_info();
                // read all tsr here, then:
                for (auto&bind_info : ts_refs) {
                    cout<<"bind:"<<bind_info.reference<<endl;
                    bind_info.ts.bind(dummy_ts);
                }
            }
            //-- evaluate, when all binding is done (vectorized calc.
            std::vector<api::apoint_ts> evaluated_tsv;
            for(auto &ats:atsv)
                evaluated_tsv.emplace_back(ats.time_axis(),ats.values(),ats.point_interpretation());
            write_ts_vector(evaluated_tsv,out);
        }
    }

};
api::apoint_ts mk_expression(int kb=1000) {
    calendar utc;
    size_t n = 1*kb*1000;// gives 8 Mb memory
    std::vector<double> x;x.reserve(n);
    for (size_t i = 0;i < n;++i)
        x.push_back(-double(n)/2.0 + i);
    api::apoint_ts aa(api::gta_t(utc.time(2016, 1, 1), deltahours(1), n), x);
    auto a = aa*3.0 + aa;
    return a;
}

void serialization_test::test_dlib_server() {
   try
    {
        shyft_server our_server;

        // set up the server object we have made
        int port_no=1234;
        our_server.set_listening_port(port_no);
        // Tell the server to begin accepting connections.
        our_server.start_async();
        {
            cout << "sending an expression ts:\n";
            iosockstream s0(string("localhost:")+to_string(port_no));

            std::vector<api::apoint_ts> tsl;
            for(size_t kb=4;kb<16;kb+=2)
                tsl.push_back(mk_expression(kb)*api::apoint_ts(string("netcdf://group/path/ts")+ std::to_string(kb)));
            write_ts_vector(tsl,s0);
            auto ts_b=read_ts_vector(s0);
            cout<<"Got vector back, size= "<<ts_b.size()<<"\n";
            for(const auto& ts:ts_b)
                cout<<"\n\t ts.size()"<<ts.size();
            cout<<"done"<<endl;
        }

    }
    catch (exception& e)
    {
        cout << e.what() << endl;
    }
}
