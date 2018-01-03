#include "test_pch.h"
#include "core/expression_serialization.h"
#include "core/core_archive.h"

using namespace std;
using namespace shyft;
using namespace shyft::core;

template <class T>
static T serialize_loop(const T& o, int c_a_flags = core_arch_flags) {
    stringstream xmls;
    core_oarchive oa(xmls, c_a_flags);
    oa << core_nvp("o", o);
    xmls.flush();
    core_iarchive ia(xmls, c_a_flags);
    T o2;
    ia>>core_nvp("o", o2);
    return o2;
}



template<class TA>
static bool is_equal(const time_series::point_ts<TA>& a, const time_series::point_ts<TA>&b, double eps = 1e-12) {
    if (a.size()!=b.size())
        return false;
    if (a.time_axis().total_period()!=b.time_axis().total_period())
        return false;
    if (a.fx_policy!= b.fx_policy)
        return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a.v[i]) && !std::isfinite(b.v[i]))
            continue; //both nans, ok.
        if (fabs(a.v[i] - b.v[i]) > eps) {
            cout << " values differs as index=" << i << " a= " << a.v[i] << ", b=" << b.v[i] << endl;
            return false;
        }
    }
    return true;
}


template<class Ts>
static bool is_equal(const Ts& a, const Ts &b) {
    if (a.size()!=b.size())
        return false;
    if (a.time_axis().total_period()!=b.time_axis().total_period())
        return false;
    if (a.point_interpretation()!= b.point_interpretation())
        return false;
    const double eps = 1e-12;
    for (size_t i = 0; i<a.size(); ++i)
        if (fabs(a.value(i)-b.value(i))> eps)
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
	using namespace time_series;
	using dd::gpoint_ts;
	using dd::apoint_ts;
	using dd::ipoint_ts;
    gpoint_ts gts(tag,10.0);
    auto gts2=serialize_loop(gts);
    TS_ASSERT(is_equal(gts,gts2));


    auto igts=make_shared<gpoint_ts>(tag,2.5);
    auto igts2 = serialize_loop(igts);
    TS_ASSERT(is_equal(*igts,*igts2));
    shared_ptr<ipoint_ts> iigts=igts;
    apoint_ts iagts(iigts);
    string xiagts=iagts.serialize();
    auto iagts2 = serialize_loop(iagts);
    TS_ASSERT(is_equal(iagts,iagts2));
    apoint_ts agts(tag,20.0);
    auto agts2 = serialize_loop(agts);
    TS_ASSERT(is_equal(agts,agts2));

    dd::average_ts gtsavg(tag,agts);
    auto gtsavg2 = serialize_loop(gtsavg);
    TS_ASSERT(is_equal(gtsavg,gtsavg2));

    dd::accumulate_ts gtsacc(tag,agts);
    auto gtsacc2 = serialize_loop(gtsacc);
    TS_ASSERT(is_equal(gtsacc,gtsacc2));
    dd::time_shift_ts atsts(igts,deltahours(24));
    auto atsts2= serialize_loop(atsts);
    TS_ASSERT(is_equal(atsts,atsts2));

    dd::periodic_ts apts(vector<double>{1.0,10.0,5.0,2.0},deltahours(1),tag);
    auto apts2= serialize_loop(apts);
    TS_ASSERT(is_equal(apts,apts2));

    dd::aref_ts arts("netcdf://file.nc");
    arts.rep=make_shared<gpoint_ts>(tag,1.0,time_series::ts_point_fx::POINT_AVERAGE_VALUE);
    auto arts2=serialize_loop(arts);
    TS_ASSERT_EQUALS(arts.id,arts2.id);
    TS_ASSERT(is_equal(arts,arts2));

    auto aexpr = (agts*2.0 + agts/4.0 + 12)/agts;
    auto aexpr2 = serialize_loop(aexpr);
    TS_ASSERT(is_equal(aexpr,aexpr2));


    // verify vector stuff.
    vector<apoint_ts> tsv;
    tsv.push_back(agts);
    tsv.push_back(3.0*agts+agts);
    tsv.push_back(10.0*agts+ 1.0/agts);
    auto tsv2 = serialize_loop(tsv);

    TS_ASSERT_EQUALS(tsv.size(), tsv2.size());
    for (size_t i = 0;i < tsv.size();++i)
        TS_ASSERT(is_equal(tsv[i], tsv2[i]));


}



TEST_CASE("test_api_ts_ref_binding") {
	using namespace time_series;
	using dd::gpoint_ts;
	using dd::apoint_ts;
	using dd::ipoint_ts;

    calendar utc;
    time_axis::generic_dt ta(utc.time(2016,1,1),deltahours(1),24);
    apoint_ts a(ta,1.0);
    apoint_ts b(ta,2.0);
    string s_c="fm::/nordic_main/xyz";
    apoint_ts c(s_c);
    string s_d="netcdf://arome_2016_01_01T00:00/UTM32/E12.123/N64.222";
    apoint_ts d(s_d);
    auto f = 3.0*a*(b+(c*d)*4);
    auto tsr=f.find_ts_bind_info();

    auto xmls_unbound = f.serialize();

    TS_ASSERT_EQUALS(tsr.size(),2u);
    CHECK_THROWS_AS(f.value(0), runtime_error);
    // -now bind the variables
    apoint_ts b_c(ta,5.0);
    apoint_ts b_d(ta,3.0);

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
    auto a_f = apoint_ts::deserialize(xmls_unbound);
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
TEST_CASE("study_vector_serialization_performance") {
	/** The purpose of this 'test' is just to study the
	 * number of object x items, vs. typical ts
	 * serialization performance.
	 * Ideally we would like memcpy speed(~10GB for sizes that matters,faster (like 10x) for smaller sizes that fits to cache)
	 * but boost serialization overhead applies,
	 * first of all, its a 1. copy from working memory to string-stream buffer, (serialize_loop)
	 * then its a 2nd copy/alloc from the serialized form back to the cloned image.
	 *
	 * typical numbers here shows that we get only 1/10th of practical memcpy speed, but accounted for
	 * the double copy as explained above, we get  1/5th of practical memcpy speed.
	 */
	using shyft::time_series::dd::apoint_ts;
	bool verbose = getenv("SHYFT_VERBOSE");
	size_t n = 8 * 365 * 5;
	size_t n_o = 10; // 100 here gives approx 1.16 GB to copy, 10 is ok for testing
	size_t n_ts = 100;
	vector<vector<vector<double>>> o; o.reserve(n_o);
	auto t0 = timing::now();
	for (size_t i = 0; i < n_o; ++i) {
		vector<vector<double>> tsv; tsv.reserve(n_ts);
		for (size_t j = 0; j < n_ts; ++j)
			tsv.emplace_back(vector<double>(n, double(i)));
		o.emplace_back(move(tsv));
	}
	auto t1 = timing::now();
	auto _0_1_s = elapsed_us(t0, t1) / 1e6;
	if(verbose) cout << "create "<<n_o<<"o x "<<n_ts<<"ts ("<<n_o*n_ts*n/1000000<<"Mpts) took " <<_0_1_s  << " s \n";
	auto t2 = timing::now();
	auto o2 = serialize_loop(o);
	auto t3 = timing::now();
	auto _2_3_s = elapsed_us(t2, t3) / 1e6;
	cout << "serialize loop " << _2_3_s << "s "<< _2_3_s/_0_1_s << "x slower(typical 6x)" << endl;
	vector<vector<apoint_ts>> otsv; otsv.reserve(n_o);
	calendar utc;
	auto dt = deltahours(3);
	time_axis::generic_dt ta(utc.time(2016, 1, 1), dt, n);
	for (size_t i = 0; i < n_o; ++i) {
		vector<apoint_ts> tsv; tsv.reserve(n_ts);
		for(size_t j=0;j<n_ts;++j)
			tsv.emplace_back(apoint_ts(ta, double(i + 1), time_series::POINT_AVERAGE_VALUE));
		otsv.emplace_back(move(tsv));
	}
	auto t4 = timing::now();
	auto tsv2 = serialize_loop(otsv);
	auto t5 = timing::now();
	auto _4_5_s = elapsed_us(t4, t5) / 1e6;
	if(verbose) cout << "similar with ts took " << _4_5_s << "s\n";
}
TEST_CASE("study_serialization_performance") {
	/**This test is for study if/how serialization can benefit from threading
	 * The answer is 'to a certain degree and it depends'.
	 * Large memory footprint serialization does not,
	 * a lot of small objects does do a degree, but other
	 * approaches that flattens the expression graphs so far shows that its
	 * more to gain doing optimization the way we represents expresssions before
	 * doing the serialization.
	 */
	using namespace time_series;
	using dd::gpoint_ts;
	using dd::gta_t;
	using dd::apoint_ts;
	using dd::ipoint_ts;

    bool verbose = getenv("SHYFT_VERBOSE") ? true : false;
    //
    // 1. create one large ts, do loop it.
    //
    for (size_t n_threads = 1;n_threads < (verbose?6:1);++n_threads) {
        calendar utc;
        size_t n = 10 * 1000 * 1000;// gives 80 Mb memory
        vector<double> x(n, 0.0);
        vector<apoint_ts> av;
        for (size_t i = 0;i < n_threads;++i)
            av.emplace_back(gta_t(utc.time(2016, 1, 1), deltahours(1), n), x);

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

        auto b = apoint_ts::deserialize(xmls);
        t0 = clock();
        vector<future<void>> calcs2;
        for (size_t i = 0;i < n_threads;++i) {
            calcs2.emplace_back(
                async(launch::async, [&xmlsv, i]() {
                auto b = apoint_ts::deserialize(xmlsv[i]);
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

TEST_CASE("study_serialization_memcpy_performance") {
	/** Would more threads copy memory speed up ?
	 *  Depends on memory size copied and cpu-cache
	 */
    bool verbose = getenv("SHYFT_VERBOSE") ? true : false;
    //
    // 1. create one large ts, do loop it.
    //
    for (size_t n_threads = 1;n_threads < (verbose ? 6 : 1);++n_threads) {
        calendar utc;
        size_t n = 10 * 1000 * 1000;// gives 80 Mb memory
		vector<vector<double>> av; av.reserve(n_threads);
        for (size_t i = 0;i < n_threads;++i)
            av.emplace_back(n,0.0);
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

                    delete []y;
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
#if 0
api::ats_vector expr;
for (size_t o = 0; o<n_obj; ++o) {
	api::ats_vector o_ts;
	api::ats_vector o_ts_sym;
	apoint_ts o_expr(ta, double(o), time_series::ts_point_fx::POINT_AVERAGE_VALUE);
	apoint_ts o_expr_sym(string("obj_") + to_string(o));

	for (size_t i = 0; i < n_ts; ++i) {
		apoint_ts term(ta, double(i), time_series::ts_point_fx::POINT_AVERAGE_VALUE);
		apoint_ts term_sym(string("obj_") + to_string(o) + ".id_" + to_string(i));
		o_ts.push_back((1 + i)*o_expr*term - double(o));
		o_ts_sym.push_back((1 + i)*o_expr_sym*term_sym - double(o));
	}
	expr = expr.size() ? expr + o_ts : o_ts;
	expr_sym = expr_sym.size() ? expr_sym + o_ts_sym : o_ts_sym;
}

auto expr_avg = expr.average(ta24);
std::vector<double> avg_v; avg_v.reserve(n);
vector<int> pct = { 0,10, -1,90,100 };
auto t0 = timing::now();
auto expr_pct = expr.percentiles(ta24, pct);
auto us = elapsed_us(t0, timing::now());
if (verbose) {
	cout << "percentile 24h " << n_obj << " x " << n_ts << " ts, each with "
		<< n << " values to one ts took " << us / 1e6 << " s"
		<< ",\nsize of final result is " << expr_pct.size()
		<< endl;
}
FAST_WARN_LE(us / 1e6, 1.0);// 0.1 second on fast computer
t0 = timing::now();
auto expr_e = api::deflate_ts_vector<apoint_ts>(expr_avg);
us = elapsed_us(t0, timing::now());
if (verbose) {
	cout << "deflate 24h " << n_obj << " x " << n_ts << " ts, each with "
		<< n << " values to one ts took " << us / 1e6 << " s"
		<< ",\nsize of final result is " << expr_e.size()
		<< endl;
}
FAST_WARN_LE(us / 1e6, 1.0);// 0.1 second on fast computer
#endif

TEST_CASE("study_apoint_ts_expression_speed") {
    /**
	* This is about how fast we can serialize an expression.
	* typical cases from the LTM-AP projects,
	* 100 objects, each with 100 time-series (scenario-dimension), each with terminals that have 5years 3h resolution time-series(14600 points/ts)
	* are summed together into 100 expressions( like sum of productionx price for each scenario)
	* then percentiles are computed.
	*
	* It appears that the serialization speed of the expression-tree takes significant time
	*  to transfer. The example above with some computational complexity takes ~150 ms to round-trip in memory.
	*
	* This is due to the excellent object-tracking (needed) provided as standard from the boost
	* serialization library. .. and also from the fact that the expression tree will be n-objects (at least ) deep. (a+b+c+d..)
	*
	* By tweaking the expression representation for before serialization, flatten the nodes to binary-serializable structure
	* we remove the tracking-overhead
	*
	* In addition, the terminals in these cases are aref_ts (symbolic ts that needs binding at server-side).
	*  - so serializing the strings instead of aref_ts reduces object tracking work of boost.
	*
	*/
    bool verbose = getenv("SHYFT_VERBOSE") ? true : false;
	using namespace time_series;
	using dd::gpoint_ts;
	using dd::gta_t;
	using dd::apoint_ts;
	using dd::ipoint_ts;
	using dd::ats_vector;

    calendar utc;
    size_t n_steps_pr_day = 8;
    auto dt=deltahours(24/n_steps_pr_day);
    const size_t n = 1*100*n_steps_pr_day; // could be 5*365
    const size_t n_ts= 100;//00;
    const size_t n_obj=100;//00;
    static_assert(n_obj>=3,"Need at least 3 objects for this test to run");
    auto t_start=utc.time(2016,1,1);
    time_axis::generic_dt ta(t_start,dt,n);
    time_axis::generic_dt ta24(t_start,deltahours(24),n/n_steps_pr_day);

    ats_vector expr_sym;
    for(size_t o=0;o<n_obj;++o) {
        ats_vector o_ts_sym;
		apoint_ts o_expr_sym(string("obj_") + to_string(o));
		if(false) // put true to do a mixed payload
			o_expr_sym.bind(apoint_ts(ta, double(o), time_series::POINT_AVERAGE_VALUE));

        for (size_t i = 0;i < n_ts;++i) {
			if (i == n_ts+1) { /// put ==0 to give a heavier payload
				o_ts_sym.push_back((1+i)*o_expr_sym*apoint_ts(ta, double((o + 1)*(i + 1) / 10.0),time_series::POINT_AVERAGE_VALUE)-double(o));
			} else {
				apoint_ts term_sym(string("obj_") + to_string(o) + ".id_" + to_string(i));
				o_ts_sym.push_back((1 + i)*o_expr_sym*term_sym - double(o));
			}
        }
        expr_sym =expr_sym.size()?expr_sym + o_ts_sym:o_ts_sym;
    }
    //expr_sym = expr_sym.average(ta24).accumulate(ta24).integral(ta24);// ensure to test cycle all avg
    expr_sym.push_back(apoint_ts(vector<double>{1.0,2,3,4,5,6,7,8},deltahours(3),ta ));// ensure to test pattern ts.
	auto pattern_ts_idx = expr_sym.size() - 1;// remember for krls
    expr_sym.push_back(expr_sym[0].convolve_w(vector<double>{0.2,0.3,0.5,0.3,0.2},time_series::convolve_policy::USE_ZERO));// test convolve_w
    expr_sym.push_back(expr_sym[1].extend(expr_sym[2],dd::extend_ts_split_policy::EPS_VALUE,dd::extend_ts_fill_policy::EPF_FILL,t_start+deltahours(24),3.14));
    time_series::rating_curve_segment rc_s0(0.0,1.0,2.0,3.0);
    time_series::rating_curve_function rc_f0;rc_f0.add_segment(rc_s0);
    time_series::rating_curve_parameters rc_param;rc_param.add_curve(t_start,rc_f0);
    expr_sym.push_back(expr_sym[0].rating_curve(rc_param));
    expr_sym.push_back(expr_sym[pattern_ts_idx].krls_interpolation(deltahours(24),1e-3,0.1,100));
	auto krls_idx = expr_sym.size()-1;
    expr_sym.push_back(expr_sym[0].min_max_check_linear_fill(0,100.0,deltahours(1000)));
    expr_sym.push_back(expr_sym[0].min_max_check_ts_fill(0,100.0,deltahours(1000),expr_sym[1]));
	expr_sym.push_back(expr_sym[0].average(ta24).accumulate(ta24).integral(ta24));
    auto t0=timing::now();
    decltype(t0) t1_0,t1_1,t1_2,t1_3;
	vector<apoint_ts> expr_dz;
	using dd::expression_compressor;
    using dd::expression_decompressor;// convert_to_ts_vector;
    {
        t1_0=timing::now();
		auto expr_converted = expression_compressor::compress(expr_sym);
		t1_1=timing::now();
		auto expr_transported = serialize_loop(expr_converted, core_arch_flags | boost::archive::archive_flags::no_codecvt);//|boost::archive::archive_flags::no_tracking);
		t1_2=timing::now();
		expr_dz = expression_decompressor::decompress(expr_transported);
		t1_3=timing::now();
	}

	auto t1 = timing::now();
	auto expr_dz2 = serialize_loop(expr_sym, core_arch_flags | boost::archive::archive_flags::no_codecvt);
	auto us2 = elapsed_us(t1, timing::now());
    if(verbose) {
		cout << "Expression-serialization: in-memory sym " << n_obj << " x " << n_ts << " ts, each with "
			<< "\n\tsize of final result is " << expr_dz.size() << endl
			<<"\t -> ts_expr_rep "<<elapsed_us(t1_0,t1_1)/1e6<< " ts_serialize-loop expr_rep "<<elapsed_us(t1_1,t1_2)/1e6 << " -> atsv "<< elapsed_us(t1_2,t1_3)/1e6<<endl
            <<"\t -> total time used : "<<elapsed_us(t1_0,t1_3)/1e6<<"s\n"
			<< "\t std serialization took " << us2 / 1e6 << " s" << endl;
		;
    }
	// finale: do binding on expr_sym (the original), then on expr_dz (expression serialized)
	// and compare the results
	FAST_REQUIRE_EQ(expr_sym.size(), expr_dz.size());
	double fill_value = 1.0;
	size_t expr_sym_count = 0;
	size_t expr_dz_count = 0;
	for (size_t i = 0; i < expr_sym.size();++i) {
		apoint_ts bts(ta, double(fill_value), time_series::POINT_AVERAGE_VALUE);
		/* */ {
			auto unbound_ts_list = expr_sym[i].find_ts_bind_info();
			for (auto& bind_info : unbound_ts_list) {
				bind_info.ts.bind(bts);
				++expr_sym_count;
			}
			expr_sym[i].do_bind();
		}
		/* */ {
			auto unbound_ts_list = expr_dz[i].find_ts_bind_info();
			for (auto& bind_info : unbound_ts_list) {
				bind_info.ts.bind(bts);
				++expr_dz_count;
			}
			expr_dz[i].do_bind();
		}
		fill_value += 1.0;
	}
	FAST_CHECK_EQ(expr_dz_count, expr_sym_count);
	auto expr_sym_evaluated = dd::deflate_ts_vector<dd::gts_t>(expr_sym);
	auto expr_dz_evaluated = dd::deflate_ts_vector<dd::gts_t>(expr_dz);
	FAST_REQUIRE_EQ(expr_sym_evaluated.size(), expr_dz_evaluated.size());
	for (size_t i = 0; i < expr_sym_evaluated.size(); ++i) {
		double eps = i == krls_idx ? 0.5 : 1e-12;// allow for slack in krls ?
		bool ts_equal_for_expr_and_classic = is_equal(expr_sym_evaluated[i], expr_dz_evaluated[i], eps);
		if (!ts_equal_for_expr_and_classic) {
			cout << " evaluated result differs for time-series number " << i << " krls indx is "<< krls_idx <<endl;
			FAST_WARN_EQ(ts_equal_for_expr_and_classic, true);
		}
	}
}
TEST_CASE("test_tuple_serialization") {
    using namespace shyft::time_series::dd;
	compressed_ts_expression xtra;
    srep::sbinop_op_ts b1{ iop_t::OP_ADD, o_index<aref_ts>{1},o_index<gpoint_ts>{2}};
	//std::get<0>(xtra.ts_reps).push_back( b1);
    xtra.append(b1);
	auto xtra2 = serialize_loop(xtra);
	FAST_CHECK_EQ(std::get<0>(xtra.ts_reps).size(), std::get<0>(xtra2.ts_reps).size());
    FAST_CHECK_EQ(std::get<0>(xtra.ts_reps)[0], std::get<0>(xtra2.ts_reps)[0]);
}

} // end TEST_SUITE
