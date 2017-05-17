#include "test_pch.h"

// from core
#include "core/utctime_utilities.h"
#include "core/cell_model.h"
#include "core/pt_gs_k_cell_model.h"


#include "api/api.h"
#include "api/pt_gs_k.h"
#include "api/pt_ss_k.h"
#include "api/pt_hs_k.h"
#include "api/api_state.h"

using namespace std;
using namespace shyft::core;
using namespace shyft::time_series;

using namespace shyft::api;

TEST_SUITE("api") {
TEST_CASE("test_ptgsk_state_io") {
	pt_gs_k_state_t s;
	s.gs.albedo = 0.5;
	s.gs.alpha = 0.8;
	s.gs.sdc_melt_mean = 12.2;
	s.gs.acc_melt = 100;
	s.gs.iso_pot_energy = 12003;
	s.gs.temp_swe = 2.1;
	s.gs.lwc = 1.2;
	s.gs.surface_heat = 3000.0;
	s.kirchner.q = 12.2;
	pt_gs_k_state_io sio;
	string s1 = sio.to_string(s);
	pt_gs_k_state_t  sr;
	TS_ASSERT(sio.from_string(s1, sr));
	TS_ASSERT_EQUALS(s, sr);
	vector<pt_gs_k_state_t> sv;
	size_t n = 20 * 20;
	sv.reserve(n);
	for (size_t i = 0; i<n; ++i) {
		sv.emplace_back(s);
		s.gs.temp_swe += 0.01;
	}
	string ssv = sio.to_string(sv);
	vector<pt_gs_k_state_t> rsv;
	rsv = sio.vector_from_string(ssv);
	TS_ASSERT_EQUALS(rsv.size(), sv.size());
	for (size_t i = 0; i<sv.size(); ++i) {
		TS_ASSERT_EQUALS(rsv[i], sv[i]);
	}

}
TEST_CASE("test_ptssk_state_io") {
	using namespace shyft::api;
	pt_ss_k_state_t s;
	s.snow.nu = 5;
	s.snow.alpha = 0.8;
	s.snow.sca = 12.2;
	s.snow.swe = 100;
	s.snow.free_water = 12003;
	s.snow.residual = 2.1;
	s.snow.num_units = 1001;
	s.kirchner.q = 12.2;
	pt_ss_k_state_io sio;
	string s1 = sio.to_string(s);
	pt_ss_k_state_t  sr;
	TS_ASSERT(sio.from_string(s1, sr));
	TS_ASSERT_EQUALS(s, sr);
	vector<pt_ss_k_state_t> sv;
	size_t n = 20 * 20;
	sv.reserve(n);
	for (size_t i = 0; i < n; ++i) {
		sv.emplace_back(s);
		s.snow.swe += 0.01;
		s.snow.num_units++;
	}
	string ssv = sio.to_string(sv);
	vector<pt_ss_k_state_t> rsv;
	rsv = sio.vector_from_string(ssv);
	TS_ASSERT_EQUALS(rsv.size(), sv.size());
	for (size_t i = 0; i < sv.size(); ++i) {
		TS_ASSERT_EQUALS(rsv[i], sv[i]);
	}

}

TEST_CASE("test_pthsk_state_io") {
	using namespace shyft::api;
	pt_hs_k_state_t s;
	s.snow.sca = 12.2;
	s.snow.swe = 100;
	s.kirchner.q = 12.2;
	pt_hs_k_state_io sio;
	string s1 = sio.to_string(s);
	pt_hs_k_state_t  sr;
	TS_ASSERT(sio.from_string(s1, sr));
	TS_ASSERT_EQUALS(s, sr);
	vector<pt_hs_k_state_t> sv;
	size_t n = 2 * 2;
	sv.reserve(n);
	for (size_t i = 0; i < n; ++i) {
		sv.emplace_back(s);
		s.snow.swe += 0.01;
	}
	string ssv = sio.to_string(sv);
	vector<pt_hs_k_state_t> rsv;
	rsv = sio.vector_from_string(ssv);
	TS_ASSERT_EQUALS(rsv.size(), sv.size());
	for (size_t i = 0; i < sv.size(); ++i) {
		TS_ASSERT_EQUALS(rsv[i], sv[i]);
	}

}

TEST_CASE("test_geo_cell_data_io") {
    geo_cell_data gcd(geo_point(1,2,3),4,5,0.6,land_type_fractions(2,4,6,8,10));

    auto gcd_s=geo_cell_data_io::to_vector(gcd);
    TS_ASSERT(gcd_s.size()>0);
    geo_cell_data gcd2= geo_cell_data_io::from_vector(gcd_s);
    double eps=1e-12;
    TS_ASSERT_DELTA(gcd2.area(),gcd.area(),eps);
    TS_ASSERT_EQUALS(gcd2.mid_point(),gcd.mid_point());
    TS_ASSERT_EQUALS(gcd2.catchment_id(),gcd.catchment_id());
    TS_ASSERT_DELTA(gcd2.radiation_slope_factor(),gcd.radiation_slope_factor(),eps);
    TS_ASSERT_DELTA(gcd2.land_type_fractions_info().glacier(),gcd.land_type_fractions_info().glacier(),eps);
    TS_ASSERT_DELTA(gcd2.land_type_fractions_info().lake(),gcd.land_type_fractions_info().lake(),eps);
    TS_ASSERT_DELTA(gcd2.land_type_fractions_info().reservoir(),gcd.land_type_fractions_info().reservoir(),eps);
    TS_ASSERT_DELTA(gcd2.land_type_fractions_info().forest(),gcd.land_type_fractions_info().forest(),eps);
    TS_ASSERT_DELTA(gcd2.land_type_fractions_info().unspecified(),gcd.land_type_fractions_info().unspecified(),eps);
}



/** Here we try to build a test-story from start to end that covers state-io-extract-restore*/
TEST_CASE("test_state_with_id_functionality") {
    typedef shyft::core::pt_gs_k::cell_discharge_response_t xcell_t;
    cell_state_id a{ 1,2,3,10 };
    cell_state_id b{ 1,2,4,20 };
    cell_state_id c{ 2,2,4,20 };
    TS_ASSERT_DIFFERS(a, b);
    std::map<cell_state_id, int> smap;
    smap[a] = a.area;
    smap[b] = b.area;
    TS_ASSERT_EQUALS(smap.find(c), smap.end());
    TS_ASSERT_DIFFERS(smap.find(a), smap.end());
    // ------------------------------- (x,y,z),area,cid)
    xcell_t c1{ geo_cell_data(geo_point(1,1,1),  10,  1) };c1.state.kirchner.q = 1.1;
    xcell_t c2{ geo_cell_data(geo_point(1,2,1),  10,  1) };c2.state.kirchner.q = 1.2;
    xcell_t c3{ geo_cell_data(geo_point(2,1,1),  10,  2) };c3.state.kirchner.q = 2.1;
    xcell_t c4{ geo_cell_data(geo_point(2,2,1),  10,  2) };c4.state.kirchner.q = 2.2;
    auto cv = make_shared<vector<xcell_t>>();
    cv->push_back(c1);
    cv->push_back(c2);
    cv->push_back(c3);
    cv->push_back(c4);
    state_io_handler<xcell_t> xh(cv);
    auto s0 = xh.extract_state(vector<int>());
    TS_ASSERT_EQUALS(s0->size(), cv->size());
    for (size_t i = 0;i < cv->size();++i)
        TS_ASSERT_EQUALS((*s0)[i].id, cell_state_id_of((*cv)[i].geo));// ensure we got correct id's out.
    //-- while at it, also verify serialization support
    auto bytes = serialize_to_bytes(s0);
    TS_ASSERT(bytes.size()> 10);
    shared_ptr<vector<cell_state_with_id<xcell_t::state_t>>> s0_x;
    deserialize_from_bytes(bytes, s0_x);
    TS_ASSERT_EQUALS(s0_x->size(), s0->size());
    for (size_t i = 0;i < s0->size();++i) {
        TS_ASSERT_EQUALS((*s0_x)[i].id, (*s0)[i].id);//equality by identity only check
        TS_ASSERT_DELTA( (*s0_x)[i].state.kirchner.q, (*s0)[i].state.kirchner.q, 0.01);
    }
    auto s1 = xh.extract_state(vector<int>{2}); // ok, now specify cids, 2, only two cells match
    TS_ASSERT_EQUALS(s1->size(), size_t( 2));
    for (size_t i = 0;i < s1->size();++i)
        TS_ASSERT_EQUALS((*s1)[i].id, cell_state_id_of((*cv)[i+2].geo));// ensure we got correct id's out.

    auto s_missing = xh.extract_state(vector<int>{3});
    TS_ASSERT_EQUALS(s_missing->size(), 0u);

    auto m0 = xh.apply_state(s0, vector<int>());
    TS_ASSERT_EQUALS(m0.size(), 0u);

    auto m0_x = xh.apply_state(s0, vector<int>{4});
    TS_ASSERT_EQUALS(m0_x.size(), 0u); // because we passed in states not containing 4
    (*s0)[0].id.cid = 4; //stuff in a 4, and get one missing below
    auto m0_y = xh.apply_state(s0, vector<int>{4});
    TS_ASSERT_EQUALS(m0_y.size(), 1u);
    TS_ASSERT_EQUALS(m0_y[0], 0);
}
}
