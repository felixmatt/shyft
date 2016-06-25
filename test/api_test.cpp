#include "test_pch.h"
#include "api_test.h"

// from core
#include "core/utctime_utilities.h"
#include "core/cell_model.h"
#include "core/pt_gs_k_cell_model.h"


#include "api/api.h"
#include "api/pt_gs_k.h"
#include "api/pt_ss_k.h"
#include "api/pt_hs_k.h"

using namespace std;
using namespace shyft::core;
using namespace shyft::timeseries;

using namespace shyft::api;


void api_test::test_ptgsk_state_io() {
	using namespace shyft::api;
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
void api_test::test_ptssk_state_io() {
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

void api_test::test_pthsk_state_io() {
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

void api_test::test_geo_cell_data_io() {
    geo_cell_data gcd(geo_point(1,2,3),4,5,0.6,land_type_fractions(2,4,6,8,10));

    string gcd_s=geo_cell_data_io::to_string(gcd);
    TS_ASSERT(gcd_s.size()>0);
    geo_cell_data gcd2= geo_cell_data_io::from_string(gcd_s.data());
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
