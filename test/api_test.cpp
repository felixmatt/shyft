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

bool operator==(const pt_gs_k_state_t& a, const pt_gs_k_state_t& b) {
	const double tol = 1e-9;
	return fabs(a.gs.albedo - b.gs.albedo) < tol
		&& fabs(a.gs.alpha - b.gs.alpha) < tol
		&& fabs(a.gs.sdc_melt_mean - b.gs.sdc_melt_mean) < tol
		&& fabs(a.gs.acc_melt - b.gs.acc_melt) < tol
		&& fabs(a.gs.iso_pot_energy - b.gs.iso_pot_energy) < tol
		&& fabs(a.gs.temp_swe - b.gs.temp_swe) < tol
		&& fabs(a.kirchner.q - b.kirchner.q) < tol
		&& fabs(a.gs.lwc - b.gs.lwc) < tol
		&& fabs(a.gs.surface_heat - b.gs.surface_heat) < tol;

}

bool operator==(const pt_ss_k_state_t& a, const pt_ss_k_state_t& b) {
	const double tol = 1e-9;
	return fabs(a.snow.nu - b.snow.nu) < tol
		&& fabs(a.snow.alpha - b.snow.alpha) < tol
		&& fabs(a.snow.sca - b.snow.sca) < tol
		&& fabs(a.snow.swe - b.snow.swe) < tol
		&& fabs(a.snow.free_water - b.snow.free_water) < tol
		&& fabs(a.snow.residual - b.snow.residual) < tol
		&& fabs(a.kirchner.q - b.kirchner.q) < tol
		&& (a.snow.num_units == b.snow.num_units);

}
bool operator==(const pt_hs_k_state_t& a, const pt_hs_k_state_t& b) {
	const double tol = 1e-9;
	return
		   fabs(a.snow.sca - b.snow.sca) < tol
		&& fabs(a.snow.swe - b.snow.swe) < tol
		&& fabs(a.kirchner.q - b.kirchner.q) < tol;

}

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
