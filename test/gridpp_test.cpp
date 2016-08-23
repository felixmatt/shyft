#include "test_pch.h"
#include "gridpp_test.h"
#include "mocks.h"

using namespace shyft::core;
using namespace shyfttest;
using namespace shyfttest::idw;

void gridpp_test::test_interpolate_sources_should_populate_grids() {

	utctime Tstart = calendar().time(YMDhms(2000, 1, 1));
	utctimespan dt = 3600L;
	const int nt = 24*36;
	TimeAxis ta(Tstart, dt, nt);
	
	const int nss = 3; // Number of source samples in each direction
	const double dss = 3000; // Sampling distance for sources is typical 3000 m
	const int ngs = 3 * nss; // Number of grid samples
	const double g0 = -500; // Coordinate of grid origin in m
	Parameter p(2 * dss, 4); // Distance and neighbors

	auto s(move(Source::GenerateTestSourceGrid(ta, nss, nss, g0, g0, dss)));
	auto d(move(MCell::GenerateTestGrid(ngs, ngs)));

	run_interpolation<TestTemperatureModel>(begin(s), end(s), begin(d), end(d), idw_timeaxis<TimeAxis>(ta), 
		p, [](MCell& d, size_t ix, double v) {d.set_value(ix, v); });

	TS_ASSERT_EQUALS(count_if(begin(d), end(d), [nt](const MCell& d) {return d.set_count == nt; }), ngs*ngs);
	TS_ASSERT_EQUALS(count_if(begin(d), end(d), [](const MCell& d) {return d.v > 0; }), ngs*ngs);
}

void gridpp_test::test_main_workflow_should_populate_grids() {
	// The main workflow for offset-bias is
	// T_forecast_1x1 = IDW(T_arome_2.5x2.5, 1x1, idw-parameters) + T_bias
	// Do the same correction for scaled-bias variables

	utctime t0 = calendar().time(YMDhms(2000, 1, 1));
	utctimespan dt = 3600L;
	const int nt = 24*36;
	TimeAxis ta(t0, dt, nt);

	const int nsx = 1;
	const int nsy = 1;
	const int ngx = 3 * nsx;
	const int ngy = 3 * nsy;
	const double s0 = -500;
	const double dss = 2500;
	Parameter p(2 * dss, 4);

	//auto Tsour(move(Source::GenerateTestSourceGrid(ta, nsx, nsy, s0, s0, dss)));
	auto Tsour(move(PointTimeSerieSource::GenerateTestSources(ta, nsx, nsy, s0, s0, dss)));
	auto Tdest(move(MCell::GenerateTestGrid(ngx, ngy)));
	auto Tbias(move(MCell::GenerateTestGrid(ngx, ngy)));

	// Tdest = IDW(Tsour, Tdest) // TODO: test with fixed_dt
	run_interpolation<TestTemperatureModel_1>(begin(Tsour), end(Tsour), begin(Tdest), end(Tdest), idw_timeaxis<TimeAxis>(ta),
		p, [](MCell& d, size_t ix, double v) {d.set_value(ix, v); });
	
	//for_each(begin(Tdest), end(Tdest), [](auto d) { cout << '\n' << d.v; });
	//cout << '\n';

	// Tdest += Tbias
	for (auto itdest = Tdest.begin(), itbias = Tbias.begin(); itdest != Tdest.end() || itbias != Tbias.end(); ++itdest, ++itbias) {
		(*itdest).v += (*itbias).v;
	}
	
	//for_each(begin(Tdest), end(Tdest), [](auto d) { cout << '\n' << d.v; });
	//cout << '\n';
}
