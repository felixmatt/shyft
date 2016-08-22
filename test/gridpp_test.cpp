#include "test_pch.h"
#include "gridpp_test.h"
#include "mocks.h"

using namespace shyft::core;
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
