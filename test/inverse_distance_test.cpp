#include "test_pch.h"
#include "inverse_distance_test.h"
#include "mocks.h"

#include <ctime>
#include <cmath>

#ifdef WIN32
#if _MSC_VER < 1800
const unsigned long nanx[2] = { 0xffffffff, 0x7fffffff };
const double NAN = *(double*)nanx;
#endif
#endif
#include <armadillo>

using namespace shyft::core;
using namespace shyfttest::idw;

void inverse_distance_test::test_temperature_model() {
	//
	// Verify temperature gradient calculator, needs to be robust ...
	//
	Parameter p(100 * 1000.0, 10);
	TestTemperatureModel::scale_computer gc(p);
	TS_ASSERT_DELTA(gc.compute(), p.default_gradient(), TEST_EPS); // should give default gradient by default

	geo_point p1(1000, 1000, 100);
	Source   s1(p1, 10);
	utctime  t0 = 3600L * 24L * 365L * 44L;

	gc.add(s1, t0);
	TS_ASSERT_DELTA(gc.compute(), p.default_gradient(), TEST_EPS); // should give default gradient if just one point

	geo_point p1b(1000, 1000, 149);
	Source   s1b(p1b, 10 - 0.005 * 59);
	gc.add(s1b, t0);

	TS_ASSERT_DELTA(gc.compute(), p.default_gradient(), TEST_EPS); // mi-max z-distance less than 50 m., return default.


	geo_point p2(2000, 2000, 200);
	Source   s2(p2, 9.5);
	gc.add(s2, t0);
	TS_ASSERT_DELTA(gc.compute(), -0.005, TEST_EPS); // should give -0.005 gradient if just two points

	geo_point p3(3000, 3000, 300);
	Source   s3(p3, 9.0);
	gc.add(s3, t0);
	TS_ASSERT_DELTA(gc.compute(), -0.005, TEST_EPS); // should give -0.005 gradient for these 3 points

	geo_point p4(4000, 4000, 500);
	Source   s4(p4, 8.0);
	gc.add(s4, t0);
	TS_ASSERT_DELTA(gc.compute(), -0.005, TEST_EPS);// should give -0.005 gradient for these 4 points

	geo_point p5(4000, 4000, 600);
	Source   s5(p5, 10 - 0.006*(600 - 100));
	gc.add(s5, t0);
	TS_ASSERT_DELTA(gc.compute(), -0.006, TEST_EPS); // mi-max alg. should only consider high/low ..


	//
	// Verify the TestTemperatureModel::transform, should do temp.gradient computation based on height difference.
	//
	MCell d1(geo_point(1500, 1500, 200));
	double sourceValue = 10.0;
	double scaleValue = -0.005;

	double transformedValue = TestTemperatureModel::transform(sourceValue, scaleValue, s1, d1);

	TS_ASSERT_DELTA(transformedValue, sourceValue + scaleValue*(d1.point.z - s1.point.z), TEST_EPS);
}

void inverse_distance_test::test_temperature_model_default_gradient() {
	inverse_distance::temperature_parameter p;
	p.default_temp_gradient = 1.0;
	inverse_distance::temperature_default_gradient_scale_computer gsc(p);
	TS_ASSERT_DELTA(p.default_temp_gradient, gsc.compute(), TEST_EPS);
	TS_ASSERT(inverse_distance::temperature_default_gradient_scale_computer::is_source_based() == false);
}

void inverse_distance_test::test_radiation_model() {
	//
	// Verify temperature gradient calculator, needs to be robust ...
	//
	Parameter p(100 * 1000.0, 10);
	TestRadiationModel::scale_computer gc(p);
	TS_ASSERT_DELTA(gc.compute(), 1.0, TEST_EPS); // should give 1.0 gradient by default

	geo_point p1(1000, 1000, 100);
	Source   s1(p1, 10);
	utctime  t0 = 3600L * 24L * 365L * 44L;

	gc.add(s1, t0);
	TS_ASSERT_DELTA(gc.compute(), 1.0, TEST_EPS); // should give 1.0 gradient if just one point

	geo_point p2(2000, 2000, 200);
	Source   s2(p2, 9.5);
	gc.add(s2, t0);
	TS_ASSERT_DELTA(gc.compute(), 1.0, TEST_EPS); // should give -0.005 gradient if just two points

	//
	// Verify the TestTemperatureModel::transform, should do temp.gradient computation based on height difference.
	//
	MCell d1(geo_point(1500, 1500, 200));
	d1.set_slope_factor(0.5);
	double sourceValue = 10.0;
	double scaleValue = 1.0;

	double transformedValue = TestRadiationModel::transform(sourceValue, scaleValue, s1, d1);

	TS_ASSERT_DELTA(transformedValue, sourceValue*d1.slope_factor(), TEST_EPS);
}

void inverse_distance_test::test_precipitation_model() {
	//
	// Verify temperature gradient calculator, needs to be robust ...
	//
	Parameter p(100 * 1000.0, 10);
	TestPrecipitationModel::scale_computer gc(p);
	TS_ASSERT_DELTA(gc.compute(), p.precipitation_scale_factor(), TEST_EPS);// should give 1.0 gradient by default

	geo_point p1(1000, 1000, 100);
	Source   s1(p1, 10);
	utctime  t0 = 3600L * 24L * 365L * 44L;

	gc.add(s1, t0);
	TS_ASSERT_DELTA(gc.compute(), p.precipitation_scale_factor(), TEST_EPS);// should give 1.0 gradient if just one point

	geo_point p2(2000, 2000, 200);
	Source   s2(p2, 9.5);
	gc.add(s2, t0);
	TS_ASSERT_DELTA(gc.compute(), p.precipitation_scale_factor(), TEST_EPS);// should give -0.005 gradient if just two points

	//
	// Verify the TestTemperatureModel::transform, should do temp.gradient computation based on height difference.
	//
	MCell d1(geo_point(1500, 1500, 200));
	double sourceValue = 10.0;
	double scaleValue = gc.compute();

	double transformedValue = TestPrecipitationModel::transform(sourceValue, scaleValue, s1, d1);

	TS_ASSERT_DELTA(transformedValue, sourceValue*pow(scaleValue, (d1.point.z - s1.point.z) / 100.0), TEST_EPS);

	// Verify that 0.0 transformation to zero
	// Destination d0(GeoPoint(1500,1500,0))
	// TS_ASSERT_DELTA(0.0, TestPrecipitationModel::transform(0.0, scaleValue, s1, d0), TEST_EPS);
}

void inverse_distance_test::test_one_source_one_dest_calculation() {
	//
	// Arrange
	//
	utctime Tstart = 3600L * 24L * 365L * 44L;
	utctimespan dt = 3600L;
	int n = 1; // 24*10;
	const int nx = 1;
	const int ny = 1;
	const int n_sources = 1;
	TimeAxis ta(Tstart, dt, n); // hour, 10 steps
	vector<Source> s(Source::GenerateTestSources(ta, n_sources, 0.5*nx * 1000, 0.5*ny * 1000, 0.25*0.5*(nx + ny) * 1000));// 40 sources, radius 50km, starting at 100,100 km center
	vector<MCell> d(MCell::GenerateTestGrid(nx, ny)); // 200x200 km
	Parameter p(2.75 * 0.5 * (nx + ny) * 1000, max(8, 1 + n_sources / 2));

	//
	// Act
	//

	run_interpolation<TestTemperatureModel>(begin(s), end(s), begin(d), end(d), idw_timeaxis<TimeAxis>(ta), p,
		[](MCell& d, size_t ix, double v) { d.set_value(ix, v); });

	//
	// Assert
	//
	TS_ASSERT_EQUALS(count_if(begin(d), end(d), [n](const MCell &d) { return d.set_count == n; }), nx*ny);

	double expected_v = TestTemperatureModel::transform(s[0].value(utcperiod(Tstart, Tstart + dt)), p.default_gradient(), s[0], d[0]);
	TS_ASSERT_EQUALS(count_if(begin(d), end(d), [n, expected_v](const MCell&d) { return fabs(d.v - expected_v) < 1e-7; }), nx*ny);
}

void inverse_distance_test::test_two_sources_one_dest_calculation() {
	//
	// Arrange
	//
	utctime Tstart = 3600L * 24L * 365L * 44L;
	utctimespan dt = 3600L;
	int n = 1;
	const int nx = 1;
	const int ny = 1;
	const int n_sources = 2;
	TimeAxis ta(Tstart, dt, n);
	vector<Source> s(Source::GenerateTestSources(ta, n_sources, 0.5 * nx * 1000, 0.5 * ny * 1000, 0.25*0.5*(nx + ny) * 1000)); // n sources, radius 50km, starting at 100,100 km center
	vector<MCell> d(MCell::GenerateTestGrid(nx, ny)); // 200x200 km
	Parameter p(2.75 * 0.5 * (nx + ny) * 1000, 1 + n_sources / 2);

	//
	// Act
	//

	run_interpolation<TestTemperatureModel>(begin(s), end(s), begin(d), end(d), idw_timeaxis<TimeAxis>(ta), p,
		[](MCell& d, size_t ix, double v) { d.set_value(ix, v); });

	//
	// Assert
	//
	TS_ASSERT_EQUALS(count_if(begin(d), end(d), [n](const MCell& d) { return d.set_count == n; }), nx*ny);

	TestTemperatureModel::scale_computer gc(p);
	gc.add(s[0], Tstart);
	gc.add(s[1], Tstart);
	double comp_gradient = gc.compute();
	double w1 = 1.0 / TestTemperatureModel::distance_measure(s[0].mid_point(), d[0].mid_point(), 2.0, 1.0);
	double w2 = 1.0 / TestTemperatureModel::distance_measure(s[1].mid_point(), d[0].mid_point(), 2.0, 1.0);
	double v1 = w1*TestTemperatureModel::transform(s[0].value(Tstart), comp_gradient, s[0], d[0]);
	double v2 = w2*TestTemperatureModel::transform(s[1].value(Tstart), comp_gradient, s[1], d[0]);
	double expected_v = (v1 + v2) / (w1 + w2);
	TS_ASSERT_EQUALS(count_if(begin(d), end(d), [n, expected_v](const MCell& d) { return fabs(d.v - expected_v) < 1e-7; }), nx*ny);
}

void inverse_distance_test::test_using_finite_sources_only() {
	//
	// Arrange
	//
	utctime Tstart = 3600L * 24L * 365L * 44L;
	utctimespan dt = 3600L;
	int n = 1;
	const int nx = 1;
	const int ny = 1;
	const int n_sources = 3;
	TimeAxis ta(Tstart, dt, n); // hour, 10 steps
	vector<Source> s(Source::GenerateTestSources(ta, n_sources, 0.5*nx * 1000, 0.5*ny * 1000, 0.25*0.5*(nx + ny) * 1000));// n sources, radius 50km, starting at 100,100 km center
	vector<MCell> d(MCell::GenerateTestGrid(nx, ny)); // 200x200 km
	Parameter p(2.75 * 0.5 * (nx + ny) * 1000, n_sources);

	s[2].set_value(NAN);
	//
	// Act
	//

	run_interpolation<TestTemperatureModel>(begin(s), end(s), begin(d), end(d), idw_timeaxis<TimeAxis>(ta), p,
		[](MCell& d, size_t ix, double v) { d.set_value(ix, v); });

	//
	// Assert
	//
	TS_ASSERT_EQUALS(count_if(begin(d), end(d), [n](const MCell& d) { return d.set_count == n; }), nx*ny);

	TestTemperatureModel::scale_computer gc(p);
	gc.add(s[0], Tstart); gc.add(s[1], Tstart);
	double comp_gradient = gc.compute();
	double w1 = 1.0 / TestTemperatureModel::distance_measure(s[0].mid_point(), d[0].mid_point(), 2.0, 1.0);
	double w2 = 1.0 / TestTemperatureModel::distance_measure(s[1].mid_point(), d[0].mid_point(), 2.0, 1.0);
	double v1 = w1*TestTemperatureModel::transform(s[0].value(Tstart), comp_gradient, s[0], d[0]);
	double v2 = w2*TestTemperatureModel::transform(s[1].value(Tstart), comp_gradient, s[1], d[0]);
	double expected_v = (v1 + v2) / (w1 + w2);
	TS_ASSERT_EQUALS(count_if(begin(d), end(d), [n, expected_v](const MCell& d) { return fabs(d.v - expected_v) < 1e-7; }), nx*ny);
}

void inverse_distance_test::test_eliminate_far_away_sources() {
	//
	// Arrange
	//
	utctime Tstart = 3600L * 24L * 365L * 44L;
	utctimespan dt = 3600L;
	int n = 1;
	const int nx = 1;
	const int ny = 1;
	const int n_sources = 3;
	TimeAxis ta(Tstart, dt, n);
	vector<Source> s(Source::GenerateTestSources(ta, n_sources, 0.5*nx * 1000, 0.5*ny * 1000, 0.25*0.5*(nx + ny) * 1000));// n sources, radius 50km, starting at 100,100 km center
	vector<MCell> d(MCell::GenerateTestGrid(nx, ny)); // 200x200 km
	Parameter p(2.75*0.5*(nx + ny) * 1000, n_sources);
	s[2].point = geo_point(p.max_distance + 1000, p.max_distance + 1000, 300);// place a point far away to ensure it's not part of interpolation
	//
	// Act
	//

	run_interpolation<TestTemperatureModel>(begin(s), end(s), begin(d), end(d), idw_timeaxis<TimeAxis>(ta), p,
		[](MCell& d, size_t ix, double v) { d.set_value(ix, v); });

	//
	// Assert
	//
	TS_ASSERT_EQUALS(count_if(begin(d), end(d), [n](const MCell&d) { return d.set_count == n; }), nx*ny);

	TestTemperatureModel::scale_computer gc(p);
	gc.add(s[0], Tstart); gc.add(s[1], Tstart);
	double comp_gradient = gc.compute();
	double w1 = 1.0 / TestTemperatureModel::distance_measure(s[0].mid_point(), d[0].mid_point(), 2.0, 1.0);
	double w2 = 1.0 / TestTemperatureModel::distance_measure(s[1].mid_point(), d[0].mid_point(), 2.0, 1.0);
	double v1 = w1*TestTemperatureModel::transform(s[0].value(Tstart), comp_gradient, s[0], d[0]);
	double v2 = w2*TestTemperatureModel::transform(s[1].value(Tstart), comp_gradient, s[1], d[0]);
	double expected_v = (v1 + v2) / (w1 + w2);
	TS_ASSERT_EQUALS(count_if(begin(d), end(d), [n, expected_v](const MCell&d) { return fabs(d.v - expected_v) < 1e-7; }), nx*ny);
}

void inverse_distance_test::test_using_up_to_max_sources() {
	//
	// Arrange
	//
	utctime Tstart = 3600L * 24L * 365L * 44L;
	utctimespan dt = 3600L;
	int n = 1;
	const int nx = 1;
	const int ny = 1;
	const int n_sources = 3;
	TimeAxis ta(Tstart, dt, n); //hour, 10 steps
	vector<Source> s(Source::GenerateTestSources(ta, n_sources, 0.5*nx * 1000, 0.5*ny * 1000, 0.25*0.5*(nx + ny) * 1000));// n sources, radius 50km, starting at 100,100 km center
	vector<MCell> d(MCell::GenerateTestGrid(nx, ny)); // 200x200 km
	Parameter p(2.75*0.5*(nx + ny) * 1000, n_sources - 1);
	//
	// Act
	//

	run_interpolation<TestTemperatureModel>(begin(s), end(s), begin(d), end(d), idw_timeaxis<TimeAxis>(ta), p,
		[](MCell& d, size_t ix, double v) {d.set_value(ix, v); });

	//
	// Assert
	//
	TS_ASSERT_EQUALS(count_if(begin(d), end(d), [n](const MCell&d) { return d.set_count == n; }), nx*ny);

	TestTemperatureModel::scale_computer gc(p);
	gc.add(s[0], Tstart); gc.add(s[1], Tstart);
	double comp_gradient = gc.compute();
	double w1 = 1.0 / TestTemperatureModel::distance_measure(s[0].mid_point(), d[0].mid_point(), 2.0, 1.0);
	double w2 = 1.0 / TestTemperatureModel::distance_measure(s[1].mid_point(), d[0].mid_point(), 2.0, 1.0);
	double v1 = w1*TestTemperatureModel::transform(s[0].value(Tstart), comp_gradient, s[0], d[0]);
	double v2 = w2*TestTemperatureModel::transform(s[1].value(Tstart), comp_gradient, s[1], d[0]);
	double expected_v = (v1 + v2) / (w1 + w2);
	TS_ASSERT_EQUALS(count_if(begin(d), end(d), [n, expected_v](const MCell&d) { return fabs(d.v - expected_v) < 1e-7; }), nx*ny);
}

void inverse_distance_test::test_handling_different_sources_pr_timesteps() {
	//
	// Arrange
	//
	utctime Tstart = 3600L * 24L * 365L * 44L;
	utctimespan dt = 3600L;
	int n = 2; // 24*10;
	const int nx = 1;
	const int ny = 1;
	const int n_sources = 3;
	TimeAxis ta(Tstart, dt, n);
	vector<Source> s(Source::GenerateTestSources(ta, n_sources, 0.5*nx * 1000, 0.5*ny * 1000, 0.25*0.5*(nx + ny) * 1000));// n sources, radius 50km, starting at 100,100 km center
	vector<MCell> d(MCell::GenerateTestGrid(nx, ny)); // 200x200 km
	Parameter p(2.75 * 0.5 * (nx + ny) * 1000, n_sources);
	s[2].set_value_at_t(Tstart + dt, NAN);

	// at second, and last.. , timestep, only s[0] and s[1] are valid, 
	// so diff. calc. applies to second step, 
	// the other tests verifies that it can calculate ok first n steps.

	//
	// Act
	//

	run_interpolation<TestTemperatureModel>(begin(s), end(s), begin(d), end(d), idw_timeaxis<TimeAxis>(ta), p,
		[](MCell& d, size_t ix, double v) { d.set_value(ix, v); });

	//
	// Assert
	//
	TS_ASSERT_EQUALS(count_if(begin(d), end(d), [n](const MCell&d) { return d.set_count == n; }), nx*ny);

	TestTemperatureModel::scale_computer gc(p);
	gc.add(s[0], Tstart); gc.add(s[1], Tstart);
	double comp_gradient = gc.compute();
	double w1 = 1.0 / TestTemperatureModel::distance_measure(s[0].mid_point(), d[0].mid_point(), 2.0, 1.0);
	double w2 = 1.0 / TestTemperatureModel::distance_measure(s[1].mid_point(), d[0].mid_point(), 2.0, 1.0);
	double v1 = w1*TestTemperatureModel::transform(s[0].value(Tstart), comp_gradient, s[0], d[0]);
	double v2 = w2*TestTemperatureModel::transform(s[1].value(Tstart), comp_gradient, s[1], d[0]);
	double expected_v = (v1 + v2) / (w1 + w2);
	TS_ASSERT_EQUALS(count_if(begin(d), end(d), [n, expected_v](const MCell&d) { return fabs(d.v - expected_v) < 1e-7; }), nx*ny);
}

void inverse_distance_test::test_performance() {
	//
	// Arrange
	//
	utctime Tstart = calendar().time(2000, 1, 1);
	utctimespan dt = 3600L;
	#ifdef _DEBUG
	int n = 4;// just speed up test.
	#else
	int n = 24 * 36; // number of timesteps
	#endif

	const int n_xy = 3; // number for xy-squares for sources
	const int nx = 3 * n_xy; // 3 times more for grid-cells, typical arome -> cell
	const int ny = 3 * n_xy;
	const int s_nx = n_xy;
	const int s_ny = n_xy;
	const int n_sources = s_nx * s_ny;
	double s_dxy = 3 * 1000; // arome typical 3 km.
	TimeAxis ta(Tstart, dt, n); // hour, 10 steps
	vector<Source> s(move(Source::GenerateTestSourceGrid(ta, s_nx, s_ny, -0.5 * 1000, -0.5 * 1000, s_dxy)));
	vector<MCell> d(move(MCell::GenerateTestGrid(nx, ny)));
	Parameter p(s_dxy * 2, min(8, n_sources / 2)); // for practical purposes, 8 neighbours or less.

	//
	// Act
	//
	const clock_t start = clock();

	run_interpolation<TestTemperatureModel>(begin(s), end(s), begin(d), end(d), idw_timeaxis<TimeAxis>(ta), p,
		[](MCell& d, size_t ix, double v) {d.set_value(ix, v); });
	const clock_t total = clock() - start;
	cout << "\nAlg. IDW2i:\n";
	double time = 1000 * double(total) / double(CLOCKS_PER_SEC);
	cout << "Temperature interpolation took: " << time << " ms" << endl;
	cout << "Each temperature timestep took: " << time / double(n) << " ms" << endl;

	//
	// Assert
	//
	TS_ASSERT_EQUALS(count_if(begin(d), end(d), [n](const MCell& d) {return d.set_count == n; }), nx*ny);
	TS_ASSERT_EQUALS(count_if(begin(d), end(d), [n](const MCell& d) {return d.v >= 0.0; }), nx*ny);
}

static inline
arma::vec3 p_vec(geo_point a, geo_point b) {
	return arma::vec3({ b.x - a.x, b.y - a.y, b.z - a.z });
}

void inverse_distance_test::test_temperature_gradient_model() {
	using namespace arma;
	geo_point p0(0, 0, 10); // 10 deg
	geo_point p1(1000, 0, 110); // 9.4 deg
	geo_point p2(0, 1000, 110);
	geo_point p3(1000, 1000, 220);
	geo_point px(500, 500, 50);

	vec3 dTv({ 0.001, 0.002, 0.001 }); // temp. gradient in x, y, z direction
	auto dT = dTv.t();
	double t0 = 10.0;
	auto p01 = p_vec(p0, p1);
	auto p02 = p_vec(p0, p2);
	auto p03 = p_vec(p0, p3);
	mat33 P(temperature_gradient_scale_computer::p_mat(p0, p1, p2, p3));
	auto t1 = t0 + as_scalar(dT * p01);
	auto t2 = t0 + as_scalar(dT * p02);
	auto t3 = t0 + as_scalar(dT * p03);
	temperature_parameter p(-0.0065, 5, 5 * 1000, true); // turn on using equations to solve gradient
	temperature_gradient_scale_computer sc(p);
	vector<Source> s;
	s.emplace_back(p0, t0);
	s.emplace_back(p1, t1);
	s.emplace_back(p2, t2);
	s.emplace_back(p3, t3);
	utctime tx = calendar().time(YMDhms(2000, 1, 1));
	sc.add(s[0], tx);
	TS_ASSERT_DELTA(sc.compute(), p.default_gradient(), 0.000001); // with one point, default should be returned
	sc.add(s[1], tx);
	TS_ASSERT_DELTA(sc.compute(), (t1 - t0) / (p1.z - p0.z), 0.000001); // with more than one, use min/max method
	sc.add(s[2], tx);
	TS_ASSERT_DELTA(sc.compute(), (t1 - t0) / (p1.z - p0.z), 0.000001); // still min/max method
	sc.add(s[2], tx); // add redundant point, gives singularity, so we should fallback to min-max method.
	TS_ASSERT_DELTA(sc.compute(), (t1 - t0) / (p1.z - p0.z), 0.000001); // still min/max method
	sc.clear(); // forget all points
	for (size_t i = 0; i < s.size(); ++i)
		sc.add(s[i], tx); // fill up with distinct points
	TS_ASSERT_DELTA(sc.compute(), as_scalar(dTv(2)), 0.00001); // now we should get the correct linear vertical
}

void inverse_distance_test::test_zscale_distance() {
	geo_point p0(0, 0, 0);
	geo_point p1(1, 1, 1);
	TS_ASSERT_DELTA(geo_point::distance_measure(p0, p1, 1, 10), pow(1+1+10*10*1,0.5), 1e-9);
	TS_ASSERT_DELTA(geo_point::distance_measure(p0, p1, 2.0, 1.0), pow(1 + 1 + 1, 2.0 / 2.0), 1e-9);
}

/* vim: set filetype=cpp: */
