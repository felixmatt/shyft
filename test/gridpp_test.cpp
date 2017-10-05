#include "test_pch.h"
#include "mocks.h"
#include "core/region_model.h"
#include "api/api.h" // looking for GeoPointSource, and TemperatureSource(realistic case)
#include "api/time_series.h" // looking for apoint_ts, the api exposed ts-type(realistic case)

using namespace shyft::core;
using namespace shyfttest;
using namespace shyfttest::idw;
namespace ta = shyft::time_axis;

namespace shyfttest {
    using namespace shyft::core;
    struct mock_cell {
        mock_cell(geo_point loc=geo_point()):location(loc){}
        geo_point location;
        const geo_point& mid_point() const {return location;}
        pts_t ts;
        void initialize(const ta::fixed_dt& ta) {
            ts=pts_t(ta,0.0);/// initialize and prepare cell before interpolation step, notice that the lambda to idw uses ts.set(ix,value)
        }
    };

#define TS0V_PRINT(tsv) cout << "\n" #tsv "\n"; \
	for_each(tsv.begin(), tsv.end(), [](auto& a) { cout << a.value(0) << ' '; }); \
	cout << '\n';

#define TS0_EQUAL(ts, v) fabs(ts.value(0) - (v)) < 1e-9
}


TEST_SUITE("gridpp") {
TEST_CASE("test_sih_workbench") {
    // from region_model::run_interpolation, we copy some typedefs to setup
    // a realistic IDW run
    // Triple AAA:
    // Arrange-section:
    using namespace shyft::core;
    using namespace std;
    namespace idw = shyft::core::inverse_distance;

    using ats_t = shyft::api::apoint_ts; // potential break-point time-series, irregular intervals
    using temperature_source = shyft::api::TemperatureSource;
    typedef shyft::time_series::average_accessor<ats_t, ta::fixed_dt> atsa_t;// accessor to ensure bp. ts is projected to fixed interval ta
    typedef idw_compliant_geo_point_ts< temperature_source, atsa_t, ta::fixed_dt> idw_compliant_gts_t;// gts =geo located ts , and idw_compliant to!
	typedef idw::temperature_model<idw_compliant_gts_t, mock_cell , idw::temperature_parameter, geo_point, idw::temperature_gradient_scale_computer> idw_temperature_model_t; // how to compensate for the height at different locations using temp.gradient


    // prepare the time-dimension, using time-axis
    calendar utc;
    utctimespan dt=deltahours(1);
    size_t n=24;
    ta::fixed_dt ta(utc.time(2000,1,1),dt,n); /// for the test,this is the governing time-axis (corresponding to region_model.time_axis during run/interpolation)

    // prepare the geo dimension, the input(s) and the cell grid
    size_t n2_5=10;// gives nice number 25 kilometers in each direction
    size_t n1=25;// that adds up into 25 pieces of 1x1

    vector<temperature_source> arome_2_5km_grid;// ref. algorithm email!
    // generate this 2.5km grid of geo-located temperatures (sources in this context)
    for(size_t x=0;x<n2_5;++x) {
        for(size_t y=0;y<n2_5;++y) { // construct points in a grid, and corresponding ts., summer at sea-level, decreasing to 10.deg at 1000 masl
            arome_2_5km_grid.emplace_back(
                geo_point(x*2500,y*2500,1000.0*(x+y)/(n2_5+n2_5)),// the arome grid midpoint location
                ats_t(ta,20.0 - 10.0*(x+y)/(n2_5+n2_5)) // the fake ts-values at this location, just a constant(t)
            );
        }
    }
    vector<mock_cell> cell_1km_grid;// ref. algorithm email!
    for(size_t x=0;x<n1;++x)
        for(size_t y=0;y<n1;++y)
            cell_1km_grid.emplace_back(geo_point(x*1000,y*1000,1000.0*(x+y)/(n1+n1)));

    //  generate this 1km grid nx1 x ny1
    for(auto &c:cell_1km_grid) /// prep. the cell for receiving interpolated ts.
        c.initialize(ta);

    idw::temperature_parameter idw_parameters;

    //
    // Act-section: (that we later factor out to a function taking a suitable number of template arguments!)
    //  This currently just form the pseudo-code like simple algorithm

    /// 1. IDW( arome_2_5km -> cell 1km, using idw_parameters above)
    ///    after the run, mock_cell.ts will have filled in values as a result of IDW (already tested!)
    idw::run_interpolation<idw_temperature_model_t, idw_compliant_gts_t>(
                                        ta, arome_2_5km_grid, idw_parameters, cell_1km_grid,
                                        [](mock_cell &d, size_t ix, double value) { d.ts.set(ix, value); }
                                );
    /// 2. compute (or mock) the bias
    ///    for now, we just make a vector<geo_located_bias> where
    ///
    using geo_located_bias_ts = geo_point_ts<pts_t>;
    vector<geo_located_bias_ts> bias_1km_grid;

    for(size_t x=0;x<n1;++x) // TODO: instead of this simple filler, replace with bias-computation using historical learning algorithm
        for(size_t y=0;y<n1;++y) {
            geo_located_bias_ts gbts;
            gbts.ts=pts_t(ta,-31.0);// all bias -30.0, so after the end, all result should be less than -10
            gbts.location= geo_point(x*1000,y*1000,1000.0*(x+y)/(n1+n1));
            bias_1km_grid.emplace_back(gbts);
        }

    /// 3. apply bias inplace to the mock_cell.ts
    for(size_t i=0;i<cell_1km_grid.size();++i) {
        //todo: assert same_location(cell_1km_grid[i].mid_point(),bias_1km_grid[i])
        cell_1km_grid[i].ts.add(bias_1km_grid[i].ts);//in-place add values, assume same size/time-axis
    }

    /// Assert-section:
    /// now, since we already have tested IDW, and ts.add etc, we can just
    /// verify some few points
    for(size_t x=0;x<n1;++x)
        for(size_t y=0;y<n1;++y) {
            for(size_t i=0;i<ta.size();++i) {
                double value_at_cell = cell_1km_grid[y*n1+x].ts.v[i];
                TS_ASSERT_LESS_THAN(value_at_cell,-10.0);// since bias is -31.0, we get a range
                TS_ASSERT_LESS_THAN( -31.0, value_at_cell );// -31..-10 approx.
                //ok, we could compute the expected value more accurate
                // but after all, it's based upon already tested functions
            }
        }

}

TEST_CASE("test_interpolate_sources_should_populate_grids") {

    calendar utc;
	utctime Tstart = utc.time(2000, 1, 1);
	utctimespan dt = deltahours(1);
	const int nt = 24*36;
	ta::fixed_dt ta(Tstart, dt, nt);

	const int nss = 3; // Number of source samples in each direction
	const double dss = 3000; // Sampling distance for sources is typical 3000 m
	const int ngs = 3 * nss; // Number of grid samples
	const double g0 = -500; // Coordinate of grid origin in m
	Parameter p(2 * dss, 4); // Distance and neighbors

	auto s{Source::GenerateTestSourceGrid(ta, nss, nss, g0, g0, dss)};
	auto d{MCell::GenerateTestGrid(ngs, ngs)};

	run_interpolation<TestTemperatureModel>(begin(s), end(s), begin(d), end(d), idw_timeaxis<ta::fixed_dt>(ta),
		p, [](MCell& d, size_t ix, double v) {d.set_value(ix, v); });

	TS_ASSERT_EQUALS(count_if(begin(d), end(d), [nt](const MCell& d) {return d.set_count == nt; }), ngs*ngs);
	TS_ASSERT_EQUALS(count_if(begin(d), end(d), [](const MCell& d) {return d.v > 0; }), ngs*ngs);
}

TEST_CASE("test_main_workflow_should_populate_grids") {
	// The main workflow for offset-bias is
	// T_forecast_1x1 = IDW(T_arome_2.5x2.5, 1x1, idw-parameters) + T_bias
	// Do the same correction for scaled-bias variables
	// Test with fixed_dt ta::fixed_dt in Source and compare performance
    calendar utc;
	utctime t0 = utc.time(2000, 1, 1);
	utctimespan dt = deltahours(1);
	const int nt = 24*36;
	ta::fixed_dt ta(t0, dt, nt);

	const int nx = 2;
	const int ny = 2;
	const int ngx = 3 * nx;
	const int ngy = 3 * ny;
	const double temp = 15;
	auto const_ts = point_ts<ta::fixed_dt>(ta, temp);

	// Tsour = vector<Source(geopoint)>(ts<> = 1)
	auto temp_set{PointTimeSerieSource::make_source_set(ta, nx, ny)};
	for_each(temp_set.begin(), temp_set.end(), [&](auto& a) { a.SetTs(const_ts); });

	// Sanity check
	TS_ASSERT_EQUALS(count_if(temp_set.begin(), temp_set.end(), [=](auto& a) { return a.value(0) == temp; }), nx * ny);

	// Tdest = vector<Cell(grid)>(ts<> = 0) => IDW<TemperatureModel>(Tsour, Tdest, fixed_dt)
	auto temp_grid{PointTimeSerieCell::make_cell_grid(ta, ngx, ngy)};
	Parameter p;
	run_interpolation<TestTemperatureModel_1>(temp_set.begin(), temp_set.end(), temp_grid.begin(), temp_grid.end(), idw_timeaxis<ta::fixed_dt>(ta),
		p, [](auto& d, size_t ix, double v) { d.set_value(ix, v); });

	// Expected IDW result
	TS_ASSERT_EQUALS(count_if(temp_grid.begin(), temp_grid.end(), [=](auto& a) {return TS0_EQUAL(a, temp); }), ngx * ngy);

	// Tbias = vector<MCell(grid)>(ts<> = 1, fixed_dt)
	auto bias_grid{PointTimeSerieCell::make_cell_grid(ta, ngx, ngy)};
	const double bias = 1;
	auto bias_ts = point_ts<ta::fixed_dt>(ta, bias);
	for_each(bias_grid.begin(), bias_grid.end(), [&](auto& b) { b.SetTs(bias_ts); });

	// Tdest(ts) += Tbias(ts)
	for (auto itdest = temp_grid.begin(), itbias = bias_grid.begin(); itdest != temp_grid.end() || itbias != bias_grid.end(); ++itdest, ++itbias)
		(*itdest).pts.add((*itbias).pts);

	TS_ASSERT_EQUALS(count_if(temp_grid.begin(), temp_grid.end(), [=](auto& a) {return TS0_EQUAL(a, temp + bias); }), ngx * ngy);
}

TEST_CASE("test_calc_bias_should_match_observations") {
	// Timeaxis
	calendar utc;
	utctime t0 = utc.time(2000, 1, 1);
	utctimespan dt = deltahours(1);
	const int nt = 24*36;
	ta::fixed_dt ta(t0, dt, nt);

	// Make observation set of 3 sources distributed in a 10 x 10 km grid
	// Temperatures are calculated from regression test
	vector<PointTimeSerieSource> obs_set;
	obs_set.reserve(3);
	obs_set.emplace_back(geo_point(100,  100, 1000), point_ts<ta::fixed_dt>(ta, 14.97));
	obs_set.emplace_back(geo_point(5100, 100, 1150), point_ts<ta::fixed_dt>(ta, 13.12));
	obs_set.emplace_back(geo_point(100, 5100,  850), point_ts<ta::fixed_dt>(ta, 14.92));

	// IDW transform observation from set to grid 10 x 10 km. Call it forecast grid
	const int ng = 10;
	auto fc_grid{PointTimeSerieCell::make_cell_grid(ta, ng, ng)};
	Parameter p;
	run_interpolation<TestTemperatureModel_1>(obs_set.begin(), obs_set.end(), fc_grid.begin(), fc_grid.end(), idw_timeaxis<ta::fixed_dt>(ta),
		p, [](auto& d, size_t ix, double v) { d.set_value(ix, v); });

	// Simulate forecast offset of -2 degC
	auto off_ts = point_ts<ta::fixed_dt>(ta, -2.0);
	for_each(fc_grid.begin(), fc_grid.end(), [&](auto& a) { a.pts.add(off_ts); });

	// IDW transform forecast from frid to set
	vector<PointTimeSerieSource> fc_set = obs_set;
	run_interpolation<TestTemperatureModel_2>(fc_grid.begin(), fc_grid.end(), fc_set.begin(), fc_set.end(), idw_timeaxis<ta::fixed_dt>(ta),
		p, [](auto& d, size_t ix, double v) { d.set_value(ix, v); });

	// Calculate bias set = observation set - forecast set
	vector<PointTimeSerieSource> bias_set = obs_set;
	for (auto it_bias = bias_set.begin(), it_fc = fc_set.begin(); it_bias != bias_set.end() || it_fc != fc_set.end(); ++it_bias, ++it_fc)
		(*it_bias).pts.add_scale((*it_fc).pts, -1);

	// IDW transform bias from set to grid
	auto bias_grid{(PointTimeSerieCell::make_cell_grid(ta, ng, ng))};
	run_interpolation<TestTemperatureModel_1>(bias_set.begin(), bias_set.end(), bias_grid.begin(), bias_grid.end(), idw_timeaxis<ta::fixed_dt>(ta),
		p, [](auto& d, size_t ix, double v) { d.set_value(ix, v); });

	// Add bias grid to forecast grid
	for (auto it_fc = fc_grid.begin(), itbias = bias_grid.begin(); it_fc != fc_grid.end() || itbias != bias_grid.end(); ++it_fc, ++itbias)
		(*it_fc).pts.add((*itbias).pts);

	// IDW transform corrected forecast from grid to set
	run_interpolation<TestTemperatureModel_2>(fc_grid.begin(), fc_grid.end(), fc_set.begin(), fc_set.end(), idw_timeaxis<ta::fixed_dt>(ta),
		p, [](auto& d, size_t ix, double v) { d.set_value(ix, v); });

	// Compare forecast to observation set => differences should be close to null
	for (auto it_obs = obs_set.begin(), it_fc = fc_set.begin(); it_obs != obs_set.end() || it_fc != fc_set.end(); ++it_obs, ++it_fc)
		TS_ASSERT_LESS_THAN(fabs((*it_obs).value(0) - (*it_fc).value(0)), 1e-2);
}
}
