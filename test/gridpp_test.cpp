#include "test_pch.h"
#include "gridpp_test.h"
#include "mocks.h"
#include "core/region_model.h"
#include "api/api.h" // looking for GeoPointSource, and TemperatureSource(realistic case)
#include "api/timeseries.h" // looking for apoint_ts, the api exposed ts-type(realistic case)

using namespace shyft::core;
using namespace shyfttest;
using namespace shyfttest::idw;

namespace shyfttest {
    using namespace shyft::core;
    struct mock_cell {
        mock_cell(geo_point loc=geo_point()):location(loc){}
        geo_point location;
        const geo_point& mid_point() const {return location;}
        pts_t ts;
        void initialize(const timeaxis_t& ta) {
            ts=pts_t(ta,0.0);/// initialize and prepare cell before interpolation step, notice that the lambda to idw uses ts.set(ix,value)
        }
    };
}

void gridpp_test::test_sih_workbench() {
    // from region_model::run_interpolation, we copy some typedefs to setup
    // a realistic IDW run
    // Triple AAA:
    // Arrange-section:
    using namespace shyft::core;
    using namespace std;
    namespace idw = shyft::core::inverse_distance;

    using ats_t=shyft::api::apoint_ts; // potential break-point time-series, irregular intervals
    using temperature_source= shyft::api::TemperatureSource;
    typedef shyft::timeseries::average_accessor<ats_t, timeaxis_t> atsa_t;// accessor to ensure bp. ts is projected to fixed interval ta
    typedef idw_compliant_geo_point_ts< temperature_source, atsa_t, timeaxis_t> idw_compliant_gts_t;// gts =geo located ts , and idw_compliant to!
	typedef idw::temperature_model<idw_compliant_gts_t, mock_cell , idw::temperature_parameter, geo_point, idw::temperature_gradient_scale_computer> idw_temperature_model_t; // how to compensate for the height at different locations using temp.gradient


    // prepare the time-dimension, using time-axis
    calendar utc;
    utctimespan dt=deltahours(1);
    size_t n=24;
    timeaxis_t ta(utc.time(2000,1,1),dt,n); /// for the test,this is the governing time-axis (corresponding to region_model.time_axis during run/interpolation)

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
                TS_ASSERT( value_at_cell > -31.0);// -31..-10 approx.
                //ok, we could compute the expected value more accurate
                // but after all, it's based upon already tested functions
            }
        }

}

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
	// Test with fixed_dt timeaxis in Source and compare performance

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
	auto pts = move(create_time_serie(t0, dt, nt));
	auto cts = move(create_const_time_serie(ta, 1));

	// Tsour = vector<Source(geopoint)>(ts)
	auto Tsour(move(PointTimeSerieSource::GenerateTestSources(nsx, nsy, s0, s0, dss)));
	for_each(Tsour.begin(), Tsour.end(), [&](auto& s) { s.SetTs(cts); });

	// Tdest = vector<Cell(grid)>() => IDW<TemperatureModel>(Tsour, Tdest, fixed_dt)
	auto Tdest(move(PointTimeSerieCell::GenerateTestGrids(ngx, ngy)));
	run_interpolation<TestTemperatureModel_1>(Tsour.begin(), Tsour.end(), Tdest.begin(), Tdest.end(), idw_timeaxis<TimeAxis>(ta),
		p, [](auto& d, size_t ix, double v) {d.set_value(ix, v); });

	// Tbias = vector<MCell(grid)>(bias, fixed_dt)
	auto Tbias(move(PointTimeSerieCell::GenerateTestGrids(ngx, ngy)));
	for_each(Tbias.begin(), Tbias.end(), [&](auto& b) { b.SetTs(cts); });

	// Tdest(ts) += Tbias(ts)
	// for (auto itdest = Tdest.begin(), itbias = Tbias.begin(); itdest != Tdest.end() || itbias != Tbias.end(); ++itdest, ++itbias)
	//	(*itdest).ts += (*itbias).ts;

	TS_ASSERT_EQUALS(count_if(Tdest.begin(), Tdest.end(), [](auto& c) {return c.TsAvg() == 0; }), ngx * ngy);
}
