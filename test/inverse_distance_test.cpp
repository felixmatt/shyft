#include "test_pch.h"
#include "inverse_distance_test.h"
#include "core/inverse_distance.h"
#include "core/utctime_utilities.h"
#include "core/timeseries.h"
#include "core/geo_point.h"

#include <ctime>
#include <cmath>

#ifdef WIN32
#if _MSC_VER < 1800
	const unsigned long nanx[2]={0xffffffff, 0x7fffffff};
	const double NAN=  *( double* )nanx;
#endif
#endif
#include <armadillo>

using namespace shyft::core::inverse_distance;

// Test method implementation
namespace shyfttest_idw {
    using namespace std;
    using namespace shyft::core;
    using namespace shyft::timeseries;
    const double TEST_EPS = 0.00000001;
    typedef shyft::timeseries::timeaxis TimeAxis;

    struct Source {
        Source(geo_point p,double v)
         : point(p), v(v), t_special(0), v_special(v), get_count(0) {}
        geo_point point;

        typedef geo_point geo_point_t;

        double v;
        // for testing failure at special time
        utctime t_special;
        double  v_special;

        mutable int get_count;
        geo_point mid_point() const { return point; }

        double   value(utcperiod p) const { return value(p.start);}
        double   value(utctime t) const {
            get_count++;
            return t == t_special ? v_special : v;
        }


        // for testing
        void set_value(double vx) {v=vx;}
        void value_at_t(utctime tx,double vx) {t_special=tx;v_special=vx;}
        static vector<Source> GenerateTestSources(const TimeAxis& time_axis,size_t n, double x, double y, double radius) {
            vector<Source> r; r.reserve(n);
            const double pi = 3.1415;
            double delta= 2*pi/n;
            for(double angle=0; angle < 2*pi; angle += delta) {
                double xa = x + radius*sin(angle);
                double ya = y + radius*cos(angle);
                double za = (xa + ya)/1000.0;
                r.emplace_back(geo_point(xa, ya, za), 10.0 + za*-0.006);// reasonable temperature, dependent on height
            }
            return move(r);
        }
        static vector<Source> GenerateTestSourceGrid(const TimeAxis& time_axis,size_t nx,size_t ny, double x, double y, double dxy) {
            vector<Source> r; r.reserve(nx*ny);
            const double max_dxy= dxy*(nx+ny);
            for(size_t i=0;i<nx;++i) {
                double xa = x + i*dxy;
                for(size_t j=0; j < ny; ++j) {
                    double ya = y + j*dxy;
                    double za = 1000.0*(xa + ya)/max_dxy;
                    r.emplace_back(geo_point(xa, ya, za), 10.0 + za*-0.006);// reasonable temperature, dependent on height
                }
            }
            return move(r);
        }
    };

    struct MCell
    {
        MCell():point(),v(-1.0),set_count(0),slope(1.0){}
        MCell(geo_point p) : point(p), v(-1.0), set_count(0), slope(1.0) {}
        geo_point point;
        double v;
        int set_count;
        double slope;
        geo_point mid_point() const { return point; }
        double slope_factor() const {return 1.0; }
        void set_slope_factor(double x) { slope = x; }
        void set_value(size_t t, double vt)
        {
            set_count++;
            v = vt;
        }
        static vector<MCell> GenerateTestGrid(size_t nx,size_t ny)
        {
            vector<MCell> r; r.reserve(nx*ny);
            const double z_min=100.0, z_max=800.0, dz=(z_max-z_min)/(nx+ny);
            for(size_t x=0; x < nx; ++x)
                for(size_t y=0; y < ny; ++y)
                    r.emplace_back(geo_point(500.0 + x*1000, 500.0 + y*1000, z_min + (x + y)*dz));
            return move(r);
        }
    };

    struct Parameter
    {
        Parameter(double max_distance, size_t max_number_of_neigbours)
            : max_distance(max_distance), max_members(max_number_of_neigbours) {}
        double max_distance;
        size_t max_members;
        bool gradient_by_equation=false;// just use min/max for existing tests(bw compatible)
        double default_gradient() const {return  -0.006;}  // C/m decrease 0.6 degC/100m
        double precipitation_scale_factor() const { return 1.0 + 2.0/100.0; } // 2 pct /100m
        double distance_measure_factor=2.0; // Square distance
    };

    typedef temperature_model  <Source, MCell, Parameter, geo_point,temperature_gradient_scale_computer> TestTemperatureModel;
    typedef radiation_model    <Source, MCell, Parameter, geo_point> TestRadiationModel;
    typedef precipitation_model<Source, MCell, Parameter, geo_point> TestPrecipitationModel;

  };

    using namespace shyfttest_idw;
    using namespace std;

void inverse_distance_test::test_temperature_model() {
    //
    // Verify temperature gradient calculator, needs to be robust ...
    //
    Parameter p(100*1000.0, 10);
    TestTemperatureModel::scale_computer gc(p);
    TS_ASSERT_DELTA(gc.compute(), p.default_gradient(), TEST_EPS); // should give default gradient by default

    geo_point p1(1000,1000,100);
    Source   s1(p1,10);
    utctime  t0=3600L*24L*365L*44L;

    gc.add(s1,t0);
    TS_ASSERT_DELTA(gc.compute(), p.default_gradient(), TEST_EPS); // should give default gradient if just one point

    geo_point p1b(1000,1000,149);
    Source   s1b(p1b,10-0.005*59);
    gc.add(s1b,t0);

    TS_ASSERT_DELTA(gc.compute(),p.default_gradient(),TEST_EPS); // mi-max z-distance less than 50 m., return default.


    geo_point p2(2000,2000,200);
    Source   s2(p2,9.5);
    gc.add(s2,t0);
    TS_ASSERT_DELTA(gc.compute(), -0.005, TEST_EPS); // should give -0.005 gradient if just two points

    geo_point p3(3000,3000,300);
    Source   s3(p3,9.0);
    gc.add(s3,t0);
    TS_ASSERT_DELTA(gc.compute(), -0.005, TEST_EPS); // should give -0.005 gradient for these 3 points

    geo_point p4(4000,4000,500);
    Source   s4(p4,8.0);
    gc.add(s4,t0);
    TS_ASSERT_DELTA(gc.compute(), -0.005, TEST_EPS);// should give -0.005 gradient for these 4 points

    geo_point p5(4000,4000,600);
    Source   s5(p5,10-0.006*(600-100));
    gc.add(s5,t0);
    TS_ASSERT_DELTA(gc.compute(), -0.006,TEST_EPS); // mi-max alg. should only consider high/low ..


    //
    // Verify the TestTemperatureModel::transform, should do temp.gradient computation based on height difference.
    //
    MCell d1(geo_point(1500,1500,200));
    double sourceValue=10.0;
    double scaleValue=-0.005;

    double transformedValue=TestTemperatureModel::transform(sourceValue, scaleValue, s1, d1);

    TS_ASSERT_DELTA(transformedValue, sourceValue + scaleValue*(d1.point.z - s1.point.z),TEST_EPS);

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
    Parameter p(100*1000.0,10);
    TestRadiationModel::scale_computer gc(p);
    TS_ASSERT_DELTA(gc.compute(), 1.0, TEST_EPS);// should give 1.0 gradient by default

    geo_point p1(1000,1000,100);
    Source   s1(p1,10);
    utctime  t0=3600L*24L*365L*44L;

    gc.add(s1,t0);
    TS_ASSERT_DELTA(gc.compute(), 1.0, TEST_EPS);// should give 1.0 gradient if just one point

    geo_point p2(2000,2000,200);
    Source   s2(p2,9.5);
    gc.add(s2,t0);
    TS_ASSERT_DELTA(gc.compute(), 1.0, TEST_EPS);// should give -0.005 gradient if just two points


    //
    // Verify the TestTemperatureModel::transform, should do temp.gradient computation based on height difference.
    //
    MCell d1(geo_point(1500,1500,200));
    d1.set_slope_factor(0.5);
    double sourceValue=10.0;
    double scaleValue=1.0;

    double transformedValue=TestRadiationModel::transform(sourceValue,scaleValue,s1,d1);

    TS_ASSERT_DELTA(transformedValue, sourceValue*d1.slope_factor(), TEST_EPS);
}

void inverse_distance_test::test_precipitation_model(){
    //
    // Verify temperature gradient calculator, needs to be robust ...
    //
    Parameter p(100*1000.0,10);
    TestPrecipitationModel::scale_computer gc(p);
    TS_ASSERT_DELTA(gc.compute(), p.precipitation_scale_factor(), TEST_EPS);// should give 1.0 gradient by default

    geo_point p1(1000,1000,100);
    Source   s1(p1,10);
    utctime  t0=3600L*24L*365L*44L;

    gc.add(s1,t0);
    TS_ASSERT_DELTA(gc.compute(), p.precipitation_scale_factor(), TEST_EPS);// should give 1.0 gradient if just one point

    geo_point p2(2000,2000,200);
    Source   s2(p2,9.5);
    gc.add(s2,t0);
    TS_ASSERT_DELTA(gc.compute(), p.precipitation_scale_factor(), TEST_EPS);// should give -0.005 gradient if just two points


    //
    // Verify the TestTemperatureModel::transform, should do temp.gradient computation based on height difference.
    //
    MCell d1(geo_point(1500,1500,200));
    double sourceValue=10.0;
    double scaleValue=gc.compute();

    double transformedValue=TestPrecipitationModel::transform(sourceValue,scaleValue,s1,d1);

    TS_ASSERT_DELTA(transformedValue, sourceValue*pow(scaleValue, (d1.point.z - s1.point.z)/100.0), TEST_EPS);

	//
	// Verify that 0.0 transformation to zero
	//
	//Destination d0(GeoPoint(1500,1500,0))
	//TS_ASSERT_DELTA(0.0, TestPrecipitationModel::transform(0.0, scaleValue, s1, d0), TEST_EPS);


}

void inverse_distance_test::test_one_source_one_dest_calculation() {
    //
    // Arrange
    //
    utctime Tstart=3600L*24L*365L*44L;
    utctimespan dt=3600L;
    int n=1;//24*10;
    const int nx=1;
    const int ny=1;
    const int n_sources=1;
    TimeAxis ta(Tstart,dt,n);//hour, 10 steps
    vector<Source> s(Source::GenerateTestSources(ta,n_sources,0.5*nx*1000,0.5*ny*1000,0.25*0.5*(nx+ny)*1000));// 40 sources, radius 50km, starting at 100,100 km center
    vector<MCell> d(MCell::GenerateTestGrid(nx,ny));// 200x200 km
    Parameter p(2.75*0.5*(nx+ny)*1000,max(8,1+n_sources/2));

    //
    // Act
    //

    run_interpolation<TestTemperatureModel>(begin(s),end(s),begin(d),end(d),idw_timeaxis<TimeAxis>(ta),p,
                                                       [](MCell&d ,size_t ix,double v) {d.set_value(ix,v);});

    //
    // Assert
    //
    TS_ASSERT_EQUALS(count_if(begin(d), end(d), [n](const MCell &d) { return d.set_count == n ;}), nx*ny);

    double expected_v= TestTemperatureModel::transform(s[0].value(utcperiod(Tstart,Tstart+dt)), p.default_gradient(),s[0],d[0]);
    TS_ASSERT_EQUALS(count_if(begin(d), end(d), [n,expected_v](const MCell&d) { return fabs(d.v - expected_v) < 1e-7;}), nx*ny);


}

void inverse_distance_test::test_two_sources_one_dest_calculation() {
    //
    // Arrange
    //
    utctime Tstart=3600L*24L*365L*44L;
    utctimespan dt=3600L;
    int n=1;
    const int nx=1;
    const int ny=1;
    const int n_sources=2;
    TimeAxis ta(Tstart,dt,n);
    vector<Source> s(Source::GenerateTestSources(ta,n_sources,0.5*nx*1000,0.5*ny*1000,0.25*0.5*(nx+ny)*1000));// n sources, radius 50km, starting at 100,100 km center
    vector<MCell> d(MCell::GenerateTestGrid(nx,ny));// 200x200 km
    Parameter p(2.75*0.5*(nx+ny)*1000,1+n_sources/2);

    //
    // Act
    //

	run_interpolation<TestTemperatureModel>(begin(s), end(s), begin(d), end(d), idw_timeaxis<TimeAxis>(ta), p,
                                                       [](MCell&d ,size_t ix,double v) {d.set_value(ix,v);});

    //
    // Assert
    //
    TS_ASSERT_EQUALS(count_if(begin(d),end(d),[n](const MCell&d) { return d.set_count==n ;}), nx*ny);

    TestTemperatureModel::scale_computer gc(p);
    gc.add(s[0],Tstart); gc.add(s[1],Tstart);
    double comp_gradient=gc.compute();
    double w1= 1.0/TestTemperatureModel::distance_measure(s[0].mid_point(), d[0].mid_point(), 2.0);
    double w2= 1.0/TestTemperatureModel::distance_measure(s[1].mid_point(), d[0].mid_point(), 2.0);
    double v1= w1*TestTemperatureModel::transform(s[0].value(Tstart), comp_gradient,s[0],d[0]);
    double v2= w2*TestTemperatureModel::transform(s[1].value(Tstart), comp_gradient,s[1],d[0]);
    double expected_v= (v1+v2)/(w1+w2);
    TS_ASSERT_EQUALS(count_if(begin(d),end(d),[n,expected_v](const MCell&d) { return fabs(d.v-expected_v) <1e-7;}), nx*ny);
}

void inverse_distance_test::test_using_finite_sources_only() {
    //
    // Arrange
    //
    utctime Tstart=3600L*24L*365L*44L;
    utctimespan dt=3600L;
    int n=1;
    const int nx=1;
    const int ny=1;
    const int n_sources=3;
    TimeAxis ta(Tstart,dt,n);//hour, 10 steps
    vector<Source> s(Source::GenerateTestSources(ta,n_sources,0.5*nx*1000,0.5*ny*1000,0.25*0.5*(nx+ny)*1000));// n sources, radius 50km, starting at 100,100 km center
    vector<MCell> d(MCell::GenerateTestGrid(nx,ny));// 200x200 km
    Parameter p(2.75*0.5*(nx+ny)*1000,n_sources);

    s[2].set_value(NAN);
    //
    // Act
    //

	run_interpolation<TestTemperatureModel>(begin(s), end(s), begin(d), end(d), idw_timeaxis<TimeAxis>(ta), p,
                                                       [](MCell&d ,size_t ix,double v) {d.set_value(ix,v);});

    //
    // Assert
    //
    TS_ASSERT_EQUALS(count_if(begin(d),end(d),[n](const MCell&d) { return d.set_count==n ;}), nx*ny);

    TestTemperatureModel::scale_computer gc(p);
    gc.add(s[0],Tstart); gc.add(s[1],Tstart);
    double comp_gradient=gc.compute();
    double w1 = 1.0/TestTemperatureModel::distance_measure(s[0].mid_point(), d[0].mid_point(), 2.0);
    double w2 = 1.0/TestTemperatureModel::distance_measure(s[1].mid_point(), d[0].mid_point(), 2.0);
    double v1 = w1*TestTemperatureModel::transform(s[0].value(Tstart), comp_gradient,s[0],d[0]);
    double v2 = w2*TestTemperatureModel::transform(s[1].value(Tstart), comp_gradient,s[1],d[0]);
    double expected_v= (v1+v2)/(w1+w2);
    TS_ASSERT_EQUALS(count_if(begin(d),end(d),[n,expected_v](const MCell&d) { return fabs(d.v-expected_v) <1e-7;}), nx*ny);
}

void inverse_distance_test::test_eliminate_far_away_sources() {
    //
    // Arrange
    //
    utctime Tstart=3600L*24L*365L*44L;
    utctimespan dt=3600L;
    int n=1;
    const int nx=1;
    const int ny=1;
    const int n_sources=3;
    TimeAxis ta(Tstart,dt,n);
    vector<Source> s(Source::GenerateTestSources(ta,n_sources,0.5*nx*1000,0.5*ny*1000,0.25*0.5*(nx+ny)*1000));// n sources, radius 50km, starting at 100,100 km center
    vector<MCell> d(MCell::GenerateTestGrid(nx,ny));// 200x200 km
    Parameter p(2.75*0.5*(nx+ny)*1000,n_sources);
    s[2].point=geo_point(p.max_distance+1000,p.max_distance+1000,300);// place a point far away to ensure it's not part of interpolation
    //
    // Act
    //

	run_interpolation<TestTemperatureModel>(begin(s), end(s), begin(d), end(d), idw_timeaxis<TimeAxis>(ta), p,
                                                       [](MCell&d ,size_t ix,double v) {d.set_value(ix,v);});

    //
    // Assert
    //
    TS_ASSERT_EQUALS(count_if(begin(d),end(d),[n](const MCell&d) { return d.set_count==n ;}), nx*ny);

    TestTemperatureModel::scale_computer gc(p);
    gc.add(s[0],Tstart); gc.add(s[1],Tstart);
    double comp_gradient=gc.compute();
    double w1 = 1.0/TestTemperatureModel::distance_measure(s[0].mid_point(),d[0].mid_point(), 2.0);
    double w2 = 1.0/TestTemperatureModel::distance_measure(s[1].mid_point(),d[0].mid_point(), 2.0);
    double v1 = w1*TestTemperatureModel::transform(s[0].value(Tstart), comp_gradient,s[0],d[0]);
    double v2 = w2*TestTemperatureModel::transform(s[1].value(Tstart), comp_gradient,s[1],d[0]);
    double expected_v= (v1+v2)/(w1+w2);
    TS_ASSERT_EQUALS(count_if(begin(d),end(d),[n,expected_v](const MCell&d) { return fabs(d.v-expected_v) <1e-7;}), nx*ny);
}

void inverse_distance_test::test_using_up_to_max_sources() {
    //
    // Arrange
    //
    utctime Tstart=3600L*24L*365L*44L;
    utctimespan dt=3600L;
    int n=1;
    const int nx=1;
    const int ny=1;
    const int n_sources=3;
    TimeAxis ta(Tstart,dt,n);//hour, 10 steps
    vector<Source> s(Source::GenerateTestSources(ta,n_sources,0.5*nx*1000,0.5*ny*1000,0.25*0.5*(nx+ny)*1000));// n sources, radius 50km, starting at 100,100 km center
    vector<MCell> d(MCell::GenerateTestGrid(nx,ny));// 200x200 km
    Parameter p(2.75*0.5*(nx+ny)*1000,n_sources-1);
    //
    // Act
    //

    run_interpolation<TestTemperatureModel>(begin(s),end(s),begin(d),end(d),idw_timeaxis<TimeAxis>(ta),p,
                                                       [](MCell&d ,size_t ix,double v) {d.set_value(ix,v);});

    //
    // Assert
    //
    TS_ASSERT_EQUALS(count_if(begin(d),end(d),[n](const MCell&d) { return d.set_count==n ;}), nx*ny);

    TestTemperatureModel::scale_computer gc(p);
    gc.add(s[0],Tstart); gc.add(s[1],Tstart);
    double comp_gradient=gc.compute();
    double w1 = 1.0/TestTemperatureModel::distance_measure(s[0].mid_point(),d[0].mid_point(), 2.0);
    double w2 = 1.0/TestTemperatureModel::distance_measure(s[1].mid_point(),d[0].mid_point(), 2.0);
    double v1 = w1*TestTemperatureModel::transform(s[0].value(Tstart), comp_gradient,s[0],d[0]);
    double v2 = w2*TestTemperatureModel::transform(s[1].value(Tstart), comp_gradient,s[1],d[0]);
    double expected_v= (v1+v2)/(w1+w2);
    TS_ASSERT_EQUALS(count_if(begin(d),end(d),[n,expected_v](const MCell&d) { return fabs(d.v-expected_v) <1e-7;}), nx*ny);
}

void inverse_distance_test::test_handling_different_sources_pr_timesteps() {
    //
    // Arrange
    //
    utctime Tstart=3600L*24L*365L*44L;
    utctimespan dt=3600L;
    int n=2;//24*10;
    const int nx=1;
    const int ny=1;
    const int n_sources=3;
    TimeAxis ta(Tstart,dt,n);
    vector<Source> s(Source::GenerateTestSources(ta,n_sources,0.5*nx*1000,0.5*ny*1000,0.25*0.5*(nx+ny)*1000));// n sources, radius 50km, starting at 100,100 km center
    vector<MCell> d(MCell::GenerateTestGrid(nx,ny));// 200x200 km
    Parameter p(2.75*0.5*(nx+ny)*1000,n_sources);
    s[2].value_at_t(Tstart+dt,NAN); // at second, and last.. , timestep, only s[0] and s[1] are valid, so diff. calc. applies to second step, the other tests verifies that it can calculate ok first n steps.
    //
    // Act
    //

    run_interpolation<TestTemperatureModel>(begin(s),end(s),begin(d),end(d),idw_timeaxis<TimeAxis>(ta),p,
                                                       [](MCell&d ,size_t ix,double v) {d.set_value(ix,v);});

    //
    // Assert
    //
    TS_ASSERT_EQUALS(count_if(begin(d),end(d),[n](const MCell&d) { return d.set_count==n ;}), nx*ny);

    TestTemperatureModel::scale_computer gc(p);
    gc.add(s[0],Tstart); gc.add(s[1],Tstart);
    double comp_gradient=gc.compute();
    double w1 = 1.0/TestTemperatureModel::distance_measure(s[0].mid_point(),d[0].mid_point(), 2.0);
    double w2 = 1.0/TestTemperatureModel::distance_measure(s[1].mid_point(),d[0].mid_point(), 2.0);
    double v1 = w1*TestTemperatureModel::transform(s[0].value(Tstart), comp_gradient,s[0],d[0]);
    double v2 = w2*TestTemperatureModel::transform(s[1].value(Tstart), comp_gradient,s[1],d[0]);
    double expected_v= (v1+v2)/(w1+w2);
    TS_ASSERT_EQUALS(count_if(begin(d),end(d),[n,expected_v](const MCell&d) { return fabs(d.v-expected_v) <1e-7;}), nx*ny);
}

void inverse_distance_test::test_performance() {
    //
    // Arrange
    //
    utctime Tstart = calendar().time(YMDhms(2000, 1, 1));
    utctimespan dt = 3600L;
    int n = 24 * 36; // number of timesteps
    const int n_xy = 3; // number for xy-squares for sources
    const int nx = 3 * n_xy; // 3 times more for grid-cells, typical arome -> cell
    const int ny = 3 * n_xy;
    const int s_nx = n_xy;
    const int s_ny = n_xy;
    const int n_sources = s_nx * s_ny;
    double s_dxy = 3 * 1000; // arome typical 3 km.
    TimeAxis ta(Tstart, dt, n); // hour, 10 steps
    vector<Source> s(move(Source::GenerateTestSourceGrid(ta, s_nx, s_ny, -0.5*1000, -0.5*1000, s_dxy)));
    vector<MCell> d(move(MCell::GenerateTestGrid(nx, ny)));
    Parameter p(s_dxy*2, min(8, n_sources/2)); // for practical purposes, 8 neighbours or less.

    //
    // Act
    //
    const clock_t start = clock();

    run_interpolation<TestTemperatureModel>(begin(s), end(s), begin(d), end(d), idw_timeaxis<TimeAxis>(ta), p,
                                                       [](MCell& d, size_t ix, double v) {d.set_value(ix, v);} );
    const clock_t total = clock() - start;
    cout << "\nAlg. IDW2i:\n";
	double time = 1000 * double(total) / double(CLOCKS_PER_SEC);
    cout << "Temperature interpolation took: " << time << " ms" << endl;
    cout << "Each temperature timestep took: " << time / double(n) << " ms" << endl;

    //
    // Assert
    //
    TS_ASSERT_EQUALS(count_if( begin(d), end(d), [n](const MCell& d) {return d.set_count == n;} ), nx*ny);
    TS_ASSERT_EQUALS(count_if( begin(d), end(d), [n](const MCell& d) {return d.v >= 0.0;} ), nx*ny);
}

static inline
arma::vec3 p_vec(geo_point a, geo_point b) {
        return arma::vec3({b.x-a.x,b.y-a.y,b.z-a.z});
}

void inverse_distance_test::test_temperature_gradient_model() {
    using namespace arma;
    geo_point   p0(   0,    0,  10),// 10 deg.
                p1(1000,    0, 110),// 9.4 deg.
                p2(   0, 1000, 110),
                p3(1000, 1000, 220),
                px( 500,  500,  50);
    vec3 dTv({0.001 , 0.002, +0.1/100}); // temp. gradient in x, y, z direction
    auto dT=dTv.t();
    double t0=10.0;
    auto p01=p_vec(p0,p1);
    auto p02=p_vec(p0,p2);
    auto p03=p_vec(p0,p3);
    mat33  P(temperature_gradient_scale_computer::p_mat(p0,p1,p2,p3));
    auto t1= t0 + as_scalar(dT*p01);
    auto t2= t0 + as_scalar(dT*p02);
    auto t3= t0 + as_scalar(dT*p03);
    temperature_parameter p(-0.0065,5,5*1000,true);// turn on using equations to solve gradient
    temperature_gradient_scale_computer sc(p);
    vector<Source> s;
    s.emplace_back(p0,t0);
    s.emplace_back(p1,t1);
    s.emplace_back(p2,t2);
    s.emplace_back(p3,t3);
    utctime tx=calendar().time(YMDhms(2000,1,1));
    sc.add(s[0],tx);
    TS_ASSERT_DELTA(sc.compute(),p.default_gradient(),0.000001);// with one point, default should be returned
    sc.add(s[1],tx);
    TS_ASSERT_DELTA(sc.compute(),(t1-t0)/(p1.z-p0.z),0.000001);// with more than one, use min/max method
    sc.add(s[2],tx);
    TS_ASSERT_DELTA(sc.compute(),(t1-t0)/(p1.z-p0.z),0.000001);// still min/max method
    sc.add(s[2],tx);// add redundant point, gives singularity, so we should fallback to min-max method.
    TS_ASSERT_DELTA(sc.compute(),(t1-t0)/(p1.z-p0.z),0.000001);// still min/max method
    sc.clear();//forget all points
    for(size_t i=0;i<s.size();++i)
        sc.add(s[i],tx);//fill up with distinct points
    TS_ASSERT_DELTA(sc.compute(),as_scalar(dTv(2)),0.00001);// now we should get the correct linear vertical


}
/* vim: set filetype=cpp: */
