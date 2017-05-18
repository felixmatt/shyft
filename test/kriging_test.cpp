#include "test_pch.h"
#include "core/kriging.h"
#include "core/time_series.h"
#include "core/geo_point.h"


namespace shyfttest {
    using namespace shyft::core;
    /** just a stub class for location concept*/
    struct location{
        geo_point p;
        double v;
        geo_point mid_point() const { return p; }
        double value(utctime t) const {
            return v ;
        }
    };

    /** just to generate test-pattern we can verify in the matrices */
    struct cov_test {
        double operator()( const location& a, const location& b) const {
            return (a.p.x) * 10 + (b.p.x);
        }
    };
}
using namespace shyft::core;
using namespace std;
using namespace shyfttest;
using int_t=arma::uword;
TEST_SUITE("kriging") {
TEST_CASE("test_covariance_calculation") {
    kriging::covariance::exponential e(1.0,1000.0);
    kriging::covariance::gaussian g(1.0,1000.0);
    TS_ASSERT_DELTA(e(0.0), 1.0,0.00001);
    TS_ASSERT_DELTA(e(1000.0), exp(1000.0*-3.0/1000.0),0.0001);
    TS_ASSERT_DELTA(g(0.0), 1.0,0.00001);
    TS_ASSERT_DELTA(g(50*50.0), exp(50.0*50.0*-3.0/(1000.0*1000.0)),0.0001);

}

TEST_CASE("test_build_covariance_obs_to_obs_matrix") {

    /* verify covariance::build observation vs observation produces the correct matrix */
    vector<location> s;
    s.push_back(location{ geo_point(1,   0,  0),10.0 });
    s.push_back(location{ geo_point(2, 100, 10),10.0 });
    s.push_back(location{ geo_point(3, 100, 20),10.0 });

    auto c = kriging::ordinary::build(begin(s), end(s), cov_test());
    TS_ASSERT_DELTA(c(0, 0), 11.0, 0.001);
    TS_ASSERT_DELTA(c(1, 1), 22.0, 0.001);
    TS_ASSERT_DELTA(c(2, 2), 33.0, 0.001);
    TS_ASSERT_DELTA(c(0, 1), 12.0, 0.001);
    TS_ASSERT_DELTA(c(0, 2), 13.0, 0.001);
    TS_ASSERT_DELTA(c(1, 0), 12.0, 0.001);
    TS_ASSERT_DELTA(c(2, 0), 13.0, 0.001);
    TS_ASSERT_DELTA(c(1, 2), 23.0, 0.001);
    TS_ASSERT_DELTA(c(2, 1), 23.0, 0.001);
    //- check that weight sum = 1.0 requirement is in place
    TS_ASSERT_DELTA(c(0,3), 1.0,0.001);
    TS_ASSERT_DELTA(c(1,3), 1.0,0.001);
    TS_ASSERT_DELTA(c(2,3), 1.0,0.001);
    TS_ASSERT_DELTA(c(3,3), 0.0,0.001);
    TS_ASSERT_DELTA(c(3, 0), 1.0, 0.001);
    TS_ASSERT_DELTA(c(3, 1), 1.0, 0.001);
    TS_ASSERT_DELTA(c(3, 2), 1.0, 0.001);

    //std::cout << "\n" << c << endl;
    //std::cout << "\nc.inv()\n" << c.i() << endl;
}
TEST_CASE("test_build_covariance_obs_to_grid_matrix") {

    vector<location> s;
    s.push_back(location{ geo_point(1,   0,  0),10.0 });
    s.push_back(location{ geo_point(2, 100, 10),10.0 });
    s.push_back(location{ geo_point(3, 100, 20),10.0 });

    vector<location> d;
    for (double x = 1;x < 10.0;x += 1.0) d.push_back(location{ geo_point(x,x,x),10.0 });

    auto c = kriging::ordinary::build(begin(s), end(s),begin(d),end(d), cov_test());
    TS_ASSERT_EQUALS(c.n_rows, int_t(s.size()+1));
    TS_ASSERT_EQUALS(c.n_cols, int_t(d.size()));
    cov_test fx;
    for (arma::uword i = 0;i < c.n_rows;++i)
        for (arma::uword j = 0;j < c.n_cols;++j)
            TS_ASSERT_DELTA(c(i, j), i<c.n_rows-1?fx(s[i],d[j]):1.0, 0.001);
    //std::cout << "\n" << c << endl;
}

TEST_CASE("test_interpolation") {
        /* numbers and coordinates are taken from met.no/gridpp  10x10 netcdf file */
        vector<double> x = { -487442.2 ,-484942.2,	 -482442.2,	-479942.2,	-477442.2,	-474942.2,	-472442.2,	-469942.2,	-467442.2,	-464942.2 };
        vector<double> y = { -269321.8,-266821.8,-264321.8,-261821.8,-259321.8,-256821.8,-254321.8,-251821.8,-249321.8,-246821.8 };
        vector<location> grid_10x10 = {
            location{ geo_point(x[0],y[0],160.9),0.0},location{geo_point(x[1],y[0],11.7) ,0.0},location{geo_point(x[2],y[0],167.9),0.0},location{geo_point(x[3],y[0],297.4),0.0},location{geo_point(x[4],y[0],416.1) ,0.0},location{geo_point(x[5],y[0],556.6) ,0.0},location{geo_point(x[6],y[0],818.7) ,0.0},location{geo_point(x[7],y[0],1065.2),0.0},location{geo_point(x[8],y[0],1158.1),0.0},location{geo_point(x[9],y[0],1059.0),0.0},
            location{ geo_point(x[0],y[1],295.4),0.0},location{geo_point(x[1],y[1],0.0)  ,0.0},location{geo_point(x[2],y[1],0.0)  ,0.0},location{geo_point(x[3],y[1],47.3) ,0.0},location{geo_point(x[4],y[1],255.3) ,0.0},location{geo_point(x[5],y[1],425.8) ,0.0},location{geo_point(x[6],y[1],610.3) ,0.0},location{geo_point(x[7],y[1],837.8) ,0.0},location{geo_point(x[8],y[1],960.3) ,0.0},location{geo_point(x[9],y[1],1046.3),0.0},
            location{ geo_point(x[0],y[2],418.0),0.0},location{geo_point(x[1],y[2],94.2) ,0.0},location{geo_point(x[2],y[2],0.0)  ,0.0},location{geo_point(x[3],y[2],0.0)  ,0.0},location{geo_point(x[4],y[2],43.3)  ,0.0},location{geo_point(x[5],y[2],209.7) ,0.0},location{geo_point(x[6],y[2],410.1) ,0.0},location{geo_point(x[7],y[2],707.4) ,0.0},location{geo_point(x[8],y[2],908.7) ,0.0},location{geo_point(x[9],y[2],1036.7),0.0},
            location{ geo_point(x[0],y[3],466.3),0.0},location{geo_point(x[1],y[3],202.2),0.0},location{geo_point(x[2],y[3],16.9) ,0.0},location{geo_point(x[3],y[3],0.0)  ,0.0},location{geo_point(x[4],y[3],0.0)   ,0.0},location{geo_point(x[5],y[3],150.0) ,0.0},location{geo_point(x[6],y[3],257.0) ,0.0},location{geo_point(x[7],y[3],434.1) ,0.0},location{geo_point(x[8],y[3],644.9) ,0.0},location{geo_point(x[9],y[3],770.5),0.0},
            location{ geo_point(x[0],y[4],277.8),0.0},location{geo_point(x[1],y[4],92.8) ,0.0},location{geo_point(x[2],y[4],5.2)  ,0.0},location{geo_point(x[3],y[4],33.8) ,0.0},location{geo_point(x[4],y[4],0.0)   ,0.0},location{geo_point(x[5],y[4],45.0)  ,0.0},location{geo_point(x[6],y[4],0.0)   ,0.0},location{geo_point(x[7],y[4],0.0)   ,0.0},location{geo_point(x[8],y[4],88.4)  ,0.0},location{geo_point(x[9],y[4],591.4),0.0},
            location{ geo_point(x[0],y[5],459.4),0.0},location{geo_point(x[1],y[5],280.4),0.0},location{geo_point(x[2],y[5],159.6),0.0},location{geo_point(x[3],y[5],166.8),0.0},location{geo_point(x[4],y[5],202.5) ,0.0},location{geo_point(x[5],y[5],142.1) ,0.0},location{geo_point(x[6],y[5],2.4)   ,0.0},location{geo_point(x[7],y[5],0.0)   ,0.0},location{geo_point(x[8],y[5],0.0)   ,0.0},location{geo_point(x[9],y[5],346.9),0.0},
            location{ geo_point(x[0],y[6],744.0),0.0},location{geo_point(x[1],y[6],487.3),0.0},location{geo_point(x[2],y[6],400.4),0.0},location{geo_point(x[3],y[6],339.9),0.0},location{geo_point(x[4],y[6],623.7) ,0.0},location{geo_point(x[5],y[6],710.8) ,0.0},location{geo_point(x[6],y[6],445.0) ,0.0},location{geo_point(x[7],y[6],114.3) ,0.0},location{geo_point(x[8],y[6],26.8)  ,0.0},location{geo_point(x[9],y[6],2.3),0.0},
            location{ geo_point(x[0],y[7],815.0),0.0},location{geo_point(x[1],y[7],656.8),0.0},location{geo_point(x[2],y[7],641.4),0.0},location{geo_point(x[3],y[7],489.7),0.0},location{geo_point(x[4],y[7],821.3) ,0.0},location{geo_point(x[5],y[7],929.2) ,0.0},location{geo_point(x[6],y[7],854.2) ,0.0},location{geo_point(x[7],y[7],811.2) ,0.0},location{geo_point(x[8],y[7],695.1) ,0.0},location{geo_point(x[9],y[7],223.2),0.0},
            location{ geo_point(x[0],y[8],936.9),0.0},location{geo_point(x[1],y[8],914.8),0.0},location{geo_point(x[2],y[8],748.9),0.0},location{geo_point(x[3],y[8],594.4),0.0},location{geo_point(x[4],y[8],846.6) ,0.0},location{geo_point(x[5],y[8],974.0) ,0.0},location{geo_point(x[6],y[8],961.5) ,0.0},location{geo_point(x[7],y[8],959.5) ,0.0},location{geo_point(x[8],y[8],898.3) ,0.0},location{geo_point(x[9],y[8],866.9),0.0},
            location{ geo_point(x[0],y[9],901.7),0.0},location{geo_point(x[1],y[9],899.5),0.0},location{geo_point(x[2],y[9],735.9),0.0},location{geo_point(x[3],y[9],783.0),0.0},location{geo_point(x[4],y[9],1004.7),0.0},location{geo_point(x[5],y[9],1085.4),0.0},location{geo_point(x[6],y[9],1053.8),0.0},location{geo_point(x[7],y[9],986.6) ,0.0},location{geo_point(x[8],y[9],789.4) ,0.0},location{geo_point(x[9],y[9],758.4),0.0}
        };

        // the case is realistic, let's say it's precipitation correction factors, known at some few locations, range 0.7..1.2
        vector<location> obs = {
            location {geo_point(x[0]-1000,y[0]+1000,10.0),0.7},
            location {geo_point(x[5],y[5],140.0),+1.2},
            location {geo_point(x[9],y[9],800.0),1.0},
            location {geo_point(x[9],y[8],850.0), 0.7}
        };
        kriging::covariance::exponential ex(1.0,3*2.5*1000.0); // give 3x2.5 km range where covariance blends out

        auto fx_cov = [&ex](const location&o1, const location& o2)->double { // notice how we compose the covariance fx
            return ex(sqrt(geo_point::distance2(o1.p,o2.p)));
        };
        //-setup kriging equations and matrices
        auto A = kriging::ordinary::build(begin(obs), end(obs),fx_cov);
        auto B = kriging::ordinary::build(begin(obs), end(obs), begin(grid_10x10), end(grid_10x10), fx_cov);

        // the ordinary kriging system is now
        //   Ax =b, b is a column-vector in our B matrix, so effectively we have 10x10 grid points
        // we solve this by inverting A, then we can multiply with any column b in B matrix to get the x weights
        //
        auto X = (A.i()*B).eval();// .eval forces expression to be evaluated and flattened
        auto weights = X.head_rows(obs.size()); // the last row is for the sum of weights=1.0 requirement, we dont use that for now

        //- now use the weights to compute grid-values given observation values.
        arma::mat obs_values(1, obs.size(), arma::fill::none);
        for (int_t c = 0;c < obs.size();c++)
            obs_values(0, c) = obs[c].value(0);
        auto grid_values = (obs_values * weights).eval();
        grid_values.reshape(10, 10);
        // verify specific places:
        TS_ASSERT_DELTA(grid_values(0,0),0.7,0.1);
        TS_ASSERT_DELTA(grid_values(5,5),1.2,0.1);
        TS_ASSERT_DELTA(grid_values(9,9),1.0,0.1);
        TS_ASSERT_DELTA(grid_values(9,8),0.7,0.1);
        // verify overall range
        for(int_t i=0;i<10;++i) {
            for(int_t j=0;j<10;++j) {
                TS_ASSERT_LESS_THAN(0.7,grid_values(i,j));
                TS_ASSERT_LESS_THAN(grid_values(i, j), 1.2);
            }
        }
        // for debug/validation
        //cout<<"\ngrid values (should be between 0.6..1.2)\n"<<grid_values<<endl;
}
}
