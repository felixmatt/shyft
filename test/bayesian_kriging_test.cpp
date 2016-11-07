#include "test_pch.h"
#include "bayesian_kriging_test.h"
#include "mocks.h"
#include "core/inverse_distance.h"
#include "core/bayesian_kriging.h"
#include "core/timeseries.h"
#include "core/geo_point.h"

#include <armadillo>
#include <ctime>
#include <random>
#include <iomanip>

using namespace std;
namespace shyfttest {
	const double EPS = 1.0e-8;
	using namespace shyft::core;
	using namespace shyft::timeseries;
	using shyft::core::geo_point;

	namespace btk_structs {

		//! Simple source class for testing Bayesian Temperature Kriging.
		class Source {
		public:
			typedef xpts_t ts_source;
			typedef shyft::timeseries::average_accessor<ts_source, shyfttest::point_timeaxis> source_accessor;
		private:
			geo_point coord;
			const ts_source temperature_ts;
		public:
			Source(const geo_point& coord, const ts_source& temperatures)
				: coord(coord), temperature_ts(std::move(temperatures)) {
				// Do nothing
			}
			const geo_point& mid_point() const { return coord; }
			const ts_source& temperatures() const { return temperature_ts; }
			source_accessor temperature_accessor(const shyfttest::point_timeaxis& time_axis) const { return source_accessor(temperature_ts, time_axis); }
		};

		//! Simple destination class for use in the IWD algorithms to represent a cell that can have temperature,
		//! radiation and precipitation set.
		class MCell {
		private:
			geo_point coord;
		public:
			arma::vec temperatures;
			template<typename T> explicit MCell(const geo_point& coord, const T& time_axis)
				: coord(coord),
				temperatures((arma::uword)time_axis.size()) {
				temperatures.fill(std::nan(""));
			}
			const geo_point& mid_point() const { return coord; }
			void set_temperature(const size_t time_idx, const double temperature) {
				temperatures[(arma::uword)time_idx] = temperature;
			}
			const double temperature(const size_t time_idx) const { return temperatures[(arma::uword)time_idx]; }

		};


		//! Simple parameter class
		class Parameter {
		private:
			double gradient = -0.006; // Prior expectation of temperature gradient [C/m]
			double gradient_sd = 0.0025; // Prior standard deviation of temperature gradient in [C/m]
			double sill_value = 25.0; // Value of semivariogram at range
			double nug_value = 0.5; // Nugget magnitude
			double range_value = 200000.0; // Point where semivariogram flattens out
			double  zscale_value = 20.0; // Height scale used during distance computations.
		public:
			Parameter() {}
			Parameter(double temperature_gradient, double temperature_gradient_sd)
				: gradient(temperature_gradient / 100),
				gradient_sd(temperature_gradient_sd / 100) {}

			const double temperature_gradient(shyft::core::utcperiod period) const {
				return gradient;
			}

			const double& temperature_gradient_sd() const {
				return gradient_sd;
			}

			double sill() const { return sill_value; }
			double nug() const { return nug_value; }
			double range() const { return range_value; }
			double zscale() const { return zscale_value; }
		};


		//! TODO: Accessor class
		class AccessorMocker {};


		// Make some aliases for the tests below
		typedef std::vector<Source> SourceList;
		typedef std::vector<MCell> DestinationList;
	}; // Namespace btk_structs
}; // Namespace test

using std::begin;
using std::end;
using namespace shyfttest::btk_structs;
using shyft::timeseries::point_timeaxis;
using namespace shyft::core;

typedef shyfttest::xpts_t xpts_t;
std::vector<utctime> ctimes{ 0, 3600 };
typedef shyft::timeseries::point_timeaxis point_timeaxis;
typedef std::vector<shyft::timeseries::point> point_vector_t;

void build_sources_and_dests(const size_t num_sources_x, const size_t num_sources_y,
	const size_t num_dests_x, const size_t num_dests_y,
	const size_t ts_size, const shyft::timeseries::utctimespan dt,
	const point_timeaxis& time_axis, bool insert_nans, SourceList& sources, DestinationList& dests) {
	const double x_min = 0.0; // [m]
	const double x_max = 100000.0; // [m]
	const double y_min = 0.0; // [m]
	const double y_max = 1000000.0; // [m]

	sources.reserve(num_sources_x*num_sources_y);
	dests.reserve(num_dests_x*num_dests_y);
	geo_point pt;
	double lower_bound = 0.0;
	double upper_bound = 10.0;
	std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
	std::default_random_engine re;
    vector<utctime> times;times.reserve(ts_size);
    for (size_t l = 0; l < ts_size; ++l)
        times.emplace_back(l*dt);
    times.emplace_back(shyft::core::max_utctime);
    point_timeaxis dta(times);
	for (size_t i = 0; i < num_sources_x; ++i) {
		pt.x = x_min + i*(x_max - x_min) / (num_sources_x - 1);
		for (size_t j = 0; j < num_sources_y; ++j) {
			pt.y = y_min + j*(y_max - y_min) / (num_sources_y - 1);
			pt.z = 500 * std::sin(pt.x / x_max) + std::sin(pt.y / y_max) / 2;
			vector<double> pts; pts.reserve(ts_size);
			double b_t = unif(re);
			//std::cout << "Base temp at pos (i,j) = " << i << ", " << j << ") = " << b_t << std::endl;
			for (size_t l = 0; l < ts_size; ++l)
				pts.emplace_back( b_t + pt.z*(0.6 / 100));
			sources.emplace_back(pt, xpts_t(dta,pts));
		}
	}
	for (size_t i = 0; i < num_dests_x; ++i) {
		pt.x = x_min + i*(x_max - x_min) / (num_dests_x - 1);
		for (size_t j = 0; j < num_dests_y; ++j) {
			pt.y = y_min + j*(y_max - y_min) / (num_dests_y - 1);
			pt.z = 500 * (std::sin(pt.x / x_max) + std::sin(pt.y / y_max)) / 2;
			dests.emplace_back(pt, time_axis);
		}
	}
}
using namespace shyft::core::bayesian_kriging;

void bayesian_kriging_test::test_covariance_calculation() {
	Parameter params;

	const arma::uword n = 100;
	arma::mat C(n, n, arma::fill::eye);
	arma::vec dqs(n*(n - 1) / 2); // Strict upper part of distance matrix, as vector
	arma::vec covs(n*(n - 1) / 2); // Strict upper part of cov matrix, as vector
	for (arma::uword i = 0; i < n*(n - 1) / 2; ++i) dqs.at(i) = static_cast<double>((i + 1)*(i + 1)); // NB: Not valid dists
	// Start timer
    bool verbose = getenv("SHYFT_VERBOSE")!=nullptr;
	const std::clock_t start = std::clock();
	utils::cov(dqs, covs, params);
	arma::uword v_idx = 0;
	for (arma::uword i = 0; i < n; ++i)
		for (arma::uword j = i + 1; j < n; ++j)
			C.at(i, j) = covs.at(v_idx++);
	C.diag() *= utils::zero_dist_cov(params);
    if (verbose) {
        const std::clock_t total = std::clock() - start;
        std::cout << "\nComputing upper covariance matrix with nnz " << n*(n + 1) / 2 << " took: " << 1000 * (total) / (double)(CLOCKS_PER_SEC) << " ms" << std::endl;
    }

	// Compute source-target cov
	const arma::uword m = 100 * 100;
	arma::vec dqms(n*m), covs2(n*m);
	v_idx = 0;
	for (arma::uword i = 0; i < n; ++i)
		for (arma::uword j = 0; j < m; ++j)
			dqms.at(i*m + j) = static_cast<double>((i + 1)*(j + 1));
	const std::clock_t start2 = std::clock();
	utils::cov(dqms, covs2, params);
	arma::mat CC(covs2);
	CC.reshape(n, m);
	if(verbose) {
        const std::clock_t total2 = std::clock() - start2;
        std::cout << "Computing full covariance matrix with nnz " << n*m << " took: " << 1000 * (total2) / (double)(CLOCKS_PER_SEC) << " ms" << std::endl;
	}
}

void bayesian_kriging_test::test_build_covariance_matrices() {
	Parameter params;
	const point_timeaxis time_axis(ctimes);
	SourceList sources;
	DestinationList destinations;
	build_sources_and_dests(3, 3, 15, 15, 2, 10, time_axis, false, sources, destinations);
	arma::mat K, k;
	utils::build_covariance_matrices(begin(sources), end(sources), begin(destinations), end(destinations), params, K, k);
	TS_ASSERT_EQUALS(K.n_rows, (size_t)9);
	TS_ASSERT_EQUALS(K.n_cols, (size_t)9);
    for (size_t i = 0;i < 3;++i) {
        TS_ASSERT_DELTA(K(i, i), params.sill() - params.nug(), 0.00001);
    }
    // verify symmetry
    for (size_t r = 0;r < 3;++r) {
        for (size_t c = 0;c < 3;++c) {
            TS_ASSERT_DELTA(K(r, c), K(c, r), 1e-9);
        }
    }
    // verify two off diagnoal values
    TS_ASSERT_DELTA(K(0, 1), 2.011082466, 1e-6);
    TS_ASSERT_DELTA(K(0, 2), 0.165079701, 1e-6);
    // verify size of the other matrix
	TS_ASSERT_EQUALS(k.n_rows, (size_t)9);
	TS_ASSERT_EQUALS(k.n_cols, (size_t)15 * 15);
}

void bayesian_kriging_test::test_build_elevation_matrices() {
	Parameter params;
	const point_timeaxis time_axis(ctimes);
	SourceList sources;
	DestinationList destinations;
	build_sources_and_dests(3, 3, 15, 15, 2, 10, time_axis, false, sources, destinations);
	arma::mat S_e, D_e;
	utils::build_elevation_matrices(begin(sources), end(sources), begin(destinations), end(destinations), S_e, D_e);
	TS_ASSERT_EQUALS(S_e.n_cols, (size_t)2);
	TS_ASSERT_EQUALS(S_e.n_rows, (size_t)9);
	TS_ASSERT_EQUALS(D_e.n_cols, (size_t)15 * 15);
	TS_ASSERT_EQUALS(D_e.n_rows, (size_t)2);
}

void bayesian_kriging_test::test_interpolation() {
	Parameter params;
	SourceList sources;
	DestinationList destinations;
	using namespace shyft::timeseries;
	using namespace shyft::core;
	using namespace shyfttest;
	size_t n_s = 3;
	size_t n_d = 9;
	size_t n_times = 2;
	shyft::timeseries::utctime dt = 10;
	vector<utctime> times; times.reserve(n_times);
	for (size_t i = 0; i < n_times; ++i)
		times.emplace_back(dt*i);
	const point_timeaxis time_axis(times);
	build_sources_and_dests(n_s, n_s, n_d, n_d, n_times, dt, time_axis, true, sources, destinations);
	const std::clock_t start = std::clock();
	btk_interpolation<average_accessor<shyfttest::xpts_t, point_timeaxis>>(begin(sources), end(sources), begin(destinations), end(destinations), time_axis, params);
	const std::clock_t total = std::clock() - start;
    double e_temp[6]{ 1.35,5.31,7.06,7.778,7.78,8.55 };
    for(size_t i=0;i<6;++i)
        TS_ASSERT_DELTA(destinations[i].temperatures[0], e_temp[i], 0.01);

	bool verbose = getenv("SHYFT_VERBOSE") != nullptr;
	if(verbose) std::cout << "Calling compute with n_sources, n_dests, and n_times = " << n_s*n_s << ", " << n_d*n_d << ", " << n_times << " took: " << 1000 * (total) / (double)(CLOCKS_PER_SEC) << " ms" << std::endl;
	if (verbose) {
		std::cout << "\taltitude\tmax\tmin\n ";
		for (auto d : destinations) {
			std::cout <<std::setprecision(3)<<"\t "<< d.mid_point().z <<"\t "
			<< *std::max_element(d.temperatures.begin(), d.temperatures.end())<<"\t "
            << *std::min_element(d.temperatures.begin(), d.temperatures.end()) << std::endl;
		}
	}
}

namespace shyft {
    namespace core {
        namespace bayesian_kriging {
            /** play ground for met.no stuff (only needed parts will be promoted) */
            namespace covariance {
                
                /** cressman_3d have a covariance in linear^2 in 3d distance
                 * <a href="http://modb.oce.ulg.ac.be/wiki/upload/diva_intro.pdf">References to relevant theory chapter 1.5 </a> 
                 */
                struct cressman_3d {
                    double max_radius2;// just keep ready made squared value 
                    double fold_distance2;
                    double max_elev_diff2;
                    double scale;
                    cressman_3d(double fold_distance,double max_radius, double max_elev_diff,double scale=1.0)
                        :max_radius2(max_radius*max_radius),
                        fold_distance2(fold_distance*fold_distance), 
                        max_elev_diff2(max_elev_diff*max_elev_diff),scale(scale) {
                    }

                    double operator()(const geo_point&p1, const geo_point&p2) {
                        double h_dist2 = geo_point::xy_distance2(p1,p2);
                        double v_dist2 = geo_point::z_distance2(p1,p2);
                        if (h_dist2 > max_radius2 || h_dist2>fold_distance2 || v_dist2 > max_elev_diff2)
                            return 0.0;
                        return scale* (fold_distance2 - h_dist2)*(max_elev_diff2 - v_dist2) /
                            ((fold_distance2 + h_dist2)*(max_elev_diff2 + v_dist2));
                    }
                };
                /** cressman_2d have a covariance in linear^2 in 2d horizontal distance
                 * <a href="http://modb.oce.ulg.ac.be/wiki/upload/diva_intro.pdf">References to relevant theory chapter 1.5 </a> 
                 */
                struct cressman_2d {
                    double fold_distance2;
                    double max_radius2;// just keep ready made squared value 
                    cressman_2d(double fold_distance, double max_radius)
                        :fold_distance2(fold_distance*fold_distance),
                        max_radius2(max_radius*max_radius) {
                    }

                    double operator()(const geo_point&p1, const geo_point&p2) {
                        double h_dist2 = geo_point::xy_distance2(p1, p2);
                        double v_dist2 = geo_point::z_distance2(p1, p2);
                        if (h_dist2 > max_radius2 || h_dist2>fold_distance2)
                            return 0.0;
                        return (fold_distance2 - h_dist2) /
                            (fold_distance2 + h_dist2);
                    }
                };

                /** barnes_2d have a covariance in linear^2 in 2d horizontal distance
                 * <a href="http://modb.oce.ulg.ac.be/wiki/upload/diva_intro.pdf">References to relevant theory chapter 1.5 </a>
                 */
                struct barnes_2d {
                    double fold_distance2;
                    double max_radius2;// just keep ready made squared value 
                    barnes_2d(double fold_distance,double max_radius )
                        :fold_distance2(fold_distance*fold_distance),
                         max_radius2(max_radius*max_radius)
                         {
                    }

                    double operator()(const geo_point&p1, const geo_point&p2) {
                        double h_dist2 = geo_point::xy_distance2(p1, p2);
                        double v_dist2 = geo_point::z_distance2(p1, p2);
                        if (h_dist2 > max_radius2)
                            return 0.0;
                        return exp( - 0.5*(fold_distance2 - h_dist2)/(fold_distance2 + h_dist2));
                    }
                };

                /** barnes_3d have a covariance in exp (linear^2 in 3d distance) 
                * <a href="http://modb.oce.ulg.ac.be/wiki/upload/diva_intro.pdf">References to relevant theory chapter 1.5 </a>
                */
                struct barnes_3d {
                    double max_radius2;// just keep ready made squared value 
                    double fold_distance2;
                    double max_elev_diff2;
                    double scale;
                    barnes_3d(double fold_distance, double max_radius, double max_elev_diff, double scale=1.0)
                        :max_radius2(max_radius*max_radius),
                        fold_distance2(fold_distance*fold_distance),
                        max_elev_diff2(max_elev_diff*max_elev_diff),scale(scale) {
                    }

                    double operator()(const geo_point&p1, const geo_point&p2) {
                        double h_dist2 = geo_point::xy_distance2(p1, p2);
                        double v_dist2 = geo_point::z_distance2(p1, p2);
                        if (h_dist2 > max_radius2 || v_dist2 > max_elev_diff2)
                            return 0.0;
                        return scale*exp(
                            -0.5*(
                                  (fold_distance2 - h_dist2) / (fold_distance2 + h_dist2) 
                                + (max_elev_diff2 - v_dist2) / (max_elev_diff2 + v_dist2)
                            )
                        );
                    }
                };


                /** build covariance matrix between observations [s..s_end>
                * given the covariance function.
                * \tparam S source type iterators, the value type must provide mid_point() ->geo_point
                * \tparam F_cov the type of a covariance callable function
                * \param s the begin of the source range
                * \param s_end the end of the source range
                * \param f_cov a callable(geo_point p1,geo_point p2)->double that computes the covariance between p1 and p2
                * \return the covariance matrix(n x n), symmetric, diagonal=1.
                */
                template <class S, class F_cov>
                arma::mat build(S s, S s_end, F_cov && f_cov) {
                    using int_t = arma::uword;
                    const int_t n = (int_t)std::distance(s, s_end);
                    arma::mat c(n, n, arma::fill::eye);
                    for (int_t i = 0;i < n; ++i)
                        for (int_t j = i + 1;j < n; ++j)
                            c.at(j, i) = c.at(i, j) = f_cov((s + i)->mid_point(), (s + j)->mid_point());
                    return std::move(c);
                }

                /** build covariance matrix between observations [s..s_end> and destinations(grid) [d..d_end>
                * given the covariance function.
                * \tparam S source type iterators, the value type must provide mid_point() ->geo_point
                * \tparam D destination type iterators, the value type must provide mid_point() ->geo_point
                * \tparam F_cov the type of a covariance callable function
                * \param s the begin of the source range
                * \param s_end the end of the source range
                * \param d the begin of the destination range
                * \param d_end the end of the destination range
                * \param f_cov a callable(geo_point p1,geo_point p2)->double that computes the covariance between p1 and p2
                * \return the covariance matrix(n x n), symmetric, diagonal=1.
                */template <class S, class D,class F_cov>
                arma::mat build(S s, S s_end,D d,D d_end, F_cov && f_cov) {
                    using int_t = arma::uword;
                    const int_t n = (int_t)std::distance(s, s_end);
                    const int_t m = (int_t)std::distance(d, d_end);
                    arma::mat c(n, m, arma::fill::none);
                    for (int_t i = 0;i < n; ++i)
                        for (int_t j = 0;j < m; ++j)
                            c.at(i, j) = f_cov((s + i)->mid_point(), (d + j)->mid_point());
                    return std::move(c);
                }
            }
        }
    }
}
void bayesian_kriging_test::test_met_no() {
    using namespace shyft::core;
    using namespace std;
    struct location{
        geo_point p;
        double v;
        geo_point mid_point() const { return p; }
        double value(utctime t) const {
            return v ;
        }
    };
    struct cov_test {
        double operator()(geo_point a, geo_point b) const {
            return (a.x) * 10 + (b.x);
        }
    };
    /* verify covariance::build observation vs observation produces the correct matrix */ {
        vector<location> s;
        s.push_back(location{ geo_point(1,   0,  0),10.0 });
        s.push_back(location{ geo_point(2, 100, 10),10.0 });
        s.push_back(location{ geo_point(3, 100, 20),10.0 });

        auto c = bayesian_kriging::covariance::build(begin(s), end(s), cov_test());
        TS_ASSERT_DELTA(c(0, 0), 1.0, 0.001);
        TS_ASSERT_DELTA(c(1, 1), 1.0, 0.001);
        TS_ASSERT_DELTA(c(2, 2), 1.0, 0.001);
        TS_ASSERT_DELTA(c(0, 1), 12.0, 0.001);
        TS_ASSERT_DELTA(c(0, 2), 13.0, 0.001);
        TS_ASSERT_DELTA(c(1, 0), 12.0, 0.001);
        TS_ASSERT_DELTA(c(2, 0), 13.0, 0.001);
        TS_ASSERT_DELTA(c(1, 2), 23.0, 0.001);
        TS_ASSERT_DELTA(c(2, 1), 23.0, 0.001);

        std::cout << "\n" << c << endl;
        std::cout << "\nc.inv()\n" << c.i() << endl;
    }
    /* verify covariance::build observation vs grid(aka destination) produces the correct matrix */ {
        vector<location> s;
        s.push_back(location{ geo_point(1,   0,  0),10.0 });
        s.push_back(location{ geo_point(2, 100, 10),10.0 });
        s.push_back(location{ geo_point(3, 100, 20),10.0 });
        vector<location> d;
        for (double x = 1;x < 10.0;x += 1.0) d.push_back(location{ geo_point(x,x,x),10.0 });

        auto c = bayesian_kriging::covariance::build(begin(s), end(s),begin(d),end(d), cov_test());
        TS_ASSERT_EQUALS(c.n_rows, 3);
        TS_ASSERT_EQUALS(c.n_cols, 9);
        cov_test fx;
        for (arma::uword i = 0;i < c.n_rows;++i)
            for (arma::uword j = 0;j < c.n_cols;++j)
                TS_ASSERT_DELTA(c(i, j), fx(s[i].mid_point(),d[j].mid_point()) /*1.0*(i + 1) + double(j+1)*/, 0.001);
        std::cout << "\n" << c << endl;
    }
    /* verify kriging setup */ {
        using int_t = arma::uword;
        /* met.no 10x10 case */ {
 
            vector<double> x = { -487442.2 ,-484942.2,	 -482442.2,	-479942.2,	-477442.2,	-474942.2,	-472442.2,	-469942.2,	-467442.2,	-464942.2 };
            vector<double> y = { -269321.8,-266821.8,-264321.8,-261821.8,-259321.8,-256821.8,-254321.8,-251821.8,-249321.8,-246821.8 };
            vector<location> grid_10x10 =
            {
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
            vector<location> obs1 = {
                location {geo_point(x[5],y[5],140.0),+4.2}, // +4.2
                location {geo_point(x[9],y[9],800.0),-1.1}, // -1.1
                location {geo_point(x[9],y[8],850.0), -0.3} // -0.3
            };
            bayesian_kriging::covariance::cressman_3d cressman_cov(200000, 10*1000.0, 10000, 0.414 / 0.5);

            auto Coo = bayesian_kriging::covariance::build(begin(obs1), end(obs1), cressman_cov);
            cressman_cov.scale = 1.0;
            auto Cod = bayesian_kriging::covariance::build(begin(obs1), end(obs1), begin(grid_10x10), end(grid_10x10), cressman_cov);
            auto ww = (Coo.i()*Cod).eval();
            arma::mat o(1, 3, arma::fill::none);
            for (int_t c = 0;c < 3;c++)
                o(0, c) = obs1[c].value(0);
            auto grd_values = (o * ww).eval();
            grd_values.reshape(10, 10);
            cout << "\ngrd_values\n" << grd_values << endl;
            //TODO: 
            // parameters for kriging: radius=1, efold. 200000, max elv diff=100, cressman
            // expect bias at [5,5] = 0.992003, , elevation diff =140.0-142.1
            // [0,0] = 1.0, [4,5]=1.0, hmm.
            //
        }
        vector<location> s;
        s.push_back(location{ geo_point(    0,     0,  0),-0.2 });
        s.push_back(location{ geo_point(    0, 1000, 100),0.1 });
        s.push_back(location{ geo_point(10000, 10000, 1000),0.2 });
        vector<location> d;
        size_t n_x = 3;
        size_t n_y = 4;
        for (size_t i = 0;i < n_x;++i)
            for (size_t j = 0;j < n_y;++j) 
                d.push_back(location{ geo_point(i*1000.0,j*1000.0,(i+j)*100),0.0 });
        
        bayesian_kriging::covariance::barnes_3d cov_b3d(50*1000.0,300*1000.0,1200.0,0.414/0.5);// ref.met.no Kriging.cpp line 206
        auto K = bayesian_kriging::covariance::build(begin(s), end(s), cov_b3d);
        cov_b3d.scale = 1.0;// not for source-dest
        auto k = bayesian_kriging::covariance::build(begin(s), end(s), begin(d), end(d), cov_b3d);
        auto K_i = K.i().eval();
        auto w = (K_i*k).eval(); // weights, nxm, we need sum weigths for each row to be one
        // do it the met-no way grid-point by gridpoint
        size_t n_obs = s.size();
        arma::cube S(n_y,n_x,n_obs,arma::fill::zeros);
        for (int_t i = 0;i < n_y;++i) 
            for (int_t j = 0;j < n_x;++j)
                for (int_t k = 0;k < n_obs;++k) 
                    S(i, j, k) = cov_b3d( s[k].mid_point(), d[i*n_x + j].mid_point());
                
        
        std::cout <<"\nSk\n"<<S<< "\nK:\n" << K << "\nk:\n" << k << "\nK.i()\n" << K.i() << "\nw:\n" << w << endl;
        utctime t = 0;
        arma::mat obs(1, 3, arma::fill::none);
        for (int_t c = 0;c < 3;c++) 
            obs(0, c) = s[c].value(t);
        auto grid_values = (obs * w).eval();
        grid_values.reshape(n_y, n_x);
        std::cout << "\nobs \n" << obs << endl;
        std::cout << "\ngrid\n" << grid_values << endl;
        // now, do it the met.no way, using S.
        for(int_t i=0;i<n_y;++i)
            for (int_t j = 0;j < n_x;++j) {
                vector<double> weights;
                weights.resize(n_obs, 0.0);
                cout << "met.no weights " << i << "," << j << ":";
                for (int_t ii = 0;ii < n_obs;++ii)
                    for (int_t jj = 0;jj < n_obs;++jj) 
                        weights[ii] += K_i(ii, jj)* S(i, j, jj);
                double grid_value = 0.0;
                for (int_t ii = 0;ii < n_obs;++ii) {
                    grid_value += weights[ii] * obs(0, ii);
                    cout << weights[ii] << ",";
                }
                cout << " -> grid_value "<< grid_value<<endl;
                
            }

    }

}