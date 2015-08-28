#ifdef ENKI_HAS_GRF
#include "test_pch.h"

#include "gaussian_random_field_test.h"
#include "core/gaussian_random_field.h"
#include "mocks.h"
#include <armadillo>

namespace grf_test  {
    struct Point {
        double x;
        double y;
        Point(double x, double y) : x(x), y(y) {}
    };

    struct Source {
        Point pt = {0.0, 0.0};
        Source() : pt(0.0, 0.0) {}
        Source(double x, double y) : pt(x, y) {}
        const Point& geo_point() const { return pt; }
    };
}

void gaussian_random_field_test::test_calculate_anisotropy_distance() {
    size_t n_points_x = 2;
    size_t n_points_y = 2;
    double dx = 1000.0;
    double dy = 1000.0;
    std::vector<grf_test::Source> sources;
    sources.reserve(n_points_x*n_points_y);
    for (size_t i = 0; i < n_points_x; ++i)
        for (size_t j = 0; j < n_points_y; ++j)
            sources.emplace_back(grf_test::Source(i*dx, j*dy));

    std::vector<grf_test::Source*> sources_p;
    for (size_t i = 0; i < sources.size(); ++i)
        sources_p.emplace_back(&sources[i]);


    arma::mat AD;
    TS_ASSERT_THROWS_NOTHING(shyft::core::grf::calculate_anisotropy_distances(begin(sources_p), end(sources_p), 1.0, 90.0, AD));
    arma::mat AD2;
    shyft::core::grf::calculate_anisotropy_distances(begin(sources_p), end(sources_p), 1.0, 90.0, AD2);
    TS_ASSERT_DELTA(arma::norm(AD - AD2), 0.0, 1.0e-12);

    shyft::core::grf::calculate_anisotropy_distances(begin(sources_p), end(sources_p), 2.0, 120.0, AD2);
    std::cout << std::endl << AD2 << std::endl;
}

void gaussian_random_field_test::test_gaussian_model() {
    std::cout << std::endl << "Gaussian model" << std::endl;
    std::cout << "==============" << std::endl;
    arma::mat covariances;
    arma::mat semivariances;
    arma::mat gammas;
    arma::mat::fixed<2,2> distances = {0.0, 0.0, 1.0, 0.0};
    distances.operator=( arma::symmatu(distances));
    const double nugget = 1.0;
    const double sill = 2.0;
    const double range = 2.0;
    shyft::core::grf::semi_variogram semi_var(nugget, sill, range);


    std::cout << "Covariances:" << std::endl;
    shyft::core::grf::gaussian_model::covariance(distances, semi_var, covariances);
    std::cout << covariances;

    std::cout << "Semivariances:" << std::endl;
    shyft::core::grf::gaussian_model::semivariance(distances, semi_var, semivariances);
    std::cout << semivariances;

    std::cout << "Gammas:" << std::endl;
    shyft::core::grf::gaussian_model::semivariogram(distances, semi_var, gammas);
    std::cout << gammas;

    arma::mat diff = gammas - 2*semivariances;
    TS_ASSERT_EQUALS(std::count_if(std::begin(diff), std::end(diff), [] (double d) { return fabs(d) < 1.0e-14; }),
                     std::distance(std::begin(diff), std::end(diff)));

    diff = gammas - (sill - covariances);
    TS_ASSERT_EQUALS(std::count_if(std::begin(diff), std::end(diff), [] (double d) { return fabs(d) < 1.0e-14; }),
                     std::distance(std::begin(diff), std::end(diff)));
}

void gaussian_random_field_test::test_spherical_model() {
    std::cout << "Spherical model" << std::endl;
    std::cout << "===============" << std::endl;
    arma::mat covariances;
    arma::mat semivariances;
    arma::mat gammas;
    arma::mat::fixed<2,2> distances = {0.0, 0.0, 1.0, 0.0};
    distances.operator = (arma::symmatu(distances));
    const double nugget = 1.0;
    const double sill = 2.0;
    const double range = 2.0;
    shyft::core::grf::semi_variogram semi_var(nugget, sill, range);

    std::cout << "Covariances:" << std::endl;
    shyft::core::grf::spherical_model::covariance(distances, semi_var, covariances);
    std::cout << std::endl << covariances;

    std::cout << "Semivariances:" << std::endl;
    shyft::core::grf::spherical_model::semivariance(distances, semi_var, semivariances);
    std::cout << semivariances;

    std::cout << "Gammas:" << std::endl;
    shyft::core::grf::spherical_model::semivariogram(distances, semi_var, gammas);
    std::cout << gammas;

    arma::mat diff = covariances - semivariances;
    TS_ASSERT_EQUALS(std::count_if(std::begin(diff), std::end(diff), [] (double d) { return fabs(d) < 1.0e-14; }),
                     std::distance(std::begin(diff), std::end(diff)));
}

void gaussian_random_field_test::test_exponential_model() {
    std::cout << "Exponential model" << std::endl;
    arma::mat covariances;
    arma::mat semivariances;
    arma::mat gammas;
    arma::mat::fixed<2,2> distances = {0.0, 0.0, 1.0, 0.0};
    distances.operator = (arma::symmatu(distances));
    const double nugget = 1.0;
    const double sill = 2.0;
    const double range = 2.0;
    shyft::core::grf::semi_variogram semi_var(nugget, sill, range);

    std::cout << "Covariances:" << std::endl;
    shyft::core::grf::exponential_model::covariance(distances, semi_var, covariances);
    std::cout << std::endl << covariances;

    std::cout << "Semivariances:" << std::endl;
    shyft::core::grf::exponential_model::semivariance(distances, semi_var, semivariances);
    std::cout << semivariances;

    std::cout << "Gammas:" << std::endl;
    shyft::core::grf::exponential_model::semivariogram(distances, semi_var, gammas);
    std::cout << gammas;

    arma::mat diff = gammas - 2*semivariances;
    TS_ASSERT_EQUALS(std::count_if(std::begin(diff), std::end(diff), [] (double d) { return fabs(d) < 1.0e-14; }),
                     std::distance(std::begin(diff), std::end(diff)));

    diff = gammas - (sill - covariances);
    TS_ASSERT_EQUALS(std::count_if(std::begin(diff), std::end(diff), [] (double d) { return fabs(d) < 1.0e-14; }),
                     std::distance(std::begin(diff), std::end(diff)));
}

void gaussian_random_field_test::test_matern_model() {
    std::cout << "Matern model" << std::endl;
    arma::mat covariances;
    arma::mat semivariances;
    arma::mat gammas;
    arma::mat::fixed<2,2> distances = {0.0, 0.0, 1.0, 0.0};
    distances.operator=( arma::symmatu(distances));
    const double nugget = 1.0;
    const double sill = 2.0;
    const double range = 2.0;
    const double shape = 0.5;
    shyft::core::grf::semi_variogram semi_var(nugget, sill, range, shape);

    std::cout << "Covariances:" << std::endl;
    shyft::core::grf::matern_model::covariance(distances, semi_var, covariances);
    std::cout << std::endl << covariances;

    std::cout << "Semivariances:" << std::endl;
    shyft::core::grf::matern_model::semivariance(distances, semi_var, semivariances);
    std::cout << semivariances;

    std::cout << "Gammas:" << std::endl;
    shyft::core::grf::matern_model::semivariogram(distances, semi_var, gammas);
    std::cout << gammas;

    arma::mat diff = covariances - semivariances;
    TS_ASSERT_EQUALS(std::count_if(std::begin(diff), std::end(diff), [] (double d) { return fabs(d) < 1.0e-14; }),
                     std::distance(std::begin(diff), std::end(diff)));
}

void gaussian_random_field_test::test_calculate_gibbs_weights() {
    std::cout << std::endl << "Test calculate gibbs weights" << std::endl;

    arma::mat covariances;
    arma::mat::fixed<3,3> distances = {0.0, 1.0, 2.0,
                                       1.0, 0.0, 1.0,
                                       2.0, 1.0, 0.0};
    distances.operator= ( arma::symmatu(distances) );
    const double nugget = 0.0;
    const double sill = 1.00;
    const double range = 2.0;
    shyft::core::grf::semi_variogram semi_var(nugget, sill, range);

    const std::vector<double> precipitation = {1.0, 0.0, 1.0};

    arma::mat krig_weights;
    std::vector<unsigned int> sub_inds = {0, 1, 2};

    typedef shyft::core::grf::spherical_model model;

    model::covariance(distances, semi_var, covariances);
    shyft::core::grf::calculate_gibbs_weights(covariances, sub_inds, krig_weights);
    std::cout << "Krig weights" << std::endl;
    std::cout << krig_weights << std::endl;

    std::vector<double> mu, var;
    shyft::core::grf::calculate_gibbs_estimates(sub_inds, precipitation, krig_weights, covariances, semi_var.sill(), mu, var);
    std::cout << "Kriging estimates:" << std::endl;
    for (auto &m: mu) std::cout << m << " ";
    std::cout << std::endl;
    std::cout << "Kriging variances:" << std::endl;
    for (auto &v: var) std::cout << v << " ";
    std::cout << std::endl;
}

void gaussian_random_field_test::test_calculate_local_krig_weights() {
    std::cout << std::endl << "Test calculate local krig weights" << std::endl;

    size_t n_sources_x = 2;
    size_t n_sources_y = 2;
    double dx = 1000.0;
    double dy = 1000.0;
    std::vector<grf_test::Source> sources;
    std::vector<grf_test::Source*> sources_p;
    sources.reserve(n_sources_x*n_sources_y);
    for (size_t i = 0; i < n_sources_x; ++i)
        for (size_t j = 0; j < n_sources_y; ++j)
            sources.emplace_back(grf_test::Source(i*dx, j*dy));

    for (size_t i = 0; i < sources.size(); ++i)
        sources_p.emplace_back(&sources[i]);

    size_t n_destination_x = 3;
    size_t n_destination_y = 3;
    double dest_dx = 500.0;
    double dest_dy = 500.0;
    double dest_x0 = 250.0;
    double dest_y0 = 250.0;
    const double nugget = 1.0;
    const double sill = 2.0;
    const double range = 2000.0;
    shyft::core::grf::semi_variogram semi_var(nugget, sill, range);
    typedef shyft::core::grf::gaussian_model model;
    typedef shyfttest::mock::GRFDestination<grf_test::Point> SimplePointDestination;

    std::vector<SimplePointDestination> destinations;

    for (size_t i = 0; i < n_destination_x; ++i)
        for (size_t j = 0; j < n_destination_y; ++j)
            destinations.emplace_back(SimplePointDestination(grf_test::Point(dest_x0 + i*dest_dx, dest_y0 + j*dest_dy)));

    arma::mat distances;
    arma::mat covariances;
    arma::mat station_distances;
    arma::mat station_covariances;
    arma::umat sort_order;

    shyft::core::grf::calculate_anisotropy_distances(begin(sources_p),
                                                    end(sources_p),
                                                    begin(destinations),
                                                    end(destinations),
                                                    1.0,
                                                    90.0,
                                                    distances);
    shyft::core::grf::calculate_anisotropy_distances(begin(sources_p),
                                                    end(sources_p),
                                                    1.0,
                                                    90.0,
                                                    station_distances);
    model::covariance(station_distances, semi_var, station_covariances);
    model::covariance(distances, semi_var, covariances);

	const arma::uword max_neighbours = 4;
    shyft::core::grf::distance_sort_order(distances, max_neighbours, sort_order);
    for (arma::uword i = 0; i < destinations.size(); ++i) {
        //std::cout << "i = " << i << std::endl;
        destinations[i].set_source_covariances(covariances.row(i).t());
        destinations[i].set_source_sort_order(sort_order.row(i).t());
        shyft::core::grf::calculate_local_weights_data(station_covariances, max_neighbours, true, destinations[i]);
        shyft::core::grf::calculate_local_weights_data(station_covariances, max_neighbours, false, destinations[i]);
        //std::cout << w1 - w2;
    }
    //std::cout << sort_order;
    //std::cout << covariances;
}

void gaussian_random_field_test::test_gamma_transform() {
    std::cout << "Test gamma transform" << std::endl;
    const std::vector<double> prec = {5.0, 0.0, 10.0};
    const std::vector<double> m =    {5.0, 0.5, 3.0};
    const std::vector<double> cv =   {0.2, 0.2, 0.3};
    const std::vector<double> p0 =   {0.4, 0.4, 0.5};
    std::vector<double> res;
    std::vector<double> recovered_prec;
    shyft::core::grf::gamma_transform(prec, m, cv, p0, res);
    shyft::core::grf::inv_gamma_transform(res, m, cv, p0, recovered_prec);

    arma::vec err = arma::vec(prec) - arma::vec(recovered_prec);
    TS_ASSERT_EQUALS(std::count_if(err.begin(), err.end(), [] (double d) { return fabs(d) < 1.0e-8; }), err.n_rows);
}

void gaussian_random_field_test::test_gibbs_sampler() {
    return;
    std::cout << "Test Gibbs sampler" << std::endl;
    size_t n_sources_x = 2;
    size_t n_sources_y = 2;
    double dx = 1000.0;
    double dy = 1000.0;
    std::vector<double> precips; precips.reserve(n_sources_x*n_sources_y);
    std::vector<double> p_mean; p_mean.reserve(n_sources_x*n_sources_y);
    std::vector<double> p_cv; p_cv.reserve(n_sources_x*n_sources_y);
    std::vector<double> p_p0; p_p0.reserve(n_sources_x*n_sources_y);

    std::vector<unsigned int> pos_inds; pos_inds.reserve(n_sources_x*n_sources_y);
    std::vector<unsigned int> zero_inds; zero_inds.reserve(n_sources_x*n_sources_y);
    std::vector<grf_test::Source> sources;
    std::vector<grf_test::Source*> sources_p;
    sources.reserve(n_sources_x*n_sources_y);
    for (size_t i = 0; i < n_sources_x; ++i)
        for (size_t j = 0; j < n_sources_y; ++j) {
            sources.emplace_back(grf_test::Source(i*dx, j*dy));
            precips.emplace_back(0.1);
            p_mean.emplace_back(0.5);
            p_cv.emplace_back(1.2);
            p_p0.emplace_back(0.4);
        }

    for (size_t i = 0; i < sources.size(); ++i)
        sources_p.emplace_back(&sources[i]);


    precips[precips.size() - 1] = 0.0; // Zero precip at one station

    for(size_t i = 0; i < precips.size(); ++i)
        if (precips[i] > 0.0)
            pos_inds.emplace_back(i);
        else
            zero_inds.emplace_back(i);

    std::vector<double> trans_precip;

    shyft::core::grf::gamma_transform(precips, p_mean, p_cv, p_p0, trans_precip);

    arma::mat dists;
    shyft::core::grf::calculate_anisotropy_distances(begin(sources_p),
                                                    end(sources_p),
                                                    1.0,
                                                    90.0,
                                                    dists);
    arma::mat weights;
    arma::mat cov;
    std::vector<double> max_zero_trans_precip;

    typedef boost::math::policies::policy<boost::math::policies::digits10<3>> acc_policy;
    typedef boost::math::normal_distribution<double, acc_policy> normal_dist;
    normal_dist n_d;

    const size_t n_z = zero_inds.size();

    for (size_t i = 0; i < n_z; ++i) {
        const double t_p = trans_precip[zero_inds[i]];
        max_zero_trans_precip.emplace_back(t_p);
        const double p_norm = boost::math::cdf(n_d, t_p);
        // Initialize to truncated expectation
        trans_precip[zero_inds[i]] = boost::math::quantile(n_d, p_norm/2.0);
    }

    for (size_t i = 0; i < trans_precip.size(); ++i)
        std::cout << "Gaussian precipitation[" << i << "] = " << trans_precip[i] << std::endl;

    const double positive_mean = shyft::core::grf::subset_mean(trans_precip, pos_inds);
    const double positive_var = shyft::core::grf::subset_var(trans_precip, pos_inds);

    const double zero_var = 0.35; // TODO: Why?
    const double zero_mean = -0.75; // TODO: Why?
    const size_t n_p = pos_inds.size();

    const double tmp = (positive_mean - zero_mean)/(n_p + n_z);
    const double total_var = (positive_var*n_p + zero_var*n_z)/(n_p + n_z) + n_p*n_z*tmp*tmp;

    const double nugget = 1.0;
    const double sill = nugget + total_var;

    const double range = 2000.0;

    typedef shyft::core::grf::spherical_model model_t;

    shyft::core::grf::semi_variogram semi_var(nugget, sill, range);
    model_t::covariance(dists, semi_var, cov);
    shyft::core::grf::calculate_gibbs_weights(cov, zero_inds, weights);
    std::cout << "W.m = " << weights.n_rows << std::endl;
    std::cout << "W.n = " << weights.n_cols << std::endl;
    shyft::core::grf::gibbs_sampler<model_t>(5,
                                            5,
                                            weights,
                                            cov,
                                            dists,
                                            zero_inds,
                                            max_zero_trans_precip,
                                            semi_var,
                                            trans_precip);

    for (size_t i = 0; i < trans_precip.size(); ++i)
        std::cout << "Gaussian precipitation[" << i << "] = " << trans_precip[i] << std::endl;
}

void gaussian_random_field_test::test_random_fields() {

    std::cout << "Testing random fields" << std::endl;

    const double x_min = 0.0;
    const double x_max = 1000.0;
    const double dx = 100.0;
    const int n_x = (x_max - x_min)/dx + 1;

    const double y_min = 0.0;
    const double y_max = 1000.0;
    const double dy = 100.0;
    const int n_y = (y_max - y_min)/dy + 1;



    std::vector<double> grid_info = {x_min, x_max, dx, y_min, y_max, dy};

    //std::vector<double> x = {0.0, 1000.0, 0.0, 1000.0, 500.0};
    //std::vector<double> y = {0.0, 0.0, 1000.0, 1000.0, 500.0};
    //std::vector<double> xy = x;
    //xy.insert(end(xy), begin(y), end(y));

    double T = 1.0;
    int dim = 2;
    int lx = 3;
    int grid = 1;
    int with_time = 0;
    int model_number = GetModelNr_Mod((char*)"spherical");
    std::vector<double> sim_parameter = {0.5, 1000.0};
    int n_sim_parameter = sim_parameter.size();
    int n_cov = 1;
    int anisotropy = 0;
    int op = 0;
    int method_number = GetMethodNr_Mod((char*)"circulant embedding"); // ERROR; only for grids
    //int method_number = GetMethodNr_Mod((char*)"intrinsic CE"); // Seg. fault
    //int method_number = GetMethodNr_Mod((char*)"cutoff CE"); // Seg. fault
    //int method_number = GetMethodNr_Mod((char*)"TBM2"); // Seg. fault
    //int method_number = GetMethodNr_Mod((char*)"TBM3"); // Seg. fault
    //int method_number = GetMethodNr_Mod((char*)"spectral TBM"); // Seg. fault
    //int method_number = GetMethodNr_Mod((char*)"direct matrix decomposition"); // Seg. fault
    //int method_number = GetMethodNr_Mod((char*)"nugget");
    //int method_number = GetMethodNr_Mod((char*)"add.MPP");
    //int method_number = GetMethodNr_Mod((char*)"hyperplanes"); // Seg. fault
    int distribution = 0;
    int key = 0;
    int error = 0;

    InitSimulateRF(grid_info.data(),    //
                   &T,                  //
                   &dim,                //
                   &lx,                 //
                   &grid,               //
                   &with_time,          //
                   &model_number,       //
                   sim_parameter.data(),//
                   &n_sim_parameter,    //
                   &n_cov,              //
                   &anisotropy,         //
                   &op,                 //
                   &method_number,      //
                   &distribution,       //
                   &key,                //
                   &error);             //
    int modus = 0;
    double lambda = 0.0; // Mean value of the field

    StoreTrend_mod(&key,
                   &modus,
                   &lambda,
                   &error);

    int paired = 0;
    int n_r = 3;
    std::vector<double> gaussian_field(n_r*n_x*n_y, 0.0);
    //std::fill(begin(gaussian_field), end(gaussian_field), 0.0);
    std::cout << "Simulating!" << std::endl;
    DoSimulateRF(&key, &n_r, &paired, gaussian_field.data(), &error);
    std::cout << "Done DoSimulateRF" << std::endl;
    DeleteKey(&key);
    std::cout << "Done DeleteKey" << std::endl;

    for (size_t s = 0; s < (size_t)n_r; ++s) {
        for (size_t i = 0; i < (size_t)n_x; ++i) {
            for (size_t j = 0; j < (size_t)n_y; ++j)
                std::cout << gaussian_field[s*(n_x*n_y) + i*n_x + j] << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "Done testing the random fields functionality from R" << std::endl;

}

#endif
