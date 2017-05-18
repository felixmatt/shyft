#include "test_pch.h"
#include "mocks.h"
#include "core/kirchner.h"


namespace shyfttest {
    const double EPS = 1.0e-6;

}
using namespace shyft::core::kirchner;
TEST_SUITE("kirchner") {
TEST_CASE("test_single_solve") {
    using namespace shyft::core;
    parameter p;
    calculator<kirchner::trapezoidal_average, parameter> k(p);
    const double P = 2.0;
    const double E = 0.5;
    double q1 = 1.0;
    double q2 = 1.0;
    double qa1 = 0.0;
    double qa2 = 0.0;
    k.step(0, 1, q1, qa1, P, E);
    k.step(0, 1, q2, qa2, P, E);
    TS_ASSERT_DELTA(q1, q2, shyfttest::EPS);
    TS_ASSERT_DELTA(qa1, qa2, shyfttest::EPS);
}
TEST_CASE("test_solve_from_zero_q") {
    using namespace shyft::core;
    parameter p;
    calculator<kirchner::trapezoidal_average, parameter> k(p);
    double P = 10.0;
    double E = 0.0;
    double q_state = 0.0;
    double q_response=0.0;
    // after a number of iterations, the output should equal the input
    for(int i=0;i<10000000;i++) {
        k.step(0, 3600, q_state, q_response, P, E);
        if(fabs(q_state -P) < 0.001 && fabs(q_response-P)<0.001 )
            break;
    }
    TS_ASSERT_DELTA(q_state, P, 0.001);
    TS_ASSERT_DELTA(q_response,P, 0.001);

}

TEST_CASE("test_hard_case") {
    using namespace shyft::core;
    parameter p;
    const double P = 5.93591;
    const double E = 0.0;
    double q = 2.29339;
    double q_a = 0.0;
    double atol = 1.0e-2;
    double rtol = 1.0e-4;
	bool verbose = getenv("SHYFT_VERBOSE")!=nullptr;
    for (size_t i = 0; i < 10; ++i) {
        calculator<kirchner::trapezoidal_average, parameter> k(atol, rtol, p);
        k.step(0, deltahours(1), q, q_a, P, E);
        if(verbose) std::cout << "r_tol = " << rtol << ", a_tol = " << atol << ", q = " << q << ", q_a = "<< q_a << std::endl;
        atol /= 2.0;
        rtol /= 2.0;
    }
    TS_ASSERT(q < 100.0);
}

TEST_CASE("test_simple_average_loads") {
    using namespace shyft::core;
    parameter p;
    calculator<kirchner::trapezoidal_average, parameter> k(1.0e-6, 1.0e-5, p);
    size_t n_x = 200;  // Simulated grid size in x direction
    size_t n_y = 300;  // Simulated grid size in x direction
    double sum_q = 0.0;
    double sum_qa = 0.0;
    const std::clock_t start = std::clock();
    double Q, Q_avg;
    const double P = 0.0; // Zero precipitation makes the ode system quite simple to solve
    const double E = 0.2;
    for (size_t i=0; i < n_x*n_y; ++i) {
        Q = 1.0;
        k.step(0, 1, Q, Q_avg, P, E);
        sum_q += Q;
        sum_qa += Q_avg;
    }
    if(getenv("SHYFT_VERBOSE")) {
        const std::clock_t total = std::clock() - start;
        std::cout << "Stepping simple avg Kirchner model " << n_x*n_y << " times took " << 1000*(total)/(double)(CLOCKS_PER_SEC) << " ms" << std::endl;
    }
}
using namespace std;
TEST_CASE("test_composite_average_loads") {
    using namespace shyft::core;
    parameter p;
    calculator<kirchner::composite_trapezoidal_average, parameter> k(1.0e-6, 1.0e-5, p);
    size_t n_x = 200;  // Simulated grid size in x direction
    size_t n_y = 300;  // Simulated grid size in x direction
    double sum_q = 0.0;
    double sum_qa = 0.0;
    const std::clock_t start = std::clock();
    double Q, Q_avg;
    const double P = 0.0; // Zero precipitation makes the ode system quite simple to solve
    const double E = 0.2;
    for (size_t i=0; i < n_x*n_y; ++i) {
        Q = 1.0;
        k.step(0, deltahours(1), Q, Q_avg, P, E);
        sum_q += Q;
        sum_qa += Q_avg;
    }
    if(getenv("SHYFT_VERBOSE")) {
        const std::clock_t total = std::clock() - start;
        std::cout<< " Q: "<< sum_q/n_x/n_y<<" Qa: "<< sum_qa/(n_x*n_y)<<endl;
        std::cout << "Stepping composite avg Kirchner model " << n_x*n_y << " times took " << 1000*(total)/(double)(CLOCKS_PER_SEC) << " ms" << std::endl;
    }
}

}
