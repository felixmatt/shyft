#include "test_pch.h"
#include "core/pt_ss_k.h"
#include "mocks.h"
#include "core/time_series.h"
#include "core/utctime_utilities.h"

// Some typedefs for clarity
using namespace shyft::core;
using namespace shyft::time_series;

using namespace shyft::core::pt_ss_k;
using namespace shyfttest;
using namespace shyfttest::mock;

namespace pt = shyft::core::priestley_taylor;
namespace ss = shyft::core::skaugen;
namespace kr = shyft::core::kirchner;
namespace ae = shyft::core::actual_evapotranspiration;
namespace pc = shyft::core::precipitation_correction;
namespace ta = shyft::time_axis;

typedef TSPointTarget<ta::point_dt> catchment_t;
typedef shyft::core::pt_ss_k::response response_t;
typedef shyft::core::pt_ss_k::state state_t;
typedef shyft::core::pt_ss_k::parameter parameter_t;

namespace shyfttest {
    namespace mock {
        template<> template<>
        void ResponseCollector<ta::fixed_dt>::collect<response>(size_t idx, const response& response) {
            _snow_output.set(idx, response.snow.outflow);
            _snow_swe.set(idx, response.snow.total_stored_water);
        }

        template <> template <>
        void DischargeCollector<ta::fixed_dt>::collect<response>(size_t idx, const response& response) {
            avg_discharge.set(idx, destination_area*response.kirchner.q_avg/1000.0/3600.0); // q_avg is given in mm, so compute the totals
        }
    } // mock
} // shyfttest

TEST_SUITE("pt_ss_k") {
TEST_CASE("test_call_stack") {
    xpts_t temp;
    xpts_t prec;
    xpts_t rel_hum;
    xpts_t wind_speed;
    xpts_t radiation;

    calendar cal;
    utctime t0 = cal.time(YMDhms(2014, 8, 1, 0, 0, 0));
    size_t n_ts_points = 3*24;
    utctimespan dt  = deltahours(1);
    utctime t1 = t0 + n_ts_points*dt;
    shyfttest::create_time_series(temp, prec, rel_hum, wind_speed, radiation, t0, dt, n_ts_points);

    utctime model_dt = deltahours(24);
    vector<utctime> times;
    for (utctime i=t0; i <= t1; i += model_dt)
        times.emplace_back(i);
    ta::fixed_dt time_axis(t0, dt, n_ts_points);
	ta::fixed_dt state_time_axis(t0, dt, n_ts_points + 1);
    // Initialize parameters
    // Snow model parameters
    const double d_range = 113.0;
    const double unit_size = 0.1;
    const double alpha_0 = 40.77;
    const double max_water_fraction = 0.1;
    const double tx = 0.16;
    const double cx = 2.50;
    const double ts = 0.14;
    const double cfr = 0.01;
    ss::parameter snow_param(alpha_0, d_range, unit_size, max_water_fraction, tx, cx, ts, cfr);
    pt::parameter pt_param;
    ae::parameter ae_param;
    kr::parameter k_param;
    pc::parameter p_corr_param;

    // Initialize the state vectors
    // Snow model state variables
    const double alpha = alpha_0;
    const double nu = alpha_0*unit_size;
    const double sca = 0.0;
    const double swe = 0.0;
    const double free_water = 0.0;
    const double residual = 0.0;
    const unsigned long nnn = 0;
    kr::state kirchner_state = {5.0};
    ss::state snow_state(nu, alpha, sca, swe, free_water, residual, nnn);

    // Initialize response
    response_t response;
    //
    // Initialize collectors
    shyfttest::mock::ResponseCollector<ta::fixed_dt> response_collector(1000*1000, time_axis);
    shyfttest::mock::StateCollector<ta::fixed_dt> state_collector(state_time_axis);

    state_t state {snow_state, kirchner_state};
    parameter_t parameter(pt_param, snow_param, ae_param, k_param, p_corr_param);
    geo_cell_data geo_cell_data;
    pt_ss_k::run<direct_accessor, response_t>(geo_cell_data, parameter, time_axis,0,0, temp, prec,
                                              wind_speed, rel_hum, radiation, state, state_collector,
                                              response_collector);

    auto snow_swe = response_collector.snow_swe();
    for (size_t i = 0; i < snow_swe.size(); ++i)
        TS_ASSERT(std::isfinite(snow_swe.get(i).v) && snow_swe.get(i).v >= 0);
}
}
