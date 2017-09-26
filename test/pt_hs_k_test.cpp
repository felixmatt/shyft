
#include "test_pch.h"
#include "core/pt_hs_k.h"
#include "mocks.h"
#include "core/time_series.h"
#include "core/utctime_utilities.h"

// Some typedefs for clarity
using namespace shyft::core;
using namespace shyft::time_series;
using namespace shyft::core::pt_hs_k;

using namespace shyfttest::mock;
using namespace shyfttest;

namespace pt = shyft::core::priestley_taylor;
namespace hs = shyft::core::hbv_snow;
namespace gm = shyft::core::glacier_melt;
namespace kr = shyft::core::kirchner;
namespace ae = shyft::core::actual_evapotranspiration;
namespace pc = shyft::core::precipitation_correction;
namespace ta = shyft::time_axis;
typedef TSPointTarget<ta::point_dt> catchment_t;

namespace shyfttest {
    namespace mock {
        // need specialization for pthsk_response_t above
        template<> template<>
        void ResponseCollector<ta::fixed_dt>::collect<response>(size_t idx, const response& response) {
            _snow_output.set(idx, response.snow.outflow);
        }
        template <> template <>
        void DischargeCollector<ta::fixed_dt>::collect<response>(size_t idx, const response& response) {
            // q_avg is given in mm, so compute the totals
            avg_discharge.set(idx, destination_area*response.kirchner.q_avg / 1000.0 / 3600.0);
        }
    };
}; // End namespace shyfttest
TEST_SUITE("pt_hs_k") {
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
    std::vector<double> s = {1.0, 1.0, 1.0, 1.0, 1.0}; // Zero cv distribution of snow (i.e. even)
    std::vector<double> a = {0.0, 0.25, 0.5, 0.75, 1.0};
    pt::parameter pt_param;
    hs::parameter snow_param(s, a);
    ae::parameter ae_param;
    kr::parameter k_param;
    pc::parameter p_corr_param;

    // Initialize the state vectors
    kr::state kirchner_state {5.0};
    hs::state snow_state(10.0, 0.5);

    // Initialize response
    response run_response;

    // Initialize collectors
    shyfttest::mock::ResponseCollector<ta::fixed_dt> response_collector(1000*1000, time_axis);
    shyfttest::mock::StateCollector<ta::fixed_dt> state_collector(state_time_axis);

    state state {snow_state, kirchner_state};
    parameter parameter(pt_param, snow_param, ae_param, k_param, p_corr_param);
    geo_cell_data geo_cell_data;
    pt_hs_k::run<direct_accessor, response>(geo_cell_data, parameter, time_axis,0,0, temp,
                                              prec, wind_speed, rel_hum, radiation, state,
                                              state_collector, response_collector);

    auto snow_swe = response_collector.snow_swe();
    for (size_t i = 0; i < snow_swe.size(); ++i)
        TS_ASSERT(std::isfinite(snow_swe.get(i).v) && snow_swe.get(i).v >= 0);
}
}
