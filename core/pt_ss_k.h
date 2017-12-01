#pragma once
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "core_pch.h"

#include "priestley_taylor.h"
#include "kirchner.h"
#include "skaugen.h"
#include "actual_evapotranspiration.h"
#include "precipitation_correction.h"
#include "glacier_melt.h"
#include "unit_conversion.h"
#include "routing.h"
namespace shyft {
  namespace core {
    namespace pt_ss_k {
        using namespace std;
        struct parameter {
            typedef priestley_taylor::parameter pt_parameter_t;
            typedef skaugen::parameter snow_parameter_t;
            typedef actual_evapotranspiration::parameter ae_parameter_t;
            typedef kirchner::parameter kirchner_parameter_t;
            typedef precipitation_correction::parameter precipitation_correction_parameter_t;
            typedef glacier_melt::parameter glacier_melt_parameter_t;
            typedef routing::uhg_parameter routing_parameter_t;
            pt_parameter_t pt;
            snow_parameter_t ss;
            ae_parameter_t ae;
            kirchner_parameter_t  kirchner;
            precipitation_correction_parameter_t p_corr;
            glacier_melt_parameter_t gm;
            routing_parameter_t routing;
            parameter(const pt_parameter_t& pt,
                      const snow_parameter_t& snow,
                      const ae_parameter_t& ae,
                      const kirchner_parameter_t& kirchner,
                      const precipitation_correction_parameter_t& p_corr,
                      glacier_melt_parameter_t gm=glacier_melt_parameter_t(),
                      routing_parameter_t routing=routing_parameter_t())
             : pt(pt), ss(snow), ae(ae), kirchner(kirchner), p_corr(p_corr),gm(gm),routing(routing) { /* Do nothing */ }

			parameter()=default;
			parameter(const parameter&)=default;
			parameter(parameter&&)=default;
			parameter& operator=(const parameter &c)=default;
			parameter& operator=(parameter&&c)=default;
            ///< Calibration support, size is the total number of calibration parameters
            size_t size() const { return 20; }

            void set(const vector<double>& p) {
                if (p.size() != size())
                    throw runtime_error("pt_ss_k parameter accessor: .set size mismatch");
                int i = 0;
                kirchner.c1 = p[i++];
                kirchner.c2 = p[i++];
                kirchner.c3 = p[i++];
                ae.ae_scale_factor = p[i++];
                ss.alpha_0 = p[i++];
                ss.d_range = p[i++];
                ss.unit_size = p[i++];
                ss.max_water_fraction = p[i++];
                ss.tx = p[i++];
                ss.cx = p[i++];
                ss.ts = p[i++];
                ss.cfr = p[i++];
                p_corr.scale_factor = p[i++];
				pt.albedo = p[i++];
				pt.alpha = p[i++];
                gm.dtf = p[i++];
                routing.velocity = p[i++];
                routing.alpha = p[i++];
                routing.beta  = p[i++];
                gm.direct_response = p[i++];
            }
            //
            ///< calibration support, get the value of i'th parameter
            double get(size_t i) const {
                switch (i) {
                    case  0:return kirchner.c1;
                    case  1:return kirchner.c2;
                    case  2:return kirchner.c3;
                    case  3:return ae.ae_scale_factor;
                    case  4:return ss.alpha_0;
                    case  5:return ss.d_range;
                    case  6:return ss.unit_size;
                    case  7:return ss.max_water_fraction;
                    case  8:return ss.tx;
                    case  9:return ss.cx;
                    case 10:return ss.ts;
                    case 11:return ss.cfr;
                    case 12:return p_corr.scale_factor;
					case 13:return pt.albedo;
					case 14:return pt.alpha;
                    case 15:return gm.dtf;
                    case 16:return routing.velocity;
                    case 17:return routing.alpha;
                    case 18:return routing.beta;
                    case 19:return gm.direct_response;

                default:
                    throw runtime_error("pt_ss_k parameter accessor:.get(i) Out of range.");
                }
            return 0.0;
            }

            ///< calibration and python support, get the i'th parameter name
            string get_name(size_t i) const {
                static const char *names[] = {
                    "kirchner.c1",
                    "kirchner.c2",
                    "kirchner.c3",
                    "ae.ae_scale_factor",
                    "ss.alpha_0",
                    "ss.d_range",
                    "ss.unit_size",
                    "ss.max_water_fraction",
                    "ss.tx",
                    "ss.cx",
                    "ss.ts",
                    "ss.cfr",
                    "p_corr.scale_factor",
                    "pt.albedo",
                    "pt.alpha",
                    "gm.dtf",
                    "routing.velocity",
                    "routing.alpha",
                    "routing.beta",
                    "gm.direct_response"
				};
                if (i >= size())
                    throw runtime_error("pt_ss_k parameter accessor:.get_name(i) Out of range.");
                return names[i];
            }

        };

        struct state {
            typedef skaugen::state snow_state_t;
            typedef kirchner::state kirchner_state_t;
            snow_state_t snow;
            kirchner_state_t kirchner;
            state() {}
            state(const snow_state_t& snow, const kirchner_state_t& kirchner)
             : snow(snow), kirchner(kirchner) { /* Do nothing */ }
            //xx state(const state& state) : snow(state.snow), kirchner(state.kirchner) {}
            bool operator==(const state& x) const {return kirchner==x.kirchner && snow==x.snow;}
            x_serialize_decl();
        };


        struct response {
            // Model responses
            typedef priestley_taylor::response  pt_response_t;
            typedef skaugen::response snow_response_t;
            typedef actual_evapotranspiration::response  ae_response_t;
            typedef kirchner::response kirchner_response_t;
            pt_response_t pt;
            snow_response_t snow;
            ae_response_t ae;
            kirchner_response_t kirchner;
            double gm_melt_m3s;
            // PTSSK response
            double total_discharge;
            double charge_m3s;
        };


        template<template <typename, typename> class A, class R, class T_TS, class P_TS,
                 class WS_TS, class RH_TS, class RAD_TS, class T, class S, class GCD,
                 class P, class SC, class RC>
        void run(const GCD& geo_cell_data,
            const P& parameter,
            const T& time_axis, int start_step,int  n_steps,
            const T_TS& temp,
            const P_TS& prec,
            const WS_TS& wind_speed,
            const RH_TS& rel_hum,
            const RAD_TS& rad,
            S& state,
            SC& state_collector,
            RC& response_collector
            ) {
            // Access time series input data through accessors of template A (typically a direct accessor).
            using temp_accessor_t = A<T_TS, T>;
            using prec_accessor_t = A<P_TS, T>;
            using rel_hum_accessor_t = A<RH_TS, T>;
            using rad_accessor_t = A<RAD_TS, T>;
            using ws_accessor_t = A<WS_TS, T>;

            auto temp_accessor = temp_accessor_t(temp, time_axis);
            auto prec_accessor = prec_accessor_t(prec, time_axis);
            auto rel_hum_accessor = rel_hum_accessor_t(rel_hum, time_axis);
            auto rad_accessor = rad_accessor_t(rad, time_axis);
            auto wind_speed_accessor = ws_accessor_t(wind_speed, time_axis);

            R response;
            const double glacier_fraction = geo_cell_data.land_type_fractions_info().glacier();
            const double gm_direct = parameter.gm.direct_response; //glacier melt directly out of cell
            const double gm_routed = 1-gm_direct; // glacier melt routed through kirchner
            const double direct_response_fraction = glacier_fraction*gm_direct + geo_cell_data.land_type_fractions_info().reservoir();// only direct response on reservoirs
            const double kirchner_fraction = 1 - direct_response_fraction;
            const double cell_area_m2 = geo_cell_data.area();
            const double glacier_area_m2 = geo_cell_data.area()*glacier_fraction;

            // Initialize the method stack
            precipitation_correction::calculator p_corr(parameter.p_corr.scale_factor);
            priestley_taylor::calculator pt(parameter.pt.albedo, parameter.pt.alpha);
            skaugen::calculator<typename P::snow_parameter_t, typename S::snow_state_t, typename R::snow_response_t> skaugen_snow;
            kirchner::calculator<kirchner::trapezoidal_average, typename P::kirchner_parameter_t> kirchner(parameter.kirchner);

            size_t i_begin = n_steps > 0 ? start_step : 0;
            size_t i_end = n_steps > 0 ? start_step + n_steps : time_axis.size();
            for (size_t i = i_begin; i < i_end; ++i) {
                utcperiod period = time_axis.period(i);
                double temp = temp_accessor.value(i);
                double rad = rad_accessor.value(i);
                double rel_hum = rel_hum_accessor.value(i);
                double prec = p_corr.calc(prec_accessor.value(i));
                double wind_speed = wind_speed_accessor.value(i);
                state_collector.collect(i, state);

                skaugen_snow.step(period.timespan(), parameter.ss, temp, prec, rad, wind_speed, state.snow, response.snow);
                response.gm_melt_m3s = glacier_melt::step(parameter.gm.dtf, temp, geo_cell_data.area()*state.snow.sca, glacier_area_m2);// m3/s, that is, how much flow from the snow free glacier parts
                response.pt.pot_evapotranspiration = pt.potential_evapotranspiration(temp, rad, rel_hum)*calendar::HOUR;// mm/s -> mm/h, interpreted as over the entire area(!)
                response.ae.ae = actual_evapotranspiration::calculate_step(state.kirchner.q, response.pt.pot_evapotranspiration,
                    parameter.ae.ae_scale_factor, std::max(state.snow.sca, glacier_fraction),  // a evap only on non-snow/non-glac area
                    period.timespan());
                double gm_mmh= shyft::m3s_to_mmh(response.gm_melt_m3s, cell_area_m2);
                kirchner.step(period.start, period.end, state.kirchner.q, response.kirchner.q_avg, response.snow.outflow + gm_routed*gm_mmh, response.ae.ae); //all units mm/h over 'same' area

                response.total_discharge =
                      std::max(0.0, prec - response.ae.ae)*direct_response_fraction // when it rains, remove ae. from direct response
                    + gm_direct*gm_mmh  // glacier melt direct response
                    + response.kirchner.q_avg*kirchner_fraction;
                response.charge_m3s =
                    + shyft::mmh_to_m3s(prec, cell_area_m2)
                    - shyft::mmh_to_m3s(response.ae.ae, cell_area_m2)
                    + response.gm_melt_m3s
                    - shyft::mmh_to_m3s(response.total_discharge, cell_area_m2);
                // Possibly save the calculated values using the collector callbacks.
                response_collector.collect(i, response);

                if(i+1==i_end)
                    state_collector.collect(i+1, state);///< \note last iteration,collect the  final state as well.
            }
            response_collector.set_end_response(response);
        }
    }
  }
}
//-- serialization support shyft
x_serialize_export_key(shyft::core::pt_ss_k::state);
