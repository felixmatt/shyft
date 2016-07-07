#pragma once

#include "priestley_taylor.h"
#include "kirchner.h"
#include "skaugen.h"
#include "actual_evapotranspiration.h"
#include "precipitation_correction.h"


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
            pt_parameter_t pt;
            snow_parameter_t ss;
            ae_parameter_t ae;
            kirchner_parameter_t  kirchner;
            precipitation_correction_parameter_t p_corr;

            parameter(pt_parameter_t pt,
                      snow_parameter_t snow,
                      ae_parameter_t ae,
                      kirchner_parameter_t kirchner,
                      precipitation_correction_parameter_t p_corr)
             : pt(pt), ss(snow), ae(ae), kirchner(kirchner), p_corr(p_corr) { /* Do nothing */ }

			parameter(const parameter &c) : pt(c.pt), ss(c.ss), ae(c.ae), kirchner(c.kirchner), p_corr(c.p_corr) {}
			parameter(){}
			parameter& operator=(const parameter &c) {
                if(&c != this) {
                    pt = c.pt;
                    ss = c.ss;
                    ae = c.ae;
                    kirchner = c.kirchner;
                    p_corr = c.p_corr;
                }
                return *this;
			}
            ///< Calibration support, size is the total number of calibration parameters
            size_t size() const { return 15; }

            void set(const vector<double>& p) {
                if (p.size() != size())
                    throw runtime_error("pt_ss_k parameter accessor: .set size missmatch");
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
                    "pt.alpha"
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
            state(const state& state) : snow(state.snow), kirchner(state.kirchner) {}
            bool operator==(const state& x) const {return kirchner==x.kirchner && snow==x.snow;}
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

            // PTSSK response
            double total_discharge;
        };


        template<template <typename, typename> class A, class R, class T_TS, class P_TS,
                 class WS_TS, class RH_TS, class RAD_TS, class T, class S, class GCD,
                 class P, class SC, class RC>
        void run(const GCD& geo_cell_data,
            const P& parameter,
            const T& time_axis,
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

            // Get the initial states
            auto &snow_state = state.snow;
            double q = state.kirchner.q;
            R response;

            // Initialize the method stack
            precipitation_correction::calculator p_corr(parameter.p_corr.scale_factor);
            priestley_taylor::calculator pt(parameter.pt.albedo, parameter.pt.alpha);
            skaugen::calculator<typename P::snow_parameter_t, typename S::snow_state_t, typename R::snow_response_t> skaugen_snow;
            kirchner::calculator<kirchner::trapezoidal_average, typename P::kirchner_parameter_t> kirchner(parameter.kirchner);
            // Step through times in axis
            for (size_t i=0; i < time_axis.size(); ++i) {
                utcperiod period = time_axis.period(i);
                double temp = temp_accessor.value(i);
                double rad = rad_accessor.value(i);
                double rel_hum = rel_hum_accessor.value(i);
                double prec = p_corr.calc(prec_accessor.value(i));
                double wind_speed = wind_speed_accessor.value(i);
                state_collector.collect(i, state);
                //
                // Land response:
                //

                // PriestleyTaylor (scale by timespan since it delivers results in mm/s)
                double pot_evap = pt.potential_evapotranspiration(temp, rad, rel_hum)*period.timespan()/calendar::HOUR;
                response.pt.pot_evapotranspiration = pot_evap;

                // Snow
                skaugen_snow.step(period.timespan(), parameter.ss, temp, prec, rad, wind_speed, snow_state, response.snow);

                // TODO: Snow transport
                // At my pos xx mm of snow moves in direction d.

                // Actual Evapotranspiration
                double act_evap = actual_evapotranspiration::calculate_step(q, pot_evap, parameter.ae.ae_scale_factor, state.snow.sca, period.timespan());
                response.ae.ae = act_evap;

                // Use responses from PriestleyTaylor and Snow in Kirchner
                double q_avg;
                kirchner.step(period.start, period.end, q, q_avg, response.snow.outflow, act_evap);
                state.kirchner.q = q; // Save discharge state variable
                response.kirchner.q_avg = q_avg;

                //
                // Adjust land response for lakes and reservoirs (Treat them the same way for now)
                double total_lake_fraction = geo_cell_data.land_type_fractions_info().lake() +
                    geo_cell_data.land_type_fractions_info().reservoir();
                double lake_corrected_discharge = prec*total_lake_fraction +
                    (1.0 - total_lake_fraction)*q_avg;
                response.total_discharge = lake_corrected_discharge;

                // Possibly save the calculated values using the collector callbacks.
                response_collector.collect(i, response);

                if(i+1==time_axis.size())
                    state_collector.collect(i+1, state);///< \note last iteration,collect the  final state as well.
            }
            response_collector.set_end_response(response);
        }
    } // pt_hs_k
  } // core
} // shyft
