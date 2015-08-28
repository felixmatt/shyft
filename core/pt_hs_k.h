#pragma once

#include "priestley_taylor.h"
#include "kirchner.h"
#include "hbv_snow.h"
#include "actual_evapotranspiration.h"
#include "precipitation_correction.h"

namespace shyft {
  namespace core {
    namespace pt_hs_k {

        struct parameter_t {
            typedef priestley_taylor::parameter pt_parameter_t;
            typedef hbv_snow::parameter snow_parameter_t;
            typedef actual_evapotranspiration::parameter ae_parameter_t;
            typedef kirchner::parameter kirchner_parameter_t;
            typedef precipitation_correction::parameter precipitation_correction_parameter_t;

            pt_parameter_t pt;
            snow_parameter_t snow;
            ae_parameter_t ae;
            kirchner_parameter_t  kirchner;
            precipitation_correction_parameter_t p_corr;

            parameter_t(pt_parameter_t& pt,
                        snow_parameter_t& snow,
                        ae_parameter_t& ae,
                        kirchner_parameter_t& kirchner,
                        precipitation_correction_parameter_t p_corr)
             : pt(pt), snow(snow), ae(ae), kirchner(kirchner), p_corr(p_corr) { /* Do nothing */ }
        };


        struct state_t {
            typedef hbv_snow::state snow_state_t;
            typedef kirchner::state kirchner_state_t;

            snow_state_t snow;
            kirchner_state_t kirchner;
            state_t() {}
            state_t(snow_state_t& snow, kirchner_state_t& kirchner)
             : snow(snow), kirchner(kirchner) { /* Do nothing */ }
            state_t(const state_t& state) : snow(state.snow), kirchner(state.kirchner) {}
        };

        struct response_t {
            typedef priestley_taylor::response pt_response_t;
            typedef hbv_snow::response snow_response_t;
            typedef actual_evapotranspiration::response ae_response_t;
            typedef kirchner::response kirchner_response_t;
            pt_response_t pt;
            snow_response_t snow;
            ae_response_t ae;
            kirchner_response_t kirchner;

            // Stack response
            double total_discharge;
        };


#ifndef SWIG
        template<template <typename, typename> class A, class R, class T_TS, class P_TS, class WS_TS, class RH_TS, class RAD_TS, class T,
        class S, class GEOCELLDATA, class P, class SC, class RC >
        void run(const GEOCELLDATA& geo_cell_data,
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
            using temp_accessor_t = A < T_TS, T > ;
            using prec_accessor_t = A < P_TS, T > ;
            using rel_hum_accessor_t = A < RH_TS, T > ;
            using rad_accessor_t = A < RAD_TS, T > ;

            auto temp_accessor = temp_accessor_t(temp, time_axis);
            auto prec_accessor = prec_accessor_t(prec, time_axis);
            auto rel_hum_accessor = rel_hum_accessor_t(rel_hum, time_axis);
            auto rad_accessor = rad_accessor_t(rad, time_axis);

            // Get the initial states
            auto &snow_state = state.snow;
            double q = state.kirchner.q;
            R response;

            // Initialize the method stack
            precipitation_correction::calculator p_corr(parameter.p_corr.scale_factor);
            priestley_taylor::calculator pt(parameter.pt.albedo, parameter.pt.alpha);
            hbv_snow::calculator<typename P::snow_parameter_t, typename S::snow_state_t> hbv_snow(parameter.snow, state.snow);
            kirchner::calculator<kirchner::trapezoidal_average, typename P::kirchner_parameter_t> kirchner(parameter.kirchner);

            // Step through times in axis
            for (size_t i = 0; i < time_axis.size(); ++i) {
                utcperiod period = time_axis(i);
                double temp = temp_accessor.value(i);
                double rad = rad_accessor.value(i);
                double rel_hum = rel_hum_accessor.value(i);
                double prec = p_corr.calc(prec_accessor.value(i));

                //
                // Land response:
                //

                // PriestleyTaylor (scale by timespan since it delivers results in mm/s)
                double pot_evap = pt.potential_evapotranspiration(temp, rad, rel_hum)*period.timespan();
                response.pt.pot_evapotranspiration=pot_evap;

                // HBVSnow
                hbv_snow.step(snow_state, response.snow, period.start, period.end, parameter.snow, prec, temp);
                response.snow.sca = snow_state.sca;
                response.snow.swe = snow_state.swe; // TODO: Should swe really be a state variable?

                // Communicate snow
                // At my pos xx mm of snow moves in direction d.

                // Actual Evapotranspiration
                double act_evap = actual_evapotranspiration::calculate_step(q, pot_evap, parameter.ae.ae_scale_factor, state.snow.sca, period.timespan());
                response.ae.ae=act_evap;

                // Use responses from PriestleyTaylor and HBVSnow in Kirchner
                double q_avg;
                kirchner.step(period.start, period.end, q, q_avg, response.snow.outflow, act_evap);
                state.kirchner.q=q; // Save discharge state variable
                response.kirchner.q_avg=q_avg;

                //
                // Adjust land response for lakes and reservoirs (Treat them the same way for now)
                //
                double total_lake_fraction = geo_cell_data.land_type_fractions_info().lake() + geo_cell_data.land_type_fractions_info().reservoir();
                double lake_corrected_discharge = prec*total_lake_fraction + (1.0 - total_lake_fraction)*q_avg;

                response.total_discharge = lake_corrected_discharge;

                // Possibly save the calculated values using the collector callbacks.
                response_collector.collect(i, response);
                state_collector.collect(i, state);
            }
            response_collector.set_end_response(response);
        }
#endif
    }
  } // core
} // shyft
