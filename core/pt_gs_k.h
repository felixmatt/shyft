#pragma once

#include "priestley_taylor.h"
#include "kirchner.h"
#include "gamma_snow.h"
#include "actual_evapotranspiration.h"
#include "precipitation_correction.h"


namespace shyft {
  namespace core {
    namespace pt_gs_k {
        using namespace std;

        /** \brief Simple parameter struct for the PTGSK method stack
         *
         * This struct contains the parameters to the methods used in the PTGSK assembly.
         *
         * \tparam PTParameter PriestleyTaylor parameter type that implements the interface:
         *    - PTParameter.albedo const --> double, land albedo parameter in PriestleyTaylor.
         *    - PTParameter.alpha const --> double, alpha parameter in PriestleyTaylor.
         * \tparam GSState GammaSnow parameter type that implements the parameter interface for GammaSnow.
         * \tparam KState Kirchner parameter type that implements the parameter interface for Kirchner.
         * \sa GammaSnowParameter \sa Kirchner \sa PTGSK  \sa PriestleyTaylor
         */
        struct parameter {
            typedef priestley_taylor::parameter pt_parameter_t;
            typedef gamma_snow::parameter gs_parameter_t;
            typedef actual_evapotranspiration::parameter ae_parameter_t;
            typedef kirchner::parameter kirchner_parameter_t;
            typedef precipitation_correction::parameter precipitation_correction_parameter_t;

            parameter(pt_parameter_t pt,
                      gs_parameter_t gs,
                      ae_parameter_t ae,
                      kirchner_parameter_t k,
                      precipitation_correction_parameter_t p_corr)
             : pt(pt), gs(gs), ae(ae), kirchner(k), p_corr(p_corr) { /*Do nothing */ }
            parameter() {}
            parameter(const parameter& other)
             : pt(other.pt), gs(other.gs), ae(other.ae),
               kirchner(other.kirchner), p_corr(other.p_corr) { /*Do nothing */ }

            pt_parameter_t pt;
            gs_parameter_t gs;
            ae_parameter_t ae;
            kirchner_parameter_t  kirchner;
            precipitation_correction_parameter_t p_corr;
            ///<calibration support, needs vector interface to params, size is the total count
            size_t size() const { return 21; }
            ///<calibration support, need to set values from ordered vector
            void set(const vector<double>& p) {
                if (p.size() != size())
                    throw runtime_error("PTGSK Parameter Accessor: .set size missmatch");
                int i = 0;
                kirchner.c1 = p[i++];
                kirchner.c2 = p[i++];
                kirchner.c3 = p[i++];
                ae.ae_scale_factor = p[i++];
                gs.tx = p[i++];
                gs.wind_scale = p[i++];
                gs.max_water = p[i++];
                gs.wind_const = p[i++];
                gs.fast_albedo_decay_rate = p[i++];
                gs.slow_albedo_decay_rate = p[i++];
                gs.surface_magnitude = p[i++];
                gs.max_albedo = p[i++];
                gs.min_albedo = p[i++];
                gs.snowfall_reset_depth = p[i++];
                gs.snow_cv = p[i++];
                gs.glacier_albedo = p[i++];
                p_corr.scale_factor = p[i++];
                gs.snow_cv_forest_factor=p[i++];
                gs.snow_cv_altitude_factor=p[i++];
				pt.albedo = p[i++];
				pt.alpha = p[i++];
            }

            ///< calibration support, get the value of i'th parameter
            double get(size_t i) const {
                switch (i) {
                    case  0:return kirchner.c1;
                    case  1:return kirchner.c2;
                    case  2:return kirchner.c3;
                    case  3:return ae.ae_scale_factor;
                    case  4:return gs.tx;
                    case  5:return gs.wind_scale;
                    case  6:return gs.max_water;
                    case  7:return gs.wind_const;
                    case  8:return gs.fast_albedo_decay_rate;
                    case  9:return gs.slow_albedo_decay_rate;
                    case 10:return gs.surface_magnitude;
                    case 11:return gs.max_albedo;
                    case 12:return gs.min_albedo;
                    case 13:return gs.snowfall_reset_depth;
                    case 14:return gs.snow_cv;
                    case 15:return gs.glacier_albedo;
                    case 16:return p_corr.scale_factor;
                    case 17:return gs.snow_cv_forest_factor;
                    case 18:return gs.snow_cv_altitude_factor;
					case 19:return pt.albedo;
					case 20:return pt.alpha;
                default:
                    throw runtime_error("PTGSK Parameter Accessor:.get(i) Out of range.");
                }
                return 0;
            }

            ///< calibration and python support, get the i'th parameter name
            string get_name(size_t i) const {
                static const char *names[] = {
                    "kirchner.c1",
                    "kirchner.c2",
                    "kirchner.c3",
                    "ae.ae_scale_factor",
                    "gs.tx",
                    "gs.wind_scale",
                    "gs.max_water",
                    "gs.wind_const",
                    "gs.fast_albedo_decay_rate",
                    "gs.slow_albedo_decay_rate",
                    "gs.surface_magnitude",
                    "gs.max_albedo",
                    "gs.min_albedo",
                    "gs.snowfall_reset_depth",
                    "gs.snow_cv",
                    "gs.glacier_albedo",
                    "p_corr.scale_factor",
                    "gs.snow_cv_forest_factor",
                    "gs.snow_cv_altitude_factor",
                    "pt.albedo",
                    "pt.alpha"
                };
                if (i >= size())
                    throw runtime_error("PTGSK Parameter Accessor:.get_name(i) Out of range.");
                return names[i];
            }
        };

        /** \brief Simple state struct for the PTGSK method stack
         *
         * This struct contains the states of the methods used in the PTGSK assembly.
         *
         * \tparam GSState GammaSnow state type that implements the state interface for GammaSnow.
         * \tparam KState Kirchner state type that implements the state interface for Kirchner.
         * \sa GammaSnowState \sa Kirchner \sa PTGSK \sa PriestleyTaylor \sa GammaSnow
         */
        struct state {
            typedef gamma_snow::state gs_state_t;
            typedef kirchner::state kirchner_state_t;
            state() {}
            state(gs_state_t gs, kirchner_state_t k) : gs(gs), kirchner(k) {}
            gs_state_t gs;
            kirchner_state_t kirchner;
            bool operator==(const state& x) const {return gs==x.gs && kirchner==x.kirchner;}
        };


        /** \brief Simple response struct for the PTGSK method stack
         *
         * This struct contains the responses of the methods used in the PTGSK assembly.
         */
        struct response {
            // Model responses
            typedef priestley_taylor::response  pt_response_t;
            typedef gamma_snow::response gs_response_t;
            typedef actual_evapotranspiration::response  ae_response_t;
            typedef kirchner::response kirchner_response_t;
            pt_response_t pt;
            gs_response_t gs;
            ae_response_t ae;
            kirchner_response_t kirchner;

            // Stack response
            double total_discharge;
        };

        /** \brief Calculation Model using assembly of PriestleyTaylor, GammaSnow and Kirchner
         *
         * This model first uses PriestleyTaylor for calculating the potential
         * evapotranspiration based on time series data for temperature,
         * radiation and relative humidity. Then it uses the GammaSnow method
         * to calculate the snow/ice adjusted runoff using time series data for
         * precipitation and wind speed in addition to the time series used in
         * the PriestleyTaylor method. The PriestleyTaylor potential evaporation is
         * is used to calculate the actual evapotranspiration that is passed on to the
         * last step, Kirchner.
         * Kirchner is run with the gamma snow output
         * and actual evaporation response from the two methods above to
         * calculate the discharge.
         *
         * TODO: This method should construct an internal time stepping ptgsk struct.
         * This stepper should be used as an iterator in time integration loops. This
         * paves the way for inter-cell communication (e.g. snow transport) without
         * touching this simple interface.
         *
         * \tparam TS Time series type that implements:
         *    - TS::source_accessor_type --> Type of accessor used to retrieve time series data.
         *    - TS.accessor(const T& time_axis) const --> TS::source_accessor_type. Accessor object
         *      for this time series.
         *
         * \tparam TS::source_accessor_type Time series source accessor type that implements:
         *    - TS::source_accessor_type(const TS& time_series, const T& time_axis) --> Construct
         *      accessor for the given time series and time axis.
         *    - TS::source_accessor_type.value(size_t i) --> double, -value of the time series for
         *      period i in the time axis.
         * \tparam T Time axis type that implements:
         *    - T.size() const --> Number of periods in the time axis.
         *    - T(size_t i) const --> shyft::core::utcperiod, - Start and end as shyft::core::utctime
         *      of the i'th period.
         * \tparam S State type that implements:
         *    - S::gs_state_type --> State type for the GammaSnow method.
         *    - S::kirchner_state_type --> State type for the Kirchner method.
         *    - S.gs --> S::gs_state_type, - State variables for the GammaSnow method
         *    - S.kirchner --> S::kirchner_state_type, - State variables for the Kirchner method
         * \tparam R Response type that implements:
         *    - R::gs_response_type --> Response type for the GammaSnow routine.
         *    - R.gs --> R::gs_response_type, -Response object passed to the GammaSnow routine.
         * \tparam P Parameter type that implements:
         *    - P::pt_parameter_type --> Parameter type for the PriestleyTaylor method
         *    - P::gs_parameter_type --> Parameter type for the GammaSnow method.
         *    - P::ae_parameter_type --> Parameter type for the ActualEvapotranspiration method.
         *    - P::kirchner_parameter_type --> Parameter type for the Kirchner method.
         *    - P.pt --> P::pt_parameter_type --> Parameters for the PriestleyTaylor method.
         *    - P.gs --> P::gs_parameter_type --> Parameters for thge GammaSnow method.
         *    - P.ae --> P::ae_parameter_type --> Parameters for thge ActualEvapotranspiration method.
         *    - P.kirchner --> P::kirchner_parameter_type --> Parameters for the Kirchner method.
         * \tparam SC State collector type that implements:
         *    - SC.collect(utctime t, const S& state) --> Possibly save some states at time t.
         * \tparam RC Response collector type that implements:
         *    - RC.collect(utctime t, const R& response) --> Possibly save some responses at time t.
         */
#ifndef SWIG
        template<template <typename, typename> class A, class R, class T_TS, class P_TS, class WS_TS, class RH_TS, class RAD_TS, class T,
        class S, class GCD, class P, class SC, class RC>
        void run_pt_gs_k(const GCD& geo_cell_data,
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
            using wind_speed_accessor_t = A<WS_TS, T>;
            using rel_hum_accessor_t = A<RH_TS, T>;
            using rad_accessor_t = A<RAD_TS, T>;

            auto temp_accessor = temp_accessor_t(temp, time_axis);
            auto prec_accessor = prec_accessor_t(prec, time_axis);
            auto wind_speed_accessor = wind_speed_accessor_t(wind_speed, time_axis);
            auto rel_hum_accessor = rel_hum_accessor_t(rel_hum, time_axis);
            auto rad_accessor = rad_accessor_t(rad, time_axis);

            // Initialize the method stack
            precipitation_correction::calculator p_corr(parameter.p_corr.scale_factor);
            priestley_taylor::calculator pt(parameter.pt.albedo, parameter.pt.alpha);
            gamma_snow::calculator<typename P::gs_parameter_t, typename S::gs_state_t, typename R::gs_response_t> gs;
            kirchner::calculator<kirchner::trapezoidal_average, typename P::kirchner_parameter_t> kirchner(parameter.kirchner);
            //
            gs.set_glacier_fraction(geo_cell_data.land_type_fractions_info().glacier());
            // Get the initial states
            auto &gs_state = state.gs;
            double q = state.kirchner.q;
            R response;
            const double forest_fraction=geo_cell_data.land_type_fractions_info().forest();
            const double altitude= geo_cell_data.mid_point().z;
            // Step through times in axis
            for (size_t i = 0; i < time_axis.size(); ++i) {
                utcperiod period = time_axis.period(i);
                double temp = temp_accessor.value(i);
                double rad = rad_accessor.value(i);
                double rel_hum = rel_hum_accessor.value(i);
                double prec = p_corr.calc(prec_accessor.value(i));
                state_collector.collect(i, state);///< \note collect the state at the beginning of each period (the end state is saved anyway)
                //
                // Land response:
                //

                // PriestleyTaylor (scale by timespan since it delivers results in mm/s)
                double pot_evap = pt.potential_evapotranspiration(temp, rad, rel_hum)*period.timespan();
                response.pt.pot_evapotranspiration=pot_evap;

                // Gamma Snow
                gs.step(gs_state, response.gs, period.start, period.timespan(), parameter.gs,
                        temp, rad, prec, wind_speed_accessor.value(i), rel_hum,forest_fraction,altitude);

                // TODO: Communicate snow
                // At my pos xx mm of snow moves in direction d.

                // Actual Evapotranspiration
                double act_evap = actual_evapotranspiration::calculate_step(q, pot_evap,
                                  parameter.ae.ae_scale_factor, response.gs.sca, period.timespan());
                response.ae.ae = act_evap;

                // Use responses from PriestleyTaylor and GammaSnow in Kirchner
                double q_avg;
                kirchner.step(period.start, period.end, q, q_avg, response.gs.outflow, act_evap);
                state.kirchner.q = q; // Save discharge state variable
                response.kirchner.q_avg = q_avg;

                //
                // Adjust land response for lakes and reservoirs (Treat them the same way for now)
                double total_lake_fraction = geo_cell_data.land_type_fractions_info().lake() +
                    geo_cell_data.land_type_fractions_info().reservoir();
                double corrected_discharge = prec*total_lake_fraction +
                    response.gs.outflow*geo_cell_data.land_type_fractions_info().glacier() +
                    (1.0 - total_lake_fraction - geo_cell_data.land_type_fractions_info().glacier())*q_avg;
                response.total_discharge = corrected_discharge;

                // Possibly save the calculated values using the collector callbacks.
                response_collector.collect(i, response);///< \note collect the response valid for the i'th period (current state is now at the end of period)
                if(i+1==time_axis.size())
                    state_collector.collect(i+1, state);///< \note last iteration,collect the  final state as well.
            }
            response_collector.set_end_response(response);
        }
#endif
    } // pt_gs_k
  } // core
} // shyft
