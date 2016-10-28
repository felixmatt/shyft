#pragma once

#include "priestley_taylor.h"
#include "kirchner.h"
#include "hbv_snow.h"
#include "glacier_melt.h"
#include "actual_evapotranspiration.h"
#include "precipitation_correction.h"

namespace shyft {
  namespace core {
    namespace pt_hs_k {
        using namespace std;

        struct parameter {
            typedef priestley_taylor::parameter pt_parameter_t;
            typedef hbv_snow::parameter snow_parameter_t;
            typedef actual_evapotranspiration::parameter ae_parameter_t;
            typedef kirchner::parameter kirchner_parameter_t;
            typedef precipitation_correction::parameter precipitation_correction_parameter_t;
            typedef glacier_melt::parameter glacier_parameter_t;

            pt_parameter_t pt;
            snow_parameter_t hs;
            ae_parameter_t ae;
            kirchner_parameter_t  kirchner;
            precipitation_correction_parameter_t p_corr;
            glacier_parameter_t gm;


            parameter(const pt_parameter_t& pt,
                        const snow_parameter_t& snow,
                        const ae_parameter_t& ae,
                        const kirchner_parameter_t& kirchner,
                        const precipitation_correction_parameter_t& p_corr,
                        glacier_parameter_t gm = glacier_parameter_t()) // for backwards compatability pass default glacier parameter
             : pt(pt), hs(snow), ae(ae), kirchner(kirchner), p_corr(p_corr), gm(gm) { /* Do nothing */ }
             			parameter(const parameter &c) : pt(c.pt), hs(c.hs), ae(c.ae), kirchner(c.kirchner), p_corr(c.p_corr), gm(c.gm) {}
			parameter(){}
			parameter& operator=(const parameter &c) {
                if(&c != this) {
                    pt = c.pt;
                    hs = c.hs;
                    gm = c.gm;
                    ae = c.ae;
                    kirchner = c.kirchner;
                    p_corr = c.p_corr;
                }
                return *this;
			}
            ///< Calibration support, size is the total number of calibration parameters
            size_t size() const { return 13; }

            void set(const vector<double>& p) {
                if (p.size() != size())
                    throw runtime_error("pt_ss_k parameter accessor: .set size missmatch");
                int i = 0;
                kirchner.c1 = p[i++];
                kirchner.c2 = p[i++];
                kirchner.c3 = p[i++];
                ae.ae_scale_factor = p[i++];
                hs.lw = p[i++];
                hs.tx = p[i++];
                hs.cx = p[i++];
                hs.ts = p[i++];
                hs.cfr = p[i++];
                gm.dtf = p[i++];
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
                    case  4:return hs.lw;
                    case  5:return hs.tx;
                    case  6:return hs.cx;
                    case  7:return hs.ts;
                    case  8:return hs.cfr;
                    case  9:return gm.dtf;
                    case 10:return p_corr.scale_factor;
					case 11:return pt.albedo;
					case 12:return pt.alpha;
                default:
                    throw runtime_error("pt_hs_k parameter accessor:.get(i) Out of range.");
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
                    "hs.lw",
                    "hs.tx",
                    "hs.cx",
                    "hs.ts",
                    "hs.cfr",
                    "gm.dtf",
                    "p_corr.scale_factor",
                    "pt.albedo",
                    "pt.alpha"
				};
                if (i >= size())
                    throw runtime_error("pt_hs_k parameter accessor:.get_name(i) Out of range.");
                return names[i];
            }

        };


        struct state {
            typedef hbv_snow::state snow_state_t;
            typedef kirchner::state kirchner_state_t;

            snow_state_t snow;
            kirchner_state_t kirchner;
            state() {}
            state(const snow_state_t& snow, const kirchner_state_t& kirchner)
             : snow(snow), kirchner(kirchner) { /* Do nothing */ }
            state(const state& state) : snow(state.snow), kirchner(state.kirchner) {}
            bool operator==(const state& x) const {return snow==x.snow && kirchner==x.kirchner;}
        };

        struct response {
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



        template<template <typename, typename> class A, class R, class T_TS, class P_TS, class WS_TS, class RH_TS, class RAD_TS, class T,
        class S, class GEOCELLDATA, class P, class SC, class RC >
        void run(const GEOCELLDATA& geo_cell_data,
            const P& parameter,
            const T& time_axis, int start_step, int  n_steps,
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
            //double q = state.kirchner.q;
            R response;

            // Initialize the method stack
            precipitation_correction::calculator p_corr(parameter.p_corr.scale_factor);
            priestley_taylor::calculator pt(parameter.pt.albedo, parameter.pt.alpha);
            hbv_snow::calculator<typename P::snow_parameter_t, typename S::snow_state_t> hbv_snow(parameter.hs, state.snow);
            kirchner::calculator<kirchner::trapezoidal_average, typename P::kirchner_parameter_t> kirchner(parameter.kirchner);

            // Step through times in axis
            size_t i_begin = n_steps > 0 ? start_step : 0;
            size_t i_end = n_steps > 0 ? start_step + n_steps : time_axis.size();
            for (size_t i = i_begin; i < i_end; ++i) {
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
                response.pt.pot_evapotranspiration = pt.potential_evapotranspiration(temp, rad, rel_hum)*calendar::HOUR;// mm/h

                // HBVSnow
                hbv_snow.step(state.snow, response.snow, period.start, period.end, parameter.hs, prec, temp);

                // Communicate snow
                // At my pos xx mm of snow moves in direction d.

                // Glacier Melt
                double outflow = glacier_melt::step(period.timespan(), parameter.gm.dtf, temp, state.snow.sca, geo_cell_data.land_type_fractions_info().glacier());
                outflow += response.snow.outflow;

                // Actual Evapotranspiration
                response.ae.ae = actual_evapotranspiration::calculate_step(state.kirchner.q, response.pt.pot_evapotranspiration, parameter.ae.ae_scale_factor, state.snow.sca, period.timespan());

                // Use responses from PriestleyTaylor and HBVSnow in Kirchner
                kirchner.step(period.start, period.end, state.kirchner.q, response.kirchner.q_avg, outflow, response.ae.ae);

                //
                // Adjust land response for lakes and reservoirs (Treat them the same way for now)
                //
                double total_lake_fraction = geo_cell_data.land_type_fractions_info().lake() + geo_cell_data.land_type_fractions_info().reservoir();
                double lake_corrected_discharge = prec*total_lake_fraction + (1.0 - total_lake_fraction)*response.kirchner.q_avg;

                response.total_discharge = lake_corrected_discharge;
                response.snow.snow_state=state.snow;//< note/sih: we need snow in the response due to calibration

                // Possibly save the calculated values using the collector callbacks.
                response_collector.collect(i, response);
                if(i+1==i_end)
                    state_collector.collect(i+1, state);///< \note last iteration,collect the  final state as well.

            }
            response_collector.set_end_response(response);
        }
    }
  } // core
} // shyft
