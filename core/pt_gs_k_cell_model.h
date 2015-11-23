#pragma once
#include "cell_model.h"
#include "pt_gs_k.h"

namespace shyft {
    namespace core {
        /** \brief pt_gs_k namespace declares the needed pt_gs_k specific parts of a cell
        * that is:
        * -# the pt_gs_k parameter, state and response types
        * -# the response collector variants
        */
        namespace pt_gs_k {
            using namespace std;
            typedef parameter parameter_t;
            typedef state state_t;
            typedef response response_t;
            typedef shared_ptr<parameter_t> parameter_t_;
            typedef shared_ptr<state_t>     state_t_;
            typedef shared_ptr<response_t>  response_t_;

            /** \brief all_reponse_collector aims to collect all output from a cell run so that it can be studied afterwards.
            *
            * \note could be quite similar between variants of a cell, e.g. ptgsk pthsk ptssk, ptss..
            * TODO: could we use another approach to limit code pr. variant ?
            * TODO: Really make sure that units are correct, precise and useful..
            *       both with respect to measurement unit, and also specifying if this
            *       a 'state in time' value or a average-value for the time-step.
            */
            struct all_response_collector {
                double destination_area;///< in [m²]
                // these are the one that we collects from the response, to better understand the model::
                pts_t avg_discharge; ///< Kirchner Discharge given in [m³/s] for the timestep
                pts_t snow_sca; ///< gamma snow covered area fraction, sca.. 0..1 - at the end of timestep (state)
                pts_t snow_swe;///< gamma snow swe, [mm] over the cell sca.. area, - at the end of timestep ?
                pts_t snow_output;///< gamma snow output [m³/s] for the timestep
                pts_t ae_output;///< actual evap mm/h
                pts_t pe_output;///< actual evap mm/h
                response_t end_reponse;///<< end_response, at the end of collected

                all_response_collector() : destination_area(0.0) {}
                all_response_collector(const double destination_area) : destination_area(destination_area) {}
                all_response_collector(const double destination_area, const timeaxis_t& time_axis)
                    : destination_area(destination_area), avg_discharge(time_axis, 0.0), snow_sca(time_axis, 0.0), snow_swe(time_axis, 0.0), snow_output(time_axis, 0.0), ae_output(time_axis, 0.0), pe_output(time_axis, 0.0) {}

                /**\brief called before run to allocate space for results */
                void initialize(const timeaxis_t& time_axis, double area) {
                    destination_area = area;
                    avg_discharge = pts_t(time_axis, 0.0);
                    snow_sca = pts_t(time_axis, 0.0);
                    snow_swe = pts_t(time_axis, 0.0);
                    snow_output = pts_t(time_axis, 0.0);
                    ae_output = pts_t(time_axis, 0.0);
                    pe_output = pts_t(time_axis, 0.0);
                }

                /**\brief Call for each time step, to collect needed information from R
                 *
                 * The R in this case is the Response type defined for the pt_g_s_k stack
                 * and in principle, we can pick out all needed information from this.
                 * The values are put into the plain point time-series at the i'th position
                 * corresponding to the i'th simulation step, .. on the time-axis.. that
                 * again gives us the concrete period in time that this value applies to.
                 *
                 */
                //template<class R>
                void collect(size_t idx, const response_t& response) {
                    avg_discharge.set(idx, mmh_to_m3s(response.total_discharge, destination_area)); // wants m3/s, q_avg is given in mm/h, so compute the totals in  mm/s
                    snow_sca.set(idx, response.gs.sca);
                    snow_output.set(idx, response.gs.outflow); //TODO: current mm/h.. but  want m3/s, but we get mm/h from snow output
                    snow_swe.set(idx, response.gs.storage);
                    ae_output.set(idx, response.ae.ae);
                    pe_output.set(idx, response.pt.pot_evapotranspiration);
                }
                //template<class R>
                void set_end_response(const response_t& r) {end_reponse=r;}
            };

            /** \brief a collector that collects/keep discharge only */
            struct discharge_collector {
                double destination_area;
                pts_t avg_discharge; ///< Discharge given in [m³/s] as the average of the timestep
                response_t end_response;///<< end_response, at the end of collected
                discharge_collector() : destination_area(0.0) {}
                discharge_collector(const double destination_area) : destination_area(destination_area) {}
                discharge_collector(const double destination_area, const timeaxis_t& time_axis)
                    : destination_area(destination_area), avg_discharge(time_axis, 0.0) {}

                void initialize(const timeaxis_t& time_axis, double area) {
                    destination_area = area;
                    avg_discharge = pts_t(time_axis, 0.0);
                }


                void collect(size_t idx, const response_t& response) {
                    avg_discharge.set(idx, mmh_to_m3s(response.total_discharge, destination_area)); // q_avg is given in mm, so compute the totals
                }

                void set_end_response(const response_t& response) {end_response=response;}
            };
            /**\brief a state null collector
             *
             * Used during calibration/optimization when there is no need for state,
             * and we need all the RAM for useful purposes.
             */
            struct null_collector {
                void initialize(const timeaxis_t& time_axis, double area=0.0) {}
                void collect(size_t i, const state_t& response) {}
            };

            /** \brief the state_collector collects all state if enabled
             *
             *  \note that the state collected is instant in time, valid at the beginning of period
             */
            struct state_collector {
                bool collect_state;///< if true, collect state, otherwise ignore (and the state of time-series are undefined/zero)
                // these are the one that we collects from the response, to better understand the model::
                double destination_area;
                pts_t kirchner_discharge; ///< Kirchner state instant Discharge given in m^3/s
                pts_t gs_albedo;
                pts_t gs_lwc;
                pts_t gs_surface_heat;
                pts_t gs_alpha;
                pts_t gs_sdc_melt_mean;
                pts_t gs_acc_melt;
                pts_t gs_iso_pot_energy;
                pts_t gs_temp_swe;

                state_collector() : collect_state(false), destination_area(0.0) { /* Do nothing */ }
                state_collector(const timeaxis_t& time_axis)
                 : collect_state(false), destination_area(0.0),
                   kirchner_discharge(time_axis, 0.0),
                   gs_albedo(time_axis, 0.0),
                   gs_lwc(time_axis, 0.0),
                   gs_surface_heat(time_axis, 0.0),
                   gs_alpha(time_axis, 0.0),
                   gs_sdc_melt_mean(time_axis, 0.0),
                   gs_acc_melt(time_axis, 0.0),
                   gs_iso_pot_energy(time_axis, 0.0),
                   gs_temp_swe(time_axis, 0.0)
                   { /* Do nothing */ }

                /** brief called before run, prepares state time-series
                 *
                 * with preallocated room for the supplied time-axis.
                 *
                 * \note if collect_state is false, a zero length time-axis is used to ensure
                 * data is wiped/out.
                 */
                void initialize(const timeaxis_t& time_axis, double area) {
                    destination_area = area;
                    timeaxis_t ta = collect_state ? time_axis : timeaxis_t(time_axis.start(), time_axis.delta(), 0);
                    kirchner_discharge = pts_t(ta, 0.0);
                    gs_albedo = pts_t(ta, 0.0);
                    gs_lwc = pts_t(ta, 0.0);
                    gs_surface_heat = pts_t(ta, 0.0);
                    gs_alpha = pts_t(ta, 0.0);
                    gs_sdc_melt_mean = pts_t(ta, 0.0);
                    gs_acc_melt = pts_t(ta, 0.0);
                    gs_iso_pot_energy = pts_t(ta, 0.0);
                    gs_temp_swe = pts_t(ta, 0.0);
                }
                /** called by the cell.run for each new state*/
                void collect(size_t idx, const state_t& state) {
                    if (collect_state) {
                        kirchner_discharge.set(idx, mmh_to_m3s(state.kirchner.q, destination_area));
                        gs_albedo.set(idx, state.gs.albedo);
                        gs_lwc.set(idx, state.gs.lwc);
                        gs_surface_heat.set(idx, state.gs.surface_heat);
                        gs_alpha.set(idx, state.gs.alpha);
                        gs_sdc_melt_mean.set(idx, state.gs.sdc_melt_mean);
                        gs_acc_melt.set(idx, state.gs.acc_melt);
                        gs_iso_pot_energy.set(idx, state.gs.iso_pot_energy);
                        gs_temp_swe.set(idx, state.gs.temp_swe);
                    }
                }
            };
            // typedef the variants we need exported.
            typedef cell<parameter_t, environment_t, state_t, state_collector, all_response_collector> cell_complete_response_t;///< used for usual/explorative runs, where we would like all possible info, result and state
            typedef cell<parameter_t, environment_t, state_t, null_collector, discharge_collector> cell_discharge_response_t; ///<used for operational or calibration runs, only needed info is collected.

        }

        //specialize run method for all_response_collector
        template<>
        inline void cell<pt_gs_k::parameter_t, environment_t, pt_gs_k::state_t,
                         pt_gs_k::state_collector, pt_gs_k::all_response_collector>
            ::run(const timeaxis_t& time_axis) {
            if (parameter.get() == nullptr)
                throw std::runtime_error("pt_gs_k::run with null parameter attempted");
            begin_run(time_axis);
            pt_gs_k::run_pt_gs_k<direct_accessor, pt_gs_k::response_t>(
                geo,
                *parameter,
                time_axis,
                env_ts.temperature,
                env_ts.precipitation,
                env_ts.wind_speed,
                env_ts.rel_hum,
                env_ts.radiation,
                state,
                sc,
                rc);
        }
        template<>
        inline void cell<pt_gs_k::parameter_t, environment_t, pt_gs_k::state_t,
                         pt_gs_k::state_collector, pt_gs_k::all_response_collector>
            ::set_state_collection(bool on_or_off) {
            sc.collect_state = on_or_off;
        }
        //specialize run method for discharge_collector
        template<>
        inline void cell<pt_gs_k::parameter_t, environment_t, pt_gs_k::state_t,
                         pt_gs_k::null_collector, pt_gs_k::discharge_collector>
            ::run(const timeaxis_t& time_axis) {
            if (parameter.get() == nullptr)
                throw std::runtime_error("pt_gs_k::run with null parameter attempted");
            begin_run(time_axis);
            pt_gs_k::run_pt_gs_k<direct_accessor, pt_gs_k::response_t>(
                geo,
                *parameter,
                time_axis,
                env_ts.temperature,
                env_ts.precipitation,
                env_ts.wind_speed,
                env_ts.rel_hum,
                env_ts.radiation,
                state,
                sc,
                rc);
        }
        template<>
        inline void cell<pt_gs_k::parameter_t, environment_t, pt_gs_k::state_t,
                         pt_gs_k::null_collector, pt_gs_k::discharge_collector>
            ::set_state_collection(bool on_or_off) {
            /* not possible always off.*/
        }
    }
}
