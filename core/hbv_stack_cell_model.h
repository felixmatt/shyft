#pragma once
#ifdef SHYFT_NO_PCH
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <algorithm>
#include <stdexcept>

#include "core_pch.h"
#endif // SHYFT_NO_PCH

#include "cell_model.h"
#include "hbv_stack.h"

namespace shyft {
	namespace core {

		namespace hbv_stack {
			using namespace std;

			typedef parameter parameter_t;
			typedef state state_t;
			typedef response response_t;
			typedef shared_ptr<parameter_t> parameter_t_;
			typedef shared_ptr<state_t>     state_t_;
			typedef shared_ptr<response_t>  response_t_;

			/** \brief all_reponse_collector aims to collect all output from a cell run so that it can be studied afterwards.
			*
			* \note could be quite similar between variants of a cell, e.g. pt_gs_k pt_hs_k pt_ss_k, ptss..
			* TODO: could we use another approach to limit code pr. variant ?
			* TODO: Really make sure that units are correct, precise and useful..
			*       both with respect to measurement unit, and also specifying if this
			*       a 'state in time' value or a average-value for the time-step.
			*/
			struct all_response_collector {
				double destination_area;///< in [m^2]
										// these are the one that we collects from the response, to better understand the model::
				pts_t pe_output;///< potential evap mm/h
				pts_t snow_outflow;///< HBV snow output [m^3/s] for the timestep
                pts_t glacier_melt;///< [m^3/s] for the timestep
                pts_t snow_sca;
				pts_t snow_swe;
				pts_t ae_output;///< actual evap mm/h
				pts_t soil_outflow; ///< Tank outflow given in [m^3/s] for the timestep
				pts_t avg_discharge; ///< Tank outflow given in [m^3/s] for the timestep
                pts_t charge_m3s; ///< = precip + glacier - act_evap - avg_discharge [m^3/s] for the timestep
				response_t end_reponse;///<< end_response, at the end of collected

				all_response_collector() : destination_area(0.0) {}
				explicit all_response_collector(const double destination_area) : destination_area(destination_area) {}
				all_response_collector(const double destination_area, const timeaxis_t& time_axis)
					: destination_area(destination_area), pe_output(time_axis, 0.0), snow_outflow(time_axis, 0.0),glacier_melt(time_axis,0.0),snow_sca(time_axis,0.0),snow_swe(time_axis,0), ae_output(time_axis, 0.0),
						soil_outflow(time_axis, 0.0), avg_discharge(time_axis, 0.0),charge_m3s(time_axis, 0.0) {}

				/**\brief called before run to allocate space for results */
				void initialize(const timeaxis_t& time_axis,int start_step,int n_steps, double area) {
					destination_area = area;
                    ts_init(pe_output, time_axis, start_step, n_steps, ts_point_fx::POINT_AVERAGE_VALUE);
                    ts_init(snow_outflow, time_axis, start_step, n_steps, ts_point_fx::POINT_AVERAGE_VALUE);
                    ts_init(glacier_melt, time_axis, start_step, n_steps, ts_point_fx::POINT_AVERAGE_VALUE);
                    ts_init(snow_sca, time_axis, start_step, n_steps, ts_point_fx::POINT_AVERAGE_VALUE);
                    ts_init(snow_swe, time_axis, start_step, n_steps, ts_point_fx::POINT_AVERAGE_VALUE);
                    ts_init(ae_output, time_axis, start_step, n_steps, ts_point_fx::POINT_AVERAGE_VALUE);
                    ts_init(soil_outflow, time_axis, start_step, n_steps, ts_point_fx::POINT_AVERAGE_VALUE);
                    ts_init(avg_discharge, time_axis, start_step, n_steps, ts_point_fx::POINT_AVERAGE_VALUE);
                    ts_init(charge_m3s, time_axis, start_step, n_steps, ts_point_fx::POINT_AVERAGE_VALUE);
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
				void collect(size_t idx, const response_t& response) {
					pe_output.set(idx, response.pt.pot_evapotranspiration);
					snow_outflow.set(idx, response.snow.outflow);//mm ?? //TODO: current mm/h. Want m3/s, but we get mm/h from snow output
                    glacier_melt.set(idx, response.gm_melt_m3s);
                    snow_sca.set(idx, response.snow.snow_state.sca);
					snow_swe.set(idx, response.snow.snow_state.swe);
					ae_output.set(idx, response.ae.ae);
					soil_outflow.set(idx, response.soil.outflow);//mm ?? //TODO: current mm/h. Want m3/s, but we get mm/h from soil output
					avg_discharge.set(idx, mmh_to_m3s(response.total_discharge, destination_area)); // wants m3/s, outflow is given in mm/h, so compute the totals in  mm/s
                    charge_m3s.set(idx, response.charge_m3s);
                }
				void set_end_response(const response_t& r) { end_reponse = r; }
			};

			/** \brief a collector that collects/keep discharge only */
			struct discharge_collector {
				double destination_area;
				pts_t avg_discharge; ///< Discharge given in [m^3/s] as the average of the timestep
                pts_t charge_m3s; ///< = precip + glacier - act_evap - avg_discharge [m^3/s] for the timestep
				response_t end_response;///<< end_response, at the end of collected
				bool collect_snow;
				pts_t snow_sca;
				pts_t snow_swe;

				discharge_collector() : destination_area(0.0), collect_snow(false) {}
				explicit discharge_collector(const double destination_area) : destination_area(destination_area), collect_snow(false) {}
				discharge_collector(const double destination_area, const timeaxis_t& time_axis)
					: destination_area(destination_area), avg_discharge(time_axis, 0.0),charge_m3s(time_axis, 0.0), collect_snow(false),
					snow_sca(timeaxis_t(time_axis.start(), time_axis.delta(), 0), 0.0),
					snow_swe(timeaxis_t(time_axis.start(), time_axis.delta(), 0), 0.0) {}

				void initialize(const timeaxis_t& time_axis, int start_step, int n_steps, double area) {
					destination_area = area;
                    ts_init(avg_discharge, time_axis, start_step, n_steps, ts_point_fx::POINT_AVERAGE_VALUE);
                    ts_init(charge_m3s, time_axis, start_step, n_steps, ts_point_fx::POINT_AVERAGE_VALUE);
					auto ta = collect_snow ? time_axis : timeaxis_t(time_axis.start(), time_axis.delta(), 0);
                    ts_init(snow_swe, ta, start_step, n_steps, ts_point_fx::POINT_AVERAGE_VALUE);
                    ts_init(snow_sca, ta, start_step, n_steps, ts_point_fx::POINT_AVERAGE_VALUE);
				}


				void collect(size_t idx, const response_t& response) {
					avg_discharge.set(idx, mmh_to_m3s(response.total_discharge, destination_area)); // outflow is given in mm, so compute the totals
                    charge_m3s.set(idx, response.charge_m3s);
					if (collect_snow) {
						snow_sca.set(idx, response.snow.snow_state.sca);
						snow_swe.set(idx, response.snow.snow_state.swe);
					}
				}

				void set_end_response(const response_t& response) { end_response = response; }
			};

			/**\brief a state null collector
			*
			* Used during calibration/optimization when there is no need for state,
			* and we need all the RAM for useful purposes.
			*/
			struct null_collector {
				void initialize(const timeaxis_t& time_axis,int start_step=0,int n_steps=0, double area = 0.0) {}
				void collect(size_t i, const state_t& response) {}
			};

			/** \brief the state_collector collects all state if enabled
			*
			*  \note that the state collected is instant in time, valid at the beginning of period
			*/
			struct state_collector {
				bool collect_state;  ///< if true, collect state, otherwise ignore (and the state of time-series are undefined/zero)
									 // these are the one that we collects from the response, to better understand the model::
				double destination_area;
				pts_t snow_swe;
				pts_t snow_sca;
				pts_t soil_moisture;
				pts_t tank_uz;
				pts_t tank_lz;

				state_collector() : collect_state(false), destination_area(0.0) {}
				explicit state_collector(const timeaxis_t& time_axis)
					: collect_state(false), destination_area(0.0), snow_swe(time_axis, 0.0), snow_sca(time_axis, 0.0),
						soil_moisture(time_axis, 0.0), tank_uz(time_axis, 0.0), tank_lz(time_axis, 0.0) { /* Do nothing */}
				/** brief called before run, prepares state time-series
				*
				* with preallocated room for the supplied time-axis.
				*
				* \note if collect_state is false, a zero length time-axis is used to ensure data is wiped/out.
				*/
				void initialize(const timeaxis_t& time_axis,int start_step,int n_steps, double area) {
					destination_area = area;
					timeaxis_t ta = collect_state ? time_axis : timeaxis_t(time_axis.start(), time_axis.delta(), 0);
                    ts_init(snow_sca, ta, start_step, n_steps, ts_point_fx::POINT_INSTANT_VALUE);
                    ts_init(snow_swe, ta, start_step, n_steps, ts_point_fx::POINT_INSTANT_VALUE);
                    ts_init(soil_moisture, ta, start_step, n_steps, ts_point_fx::POINT_INSTANT_VALUE);
                    ts_init(tank_uz, ta, start_step, n_steps, ts_point_fx::POINT_INSTANT_VALUE);
                    ts_init(tank_lz, ta, start_step, n_steps, ts_point_fx::POINT_INSTANT_VALUE);
				}
				/** called by the cell.run for each new state*/
				void collect(size_t idx, const state_t& state) {
					if (collect_state) {
						snow_sca.set(idx, state.snow.sca);
						snow_swe.set(idx, state.snow.swe);
						soil_moisture.set(idx, state.soil.sm);
						tank_uz.set(idx, state.tank.uz);
						tank_lz.set(idx, state.tank.lz);
					}
				}
			};

			// typedef the variants we need exported.
			typedef cell<parameter_t, environment_t, state_t, state_collector, all_response_collector> cell_complete_response_t;
			typedef cell<parameter_t, environment_t, state_t, null_collector, discharge_collector> cell_discharge_response_t;

		} // pt_hs_k

		  //specialize run method for all_response_collector
		template<>
		inline void cell<hbv_stack::parameter_t, environment_t, hbv_stack::state_t,
			hbv_stack::state_collector, hbv_stack::all_response_collector>
			::run(const timeaxis_t& time_axis, int start_step, int n_steps) {
			if (parameter.get() == nullptr)
				throw std::runtime_error("pt_hs_k::run with null parameter attempted");
			begin_run(time_axis,start_step,n_steps);
			hbv_stack::run_hbv_stack<direct_accessor, hbv_stack::response_t>(
				geo,
				*parameter,
				time_axis, start_step, n_steps,
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
		inline void cell<hbv_stack::parameter_t, environment_t, hbv_stack::state_t,
			hbv_stack::state_collector, hbv_stack::all_response_collector>
			::set_state_collection(bool on_or_off) {
			sc.collect_state = on_or_off;
		}

		//specialize run method for discharge_collector
		template<>
		inline void cell<hbv_stack::parameter_t, environment_t, hbv_stack::state_t,
			hbv_stack::null_collector, hbv_stack::discharge_collector>
			::run(const timeaxis_t& time_axis, int start_step, int n_steps) {
			if (parameter.get() == nullptr)
				throw std::runtime_error("pt_hs_k::run with null parameter attempted");
			begin_run(time_axis, start_step, n_steps);
			hbv_stack::run_hbv_stack<direct_accessor, hbv_stack::response_t>(
				geo,
				*parameter,
				time_axis, start_step, n_steps,
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
		inline void cell<hbv_stack::parameter_t, environment_t, hbv_stack::state_t,
			hbv_stack::null_collector, hbv_stack::discharge_collector>
			::set_snow_sca_swe_collection(bool on_or_off) {
			rc.collect_snow = on_or_off;// possible, if true, we do collect both swe and sca, default is off
		}
	} // core
} // shyft
