#pragma once

#ifdef SHYFT_NO_PCH
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>
#include <limits>
#include <future>
#include <utility>
#include <memory>
#include <stdexcept>
#include <future>
#include <mutex>

#include "core_pch.h"
#endif // SHYFT_NO_PCH


#include "bayesian_kriging.h"
#include "inverse_distance.h"
#include "kirchner.h"
#include "gamma_snow.h"
#include "priestley_taylor.h"
#include "pt_gs_k.h"
#include "pt_hs_k.h"
#include "geo_cell_data.h"
#include "routing.h"

/**
 * This file now contains mostly things to provide the PTxxK model,or
 * in general a region model, based on distributed cells where
 *  each cell keep its local properties, state, and local. env. data,
 *   and when run, starts from an initial state, and steps through the time-axis giving response and state-changes as output.
 *
 *   Running this kind of region model, is two steps:
 *    -#: Interpolation step interpolating zero or more observation sources of temperature,precipitation radiation etc
 *        into the cell- midpoint.
 *        After the interpolation phase, all cells have temp,precip,wind,rel-hum,rad locally calculated.
 *    -#: Then a cell-layer, runs cell computations
 *
 *  Each cell model have unique cell-types, including parameters, state, response.
 *  A region model consists of one or more catchments, polygons, where the response from member-cells is collected.
 *
 *  Usage of a region-model is either ordinary run, where we collect everything, responses, states for the complete cells,
 *  and catchment-level results (aggregates pr. catchment).
 *
 *  -secondly, we run the region-model as part of calibration/parameter-optimization model, where the primary output of the run
 *  should be sufficient to provide data to calculate the  goal-function(s) for the parameter optimization.
 *  In order to provide maximum speed, we minimize and specialize these computational models, so that cpu and memory footprint is minimal.
 *
 *
 */

namespace shyft {
    namespace core {
        namespace idw = inverse_distance;
        namespace btk = bayesian_kriging;
        using namespace std;

        /** \brief the interpolation_parameter keeps parameter needed to perform the
         * interpolation steps as specified by the user.
         */
        struct interpolation_parameter {
            typedef btk::parameter btk_parameter_t;
            typedef idw::precipitation_parameter idw_precipitation_parameter_t;
            typedef idw::temperature_parameter idw_temperature_parameter_t;
            typedef idw::parameter idw_parameter_t;

            btk::parameter temperature;
            bool use_idw_for_temperature = false;
            idw::temperature_parameter temperature_idw;
            idw::precipitation_parameter precipitation;
            idw::parameter wind_speed;
            idw::parameter radiation;
            idw::parameter rel_hum;

            interpolation_parameter() {}
            interpolation_parameter(const btk::parameter& temperature,
                                    const idw::precipitation_parameter& precipitation,
                                    const idw::parameter& wind_speed,
                                    const idw::parameter& radiation,
                                    const idw::parameter& rel_hum)
             : temperature(temperature), precipitation(precipitation),
               wind_speed(wind_speed), radiation(radiation), rel_hum(rel_hum) {}
            interpolation_parameter(const idw::temperature_parameter& temperature,
                                    const idw::precipitation_parameter& precipitation,
                                    const idw::parameter& wind_speed,
                                    const idw::parameter& radiation,
                                    const idw::parameter& rel_hum)
             : use_idw_for_temperature(true), temperature_idw(temperature),
               precipitation(precipitation), wind_speed(wind_speed),
               radiation(radiation), rel_hum(rel_hum) {}
        };

        /** \brief point_source contains common properties,functions
        * for the point sources in Enki.
        * Typically it contains a geo_point (3d position),plus a time_series
        * \tparam T a time_series, that supplies:
        *  -# type T::timeaxis_t
        *  -# double .value(size_t i)
        *  and we need that the ts-type
        *  can be accessed via an xxx_accessor(ts,other_time_axis),
        *  where xxx is direct,average or constant
        */
        template <class T>
        struct geo_point_ts {
            typedef T ts_t;
            geo_point location;//< the location of the observed/forecasted property ts
            T ts;//< a ts that can be transformed to a time-axis through an accessor..
            //geo_point_ts(){}
            // make it move-able
            //geo_point_ts(const geo_point_ts &c) : location(c.location),ts(c.ts) {}
            //geo_point_ts(geo_point_ts&& c) : location(c.location),ts(std::move(c.ts)) {}
            //geo_point_ts& operator=(geo_point_ts&& c) {location=c.location;ts=std::move(c.ts);return *this;}

            geo_point mid_point() const { return location; }
            const T& temperatures() const { return ts; }
        };

        /** \brief the idw_compliant geo_point_ts is a special variant,(or you could call it a fix),
        * so that we can feed 'data' sources to the IDW routines, and ensure that they execute at
        * maximum speed using time-series accessor suitable (with caching) so that true-average is
        * only computed once for each time-step.
        * This allows the input time-series to have any resolution/timestep, and ensures that
        * the resulting series at the cell.env_ts level are all aligned to the common execution time-axis.
        * \note this class is for local use inside the region_model::run_interpolation step. do not use elsewhere.
        *
        * This allows us to use direct_accessor in the run_cell processing step
        * \tparam GPTS \ref geo_point_ts the supplied geo located time-series.
        * \tparam TSA  time-series accessor to use for converting the ts to the execution time-axis.
        * \tparam TA   time-axis \ref shyft::time_series::timeaxis that goes into the constructor of TSA object
        */
        template <class GPTS, class TSA, class TA>
        class idw_compliant_geo_point_ts {
            const GPTS& s; //< type geo_point_ts, and reference is ok because we use this almost internally... hm.. in run_idw_interpolation, (so we should define it there..)
            mutable TSA tsa;// because .value(i) is const, and we use tsa.value(i) (tsa have local cache ix ) that is modified
        public:
            typedef geo_point geo_point_t;
            //run_idw compliant ct and methods
            idw_compliant_geo_point_ts(const GPTS &gpts, const TA& ta) : s(gpts), tsa(gpts.ts, ta) {}
            geo_point mid_point() const { return s.mid_point(); }
            double value(size_t i) const { return tsa.value(i); }
        };

        /** \brief region_environment contains the measured/forecasted sources of
        *  environmental properties, each that contains a geo_point and a time-series
        *  representation, like the \ref geo_point_ts type .
        *  This class comes into play in the interpolation step
        *  where each env.property is projected to the region.cells, using interpolation
        *  algorithm.
        *  For performance, it's imperative that access to each time-series value
        *  is efficient, and repeated calls should have very low cost (implies possible caching)
        *
        *  Each property is a share_ptr<vector<T>>
        *  such that it is easy to maintain one consistent copy of the inputs
        *
        * \tparam PS precipitation ts [mm/h] with geo_point
        * \tparam TS temperature ts [degC] with geo_point
        * \tparam RS radiation ts [xx] with  geo_point
        * \tparam HS relative humidity [rh] ts with geo_point
        * \tparam WS wind_speed ts [m/s] with geo_point
        */
        template <class PS, class TS, class RS, class HS, class WS>
        struct region_environment {
            typedef PS precipitation_t;
            typedef TS temperature_t;
            typedef RS radiation_t;
            typedef HS rel_hum_t;
            typedef WS wind_speed_t;
            typedef vector<PS> precipitation_vec_t;
            typedef vector<TS> temperature_vec_t;
            typedef vector<RS> radiation_vec_t;
            typedef vector<HS> rel_hum_vec_t;
            typedef vector<WS> wind_speed_vec_t;

            shared_ptr<temperature_vec_t>   temperature;
            shared_ptr<precipitation_vec_t> precipitation;
            shared_ptr<radiation_vec_t>     radiation;
            shared_ptr<wind_speed_vec_t>    wind_speed;
            shared_ptr<rel_hum_vec_t>       rel_hum;

        };
        ///< needs definition of the core time-series
        typedef shyft::time_series::point_ts<shyft::time_axis::fixed_dt> pts_t;
        /** \brief region_model is the calculation model for a region, where we can have
        * one or more catchments.
        * The role of the region_model is to describe region, so that we can run the
        * region computational model efficiently for a number of type of cells, interpolation and
        * catchment level algorihtms.
        *
        * The region model keeps a list of cells, of specified type C
        *
        *
        * \tparam C \see cell type parameter, describes the distributed cell calculation stack.
        * \tparam RE region environment class
        *  The cell-type must provide the following types:
        *  -# parameter_t parameter type for the cell calculation stack
        *  -# state_t     state type for the cell
        *  -# response_t  response type for cell
        *  -# timeaxis_t timeaxis type for the cell environment input/result ts.
        *  and following members/methods must be available
        *  -# .run(time_axis) executes the method stack for the cell, (must be thread-safe!)
        *  -# .env_ts with env. parameters
        *
        */

        template<class C, class RE = region_environment<pts_t,pts_t,pts_t,pts_t,pts_t>>
        class region_model {
        public:
            typedef C cell_t;
            typedef typename C::state_t state_t;
            typedef typename C::parameter_t parameter_t;
            typedef typename C::timeaxis_t timeaxis_t;
            typedef std::vector<cell_t> cell_vec_t;
            typedef std::shared_ptr<cell_vec_t > cell_vec_t_;
            typedef std::shared_ptr<parameter_t> parameter_t_;
            typedef typename cell_vec_t::iterator cell_iterator;
            typedef RE region_env_t;
        protected:

            cell_vec_t_ cells;///< a region consists of cells that orchestrate the distributed correlation
            parameter_t_ region_parameter;///< applies to all cells, except those with catchment override
            std::map<int, parameter_t_> catchment_parameters;///<  for each catchment (with cid) parameter is possible

            std::vector<bool> catchment_filter;///<if active (alias .size()>0), only calc if catchment_filter[catchment_id] is true.
            std::vector<int> cix_to_cid;///< maps internal zero-based catchment index ix to externally supplied catchment id.
            std::map<int,int> cid_to_cix;///< map external catchment id to internal index

            void update_ix_to_id_mapping() {
                // iterate over cell-vector
                // map<id,ix>
                // if new id, push it, map-it
                cid_to_cix.clear();
                cix_to_cid.clear();
                for(auto&c:*cells) {
					auto found = cid_to_cix.find(c.geo.catchment_id());
                    if(found==cid_to_cix.end()) {
                        cid_to_cix[c.geo.catchment_id()]=cix_to_cid.size();
                        c.geo.catchment_ix=cix_to_cid.size();// assign catchment ix to cell.
                        cix_to_cid.push_back(c.geo.catchment_id());// keep map
					} else {
						c.geo.catchment_ix = found->second;// assign corresponding ix.
					}
                }
            }

            size_t n_catchments=0;///< optimized//extracted as max(cell.geo.catchment_id())+1 in run interpolate

            void clone(const region_model& c) {
                // First, clear own content
                ncore = c.ncore;
                time_axis = c.time_axis;
                catchment_filter = c.catchment_filter;
                n_catchments = c.n_catchments;
				ip_parameter = c.ip_parameter;
                region_env = c.region_env;// todo: verify it is deep or shallow copy
                catchment_parameters.clear();
                // Then, clone from c
                cix_to_cid=c.cix_to_cid;
                cid_to_cix=c.cid_to_cix;
                initial_state = c.initial_state;
                cells = cell_vec_t_(new cell_vec_t(*(c.cells)));
                river_network=c.river_network;
                set_region_parameter(*(c.region_parameter));
                for(const auto& pair:c.catchment_parameters)
                    set_catchment_parameter(pair.first, *(pair.second));
            }


        public:
            /** \brief construct a region model,
             *  by supplying the following minimum set of parameters.
             * \param region_param ref to parameters to apply to all cells, the constructor loops over cells and implants a ref to a copy of this param.
             * \param cells a shared pointer to a vector of cells, notice, we do share the cell-vector!
             */
            region_model(std::shared_ptr<std::vector<C> >& cells, const parameter_t &region_param)
             : cells(cells) {
                set_region_parameter(region_param);// ensure we have a correct model
                ncore = thread::hardware_concurrency();
                update_ix_to_id_mapping();
            }
            region_model(const std::vector<geo_cell_data>& geov, const parameter_t &region_param)
              :cells(std::make_shared<std::vector<C>>()){
                state_t s0;
                auto global_parameter = make_shared<typename cell_t::parameter_t>();
                for (const auto&gcd : geov)  cells->push_back(cell_t{ gcd, global_parameter, s0 });
                set_region_parameter(region_param);// ensure we have a correct region_param for all cells
                ncore = thread::hardware_concurrency() ;
                update_ix_to_id_mapping();
            }
            region_model(std::shared_ptr<std::vector<C> >& cells,
                         const parameter_t &region_param,
                         const std::map<int, parameter_t>& catchment_parameters)
             : cells(cells) {
                set_region_parameter(region_param);
                update_ix_to_id_mapping();
                for(const auto & pair:catchment_parameters)
                     set_catchment_parameter(pair.first, pair.second);
                ncore = thread::hardware_concurrency();
            }
            region_model(const region_model& model) { clone(model); }
			region_model& operator=(const region_model& c) {
                if (&c != this)
                    clone(c);
                return *this;
            }
            ///-- properties accessible to user
            timeaxis_t time_axis; ///<The time_axis as set from run_interpolation, determines the axis for run()..
            size_t ncore = 0; ///<< defaults to 4x hardware concurrency, controls number of threads used for cell processing
			interpolation_parameter ip_parameter;///< the interpolation parameter as passed to interpolate/run_interpolation
            region_env_t region_env;///< the region environment (shallow-copy?) as passed to the interpolation/run_interpolation
            std::vector<state_t> initial_state; ///< the initial state, set explicit, or by the first call to .set_states(..) or run_cells()
            routing::river_network river_network;///< the routing river_network, can be empty
            /** \brief compute and return number of catchments inspecting call cells.geo.catchment_id() */
            size_t number_of_catchments() const { return cix_to_cid.size(); }

            /** connect all cells in a catchment to a river
             * \param cid catchment id for the cells to be connected to the specified river
             * \param rid river id for the target river. Note it can be 0 to set no routing for the cells
             */
            void connect_catchment_to_river(int cid,int rid) {
                if(cid_to_cix.find(cid)==cid_to_cix.end()) throw std::runtime_error(string("specified catchment id=") + std::to_string(cid)+string(" not found"));
                if(routing::valid_routing_id(rid)) river_network.check_rid(rid);// verify it exists.
                for(auto&c:*cells)
                    if(int(c.geo.catchment_id())==cid) c.geo.routing.id=rid;
            }

            bool has_routing() const {
                for(auto&c:*cells) {
                    if(routing::valid_routing_id(c.geo.routing.id))
                        return true;
                }
                return false;
            }
            /**\brief extracts the geo-cell data part out from the cells */
            std::vector<geo_cell_data> extract_geo_cell_data() const {
                std::vector<geo_cell_data> r; r.reserve(cells->size());
                for (const auto&c : *cells) r.push_back(c.geo);
                return r;
            }
			/** \brief Initializes the cell enviroment (cell.env.ts* )
			 *
			 * The initializes the cell environment, that keeps temperature, precipitation etc
			 * that is local to the cell. The intial values of these time-series is set to zero.
			 * The region-model time-axis is set to the supplied time-axis, so that
			 * the any calculation steps will use the supplied time-axis.
			 * This call is needed prior to call to interpolate() or run_cells().
			 * The call ensures that all cells.env ts are reset to zero, with a time-axis and
			 * value-vectors according to the supplied time_axis.
			 * Also note that the region_model.time_axis is set to the supplied time-axis.
			 *
			 * \param time_axis specifies the time-axis for the region-model, and thus the cells.
			 * \return void
			 */
			void initialize_cell_environment(const timeaxis_t& time_axis) {
				for (auto&c : *cells) {
					c.init_env_ts(time_axis);
				}
				n_catchments = number_of_catchments();// keep this/assume invariant..
				this->time_axis = time_axis;
			}

			/** \brief interpolate the supplied region_environment to the cells
			*
			* \note initialize_cell_environment should be called prior to this
			*
			* Only supplied vectors of temp, precipitation etc. are interpolated, thus
			* the user of the class can choose to put in place distributed series in stead.
			*
			* \tparam RE \ref region_environment type
			*
			* \param ip_parameter contains wanted parameters for the interpolation
			* \param env contains the \ref region_environment type
			* \param best_effort controls if the entire calculation should be aborted in case of one ip-going wrong(leaving nans @ cells)
			* \return true if everything went ok, false if exceptions, doing best effort
			*
			*/

			bool interpolate(const interpolation_parameter& ip_parameter, const region_env_t& env, bool best_effort=true) {
				using namespace shyft::core;
				using namespace std;
				namespace idw = shyft::core::inverse_distance;
				namespace btk = shyft::core::bayesian_kriging;
				// we use local scoped cell_proxy to support
				// filtering interpolation to the cells set by
				// calculation filter
                struct cell_proxy {
                    cell_proxy():cell(nullptr) {}
                    explicit cell_proxy(cell_t *c):cell(c){}
                    cell_proxy(cell_proxy const &o):cell(o.cell) {}
                    cell_proxy(cell_proxy&&o):cell(o.cell){}
                    cell_proxy& operator=(cell_proxy const&o){cell=o.cell;return *this;}
                    cell_proxy& operator=(cell_proxy&&o){cell=o.cell;return *this;}
                    cell_t *cell;// ptr ok, because it's only within this scope, no life-time stuff needed, just ref
                    // support enough methods to make it look like a cell during idw/btk
                    geo_point mid_point()const {return cell->mid_point();}
                    //< support for the interpolation phase, executes before the cell-run
			        void   set_temperature(size_t ix, double temperature_value) { cell->env_ts.temperature.set(ix, temperature_value); }
			        ///< used by the radiation interpolation to adjust the local radiation with respect to idw sources.
			        double slope_factor() const { return cell->geo.radiation_slope_factor(); }
                };
                std::vector<cell_proxy> cell_ps;cell_ps.reserve(cells->size());
                for(auto&c:*cells)
                    if(is_calculated_by_catchment_ix(c.geo.catchment_ix))
                        cell_ps.emplace_back(&c);


				typedef shyft::time_series::average_accessor<typename region_env_t::temperature_t::ts_t, timeaxis_t> temperature_tsa_t;
				typedef shyft::time_series::average_accessor<typename region_env_t::precipitation_t::ts_t, timeaxis_t> precipitation_tsa_t;
				typedef shyft::time_series::average_accessor<typename region_env_t::radiation_t::ts_t, timeaxis_t> radiation_tsa_t;
				typedef shyft::time_series::average_accessor<typename region_env_t::wind_speed_t::ts_t, timeaxis_t> wind_speed_tsa_t;
				typedef shyft::time_series::average_accessor<typename region_env_t::rel_hum_t::ts_t, timeaxis_t> rel_hum_tsa_t;


				typedef idw_compliant_geo_point_ts<typename region_env_t::temperature_t, temperature_tsa_t, timeaxis_t> idw_compliant_temperature_gts_t;

				typedef idw_compliant_geo_point_ts<typename region_env_t::precipitation_t, precipitation_tsa_t, timeaxis_t> idw_compliant_precipitation_gts_t;
				typedef idw_compliant_geo_point_ts<typename region_env_t::radiation_t, radiation_tsa_t, timeaxis_t> idw_compliant_radiation_gts_t;
				typedef idw_compliant_geo_point_ts<typename region_env_t::wind_speed_t, wind_speed_tsa_t, timeaxis_t> idw_compliant_wind_speed_gts_t;
				typedef idw_compliant_geo_point_ts<typename region_env_t::rel_hum_t, rel_hum_tsa_t, timeaxis_t> idw_compliant_rel_hum_gts_t;

				typedef idw::temperature_model  <idw_compliant_temperature_gts_t, cell_proxy, typename interpolation_parameter::idw_temperature_parameter_t, geo_point, idw::temperature_gradient_scale_computer> idw_temperature_model_t;
				typedef idw::precipitation_model<idw_compliant_precipitation_gts_t, cell_proxy, typename interpolation_parameter::idw_precipitation_parameter_t, geo_point> idw_precipitation_model_t;
				typedef idw::radiation_model    <idw_compliant_radiation_gts_t, cell_proxy, typename interpolation_parameter::idw_parameter_t, geo_point> idw_radiation_model_t;
				typedef idw::wind_speed_model   <idw_compliant_wind_speed_gts_t, cell_proxy, typename interpolation_parameter::idw_parameter_t, geo_point> idw_windspeed_model_t;
				typedef idw::rel_hum_model      <idw_compliant_rel_hum_gts_t, cell_proxy, typename interpolation_parameter::idw_parameter_t, geo_point> idw_relhum_model_t;

				typedef  shyft::time_series::average_accessor<typename region_env_t::temperature_t::ts_t, timeaxis_t> btk_tsa_t;
				this->ip_parameter = ip_parameter;// keep the most recently used ip_parameter
                this->region_env = env;// this could be a shallow copy
				// Allocate memory for the source_destinations, put in the reference to the parameters:
				// Run one thread for each optional interpolation
				//  notice that if a source is nullptr, then we leave the allocated cell.level signal to fill-value nan
				//  the intention is that the orchestrator at the outside could provide it's own ready-made
				//  interpolated/distributed signal, e.g. temperature input from arome-data


				auto btkx = async(launch::async, [&]() {
					if (env.temperature != nullptr) {
						if (env.temperature->size()>1) {
							if (ip_parameter.use_idw_for_temperature) {
								idw::run_interpolation<idw_temperature_model_t, idw_compliant_temperature_gts_t>(
									time_axis, *env.temperature, ip_parameter.temperature_idw, cell_ps,
									[](cell_proxy &d, size_t ix, double value) { d.cell->env_ts.temperature.set(ix, value); }
								);
							} else {
								btk::btk_interpolation<btk_tsa_t>(
									begin(*env.temperature), end(*env.temperature), begin(cell_ps), end(cell_ps),
									time_axis, ip_parameter.temperature
									);
							}
						} else {
							// just one temperature ts. just a a clean copy to destinations
							btk_tsa_t tsa((*env.temperature)[0].ts, time_axis);
							typename cell_t::env_ts_t::temperature_ts_t temp_ts(time_axis, 0.0);
							for (size_t i = 0;i<time_axis.size();++i) {
								temp_ts.set(i, tsa.value(i));
							}
							for (auto& c : *cells) {
                                if(is_calculated_by_catchment_ix(c.geo.catchment_ix))
								c.env_ts.temperature = temp_ts;
							}
						}
					}
				});

				auto idw_precip = async(launch::async, [&]() {
					if (env.precipitation != nullptr)
						idw::run_interpolation<idw_precipitation_model_t, idw_compliant_precipitation_gts_t>(
							time_axis, *env.precipitation, ip_parameter.precipitation, cell_ps,
							[](cell_proxy &d, size_t ix, double value) { d.cell->env_ts.precipitation.set(ix, value); }
					);
				});

				auto idw_radiation = async(launch::async, [&]() {
					if (env.radiation != nullptr)
						idw::run_interpolation<idw_radiation_model_t, idw_compliant_radiation_gts_t>(
							time_axis, *env.radiation, ip_parameter.radiation, cell_ps,
							[](cell_proxy &d, size_t ix, double value) { d.cell->env_ts.radiation.set(ix, value); }
					);
				});

				auto idw_wind_speed = async(launch::async, [&]() {
					if (env.wind_speed != nullptr)
						idw::run_interpolation<idw_windspeed_model_t, idw_compliant_wind_speed_gts_t>(
							time_axis, *env.wind_speed, ip_parameter.wind_speed, cell_ps,
							[](cell_proxy &d, size_t ix, double value) { d.cell->env_ts.wind_speed.set(ix, value); }
					);
				});

				auto idw_rel_hum = async(launch::async, [&]() {
					if (env.rel_hum != nullptr)
						idw::run_interpolation<idw_relhum_model_t, idw_compliant_rel_hum_gts_t>(
							time_axis, *env.rel_hum, ip_parameter.rel_hum, cell_ps,
							[](cell_proxy &d, size_t ix, double value) { d.cell->env_ts.rel_hum.set(ix, value); }
					);
				});

                bool btkx_ok=true,precip_ok=true,radiation_ok=true,wind_speed_ok=true,rel_hum_ok=true;
                exception_ptr p_ex;
				try {btkx.get();} catch(...) {p_ex=current_exception();btkx_ok=false;}
				try {idw_precip.get();} catch(...) {p_ex=current_exception();precip_ok=false;}
				try {idw_radiation.get();} catch(...) {p_ex=current_exception();radiation_ok=false;}
				try {idw_wind_speed.get();} catch(...) {p_ex=current_exception();wind_speed_ok=false;}
				try {idw_rel_hum.get();} catch(...) {p_ex=current_exception();rel_hum_ok=false;}
				if(!best_effort && p_ex)
				    rethrow_exception(p_ex);
				return btkx_ok && precip_ok && radiation_ok && wind_speed_ok && rel_hum_ok;
			}

            /** \brief initializes cell.env and project region env. time_series to cells.
            *
            * \note Prior to running all cell.env_ts.xxx are reset to zero, and have a length of time_axis.size().
            *
            * Only supplied vectors of temp, precip etc. are interpolated, thus
            * the user of the class can choose to put in place distributed series in stead.
			*
			* This call simply calls initialize_cell_environment() followed by interpolate<..>(..)
            *
            *
            * \param ip_parameter contains wanted parameters for the interpolation
            * \param time_axis should be equal to the \ref shyft::time_axis the \ref region_model is prepared running for.
            * \param env contains the \ref region_environment type
            * \param best_effort (default=true)
            * \return true if entire process was done with no exceptions raised
            *
            */
            bool run_interpolation(const interpolation_parameter& ip_parameter, const timeaxis_t& time_axis, const region_env_t& env, bool best_effort=true) {
				initialize_cell_environment(time_axis);
				return interpolate(ip_parameter, env);
            }

            /** \brief run_cells calculations over specified time_axis
            *  the cell method stack is invoked for all the cells, using multicore up to a maximum number of
            *  tasks/cores. Notice that this implies that executing the cell method stack should have no
            *  side-effects except for invocation of the cell state_collector, response_collector.
            *  No thread should run simultaneously on the same cell, and the collectors must be carefully
            *  written to avoid any race-condition or memory contention (lock-free code).
            *
            * \pre cells should be prepared by prior calls so that
            *  -# all cells have env_ts intialized with env. precip,temp,rad etc.
            *  -# all cells have state set to the initial values
            *  -# catchment_parameters should be set (and applied to the matching cells)
            * \post :
            *  -# all cells have updated state variables
            *
            *
            * \param use_ncore if 0 figure out threads using hardware info,
            *   otherwise use supplied value (throws if >100x ncore)
            * \param start_step of time-axis, defaults to 0, starting at the beginning
            * \param n_steps number of steps to run from the start_step, a value of 0 means running all time-steps
            * \return void
            *
            */
            void run_cells(size_t use_ncore=0, int start_step=0, int  n_steps=0) {
                if(use_ncore == 0) {
                    if(ncore==0) ncore=4;// a reasonable minimum..
                    use_ncore = ncore;
                } else if (use_ncore > 100 * ncore) {
                    throw runtime_error(string("illegal parameter value: use_ncore(")+to_string(use_ncore)+string(" is more than 100 time available physical cores: ") + to_string(ncore));
                }
                if(! (time_axis.size()>0))
                    throw runtime_error("region_model::run with invalid time_axis invoked");
                if(start_step<0 || size_t(start_step+1)>time_axis.size())
                    throw runtime_error("region_model::run start_step must in range[0..n_steps-1>");
                if(n_steps<0)
                    throw runtime_error("region_model::run n_steps must be range[0..time-axis-steps]");
                if (size_t(start_step + n_steps) > time_axis.size())
                    throw runtime_error("region_model::run start_step+n_steps must be within time-axis range");
                if (initial_state.size() != cells->size())
                    get_states(initial_state); // snap the initial state here, unless it's already set by the user
                parallel_run(time_axis,start_step,n_steps, begin(*cells), end(*cells),use_ncore);
                run_routing(start_step,n_steps);
            }

            /** \brief set the region parameter, apply it to all cells
             *        that do not have catchment specific parameters.
             * \note that if there already exist a region parameter
             *       then the values of the supplied parameter is just copied
             *       into the region parameter-set.
             * \param p is the wanted region parameter set
             */
            void set_region_parameter(const parameter_t &p) {
                if (region_parameter == nullptr) {
                    region_parameter = parameter_t_(new parameter_t(p));
                    for(auto& c:*cells)
                        if (!has_catchment_parameter(c.geo.catchment_id()))
                            c.set_parameter(region_parameter);
                } else {
                    (*region_parameter) = p;
                }
            }

            /** \brief provide access to current region parameter-set
            * \return a ref to region_parameter (non-const by intention)
            */
            parameter_t& get_region_parameter() const {
                return *region_parameter;
            }

            /** \brief creates/modifies a pr catchment override parameter
                \param catchment_id the 0 based catchment_id that correlates to the cells catchment_id
                \param p a reference to the parameter that will be used/applied to those cells
            */
            void set_catchment_parameter(int catchment_id, const parameter_t & p) {
                if (catchment_parameters.find(catchment_id) == catchment_parameters.end()) {
                    auto shared_p = parameter_t_(new parameter_t(p));// add to map, a copy of p
                    catchment_parameters[catchment_id] = shared_p;
                    for(auto &c:*cells)
                        if (int(c.geo.catchment_id()) == catchment_id)
                            c.set_parameter(shared_p);
                } else {
                    *(catchment_parameters[catchment_id]) = p; //copy values into existing parameters
                }
            }


            /** \brief remove a catchment specific parameter override, if it exists.
            */
            void remove_catchment_parameter(int catchment_id) {
                auto it = catchment_parameters.find(catchment_id);
                if (it != catchment_parameters.end()) {
                    catchment_parameters.erase(catchment_id);// get rid of it, and update the affected cells with the global parameter
                    for(auto & c:*cells)
                        if (int(c.geo.catchment_id()) == catchment_id)
                            c.set_parameter(region_parameter);
                }
            }
            /** \brief returns true if there exist a specific parameter override for the specified 0-based catchment_id*/
            bool has_catchment_parameter(int catchment_id) const {
                return catchment_parameters.find(catchment_id) != catchment_parameters.end();
            }

            /** \brief return the parameter valid for specified catchment_id, or global parameter if not found.
            * \note Be aware that if you change the returned parameter, it will affect the related cells.
            * \param catchment_id 0 based catchment id as placed on each cell
            * \returns reference to the real parameter structure for the catchment_id if exists, otherwise the global parameters
            */
            parameter_t& get_catchment_parameter(int catchment_id) const {
                auto search=catchment_parameters.find(catchment_id);
                if ( search != catchment_parameters.end())
                    return *((*search).second);
                else
                    return *region_parameter;
            }

            /** \brief set/reset the catchment based calculation filter. This affects what get simulate/calculated during
             * the run command. Pass an empty list to reset/clear the filter (i.e. no filter).
             *
             * \param catchment_id_list is a catchment id vector
             */
            void set_catchment_calculation_filter(const std::vector<int>& catchment_id_list) {
                if (catchment_id_list.size()) {
                    if(catchment_id_list.size()> cix_to_cid.size())
                        throw std::runtime_error("set_catchment_calculation_filter: supplied list > available catchments");
                    for(auto cid:catchment_id_list) {
                        if(cid_to_cix.find(cid)==cid_to_cix.end())
                            throw std::runtime_error("set_catchment_calculation_filter: no cells have supplied cid");
                    }
                    catchment_filter = vector<bool>(cix_to_cid.size(), false);
                    for (auto i : catchment_id_list) catchment_filter[cid_to_cix[i]] = true;
                } else {
                    catchment_filter.clear();
                }

            }

            /** \brief set/reset the catchment and river based calculation filter.
             * This affects what get simulate/calculated during
             * the run command. Pass an empty list to reset/clear the filter (i.e. no filter).
             *
             * \param catchment_id_list is a catchment id vector
             * \param river_id_list is a river id vector
             */
            void set_calculation_filter(const std::vector<int>& catchment_id_list, const std::vector<int>& river_id_list) {
                set_catchment_calculation_filter(catchment_id_list);
                for (auto rid : river_id_list) {
                    auto catchments_involved = get_catchment_feeding_to_river(rid);
                    for (auto cid : catchments_involved) {
                        if(catchment_filter.size()==0) // none selected yet, allocate full vector, and
                            catchment_filter = vector<bool>(cix_to_cid.size(), false);
                        catchment_filter[cid_to_cix[cid]] = true;// then assign true
                    }
                }
            }

            /**compute the unique set of catchments feeding into this river_id, or any river upstream */
            std::set<int> get_catchment_feeding_to_river(int river_id) const {
                std::set<int> r;
                auto all_upstreams_rid = river_network.all_upstreams_by_id(river_id);
                all_upstreams_rid.push_back(river_id);// remember to add this river
                for (const auto&c : *cells) {
                    if (routing::valid_routing_id(c.geo.routing.id) &&
                        std::find(begin(all_upstreams_rid), end(all_upstreams_rid), c.geo.routing.id) != end(all_upstreams_rid) ) {
                        r.insert(c.geo.catchment_id());
                    }
                }
                return r;
            }

            /** \brief using the catchment_calculation_filter to decide if discharge etc. are calculated.
             * \param cid  catchment id
             * \returns true if catchment id is calculated during runs
             */
            bool is_calculated(size_t cid) const {
                return is_calculated_by_catchment_ix(cix_from_cid(cid));
            }

            bool is_calculated_by_catchment_ix(size_t cix) const {return catchment_filter.size() == 0 || (catchment_filter[cix]);}

            size_t cix_from_cid(size_t cid) const {
                auto cix=cid_to_cix.find(cid);
                if(cix == cid_to_cix.end())
                    throw runtime_error("region_model: no match for cid in map lookup");
                return cix->second;
            }
            /** \brief collects current state from all the cells
            * \note that catchment filter can influence which states are calculated/updated.
            *\param end_states a reference to the vector<state_t> that are filled with cell state, in order of appearance.
            */
            void get_states(std::vector<state_t>& end_states) const {
                end_states.clear();end_states.reserve(std::distance(begin(*cells), end(*cells)));
                for(const auto& cell:*cells) end_states.emplace_back(cell.state);
            }

            /**\brief set current state for all the cells in the model.
             *
             * If this is the first 'set_states()', initial_state is copied from the
             * supplied vector. The purpose of this is to ease scripting so that one
             * always get back the initial state if needed.
             *
             * \param states is a vector<state_t> of all states, must match size/order of cells.
             * \note throws runtime-error if states.size is different from cells.size
             */
            void set_states(const std::vector<state_t>& states) {
                if (states.size() != size())
                    throw runtime_error("Length of the state vector must equal number of cells");
                auto state_iter = begin(states);
                for(auto& cell:*cells) cell.set_state(*(state_iter++));
                if (initial_state.size() != states.size())
                    initial_state = states;// if first time, or different copy the state
            }
            /**\brief revert cell states to the initial state (if it exists)
            */
            void revert_to_initial_state() {
                if (initial_state.size() == 0)
                    throw runtime_error("Initial state not yet established or set");
                set_states(initial_state);
            }
            /** \brief enable state collection for specified or all cells
             * \note that this only works if the underlying cell is configured to
             *       do state collection. THis is typically not the  case for
             *       cell-types that are used during calibration/optimization
             */
            void set_state_collection(int catchment_id, bool on_or_off) {
                for(auto& cell:*cells)
                    if (catchment_id == -1 || (int)cell.geo.catchment_id() == catchment_id )
                        cell.set_state_collection(on_or_off);
            }
            /** \brief enable/disable collection of snow sca|sca for calibration purposes
             * \param catchment_id to enable snow calibration for, -1 means turn on/off for all
             * \param on_or_off true|or false.
             * \note if the underlying cell do not support snow sca|swe collection, this call has no effect
             */
            void set_snow_sca_swe_collection(int catchment_id,bool on_or_off) {
                for(auto& cell:*cells)
                    if (catchment_id == -1 || (int)cell.geo.catchment_id() == catchment_id )
                        cell.set_snow_sca_swe_collection(on_or_off);
            }
            /** \return cells as shared_ptr<vector<cell_t>> */
            cell_vec_t_ get_cells() const { return cells; }

            /** \return number of cells */
            size_t size() const { return distance(begin(*cells), end(*cells)); }

            /** \brief catchment_discharges, vital for calibration
             * \tparam TSV a vector<time_series> type, where time_series supports:
             *  -# .ct(timeaxis_t, double) fills a series with 0.0 for all time_axis elements
             *  -# .add( const some_ts & ts)
             * \param cr catchment result vector to be filled in
             * \note the ts type should have proper move/copy etc. semantics
             * \return filled in cr, dimensioned to number of catchments, where the i'th entry correspond to cid using cix_to_cid(i)
             */
            template <class TSV>
            void catchment_discharges( TSV& cr) const {
                typedef typename TSV::value_type ts_t;
                cr.clear();
                cr.reserve(n_catchments);
                for(size_t i=0;i<n_catchments;++i) {
                    cr.emplace_back(ts_t(time_axis, 0.0));
                }
                for(const auto& c: *cells) {
                    if ( is_calculated_by_catchment_ix(c.geo.catchment_ix))
                        cr[c.geo.catchment_ix].add(c.rc.avg_discharge);
                }
            }

            template <class TSV>
            void catchment_charges(TSV& cr) const {
                typedef typename TSV::value_type ts_t;
                cr.clear();
                cr.reserve(n_catchments);
                for (size_t i = 0;i < n_catchments;++i) {
                    cr.emplace_back(ts_t(time_axis, 0.0));
                }
                for (const auto& c : *cells) {
                    if (is_calculated_by_catchment_ix(c.geo.catchment_ix)) {
                        cr[c.geo.catchment_ix].add(c.rc.charge_m3s);
                    }
                }
            }

            /**\brief return all discharges at the output of the routing points
             *
             * For all routing nodes,(maybe terminal routing nodes ?)
             * compute the routed discharge.
             * \note if no routing, empty time-series is returned.
             * \return time-series by ascending routing id order where the i'th entry correspond to sorted river idents asc.
             */
            template <class TSV>
            void routing_discharges( TSV& cr) const {
                //typedef typename TSV::value_type ts_t;
                cr.clear();
                if(has_routing()) {
                    //TODO: iterate over the routing model, return results
                    routing::model<C> rn(river_network,cells,time_axis);
                    std::vector<int> rids;
                    for(auto r:river_network.rid_map)
                        rids.push_back(r.first);
                    std::sort(begin(rids),end(rids));// ascending order!
                    for(auto rid:rids) {
                        cr.emplace_back(rn.output_m3s(rid));
                    }
                }
                return cr;
            }
            std::shared_ptr<pts_t> river_output_flow_m3s(int rid) const {
                auto r= std::make_shared<pts_t>(time_axis,0.0,time_series::ts_point_fx::POINT_AVERAGE_VALUE);
                if(has_routing()) {
                    routing::model<C> rn(river_network,cells,time_axis);
                    r=std::make_shared<pts_t>(rn.output_m3s(rid));
                }
                return r;
            }
            std::shared_ptr<pts_t> river_upstream_inflow_m3s(int rid) const {
                auto r= std::make_shared<pts_t>(time_axis,0.0,time_series::ts_point_fx::POINT_AVERAGE_VALUE);
                if(has_routing()) {
                    routing::model<C> rn(river_network,cells,time_axis);
                    r=std::make_shared<pts_t>(rn.upstream_inflow(rid));
                }
                return r;
            }
            std::shared_ptr<pts_t> river_local_inflow_m3s(int rid) const {
                auto r= std::make_shared<pts_t>(time_axis,0.0,time_series::ts_point_fx::POINT_AVERAGE_VALUE);
                if(has_routing()) {
                    routing::model<C> rn(river_network,cells,time_axis);
                    r=std::make_shared<pts_t>(rn.local_inflow(rid));
                }
                return r;
            }
        protected:
            /** \brief parallell_run using a mid-point split + async to engange multicore execution
             *
             * \param time_axis forwarded to the cell.run(time_axis)
             * \param start_step of time-axis
             * \param n_steps number of steps to run
             * \param 'beg' the beginning of the cell-range
             * \param 'endc' the end of cell range
             */
            void single_run(const timeaxis_t& time_axis, int start_step, int  n_steps, cell_iterator beg, cell_iterator endc) {
                for(cell_iterator cell=beg; cell!=endc;++cell) {
                        //& cell:boost::make_iterator_range(beg,endc)) {

                     if (is_calculated_by_catchment_ix(cell->geo.catchment_ix))
                        cell->run(time_axis,start_step,n_steps);
                }
            }
            /** \brief uses async to execute the single_run, partitioning the cell range into thread-cell count
             *
             * \throw runtime_error if thread_cell_count is zero
             * \return when all cells calculated
             * \param time_axis time-axis to use
             * \param start_step of time-axis
             * \param n_steps number of steps to run
             * \param 'beg' the beginning of the cell-range
             * \param 'endc' the end of cell range
             * \param thread_cell_count number of cells given to each async thread
             */
            void parallel_run(const timeaxis_t& time_axis, int start_step, int  n_steps, cell_iterator beg, cell_iterator endc,int use_ncore) {
                size_t len = distance(beg, endc);
                if(len == 0)
                    return;
                if(use_ncore == 0)
                    throw runtime_error("parallel_run: use_ncore is zero ");
                vector<future<void>> calcs;
                mutex pos_mx;
                size_t pos = 0;
                for (int i = 0;i < use_ncore;++i) { // using ncore = logical available core saturates cpu 100%
                    calcs.emplace_back(
                        async(launch::async,
                            [this,&pos,&pos_mx,len,&time_axis,&beg,start_step,n_steps]() {
                                while (true) {
                                    size_t ci;
                                    { lock_guard<decltype(pos_mx)> lock(pos_mx);// get work item here
                                        if (pos < len)
                                            ci = pos++;
                                        else
                                            break;
                                    }
                                    this->single_run(time_axis, start_step, n_steps, beg + ci, beg + ci + 1);// just for one cell
                                }
                            }
                        )
                    );
                }
                for(auto &f:calcs)
                    f.get();
                return;
            }
            void run_routing(int start_step,int n_steps) {
                // TODO: implement
                // things to consider:
                //  a) start_step,n_steps could be a problem due to the time-delay/convolution window.
                //     so data is not available through the convolution until after window-size.
                //     possible solutions:
                //         1) leave it to the user
                //         2)  always run from start_step=0 ?
                //         3) as the routing model for the largest delay counted in timesteps, start - longest delay
                //  b) for convolution, we can base the approach on 'pull', calculate on demand, possibly with memory-cached result time-series
                //     in that scenario, we would need a 'dirty' bit to be set (initially) and when starting the run-cells step.
                //         1) optimization, maybe later
                //  c) the catchment filter, based on cids, could also be extended to routing ids (rids).
                //     also: if routing, only rids type of filter allowed ?
                //           if rids, cell to rid is possibly multi-step lookup. Need to make a rids-filter, where members are all reachable
                //           cells from wanted rids (to be calculated).
                //         1) leave it to user
                //         2) make alg. given rid's: compute cid's needed
                //         3) if river_network available, deny cids based filter, insist on rids ?
                //
                //  d) consistency: if cids are used, and routing (and rids) are enabled, auto-extend the cids so that all connected cells(and corresponding cids)
                //     are calculated (avoid partial calculation of something that goes into a routing network...
                //     question: how much of this consistency/complexity should we put to the user setting the calculation filter
                if(!has_routing())
                    return;

            }
        };

    } // core
} // shyft
