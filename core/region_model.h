#pragma once

#include "bayesian_kriging.h"
#include "inverse_distance.h"
#include "kirchner.h"
#include "gamma_snow.h"
#include "priestley_taylor.h"
#include "pt_gs_k.h"
#include "pt_hs_k.h"
#include "geo_cell_data.h"

/** \file This file now contains mostly things to provide the PTxxK model,or
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
        * Typically it contains a geo_point (3d position),plus a timeseries
        * \tparam T a timeseries, that supplies:
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
        * \tparam TA   time-axis \ref shyft::timeseries::timeaxis that goes into the constructor of TSA object
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

        template<class C>
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
        protected:

            cell_vec_t_ cells;///< a region consists of cells that orchestrate the distributed correlation
            // TODO:
            // catchment_vec_t_ catchments;//< catchment level, after cell.run is done
            // we could have kirchner for each catchments (cell-routing-delay before and or routing after kirchner)
            //
            // so the catchment concept should be modeled and handled:
            //   a catchment
            //      id
            //       a geo-shape, (we keep catch_id for each ..)
            //      catchment level model-parameters (kirchner etc.)
            //      obs. discharge
            //      obs. snow  (swe,sca ?)
            //     after run's of the model
            //     we can extract cell-level into into the catchments

            parameter_t_ region_parameter;///< applies to all cells, except those with catchment override
            std::map<size_t, parameter_t_> catchment_parameters;///<  for each catchment parameter is possible

            std::vector<bool> catchment_filter;///<if active (alias .size()>0), only calc if catchment_filter[catchment_id] is true.

            size_t n_catchments;///< optimized//extracted as max(cell.geo.catchment_id())+1 in run interpolate

            void clone(const region_model& c) {
                // First, clear own content
                ncore = c.ncore;
                time_axis = c.time_axis;
                catchment_filter = c.catchment_filter;
                n_catchments = c.n_catchments;
                catchment_parameters.clear();
                // Then, clone from c
                cells = cell_vec_t_(new cell_vec_t(*(c.cells)));
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
                ncore = thread::hardware_concurrency()*4;
            }
            region_model(std::shared_ptr<std::vector<C> >& cells,
                         const parameter_t &region_param,
                         const std::map<size_t, parameter_t>& catchment_parameters)
             : cells(cells) {
                set_region_parameter(region_param);
                for(const auto & pair:catchment_parameters)
                     set_catchment_parameter(pair.first, pair.second);
                ncore = thread::hardware_concurrency()*4;
            }
            region_model(const region_model& model) { clone(model); }
			#ifndef SWIG
			region_model& operator=(const region_model& c) {
                if (&c != this)
                    clone(c);
                return *this;
            }
			#endif
            timeaxis_t time_axis; ///<The time_axis as set from run_interpolation, determines the axis for run()..
            size_t ncore = 0; ///<< defaults to 4x hardware concurrency, controls number of threads used for cell processing
            /** \brief compute and return number of catchments inspecting call cells.geo.catchment_id() */
            size_t number_of_catchments() const {
                size_t max_catchment_id=0;
                for(const auto&c:*cells)
                    if(c.geo.catchment_id()>max_catchment_id) max_catchment_id=c.geo.catchment_id();
                return max_catchment_id+1;
            }
            /** \brief run_interpolation interpolates region_environment temp,precip,rad.. point sources
            * to a value representative for the cell.mid_point().
            *
            * \note Prior to running all cell.env_ts.xxx are reset to zero, and have a length of time_axis.size().
            *
            * Only supplied vectors of temp, precip etc. are interpolated, thus
            * the user of the class can choose to put in place distributed series in stead.
            *
            * \tparam RE \ref region_environment type
            * \tparam IP interpolation parameters
            *
            * \param interpolation_parameter contains wanted parameters for the interpolation
            * \param time_axis should be equal to the \ref timeaxis the \ref region_model is prepared running for.
            * \param env contains the \ref region_environment type
            * \return void
            *
            */
            template < class RE, class IP>
            void run_interpolation(const IP& interpolation_parameter, const timeaxis_t& time_axis, const RE& env) {
                #ifndef SWIG
                for(auto&c:*cells){
                    c.init_env_ts(time_axis);
                }
                n_catchments = number_of_catchments();// keep this/assume invariant..
                this->time_axis = time_axis;
                using namespace shyft::core;
                using namespace std;
                namespace idw = shyft::core::inverse_distance;
                namespace btk = shyft::core::bayesian_kriging;


                typedef shyft::timeseries::average_accessor<typename RE::temperature_t::ts_t, timeaxis_t> temperature_tsa_t;
                typedef shyft::timeseries::average_accessor<typename RE::precipitation_t::ts_t, timeaxis_t> precipitation_tsa_t;
                typedef shyft::timeseries::average_accessor<typename RE::radiation_t::ts_t, timeaxis_t> radiation_tsa_t;
                typedef shyft::timeseries::average_accessor<typename RE::wind_speed_t::ts_t, timeaxis_t> wind_speed_tsa_t;
                typedef shyft::timeseries::average_accessor<typename RE::rel_hum_t::ts_t, timeaxis_t> rel_hum_tsa_t;


                typedef idw_compliant_geo_point_ts< typename RE::temperature_t, temperature_tsa_t, timeaxis_t> idw_compliant_temperature_gts_t;

				typedef idw_compliant_geo_point_ts< typename RE::precipitation_t, precipitation_tsa_t, timeaxis_t> idw_compliant_precipitation_gts_t;
				typedef idw_compliant_geo_point_ts< typename RE::radiation_t, radiation_tsa_t, timeaxis_t> idw_compliant_radiation_gts_t;
				typedef idw_compliant_geo_point_ts< typename RE::wind_speed_t, wind_speed_tsa_t, timeaxis_t> idw_compliant_wind_speed_gts_t;
				typedef idw_compliant_geo_point_ts< typename RE::rel_hum_t, rel_hum_tsa_t, timeaxis_t> idw_compliant_rel_hum_gts_t;

				typedef idw::temperature_model  <idw_compliant_temperature_gts_t, cell_t, typename IP::idw_temperature_parameter_t, geo_point, idw::temperature_gradient_scale_computer> idw_temperature_model_t;
				typedef idw::precipitation_model<idw_compliant_precipitation_gts_t, cell_t, typename IP::idw_precipitation_parameter_t, geo_point> idw_precipitation_model_t;
				typedef idw::radiation_model    <idw_compliant_radiation_gts_t, cell_t, typename IP::idw_parameter_t, geo_point> idw_radiation_model_t;
				typedef idw::wind_speed_model   <idw_compliant_wind_speed_gts_t, cell_t, typename IP::idw_parameter_t, geo_point> idw_windspeed_model_t;
				typedef idw::rel_hum_model      <idw_compliant_rel_hum_gts_t, cell_t, typename IP::idw_parameter_t, geo_point> idw_relhum_model_t;

				typedef  shyft::timeseries::average_accessor<typename RE::temperature_t::ts_t, timeaxis_t> btk_tsa_t;

				// Allocate memory for the source_destinations, put in the reference to the parameters:
				// Run one thread for each optional interpolation
				//  notice that if a source is nullptr, then we leave the allocated cell.level signal to fillvalue 0.0
				//  the intention is that the orchestrator at the outside could provide it's own ready-made
				//  interpolated/distributed signal, e.g. temperature input from arome-data

				auto btkx = async(launch::async, [&]() {
					if (env.temperature != nullptr) {
                        if(env.temperature->size()>1) {
                            if(interpolation_parameter.use_idw_for_temperature) {
                                idw::run_interpolation<idw_temperature_model_t, idw_compliant_temperature_gts_t>(
                                        time_axis, *env.temperature, interpolation_parameter.temperature_idw, *cells,
                                        [](cell_t &d, size_t ix, double value) { d.env_ts.temperature.set(ix, value); }
                                );
                            } else {
                                btk::btk_interpolation<btk_tsa_t>(
                                    begin(*env.temperature), end(*env.temperature), begin(*cells), end(*cells),
                                    time_axis, interpolation_parameter.temperature
                                );
                            }
                        } else {
                            // just one temperature ts. just a a clean copy to destinations
                            btk_tsa_t tsa((*env.temperature)[0].ts, time_axis);
                            typename cell_t::env_ts_t::temperature_ts_t temp_ts(time_axis, 0.0);
                            for(size_t i=0;i<time_axis.size();++i) {
                                temp_ts.set(i, tsa.value(i));
                            }
                            for(auto& c:*cells) {
                                c.env_ts.temperature=temp_ts;
                            }
                        }
                    }
                });

                auto idw_precip = async(launch::async, [&]() {
                    if (env.precipitation != nullptr)
                        idw::run_interpolation<idw_precipitation_model_t, idw_compliant_precipitation_gts_t>(
                        time_axis, *env.precipitation, interpolation_parameter.precipitation, *cells,
                        [](cell_t &d, size_t ix, double value) { d.env_ts.precipitation.set(ix, value); }
                    );
                });

                auto idw_radiation = async(launch::async, [&]() {
                    if (env.radiation != nullptr)
                        idw::run_interpolation<idw_radiation_model_t, idw_compliant_radiation_gts_t>(
                        time_axis, *env.radiation, interpolation_parameter.radiation, *cells,
                        [](cell_t &d, size_t ix, double value) { d.env_ts.radiation.set(ix, value); }
                    );
                });

                auto idw_wind_speed = async(launch::async, [&]() {
                    if (env.wind_speed != nullptr)
                        idw::run_interpolation<idw_windspeed_model_t, idw_compliant_wind_speed_gts_t>(
                        time_axis, *env.wind_speed, interpolation_parameter.wind_speed, *cells,
                        [](cell_t &d, size_t ix, double value) { d.env_ts.wind_speed.set(ix, value); }
                    );
                    //else
                    //    for_each(begin(*cells), end(*cells), [this] (cell_t& d) { d.constant_wind_speed = region_ws; });
                });

                auto idw_rel_hum = async(launch::async, [&]() {
                    if (env.rel_hum != nullptr)
                        idw::run_interpolation<idw_relhum_model_t, idw_compliant_rel_hum_gts_t>(
                        time_axis, *env.rel_hum, interpolation_parameter.rel_hum, *cells,
                        [](cell_t &d, size_t ix, double value) { d.env_ts.rel_hum.set(ix, value); }
                    );
                    //else, set the constant ? as ts.
                    //  for_each(begin(*cells), end(*cells), [this] (cell_t& d) { d.constant_rel_hum = region_rel_hum; });
                });

                btkx.get();
                idw_precip.get();
                idw_radiation.get();
                idw_wind_speed.get();
                idw_rel_hum.get();
                #endif
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
            *  -#
            *  \note the call
            * \tparam TA time-axis, \ref shyft::timeseries::timeaxis
            * \param time_axis of type TA
            * \return void
            *
            */
            void run_cells(size_t thread_cell_count=0) {
                if(thread_cell_count==0) {
                    if(ncore==0) ncore=4;// a reasonable minimum..
                    thread_cell_count = size() <= ncore ? 1 : size()/ncore;
                }
                if(! (time_axis.size()>0))
                    throw runtime_error("region_model::run with invalid time_axis invoked");
                parallel_run(time_axis, begin(*cells), end(*cells), thread_cell_count);
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
                \param a reference to the parameter that will be kept for those cells
            */
            void set_catchment_parameter(size_t catchment_id, const parameter_t & p) {
                if (catchment_parameters.find(catchment_id) == catchment_parameters.end()) {
                    auto shared_p = parameter_t_(new parameter_t(p));// add to map, a copy of p
                    catchment_parameters[catchment_id] = shared_p;
                    for(auto &c:*cells)
                        if (c.geo.catchment_id() == catchment_id)
                            c.set_parameter(shared_p);
                } else {
                    *(catchment_parameters[catchment_id]) = p; //copy values into existing parameters
                }
            }


            /** \brief remove a catchment specific parameter override, if it exists.
            */
            void remove_catchment_parameter(size_t catchment_id) {
                auto it = catchment_parameters.find(catchment_id);
                if (it != catchment_parameters.end()) {
                    catchment_parameters.erase(catchment_id);// get rid of it, and update the affected cells with the global parameter
                    for(auto & c:*cells)
                        if (c.geo.catchment_id() == catchment_id)
                            c.set_parameter(region_parameter);
                }
            }
            /** \brief returns true if there exist a specific parameter override for the specified 0-based catchment_id*/
            bool has_catchment_parameter(size_t catchment_id) const {
                return catchment_parameters.find(catchment_id) != catchment_parameters.end();
            }

            /** \brief return the parameter valid for specified catchment_id, or global parameter if not found.
            * \note Be aware that if you change the returned parameter, it will affect the related cells.
            * \param catchment_id 0 based catchment id as placed on each cell
            * \returns reference to the real parameter structure for the catchment_id if exists, otherwise the global parameters
            */
            parameter_t& get_catchment_parameter(size_t catchment_id) const {
                auto search=catchment_parameters.find(catchment_id);
                if ( search != catchment_parameters.end())
                    return *((*search).second);
                else
                    return *region_parameter;
            }
            /** \brief set/reset the catchment based calculation filter. This affects what get simulate/calculated during
             * the run command. Pass an empty list to reset/clear the filter (i.e. no filter).
             *
             * \param catchment_id_list is a (zero-based) catchment id vector
             *
             */
            void set_catchment_calculation_filter(const std::vector<int>& catchment_id_list) {
                if (catchment_id_list.size()) {
                    int mx = 0;
                    for (auto i : catchment_id_list) mx = i > mx ? i : mx;
                    catchment_filter = vector<bool>(mx + 1, false);
                    for (auto i : catchment_id_list) catchment_filter[i] = true;
                } else {
                    catchment_filter.clear();
                }
            }

            /** \brief using the catchment_calculation_filter to decide if discharge etc. are calculated.
             * \param cid  catchment id
             * \returns true if catchment id is calculated during runs
             */
            bool is_calculated(size_t cid) const { return catchment_filter.size() == 0 || (catchment_filter[cid]); }

            /** \brief collects current state from all the cells
            * \note that catchment filter can influence which states are calculated/updated.
            *\param end_states a reference to the vector<state_t> that are filled with cell state, in order of appearance.
            */
            void get_states(std::vector<state_t>& end_states) const {
                end_states.clear();end_states.reserve(std::distance(begin(*cells), end(*cells)));
                for(const auto& cell:*cells) end_states.emplace_back(cell.state);
            }

            /**\brief set current state for all the cells in the model.
             * \param states is a vector<state_t> of all states, must match size/order of cells.
             * \note throws runtime-error if states.size is different from cells.size
             */
            void set_states(const std::vector<state_t>& states) {
                if (states.size() != size())
                    throw runtime_error("Length of the state vector must equal number of cells");
                auto state_iter = begin(states);
                for(auto& cell:*cells) cell.set_state(*(state_iter++));
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
             * \param cachment_id to enable snow calibration for, -1 means turn on/off for all
             * \param on_or_off true|or false.
             * \note if the underlying cell do not support snow sca|swe collection, this
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
             * \tparam TSV a vector<timeseries> type, where timeseries supports:
             *  -# .ct(timeaxis_t, double) fills a series with 0.0 for all time_axis elements
             *  -# .add( const some_ts & ts)
             * \note the ts type should have proper move/copy etc. semantics
             *
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
                    if (is_calculated(c.geo.catchment_id()))
                        cr[c.geo.catchment_id()].add(c.rc.avg_discharge);
                }
            }

        protected:
            /** \brief parallell_run using a mid-point split + async to engange multicore execution
             *
             * \param time_axis forwarded to the cell.run(time_axis)
             * \param beg iterator to first cell in range
             * \param endc iterator to end cell in range (one past last element)
             */
            void single_run(const timeaxis_t& time_axis, cell_iterator beg, cell_iterator endc) {
                for(auto& cell:boost::make_iterator_range(beg,endc)) {
                     if (catchment_filter.size() == 0 || (cell.geo.catchment_id() < catchment_filter.size() && catchment_filter[cell.geo.catchment_id()]))
                        cell.run(time_axis);
                }
            }
            /** \brief uses async to execute the single_run, partitioning the cell range into thread-cell count
             *
             * \throws runtime_error if thread_cell_count is zero
             * \returns when all cells calculated
             * \param time_axis time-axis to use
             * \param beg start of cell range
             * \param end end of cell range
             * \param thread_cell_count number of cells given to each async thread
             */
            void parallel_run(const timeaxis_t& time_axis, cell_iterator beg, cell_iterator endc, size_t thread_cell_count) {
                size_t len = distance(beg, endc);
                if(len == 0)
                    return;
                if(thread_cell_count == 0)
                    throw runtime_error("parallel_run:cell pr thread is zero ");
                vector<future<void>> calcs;
                for (size_t i = 0; i < len;) {
                    size_t n = thread_cell_count;
                    if (i + n > len)
                        n = len - i;
                    calcs.emplace_back(
                        async(launch::async, [this, &time_axis, beg, n]() {
                            this->single_run(time_axis, beg, beg + n); }
                        )
                    );
                    beg = beg + n;
                    i = i + n;
                }
                for(auto &f:calcs)
                    f.get();
                return;
            }
        };





    } // core
} // shyft
