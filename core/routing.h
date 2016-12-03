#pragma once
namespace shyft {
    namespace core {
        namespace routing {

            /**
             *  There are two stages in the routing:
             *
             *  1) Cell to river routing
             *
             *  Routing of water-flow from the cells to the closest river routing object, providing
             *  the lateral inflow to the river.
             *
             *  In this stage, using cell.geo.routing velocity and distance along with
             *  catchment-specific uhg shape parameters.
             *  Thus, the routing down to the first routing point, river, is then determined by each cell along with possibly
             *    catchment specific shape/routing parameters.
             *
             *  2) River network routing
             *
             *  Routing from one river object to the next. In addition to lateral inflow from shyft-cells,
             *  a river takes the output from upstream rivers.
             *  The sum of these two flows are then passed on to the downstream (if any) river.
             *
             *  This allow the user to configure and setup larger river networks to fit the purpose.
             *
             *  For some of the rivers, we might have observation of the flow, represented as time-series.
             *  These time-series can be utilized to calibrate/tune the parameters of the complete shyft-model.
             *
             *
             *  This stage of routing allow for several strategies, like
             *
             *  Skaugen: sum together all cell-responses that belong to a routing point, then
             *    use distance distribution profile to generate a uhg that together with convolution
             *    determines the response out from those cells to the first routing point
             *
             *  Time-delay-zones: Group cells output to routing time-delay points,with no local delay.
             *   then use a response function that take the shape and time-delay characteristics for the group
             *   to the 'observation routing point'
             *
             *  A number of various routing methods can be utilized, but we start out with a simple uhg based approach,
             *  enriched with a generic topology.
             *
             */


            /** The unit hydro graph parameter contains sufficient
             * description to create a unit hydro graph, that have a shape
             * and a discretized 'time-length' according to the model time-step resolution.
             *
             */
            struct uhg_parameter {
                uhg_parameter(double velocity=1.0,double alpha=3.0,double beta=0.7):velocity(velocity),alpha(alpha),beta(beta) {}
                double velocity= 1.0;
                double alpha=3.0;
                double beta =0.7;
            };

            inline std::vector<double>  make_uhg_from_gamma(int n_steps, double alpha, double beta);// fwd decl




            /**\brief A river that we use for routing
             *
             * The routing river have flow from
             * -# zero or more 'cell_nodes',  typically a cell_model type, lateral flow,like cell.rc.average_discharge [m3/s]
             * -# zero or more upstream connected rivers, taking their .output_m3s()
             * then a routing river can *optionally* be connected to a down-stream river,
             * providing a routing function (currently just a convolution of a uhg).
             *
             * This definition is recursive, and we use other means to ensure the routing graph
             * is directed and with no cycles.
             */
            struct river {
                // binding information
                //  not really needed at core level, as we could to only with ref's in the core
                //  but we plan to expose to python and external persistence models
                //  so providing some handles do make sense for now.
                int id;// self.id, >0 means valid id, 0= null
                shyft::core::routing_info downstream;
                uhg_parameter parameter;///< We assume each river do have distinct parameter, so no shared pointer
                // here we could have a link to the observed time-series (can use the self.id to associate)

                /** create the hydro-graph, taking specified delta-t, dt,
                 * static hydrological distance as well as the shape parameters
                 * alpha and beta used to form the gamma-function.
                 * The length of the uhg (delay) is determined by the downstream-distance,
                 * and the velocity parameter. The shape of the uhg is determined by alpha&beta parameters.
                 */
                std::vector<double> uhg(utctimespan dt) const {
                    double steps = (downstream.distance / parameter.velocity) / dt;// time = distance / velocity[s] // dt[s]
                    int n_steps = int(steps + 0.5);
                    return std::move(make_uhg_from_gamma(n_steps, parameter.alpha, parameter.beta));
                }
            };

            /**
             * Convenient function to get the frozen values out,
             * maybe candidate for timeseries, but keep it here for now
             */
            template <class Ts>
            inline std::vector<double> ts_values(const Ts& ts) {
                std::vector<double> r; r.reserve(ts.size());
                for (size_t i = 0;i < ts.size();++i) r.push_back(ts.value(i));
                return std::move(r);
            }
            /** A routing model
             *
             *
             * Based on modelling the routing using repeated convolution of a unit hydro-graph.
             * First from the lateral local cells that feeds into the routing-point (river/creek).
             * Then add up the flow from the upstream rivers(they might or might not have local-cells, up-streams rivers etc.)
             * Then finally compute output as the convolution using the uhg of the sum_inflow to the river.
             *
             * \tparam C
             *  Cell type that should provide
             *  -# C::ts_t the current core time-series representation, currently point_ts<fixed_dt>..
             *  -# C::output_m3s_t the type of the call C::output_m3s() const (we could rater use result_of..)
             *
             * \note implementation:
             *    technically we are currently flattening out the ts-expression tree by computing the full
             *    point representation of every flow in the directed graph.
             *    Later we could utilize a dynamic dispatch to to build the recursive
             *    accumulated expression tree at any river in the routing graph. This could
             *    improve performance and resource usage in certain scenarios.
             *
             */

            template<class C>
            struct model {
                typedef typename C::ts_t rts_t; // the result computed in the cell.rc.avg_discharge [m3/s]
                //typedef typename C::output_m3s_t ots_t; // the output result from the cell, it's a convolution_w_ts<rts_t>..
                //typedef typename shyft::timeseries::uniform_sum_ts<ots_t> sum_ts_t; // could be possible using api::apoint_ts

                std::map<int, river> river_map; ///< keeps structure and routing properties

                std::shared_ptr<std::vector<C>> cells; ///< shared with the region_model !
                timeseries::timeaxis ta;///< shared with the region_model,  should be simulation time-axis


                std::vector<double> cell_uhg(const C& c, utctimespan dt) const {
                    double steps = (c.geo.routing.distance / c.parameter->routing_uhg.velocity)/dt;// time = distance / velocity[s] // dt[s]
                    int n_steps = int(steps + 0.5);
                    return std::move(make_uhg_from_gamma(n_steps, c.parameter->routing_uhg.alpha, c.parameter->routing_uhg.beta));//std::vector<double>{0.1,0.5,0.2,0.1,0.05,0.030,0.020};
                }

                /** compute the cell_output, taking the cell-route to routing river into consideration
                 *
                 */
                timeseries::convolve_w_ts<rts_t> cell_output_m3s(const C&c ) const {
                    // return discharge, notice that this function assumes that time_axis() do have a uniform delta() (requirement)
                    return std::move(timeseries::convolve_w_ts<rts_t>(c.rc.avg_discharge,cell_uhg(c,ta.delta()),timeseries::convolve_policy::USE_ZERO));
                }


                /** compute the local lateral inflow from connected shyft-cells into given river-id
                 *
                 */
                rts_t local_inflow(int node_id) const {
                    rts_t r(ta,0.0,timeseries::POINT_AVERAGE_VALUE);// default null to null ts.
                    for (const auto& c : *cells) {
                        if (c.geo.routing.id == node_id) {
                            auto node_output_m3s (cell_output_m3s(c));
                            for (size_t t = 0;t < r.size();++t)
                                r.add(t, node_output_m3s.value(t));
                        }
                    }
                    return std::move(r);
                }

                /** Aggregate the upstream inflow that flows into this cell
                 * Notice that this is a recursive function that will go upstream
                 * and collect *all* upstream flow
                 */
                rts_t upstream_inflow(int node_id) const {
                    rts_t r(ta, 0.0, timeseries::POINT_AVERAGE_VALUE);
                    for (const auto& i : river_map) {
                        if (i.second.downstream.id == node_id) {
                            auto flow_m3s = output_m3s(i.first);
                            for (size_t t = 0;t < ta.size();++t)
                                r.add(t, flow_m3s.value(t));
                        }
                    }
                    return std::move(r);
                }

                /** Utilizing the local_inflow and upstream_inflow function,
                 * calculate the output_m3s leaving the specified river.
                 * This is a walk in the park, since we can just use
                 * already existing (possibly recursive) functions to do the work.
                 */
                rts_t output_m3s(int node_id) const {
                    utctimespan dt = ta.delta(); // for now need to pick up delta from the sources
                    std::vector<double> uhg_weights = river_map.at(node_id).uhg(dt);
                    auto sum_input_m3s = local_inflow(node_id)+ upstream_inflow(node_id);
                    auto response = timeseries::convolve_w_ts<decltype(sum_input_m3s)>(sum_input_m3s, uhg_weights, timeseries::convolve_policy::USE_ZERO);
                    return std::move(rts_t(ta, ts_values(response), timeseries::POINT_AVERAGE_VALUE)); // flatten values
                }

            };

            /** make_uhg_from_gamma a simple function to create a uhg (unit hydro graph) weight vector
             * containing n_steps, given the gamma shape factor alpha and beta.
             * ensuring the sum of the weight vector is 1.0
             * and that it has a min-size of one element (1.0)
             *
             * Later we can replace the implementation of this to depend on the configured parameters of the
             * model.
             * \param n_steps number of time-steps, elements, in the vector
             * \param alpha the gamma_distribution gamma-factor
             * \param beta the gamma_distribution beta-factor
             * \return unit hydro graph factors, normalized to sum 1.0
             */
            inline std::vector<double>  make_uhg_from_gamma(int n_steps, double alpha, double beta) {
                using boost::math::gamma_distribution;
                gamma_distribution<double> gdf(alpha, beta);
                std::vector<double> r;r.reserve(n_steps);
                double s = 0.0;
                double d = 1.0 / double(n_steps);
                for (double q = d;q < 1.0; q += d) {
                    double x = quantile(gdf, q);
                    double y = pdf(gdf, x);
                    s += y;
                    r.push_back(y);
                }
                for (auto& y : r) y /= s;
                if (r.size() == 0) r.push_back(1.0);// at a minimum 1.0, no delay
                return std::move(r);
            };

        }
    }
}
