#pragma once

#ifdef SHYFT_NO_PCH
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <limits>
#include <future>
#include <utility>
#include <memory>
#include <stdexcept>
#include <future>
#include <mutex>
#include <boost/math/distributions/gamma.hpp>
// no serialize here: #include "core_pch.h"
#endif // SHYFT_NO_PCH

#include "geo_cell_data.h"
#include "time_axis.h"
#include "time_series.h"

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
             * Currently we use a gamma function for the shape:
             * <a ref="https://en.wikipedia.org/wiki/Gamma_distribution">gamma distribution</a>
             */
            struct uhg_parameter {
                uhg_parameter(double velocity=1.0,double alpha=7.0,double beta=0.0):velocity(velocity),alpha(alpha),beta(beta) {}
                double velocity= 1.0;///< in units of [m/s]
                double alpha=7.0; ///< gamma function alpha factor
                double beta =0.0; ///< base-line offset, added to pdf(gamma(alpha,1.0))
            };

            inline std::vector<double>  make_uhg_from_gamma(int n_steps, double alpha, double beta);// fwd decl

            ///< valid routing|river id, rid, must be >0. 0 or less than 0 is interpreted as null
            inline bool valid_routing_id(int rid ) {return rid>0;}

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
                // python binding help
                river()=default;
                river(const river&)=default;
                river(river&&)=default;
                river&operator=(const river&)=default;
                river&operator=(river&&)=default;
                // python support
                river(int id,shyft::core::routing_info downstream=shyft::core::routing_info(),uhg_parameter p=uhg_parameter()):
                    id(id),downstream(downstream),parameter(p) {}
                int id;///< self.id, >0 means valid id, <=0 means null
                shyft::core::routing_info downstream;///< downstream routing info, where to, and the static distance measure
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
                    return make_uhg_from_gamma(n_steps, parameter.alpha, parameter.beta);
                }
            };

            /**
             * Convenient function to get the 'frozen'/computed ts values out,
             * \tparam Ts any ts supporting .size() and .value().
             * \param ts time-series for which we extract all values
             * \return a std::vector<double> with the values
             * \note maybe candidate for time_series, but keep it here for now
             */
            template <class Ts>
            inline std::vector<double> ts_values(const Ts& ts) {
                std::vector<double> r; r.reserve(ts.size());
                for (size_t i = 0;i < ts.size();++i) r.push_back(ts.value(i));
                return r;
            }

            /** Provide a class that enable safe manipulation of rivers.
             * ensure no cycles, no duplicate object-id's etc.
             * Partly motivated by exposure to python, providing routing-id etc.
             * to enable external simplified description and association.
             */
            struct river_network {
                std::map<int,river> rid_map;///< user operates on rivers based on river-id, we map this here

                void check_rid(int rid, bool must_exist=true) const {
                    if(!valid_routing_id(rid)) throw std::runtime_error("valid river|routing id must be >0");
                    if(must_exist)
                        if(rid_map.find(rid)==rid_map.end())
                            throw std::runtime_error(std::string("the supplied river|routing id is not registered/does not exist, id=")+std::to_string(rid));
                }

                /** verify there is no directed cycles in the river-network */
                bool network_contains_directed_cycle() const {
                    std::map<int,bool> not_visited;
                    for(auto it=rid_map.cbegin();it!=rid_map.cend();++it)
                        not_visited[it->first]=false;
                    for(auto it=rid_map.cbegin();it!=rid_map.cend();++it) {
                        auto visited=not_visited;// not optimal, but simple approach
                        visited[it->first]=true;// hitting any node visited during search
                        auto downstream_id=it->second.downstream.id;// indicates a cycle
                        while(valid_routing_id(downstream_id)) { // follow downstream
                            if(visited[downstream_id])
                                return true;// got you, this is a cycle!
                            visited[downstream_id]=true;// mark it as visited
                            downstream_id=rid_map.find(downstream_id)->second.downstream.id;// continue downstream
                        }
                    }
                    return false;// if we reach here, no cycles detected.
                }
                // help python exposure
                river_network()=default;
                river_network(const river_network&)=default;
                ~river_network()=default;
                river_network(river_network&&c)=default;
                river_network& operator=(const river_network&c)=default;
                river_network& operator=(river_network&&c)=default;
                ///< .add(river), - includes possible connection, will fail if dest. do not exis
                river_network& add(const river &r) {
                    check_rid(r.id,false);
                    if(rid_map.find(r.id)!=rid_map.end()) throw std::runtime_error("the supplied river id is already registered");
                    if(r.id == r.downstream.id) throw std::runtime_error("the supplied river.downstream.id should not point to self (cycle!)");
                    if(valid_routing_id(r.downstream.id) && rid_map.find(r.downstream.id) ==rid_map.end()) throw std::runtime_error("the river.downstream.id does not yet exist in the network, please downstream river-segments first");
                    rid_map[r.id]=r;
                    if(network_contains_directed_cycle()) {
                        rid_map.erase(r.id);
                        throw std::runtime_error("adding this river caused circular reference");
                    }
                    return *this;
                }
                ///< .remove_by_id(river-id)
                void remove_by_id(int rid) {
                    check_rid(rid);
                    auto upstreams=upstreams_by_id(rid);
                    for(auto i:upstreams)
                        rid_map[i].downstream.id=0;// zero out references (could also bypass,but leave it for now)
                    rid_map.erase(rid);
                }
                ///< .river(river-id).. --> the river-object itself, r/w
                river& river_by_id(int rid) {
                    check_rid(rid);
                    return rid_map[rid];
                }
                river river_by_id2(int rid) { check_rid(rid); return rid_map[rid];}
                const river& river_by_id(int rid) const {
                    check_rid(rid);
                    return rid_map.find(rid)->second;
                }

                std::vector<int> upstreams_by_id(int rid) const {
                    check_rid(rid);
                    std::vector<int> rids;
                    for(auto it=rid_map.cbegin();it!=rid_map.cend();++it) {
                        if(it->second.downstream.id==rid)
                            rids.push_back(it->first);
                    }
                    return rids;
                }
                std::vector<int> all_upstreams_by_id(int rid) const {
                    check_rid(rid);
                    auto rids=upstreams_by_id(rid);
                    auto n =rids.size();//since we are going to extend it in loop below
                    for (size_t i=0;i<n;++i) {
                        auto xtra = all_upstreams_by_id(rids[i]);
                        for (auto x : xtra)
                            rids.push_back(x);
                    }
                    return rids;
                }

                int downstream_by_id(int rid) const {
                    check_rid(rid);
                    return rid_map.find(rid)->second.downstream.id;
                }

                void set_downstream_by_id(int rid,int downstream_rid) {
                    check_rid(rid);
                    if(valid_routing_id(downstream_rid))
                        check_rid(downstream_rid);// maybe weak error reporting
                    // first disconnect current rid
                    unsigned rid_old_downstream=rid_map[rid].downstream.id;
                    rid_map[rid].downstream.id=downstream_rid;
                    if(network_contains_directed_cycle()) {
                        rid_map[rid].downstream.id=rid_old_downstream;// restore it
                        throw std::runtime_error("connection would create a cycle, not allowed");
                    }
                }
            };

            /** A routing model
             *
             * Based on modelling the routing using repeated convolution of a unit hydro-graph.
             * First from the lateral local cells that feeds into the routing-point (river/creek).
             * Then add up the flow from the upstream rivers(they might or might not have local-cells, up-streams rivers etc.)
             * Then finally compute output as the convolution using the uhg of the sum_inflow to the river.
             *
             * \tparam C
             *  Cell type that should provide
             *  -# C::rc.avg_discharge the current core time-series representation, currently point_ts<fixed_dt>..
             *
             * \note implementation:
             *    technically we are currently flattening out the ts-expression tree by computing the full
             *    point representation of every flow in the directed graph.
             *    Later we could utilize a dynamic dispatch to to build the recursive
             *    accumulated expression tree at any river in the routing graph. This could
             *    improve performance and resource usage in certain scenarios.
             *
             * \note usage and further work:
             *    In the current form, we can use the routing::model<C> as a standalone class, which is
             *    nice during testing. However, the time_axis (ta) member is kind of not elegant
             *    in it's current form, since it invite to duplicate data from the region-model.
             *    Thus, the current plan is to use it as a temporary object, for calculations,
             *    provided by the member-functions of region_model. The region model then keep
             *    the river-routing network, cells, etc. and this class is constructed on request,
             *    with life-time equal to  the scope of the corresponding function.
             *
             */

            template<class C>
            struct model {
                typedef ::shyft::time_series::point_ts<::shyft::time_axis::fixed_dt> rts_t; // the result computed in the cell.rc.avg_discharge [m3/s]
                //typedef typename C::output_m3s_t ots_t; // the output result from the cell, it's a convolution_w_ts<rts_t>..
                //typedef typename shyft::time_series::uniform_sum_ts<ots_t> sum_ts_t; // could be possible using api::apoint_ts

                //std::map<int, river> river_map; ///< keeps structure and routing properties
                std::shared_ptr<river_network> rivers;
                std::shared_ptr<std::vector<C>> cells; ///< shared with the region_model !
                time_axis::fixed_dt ta;///< shared with the region_model,  should be the simulation time-axis

                model(const std::shared_ptr<river_network> &rivers,
                      const std::shared_ptr<std::vector<C>>& cells,
                      const time_axis::fixed_dt& ta):rivers(rivers),cells(cells),ta(ta) {}
                model(const river_network& rivers,// hmm. clumsy, the region model keeps no shared pointer, which is ok..
                      const std::shared_ptr<std::vector<C>>& cells,
                      const time_axis::fixed_dt& ta):rivers(std::make_shared<river_network>(rivers)),cells(cells),ta(ta) {}
                // constructors etc.
                model() = default;
                model(const model&)=default;
                model(model&&) = default;
                ~model() =default;
                model&operator=(const model&c) {
                    if(&c !=this) {
                        rivers=c.rivers;
                        cells=c.cells;// shallow
                        ta =c.ta;
                    }
                    return *this;
                }
                model&operator=(model &&c) {
                    rivers=std::move(c.rivers);
                    cells=std::move(c.cells);
                    ta =std::move(c.ta);
                    return *this;
                }

                // useful functions:
                void verify_cell_river_connections() const {
                    for(const auto&c:*cells) {
                        if(c.geo.routing.id>0)
                            rivers->check_rid(c.geo.routing.id);// throws if id does not exist
                    }
                }

                std::vector<double> cell_uhg(const C& c, utctimespan dt) const {
                    double steps = (c.geo.routing.distance / c.parameter->routing.velocity)/dt;// time = distance / velocity[s] // dt[s]
                    int n_steps = int(steps + 0.5);
                    return make_uhg_from_gamma(n_steps, c.parameter->routing.alpha, c.parameter->routing.beta);//std::vector<double>{0.1,0.5,0.2,0.1,0.05,0.030,0.020};
                }

                /** compute the cell_output, taking the cell-route to routing river into consideration
                 *
                 */
                time_series::convolve_w_ts<rts_t> cell_output_m3s(const C&c ) const {
                    // return discharge, notice that this function assumes that time_axis() do have a uniform delta() (requirement)
                    return time_series::convolve_w_ts<rts_t>(c.rc.avg_discharge,cell_uhg(c,ta.delta()),time_series::convolve_policy::USE_ZERO);
                }


                /** compute the local lateral inflow from connected shyft-cells into given river-id
                 *
                 */
                rts_t local_inflow(int node_id) const {
                    rts_t r(ta,0.0,time_series::POINT_AVERAGE_VALUE);// default null to null ts.
                    for (const auto& c : *cells) {
                        if (c.geo.routing.id == node_id) {
                            auto node_output_m3s (cell_output_m3s(c));
                            for (size_t t = 0;t < r.size();++t)
                                r.add(t, node_output_m3s.value(t));
                        }
                    }
                    return r;
                }

                /** Aggregate the upstream inflow that flows into this cell
                 * Notice that this is a recursive function that will go upstream
                 * and collect *all* upstream flow
                 */
                rts_t upstream_inflow(int node_id) const {
                    rts_t r(ta, 0.0, time_series::POINT_AVERAGE_VALUE);
                    auto upstream_ids=rivers->upstreams_by_id(node_id);
                    for(auto upstream_id:upstream_ids) {
                        auto flow_m3s= output_m3s(upstream_id);
                        for (size_t t = 0;t < ta.size();++t)
                            r.add(t, flow_m3s.value(t));

                    }
                    return r;
                }

                /** Utilizing the local_inflow and upstream_inflow function,
                 * calculate the output_m3s leaving the specified river.
                 * This is a walk in the park, since we can just use
                 * already existing (possibly recursive) functions to do the work.
                 */
                rts_t output_m3s(int node_id) const {
                    utctimespan dt = ta.delta(); // for now need to pick up delta from the sources
                    std::vector<double> uhg_weights = rivers->river_by_id(node_id).uhg(dt);
                    auto sum_input_m3s = local_inflow(node_id)+ upstream_inflow(node_id);
                    auto response = time_series::convolve_w_ts<decltype(sum_input_m3s)>(sum_input_m3s, uhg_weights, time_series::convolve_policy::USE_ZERO);
                    return rts_t(ta, ts_values(response), time_series::POINT_AVERAGE_VALUE); // flatten values
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
             * \param base offset added to the gdf(alpa,1.0)
             * \return unit hydro graph factors, normalized to sum 1.0
             */
            inline std::vector<double>  make_uhg_from_gamma(int n_steps, double alpha, double base) {
                using boost::math::gamma_distribution;
                double beta = 1.0;
                gamma_distribution<double> gdf(alpha, beta);
                std::vector<double> r;r.reserve(n_steps);
                if(n_steps>1) {
                    double s = 0.0;
                    double x_max = quantile(gdf,0.99);
                    double d = x_max / double(n_steps);
                    for (int i=0;i<n_steps;++i) {
                        double x=   d*i;
                        double y = std::max(0.0,pdf(gdf, x)+base);
                        s += y;
                        r.push_back(y);
                    }
                    if (s>0.0)
                        for (auto& y : r) y /= s;
                    else
                        for (auto& y:r) y=1/double(n_steps);
                }
                if (r.size() == 0) r.push_back(1.0);// at a minimum 1.0, no delay
                return r;
            };

        }
    }
}
