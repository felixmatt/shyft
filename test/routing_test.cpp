#include "test_pch.h"
#include "routing_test.h"
#include "core/timeseries.h"
#include "core/utctime_utilities.h"
#include "core/geo_cell_data.h"
#include "api/timeseries.h"
//#include <boost/math/distributions/gamma.hpp>
//#include <iomanip>

namespace shyft {
    namespace timeseries {


        // maybe a more convenient function to get the frozen values out,
        // then we don't have to write this more than once,
        // except for the rare cases where we would like to specialize it:

        template <class Ts>
        std::vector<double> ts_values(const Ts& ts) {
            std::vector<double> r; r.reserve(ts.size());
            for (size_t i = 0;i < ts.size();++i) r.push_back(ts.value(i));
            return std::move(r);
        }

    }
}
namespace shyft {
    namespace core {
        namespace routing {

            /** make_uhg_from_gamma a simple function to create a uhg (unit hydro graph) weight vector
             * containing n_steps, given the gamma shape factor alpha and beta.
             * ensuring the sum of the weight vector is 1.0
             * and that it has a min-size of one element (1.0)
             *
             * Later we can replace the implementation of this to depend on the configured parameters of the
             * model.
             *
             * There are some variations of how this is created, and how it is applied.
             * Notice that there are two stages in the routing:
             *  First from the cells to the first routing points,
             *  Then output form the routing points can be cascaded to form
             *  larger units (rivers etc), e.g. simulating a simple river-network.
             *
             *  One or more routing-points can be classified as 'observation routing points',
             *  that is, a place where we have an observation(or a simulated observation).
             *
             *
             * #1 Using cell.geo.routing velocity and distance along with catchment-specific shape parameters.
             *    routing down to the first routing point is then determined by each cell along with possibly
             *    catchment specific shape/routing parameters.
             *
             * #2 Skaugen: sum together all cell-responses that belong to a routing point, then
             *    use distance distribution profile to generate a uhg that together with convolution
             *    determines the response out from those cells to the first routing point
             *
             * #3 time-delay-zones: Group cells output to routing time-delay points,with no local delay.
             *   then use a response function that take the shape and time-delay characteristics for the group
             *   to the 'observation routing point'
             *
             *  A number of various routing methods can be utilized, we start out with a simple uhg,
             *  and convolution to create the output response.
             *
             */
            std::vector<double>  make_uhg_from_gamma(int n_steps, double alpha, double beta) {
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
            struct cell_parameter {
                double velocity=0.0;
                double alpha =3.0;
                double beta = 0.7;
            };
            /** cell_node, a workbench stand-in for a typical shyft::core::cell_model
             *
             * Just for emulating a cell-node that do have
             * the needed properties that we will require
             * later when promoting the stuff to cell_model
             *  either by explicit requirement, or by concept
             */
            template <class Ts>
            struct cell_node {
                typedef Ts ts_t;
                typedef typename  timeseries::convolve_w_ts<ts_t> output_m3s_t;
                cell_parameter parameter;
                geo_cell_data geo;
                ts_t discharge_m3s;

            };

            struct node_parameter {
                node_parameter(double velocity=1.0,double alpha=3.0,double beta=0.7):velocity(velocity),alpha(alpha),beta(beta) {}
                double velocity= 1.0;
                double alpha=3.0;
                double beta =0.7;
            };
            /**\brief A routing node
             *
             * The routing node have flow from
             * -# zero or more 'cell_nodes',  typically a cell_model type, lateral flow (.output_m3s())
             * -# zero or more upstream connected routing nodes, taking their inputs (.output_m3s())
             * then a routing node can be connected to a down-stream node,
             * providing a routing function (currently just a convolution of a uhg).
             *
             * This definition is recursive, and we use other means to ensure the routing graph
             * is directed and with no cycles.
             */
            struct node {
                // binding information
                //  not really needed at core level, as we could to only with ref's in the core
                //  but we plan to expose to python and external persistence models
                //  so providing some handles do make sense for now.
                int id;// self.id, >0 means valid id, 0= null
                shyft::core::routing_info downstream;
                node_parameter parameter;
                // here we could have a link to the observed time-series (can use the self.id to associate)
                std::vector<double> uhg(utctimespan dt) const {
                    double steps = (downstream.distance / parameter.velocity) / dt;// time = distance / velocity[s] // dt[s]
                    int n_steps = int(steps + 0.5);
                    return std::move(make_uhg_from_gamma(n_steps, parameter.alpha, parameter.beta));//std::vector<double>{0.1,0.5,0.2,0.1,0.05,0.030,0.020};
                }
            };

            /** A simple routing model
             *
             *
             * Based on shaping the routing using repeated convolution a unit hydro-graph.
             * First from the lateral local cells that feeds into the routing-point (creek).
             * Then add up the flow from the upstream routing points(they might or might not have local-cells, upstreams routes etc.)
             * Then finally compute output as the convolution using the uhg of the sum_inflow to the node.
             * \tparam C
             *  Cell type that should provide
             *  -# C::ts_t the current core time-series representation, currently point_ts<fixed_dt>..
             *  -# C::output_m3s_t the type of the call C::output_m3s() const (we could rater use result_of..)
             *
             * \note implementation:
             *    technically we are currently flattening out the ts-expression tree.
             *    later we could utilize a dynamic dispatch to to build the recursive
             *    accumulated expression tree at any node in the routing graph.
             *
             */

            template<class C>
            struct model {
                typedef typename C::ts_t rts_t; // the result computed in the cell.rc.avg_discharge [m3/s]
                typedef typename C::output_m3s_t ots_t; // the output result from the cell, it's a convolution_w_ts<rts_t>..
                //typedef typename shyft::timeseries::uniform_sum_ts<ots_t> sum_ts_t;

                std::map<int, node> node_map; ///< keeps structure and routing properties

                std::shared_ptr<std::vector<C>> cells; ///< shared with the region_model !
                timeseries::timeaxis ta;///< shared with the region_model,  should be simulation time-axis


                std::vector<double> cell_uhg(const C& c, utctimespan dt) const {
                    double steps = (c.geo.routing.distance / c.parameter.velocity)/dt;// time = distance / velocity[s] // dt[s]
                    int n_steps = int(steps + 0.5);
                    return std::move(make_uhg_from_gamma(n_steps, c.parameter.alpha, c.parameter.beta));//std::vector<double>{0.1,0.5,0.2,0.1,0.05,0.030,0.020};
                }

                timeseries::convolve_w_ts<rts_t> cell_output_m3s(const C&c ) const {
                    // return discharge, notice that this function assumes that time_axis() do have a uniform delta() (requirement)
                    return std::move(timeseries::convolve_w_ts<rts_t>(c.discharge_m3s,cell_uhg(c,ta.delta()),timeseries::convolve_policy::USE_ZERO));
                }
                /** compute the local lateral inflow from connected shyft-cells into given node-id
                 *
                 */
                rts_t local_inflow(int node_id) const {
                    rts_t r(ta,0.0,timeseries::POINT_AVERAGE_VALUE);// default null to null ts.
                    for (const auto& c : *cells) {
                        if (c.geo.routing.id == node_id) {
                            auto node_output_m3s = cell_output_m3s(c);
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
                    for (const auto& i : node_map) {
                        if (i.second.downstream.id == node_id) {
                            auto flow_m3s = output_m3s(i.first);
                            for (size_t t = 0;t < ta.size();++t)
                                r.add(t, flow_m3s.value(t));
                        }
                    }
                    return std::move(r);
                }

                /** Utilizing the local_inflow and upstream_inflow function,
                 * calculate the output_m3s leaving the specified node.
                 * This is a walk in the park, since we can just use
                 * already existing (possibly recursive) functions to do the work.
                 */
                rts_t output_m3s(int node_id) const {
                    utctimespan dt = ta.delta(); // for now need to pick up delta from the sources
                    std::vector<double> uhg_weights = node_map.at(node_id).uhg(dt);
                    auto sum_input_m3s = local_inflow(node_id)+ upstream_inflow(node_id);
                    auto response = timeseries::convolve_w_ts<decltype(sum_input_m3s)>(sum_input_m3s, uhg_weights, timeseries::convolve_policy::USE_ZERO);
                    return std::move(rts_t(ta, timeseries::ts_values(response), timeseries::POINT_AVERAGE_VALUE)); // flatten values
                }

            };

        }
    }
}



void routing_test::test_hydrograph() {
    //
    //  Just for playing around with sum of routing node response
    //

}

//-- consider using dlib graph to do the graph stuff

//#include <dlib/graph_utils.h>
//#include <dlib/graph.h>
//#include <dlib/directed_graph.h>

void routing_test::test_routing_model() {
    using namespace shyft::core;
    using ta_t = shyft::time_axis::fixed_dt;
    using ts_t = shyft::timeseries::point_ts<ta_t>;
    using cell_t = routing::cell_node<ts_t>;
    // setup a simple network
    // a->b-> d
    //    c-/
    //
    // node a and c do have local inflow through cell_t (alias shyft region model cell)
    // node b is just a routing transport node
    // node d is an endpoint node, observation point routing node where we
    // would like to observed the local_inflow + the upstream_inflow
    // --
    // for now we just set routing parameters to
    // values suitable for demo and verification
    //--
    calendar utc;
    ta_t ta(utc.time(2016, 1, 1), deltahours(1), 24);
    int a_id = 1;
    int b_id = 2;
    int c_id = 3;
    int d_id = 4;

    // build the shyft region model cells (we use local types here, so we have maximum control)
    auto cells = std::make_shared<std::vector<cell_t>>();
    cell_t cx;
    cx.geo.routing.id = a_id;
    cx.geo.routing.distance = 10000;
    cx.parameter.velocity = cx.geo.routing.distance / (10 * 3600.0);// takes 10 hours to propagate the distance
    cx.discharge_m3s= ts_t(ta, 0.0, shyft::timeseries::POINT_AVERAGE_VALUE);
    cx.discharge_m3s.set(0, 10.0);// set 10 m3/s at timestep 0.
    cells->push_back(cx); //ship it to cell-vector
    cx.parameter.velocity = cx.geo.routing.distance / (7 * 3600.0);// takes 7 hours to propagate the distance
    cx.discharge_m3s = ts_t(ta, 0.0, shyft::timeseries::POINT_AVERAGE_VALUE);
    cx.discharge_m3s.set(0, 7.0);
    cx.discharge_m3s.set(6, 6.0); // a second pulse after 6 hours
    cells->push_back(cx);
    cx.geo.routing.id = c_id;// route it to cell c
    cx.parameter.velocity = cx.geo.routing.distance / (24 * 3600.0);// takes 14 hours to propagate the distance
    cx.discharge_m3s = ts_t(ta, 0.0, shyft::timeseries::POINT_AVERAGE_VALUE);
    cx.discharge_m3s.set(0, 50.0);// just one large pulse, that will spread over 24 hours
    cells->push_back(cx);

    // build the routing network as described,
    routing::node a{ a_id,routing_info(b_id, 2*3600.0),routing::node_parameter(1.0) };// two hour delay from
    routing::node b{ b_id,routing_info(d_id, 0),routing::node_parameter(1.0) }; // give zero delay
    routing::node c{ c_id,routing_info(d_id, 3600.0),routing::node_parameter(1.0) };// give one hour delay
    routing::node d{ d_id,routing_info(0) }; // here is our observation point node

    routing::model<cell_t> m;
    m.cells = cells;
    m.ta = ta;
    m.node_map[a.id]=a;
    m.node_map[b.id]=b;
    m.node_map[c.id]=c;
    m.node_map[d.id]=d;

    /// now, with the model in place, including some fake-timeseries at cell-level, we can expect things to happen:
    // for now just print out the response
    auto observation_m3s = m.local_inflow(d_id) + m.upstream_inflow(d_id);// this arrives into node d:
    std::cout << "Resulting response at observation node d:\n";
    std::cout << "timestep\t d.flow.[m3s]\n";
    for (size_t t = 0; t < observation_m3s.size();++t) {
        std::cout << t << "\t" << std::setprecision(2) << observation_m3s.value(t) << std::endl;
    }

#if 0
    // this is what we can easily do with dlib
    //
    dlib::directed_graph<routing::node>::kernel_1a_c md;
    md.set_number_of_nodes(4);
    md.add_edge(0,1);
    md.add_edge(1,3);
    md.add_edge(2,3);
    md.node(0).data=a;
    md.node(1).data=b;
    md.node(2).data=c;
    md.node(3).data=d;

    TS_ASSERT_EQUALS( md.node(3).number_of_parents(),2);
    TS_ASSERT_EQUALS( md.node(0).number_of_parents(),0);
    TS_ASSERT_EQUALS( md.node(0).number_of_children(),1);
    TS_ASSERT_EQUALS( md.node(3).number_of_children(),0);
#endif
}
