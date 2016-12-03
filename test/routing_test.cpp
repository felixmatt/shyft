#include "test_pch.h"
#include "routing_test.h"
#include "core/timeseries.h"
#include "core/utctime_utilities.h"
#include "core/geo_cell_data.h"
#include "core/routing.h"
#include "api/timeseries.h"

namespace shyft {
    namespace core {
        namespace routing {



            /**just a mock for the relevant cell parameter used in cell_node below*/
            struct cell_parameter {
                uhg_parameter routing_uhg;
            };

            /** cell_node, a workbench stand-in for a typical shyft::core::cell_model
             *
             * Just for emulating a cell-river that do have
             * the needed properties that we will require
             * later when promoting the stuff to cell_model
             *  either by explicit requirement, or by concept
             */
            template <class Ts>
            struct cell_node {
                typedef Ts ts_t;
                typedef typename  timeseries::convolve_w_ts<ts_t> output_m3s_t;
                std::shared_ptr<cell_parameter> parameter;
                geo_cell_data geo;
                //ts_t discharge_m3s;
                struct resource_collector {
                    ts_t avg_discharge;
                };
                resource_collector rc;
            };

        }
    }
}



void routing_test::test_hydrograph() {
    //
    //  Just for playing around with sum of routing river response
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
    // river a and c do have local inflow through cell_t (alias shyft region model cell)
    // river b is just a routing transport river
    // river d is an endpoint river, observation point routing river where we
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
    cx.parameter = std::make_shared<routing::cell_parameter>();
    cx.parameter->routing_uhg.velocity = cx.geo.routing.distance / (10 * 3600.0);// takes 10 hours to propagate the distance
    cx.rc.avg_discharge= ts_t(ta, 0.0, shyft::timeseries::POINT_AVERAGE_VALUE);
    cx.rc.avg_discharge.set(0, 10.0);// set 10 m3/s at timestep 0.
    cells->push_back(cx); //ship it to cell-vector
    cx.parameter = std::make_shared<routing::cell_parameter>();
    cx.parameter->routing_uhg.velocity = cx.geo.routing.distance / (7 * 3600.0);// takes 7 hours to propagate the distance
    cx.rc.avg_discharge = ts_t(ta, 0.0, shyft::timeseries::POINT_AVERAGE_VALUE);
    cx.rc.avg_discharge.set(0, 7.0);
    cx.rc.avg_discharge.set(6, 6.0); // a second pulse after 6 hours
    cells->push_back(cx);
    cx.geo.routing.id = c_id;// route it to cell c
    cx.parameter = std::make_shared<routing::cell_parameter>();
    cx.parameter->routing_uhg.velocity = cx.geo.routing.distance / (24 * 3600.0);// takes 14 hours to propagate the distance
    cx.rc.avg_discharge = ts_t(ta, 0.0, shyft::timeseries::POINT_AVERAGE_VALUE);
    cx.rc.avg_discharge.set(0, 50.0);// just one large pulse, that will spread over 24 hours
    cells->push_back(cx);

    // build the routing network as described,
    routing::river a{ a_id,routing_info(b_id, 2*3600.0),routing::uhg_parameter(1.0) };// two hour delay from
    routing::river b{ b_id,routing_info(d_id, 0),routing::uhg_parameter(1.0) }; // give zero delay
    routing::river c{ c_id,routing_info(d_id, 3600.0),routing::uhg_parameter(1.0) };// give one hour delay
    routing::river d{ d_id,routing_info(0) }; // here is our observation point river

    routing::model<cell_t> m;
    m.cells = cells;
    m.ta = ta;
    m.river_map[a.id]=a;
    m.river_map[b.id]=b;
    m.river_map[c.id]=c;
    m.river_map[d.id]=d;

    /// now, with the model in place, including some fake-timeseries at cell-level, we can expect things to happen:
    // fto establish regression, uncomment and print out out the response
    auto observation_m3s = m.local_inflow(d_id) + m.upstream_inflow(d_id);// this arrives into river d:
    //std::cout << "Resulting response at observation river d:\n";
    //std::cout << "timestep\t d.flow.[m3s]\n";
    //for (size_t t = 0; t < observation_m3s.size();++t) {
    //    std::cout <<std::setprecision(4) << observation_m3s.value(t) << ",";//std::endl;
    //}
    double expected_m3s[]= {3.9,4.987,5.4,5.421,5.163,4.68,5.083,5.012,4.633,4.028,3.676,3.213,2.629,2.481,2.313,2.127,1.924,1.704,1.468,1.215,0.9441,0.6551,0.3445,4.736e-15};
    for(size_t i=0;i<observation_m3s.size();++i)
        TS_ASSERT_DELTA(observation_m3s.value(i),expected_m3s[i],0.001);

#if 0
    double expected_m3s= {3.9,4.987,5.4,5.421,5.163,4.68,5.083,5.012,4.633,4.028,3.676,3.213,2.629,2.481,2.313,2.127,1.924,1.704,1.468,1.215,0.9441,0.6551,0.3445,4.736e-15};
    // this is what we can easily do with dlib
    //
    dlib::directed_graph<routing::river>::kernel_1a_c md;
    md.set_number_of_nodes(4);
    md.add_edge(0,1);
    md.add_edge(1,3);
    md.add_edge(2,3);
    md.node(0).data=a;
    md.node(1).data=b;
    md.node(2).data=c;
    md.node(3).data=d;

    TS_ASSERT_EQUALS( md.river(3).number_of_parents(),2);
    TS_ASSERT_EQUALS( md.river(0).number_of_parents(),0);
    TS_ASSERT_EQUALS( md.river(0).number_of_children(),1);
    TS_ASSERT_EQUALS( md.river(3).number_of_children(),0);
#endif
}
