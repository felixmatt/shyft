#include "boostpython_pch.h"
#include "api/api.h"
#include "core/routing.h"

namespace expose {
    using namespace shyft::core;
    using namespace boost::python;

    void routing_path_info() {

        class_<routing_info>("RoutingInfo","Describe the hydrological distance and the id of the target routing element (river)")
         .def(init<optional<int,double>>(args("id","distance"),"create an object with the supplied parameters"))
         .def_readwrite("id",&routing_info::id,"id of the target,down-stream river")
         .def_readwrite("distance",&routing_info::distance,"the hydrological distance, in unit of [m]")
         ;
    }

    void routing_ugh_parameter() {

        class_<routing::uhg_parameter>("UHGParameter",
            "The Unit Hydro Graph Parameter contains sufficient\n"
            "description to create a unit hydro graph, that have a shape\n"
            "and a discretized 'time-length' according to the model time-step resolution.\n"
            "Currently we use a gamma function for the shape:\n"
            "<a ref='https://en.wikipedia.org/wiki/Gamma_distribution'>gamma distribution</a>\n")
          .def(init<optional<double,double,double>>(args("velocity","alpha","beta"),"a new object with specified parameters"))
            .def_readwrite("velocity",&routing::uhg_parameter::velocity,"default 1.0, unit [m/s]")
            .def_readwrite("alpha", &routing::uhg_parameter::alpha,"default 3.0,unit-less, ref. shape of gamma-function")
            .def_readwrite("beta", &routing::uhg_parameter::beta,"default 0.0, unit-less, added to pdf of gamma(alpha,1.0)")
        ;

        def("make_uhg_from_gamma",&routing::make_uhg_from_gamma,args("n_steps","alpha","beta"),
             "make_uhg_from_gamma a simple function to create a uhg (unit hydro graph) weight vector\n"
             "containing n_steps, given the gamma shape factor alpha and beta.\n"
             "ensuring the sum of the weight vector is 1.0\n"
             "and that it has a min-size of one element (1.0)\n"
             "n_steps: int, number of time-steps, elements, in the vector\n"
             "alpha: float, the gamma_distribution gamma-factor\n"
             "beta: float, the base-line, added to pdf(gamma(alpha,1))\n"
             "return unit hydro graph factors, normalized to sum 1.0\n"
        );
    }
    void routing_river() {
        class_<routing::river>("River",
            "A river that we use for routing, its a single piece of a RiverNetwork\n"
            "\n"
            " The routing river have flow from\n"
            " a) zero or more 'cell_nodes',  typically a cell_model type, lateral flow,like cell.rc.average_discharge [m3/s]\n"
            " b) zero or more upstream connected rivers, taking their .output_m3s()\n"
            " then a routing river can *optionally* be connected to a down-stream river,\n"
            " providing a routing function (currently just a convolution of a uhg).\n"
            "\n"
            " This definition is recursive, and we use RiverNetwork to ensure the routing graph\n"
            " is directed and with no cycles.\n"
            )
            .def(init<int,optional<routing_info,routing::uhg_parameter>>(args("id","downstream","parameter"),
                "a new object with specified parameters, notice that a valid river-id|routing-id must be >0"
                )
            )
            .def_readonly("id",&routing::river::id,"a valid identifier >0 for the river|routing element")
            .def_readwrite("downstream",&routing::river::downstream,"routing information for downstream, target-id, and hydrological distance")
            .def_readwrite("parameter",&routing::river::parameter,"uhg_parameter, describing the downstream propagation ")
            .def("uhg",&routing::river::uhg,args("dt"),
                "create the hydro-graph for this river, taking specified delta-t, dt,\n"
                "static hydrological distance as well as the shape parameters\n"
                "alpha and beta used to form the gamma-function.\n"
                "The length of the uhg (delay) is determined by the downstream-distance,\n"
                "and the velocity parameter. \n"
                "The shape of the uhg is determined by alpha&beta parameters.\n")
            ;


    }
    void routing_river_network() {
        routing::river& (routing::river_network::*griver)(int)= &routing::river_network::river_by_id;
        class_<routing::river_network>("RiverNetwork",
            "A RiverNetwork takes care of the routing\n"
            "see also description of River\n"
            "The RiverNetwork provides all needed functions to build\n"
            "routing into the region model\n"
            "It ensures safe manipulation of rivers:\n"
            " * no cycles, \n"
            " * no duplicate object-id's etc.\n"
            )
            .def(init<const routing::river_network&>(args("clone"),"make a clone of river-network"))
            .def("add",&routing::river_network::add,args("river"),
                 "add a river to the network, verifies river id, no cycles etc.\n"
                 "\nraises exception on error\ntip: build your river-network from downstream to upstream order\n"
            ,return_internal_reference<>())
            .def("remove_by_id",&routing::river_network::remove_by_id,args("id"),"disconnect and remove river for network")
            //.def("river_by_id",&routing::river_network::river_by_id2,args("id"),"get river by id")
            .def("river_by_id",griver,args("id"),"get river by id",return_internal_reference<>())
            .def("upstreams_by_id",&routing::river_network::upstreams_by_id,args("id"),"returns a list(vector) of id of upstream rivers from the specified one\n")
            .def("downstream_by_id",&routing::river_network::downstream_by_id,args("id"),"return id of downstream river, 0 if none")
            .def("set_downstream_by_id",&routing::river_network::set_downstream_by_id,args("id","downstream_id"),"set downstream target for specified river id")
            .def("network_contains_directed_cycle",&routing::river_network::network_contains_directed_cycle,"True if network have cycles detected")
            ;
    }

    void routing() {
        routing_path_info();
        routing_ugh_parameter();
        routing_river();
        routing_river_network();
    }
}
