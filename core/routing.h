#pragma once
#pragma once
namespace shyft {
    namespace core {
        namespace routing {
            /*
            Routing concepts:

            routing point/object:
            act as *destination* for other sources, like cells, or other upstream routing objects
            have an optional state (if the implementation need it).
            have one optional output to downstream routing point|object
            and 'routing-shape' that determine the response at the output.

            cells have routing info, describing the
            routing destination identifier (a number, used to bind to a certain node the routing network)
            routing parameter(s) (how discharge are shaped before reaching the routing destination)

            implementation:

            -# routing-network [ { routing-node { id,opt output id + shape } ]
            -# region_model can be created with a routing-network
            -# cell.geo.routing { id, shape-parameters }


            */
        }
    }
}