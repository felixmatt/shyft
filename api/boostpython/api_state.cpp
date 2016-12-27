#include "boostpython_pch.h"

#include "api_state.h"
namespace expose {
    using namespace shyft::api;
    using namespace boost::python;
    using namespace std;

    void api_cell_state_id() {
        class_<cell_state_id>("CellStateId",
            "Unique cell pseudo identifier of  a state\n"
            "\n"
            "A lot of Shyft models can appear in different geometries and\n"
            "resolutions.In a few rare cases we would like to store the *state* of\n"
            "a model.This would be typical for example for hydrological new - year 1 - sept in Norway.\n"
            "To ensure that each cell - state have a unique identifier so that we never risk\n"
            "mixing state from different cells or different geometries, we create a pseudo unique\n"
            "id that helps identifying unique cell - states given this usage and context.\n"
            "\n"
            "The primary usage is to identify which cell a specific identified state belongs to.\n"

            )
            .def(init<int, int, int, int>(args("cid", "x", "y", "area"), "construct CellStateId with specified parameters"))
            .def_readwrite("cid", &cell_state_id::cid, "catchment identifier")
            .def_readwrite("x", &cell_state_id::x, "x position in [m]")
            .def_readwrite("y", &cell_state_id::y, "y position in [m]")
            .def_readwrite("area", &cell_state_id::area, "area in [m^2]")
            .def(self==self)
            ;

    }
}