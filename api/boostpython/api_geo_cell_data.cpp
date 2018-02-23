#include "boostpython_pch.h"

#include "core/geo_point.h"
#include "core/geo_cell_data.h"
namespace expose {
    using namespace shyft::core;
    using namespace boost::python;
    namespace py=boost::python;
    using namespace std;

    void api_geo_cell_data() {
        class_<land_type_fractions>("LandTypeFractions",
                              "LandTypeFractions are used to describe 'type of land'\n"
                              "like glacier, lake, reservoir and forest.\n"
                              "It is designed as a part of GeoCellData \n"
        )
        .def(init<double,double,double,double,double>(args("glacier","lake","reservoir","forest","unspecified"),"construct LandTypeFraction specifying the area of each type"))
        .def("glacier",&land_type_fractions::glacier,"returns the glacier part")
        .def("lake",&land_type_fractions::lake,"returns the lake part")
        .def("reservoir",&land_type_fractions::reservoir,"returns the reservoir part")
        .def("forest",&land_type_fractions::forest,"returns the forest part")
        .def("unspecified",&land_type_fractions::unspecified,"returns the unspecified part")
        .def("set_fractions",&land_type_fractions::set_fractions,args("glacier","lake","reservoir","forest"),"set the fractions explicit, each a value in range 0..1, sum should be 1.0")
        ;

        class_<geo_cell_data>("GeoCellData",
                              "Represents common constant geo_cell properties across several possible models and cell assemblies.\n"
                              "The idea is that most of our algorithms uses one or more of these properties,\n"
                              "so we provide a common aspect that keeps this together.\n"
                              "Currently it keep the\n"
                              "  - mid-point geo_point, (x,y,z) (default 0)\n"
                              "  the area in m^2, (default 1000 x 1000 m2)\n"
                              "  land_type_fractions (unspecified=1)\n"
                              "  catchment_id   def (-1)\n"
                              "  radiation_slope_factor def 0.9\n"
                              "  routing_info def(0,0.0), i.e. not routed and hydrological distance=0.0m\n"
        )
        .def(init<geo_point,double,int,optional<double,const land_type_fractions&,routing_info>>(
         (py::arg("self"),py::arg("mid_point"),py::arg("area"),py::arg("catchment_id"),py::arg("radiation_slope_factor"),py::arg("land_type_fractions"),py::arg("routing_info")),
         "constructs a GeoCellData with all parameters specified"))
        .def("mid_point",&geo_cell_data::mid_point,"returns the mid_point",return_value_policy<copy_const_reference>())
        .def("catchment_id",&geo_cell_data::catchment_id,"returns the catchment_id")
        .def("set_catchment_id",&geo_cell_data::catchment_id,args("catchment_id"),"set the catchment_id")
        .def("radiation_slope_factor",&geo_cell_data::radiation_slope_factor,"radiation slope factor")
        .def("land_type_fractions_info",&geo_cell_data::land_type_fractions_info,"land_type_fractions",return_value_policy<copy_const_reference>())
        .def("set_land_type_fractions",&geo_cell_data::set_land_type_fractions,args("ltf"),"set new LandTypeFractions")
        .def_readwrite("routing_info",&geo_cell_data::routing,"the routing information for the cell keep destination id and hydrological distance to destination")
        .def("area",&geo_cell_data::area,"returns the area in m^2")
        ;
    }
}
