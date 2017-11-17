#include "boostpython_pch.h"

#include "core/geo_point.h"
namespace expose {

    void api_geo_point() {
        using namespace shyft::core;
        using namespace boost::python;

        class_<geo_point>("GeoPoint",
             "GeoPoint commonly used in the shyft::core for\n"
             "  representing a 3D point in the terrain model.\n"
             "  The primary usage is in the interpolation routines\n"
             "\n"
             "  Absolutely a primitive point model aiming for efficiency and simplicity.\n"
             "\n"
             "  Units of x,y,z are metric, z positive upwards to sky, represents elevation\n"
             "  x is east-west axis\n"
             "  y is south-north axis\n"
        )
        .def(init<double,double,double>(args("x,y,z"),"construct a geo_point with x,y,z "))
        .def_readwrite("x",&geo_point::x)
        .def_readwrite("y",&geo_point::y)
        .def_readwrite("z",&geo_point::z)
        .def("distance2",&geo_point::distance2,args("a,b"),"returns the euclidian distance^2 ").staticmethod("distance2")
        .def("distance_measure",&geo_point::distance_measure,args("a,b,p"),"return sum(a-b)^p").staticmethod("distance_measure")
        .def("zscaled_distance",&geo_point::zscaled_distance,args("a,b,zscale"),"sqrt( (a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y) + (a.z - b.z)*(a.z - b.z)*zscale*zscale)").staticmethod("zscaled_distance")
        .def("xy_distance",geo_point::xy_distance,args("a,b"),"returns sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y))").staticmethod("xy_distance")
        .def("difference",geo_point::difference,args("a,b"),"returns GeoPoint(a.x - b.x, a.y - b.y, a.z - b.z)").staticmethod("difference")
        .def(self==self)
        //.def(self!=self)
        ;
    }
}
