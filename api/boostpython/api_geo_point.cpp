#include "boostpython_pch.h"

#include "core/geo_point.h"
namespace expose {

    void api_geo_point() {
        using namespace shyft::core;
        using namespace boost::python;
        namespace py=boost::python;
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
        .def(init<double,double,double>((py::arg("x"),py::arg("y"),py::arg("z")),
                                        doc_intro("construct a geo_point with x,y,z ")
                                        doc_parameters()
                                        doc_parameter("x","float","meter units")
                                        doc_parameter("y","float","meter units")
                                        doc_parameter("z","float","meter units")
                                        )
        )
        .def_readwrite("x",&geo_point::x)
        .def_readwrite("y",&geo_point::y)
        .def_readwrite("z",&geo_point::z)
        .def("distance2",&geo_point::distance2,(py::arg("a"),py::arg("b")),
             "returns the euclidian distance^2 ").staticmethod("distance2")
        .def("distance_measure",&geo_point::distance_measure,(py::arg("a"),py::arg("b"),py::arg("p")),"return sum(a-b)^p").staticmethod("distance_measure")
        .def("zscaled_distance",&geo_point::zscaled_distance,(py::arg("a"),py::arg("b"),py::arg("zscale")),"sqrt( (a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y) + (a.z - b.z)*(a.z - b.z)*zscale*zscale)").staticmethod("zscaled_distance")
        .def("xy_distance",geo_point::xy_distance,(py::arg("a"),py::arg("b")),"returns sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y))").staticmethod("xy_distance")
        .def("difference",geo_point::difference,(py::arg("a"),py::arg("b")),"returns GeoPoint(a.x - b.x, a.y - b.y, a.z - b.z)").staticmethod("difference")
        .def(self==self)
        //.def(self!=self)
        ;
    }
}
