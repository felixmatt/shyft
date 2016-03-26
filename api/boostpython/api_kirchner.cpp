#include "boostpython_pch.h"

#include "core/kirchner.h"

using namespace shyft::core::kirchner;
using namespace boost::python;
using namespace std;

void def_kirchner() {

    class_<parameter>("KirchnerParameter")
        .def(init<double,optional<double,double>>(args("c1","c2","c3"),"creates parameter object according to parameters"))
        .def_readwrite("c1",&parameter::c1,"default =2.439")
        .def_readwrite("c2",&parameter::c2,"default= 0.966")
        .def_readwrite("c3",&parameter::c3,"default = -0.10")
        ;
    class_<state>("KirchnerState")
        .def(init<double>(args("q"),"create a state specifying initial content q"))
        .def_readwrite("q",&state::q,"state water 'content' in [mm/h], it defaults to 0.0001 mm, zero is not a reasonable valid value")
        ;
    class_<response>("KirchnerResponse")
        .def_readwrite("q_avg",&response::q_avg,"average discharge over time-step in [mm/h]")
        ;
    typedef calculator<trapezoidal_average,parameter> KirchnerCalculator;
    class_<KirchnerCalculator>("KirchnerCalculator",
            "Kirchner model for computing the discharge based on precipitation and evapotranspiration data.\n"
            "\n"
             "This algorithm is based on the log transform of the ode formulation of the time change in discharge as a function\n"
             "of measured precipitation, evapo-transpiration and discharge, i.e. equation 19 in the publication\n"
             "Catchments as simple dynamic systems: Catchment characterization, rainfall-runoff modeling, and doing\n"
             "'hydrology backward' by James W. Kirchner, published in Water Resources Research, vol. 45, W02429,\n"
             "doi: 10.1029/2008WR006912, 2009.\n"
             "\n",no_init
        )
        .def(init<const parameter&>(args("param"),"create a calculator using supplied parameter"))
        .def(init<double,double,const parameter&>(args("abs_err","rel_err","param"),"create a calculator using supplied parameter, also specifying the ODE error parameters"))
        .def("step",&KirchnerCalculator::step,args("t0","t1","q","q_avg","precipitation","evapotranspiration"),
             "step Kirchner model forward from time t0 to time t1\n"
             "  note: If the supplied q (state) is less than min_q(0.00001, it represents mm water..),\n"
             "        it is forced to min_q to ensure numerical stability\n"
             )
        ;

 }
