#include "boostpython_pch.h"

#include "core/skaugen.h"

namespace expose {

    void skaugen_snow() {
        using namespace shyft::core::skaugen;
        using namespace boost::python;
        using namespace std;
        class_<parameter>("SkaugenParameter")
        .def(init<double,double,double,double,double,double,double,double>(args("alpha_0","d_range","unit_size","max_water_fraction","tx","cx","ts","cfr"),"create parameter object with specifed values"))
        .def(init<const parameter&>(args("p"),"create a clone of p"))
        .def_readwrite("alpha_0",&parameter::alpha_0,"default = 40.77")
        .def_readwrite("d_range",&parameter::d_range,"default = 113.0")
        .def_readwrite("unit_size",&parameter::unit_size,"default = 0.1")
        .def_readwrite("max_water_fraction",&parameter::max_water_fraction,"default = 0.1")
        .def_readwrite("tx",&parameter::tx,"default = 0.16")
        .def_readwrite("cx",&parameter::cx,"default = 2.5")
        .def_readwrite("ts",&parameter::ts,"default = 0.14")
        .def_readwrite("cfr",&parameter::cfr,"default = 0.01")
         ;

        class_<state>("SkaugenState")
         .def(init<double,optional<double,double,double,double,double,double>>(args("","alpha","sca","swe","free_water","residual","num_units"),"create a state with specified values"))
         .def(init<const state&>(args("s"),"create a clone of s"))
         .def_readwrite("nu",&state::nu,"")
         .def_readwrite("alpha",&state::alpha,"")
         .def_readwrite("sca",&state::sca,"")
         .def_readwrite("swe",&state::swe,"")
         .def_readwrite("free_water",&state::free_water,"")
         .def_readwrite("residual",&state::residual,"")
         .def_readwrite("num_units",&state::num_units,"")
         ;

        class_<response>("SkaugenResponse")
         .def_readwrite("sca",&response::sca,"from snow-routine in [m3/s]")
         .def_readwrite("swe",&response::swe,"from snow-routine in [m3/s]")
         .def_readwrite("outflow",&response::outflow,"from snow-routine in [m3/s]")
         .def_readwrite("total_stored_water",&response::total_stored_water,"def. as sca*(swe+lwc)")
         ;

        typedef  calculator<parameter,state,response> SkaugenCalculator;
        class_<SkaugenCalculator>("SkaugenCalculator",
                "Skaugen snow model method\n"
                "\n"
                "This algorithm uses theory from Skaugen \n"
                "\n"
            )
            .def("step",&SkaugenCalculator::step,args("delta_t","parameter","temperature","precipitation","radiation","wind_speed","state","response"),"steps the model forward delta_t seconds, using specified input, updating state and response")
            ;

        /*typedef statistics SkaugenStatistics;
        double (statistics::*sca_rel_red_1)(unsigned long , unsigned long , double , double , double )=&statistics::sca_rel_red;
        double (statistics::*sca_rel_red_2)(unsigned long , unsigned long , double , double ) const=&statistics::sca_rel_red;

        class_<SkaugenStatistics>("SkaugenStatistics")
            .def(init<double,double,double>(args("alpha_0","d_range","unit_size"),"create a new object with given constants"))
            .def("c",&SkaugenStatistics::c,args("n","d_range"),"return exp(-(double(n)/d_range)")
            .def("sca_rel_red",sca_rel_red_2,args("u","n","nu_a","alpha"),"tbd")
            ;
            */

    }
}
