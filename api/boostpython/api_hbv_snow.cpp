#include "boostpython_pch.h"

#include "core/hbv_snow.h"

namespace expose {

    void hbv_snow() {
        using namespace shyft::core::hbv_snow;
        using namespace boost::python;
        namespace py = boost::python;
        using namespace std;
        class_<parameter>("HbvSnowParameter")
        .def(init<double,optional<double,double,double,double>>(args("tx","cx","ts","lw","cfr"),"create parameter object with specifed values"))
        .def(init<const vector<double>&,const vector<double>&,optional<double,double,double,double,double>>(
            args("snow_redist_factors","quantiles","tx","cx","ts","lw","cfr"),"create a parameter with snow re-distribution factors, quartiles and optionally the other parameters"))
        .def("set_snow_redistribution_factors",&parameter::set_snow_redistribution_factors,args("snow_redist_factors"))
        .def("set_snow_quantiles",&parameter::set_snow_quantiles,args("quantiles"))
        .def_readwrite("tx",&parameter::tx,"threshold temperature determining if precipitation is rain or snow")
        .def_readwrite("cx",&parameter::cx,"temperature index, i.e., melt = cx(t - ts) in mm per degree C")
        .def_readwrite("ts",&parameter::ts,"threshold temperature for melt onset")
        .def_readwrite("lw",&parameter::lw,"max liquid water content of the snow")
        .def_readwrite("cfr",&parameter::cfr,"")
        .def_readwrite("s",&parameter::s,"snow redistribution factors,default =1.0..")
        .def_readwrite("intervals",&parameter::intervals,"snow quantiles list default 0, 0.25 0.5 1.0")
         ;

        class_<state>("HbvSnowState")
         .def(init<double,optional<double>>(args("swe","sca"),"create a state with specified values"))
         .def_readwrite("swe",&state::swe,"snow water equivalent[mm]")
         .def_readwrite("sca",&state::sca,"snow covered area [0..1]")
         .def("distribute", &state::distribute, (py::arg("self"), py::arg("p"),py::arg("force")=true),
            doc_intro("Distribute state according to parameter settings.")
            doc_parameters()
            doc_parameter("p", "HbvSnowParameter", "descr")
            doc_parameter("force","bool","default true, if false then only distribute if state vectors are of different size than parameters passed")
            doc_returns("", "None", "")
         )
         .def_readwrite("sw",&state::sw,"snow water[mm]")
         .def_readwrite("sp",&state::sp,"snow dry[mm]")
         ;

        class_<response>("HbvSnowResponse")
         .def_readwrite("outflow",&response::outflow,"from snow-routine in [mm]")
         .def_readwrite("snow_state",&response::snow_state,"swe and snow covered area")
         ;

        typedef  calculator<parameter,state> HbvSnowCalculator;
        class_<HbvSnowCalculator>("HbvSnowCalculator",
                "Generalized quantile based HBV Snow model method\n"
                "\n"
                "This algorithm uses arbitrary quartiles to model snow. No checks are performed to assert valid input.\n"
                "The starting points of the quantiles have to partition the unity, \n"
                "include the end points 0 and 1 and must be given in ascending order.\n"
                "\n",no_init
            )
            .def(init<const parameter&>(args("parameter"),"creates a calculator with given parameter"))
            .def("step",&HbvSnowCalculator::step<response>,(py::arg("self"),py::arg("state"),py::arg("response"), py::arg("t0"), py::arg("t1"), py::arg("precipitation"), py::arg("temperature")),
                 doc_intro("steps the model forward from t0 to t1, updating state and response")
            )

            ;


    }
}
