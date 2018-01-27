#include "boostpython_pch.h"

#include "core/hbv_physical_snow.h"

namespace expose {

    void hbv_physical_snow() {
        using namespace shyft::core::hbv_physical_snow;
        using namespace boost::python;
        using namespace std;
        namespace py=boost::python;
        class_<parameter>("HbvPhysicalSnowParameter")
        .def(init<optional<double,double,double,double,double,double,double,double,double,double,double,bool>>(args("tx","lw","cfr","wind_scale","wind_const","surface_magnitude","max_albedo","min_albedo","fast_albedo_decay_rate","slow_albedo_decay_rate","snowfall_reset_depth","calculate_iso_pot_energy"),"create parameter object with specifed values"))
        .def(init<const vector<double>&,const vector<double>&,optional<double,double,double,double,double,double,double,double,double,double,double,bool>>(
            args("snow_redist_factors","quantiles","tx","lw","cfr","wind_scale","wind_const","surface_magnitude","max_albedo","min_albedo","fast_albedo_decay_rate","slow_albedo_decay_rate","snowfall_reset_depth","calculate_iso_pot_energy"),"create a parameter with snow re-distribution factors, quartiles and optionally the other parameters"))
        .def("set_snow_redistribution_factors",&parameter::set_snow_redistribution_factors,args("snow_redist_factors"))
        .def("set_snow_quantiles",&parameter::set_snow_quantiles,args("quantiles"))
        .def_readwrite("tx",&parameter::tx,"threshold temperature determining if precipitation is rain or snow")
        .def_readwrite("lw",&parameter::lw,"max liquid water content of the snow")
        .def_readwrite("cfr",&parameter::cfr,"")
        .def_readwrite("wind_scale",&parameter::wind_scale,"slope in turbulent wind function [m/s]")
        .def_readwrite("wind_const",&parameter::wind_const,"intercept in turbulent wind function")
        .def_readwrite("surface_magnitude",&parameter::surface_magnitude,"surface layer magnitude")
        .def_readwrite("max_albedo",&parameter::max_albedo,"maximum albedo value")
        .def_readwrite("min_albedo",&parameter::min_albedo,"minimum albedo value")
        .def_readwrite("fast_albedo_decay_rate",&parameter::fast_albedo_decay_rate,"albedo decay rate during melt [days]")
        .def_readwrite("slow_albedo_decay_rate",&parameter::slow_albedo_decay_rate,"albedo decay rate in cold conditions [days]")
        .def_readwrite("snowfall_reset_depth",&parameter::snowfall_reset_depth,"snowfall required to reset albedo [mm]")
        .def_readwrite("calculate_iso_pot_energy",&parameter::calculate_iso_pot_energy,"whether or not to calculate the potential energy flux")
        .def_readwrite("s",&parameter::s,"snow redistribution factors,default =1.0..")
        .def_readwrite("intervals",&parameter::intervals,"snow quantiles list default 0, 0.25 0.5 1.0")
         ;

        class_<state>("HbvPhysicalSnowState")
         .def(init<const vector<double>&, const vector<double>&,optional<double, double, double>>(args("albedo", "iso_pot_energy", "surface_heat", "swe","sca"),"create a state with specified values"))
         .def_readwrite("albedo",&state::albedo,"albedo (Broadband snow reflectivity fraction)")
         .def_readwrite("iso_pot_energy",&state::iso_pot_energy,"iso_pot_energy (Accumulated energy assuming isothermal snow surface) [J/m2]")
         .def_readwrite("surface_heat",&state::surface_heat,"surface_heat (Snow surface cold content) [J/m2]")
         .def_readwrite("swe",&state::swe,"snow water equivalent[mm]")
         .def_readwrite("sca",&state::sca,"snow covered area [0..1]")
         .def("distribute", &state::distribute,(py::arg("self"), py::arg("p"),py::arg("force")=true),
             doc_intro("Distribute state according to parameter settings.")
             doc_parameters()
             doc_parameter("p", "HbvPhysicalSnowParameter", "descr")
             doc_parameter("force","bool","default true, if false then only distribute if state vectors are of different size than parameters passed")
             doc_returns("", "None", "")
         )
         ;

        class_<response>("HbvPhysicalSnowResponse")
         .def_readwrite("outflow",&response::outflow,"from snow-routine in [mm]")
         .def_readwrite("hps_state",&response::hps_state,"current state instance")
         .def_readwrite("sca",&response::sca,"snow-covered area")
         .def_readwrite("storage",&response::storage,"snow storage [mm]")
         ;

        typedef  calculator<parameter,state,response> HbvPhysicalSnowCalculator;
        class_<HbvPhysicalSnowCalculator>("HbvPhysicalSnowCalculator",
                "Generalized quantile based HBV Physical Snow model method\n"
                "\n"
                "This algorithm uses arbitrary quartiles to model snow. No checks are performed to assert valid input.\n"
                "The starting points of the quantiles have to partition the unity, \n"
                "include the end points 0 and 1 and must be given in ascending order.\n"
                "\n",no_init
            )
            .def(init<const parameter&>(args("parameter"),"creates a calculator with given parameter and initial state, notice that state is updated in this call(hmm)"))
            .def("step",&HbvPhysicalSnowCalculator::step,args("state","response","t","dt","temperature","rad", "prec_mm_h","wind_speed","rel_hum"),"steps the model forward from t to t+dt, updating state and response")

            ;


    }
}
