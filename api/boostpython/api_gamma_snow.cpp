#include "boostpython_pch.h"

#include "core/gamma_snow.h"

namespace expose {

    using namespace shyft::core::gamma_snow;
    using namespace boost::python;
    using namespace std;
    namespace py=boost::python;
    void gamma_snow() {
        class_<parameter>("GammaSnowParameter")
            .def(init<int,optional<double,double,double,double,double,double,double,double,double,double,double,double,double>>(
              args("winter_end_day_of_year","initial_bare_ground_fraction","snow_cv","tx","wind_scale","wind_const","max_water","surface_magnitude","max_albedo","min_albedo","fast_albedo_decay_rate","slow_albedo_decay_rate","snowfall_reset_depth","glacier_albedo"),"specifying most of the parameters"))
             //Note: due to max arity of 15, the init function does not provide all the params, TODO: consider kwargs etc. instead
            .def_readwrite("winter_end_day_of_year", &parameter::winter_end_day_of_year,"Last day of accumulation season,default= 100")
            .def_readwrite("n_winter_days",&parameter::n_winter_days,"number of winter-days, default 221")
            .def_readwrite("initial_bare_ground_fraction", &parameter::initial_bare_ground_fraction,"Bare ground fraction at melt onset,default= 0.04")
            .def_readwrite("snow_cv" ,&parameter:: snow_cv,"Spatial coefficient variation of fresh snowfall, default= 0.4")
            .def_readwrite("tx", &parameter::tx,"default= -0.5 [degC]")
            .def_readwrite("wind_scale", &parameter::wind_scale,"Slope in turbulent wind function default=2.0 [m/s]")
            .def_readwrite("wind_const", &parameter::wind_const,"Intercept in turbulent wind function,default=1.0")
            .def_readwrite("max_water", &parameter::max_water,"Maximum liquid water content,default=0.1")
            .def_readwrite("surface_magnitude", &parameter::surface_magnitude,"Surface layer magnitude,default=30.0")
            .def_readwrite("max_albedo", &parameter::max_albedo,"Maximum albedo value,default=0.9")
            .def_readwrite("min_albedo", &parameter::min_albedo,"Minimum albedo value,default=0.6")
            .def_readwrite("fast_albedo_decay_rate", &parameter::fast_albedo_decay_rate,"Albedo decay rate during melt [days],default=5.0")
            .def_readwrite("slow_albedo_decay_rate", &parameter::slow_albedo_decay_rate,"Albedo decay rate in cold conditions [days],default=5.0")
            .def_readwrite("snowfall_reset_depth", &parameter::snowfall_reset_depth,"Snowfall required to reset albedo [mm],default=5.0")
            .def_readwrite("glacier_albedo", &parameter::glacier_albedo,"Glacier ice fixed albedo, default=0.4")
            .def_readwrite("calculate_iso_pot_energy", &parameter::calculate_iso_pot_energy,"Whether or not to calculate the potential energy flux,default=false")
            .def_readwrite("snow_cv_forest_factor", &parameter::snow_cv_forest_factor,"default=0.0, [ratio]\n\tthe effective snow_cv gets an additional value of geo.forest_fraction()*snow_cv_forest_factor")
            .def_readwrite("snow_cv_altitude_factor", &parameter::snow_cv_altitude_factor,"default=0.0, [1/m]\n\t the effective snow_cv gets an additional value of altitude[m]* snow_cv_altitude_factor")
            .def("effective_snow_cv",&parameter::effective_snow_cv,(py::arg("self"),py::arg("forest_fraction"),py::arg("altitude")),"returns the effective snow cv, taking the forest_fraction and altitude into the equations using corresponding factors")
            .def("is_snow_season",&parameter::is_snow_season,(py::arg("self"),py::args("t")),"returns true if specified t is within the snow season, e.g. sept.. winder_end_day_of_year")
            .def("is_start_melt_season",&parameter::is_start_melt_season,(py::arg("self"),py::arg("t")),"true if specified interval t day of year is wind_end_day_of_year")
            ;
        class_<state>("GammaSnowState", "The state description of the GammaSnow routine")
          .def(init<optional<double,double,double,double,double,double,double,double>>(args("albedo","lwc","surface_heat","alpha","sdc_melt_mean","acc_melt","iso_pot_energy","temp_swe"),"the description here ?"))
          .def_readwrite("albedo",&state::albedo,"albedo (Broadband snow reflectivity fraction),default = 0.4")
          .def_readwrite("lwc",&state::lwc,"lwc (liquid water content) [mm],default = 0.1")
          .def_readwrite("surface_heat",&state::surface_heat,"surface_heat (Snow surface cold content) [J/m2],default = 30000.0")
          .def_readwrite("alpha",&state::alpha,"alpha (Dynamic shape state in the SDC),default = 1.26")
          .def_readwrite("sdc_melt_mean",&state::sdc_melt_mean,"sdc_melt_mean  (Mean snow storage at melt onset) [mm],default = 0.0")
          .def_readwrite("acc_melt",&state::acc_melt,"acc_melt (Accumulated melt depth) [mm],default = 0.0")
          .def_readwrite("iso_pot_energy",&state::iso_pot_energy,"iso_pot_energy (Accumulated energy assuming isothermal snow surface) [J/m2],default = 0.0")
          .def_readwrite("temp_swe",&state::temp_swe,"temp_swe (Depth of temporary new snow layer during spring) [mm],default = 0.0")
          ;
        class_<response>("GammaSnowResponse","The response(output) from gamma-snow for one time-step")
          .def_readwrite("sca",&response::sca,"Snow covered area [0..1]")
          .def_readwrite("storage",&response::storage,"Snow storage in [mm] over the area")
          .def_readwrite("outflow",&response::outflow,"Water out from the snow pack [mm/h]")
          ;
        typedef calculator<parameter,state,response> GammaSnowCalculator;
        class_<GammaSnowCalculator>("GammaSnowCalculator")
            .def("step",&GammaSnowCalculator::step,args("state","response","t","dt","parameter","temperature","radiation","precipitation","wind_speed","rel_hum","forest_fraction","altitude"),
                "Step the snow model forward from time t to t+dt, given state, parameters and input.\n"
                "Updates the state and response upon return.\n"
                " param state state of type S,in/out, ref template parameters\n"
                " param response result of type R, output only, ref. template parameters\n"
                " param temperature degC, considered constant over timestep dt\n"
                " param radiation in W/m2, considered constant over timestep\n"
                " param precipitation in mm/h\n"
                " param wind_speed in m/s\n"
                " param rel_hum 0..1\n"
                " param forest_fraction 0..1, influences calculation of effective snow_cv\n"
                " param altitude 0..x [m], influences calculation of effective_snow_cv\n"
            )
            ;

    }
}
