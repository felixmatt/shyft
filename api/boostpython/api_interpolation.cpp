#include "boostpython_pch.h"

#include "core/inverse_distance.h"
#include "core/bayesian_kriging.h"
#include "core/region_model.h"


using namespace boost::python;

static void expose_btk_interpolation() {
    typedef shyft::core::bayesian_kriging::parameter BTKParameter;

    class_<BTKParameter>("BTKParameter","BTKParameter class with time varying gradient based on day no")
        .def(init<double,double>(args("temperature_gradient","temperature_gradient_sd"),"specifying default temp.grad(not used) and std.dev[C/100m]"))
        .def(init<double,double,double,double,double,double>(args("temperature_gradient","temperature_gradient_sd", "sill", "nugget", "range", "zscale"),"full specification of all parameters"))
        .def("temperature_gradient",&BTKParameter::temperature_gradient,args("p"),"return default temp.gradient based on day of year calculated for midst of utcperiod p")
        .def("temperature_gradient_sd",&BTKParameter::temperature_gradient_sd,"returns Prior standard deviation of temperature gradient in [C/m]" )
        .def("sill",&BTKParameter::sill,"Value of semivariogram at range default=25.0")
        .def("nug",&BTKParameter::nug,"Nugget magnitude,default=0.5")
        .def("range",&BTKParameter::range,"Point where semivariogram flattens out,default=200000.0")
        .def("zscale",&BTKParameter::zscale,"Height scale used during distance computations,default=20.0")
        ;
}

static void expose_idw_interpolation() {
    typedef shyft::core::inverse_distance::parameter IDWParameter;



    class_<IDWParameter>("IDWParameter",
                "IDWParameter is a simple place-holder for IDW parameters\n"
                "The two most common:\n"
                "  max_distance \n"
                "  max_members \n"
                "Is used for interpolation.\n"
                "Additionally it keep distance measure-factor,\n"
                "so that the IDW distance is computed as 1 over pow(euclid distance,distance_measure_factor)\n"
        )
        .def(init<int,optional<double,double>>(args("max_members","max_distance","distance_measure_factor"),"create IDW from supplied parameters"))
        .def_readwrite("max_members",&IDWParameter::max_members,"maximum members|neighbors used to interpolate into a point,default=10")
        .def_readwrite("max_distance",&IDWParameter::max_distance,"[meter] only neighbours within max distance is used for each destination-cell,default= 200000.0")
        .def_readwrite("distance_measure_factor",&IDWParameter::distance_measure_factor,"IDW distance is computed as 1 over pow(euclid distance,distance_measure_factor), default=2.0")
        ;
    typedef shyft::core::inverse_distance::temperature_parameter IDWTemperatureParameter;
    class_<IDWTemperatureParameter,bases<IDWParameter>> ("IDWTemperatureParameter",
            "For temperature inverse distance, also provide default temperature gradient to be used\n"
            "when the gradient can not be computed.\n"
            "note: if gradient_by_equation is set true, and number of points >3, the temperature gradient computer\n"
            "      will try to use the 4 closes points and determine the 3d gradient including the vertical gradient.\n"
            "      (in scenarios with constant gradients(vertical/horizontal), this is accurate) \n"
        )
        .def(init<double,optional<int,double,bool>>(args("default_gradient","max_members","max_distance","gradient_by_equation"),"construct IDW for temperature as specified by arguments"))
        .def_readwrite("default_temp_gradient",&IDWTemperatureParameter::default_temp_gradient,"[degC/m], default=-0.006")
        .def_readwrite("gradient_by_equation",&IDWTemperatureParameter::gradient_by_equation,"if true, gradient is computed using 4 closest neighbors, solving equations to find 3D temperature gradients.")
        ;

    typedef shyft::core::inverse_distance::precipitation_parameter IDWPrecipitationParameter;
    class_<IDWPrecipitationParameter,bases<IDWParameter>>("IDWPrecipitationParameter",
                "For precipitation,the scaling model needs the increase in precipitation for each 100 meters.\n"
                "Ref to IDWParameter for the other parameters\n"
        )
        .def(init<double,optional<int,double>>(args("increase_pct_m", "max_members","max_distance"),"create IDW from supplied parameters"))
        .def_readwrite("scale_factor",&IDWPrecipitationParameter::scale_factor,"mm/m,  default=1.02, mm/m, corresponding to 2 mm increase pr. 100 m height")
    ;
}
static void expose_interpolation_parameter() {
    typedef shyft::core::interpolation_parameter InterpolationParameter;
    namespace idw = shyft::core::inverse_distance;
    namespace btk = shyft::core::bayesian_kriging;
    class_<InterpolationParameter>("InterpolationParameter",
             "The InterpolationParameter keep parameters needed to perform the\n"
             "interpolation steps, IDW,BTK etc\n"
             "It is used as parameter  in the model.run_interpolation() method\n"
        )
        .def(init<const btk::parameter&,const idw::precipitation_parameter&,const idw::parameter&,const idw::parameter&,const idw::parameter&>(args("temperature","precipitation","wind_speed","radiation","rel_hum"),"using BTK for temperature"))
        .def(init<const idw::temperature_parameter&,const idw::precipitation_parameter&,const idw::parameter&,const idw::parameter&,const idw::parameter&>(args("temperature","precipitation","wind_speed","radiation","rel_hum"),"using smart IDW for temperature, typically grid inputs"))
        .def_readwrite("use_idw_for_temperature",&InterpolationParameter::use_idw_for_temperature,"if true, the IDW temperature is used instead of BTK, useful for grid-input scenarios")
        .def_readwrite("temperature",&InterpolationParameter::temperature,"BTK for temperature (in case .use_idw_for_temperature is false)")
        .def_readwrite("temperature_idw",&InterpolationParameter::temperature_idw,"IDW for temperature(in case .use_idw_for_temperature is true)")
        .def_readwrite("precipitation",&InterpolationParameter::precipitation,"IDW parameters for precipitation")
        .def_readwrite("wind_speed", &InterpolationParameter::wind_speed,"IDW parameters for wind_speed")
        .def_readwrite("radiation", &InterpolationParameter::radiation,"IDW parameters for radiation")
        .def_readwrite("rel_hum",&InterpolationParameter::rel_hum,"IDW parameters for relative humidity")
        ;
}
void expose_interpolation() {
    expose_idw_interpolation();
    expose_btk_interpolation();
    expose_interpolation_parameter();
}
