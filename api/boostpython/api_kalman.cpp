#include "boostpython_pch.h"
#include "api/api.h"
#include "core/region_model.h"
#include "core/kalman.h"

namespace expose {
    using namespace boost::python;
    namespace sa = shyft::api;
    namespace sc = shyft::core;
	namespace sta = shyft::time_axis;
	//namespace btk = shyft::core::bayesian_kriging;
	//namespace idw = shyft::core::inverse_distance;

	typedef std::vector<sc::geo_point> geo_point_vector;
	typedef std::vector<sa::TemperatureSource> geo_temperature_vector;
	typedef std::shared_ptr<geo_temperature_vector> geo_temperature_vector_;
	typedef std::vector<sa::PrecipitationSource> geo_precipitation_vector;
	typedef std::shared_ptr<geo_precipitation_vector> geo_precipitation_vector_;

	static void kalman_parameter() {
	    typedef shyft::core::kalman::parameter KalmanParameter;
	    class_<KalmanParameter>(
            "KalmanParameter",
            "Defines the parameters that is used to tune the kalman-filter algorithm for temperature type of signals")
            .def(init<optional<int,double,double,double,double>>(args("n_daily_observations","hourly_correlation","covariance_init","std_error_bias_measurements","ratio_std_w_over_v"), "Constructs KalmanParameter with default or supplied values"))
            .def(init<const KalmanParameter&>(arg("const_ref"),"clone the supplied KalmanParameter"))
            .def_readwrite("n_daily_observations",&KalmanParameter::n_daily_observations," default = 8 each 24hour, every 3 hour")
            .def_readwrite("hourly_correlation",&KalmanParameter::hourly_correlation,"default=0.93, correlation from one-hour to the next")
            .def_readwrite("covariance_init",&KalmanParameter::covariance_init," default=0.5,  for the error covariance P matrix start-values")
            .def_readwrite("std_error_bias_measurements",&KalmanParameter::std_error_bias_measurements,"default=2.0, st.dev for the bias measurements")
            .def_readwrite("ratio_std_w_over_v",&KalmanParameter::ratio_std_w_over_v,"default=0.06, st.dev W /st.dev V ratio")
        ;
	}
	static void kalman_state() {
	    typedef shyft::core::kalman::state KalmanState;
	    class_<KalmanState>("KalmanState","keeps the state of the specialized kalman-filter" )
	    .def(init<int,double,double,double>(args("n_daily_observations","covariance_init","hourly_correlation","process_noise_init"),"create a state based on supplied parameters"))
	    .def(init<const KalmanState&>(arg("clone"),"clone the supplied state"))
	    .def(init<>("construct a default state"))
	    .def("size",&KalmanState::size,"returns the size of the state, corresponding to n_daily_observations")
	    // todo: expose arma::vec and arma::mat
	    ;
	}
	static void kalman_filter() {
	}
	static void kalman_bias_predictor() {
	}
    void kalman() {
        kalman_parameter();
        kalman_state();
        kalman_filter();
        kalman_bias_predictor();
    }
}
