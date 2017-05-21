#include "boostpython_pch.h"
#include "api/api.h"
#include "core/region_model.h"
#include "core/kalman.h"

namespace expose {
    using namespace boost::python;
    namespace sa = shyft::api;
    namespace sc = shyft::core;
	namespace sta = shyft::time_axis;

	typedef std::vector<sc::geo_point> geo_point_vector;
	typedef std::vector<sa::TemperatureSource> geo_temperature_vector;
	typedef std::shared_ptr<geo_temperature_vector> geo_temperature_vector_;
	typedef std::vector<sa::PrecipitationSource> geo_precipitation_vector;
	typedef std::shared_ptr<geo_precipitation_vector> geo_precipitation_vector_;
    typedef sa::ats_vector apoint_ts_vector; // this type is already exposed in api, so we can use it directly


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
	static std::vector<double> kalman_x(const shyft::core::kalman::state &s) {return arma::conv_to<std::vector<double>>::from(s.x);}
	static std::vector<double> kalman_k(const shyft::core::kalman::state &s) { return arma::conv_to<std::vector<double>>::from(s.k); }
	/** flattens supplied matrix into a vector row by col */
	static std::vector<double> arma_flatten(const arma::mat&m) {
		std::vector<double> r;
		for (arma::uword i = 0;i < m.n_rows;++i) {
			auto row = arma::conv_to<std::vector<double>>::from(m.row(i));
			for (auto v : row) r.push_back(v);
		}
		return r;// return flatten vector rows repeated n_cols
	}
	static std::vector<double> kalman_P(const shyft::core::kalman::state &s) { return arma_flatten(s.P);}
	static std::vector<double> kalman_W(const shyft::core::kalman::state &s) { return arma_flatten(s.W);}

	static void kalman_state() {
	    typedef shyft::core::kalman::state KalmanState;
		class_<KalmanState>("KalmanState",
			"keeps the state of the specialized kalman-filter\n\t"
			"x : vector[n=n_daily_observations] best estimate\n\t"
			"k : vector[n], gain factors\n\t"
			"P : matrix[nxn], covariance\n\t"
			"W : noise[nxn]\n"
			)
			.def(init<int, double, double, double>(args("n_daily_observations", "covariance_init", "hourly_correlation", "process_noise_init"), "create a state based on supplied parameters"))
			.def(init<const KalmanState&>(arg("clone"), "clone the supplied state"))
			.def(init<>("construct a default state"))
			.def("size", &KalmanState::size, "returns the size of the state, corresponding to n_daily_observations")
			.def("get_x", kalman_x, args("state"),"returns a copy of current bias estimate x").staticmethod("get_x")
			.def("get_k", kalman_k, args("state"), "returns a copy of current kalman gain k").staticmethod("get_k")
			.def("get_P", kalman_P, args("state"), "returns a copy of current kalman covariance matrix P").staticmethod("get_P")
			.def("get_W", kalman_W, args("state"), "returns a copy of current kalman noise matrix W").staticmethod("get_W")
	    ;
	}
	static void kalman_filter() {
		typedef shyft::core::kalman::filter KalmanFilter;
		class_<KalmanFilter>(
			"KalmanFilter",
			"Specialized kalman filter for temperature (e.g.:solar-driven bias patterns)\n\n"
			"The observation point(t, v) is folded on to corresponding period of the\n"
			"day(number of periods in a day is parameterized, typically 8).\n"
			"A simplified kalman filter algorithm using the forecast bias as\n"
			"the state - variable.\n"
			"Observed bias(fc - obs) is feed into the filter and establishes the\n"
			"kalman best predicted estimates(x) for the bias.\n"
			"This bias can then be used as a correction to forecast in the future\n"
			" to compensate for systematic forecast errors.\n"
			" Credits: Thanks to met.no for providing the original source for this algorithm.\n"
			"\n"
			" see also https ://en.wikipedia.org/wiki/Kalman_filter\n"
			)
			.def(init<>("Construct a filter with default KalmanParameter"))
			.def(init<shyft::core::kalman::parameter>(args("p"), "Construct a filter with the supplied parameter"))
			.def("create_initial_state",&KalmanFilter::create_initial_state,"returns initial state, suitable for starting, using the filter parameters")
			.def("update",&KalmanFilter::update,args("observed_bias","t","state"),
				"update the with the observed_bias for a specific period starting at utctime t\n"
				"\n"
				"Parameters\n"
				"----------\n"
				"observed_bias : double\n"
				"\tnan if no observation is available otherwise obs - fc\n"
				"t : utctime\n"
				"\tutctime of the start of the observation period, this filter utilizes daily solar patterns, so time\n"
				"\tin day - cycle is the only important aspect.\n"
				"state : KalmanState\n"
				"\t contains the kalman state, that is updated by the function upon return\n"
				)
			.def_readwrite("parameter",&KalmanFilter::p,"The KalmanParameter used by this filter")
			;

	}
    ///thin wrappers to fwd the call from py to c++
	static void update_with_forecast_geo_ts_and_obs(
		shyft::core::kalman::bias_predictor& bp,
		geo_temperature_vector_ fc,
		const shyft::api::apoint_ts& obs,
		const shyft::time_axis::generic_dt &ta) {
		std::vector<shyft::api::apoint_ts> fc_ts_set;
		for (auto& geo_ts : *fc)
			fc_ts_set.push_back(geo_ts.ts);
		bp.update_with_forecast(fc_ts_set, obs, ta);
	}
	static void update_with_forecast_ts_and_obs(
			shyft::core::kalman::bias_predictor& bp,
		const apoint_ts_vector& fc_ts_set,
		const shyft::api::apoint_ts& obs,
		const shyft::time_axis::generic_dt &ta) {
        bp.update_with_forecast(fc_ts_set, obs, ta);
	}
    static shyft::api::apoint_ts compute_running_bias(
        shyft::core::kalman::bias_predictor& bp,
        const shyft::api::apoint_ts& fc_ts,
        const shyft::api::apoint_ts& obs,
        const shyft::time_axis::generic_dt &ta) {
        if (ta.gt != ta.FIXED)
            throw std::runtime_error("The supplied time-axis must be of type FIXED for the compute_running_bias function");
        return bp.compute_running_bias<shyft::api::apoint_ts>(fc_ts, obs, ta.f);
    }

	static void kalman_bias_predictor() {
		typedef shyft::core::kalman::bias_predictor KalmanBiasPredictor;

		class_<KalmanBiasPredictor>(
			"KalmanBiasPredictor",
			"A bias predictor using a daily pattern KalmanFilter for temperature (etc.)\n"
			"(tbd)"
			)
			.def(init<>("Constructs a bias predictor with default filter, parameters and state"))
			.def(init<const shyft::core::kalman::filter&>(args("filter"),"create a bias predictor with specified filter"))
			.def(init<const shyft::core::kalman::filter&,const shyft::core::kalman::state&>(args("filter","state"), "create a bias predictor with specified filter and initial state"))
			.def("update_with_geo_forecast", update_with_forecast_geo_ts_and_obs,args("bias_predictor","temperature_sources","observation_ts","time_axis"),
               	"update the bias-predictor with forecasts and observation\n"
               	"After the update, the state is updated with new kalman estimates for the bias, .state.x\n"
				"Parameters\n"
				"----------\n"
				"bias_predictor : KalmanBiasPredictor\n"
				"\tThe bias predictor object it self\n"
				"temperature_sources : TemperatureSourceVector\n"
				"\ta set of forecasts, in the order oldest to the newest.\n"
				"\tnote that the geo part of source is not used in this context, only the ts\n"
				"\twith periods covering parts of the observation_ts and time_axis supplied\n"
				"observation ts: Timeseries\n"
				"\tthe observation time-series\n"
				"time_axis : Timeaxis\n"
				"\tcovering the period/timesteps to be updated\n"
				"\t e.g. yesterday, 3h resolution steps, according to the points in the filter\n"
			).staticmethod("update_with_geo_forecast")
			.def("update_with_forecast_vector",update_with_forecast_ts_and_obs,args("bias_predictor","temperature_sources","observation_ts","time_axis"),
               	"update the bias-predictor with forecasts and observation\n"
               	"After the update, the state is updated with new kalman estimates for the bias, .state.x\n"
				"Parameters\n"
				"----------\n"
				"bias_predictor : KalmanBiasPredictor\n"
				"\tThe bias predictor object it self\n"
				"temperature_sources : TsVector\n"
				"\ta set of forecasts, in the order oldest to the newest.\n"
				"\twith periods covering parts of the observation_ts and time_axis supplied\n"
				"observation ts: Timeseries\n"
				"\tthe observation time-series\n"
				"time_axis : Timeaxis\n"
				"\tcovering the period/timesteps to be updated\n"
				"\t e.g. yesterday, 3h resolution steps, according to the points in the filter\n"
            ).staticmethod("update_with_forecast_vector")
            .def("compute_running_bias_ts", compute_running_bias, args("bias_predictor", "forecast_ts", "observation_ts", "time_axis"),
                "compute the running bias timeseries,\n"
                "using one 'merged' - forecasts and one observation time - series.\n"
                "\n"
                "Before each day - period, the bias - values are copied out to form\n"
                "a continuous bias prediction time-series.\n"
                "Parameters\n"
                "----------\n"
                "bias_predictor : KalmanBiasPredictor\n"
                "\tThe bias predictor object it self\n"
                "forecast_ts : Timeseries\n"
                "\ta merged forecast ts\n"
                "\twith period covering the observation_ts and time_axis supplied\n"
                "observation ts: Timeseries\n"
                "\tthe observation time-series\n"
                "time_axis : Timeaxis\n"
                "\tcovering the period/timesteps to be updated\n"
                "\t e.g. yesterday, 3h resolution steps, according to the points in the filter\n"
                "\nReturns\n-------\n"
                "bias_ts:Timeseries(time_axis,bias_vector,POINT_AVERAGE)\n"
                "\t computed running bias-ts\n"
            ).staticmethod("compute_running_bias_ts")

			.def_readonly("filter",&KalmanBiasPredictor::f,"the kalman filter with parameters")
			.def_readwrite("state",&KalmanBiasPredictor::s,"current state of the predictor")
			;
	}
    void kalman() {
        kalman_parameter();
        kalman_state();
        kalman_filter();
        kalman_bias_predictor();
    }
}
