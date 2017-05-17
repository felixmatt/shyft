#include "test_pch.h"
#include "mocks.h"
#include "core/region_model.h"
#include "core/pt_gs_k.h"
#include "core/pt_gs_k_cell_model.h"
#include "core/model_calibration.h"


using namespace shyft::core;
using namespace shyft::time_series;
using namespace shyfttest::mock;
using namespace std;

namespace pt = shyft::core::priestley_taylor;
namespace gs = shyft::core::gamma_snow;
namespace kr = shyft::core::kirchner;
namespace ae = shyft::core::actual_evapotranspiration;
namespace pc = shyft::core::precipitation_correction;
namespace ta = shyft::time_axis;



typedef point_ts<ta::point_dt> xpts_t;
typedef point_ts<ta::fixed_dt> catchment_t;
typedef MCell<pt_gs_k::response_t, pt_gs_k::state_t, pt_gs_k::parameter_t, xpts_t> PTGSKCell;


typedef dlib::matrix<double, 0, 1> column_vector;


namespace shyfttest {

    class TestModel {
      public:
        TestModel(const std::vector<double> target, const std::vector<double> p_min, const std::vector<double> p_max)
         : target(target), p_min(p_min), p_max(p_max) { /* Do nothing */ }

        /** \brief Simple test function to minimize
         */
        double operator() (const column_vector& p_s) const {
            return mean(squared(dlib::mat(target) - dlib::mat(from_scaled(p_s))));
        }
        double operator() (const vector<double>& p_s) const {
            return mean(squared(dlib::mat(target) - dlib::mat(from_scaled(p_s))));
        }

        std::vector<double> to_scaled(std::vector<double>& p) {
            std::vector<double> p_s;
            p_s.reserve(p.size());
            for (size_t i = 0; i < p.size(); ++i)
                p_s.emplace_back((p[i] - p_min[i])/(p_max[i] - p_min[i]));
            return p_s;
        }

        std::vector<double> from_scaled(const column_vector& p_s) const {
            std::vector<double> p;
            p.reserve(p_s.nr());
            for (int i = 0; i < p_s.nr(); ++i)
                p.emplace_back((p_max[i] - p_min[i])*p_s(i) + p_min[i]);
            return p;
        }
        std::vector<double> from_scaled(const vector<double>& p_s) const {
            std::vector<double> p;
            p.reserve(p_s.size());
            for (size_t i = 0; i < p_s.size(); ++i)
                p.emplace_back((p_max[i] - p_min[i])*p_s[i] + p_min[i]);
            return p;
        }

      private:
        std::vector<double> target;
        std::vector<double> p_min;
        std::vector<double> p_max;
    };

    using namespace shyfttest::mock;




    template<class M>
    struct SimpleParameterAccessor {
        typename M::parameter_t& parameter;
        SimpleParameterAccessor(M& model) : parameter(model.parameter) { /* Do nothing */ }

        size_t size() const { return 4;}

        void set(const std::vector<double> p) {
            parameter.kirchner.c1=p[0];
            parameter.gs.set_snow_tx(p[1]);
            parameter.gs.set_surface_magnitude(p[2]);
            parameter.ae.set_ae_scale_factor(p[3]);
        }

        double get(size_t i) const {
            double res = 0.0;
            switch (i)
            {
                case 0:
                    res = parameter.kirchner.c1;
                    break;
                case 1:
                    res = parameter.gs.TX();
                    break;
                case 2:
                    res = parameter.gs.surface_magnitude();
                    break;
                case 3:
                    res = parameter.ae.ae_scale_factor();
                    break;
                default:
                    throw std::runtime_error("Out of range.");
            }
            return res;
        }

        std::string get_name(size_t i) const {
            std::string res;
            switch (i)
            {
                case 0:
                    res = "c1";
                    break;
                case 1:
                    res = "TX";
                    break;
                case 2:
                    res = "surface_magnitude";
                    break;
                case 3:
                    res = "ae_scale_factor";
                    break;
                default:
                    throw std::runtime_error("Out of range.");
            }
            return res;
        }


    };




    template<class PA>
    struct PTGSKTestModel {
        typedef PTGSKCell::parameter_t parameter_t;
        utctime t0;
        utctimespan dt;
        size_t n_times;
        mutable size_t n_evals = 0;
        parameter_t parameter;
        PTGSKCell::state_t s0;
        std::vector<PTGSKCell> model_cells;
        ta::fixed_dt time_axis;
        PTGSKTestModel(utctime t0,
                  utctimespan dt,
                  size_t n_times,
                  parameter_t p_init,
                  pt_gs_k::state_t s0)
          : t0(t0), dt(dt), n_times(n_times), parameter(p_init), s0(s0) {}
		std::vector<double> p_expanded;
        std::vector<double> p_min;
        std::vector<double> p_max;
		//Need to handle expanded/reduced parameter vector based on min..max range to optimize speed for bobyqa
		bool is_active_parameter(size_t i) const { return fabs(p_max[i] - p_min[i]) > 0.000001; }
		std::vector<double> reduce_p_vector(const std::vector<double>& fp) const {
			std::vector<double> r; r.reserve(fp.size());
			for (size_t i = 0; i < fp.size(); ++i) {
				if (is_active_parameter(i))
					r.push_back(fp[i]);// only pick values that are active in optimization
			}
			return r;
		}
		std::vector<double> expand_p_vector(const std::vector<double>& rp) const {
			std::vector<double> r; r.reserve(p_expanded.size());
			size_t j = 0;
			for (size_t i = 0; i < p_expanded.size(); ++i) {
				if (is_active_parameter(i))
					r.push_back(rp[j++]);// pick from reduced vector
				else
					r.push_back(p_expanded[i]); // just use already set class global parameter
			}
			return r;
		}
        void set_parameter_ranges(std::vector<double>& p_min, std::vector<double>& p_max) {
            this->p_min = p_min;
            this->p_max = p_max;
        }


        pt_gs_k::state_t state(size_t i) {
            pt_gs_k::state_t  s = s0;
            s.gs.albedo += 0.3*(double)i/(model_cells.size() - 1);
            return s;
        }

        void init(size_t n_dests, ta::fixed_dt& time_axis) {
            this->time_axis = time_axis;
            xpts_t temp;
            xpts_t prec;
            xpts_t rel_hum;
            xpts_t wind_speed;
            xpts_t radiation;
            for (size_t i = 0; i < n_dests; ++i) {
                shyfttest::create_time_series(temp, prec, rel_hum, wind_speed, radiation, t0, dt, n_times);
                pt_gs_k::state_t s = state(i);
                model_cells.emplace_back(temp, prec, wind_speed, rel_hum, radiation, s,  parameter, 0);
            }
        }

        catchment_t run(parameter_t param) {
            catchment_t catchment_discharge(time_axis, 0.0);
			//shyft::time_axis::fixed_dt
			auto state_time_axis=time_axis;
			state_time_axis.n++;//add room for end-state
            size_t i = 0;
            for_each(model_cells.cbegin(), model_cells.cend(), [this, &i, &param, &catchment_discharge,&state_time_axis] (PTGSKCell d) {
                StateCollector<ta::fixed_dt> sc(state_time_axis);
                DischargeCollector<ta::fixed_dt> rc(1000*1000, time_axis);
                pt_gs_k::state_t s = state(i++);
                pt_gs_k::response_t r;

                pt_gs_k::run_pt_gs_k<shyft::time_series::direct_accessor,pt_gs_k::response_t>(
                      d.geo_cell_info(),
                      param,
                      time_axis,0,0,
                      d.temperature(),
                      d.precipitation(),
                      d.wind_speed(),
                      d.rel_hum(),
                      d.radiation(),
                      s,
                      sc,
                      rc);
                d.add_discharge(rc.avg_discharge, catchment_discharge, time_axis); // Aggregate discharge into the catchment
            });
            return catchment_discharge;
        }

        catchment_t target;

        void set_measured_discharge(catchment_t discharge) {target = discharge;}

        double operator() (const column_vector& p_s) {
            ++n_evals;
			auto rp = from_scaled(p_s);
			auto p = expand_p_vector(rp);
            parameter.set(p);
            catchment_t result = run(parameter);
            return nash_sutcliffe_goal_function(result,target);
        }
        double operator() (const vector<double>& p_s) {
            ++n_evals;
			auto rp = from_scaled(p_s);
			auto p = expand_p_vector(rp);
            parameter.set(p);
            catchment_t result = run(parameter);
            return nash_sutcliffe_goal_function(result,target);
        }
		std::vector<double> to_scaled(std::vector<double>& rp) {
			std::vector<double> p_s;
			auto rp_min = reduce_p_vector(p_min);
			auto rp_max = reduce_p_vector(p_max);
			const size_t n_params = rp.size();
			p_s.reserve(n_params);
			for (size_t i = 0; i < n_params; ++i)
				p_s.emplace_back((rp[i] - rp_min[i]) / (rp_max[i] - rp_min[i]));
			return p_s;
		}

		std::vector<double> from_scaled(const column_vector& p_s) const {
			std::vector<double> p;
			auto rp_min = reduce_p_vector(p_min);
			auto rp_max = reduce_p_vector(p_max);
			p.reserve(p_s.nr());
			for (int i = 0; i < p_s.nr(); ++i)
				p.emplace_back((rp_max[i] - rp_min[i])*p_s(i) + rp_min[i]);
			return p;
		}
		std::vector<double> from_scaled(const vector<double>& p_s) const {
			std::vector<double> p;
			auto rp_min = reduce_p_vector(p_min);
			auto rp_max = reduce_p_vector(p_max);
			p.reserve(p_s.size());
			for (size_t i = 0; i < p_s.size(); ++i)
				p.emplace_back((rp_max[i] - rp_min[i])*p_s[i] + rp_min[i]);
			return p;
		}
    };

} //  shyfttest

TEST_SUITE("calibration") {
TEST_CASE("test_dummy") {
    std::vector<double> target = {-5.0,1.0,1.0,1.0};
    std::vector<double> lower = {-10, 0, 0, 0};
    std::vector<double> upper = {-4, 2, 2, 2};
    std::vector<double> x = {-9.0, 0.5, 0.9, 0.3};
    std::vector<double> x2 = {-9.0, 0.5, 0.9, 0.3};
    std::vector<double> x3 = {-9.0, 0.5, 0.9, 0.3};
    shyfttest::TestModel model(target, lower, upper);
    double residual = model_calibration::min_bobyqa(model, x, 500,0.05, 1.0e-16);
    TS_ASSERT_DELTA(residual, 0.0, 1.0e-16);
    for (size_t i = 0; i < x.size(); ++i)
        TS_ASSERT_DELTA(x[i], target[i], 1.0e-16);
    /// the dream algorithm seems to have very sloppy accuracy
    /// most likely a bug
    bool verbose = getenv("SHYFT_VERBOSE")!=nullptr;
    if(verbose) TS_WARN("DREAM: simple tests with just 0.3 accuracy requirement");
    double residual2=model_calibration::min_dream(model,x2,10000);
    TS_ASSERT_DELTA(residual2, 0.0, 1.0e-1);
    for (size_t i = 0; i < x2.size(); ++i)
        TS_ASSERT_DELTA(x2[i], target[i], 0.3);


    if(verbose) TS_WARN("SCEUA: simple tests with just 0.001 accuracy requirement");
    double residual3=model_calibration::min_sceua(model,x3,10000,0.001,0.0001);
    TS_ASSERT_DELTA(residual3, 0.0, 1.0e-1);
    for (size_t i = 0; i < x3.size(); ++i)
        TS_ASSERT_DELTA(x3[i], target[i], 0.02);


}


TEST_CASE("test_simple") {
    using namespace shyft::core::model_calibration;
    // Make a simple model setup
    calendar cal;
    utctime t0 = cal.time(YMDhms(2014,8,1,0,0,0));
    const int nhours=1;
    utctimespan dt = 1*deltahours(nhours);
    size_t num_days= 10;
    ta::fixed_dt time_axis(t0, dt, 24*num_days);

    size_t n_dests = 10*10;
    std::vector<PTGSKCell> destinations;
    destinations.reserve(n_dests);

    pt::parameter pt_param;
    gs::parameter gs_param;
    ae::parameter ae_param;
    kr::parameter k_param;

    pc::parameter p_corr_param;
    using TestModel = shyfttest::PTGSKTestModel<pt_gs_k::parameter_t>;
    TestModel::parameter_t parameter{pt_param, gs_param, ae_param, k_param,  p_corr_param};

    // Some initial state is needed:

	kr::state kirchner_state{ 5.0 };
    gs::state gs_state={0.6, 1.0, 0.0, 0.7 /*1.0/(gs_param.snow_cv*gs_param.snow_cv)*/, 10.0, -1.0, 0.0, 0.0};
    pt_gs_k::state_t state{ gs_state, kirchner_state};

    // Set up the model and store the synthetic discharge
	shyfttest::PTGSKTestModel<TestModel::parameter_t> model(t0, dt, 24*num_days, parameter, state);
    model.init(n_dests, time_axis);
    catchment_t synthetic_discharge = model.run(parameter);
    model.set_measured_discharge(synthetic_discharge);

    // Define parameter ranges
    const size_t n_params = model.parameter.size();
    std::vector<double> lower; lower.reserve(n_params);
    std::vector<double> upper; upper.reserve(n_params);
    const size_t n_calib_params = 4;
    for (size_t i = 0; i < n_params; ++i) {
        double v = model.parameter.get(i);
        lower.emplace_back(i < n_calib_params ? 0.5*v : v);
        upper.emplace_back(i < n_calib_params ? 1.5*v : v);
    }

    model.set_parameter_ranges(lower, upper);

    // Perturb parameter set
    std::vector<double> x(n_params);
    std::default_random_engine re; re.seed(1023);
	for (size_t i = 0; i < n_params; ++i) {
        if(i<n_calib_params) {
            x[i] = lower[i]< upper[i] ? std::uniform_real_distribution<double>(lower[i], upper[i])(re) : std::uniform_real_distribution<double>(upper[i], lower[i])(re);
        } else {
            x[i] = (lower[i] + upper[i])*0.5;
        }
	}
    using std::cout;
    using std::endl;
    model.p_expanded = x;
    bool verbose = getenv("SHYFT_VERBOSE")!=nullptr;
    if(verbose) {
        cout << "True Parameter settings:" << endl;
        cout << "========================" << endl;
        for (size_t i = 0; i < n_params; ++i)
            cout << model.parameter.get_name(i) << " = " << model.parameter.get(i) << endl;
        cout << "===============" << endl;
        cout << "Initial guess :" << endl;
        cout << "===============" << endl;
        for (size_t i = 0; i < n_params; ++i)
            cout << model.parameter.get_name(i) << " = " << x[i] << endl;
        cout << "====================" << endl;
        cout << "Found:" << endl;
        cout << "Found:" << endl;
    }
    // Solve the optimization problem
	size_t n_max = 15000;
	const double tr_start = 0.1;
	const double tr_end = 1e-6;
	auto rx = model.reduce_p_vector(x);
    double residual = min_bobyqa(model, rx, n_max, tr_start, tr_end);//min_sceua(model,rx,n_max,0.001,0.001);//min_dream(model,rx,n_max);//min_bobyqa(model, rx,n_max, tr_start,tr_end);
	x = model.expand_p_vector(rx);
	model.p_expanded = x;
    if(verbose) {
        cout << "====================" << endl;
        for (size_t i = 0; i < n_params; ++i)
            cout << model.parameter.get_name(i) << " = " << x[i] << endl;
        cout << "====================" << endl;
        cout << "Residual = " << residual << endl;
        cout << "Number of function evals = " << model.n_evals << endl;
        cout << "====================" << endl;
        cout << "Trying once more:" << endl;
    }
    model.n_evals = 0;
	rx = model.reduce_p_vector(x);
    residual = min_bobyqa(model, rx, n_max*2, tr_start/2, tr_end/2);//min_dream(model,rx,n_max);//
	x = model.expand_p_vector(rx);
	model.p_expanded = x;
	if(verbose) {
        cout << "====================" << endl;
        cout << "min_bobyqa:" << endl;
        for (size_t i = 0; i < n_params; ++i)
            cout << model.parameter.get_name(i) << " = " << x[i] << endl;
        cout << "====================" << endl;
        cout << "Residual = " << residual << endl;
        cout << "Number of function evals = " << model.n_evals << endl;
        cout << "====================" << endl;
	}

}
TEST_CASE("test_nash_sutcliffe_goal_function") {
	calendar utc;
	utctime start = utc.time(YMDhms(2000, 1, 1, 0, 0, 0));
	utctimespan dt = deltahours(1);
	size_t n = 1000;
	ta::fixed_dt ta(start, dt, n);

	pts_t obs(ta, 0.0);
	pts_t sim(ta, 0.0);
	for (size_t i = 0; i < n; i++) {
		double a = M_PI*100.0*i / n;
		double ov = 10.0 + 5 * sin(a);
		obs.set(i,ov);
		sim.set(i, i<n/2?ov:0.0);//9.5 + 4 * sin(a + 0.02)
	}
	double nsg_perfect = nash_sutcliffe_goal_function(obs, obs);
	TS_ASSERT_DELTA(nsg_perfect, 0.0, 0.000001);// because obs and obs are equal, 0.0
	double nsg_ok = nash_sutcliffe_goal_function(obs, sim);
	TS_ASSERT_DELTA(nsg_ok, 4.5, 0.000001);// because obs and obs are equal, 0.0

	// manually testing with two values, and hardcoded formula
	ta::fixed_dt ta1(start, dt, 2);
	pts_t o1(ta1, 1.0); o1.set(1, 10.0);
	pts_t s1(ta1, 2.0);
	double nsg1 = nash_sutcliffe_goal_function(o1, s1);

	TS_ASSERT_DELTA(nsg1,
		(((2.0 - 1.0)*(2.0 - 1.0)) + ((2.0 - 10.0)*(2.0 - 10.0))) /
		((1.0 - 5.5)*(1.0 - 5.5) + (10.0 - 5.5)*(10.0 - 5.5)), 0.00001);

}
TEST_CASE("test_kling_gupta_goal_function") {
    	calendar utc;
	utctime start = utc.time(YMDhms(2000, 1, 1, 0, 0, 0));
	utctimespan dt = deltahours(1);
	size_t n = 1000;
	ta::fixed_dt ta(start, dt, n);

	pts_t obs(ta, 0.0);
	pts_t sim(ta, 0.0);
	for (size_t i = 0; i < n; i++) {
		double a = M_PI*100.0*i / n;
		double ov = 10.0 + 5 * sin(a);
		obs.set(i,ov);
		sim.set(i, i<n/2?ov:0.0);//9.5 + 4 * sin(a + 0.02)
	}
	double nsg_perfect = kling_gupta_goal_function<dlib::running_scalar_covariance<double>>(obs, obs,1.0,1.0,1.0);
	TS_ASSERT_DELTA(nsg_perfect, 0.0, 0.000001);// because obs and obs are equal, 0.0
	double nsg_ok = kling_gupta_goal_function<dlib::running_scalar_covariance<double>>(obs, sim,1.0,1.0,1.0);
	TS_ASSERT_DELTA(nsg_ok, 1.0272, 0.0001);// because obs and obs are equal, 0.0

	// manually testing with two values, and hardcoded formula
	ta::fixed_dt ta1(start, dt, 3);
	pts_t o1(ta1, 1.0); o1.set(1, 10.0);
	pts_t s1(ta1, 2.0); s1.set(2, 7.0);
	double nsg1 = kling_gupta_goal_function<dlib::running_scalar_covariance<double>>(o1, s1,1.0,1.0,1.0);

	TS_ASSERT_DELTA(nsg1,1.5666, 0.0001);
    SUBCASE("avg ratio") {
        o1.set(1, 1.0);
        s1.set(2, 2.0);
        double a = kling_gupta_goal_function<dlib::running_scalar_covariance<double>>(o1, s1, 0.0, 1.0, 0.0);
        FAST_CHECK_LE( fabs(a - 1.0) , 1e-12);
    }
}
TEST_CASE("test_abs_diff_sum_goal_function") {
    calendar utc;
    utctime start = utc.time(YMDhms(2000, 1, 1, 0, 0, 0));
    utctimespan dt = deltahours(1);
    ta::fixed_dt ta1(start, dt, 3);
    pts_t o1(ta1, 1.0); o1.set(1, 10.0);
    pts_t s1(ta1, 2.0); s1.set(2, 7.0);
    double ads1 = abs_diff_sum_goal_function(o1, s1);

    TS_ASSERT_DELTA(fabs(1.0-2.0)+fabs(10.0-2.0)+fabs(1.0-7.0), ads1, 0.0001);
    SUBCASE("with-nan") {
        o1.set(1, shyft::nan);
        double ads2 = abs_diff_sum_goal_function(o1, s1);
        TS_ASSERT_DELTA(fabs(1.0 - 2.0) + fabs(0) + fabs(1.0 - 7.0), ads2, 0.0001);
    }
}

}
