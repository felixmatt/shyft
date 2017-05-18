#include "test_pch.h"
#include "mocks.h"
#include "api/api.h" // looking for GeoPointSource, and TemperatureSource(realistic case)
#include "api/time_series.h" // looking for apoint_ts, the api exposed ts-type(realistic case)
#include "core/kalman.h"


namespace shyfttest {
    /// signal generator for temperature and forecast
    /// including random noise and a constant day-hour
    /// dependent fixed bias
    struct temperature {
        double mean= 10.0;//deg C
        double w = 2*3.1415/(24.0*3600.0);
        double p = - 2*3.1415*8/24;// cross zero at 8, max at 14
        double a = 2.0;// amplitude deg/C day/night
        utctime t0=0;// 1970.
        calendar utc;

        double bias_offset_day_pattern[8]={2.0,1.9,1.8,1.7,1.8,1.8,1.9,2.0};

        double bias_offset(utctime t) const {
            size_t ix= utc.calendar_units(t).hour/3;// 0..7
            return bias_offset_day_pattern[ix];
        }
        mutable std::normal_distribution<double> bias_noise;

        temperature(double stdev_noise=0.0):bias_noise(0.0,stdev_noise) {

        }
        ///< this is the observation
        double observation(utctime t) const {
            return mean + a*sin( w *(t-t0)+ p);
        }
        double bias(utctime t) const {
            static std::default_random_engine generator;
            return bias_offset(t) + bias_noise(generator);// +  bias_noise*random()..
        }
        double forecast(utctime t) const {
            return observation(t)+bias(t);
        }
    };

    void print(std::ostream &os, const shyft::core::kalman::state& s, bool full_print=false) {
        s.x.t().print("x:");
        if(full_print) {
            s.k.print("k:");
            s.P.print("P:");
        }
    }
}
TEST_SUITE("kalman") {
TEST_CASE("test_filter") {
    using namespace shyfttest;
    // prepare the time-dimension, using time-axis
    calendar utc;
    utctimespan dt=deltahours(1);
    size_t n=24;
    timeaxis_t ta(utc.time(2000,1,1),dt,n); /// for the test,this is the governing time-axis (corresponding to region_model.time_axis during run/interpolation)
    kalman::parameter p;
    kalman::filter f(p);
    auto s = f.create_initial_state();

    temperature fx(0.1);
    /// update with n_days=3 of observations (
    for(int i=0;i<8*3;++i) {
        utctime t = fx.t0+ deltahours(3*i);
        //print(cout,p,false);
        f.update(fx.bias(t),t,s);
    }
    /// verify that bias estimate has converged to the bias_offset (despite small noise)
    for(auto i=0;i<8;++i) {
        double bias_estimate=s.x(i);
        utctime t = fx.t0 + deltahours(3*i);
        TS_ASSERT_DELTA(fx.bias_offset(t),bias_estimate,0.21);
    }
    double no_value=std::numeric_limits<double>::quiet_NaN();
    /// update//forecast up to 10 days with no observations
    auto s_last_known=s;
    for(auto i=0;i<8*3;++i) {
        auto time_step=8*3+i;
        utctime t = fx.t0+ deltahours(3*time_step);
        //print(cout,p,false);
        f.update(no_value,t,s);
    }
    /// verify p.x(i) is equal to p_last_known
    /// according to met.no code, there is no change in prediction pattern
    /// while not feeding data into the loop, only increases the error covariance P.
    for(auto i=0;i<s_last_known.size();++i)
        TS_ASSERT_DELTA(s_last_known.x(i),s.x(i),0.01);
}

TEST_CASE("test_bias_predictor") {
    using namespace shyfttest;
    using namespace std;
    using pts_t=shyft::time_series::point_ts<timeaxis_t>;
    // prepare the time-dimension, using time-axis
    calendar utc;
    utctimespan dt=deltahours(1);
    size_t n=24;
    auto t0=utc.time(2000,1,1);
    timeaxis_t ta(t0,dt,n); /// for the test,this is the governing time-axis (corresponding to region_model.time_axis during run/interpolation)
    temperature fx(0.1);

    pts_t observation(ta,0.0);
    for(size_t i=0;i<ta.size();++i) observation.set(i,fx.observation(ta.time(i)));

    vector<pts_t> fc_set;
    for(size_t i=0;i<4;++i) {
        size_t fc_n=36;//e.g. arome
        timeaxis_t fc_ta(t0+deltahours(6*i),dt,fc_n);// offset fc with 6hours, so start at 00,06,12,18
        pts_t fc(fc_ta,0.0);// generate the forecast (later, add different offset's and noise please.)
        for(size_t j=0;j<fc_ta.size();++j) fc.set(j,fx.forecast(fc_ta.time(j)));
        fc_set.push_back(fc);
    }

    kalman::parameter p(8,0.93,0.5,2.0,0.22);
    kalman::filter f(p);
    kalman::bias_predictor bias_predictor(f);

    timeaxis_t pred_ta(t0,deltahours(3),n/3);// the predictor time-axis that covers the observation time-series ta
    bias_predictor.update_with_forecast(fc_set,observation,pred_ta);

    // Now verify the bias_predictor have learned the bias:
    for(auto i=0;i<8;++i) {
        double bias_estimate=bias_predictor.s.x(i);
        utctime t = fx.t0 + deltahours(3*i);
        TS_ASSERT_DELTA(fx.bias_offset(t),bias_estimate,0.4);
    }
}

TEST_CASE("test_running_predictor") {
    using namespace shyfttest;
    using namespace std;
    using pts_t=shyft::time_series::point_ts<timeaxis_t>;
    // prepare the time-dimension, using time-axis
    calendar utc;
    utctimespan dt=deltahours(1);
    size_t n=24*10;
    auto t0=utc.time(2000,1,1);
    timeaxis_t ta(t0,dt,n); /// for the test,this is the governing time-axis (corresponding to region_model.time_axis during run/interpolation)
    temperature fx(0.1);

    pts_t observation(ta,0.0);
    pts_t merged_forecast(ta,0.0);
    for(size_t i=0;i<ta.size();++i) {
        observation.set(i,fx.observation(ta.time(i)));
        merged_forecast.set(i,fx.forecast(ta.time(i)));
    }

    kalman::parameter p;
    kalman::filter f(p);
    kalman::bias_predictor bias_predictor(f);
    timeaxis_t pred_ta(t0,deltahours(3),n/3);// the predictor time-axis that covers the observation time-series ta
    auto bias_ts = bias_predictor.compute_running_bias<pts_t>(merged_forecast,observation,pred_ta);
    TS_ASSERT_EQUALS(bias_ts.size(),pred_ta.size());
    for(size_t i=0;i<8;++i) {
        TS_ASSERT_DELTA(bias_ts.value(i),0.0,0.0001);// assume first values are 0.0 since there is no learning.
    }
    for(size_t i=bias_ts.size()-8;i<bias_ts.size();++i) {
        auto t= bias_ts.time(i);
        TS_ASSERT_DELTA(fx.bias_offset(t),bias_ts.value(i),0.2);// at the end it should have a quite correct pattern
    }
}
}
