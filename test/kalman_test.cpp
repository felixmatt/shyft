#include "test_pch.h"
#include "kalman_test.h"
#include "mocks.h"
#include "api/api.h" // looking for GeoPointSource, and TemperatureSource(realistic case)
#include "api/timeseries.h" // looking for apoint_ts, the api exposed ts-type(realistic case)
#include <armadillo>
using namespace shyft::core;

namespace shyft {
    namespace core {
        namespace kalman {
            using namespace std;

            /** \brief kalman::parameter as implemented by met.no
             * \note that n_daily_observation must be >1 and 24/x must be an integer
             */

            struct parameter {
                arma::vec x;///< best estimate of bias, size is n forecast periods each day(eg. 3h~ n=8)
                arma::vec k;///< kalman gain factors
                arma::mat P; ///< error covariance of best estimate
                int size() const {return (int)x.n_elem;}

                parameter(int n_daily_observations,double covariance_init,double hourly_correlation) {
                    x=arma::vec(n_daily_observations);x.fill(0.0);
                    k=arma::vec(n_daily_observations);k.fill(0.0);
                    P = make_covariance(n_daily_observations,covariance_init,hourly_correlation);

                }
                static arma::mat make_covariance(int n_daily_observations,double covariance_init,double hourly_correlation) {
                    arma::mat P(n_daily_observations,n_daily_observations);
                    double dt_hours=24.0/n_daily_observations;
                    for(int i=0;i<n_daily_observations;++i) {
                        for(int j=0;j<n_daily_observations;++j) {
                            int ix_diff= std::min(abs(i-j),n_daily_observations-abs(i-j));
                            P(i,j) = covariance_init*pow(hourly_correlation,ix_diff*dt_hours);
                        }
                    }
                    return P;
                }
            };
            void print(ostream &os, const parameter& p, bool full_print=false) {
                p.x.t().print("x:");
                if(full_print) {
                    p.k.print("k:");
                    p.P.print("P:");
                }
            }

            struct filter {

                int n_daily_observations=8;/// 'mDim' dt= n_daily_obserations/24hours
                double hourly_correlation=0.93;/// correlation from one-hour to the next
                double covariance_init=0.5;/// for the error covariance P matrix
                double std_error_bias_measurements=2.0;///'mV', st.dev
                double ratio_std_w_over_v=0.06;/// st.dev W /st.dev V
                arma::mat W;/// W process noise (constant after initi)

                int delta_hours() const {return 24/n_daily_observations;}

                filter(int n_daily_observations=8,/// 'mDim' dt= n_daily_obserations/24hours
                       double hourly_correlation=0.93,/// correlation from one-hour to the next
                       double covariance_init=0.5,/// for the error covariance P matrix
                       double std_error_bias_measurements=2.0,///st.dev
                       double ratio_std_w_over_v=0.06 ///'mRatio' w= process noise, v= measurement noise
                       ):
                       n_daily_observations(n_daily_observations),
                       hourly_correlation(hourly_correlation),
                       covariance_init(covariance_init),
                       std_error_bias_measurements(std_error_bias_measurements),
                       ratio_std_w_over_v(ratio_std_w_over_v) {
                    W=parameter::make_covariance(n_daily_observations,
                                                 std_error_bias_measurements!=0.0?ratio_std_w_over_v*std_error_bias_measurements:100.0,
                                                 hourly_correlation);

                }
                ///< return a kalman::parameter initialized with the filter parameters, 'p0'- startstate
                parameter initialize() {
                    return parameter(n_daily_observations,covariance_init,hourly_correlation);
                }

                ///< given any hour, fold it into the 'day' and part (index) of day.
                int compute_index_from_hour(int time_step_ix) const {
                    int ix= round(double(time_step_ix)*n_daily_observations/24);
                    ix= ix % n_daily_observations;
                    return ix;
                }

                /** \brief update the kalman::filter p with the observed_bias for
                 * a specific 'hour' (still clumsy,we need utctime here..)
                 *
                 * \param observed_bias nan if no observation is available otherwise obs-fc
                 * \param hour range 0.. n, this filter utilizes daily solar patterns, so time
                 *        in day-cycle is the only important aspect.
                 * \param p contains the kalman parameter x,k and P, updated at exit
                 * \return void
                 *
                 */
                void update(double observed_bias,int hour,parameter& p) {
                   /// Compute Pt|t-1. This increases the covariance.
                   p.P= p.P + W;
                   if(isfinite(observed_bias)) {
                        /// compute Kt
                        int ix= compute_index_from_hour(hour);
                        for(int i=0;i<p.size();++i)
                            p.k(i) = p.P(ix,i)/(p.P(ix,ix) + std_error_bias_measurements);
                        /// compute Pt|t
                        for(int i=0;i<p.size();++i)
                            for(int j=0;j<p.size();++j)
                                p.P(i,j)= (1.0 - p.k(ix)*p.P(i,j));

                        /// compute xt|t
                        double x_ix=p.x(ix);
                        p.x = p.x + p.k*(observed_bias - x_ix);
                    } else {
                        /// Missing obs or forecast. P has already been increased, do not update x.
                        /// TODO: Does the kalman gain need to be updated?
                   }
                }
            };
        }
    }
}
namespace shyfttest {
    /// signal generator for temperature and forecast
    /// including random noise and a constant day-hour
    /// dependent fixed bias
    struct temperature {
        double mean= 10.0;//deg C
        double w = 2*3.1415/(24.0*3600.0);
        double p = - 2*3.1415*8/24;// cross zero at 8, max at 14
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
            return mean + sin( w *(t-t0)+ p);
        }
        double bias(utctime t) const {
            static std::default_random_engine generator;
            return bias_offset(t) + bias_noise(generator);// +  bias_noise*random()..
        }
        double forecast(utctime t) const {
            return observation(t)+bias(t);
        }
    };
}

void kalman_test::test_filter_workbench() {
    using namespace shyfttest;
    // prepare the time-dimension, using time-axis
    calendar utc;
    utctimespan dt=deltahours(1);
    size_t n=24;
    timeaxis_t ta(utc.time(2000,1,1),dt,n); /// for the test,this is the governing time-axis (corresponding to region_model.time_axis during run/interpolation)

    kalman::filter f;
    auto p = f.initialize();

    temperature fx(0.1);
    /// update with n_days=3 of observations (
    for(int i=0;i<8*3;++i) {
        utctime t = fx.t0+ deltahours(3*i);
        //print(cout,p,false);
        f.update(fx.bias(t),3*i,p);
    }
    /// verify that bias estimate has converged to the bias_offset (despite small noise)
    for(auto i=0;i<8;++i) {
        double bias_estimate=p.x(i);
        utctime t = fx.t0 + deltahours(3*i);
        TS_ASSERT_DELTA(fx.bias_offset(t),bias_estimate,0.2);
    }
    double no_value=std::numeric_limits<double>::quiet_NaN();
    /// update//forecast up to 10 days with no observations
    auto p_last_known=p;
    for(auto i=0;i<8*3;++i) {
        auto time_step=8*3+i;
        //utctime t = fx.t0+ deltahours(3*time_step);
        //print(cout,p,false);
        f.update(no_value,time_step*3,p);
    }
    /// verify p.x(i) is equal to p_last_known
    /// according to met.no code, there is no change in prediction pattern
    /// while not feeding data into the loop, only increases the error covariance P.
    for(auto i=0;i<p_last_known.size();++i)
        TS_ASSERT_DELTA(p_last_known.x(i),p.x(i),0.01);
}

