#pragma once

#include <armadillo>

namespace shyft {
    namespace core {
        namespace kalman {
            using namespace std;
            using namespace shyft::core;


            /** \brief kalman::state represent the state of the specialized kalman-filter as implemented by met.no
             *
             */
            struct state {
                arma::vec x;///< best estimate of bias, size is n forecast periods each day(eg. 3h~ n=8)
                arma::vec k;///< kalman gain factors
                arma::mat P;///< error covariance of best estimate
                arma::mat W;///< W process noise (currently constant after init)
                int size() const {return (int)x.n_elem;}
                state(){}
                state(int n_daily_observations,double covariance_init,double hourly_correlation, double process_noise_init) {
                    x = arma::vec(n_daily_observations);x.fill(0.0);
                    k = arma::vec(n_daily_observations);k.fill(0.0);
                    P = make_covariance(n_daily_observations,  covariance_init,hourly_correlation);
                    W = make_covariance(n_daily_observations,process_noise_init,hourly_correlation);
                }
                static inline arma::mat make_covariance(int n_daily_observations,double covariance_init,double hourly_correlation) {
                    arma::mat P(n_daily_observations,n_daily_observations);
                    double dt_hours=24.0/n_daily_observations;
                    for(int i=0;i<n_daily_observations;++i) {
                        for(int j=0;j<n_daily_observations;++j) {
                            // the correlation is proportional to the daily-folded distance between i and j
                            // so the last time-step in a day is closest to the first hour of the next day etc.
                            // this is essential to the time-folding and 'solar' driven pattern of bias.
                            int ix_diff= std::min(abs(i-j),n_daily_observations-abs(i-j));
                            P.at(i,j) = covariance_init*pow(hourly_correlation,ix_diff*dt_hours);
                        }
                    }
                    return P;
                }
            };


            /** \brief parameters to tune the kalman-filter
            */
            struct parameter {
                int n_daily_observations=8;/// 'mDim' dt= n_daily_obserations/24hours
                double hourly_correlation=0.93;/// correlation from one-hour to the next
                double covariance_init=0.5;/// for the error covariance P matrix
                double std_error_bias_measurements=2.0;///'mV', st.dev
                double ratio_std_w_over_v=0.06;/// st.dev W /st.dev V
                /** \brief constructs a kalman parameter
                 *
                 * where the values have reasonable defaults
                 *
                 * \param n_daily_observations typically 8, 3h sampling frequency
                 * \param hourly_correlation correlation from one-hour to the next
                 * \param covariance_init for the error covariance P matrix
                 * \param std_error_bias_measurements std.dev for bias measurements
                 * \param ratio_std_w_over_v w= process noise, v= measurement noise
                */
                parameter(int n_daily_observations=8,
                       double hourly_correlation=0.93,
                       double covariance_init=0.5,
                       double std_error_bias_measurements=2.0,
                       double ratio_std_w_over_v=0.06
                       ):
                       n_daily_observations(n_daily_observations),
                       hourly_correlation(hourly_correlation),
                       covariance_init(covariance_init),
                       std_error_bias_measurements(std_error_bias_measurements),
                       ratio_std_w_over_v(ratio_std_w_over_v) {}

            };

            /** \brief Specialized kalman filter for temperature (e.g.:solar-driven bias patterns)
             *
             *
             * The observation point (t,v) is folded on to corresponding period
             * of the day (number of periods in a day is parameterized, typically 8).
             * A simplified kalman filter algorithm using the forecast bias as
             * the state-variable.
             * Observed bias (fc-obs) is feed into the filter and establishes the
             * kalman best predicted estimates (x) for the bias.
             * This bias can then be used as a correction to forecast in the future
             * to compensate for systematic forecast errors.
             *
             * Credits: Thanks to met.no for providing the original source for this algorithm.
             *
             * \sa <a href="https://en.wikipedia.org/wiki/Kalman_filter">Kalman Filter</a>
             *
             */
            struct filter {
                parameter p;///<< filter parameters determining number of daily points, plus kalman-parameters
                filter(){}
                filter(parameter p):p(p){}

                ///< return a kalman::state initialized with the filter parameters, 's0'- startstate
                state create_initial_state() const {
                    return state(p.n_daily_observations,p.covariance_init,p.hourly_correlation,p.std_error_bias_measurements!=0.0?p.ratio_std_w_over_v*p.std_error_bias_measurements:100.0);
                }

                ///< given any time, fold it into the 'day' and part (index) of day.
                inline int fold_to_daily_observation(utctime t) const {
                    return ((t/deltahours(1) + 200L*24*365) % 24)*p.n_daily_observations/24;// in short, add 200 years (ensure positve num before %)
                }

                /** \brief update the kalman::filter p with the observed_bias for
                 * a specific period starting with utctime t.
                 *
                 * \param observed_bias nan if no observation is available otherwise obs-fc
                 * \param t utctime of observation, this filter utilizes daily solar patterns, so time
                 *        in day-cycle is the only important aspect.
                 * \param s contains the kalman state x,k and P, updated at exit
                 *
                 */
                void update(double observed_bias,utctime t,state& s) const {
                   /// Compute Pt|t-1. This increases the covariance.
                   s.P= s.P + s.W;
                   if(isfinite(observed_bias)) {
                        /// compute Kt
                        int ix= fold_to_daily_observation(t);
                        s.k = s.P.col(ix) / (s.P.at(ix,ix) + p.std_error_bias_measurements);
                        /// compute Pt|t
                        s.P = 1.0 - s.k[ix]*s.P;
                        /// compute xt|t
                        s.x = s.x + s.k*(observed_bias - s.x[ix]);
                    } else {
                        /// Missing obs or forecast. P has already been increased, do not update x.
                        /// TODO: Does the kalman gain need to be updated?
                    }
                }
            };

            /** \brief bias_predictor for forecast-observation for solar-influenced signals
             *
             * Using a kalman filter technique developed by met.no/gridpp to
             * predict daily bias for solar driven signals (temperature)
             *
             */
            struct bias_predictor {
                filter f;///<< the filter we are using
                state s; ///<< the current state of the filter

                bias_predictor() {}
                bias_predictor(const filter& f):f(f),s(f.create_initial_state()) {}
                bias_predictor(const filter& f,const state& s):f(f),s(s){}

                /** updates the state s, that includes the predicted bias(.x),
                 * using forecasts and observation time-series.
                 *
                 * \tparam fc_ts a template class for forecast time-series
                 * \tparam obs_ts a template class for the observation time-series
                 * \tparam ta timeaxis type, e.g. fixed_dt, should(but don't need to) match filter-timestep
                 * \param fc_ts_set of type const vector<fc_ts>& contains the forecast time-series
                 * \param observation_ts of type obs_ts with the observations
                 * \param time_axis of type ta, time-axis that can be used for average_accessors
                 */
                template<class fc_ts,class obs_ts,class ta>
                void update_with_forecast(const std::vector<fc_ts>& fc_ts_set,const obs_ts& observation_ts, const ta& time_axis) {
                    shyft::timeseries::average_accessor<obs_ts,ta> obs(observation_ts,time_axis);
                    for(const auto& fcts:fc_ts_set) {
                        shyft::timeseries::average_accessor<fc_ts,ta> fc(fcts,time_axis);
                        for(size_t i=0;i<time_axis.size();++i) {
                            double bias = fc.value(i) - obs.value(i);
                            if(isfinite(bias)) // we skip nan's here, just want to 'learn' from non-nan- values
                                f.update(bias,time_axis.time(i),s);
                        }
                    }
                }
                //todo: get_bias_ts(utcperiod range) -> a timeseries with fixed_dt time-axis and values from s.x projected out on the time-axis
            };
        }
    }
}
