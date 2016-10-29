#include "test_pch.h"
#include "glacier_melt_test.h"
#include "core/glacier_melt.h"
#include "core/timeseries.h"

namespace glacier_test_constant {
    const double EPS = 1.0e-10;
}
namespace shyft {
    namespace timeseries {

        /**\brief provides glacier melt ts
         *
         * Using supplied temperature and snow covered area time-series
         * computes the glacier melt in units of [m3/s] using the
         * the following supplied parameters:
         * dtf:day temperature factor (dtf),
         * gf: glacier fraction
         * area: area in m2
         *
         * \tparam TS a time-series type
         * \note that both temperature and snow covered area (sca) TS is of same type
         * \ref shyft::core::glacier_melt::step function
         */
		template<class TS>
		struct glacier_melt_ts {
			typedef typename TS::ta_t ta_t;
			TS temperature;
			TS sca;
			double glacier_fraction;
			double dtf;
			double area_m2;
			point_interpretation_policy fx_policy;
			const ta_t& time_axis() const { return temperature.ta; }

			/** construct a glacier_melt_ts
			 * \param temperature in degree Celsius
			 * \param sca snow covered area in range [0..1]
			 * \param glacier fraction in range [0..1]
			 * \param dtf degree timestep factor [mm/day/deg.C]; lit. values for Norway: 5.5 - 6.4 in Hock, R. (2003), J. Hydrol., 282, 104-115.
			 * \param area_m2 the area of the glacier in units of [m2]
			 */
			glacier_melt_ts(const TS& temperature, const TS& sca, double glacier_fraction, double dtf, double area_m2)
				:temperature(temperature), sca(sca),glacier_fraction(glacier_fraction),dtf(dtf),area_m2(area_m2)
				, fx_policy(fx_policy_t::POINT_AVERAGE_VALUE) {
			}

			point get(size_t i) const { return point(time_axis().time(i), value(i)); }
			size_t size() const { return time_axis().size(); }
			size_t index_of(utctime t) const { return time_axis().index_of(t); }
			//--
			double value(size_t i) const {
				if (i >= time_axis().size())
					return nan;
                utcperiod p=time_axis().period(i);
				double t_i = temperature.value(i);
				size_t ix_hint=i;// assume same indexing of sca and temperature
				double sca_i= average_value(sca,p,ix_hint,sca.fx_policy==fx_policy_t::POINT_INSTANT_VALUE);
				return shyft::core::glacier_melt::step(dtf, t_i,sca_i, glacier_fraction) * area_m2* 1000.0/3600.0; // mm/h * m2  *1000m/3600s-> m3/s
			}
			double operator()(utctime t) const {
				size_t i = index_of(t);
				if (i == string::npos)
					return nan;
                return value(i);
			}
		};
		// This is to allow this ts to participate in ts-math expressions
		template<class T> struct is_ts<glacier_melt_ts<T>> {static const bool value=true;};
        template<class T> struct is_ts<shared_ptr<glacier_melt_ts<T>>> {static const bool value=true;};

    }
}
using namespace shyft::core;

void test_glacier_melt_func() {
    using namespace shyft::timeseries;
    using ts_t= point_ts<timeaxis>;
    calendar utc;
    utctimespan dt=deltahours(1);
    timeaxis ta(utc.time(2016,10,1),dt,24);
    ts_t sca(ta,0.3,fx_policy_t::POINT_AVERAGE_VALUE);
    ts_t tmp(ta,10.0,fx_policy_t::POINT_AVERAGE_VALUE);
    double glacier_fraction=0.4;
    auto melt_temperature = max(0.0,tmp);
    auto melt_area = max(0.0,glacier_fraction- sca);
    double mm_h_factor= double(calendar::HOUR)/double(dt);
    double dtf=0.6;
    auto melt = melt_temperature*melt_area*dtf*mm_h_factor;

}
void glacier_melt_test::test_melt() {
    // Model parameters
    const double dtf = 6.0;
    glacier_melt::parameter p(dtf);

 	double sca = 0.0;
	while(sca <= 1.0)
    {
        double gf = 0.0;
        while(gf <= 1.0)
        {
            double temp = -10;
            while(temp <= 10)
            {

                double melt = glacier_melt::step(p.dtf, temp, sca, gf);
                TS_ASSERT(melt >= 0.0);
                if (temp <= 0.0)
                    TS_ASSERT_DELTA(melt, 0.0, glacier_test_constant::EPS);
                if (sca >= gf)
                    TS_ASSERT_DELTA(melt, 0.0, glacier_test_constant::EPS);
                if ((temp > 0.0) && (gf>sca))
                    TS_ASSERT(melt > 0.0)
                temp += 2.0;
            }
            gf += 0.2;
        }
        sca += 0.2;
    }

    // Glacier melt with different time steps, must return same value under same conditions (mm/h)
    double temp = 1.0;
    sca = 0.0;
    double gf = 1.0;

    const double melt_hourly =  glacier_melt::step(p.dtf, temp, sca, gf);
    const double melt_daily =  glacier_melt::step(p.dtf, temp, sca, gf);


    TS_ASSERT(melt_hourly > 0.0);
    TS_ASSERT(melt_daily > 0.0);
    TS_ASSERT_DELTA(melt_hourly, melt_daily, glacier_test_constant::EPS);
    // 1 day of melt at 1 deg. C on full glaciated cell without snow
    // must be same as proportionallity parameter; must be integrated from mm/h -> mm/day
    TS_ASSERT_DELTA(melt_daily*24, dtf, glacier_test_constant::EPS)

}

void glacier_melt_test::test_melt_ts(){
    using namespace shyft::core;
    using namespace shyft::timeseries;
    using namespace shyft::time_axis;
    typedef point_ts<fixed_dt> pts_t;
    calendar utc;
    utctime t0= utc.time(2016,5,1);
    utctimespan dt= deltahours(1);
    size_t n=24;
    timeaxis ta(t0,dt,n);
    double dtf = 6.0;
    double glacier_fraction= 0.5;
    double area_m2 = 10 * 1000* 1000;
    pts_t temperature(ta,10.0,fx_policy_t::POINT_AVERAGE_VALUE);
    pts_t sca(ta,0.5,fx_policy_t::POINT_AVERAGE_VALUE);
    for(size_t i=0;i<ta.size();++i)
        sca.set(i,0.5 *(1.0 - double(i)/ta.size()));
    glacier_melt_ts<pts_t> melt(temperature,sca,glacier_fraction,dtf,area_m2);
    for(size_t i=0;i<ta.size();++i) {
        TS_ASSERT_DELTA(melt.value(i),glacier_melt::step(dtf,temperature.value(i),sca.value(i),glacier_fraction)*area_m2*1000.0/3600.0,0.0001);
    }
}
