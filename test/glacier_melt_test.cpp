#include "test_pch.h"
#include "glacier_melt_test.h"
#include "core/glacier_melt.h"
#include "core/timeseries.h"

namespace glacier_test_constant {
    const double EPS = 1.0e-10;
}

using namespace shyft::core;


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
        TS_ASSERT_DELTA(melt.value(i),glacier_melt::step(dtf,temperature.value(i),sca.value(i),glacier_fraction)*area_m2*0.001/3600.0,0.0001);
    }
    // just verify that glacier_melt_ts can participate in ts-operations
    auto a= melt*3.0 + sca; // if this compiles, then it works like we want
    auto b = a*3.0;
}
