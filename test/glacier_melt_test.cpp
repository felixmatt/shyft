#include "test_pch.h"
#include "core/glacier_melt.h"
#include "core/time_series.h"

namespace glacier_test_constant {
    const double EPS = 1.0e-10;
}

using namespace shyft::core;

TEST_SUITE("glacier_melt") {
TEST_CASE("test_melt") {
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
                if ((temp > 0.0) && (gf > sca))
                    TS_ASSERT(melt > 0.0);
                temp += 2.0;
            }
            gf += 0.2;
        }
        sca += 0.2;
    }

    // Verify one step manually:
    double temp = 1.0;
    sca = 0.0;
    double gf = 1.0;

    const double melt_m3s =  glacier_melt::step(p.dtf, temp, sca, gf);
    TS_ASSERT_DELTA(melt_m3s, dtf*(gf - sca)*temp*0.001 / 86400.0, glacier_test_constant::EPS);

}

TEST_CASE("test_melt_ts") {
#if 0
    using namespace shyft::core;
    using namespace shyft::time_series;
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
    pts_t temperature(ta,10.0,ts_point_fx::POINT_AVERAGE_VALUE);
    pts_t sca_m2(ta,0.5*area_m2,ts_point_fx::POINT_AVERAGE_VALUE);
    for(size_t i=0;i<ta.size();++i)
        sca_m2.set(i,area_m2*0.5 *(1.0 - double(i)/ta.size()));
    glacier_melt_ts<pts_t> melt(temperature,sca_m2,glacier_fraction*area_m2,dtf);
    for(size_t i=0;i<ta.size();++i) {
        TS_ASSERT_DELTA(melt.value(i),glacier_melt::step(dtf,temperature.value(i),sca_m2.value(i),area_m2*glacier_fraction),0.0001);
    }
    // just verify that glacier_melt_ts can participate in ts-operations
    auto a= melt*3.0 + sca_m2; // if this compiles, then it works like we want
    auto b = a*3.0;
#endif
}
}
