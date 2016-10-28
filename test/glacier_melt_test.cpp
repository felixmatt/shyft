#include "test_pch.h"
#include "glacier_melt_test.h"
#include "core/glacier_melt.h"

namespace glacier_test_constant {
    const double EPS = 1.0e-10;
}

using namespace shyft::core;

void glacier_melt_test::test_melt() {
    // Model parameters
    const double dtf = 6.0;
    glacier_melt::parameter p(dtf);

    // Glacier melt under various conditions
	utctimespan dt_h = deltahours(1); // hourly

	double sca = 0.0;
	while(sca <= 1.0)
    {
        double gf = 0.0;
        while(gf <= 1.0)
        {
            double temp = -10;
            while(temp <= 10)
            {

                double melt = glacier_melt::step(dt_h,p.dtf, temp, sca, gf);
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
    utctimespan dt_d = deltahours(24); // daily

    const double melt_hourly =  glacier_melt::step(dt_h,p.dtf, temp, sca, gf);
    const double melt_daily =  glacier_melt::step(dt_d,p.dtf, temp, sca, gf);


    TS_ASSERT(melt_hourly > 0.0);
    TS_ASSERT(melt_daily > 0.0);
    TS_ASSERT_DELTA(melt_hourly, melt_daily, glacier_test_constant::EPS);
    // 1 day of melt at 1 deg. C on full glaciated cell without snow
    // must be same as proportionallity parameter; must be integrated from mm/h -> mm/day
    TS_ASSERT_DELTA(melt_daily*24, dtf, glacier_test_constant::EPS)

}
