#pragma once
namespace shyft {

    constexpr double mmh_to_m3s_scale_factor = 1 / (3600.0*1000.0);

    /** \brief convert [mm/h] over an area_m^2 to [m^3/s] units */
    inline double mmh_to_m3s(double mm_pr_hour, double area_m2) {
        return area_m2*mm_pr_hour*mmh_to_m3s_scale_factor;
    }
    /** \brief convert [m^3/s] to [mm/h] over area_m^2  */
    inline double m3s_to_mmh(double m3s, double area_m2) {
        return m3s / (mmh_to_m3s_scale_factor*area_m2);
    }
}
