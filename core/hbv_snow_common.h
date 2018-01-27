#pragma once
#include <vector>
namespace shyft {
namespace core {
namespace hbv_snow_common {

using std::vector;

/** \brief integrate function f given as linear interpolated between the f_i(x_i) from a to b for a, b in x.
* If f_rhs_is_zero is set, f(b) = 0, unless b = x[i] for some i in [0,n).
*/
inline double integrate(const vector<double>& f, const vector<double>& x, size_t n, double a, double b, bool f_b_is_zero = false) {
    size_t left = 0;
    double area = 0.0;
    double f_l = 0.0;
    double x_l = a;
    while (a > x[left]) ++left;

    if (fabs(a - x[left]) > 1.0e-8 && left > 0) { // Linear interpolation of start point
        --left;
        f_l = (f[left + 1] - f[left])/(x[left + 1] - x[left])*(a - x[left]) + f[left];
    } else
        f_l = f[left];

    while (left < n - 1) {
        if (b >= x[left + 1]) {
            area += 0.5*(f_l + f[left + 1])*(x[left + 1] - x_l);
            x_l = x[left + 1];
            f_l = f[left + 1];
            ++left;
        } else {
            if (!f_b_is_zero)
                area += (f_l + 0.5*(f[left + 1] - f_l)/(x[left + 1] - x_l)*(b - x_l))*(b - x_l);
            else
                area += 0.5*f_l*(b - x_l);
            break;
        }
    }
    return area;
}

template <class parameter>
void distribute_snow(const parameter& p,vector<double>&sp, vector<double> &sw,double& swe, double &sca) {
    sp = vector<double>(p.intervals.size(), 0.0);
    sw = vector<double>(p.intervals.size(), 0.0);
    if (swe <= 1.0e-3 || sca <= 1.0e-3) {
        swe = sca = 0.0;
    } else {
        for (size_t i = 0; i<p.intervals.size(); ++i)
            sp[i] = sca < p.intervals[i] ? 0.0 : p.s[i] * swe;

        auto temp_swe = integrate(sp, p.intervals, p.intervals.size(), 0.0, sca, true);

        if (temp_swe < swe) {
            const double corr1 = swe/temp_swe*p.lw;
            const double corr2 = swe/temp_swe*(1.0 - p.lw);
            for (size_t i = 0; i<p.intervals.size(); ++i) {
                sw[i] = corr1 * sp[i];
                sp[i] *= corr2;
            }
        } else {
            sw = vector<double>(p.intervals.size(), 0.0);
        }
    }
}

}}}