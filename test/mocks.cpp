#include "test_pch.h"
#include "mocks.h"

namespace shyfttest {
	using namespace shyft::core;
	using namespace shyft;
	void create_time_series(xpts_t& temp, xpts_t& prec, xpts_t& rel_hum, xpts_t& wind_speed, xpts_t& radiation,
		utctime T0, utctimespan dt, size_t n_points) {
		vector<double> temps; temps.reserve(n_points);
		vector<double> precs; precs.reserve(n_points);
		vector<double> rel_hums; rel_hums.reserve(n_points);
		vector<double> wind_speeds; wind_speeds.reserve(n_points);
		vector<double> rads; rads.reserve(n_points);
		vector<utctime> tpoints; tpoints.reserve(n_points + 1);

		utctime T1 = T0 + n_points * dt;
		for (size_t i = 0; i < n_points; ++i) {
			utctime t = T0 + i * dt;
			temps.emplace_back(-5.0 + 10.0 * std::sin(double((t - T0) / (T1 - T0) * M_PI)));
			precs.emplace_back(i % 3 ? 0.0 : 5.0); // 5mm rain every third time step
			rel_hums.emplace_back(70.0 + 30.0 * sin(double((t - T0) / (T1 - T0) * M_PI)));
			wind_speeds.emplace_back(std::max(0.0, 5.0 * cos(double((t - T0) / (T1 - T0) * 4.0 * M_PI))));
			rads.emplace_back(10.0 + 300 * sin(double((t - T0) / (T1 - T0) * M_PI)));
			tpoints.emplace_back(t);
		}

		tpoints.emplace_back(T1);
		time_axis::point_dt ta(tpoints);
		temp = xpts_t(ta, temps);
		prec = xpts_t(ta, precs);
		rel_hum = xpts_t(ta, rel_hums);
		wind_speed = xpts_t(ta, wind_speeds);
		radiation = xpts_t(ta, rads);
	}

	xpts_t create_time_serie(utctime t0, utctimespan dt, size_t nt) {
		vector<double> vars; vars.reserve(nt);
		vector<utctime> samples; samples.reserve(nt + 1);
		utctime t1 = t0 + nt * dt;
		for (size_t i = 0; i < nt; ++i) {
			utctime t = t0 + i * dt;
			vars.emplace_back(-5.0 + 10.0 * sin(double((t - t0) / (t1 - t0) * M_PI)));
			samples.emplace_back(t);
		}
		samples.emplace_back(t1);
		time_axis::point_dt pta(samples);
		return xpts_t(pta, vars);
	}

	point_ts<time_axis::fixed_dt> create_const_time_serie(const time_axis::fixed_dt& ta, double v) {
		vector<double> vals; vals.reserve(ta.n);
		for (size_t i = 0; i < ta.n; ++i)
			vals.emplace_back(v);
		return point_ts<time_axis::fixed_dt>(ta, vals);
	}
}
