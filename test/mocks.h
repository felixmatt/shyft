#pragma once
#include "core/utctime_utilities.h"
#include "core/time_series.h"
#include "core/inverse_distance.h"
#include "core/utctime_utilities.h"
#include "core/geo_cell_data.h"
#include "core/geo_point.h"

namespace ta = shyft::time_axis;
namespace shyfttest {
	using namespace shyft::core;
	using namespace shyft::time_series;
	using namespace std::chrono;

	typedef std::vector<point> point_vector_t;
	typedef point_ts<ta::point_dt> xpts_t;

	void create_time_series(xpts_t& temp, xpts_t& prec, xpts_t& rel_hum, xpts_t& wind_speed, xpts_t& radiation, utctime T0, utctimespan dt, size_t n_points);
	xpts_t create_time_serie(utctime t0, utctimespan dt, size_t nt);
	point_ts<ta::fixed_dt> create_const_time_serie(const ta::fixed_dt& ta, double v);

	template<typename TimeT = milliseconds>
	struct measure {
		template<typename F, typename ...Args>
		static typename TimeT::rep execution(F func, Args&&... args) {
			auto start = system_clock::now();
			func(std::forward<Args>(args)...);
			auto duration = duration_cast<TimeT>(system_clock::now() - start);
			return duration.count();
		}
	};

	namespace mock {


		struct PTGSKResponseCollector {
			std::vector<shyft::time_series::point> evap;
			std::vector<shyft::time_series::point> snow_storage;
			std::vector<shyft::time_series::point> avg_discharge;

			PTGSKResponseCollector(size_t n_pts) {
				evap.reserve(n_pts);
				snow_storage.reserve(n_pts);
				avg_discharge.reserve(n_pts);
			}

			template<class R>
			void collect(const shyft::core::utctime time, const R& response) {
				evap.emplace_back(time, response.pt.pot_evapotranspiration);
				snow_storage.emplace_back(time, response.gs.storage);
				avg_discharge.emplace_back(time, response.kirchner.q_avg);
			}

			template<class R>
			void set_end_response(const R& r) {}
		};

		struct PTUSKResponseCollector {
			std::vector<shyft::time_series::point> evap;
			std::vector<shyft::time_series::point> snow_storage;
			std::vector<shyft::time_series::point> avg_discharge;

			PTUSKResponseCollector(size_t n_pts) {
				evap.reserve(n_pts);
				snow_storage.reserve(n_pts);
				avg_discharge.reserve(n_pts);
			}

			template<class R>
			void collect(const shyft::core::utctime time, const R& response) {
				evap.emplace_back(time, response.pt.pot_evapotranspiration);
				snow_storage.emplace_back(time, response.us.storage);
				avg_discharge.emplace_back(time, response.kirchner.q_avg);
			}

			template<class R>
			void set_end_response(const R& r) {}
		};
		template<class T>
		struct StateCollector {
			// Collected from the response, to better understand the model
			shyft::time_series::point_ts<T> _inst_discharge; // Kirchner instant Discharge in m^3/s

			StateCollector() {}

			StateCollector(const T& time_axis) : _inst_discharge(time_axis, 0.0) {}

			void initialize(const T& time_axis) {
				_inst_discharge = shyft::time_series::point_ts<T>(time_axis, 0.0);
			}

			template<class S>
			void collect(size_t idx, const S& state) {
				// a lot of other states in hbv_
				_inst_discharge.set(idx, state.kirchner.q);
			}

			const shyft::time_series::point_ts<T>& kirchner_state() const {
				return _inst_discharge;
			}
		};

		template<class T>
		class TSPointTarget {
		private:
			std::vector<shyft::time_series::point> _values;

		public:
			TSPointTarget() {}

			TSPointTarget(const T& time_axis) {
				_values.reserve(time_axis.size());
				for (size_t i = 0; i < time_axis.size(); ++i)
					_values.emplace_back(time_axis.period(i).end, 0.0);
			}

			shyft::time_series::point& value(size_t idx) {
				return _values[idx];
			}

			size_t size() const {
				return _values.size();
			}
		};

		template<class T>
		struct DischargeCollector {
			double destination_area;
			shyft::time_series::point_ts<T> avg_discharge; // Discharge in m^3

			DischargeCollector() : destination_area(0.0) {}
			DischargeCollector(const double destination_area) : destination_area(destination_area) {}
			DischargeCollector(const double destination_area, const T& time_axis)
				: destination_area(destination_area), avg_discharge(time_axis, 0.0) {}

			void initialize(const T& time_axis) {
				avg_discharge = shyft::time_series::point_ts<T>(time_axis, 0.0);
			}

			template<class R>
			void collect(size_t idx, const R& resp) {
				avg_discharge.set(idx, destination_area * resp.kirchner.q_avg / 1000.0 / 3600.0); // q_avg in mm
			}

			template<class R>
			void set_end_response(const R& response) {}

			const shyft::time_series::point_ts<T>& discharge() const {
				return avg_discharge;
			}
		};

		template<class T>
		struct ResponseCollector {
			double destination_area; // in m^2

			double mmh_to_m3s(double mm_pr_hour) const {
				const double mmh_to_m3s_scale_factor = 1.0 / 3600.0 / 1000.0;
				return destination_area * mm_pr_hour * mmh_to_m3s_scale_factor;
			}

			// Collected from the response, to better understand the model
			shyft::time_series::point_ts<T> avg_discharge; // Kirchner Discharge given in m^3/s
			shyft::time_series::point_ts<T> _snow_sca; // gamma snow, sca..
			shyft::time_series::point_ts<T> _snow_swe; // gamma snow swe, mm
			shyft::time_series::point_ts<T> _snow_output; // snow output in m^3/s
			shyft::time_series::point_ts<T> _total_stored_snow; // skaugen's stored water in m^3
			shyft::time_series::point_ts<T> _ae_output; // actual evap mm/h
			shyft::time_series::point_ts<T> _pe_output; // actual evap mm/h

			ResponseCollector() : destination_area(0.0) {}

			ResponseCollector(double destination_area) : destination_area(destination_area) {}

			ResponseCollector(double destination_area, const T& time_axis) : destination_area(destination_area),
				avg_discharge(time_axis, 0.0),
				_snow_sca(time_axis, 0.0),
				_snow_swe(time_axis, 0.0),
				_snow_output(time_axis, 0.0),
				_ae_output(time_axis, 0.0),
				_pe_output(time_axis, 0.0) {}

			void initialize(const T& time_axis) {
				avg_discharge = shyft::time_series::point_ts<T>(time_axis, 0.0);
				_snow_sca = shyft::time_series::point_ts<T>(time_axis, 0.0);
				_snow_swe = shyft::time_series::point_ts<T>(time_axis, 0.0);
				_snow_output = shyft::time_series::point_ts<T>(time_axis, 0.0);
				_total_stored_snow = shyft::time_series::point_ts<T>(time_axis, 0.0);
				_ae_output = shyft::time_series::point_ts<T>(time_axis, 0.0);
				_pe_output = shyft::time_series::point_ts<T>(time_axis, 0.0);
			}

			template<class R>
			void collect(size_t idx, const R& response) {
				avg_discharge.set(idx, mmh_to_m3s(response.total_discharge)); // want m3/s, q_avg is in mm/h, so compute the totals in mm/s
				_snow_sca.set(idx, response.gs.sca);
				_snow_output.set(idx, response.gs.outflow); // current mm/h, but want m3/s, but we get mm/h from snow output
				_snow_swe.set(idx, response.gs.storage);
				_ae_output.set(idx, response.ae.ae);
				_pe_output.set(idx, response.pt.pot_evapotranspiration);
			}

			template<class R>
			void set_end_response(const R& r) {}

			const shyft::time_series::point_ts<T>& discharge() const { return avg_discharge; }
			const shyft::time_series::point_ts<T>& snow_sca() const { return _snow_sca; }
			const shyft::time_series::point_ts<T>& snow_swe() const { return _snow_swe; }
			const shyft::time_series::point_ts<T>& snow_output() const { return _snow_output; }
			const shyft::time_series::point_ts<T>& ae_output() const { return _ae_output; }
			const shyft::time_series::point_ts<T>& pe_output() const { return _pe_output; }
		};

		template<class R, class S, class P, class TS>
		class MCell {
		public:
			typedef R response_t;
			typedef S state_t;
			typedef TS time_series_t;
			typedef P parameter_t;

		private:
			time_series_t _temperature;
			time_series_t _precipitation;
			time_series_t _wind_speed;
			time_series_t _rel_hum;
			time_series_t _radiation;

			state_t _state;
			response_t _response;
			parameter_t _parameter;
			geo_cell_data geo_cell_data_;

		public:
			const time_series_t& temperature() const { return _temperature; }
			const time_series_t& precipitation() const { return _precipitation; }
			const time_series_t& wind_speed() const { return _wind_speed; }
			const time_series_t& rel_hum() const { return _rel_hum; }
			const time_series_t& radiation() const { return _radiation; }
			const geo_cell_data& geo_cell_info() const { return geo_cell_data_; }

			MCell(const time_series_t& temp,
				const time_series_t& prec,
				const time_series_t& ws,
				const time_series_t& rh,
				const time_series_t& rad,
				const state_t& state,
				const parameter_t& parameter,
				size_t catchment_number) :
				_temperature(temp),
				_precipitation(prec),
				_wind_speed(ws),
				_rel_hum(rh),
				_radiation(rad),
				_state(state),
				_parameter(parameter) {
				geo_cell_data_.set_catchment_id(catchment_number);
			}

			state_t& get_state(shyft::core::utctime time) {
				return _state; // TODO Make sure the input time corresponds to the actual time stamp of the state vec!!!
			}

			void set_parameter(parameter_t& p) { _parameter = p; }
			void set_state(state_t& s) { _state = s; }
			const parameter_t& parameter() const { return _parameter; }
			size_t catchment_number() const { return geo_cell_data_.catchment_id(); }

			template<class SRC, class D, class T>
			void add_discharge(const SRC& discharge_source,
				D& discharge_target,
				const T& time_axis) const {
				auto discharge_accessor = shyft::time_series::average_accessor<SRC, T>(discharge_source, time_axis);
				for (size_t i = 0; i < time_axis.size(); ++i)
					discharge_target.add(i, discharge_accessor.value(i));
			}
		};

		template<typename P>
		class GRFDestination {
		private:
			const P _geo_point;
			arma::vec _source_anisotropic_distances; // TODO: Not sure we need this, please check when method is fully understood
			arma::vec _source_covariances;
			arma::uvec _source_sort_order; // Changes when the anisotropic distances change
			arma::vec _weights;
			arma::uvec _weight_indices;

		public:
			GRFDestination(const P& geo_point) : _geo_point(geo_point) {}

			void set_source_anisotropic_distances(const arma::mat& anisotopic_distances) { _source_anisotropic_distances = anisotopic_distances; }
			void set_source_covariances(const arma::vec& source_covariances) { _source_covariances = source_covariances; }
			void set_source_sort_order(const arma::uvec& source_sort_order) { _source_sort_order = source_sort_order; }

			void set_weights(const arma::vec& weights, const arma::uvec& indices) {
				_weights = weights;
				_weight_indices = indices;
			}

			const arma::uvec& source_sort_order() const { return _source_sort_order; }
			const arma::vec& source_covariances() const { return _source_covariances; }
			const arma::vec& weights() const { return _weights; }
			const arma::uvec& weight_indices() const { return _weight_indices; }
			const P& geo_point() const { return _geo_point; }
		};

	}; // End namespace mock

	namespace idw {

		const double TEST_EPS = 0.00000001;

		struct Source {
			typedef geo_point geo_point_t; // Why is it declared here?

			geo_point point;
			double v;
			utctime t_special;
			double v_special;
			mutable int get_count;

			Source(geo_point p, double v) : point(p), v(v), t_special(0), v_special(v), get_count(0) {}

			geo_point mid_point() const { return point; }

			double value(utcperiod p) const { return value(p.start); }

			double value(utctime t) const {
				get_count++;
				return t == t_special ? v_special : v;
			}

			// For testing
			void set_value(double vx) { v = vx; }

			void set_value_at_t(utctime tx, double vx) { t_special = tx; v_special = vx; }

			static vector<Source> GenerateTestSources(const ta::fixed_dt& time_axis, size_t n, double x, double y, double radius) {
				vector<Source> r;
				r.reserve(n);
				const double pi = 3.1415;
				double delta = 2.0 * pi / n;
				for (double angle = 0; angle < 2 * pi; angle += delta) {
					double xa = x + radius * sin(angle);
					double ya = y + radius * cos(angle);
					double za = (xa + ya) / 1000.0;
					r.emplace_back(geo_point(xa, ya, za), 10.0 + za * -0.006); // reasonable temperature, dependent on height
				}
				return r;
			}

			static vector<Source> GenerateTestSourceGrid(const ta::fixed_dt& time_axis, size_t nx, size_t ny, double x, double y, double dxy) {
				vector<Source> r;
				r.reserve(nx * ny);
				const double max_dxy = dxy * (nx + ny);
				for (size_t i = 0; i < nx; ++i) {
					double xa = x + i * dxy;
					for (size_t j = 0; j < ny; ++j) {
						double ya = y + j * dxy;
						double za = 1000.0 * (xa + ya) / max_dxy;
						r.emplace_back(geo_point(xa, ya, za), 10.0 + za * -0.006); // reasonable temperature, dependent on height
					}
				}
				return r;
			}
		};

		struct PointTimeSerieSource {
			// Interface needed for run_interpolation<>
			typedef geo_point geo_point_t;
			geo_point mid_point() const { return gp; }
			double value(utcperiod p) const { return pts(p.start); }
			double value(size_t i) const { return pts.value(i); }
			void set_value(size_t i, double v) { pts.set(i, v); }

			geo_point gp;
			point_ts<ta::fixed_dt> pts;

			PointTimeSerieSource(geo_point gp, const point_ts<ta::fixed_dt>& ts) : gp(gp), pts(ts) {}
			void SetTs(const point_ts<ta::fixed_dt>& ts) { pts = ts; }

			static vector<PointTimeSerieSource> make_source_set(const ta::fixed_dt& ta, size_t nx, size_t ny) {
				vector<PointTimeSerieSource> v;
				v.reserve(nx * ny);
				auto pts = point_ts<ta::fixed_dt>(ta, 0);
				for (size_t x = 0; x < nx; x++)
					for (size_t y = 0; y < ny; y++)
						v.emplace_back(geo_point(x * 2500, y * 2500, 1000), pts);
				return v;
			}
		};

		struct MCell {
			geo_point point;
			double v;
			int set_count;
			double slope;

			MCell() : point(), v(-1.0), set_count(0), slope(1.0) {}

			MCell(geo_point p) : point(p), v(-1.0), set_count(0), slope(1.0) {}

			geo_point mid_point() const { return point; }
			double slope_factor() const { return 1.0; }
			void set_slope_factor(double x) { slope = x; }
			void set_value(size_t t, double vt) {
				set_count++;
				v = vt;
			}

			static vector<MCell> GenerateTestGrid(size_t nx, size_t ny) {
				vector<MCell> r;
				r.reserve(nx * ny);
				const double z_min = 100.0;
				const double z_max = 800.0;
				const double dz = (z_max - z_min) / (nx + ny);
				for (size_t x = 0; x < nx; ++x)
					for (size_t y = 0; y < ny; ++y)
						r.emplace_back(geo_point(500.0 + x * 1000, 500.0 + y * 1000, z_min + (x + y) * dz));
				return r;
			}
		};

		struct PointTimeSerieCell {
			// Interface needed for run_interpolation<>
			typedef geo_point geo_point_t;
			geo_point mid_point() const { return gp; }
			double value(utcperiod p) const { return pts(p.start); }
			double value(size_t i) const { return pts.value(i); }
			void set_value(size_t i, double v) { pts.set(i, v); }

			geo_point gp;
			point_ts<ta::fixed_dt> pts;

			PointTimeSerieCell(geo_point gp, const point_ts<ta::fixed_dt>& ts) : gp(gp), pts(ts) {}
			void SetTs(const point_ts<ta::fixed_dt>& ts) { pts = ts; }

			static vector<PointTimeSerieCell> make_cell_grid(const ta::fixed_dt& ta, size_t nx, size_t ny) {
				vector<PointTimeSerieCell> v;
				v.reserve(nx * ny);
				auto pts = point_ts<ta::fixed_dt>(ta, 0);
				for (size_t x = 0; x < nx; ++x)
					for (size_t y = 0; y < ny; ++y)
						v.emplace_back(geo_point(x * 1000, y * 1000, 1000), pts);
				return v;
			}
		};

		struct Parameter {
			double max_distance;
			size_t max_members;

			Parameter(double max_distance = 200000, size_t max_neigbours = 20) : max_distance(max_distance), max_members(max_neigbours) {}

			bool gradient_by_equation = false; // just use min/max for existing tests (bw compatible)
			double default_gradient() const { return -0.006; }  // C/m decrease 0.6 degC/100m
			double precipitation_scale_factor() const { return 1.0 + 2.0 / 100.0; } // 2 pct /100m
			double distance_measure_factor = 2.0; // Square distance
			double zscale = 1.0;
		};

		using namespace shyft::core::inverse_distance;
		typedef temperature_model<Source, MCell, Parameter, geo_point, temperature_gradient_scale_computer> TestTemperatureModel;
		typedef temperature_model<PointTimeSerieSource, PointTimeSerieCell, Parameter, geo_point, temperature_gradient_scale_computer> TestTemperatureModel_1;
		typedef temperature_model<PointTimeSerieCell, PointTimeSerieSource, Parameter, geo_point, temperature_gradient_scale_computer> TestTemperatureModel_2;
		typedef radiation_model<Source, MCell, Parameter, geo_point> TestRadiationModel;
		typedef precipitation_model<Source, MCell, Parameter, geo_point> TestPrecipitationModel;

	}; // End namespace idw

}; // End namespace shyfttest
