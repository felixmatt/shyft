#include "core/utctime_utilities.h"
#include "core/timeseries.h"
#include "core/utctime_utilities.h"
#include "core/geo_cell_data.h"
#include <armadillo>

namespace shyfttest {

	using namespace shyft::core;
	using namespace shyft::timeseries;
	using namespace std::chrono;

	typedef std::vector<point> point_vector_t;
	typedef point_ts<point_timeaxis> xpts_t;

	void create_time_series(xpts_t& temp,
		xpts_t& prec,
		xpts_t& rel_hum,
		xpts_t& wind_speed,
		xpts_t& radiation,
		utctime T0,
		utctimespan dt,
		size_t n_points);

	template<typename TimeT = milliseconds>
	struct measure
	{
		template<typename F, typename ...Args>
		static typename TimeT::rep execution(F func, Args&&... args)
		{
			auto start = system_clock::now();
			func(std::forward<Args>(args)...);
			auto duration = duration_cast<TimeT>(system_clock::now() - start);
			return duration.count();
		}
	};

	namespace mock
	{
		struct PTGSKResponseCollector
		{
			std::vector<shyft::timeseries::point> evap;
			std::vector<shyft::timeseries::point> snow_storage;
			std::vector<shyft::timeseries::point> avg_discharge;

			PTGSKResponseCollector(size_t n_pts)
			{
				evap.reserve(n_pts);
				snow_storage.reserve(n_pts);
				avg_discharge.reserve(n_pts);
			}

			template<class R>
			void collect(const shyft::core::utctime time, const R& response)
			{
				evap.emplace_back(time, response.pt.pot_evapotranspiration);
				snow_storage.emplace_back(time, response.gs.storage);
				avg_discharge.emplace_back(time, response.kirchner.q_avg);
			}

			template<class R>
			void set_end_response(const R& r) {}
		};

		template<class T>
		struct StateCollector
		{
			// Collected from the response, to better understand the model
			shyft::timeseries::point_ts<T> _inst_discharge; // Kirchner instant Discharge in m^3/s

			StateCollector() {}

			StateCollector(const T& time_axis) : _inst_discharge(time_axis, 0.0) {}

			void initialize(const T& time_axis)
			{
				_inst_discharge = shyft::timeseries::point_ts<T>(time_axis, 0.0);
			}

			template<class S>
			void collect(size_t idx, const S& state)
			{
				_inst_discharge.set(idx, state.kirchner.q);
			}

			const shyft::timeseries::point_ts<T>& kirchner_state() const
			{
				return _inst_discharge;
			}
		};

		template<class T> class TSPointTarget
		{
		private:
			std::vector<shyft::timeseries::point> _values;

		public:
			TSPointTarget() {}

			TSPointTarget(const T& time_axis)
			{
				_values.reserve(time_axis.size());
				for (size_t i = 0; i < time_axis.size(); ++i)
					_values.emplace_back(time_axis.period(i).end, 0.0);
			}

			shyft::timeseries::point& value(size_t idx)
			{
				return _values[idx];
			}

			size_t size() const
			{
				return _values.size();
			}
		};

		template<class T>
		struct DischargeCollector
		{
			double destination_area;
			shyft::timeseries::point_ts<T> avg_discharge; // Discharge in m^3

			DischargeCollector() : destination_area(0.0) {}
			DischargeCollector(const double destination_area) : destination_area(destination_area) {}
			DischargeCollector(const double destination_area, const T& time_axis)
				: destination_area(destination_area), avg_discharge(time_axis, 0.0) {}

			void initialize(const T& time_axis)
			{
				avg_discharge = shyft::timeseries::point_ts<T>(time_axis, 0.0);
			}

			template<class R>
			void collect(size_t idx, const R& resp)
			{
				avg_discharge.set(idx, destination_area * resp.kirchner.q_avg / 1000.0 / 3600.0); // q_avg in mm
			}

			template<class R>
			void set_end_response(const R& response) {}

			const shyft::timeseries::point_ts<T>& discharge() const
			{
				return avg_discharge;
			}
		};

		template<class T>
		struct ResponseCollector
		{
			double destination_area; // in m^2

			double mmh_to_m3s(double mm_pr_hour) const
			{
				const double mmh_to_m3s_scale_factor = 1.0 / 3600.0 / 1000.0;
				return destination_area * mm_pr_hour * mmh_to_m3s_scale_factor;
			}

			// Collected from the response, to better understand the model
			shyft::timeseries::point_ts<T> avg_discharge; // Kirchner Discharge given in m^3/s
			shyft::timeseries::point_ts<T> _snow_sca; // gamma snow, sca..
			shyft::timeseries::point_ts<T> _snow_swe; // gamma snow swe, mm
			shyft::timeseries::point_ts<T> _snow_output; // snow output in m^3/s
			shyft::timeseries::point_ts<T> _total_stored_snow; // skaugen's stored water in m^3
			shyft::timeseries::point_ts<T> _ae_output; // actual evap mm/h
			shyft::timeseries::point_ts<T> _pe_output; // actual evap mm/h

			ResponseCollector() : destination_area(0.0) {}

			ResponseCollector(double destination_area) : destination_area(destination_area) {}

			ResponseCollector(double destination_area, const T& time_axis) : destination_area(destination_area),
				avg_discharge(time_axis, 0.0),
				_snow_sca(time_axis, 0.0),
				_snow_swe(time_axis, 0.0),
				_snow_output(time_axis, 0.0),
				_ae_output(time_axis, 0.0),
				_pe_output(time_axis, 0.0) {}

			void initialize(const T& time_axis)
			{
				avg_discharge = shyft::timeseries::point_ts<T>(time_axis, 0.0);
				_snow_sca = shyft::timeseries::point_ts<T>(time_axis, 0.0);
				_snow_swe = shyft::timeseries::point_ts<T>(time_axis, 0.0);
				_snow_output = shyft::timeseries::point_ts<T>(time_axis, 0.0);
				_total_stored_snow = shyft::timeseries::point_ts<T>(time_axis, 0.0);
				_ae_output = shyft::timeseries::point_ts<T>(time_axis, 0.0);
				_pe_output = shyft::timeseries::point_ts<T>(time_axis, 0.0);
			}

			template<class R>
			void collect(size_t idx, const R& response)
			{
				avg_discharge.set(idx, mmh_to_m3s(response.total_discharge)); // want m3/s, q_avg is in mm/h, so compute the totals in mm/s
				_snow_sca.set(idx, response.gs.sca);
				_snow_output.set(idx, response.gs.outflow); // current mm/h, but want m3/s, but we get mm/h from snow output
				_snow_swe.set(idx, response.gs.storage);
				_ae_output.set(idx, response.ae.ae);
				_pe_output.set(idx, response.pt.pot_evapotranspiration);
			}

			template<class R>
			void set_end_response(const R& r) {}

			const shyft::timeseries::point_ts<T>& discharge() const { return avg_discharge; }
			const shyft::timeseries::point_ts<T>& snow_sca() const { return _snow_sca; }
			const shyft::timeseries::point_ts<T>& snow_swe() const { return _snow_swe; }
			const shyft::timeseries::point_ts<T>& snow_output() const { return _snow_output; }
			const shyft::timeseries::point_ts<T>& ae_output() const { return _ae_output; }
			const shyft::timeseries::point_ts<T>& pe_output() const { return _pe_output; }
		};

		template<class R, class S, class P, class TS> class MCell
		{
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
				_parameter(parameter)
			{
				geo_cell_data_.set_catchment_id(catchment_number);
			}

			state_t& get_state(shyft::core::utctime time)
			{
				return _state; // TODO Make sure the input time corresponds to the actual time stamp of the state vec!!!
			}

			void set_parameter(parameter_t& p) { _parameter = p; }
			void set_state(state_t& s) { _state = s; }
			const parameter_t& parameter() const { return _parameter; }
			size_t catchment_number() const { return geo_cell_data_.catchment_id(); }

			template<class SRC, class D, class T>
			void add_discharge(const SRC& discharge_source,
				D& discharge_target,
				const T& time_axis) const
			{
				auto discharge_accessor = shyft::timeseries::average_accessor<SRC, T>(discharge_source, time_axis);
				for (size_t i = 0; i < time_axis.size(); ++i)
					discharge_target.add(i, discharge_accessor.value(i));
			}
		};

		template<typename P>
		class GRFDestination
		{
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

			void set_weights(const arma::vec& weights, const arma::uvec& indices)
			{
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

}; // End namespace shyfttest
