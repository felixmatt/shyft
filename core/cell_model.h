#pragma once


#include "time_series.h"
#include "geo_cell_data.h"

namespace shyft {
    namespace core {
		using namespace shyft::time_series;
		using namespace shyft;
		// and typedefs for commonly used types in the model
		typedef point_ts<time_axis::fixed_dt> pts_t;
		typedef constant_timeseries<time_axis::fixed_dt> cts_t;
		typedef time_axis::fixed_dt timeaxis_t;


		// cell-model goes here

		/** \brief environment supplies the temperature, precipitation,radiation,rel_hum and  wind_speed
		* time-series representative for the cell area,height and mid_point
		* Each property can have its own ts-type, so that we can easily represent constants if needed.
		* We use this template class as a building block of the cell-types, since most cells need this
		* information.
		* \see cell
		* \tparam timeaxis type, the ts-type should all be constructible/resetable with a timeaxis and a fill-value.
		*
		* \tparam temperature_ts type for the temperature ts, usually a shyft::time_series::points_ts<TA> type
		* \tparam precipitation_ts
		* \tparam radiation_ts
		* \tparam relhum_ts
		* \tparam windspeed_ts
		*/
		template<class timeaxis, class temperature_ts, class precipitation_ts, class radiation_ts, class relhum_ts, class windspeed_ts>
		struct environment {
			typedef timeaxis timeaxis_t;//< need to remember and expose this type
			typedef temperature_ts temperature_ts_t;
			temperature_ts   temperature;
			precipitation_ts precipitation;
			radiation_ts     radiation;
			relhum_ts        rel_hum;
			windspeed_ts     wind_speed;

			//< reset/initializes all series with nan and reserves size to ta. size.
			void init(const timeaxis& ta) {
                if (ta != temperature.ta) {
                    temperature = temperature_ts(ta, shyft::nan);
                    precipitation = precipitation_ts(ta, shyft::nan);
                    rel_hum = relhum_ts(ta, shyft::nan);
                    radiation = radiation_ts(ta, shyft::nan);
                    wind_speed = windspeed_ts(ta, shyft::nan);
                } else {
                    temperature.fill(shyft::nan);
                    precipitation.fill(shyft::nan);
                    rel_hum.fill( shyft::nan);
                    radiation.fill(shyft::nan);
                    wind_speed.fill(shyft::nan);
                }
			}
		};

		///< environment variant with const relative humidity and wind (speed up calc &reduce mem)
		typedef environment<timeaxis_t, pts_t, pts_t, pts_t, cts_t, cts_t> environment_const_rhum_and_wind_t;
		///< environment type with all properties as general time_series
		typedef environment<timeaxis_t, pts_t, pts_t, pts_t, pts_t, pts_t> environment_t;

		///< utility function to create an instance of a environment based on function (auto-template by arguments)
		template<class timeaxis, class temperature_ts, class precipitation_ts, class radiation_ts, class relhum_ts, class windspeed_ts>
		inline environment<timeaxis, temperature_ts, precipitation_ts, radiation_ts, relhum_ts, windspeed_ts>
			create_cell_environment(temperature_ts temp, precipitation_ts prec, radiation_ts rad, relhum_ts rhum, windspeed_ts ws) {
			return environment<timeaxis, temperature_ts, precipitation_ts, radiation_ts, relhum_ts, windspeed_ts>{temp, prec, rad, rhum, ws};
		}

		/** \brief cell template for distributed cell calculations.
		* a cell have :
		* -# geo        information (.mid_point(),area(), landtype fractions etc..)
		* -# parameters reference to parameters for the cell method-stack (subject to calibration)
		* -# state      the current state of the cell, like water-content etc.
		* -# env_ts     provision the environmental properties (prec,temp,rad,wind,rhum) at the cell.mid_point
		* -# sc         state collector, plays its role during run(), collects wanted state each step
		* -# rc         response collector,plays its role during run(), collects wanted response each step
		*
		* \tparam P  parameter type  for the cell method stack, e.g. \ref shyft::core::pt_gs_k::parameter
		* \tparam E  environment type for the cell
		* \tparam S  state type for the cell
		* \tparam SC state collector type for the cell (usually a a null collector)
		* \tparam RC response collector type for the cell, usually we look for discharge, but could be more extensive collector
		*
		*/

		template< class P, class E, class S, class SC, class RC>
		struct cell {
			typedef P parameter_t;///< export the parameter type so we can use it elsewhere
			typedef E env_ts_t;   ///< export the env_ts type so we can use it elsewhere
			typedef S state_t;    ///< export the state type ..
			typedef RC response_collector_t;
            typedef typename E::timeaxis_t timeaxis_t;///<export the timeaxis_t
			// these are the type of data the cell keeps:
			geo_cell_data geo;///< all needed (static) geo-related information goes here
			std::shared_ptr<P>  parameter;///< usually one cell shares a common set of parameters, stored elsewhere
			S state;     ///< method stack dependent
			E env_ts;    ///< environment ts inputs, precipitation, temperature etc.

			SC sc;   ///<method stack and context dependent class (we dont collect state during optimization)
			RC rc;   ///<method stack and context dependent (full response, or just minimal for optimization)

            // boost python workaround for shared ptr properties
            std::shared_ptr<P> get_parameter() {return parameter;}


			///< support to initialize env. input series, before interpolation step
			void   init_env_ts(const typename E::timeaxis_t &ta) { env_ts.init(ta); }
			///< common support during run, first step is to call begin_run, typically invoked from run() in templates, calls initialize on collectors
			void begin_run(const timeaxis_t &ta, int start_step, int n_steps) {
                rc.initialize(ta,start_step,n_steps, geo.area());
                sc.initialize(timeaxis_t(ta.start(),ta.delta(),ta.size()+1),start_step,n_steps>0?n_steps+1:0,geo.area());//state collection is both first and last state, so, time-axis +1 step
			}
			///< support for the interpolation phase, executes before the cell-run
			void   set_temperature(size_t ix, double temperature_value) { env_ts.temperature.set(ix, temperature_value); }
			///< used by the radiation interpolation to adjust the local radiation with respect to idw sources.
			double slope_factor() const { return geo.radiation_slope_factor(); }
			///< used by the interpolation routines
			const  geo_point& mid_point() const { return geo.mid_point(); }

			///< set the cell method stack parameters, typical operations at region_level, executed after the interpolation, before the run
			void   set_parameter(const std::shared_ptr<P>& p) { parameter = p; }
			///< set the cell state
			void   set_state(const S& s) { state = s; }
			///< collecting the state during run could be very useful to understand models
			void set_state_collection(bool on) {}
			///< collecting the snow sca and swe on for calibration scenarios, default throws
			void set_snow_sca_swe_collection(bool on) {/*default simply ignore*/}
			/// run the cell method stack for  a specified time-axis, to be specialized by cell type
			void run(const timeaxis_t& t, int start_step, int n_steps) {}
			///< operator equal if same midpoint and catchment-id
			bool operator==(const cell&x) const {
			    return geo.mid_point()==x.geo.mid_point()&& geo.catchment_id()==x.geo.catchment_id();
			}
		};
        /**Utility function used to  initialize a pts_t in the core, typically making space, fill a ts
        *  prior to a run to ensure values are zero */
        inline void ts_init(pts_t&ts, time_axis::fixed_dt const& ta, int start_step, int n_steps, ts_point_fx fx_policy) {
            double const fill_value=shyft::nan;
            if (ts.ta != ta || ta.size()==0 ) {
                ts = pts_t(ta, fill_value, fx_policy);
            } else {
                ts.fill_range(fill_value, start_step, n_steps);
            }
        }

        using namespace std;
        /** \brief cell statistics provides ts feature summation over cells
         *
         * Since the cells are different, like different features, based on
         * method stack, we use template to do the "main work", either to
         * do a scaled average, e.g. temperature, or plain sum (cell run_off in m3/s)
         * Using this technique, there is only one line pr. feature sum/average needed
         * -and we can keep the kloc down to a minimum.
         *
         * Using swig/python, we need a helper class in either _api.h, or api.i
         *  to provide the functions to python layer
         *
         */
        struct cell_statistics {

			/**throws runtime_error if catchment_indexes contains cid that's not part of cells */
			template<typename cell>
			static void verify_cids_exist(const vector<cell>& cells, const vector<int>& catchment_indexes) {
				if (catchment_indexes.size() == 0) return;
				map<int, bool> all_cids;
				for (const auto&c : cells) all_cids[c.geo.catchment_id()] = true;
				for(auto cid:catchment_indexes)
					if (all_cids.count(cid) == 0)
						throw runtime_error(string("one or more supplied catchment_indexes does not exist:") + to_string(cid));
			}

            /** \brief average_catchment_feature returns the area-weighted average
             *
             * \tparam cell the cell type, assumed to have .geo.area(), and geo.catchment_id()
             * \tparam cell_feature_ts a callable that takes a const cell ref, returns a ts
             * \param cells that we want to perform calculation on
             * \param catchment_indexes list of catchment-id that identifies the cells, if zero length, all are averaged
             * \param cell_ts a callable that fetches the cell feature we want to average
             * \throw runtime_error if number of cells are zero
             * \return area weighted feature sum, as a ts, a  shared_ptr<pts_ts>
             */
            template<typename cell, typename cell_feature_ts>
            static shared_ptr<pts_t> average_catchment_feature(const vector<cell>& cells, const vector<int>& catchment_indexes,
                                                               cell_feature_ts&& cell_ts) {
                if (cells.size() == 0)
                    throw runtime_error("no cells to make statistics on");
				verify_cids_exist(cells, catchment_indexes);
                shared_ptr<pts_t> r;
				double sum_area = 0.0;
				bool match_all = catchment_indexes.size() == 0;
				for (const auto& c : cells) {
                    if (match_all) {
                        if(!r) r= make_shared<pts_t>(cell_ts(c).ta, 0.0, ts_point_fx::POINT_AVERAGE_VALUE);
                        r->add_scale(cell_ts(c), c.geo.area());  // c.env_ts.temperature, could be a feature(c) func return ref to ts
                        sum_area += c.geo.area();
                    } else {
                        for (auto cid : catchment_indexes) {
                            if ( c.geo.catchment_id() == (size_t) cid) { // criteria
                                if (!r) r = make_shared<pts_t>(cell_ts(c).ta, 0.0, ts_point_fx::POINT_AVERAGE_VALUE);
                                r->add_scale(cell_ts(c), c.geo.area());  // c.env_ts.temperature, could be a feature(c) func return ref to ts
                                sum_area += c.geo.area();
                                break;
                            }
                        }
                    }
				}
				r->scale_by(1/sum_area); // sih: if no match, then you will get nan here, and I think thats reasonable
				return r;
			}

			/** \brief average_catchment_feature_value returns the area-weighted average for a timestep
			*
			* \tparam cell the cell type, assumed to have .geo.area(), and geo.catchment_id()
			* \tparam cell_feature_ts a callable that takes a const cell ref, returns a ts
			* \param cells that we want to perform calculation on
			* \param catchment_indexes list of catchment-id that identifies the cells, if zero length, all are averaged
			* \param cell_ts  a callable that fetches the cell feature we want to average
			* \param i the i'th time-step for which we compute the value
			* \throw runtime_error if number of cells are zero
			* \return area weighted feature sum, as a double value
			*/
			template<typename cell, typename cell_feature_ts>
			static double average_catchment_feature_value(const vector<cell>& cells, const vector<int>& catchment_indexes,
				cell_feature_ts&& cell_ts,size_t i) {
				if (cells.size() == 0)
					throw runtime_error("no cells to make statistics on");
				verify_cids_exist(cells, catchment_indexes);
				double r = 0.0;
				double sum_area = 0.0;
				bool match_all = catchment_indexes.size() == 0;
				for (const auto& c : cells) {
					if (match_all) {
						r += cell_ts(c).value(i)*c.geo.area();  // c.env_ts.temperature, could be a feature(c) func return ref to ts
						sum_area += c.geo.area();
					} else {
						for (auto cid : catchment_indexes) {
							if (c.geo.catchment_id() == (size_t)cid) { // criteria
								r += cell_ts(c).value(i)*c.geo.area();  // c.env_ts.temperature, could be a feature(c) func return ref to ts
								sum_area += c.geo.area();
								break;
							}
						}
					}
				}
				r= r/ sum_area; // sih: if no match, then you will get nan here, and I think thats reasonable
				return r;
			}


           /** \brief sum_catchment_feature returns the sum of cell-features(discharge etc)
             *
             * \tparam cell the cell type, assumed to have geo.catchment_id()
             * \tparam cell_feature_ts a callable that takes a const cell ref, returns a ts
             * \param cells that we want to perform calculation on
             * \param catchment_indexes list of catchment-id that identifies the cells, if zero length, all are summed
             * \param cell_ts a callable that fetches the cell feature we want to sum
             * \throw runtime_error if number of cells are zero
             * \return feature sum, as a ts, a  shared_ptr<pts_ts>
             */
			template<typename cell, typename cell_feature_ts>
            static shared_ptr<pts_t> sum_catchment_feature(const vector<cell>& cells, const vector<int>& catchment_indexes,
                                                           cell_feature_ts && cell_ts) {
                if (cells.size() == 0)
                    throw runtime_error("no cells to make statistics on");
				verify_cids_exist(cells, catchment_indexes);
                shared_ptr<pts_t> r;
				bool match_all = catchment_indexes.size() == 0;

				for (const auto& c : cells) {
                    if (match_all) {
                        if(!r) r= make_shared<pts_t>(cell_ts(c).ta, 0.0, ts_point_fx::POINT_AVERAGE_VALUE);
                        r->add(cell_ts(c));
                    } else {
                        for (auto cid : catchment_indexes) {
                            if (c.geo.catchment_id() == (size_t)cid) { //criteria
                                if (!r) r = make_shared<pts_t>(cell_ts(c).ta, 0.0, ts_point_fx::POINT_AVERAGE_VALUE);
                                r->add(cell_ts(c));  //c.env_ts.temperature, could be a feature(c) func return ref to ts
                                break;
                            }
                        }
                    }
				}
				return r;
			}
			/** \brief sum_catchment_feature_value returns the sum of cell-features(discharge etc) value at the i'th timestep
			*
			* \tparam cell the cell type, assumed to have geo.catchment_id()
			* \tparam cell_feature_ts a callable that takes a const cell ref, returns a ts
			* \param cells that we want to perform calculation on
			* \param catchment_indexes list of catchment-id that identifies the cells, if zero length, all are summed
			* \param cell_ts a callable that fetches the cell feature we want to sum
			* \param i the i'th time-step of the time-axis to use
			* \throw runtime_error if number of cells are zero
			* \return feature sum, as a ts, a  shared_ptr<pts_ts>
			*/
			template<typename cell, typename cell_feature_ts>
			static double sum_catchment_feature_value(const vector<cell>& cells, const vector<int>& catchment_indexes,
				cell_feature_ts && cell_ts, size_t i) {
				if (cells.size() == 0)
					throw runtime_error("no cells to make statistics on");
				verify_cids_exist(cells, catchment_indexes);
				double r = 0.0;
				bool match_all = catchment_indexes.size() == 0;

				for (const auto& c : cells) {
					if (match_all) {
						r += cell_ts(c).value(i);
					} else {
						for (auto cid : catchment_indexes) {
							if (c.geo.catchment_id() == (size_t)cid) { //criteria
								r+=cell_ts(c).value(i);  //c.env_ts.temperature, could be a feature(c) func return ref to ts
								break;
							}
						}
					}
				}
				return r;
			}

			/** \brief catchment_feature extracts cell-features(discharge etc) for specific i'th period of timeaxis
			*
			* \tparam cell the cell type, assumed to have geo.catchment_id()
			* \tparam cell_feature_ts a callable that takes a const cell ref, returns a ts
			* \param cells that we want to extract feature from
			* \param catchment_indexes list of catchment-id that identifies the cells, if zero length, all are summed
			* \param cell_ts a callable that fetches the cell feature ts
			* \param i the i'th step on the time-axis of the cell-feature
			* \throw runtime_error if number of cells are zero
			* \return vector filled with feature for the i'th time-step on timeaxis
			*/
			template<typename cell, typename cell_feature_ts>
			static vector<double> catchment_feature(const vector<cell>& cells, const vector<int>& catchment_indexes,
				cell_feature_ts && cell_ts,size_t i) {
				if (cells.size() == 0)
					throw runtime_error("no cells to make extract from");
				verify_cids_exist(cells, catchment_indexes);
				vector<double> r; r.reserve(cells.size());
				bool match_all = catchment_indexes.size() == 0;

				for (const auto& c : cells) {
					if (match_all) {
						r.push_back(cell_ts(c).value(i));
					} else {
						for (auto cid : catchment_indexes) {
							if (c.geo.catchment_id() == (size_t)cid) { //criteria
								r.push_back(cell_ts(c).value(i));  //c.env_ts.temperature, could be a feature(c) func return ref to ts
								break;
							}
						}
					}
				}
				return r;
			}

        };
    } // core
} // shyft
