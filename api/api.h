#pragma once
#include <string>
#include <vector>
#include <map>
#include <memory>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <thread>
#include <future>
#include <stdexcept>
#include <random>

/**
 * \file
 * contains mostly typedefs and some few helper classes to
 * provide a python/swig friendly header file to the api.python project.
 *
 * \note Since we try to follow PEP-8 on the python side,
 *   which have some minor conflicts with modern C++ standards.
 *   We solve this in this file or in the swig/api.i file using
 *   rename, typedefs, or other clever tricks where needed.
 *
 */

#include "core/utctime_utilities.h"
#include "core/time_axis.h"
#include "core/geo_point.h"
#include "core/geo_cell_data.h"
#include "core/timeseries.h"
#include "core/region_model.h"
#include "core/model_calibration.h"
#include "core/bayesian_kriging.h"
#include "core/inverse_distance.h"

#include "core/pt_gs_k_cell_model.h"
#include "core/pt_hs_k_cell_model.h"
#include "core/pt_ss_k_cell_model.h"

#include "timeseries.h"

namespace shyft {
  using namespace shyft::core;
  using namespace std;
  namespace api {


    /** \brief TsFactor provides time-series creation function using supplied primitives like vector of double, start, delta-t, n etc.
     */
    struct TsFactory {

        apoint_ts
        create_point_ts(int n, utctime tStart, utctimespan dt,
                        const std::vector<double>& values,
                        point_interpretation_policy interpretation=POINT_INSTANT_VALUE){
            return apoint_ts( time_axis::fixed_dt(tStart,dt, n), values, interpretation);
        }


        apoint_ts
        create_time_point_ts(utcperiod period, const std::vector<utctime>& times,
                             const std::vector<double>& values,
                             point_interpretation_policy interpretation=POINT_INSTANT_VALUE) {
            if (times.size() == values.size() + 1) {
                return apoint_ts( time_axis::point_dt(times), values, interpretation);
            } else if (times.size() == values.size()) {
                auto tx(times);
                tx.push_back(period.end > times.back()?period.end:times.back() + utctimespan(1));
                return apoint_ts( time_axis::point_dt(tx), values, interpretation);
            } else {
                throw std::runtime_error("create_time_point_ts times and values arrays must have corresponding count");
            }
        }
    };


    /** \brief GeoPointSource contains common properties, functions
     * for the point sources in Enki.
     * Typically it contains a GeoPoint (3d position), plus a timeseries
     */
    class GeoPointSource {
      public:
        GeoPointSource(geo_point midpoint=geo_point(), apoint_ts ts=apoint_ts())
          : mid_point_(midpoint), ts(ts) {}

        typedef apoint_ts ts_t;
        typedef geo_point geo_point_t;

        geo_point mid_point_;
        apoint_ts ts;
        // boost python fixes for attributes and shared_ptr
        apoint_ts get_ts()  {return ts;}
        void set_ts(apoint_ts x) {ts=x;}
        bool is_equal(const GeoPointSource& x) const {return mid_point_==x.mid_point_ && ts == x.ts;}
        geo_point mid_point() const { return mid_point_; }
    };

    struct TemperatureSource : GeoPointSource {
        TemperatureSource(geo_point p=geo_point(), apoint_ts ts=apoint_ts())
         : GeoPointSource(p, ts) {}
        const apoint_ts& temperatures() const { return ts; }
        bool operator==(const TemperatureSource& x) {return is_equal(x);}
    };

    struct PrecipitationSource : GeoPointSource {
        PrecipitationSource(geo_point p=geo_point(), apoint_ts ts=apoint_ts())
         : GeoPointSource(p, ts) {}
        const apoint_ts& precipitations() const { return ts; }
        bool operator==(const PrecipitationSource& x) {return is_equal(x);}
    };

    struct WindSpeedSource : GeoPointSource {
        WindSpeedSource(geo_point p=geo_point(), apoint_ts ts=apoint_ts())
         : GeoPointSource(p, ts) {}
        bool operator==(const WindSpeedSource& x) {return is_equal(x);}
    };

    struct RelHumSource : GeoPointSource {
        RelHumSource(geo_point p=geo_point(), apoint_ts ts=apoint_ts())
         : GeoPointSource(p, ts) {}
        bool operator==(const RelHumSource& x) {return is_equal(x);}
    };

    struct RadiationSource : GeoPointSource {
        RadiationSource(geo_point p=geo_point(), apoint_ts ts=apoint_ts())
         : GeoPointSource(p, ts) {}
        bool operator==(const RadiationSource& x) {return is_equal(x);}
    };

    struct a_region_environment {
            typedef PrecipitationSource precipitation_t;
            typedef TemperatureSource temperature_t;
            typedef RadiationSource radiation_t;
            typedef RelHumSource rel_hum_t;
            typedef WindSpeedSource wind_speed_t;
            /** make vectors non nullptr by  default */
            a_region_environment() {
                temperature=make_shared<vector<TemperatureSource>>();
                precipitation=make_shared<vector<PrecipitationSource>>();
                radiation=make_shared<vector<RadiationSource>>();
                rel_hum=make_shared<vector<RelHumSource>>();
                wind_speed=make_shared<vector<WindSpeedSource>>();
            }
            shared_ptr<vector<TemperatureSource>>   temperature;
            shared_ptr<vector<PrecipitationSource>> precipitation;
            shared_ptr<vector<RadiationSource>>     radiation;
            shared_ptr<vector<WindSpeedSource>>     wind_speed;
            shared_ptr<vector<RelHumSource>>        rel_hum;

            // our boost python needs these methods to get properties straight (most likely it can be fixed by other means but..)
            shared_ptr<vector<TemperatureSource>> get_temperature() {return temperature;}
            void set_temperature(shared_ptr<vector<TemperatureSource>> x) {temperature=x;}
            shared_ptr<vector<PrecipitationSource>> get_precipitation() {return precipitation;}
            void set_precipitation(shared_ptr<vector<PrecipitationSource>> x) {precipitation=x;}
            shared_ptr<vector<RadiationSource>> get_radiation() {return radiation;}
            void set_radiation(shared_ptr<vector<RadiationSource>> x) {radiation=x;}
            shared_ptr<vector<WindSpeedSource>> get_wind_speed() {return wind_speed;}
            void set_wind_speed(shared_ptr<vector<WindSpeedSource>> x) {wind_speed=x;}
            shared_ptr<vector<RelHumSource>> get_rel_hum() {return rel_hum;}
            void set_rel_hum(shared_ptr<vector<RelHumSource>> x) {rel_hum=x;}
    };

    typedef shyft::timeseries::point_ts<time_axis::fixed_dt> result_ts_t;
    typedef std::shared_ptr<result_ts_t> result_ts_t_;

    /** \brief A class that facilitates fast state io, the yaml in Python is too slow
     *
     */

    template <typename cell>
    struct basic_cell_statistics {
        shared_ptr<vector<cell>> cells;
        basic_cell_statistics( shared_ptr<vector<cell>> cells):cells(cells) {}

        result_ts_t_ discharge(const vector<int>& catchment_indexes) const {
            return shyft::core::cell_statistics::
                     sum_catchment_feature(*cells, catchment_indexes,
                            [](const cell& c){ return c.rc.avg_discharge; });
        }
		vector<double> discharge(const vector<int>& catchment_indexes,size_t ith_timestep) const {
			return shyft::core::cell_statistics::
				catchment_feature(*cells, catchment_indexes,
				[](const cell& c) { return c.rc.avg_discharge; }, ith_timestep);
		}

        result_ts_t_ temperature(const vector<int>& catchment_indexes) const {
            return shyft::core::cell_statistics::
                     average_catchment_feature(*cells, catchment_indexes,
                            [](const cell& c){ return c.env_ts.temperature; });
        }
		vector<double> temperature(const vector<int>& catchment_indexes, size_t ith_timestep) const {
			return shyft::core::cell_statistics::
				catchment_feature(*cells, catchment_indexes,
				[](const cell& c) { return c.env_ts.temperature; }, ith_timestep);
		}
        result_ts_t_ precipitation(const vector<int>& catchment_indexes) const {
            return shyft::core::cell_statistics::
                     average_catchment_feature(*cells, catchment_indexes,
                            [](const cell& c){ return c.env_ts.precipitation; });
        }
		vector<double> precipitation(const vector<int>& catchment_indexes,size_t ith_timestep) const {
			return shyft::core::cell_statistics::
				catchment_feature(*cells, catchment_indexes,
				[](const cell& c) { return c.env_ts.precipitation; }, ith_timestep);
		}
        result_ts_t_ radiation(const vector<int>& catchment_indexes) const {
            return shyft::core::cell_statistics::
                     average_catchment_feature(*cells, catchment_indexes,
                            [](const cell& c){ return c.env_ts.radiation; });
        }
		vector<double> radiation(const vector<int>& catchment_indexes,size_t ith_timestep) const {
			return shyft::core::cell_statistics::
				catchment_feature(*cells, catchment_indexes,
				[](const cell& c) { return c.env_ts.radiation; }, ith_timestep);
		}

        result_ts_t_ wind_speed(const vector<int>& catchment_indexes) const {
            return shyft::core::cell_statistics::
                     average_catchment_feature(*cells, catchment_indexes,
                            [](const cell& c){ return c.env_ts.wind_speed; });
        }
		vector<double> wind_speed(const vector<int>& catchment_indexes,size_t ith_timestep) const {
			return shyft::core::cell_statistics::
				catchment_feature(*cells, catchment_indexes,
				[](const cell& c) { return c.env_ts.wind_speed; }, ith_timestep);
		}

		result_ts_t_ rel_hum(const vector<int>& catchment_indexes) const {
            return shyft::core::cell_statistics::
                     average_catchment_feature(*cells, catchment_indexes,
                            [](const cell& c){ return c.env_ts.rel_hum; });
        }
		vector<double> rel_hum(const vector<int>& catchment_indexes, size_t ith_timestep) const {
			return shyft::core::cell_statistics::
				catchment_feature(*cells, catchment_indexes,
				[](const cell& c) { return c.env_ts.rel_hum; }, ith_timestep);
		}
    };

    template <typename cell>
    struct kirchner_cell_state_statistics {
        shared_ptr<vector<cell>> cells;
        kirchner_cell_state_statistics(shared_ptr<vector<cell>> cells) :cells(cells) {}

        result_ts_t_ discharge(const vector<int>& catchment_indexes) const {
            return shyft::core::cell_statistics::
                sum_catchment_feature(*cells, catchment_indexes,
                [](const cell& c) { return c.sc.kirchner_discharge; });
        }
		vector<double> discharge(const vector<int>& catchment_indexes, size_t ith_timestep) const {
			return shyft::core::cell_statistics::
				catchment_feature(*cells, catchment_indexes,
				[](const cell& c) { return c.sc.kirchner_discharge; }, ith_timestep);
		}
    };

    ///< cells with gamma_snow state collection gives access to time-series for state
    template <typename cell>
    struct gamma_snow_cell_state_statistics {
        shared_ptr<vector<cell>> cells;
        gamma_snow_cell_state_statistics(shared_ptr<vector<cell>> cells) :cells(cells) {}

        result_ts_t_ albedo(const vector<int>& catchment_indexes) const {
            return shyft::core::cell_statistics::
                average_catchment_feature(*cells, catchment_indexes,
                [](const cell& c) { return c.sc.gs_albedo; });
        }
		vector<double> albedo(const vector<int>& catchment_indexes,size_t ith_timestep) const {
			return shyft::core::cell_statistics::
				catchment_feature(*cells, catchment_indexes,
				[](const cell& c) { return c.sc.gs_albedo; }, ith_timestep);
		}


        result_ts_t_ lwc(const vector<int>& catchment_indexes) const {
            return shyft::core::cell_statistics::
                average_catchment_feature(*cells, catchment_indexes,
                [](const cell& c) { return c.sc.gs_lwc; });
        }
		vector<double> lwc(const vector<int>& catchment_indexes, size_t ith_timestep) const {
			return shyft::core::cell_statistics::
				catchment_feature(*cells, catchment_indexes,
				[](const cell& c) { return c.sc.gs_lwc; }, ith_timestep);
		}

		result_ts_t_ surface_heat(const vector<int>& catchment_indexes) const {
            return shyft::core::cell_statistics::
                average_catchment_feature(*cells, catchment_indexes,
                [](const cell& c) { return c.sc.gs_surface_heat; });
        }
		vector<double> surface_heat(const vector<int>& catchment_indexes,size_t ith_timestep) const {
			return shyft::core::cell_statistics::
				catchment_feature(*cells, catchment_indexes,
				[](const cell& c) { return c.sc.gs_surface_heat; }, ith_timestep);
		}

        result_ts_t_ alpha(const vector<int>& catchment_indexes) const {
            return shyft::core::cell_statistics::
                average_catchment_feature(*cells, catchment_indexes,
                [](const cell& c) { return c.sc.gs_alpha; });
        }
		vector<double> alpha(const vector<int>& catchment_indexes,size_t ith_timestep) const {
			return shyft::core::cell_statistics::
				catchment_feature(*cells, catchment_indexes,
				[](const cell& c) { return c.sc.gs_alpha; }, ith_timestep);
		}

		result_ts_t_ sdc_melt_mean(const vector<int>& catchment_indexes) const {
            return shyft::core::cell_statistics::
                average_catchment_feature(*cells, catchment_indexes,
                [](const cell& c) { return c.sc.gs_sdc_melt_mean; });
        }
		vector<double> sdc_melt_mean(const vector<int>& catchment_indexes,size_t ith_timestep) const {
			return shyft::core::cell_statistics::
				catchment_feature(*cells, catchment_indexes,
				[](const cell& c) { return c.sc.gs_sdc_melt_mean; }, ith_timestep);
		}

		result_ts_t_ acc_melt(const vector<int>& catchment_indexes) const {
            return shyft::core::cell_statistics::
                average_catchment_feature(*cells, catchment_indexes,
                [](const cell& c) { return c.sc.gs_acc_melt; });
        }
		vector<double> acc_melt(const vector<int>& catchment_indexes,size_t ith_timestep) const {
			return shyft::core::cell_statistics::
				catchment_feature(*cells, catchment_indexes,
				[](const cell& c) { return c.sc.gs_acc_melt; }, ith_timestep);
		}

		result_ts_t_ iso_pot_energy(const vector<int>& catchment_indexes) const {
            return shyft::core::cell_statistics::
                average_catchment_feature(*cells, catchment_indexes,
                [](const cell& c) { return c.sc.gs_iso_pot_energy; });
        }
		vector<double> iso_pot_energy(const vector<int>& catchment_indexes,size_t ith_timestep) const {
			return shyft::core::cell_statistics::
				catchment_feature(*cells, catchment_indexes,
				[](const cell& c) { return c.sc.gs_iso_pot_energy; }, ith_timestep);
		}

		result_ts_t_ temp_swe(const vector<int>& catchment_indexes) const {
            return shyft::core::cell_statistics::
                average_catchment_feature(*cells, catchment_indexes,
                [](const cell& c) { return c.sc.gs_temp_swe; });
        }
		vector<double> temp_swe(const vector<int>& catchment_indexes,size_t ith_timestep) const {
			return shyft::core::cell_statistics::
				catchment_feature(*cells, catchment_indexes,
				[](const cell& c) { return c.sc.gs_temp_swe; }, ith_timestep);
		}
	};

    ///< cells with gamma_snow response give access to time-series for these
    template <typename cell>
    struct gamma_snow_cell_response_statistics {
        shared_ptr<vector<cell>> cells;
        gamma_snow_cell_response_statistics(shared_ptr<vector<cell>> cells) :cells(cells) {}

        result_ts_t_ sca(const vector<int>& catchment_indexes) const {
            return shyft::core::cell_statistics::
                average_catchment_feature(*cells, catchment_indexes,
                [](const cell& c) { return c.rc.snow_sca; });
        }
		vector<double> sca(const vector<int>& catchment_indexes, size_t ith_timestep) const {
			return shyft::core::cell_statistics::
				catchment_feature(*cells, catchment_indexes,
				[](const cell& c) { return c.rc.snow_sca; }, ith_timestep);
		}

        result_ts_t_ swe(const vector<int>& catchment_indexes) const {
            return shyft::core::cell_statistics::
                average_catchment_feature(*cells, catchment_indexes,
                [](const cell& c) { return c.rc.snow_swe; });
        }
		vector<double> swe(const vector<int>& catchment_indexes, size_t ith_timestep) const {
			return shyft::core::cell_statistics::
				catchment_feature(*cells, catchment_indexes,
				[](const cell& c) { return c.rc.snow_swe; }, ith_timestep);
		}
		result_ts_t_ outflow(const vector<int>& catchment_indexes) const {
            return shyft::core::cell_statistics::
                sum_catchment_feature(*cells, catchment_indexes,
                [](const cell& c) { return c.rc.snow_outflow; });
        }
		vector<double> outflow(const vector<int>& catchment_indexes, size_t ith_timestep) const {
			return shyft::core::cell_statistics::
				catchment_feature(*cells, catchment_indexes,
				[](const cell& c) { return c.rc.snow_outflow; }, ith_timestep);
		}
	};

    ///< access to skaugen's snow routine's state statistics
    template <typename cell>
    struct skaugen_cell_state_statistics {
        shared_ptr<vector<cell>> cells;
        skaugen_cell_state_statistics(shared_ptr<vector<cell>> cells) :cells(cells) {}

        result_ts_t_ alpha(const vector<int>& catchment_indexes) const {
            return shyft::core::cell_statistics::
                average_catchment_feature(*cells, catchment_indexes,
                [](const cell& c) { return c.sc.snow_alpha; });
        }
		vector<double> alpha(const vector<int>& catchment_indexes, size_t ith_timestep) const {
			return shyft::core::cell_statistics::
				catchment_feature(*cells, catchment_indexes,
				[](const cell& c) { return c.sc.snow_alpha; }, ith_timestep);
		}

        result_ts_t_ nu(const vector<int>& catchment_indexes) const {
            return shyft::core::cell_statistics::
                average_catchment_feature(*cells, catchment_indexes,
                [](const cell& c) { return c.sc.snow_nu; });
        }
		vector<double> nu(const vector<int>& catchment_indexes, size_t ith_timestep) const {
			return shyft::core::cell_statistics::
				catchment_feature(*cells, catchment_indexes,
				[](const cell& c) { return c.sc.snow_nu; }, ith_timestep);
		}

        result_ts_t_ lwc(const vector<int>& catchment_indexes) const {
            return shyft::core::cell_statistics::
                average_catchment_feature(*cells, catchment_indexes,
                [](const cell& c) { return c.sc.snow_lwc; });
        }
		vector<double> lwc(const vector<int>& catchment_indexes, size_t ith_timestep) const {
			return shyft::core::cell_statistics::
				catchment_feature(*cells, catchment_indexes,
				[](const cell& c) { return c.sc.snow_lwc; }, ith_timestep);
		}

        result_ts_t_ residual(const vector<int>& catchment_indexes) const {
            return shyft::core::cell_statistics::
                average_catchment_feature(*cells, catchment_indexes,
                [](const cell& c) { return c.sc.snow_residual; });
        }
		vector<double> residual(const vector<int>& catchment_indexes, size_t ith_timestep) const {
			return shyft::core::cell_statistics::
				catchment_feature(*cells, catchment_indexes,
				[](const cell& c) { return c.sc.snow_residual; }, ith_timestep);
		}

        result_ts_t_ swe(const vector<int>& catchment_indexes) const {
            return shyft::core::cell_statistics::
                average_catchment_feature(*cells, catchment_indexes,
                [](const cell& c) { return c.sc.snow_swe; });
        }
		vector<double> swe(const vector<int>& catchment_indexes, size_t ith_timestep) const {
			return shyft::core::cell_statistics::
				catchment_feature(*cells, catchment_indexes,
				[](const cell& c) { return c.sc.snow_swe; }, ith_timestep);
		}

        result_ts_t_ sca(const vector<int>& catchment_indexes) const {
            return shyft::core::cell_statistics::
                average_catchment_feature(*cells, catchment_indexes,
                [](const cell& c) { return c.sc.snow_sca; });
        }
		vector<double> sca(const vector<int>& catchment_indexes, size_t ith_timestep) const {
			return shyft::core::cell_statistics::
				catchment_feature(*cells, catchment_indexes,
				[](const cell& c) { return c.sc.snow_sca; }, ith_timestep);
		}
	};

    ///< access to skaugen's snow routine response statistics
    template <typename cell>
    struct skaugen_cell_response_statistics {
        shared_ptr<vector<cell>> cells;
        skaugen_cell_response_statistics(shared_ptr<vector<cell>> cells) : cells(cells) {}

        result_ts_t_ outflow(const vector<int>& catchment_indexes) const {
            return shyft::core::cell_statistics::
                sum_catchment_feature(*cells, catchment_indexes,
                [](const cell& c) { return c.rc.snow_outflow; });
        }
		vector<double> outflow(const vector<int>& catchment_indexes, size_t ith_timestep) const {
			return shyft::core::cell_statistics::
				catchment_feature(*cells, catchment_indexes,
				[](const cell& c) { return c.rc.snow_outflow; }, ith_timestep);
		}

        result_ts_t_ total_stored_water(const vector<int>& catchment_indexes) const {
            return shyft::core::cell_statistics::
                sum_catchment_feature(*cells, catchment_indexes,
                [](const cell& c) { return c.rc.snow_total_stored_water; });
        }
		vector<double> total_stored_water(const vector<int>& catchment_indexes, size_t ith_timestep) const {
			return shyft::core::cell_statistics::
				catchment_feature(*cells, catchment_indexes,
				[](const cell& c) { return c.rc.snow_total_stored_water; }, ith_timestep);
		}
	};

    ///< access to hbv snow routine's state statistics
    template <typename cell>
    struct hbv_snow_cell_state_statistics {
        shared_ptr<vector<cell>> cells;
        hbv_snow_cell_state_statistics(shared_ptr<vector<cell>> cells) : cells(move(cells)) {}

        result_ts_t_ swe(const vector<int>& catchment_indexes) const {
            return shyft::core::cell_statistics::
                average_catchment_feature(*cells, catchment_indexes,
                [](const cell& c) { return c.sc.snow_swe; });
        }
		vector<double> swe(const vector<int>& catchment_indexes, size_t ith_timestep) const {
			return shyft::core::cell_statistics::
				catchment_feature(*cells, catchment_indexes,
				[](const cell& c) { return c.sc.snow_swe; }, ith_timestep);
		}

        result_ts_t_ sca(const vector<int>& catchment_indexes) const {
            return shyft::core::cell_statistics::
                average_catchment_feature(*cells, catchment_indexes,
                [](const cell& c) { return c.sc.snow_sca; });
        }
		vector<double> sca(const vector<int>& catchment_indexes, size_t ith_timestep) const {
			return shyft::core::cell_statistics::
				catchment_feature(*cells, catchment_indexes,
				[](const cell& c) { return c.sc.snow_sca; }, ith_timestep);
		}
    };

    ///< access to hbv snow routine response statistics
    template <typename cell>
    struct hbv_snow_cell_response_statistics {
        shared_ptr<vector<cell>> cells;
        hbv_snow_cell_response_statistics(shared_ptr<vector<cell>> cells) : cells(cells) {}

        result_ts_t_ outflow(const vector<int>& catchment_indexes) const {
            return shyft::core::cell_statistics::
                sum_catchment_feature(*cells, catchment_indexes,
                [](const cell& c) { return c.rc.snow_outflow; });
        }
		vector<double> outflow(const vector<int>& catchment_indexes, size_t ith_timestep) const {
			return shyft::core::cell_statistics::
				catchment_feature(*cells, catchment_indexes,
				[](const cell& c) { return c.rc.snow_outflow; }, ith_timestep);
		}
    };

    template <typename cell>
    struct priestley_taylor_cell_response_statistics {
        shared_ptr<vector<cell>> cells;
        priestley_taylor_cell_response_statistics(shared_ptr<vector<cell>> cells) :cells(cells) {}

        result_ts_t_ output(const vector<int>& catchment_indexes) const {
            return shyft::core::cell_statistics::
                average_catchment_feature(*cells, catchment_indexes,
                [](const cell& c) { return c.rc.pe_output; });
        }
		vector<double> output(const vector<int>& catchment_indexes, size_t ith_timestep) const {
			return shyft::core::cell_statistics::
				catchment_feature(*cells, catchment_indexes,
				[](const cell& c) { return c.rc.pe_output; }, ith_timestep);
		}
	};

    template <typename cell>
    struct actual_evapotranspiration_cell_response_statistics {
        shared_ptr<vector<cell>> cells;
        actual_evapotranspiration_cell_response_statistics(shared_ptr<vector<cell>> cells) :cells(cells) {}

        result_ts_t_ output(const vector<int>& catchment_indexes) const {
            return shyft::core::cell_statistics::
                average_catchment_feature(*cells, catchment_indexes,
                [](const cell& c) { return c.rc.ae_output; });
        }
		vector<double> output(const vector<int>& catchment_indexes, size_t ith_timestep) const {
			return shyft::core::cell_statistics::
				catchment_feature(*cells, catchment_indexes,
				[](const cell& c) { return c.rc.ae_output; }, ith_timestep);
		}
	};
  } // Namespace api
}// Namespace shyft
