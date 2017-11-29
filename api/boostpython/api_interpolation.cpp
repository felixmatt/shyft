#include "boostpython_pch.h"
#include "api/api.h"
#include "core/inverse_distance.h"
#include "core/bayesian_kriging.h"
#include "core/kriging.h"
#include "core/region_model.h"

namespace expose {
    using namespace boost::python;
    namespace sa = shyft::api;
    namespace sc = shyft::core;
	namespace sta = shyft::time_axis;
	namespace btk = shyft::core::bayesian_kriging;
	namespace idw = shyft::core::inverse_distance;

	typedef std::vector<sc::geo_point> geo_point_vector;
	typedef std::vector<sa::GeoPointSource> geo_ts_vector;
	typedef std::shared_ptr<geo_ts_vector> geo_ts_vector_;
	typedef std::vector<sa::TemperatureSource> geo_temperature_vector;
	typedef std::shared_ptr<geo_temperature_vector> geo_temperature_vector_;
	typedef std::vector<sa::PrecipitationSource> geo_precipitation_vector;
	typedef std::shared_ptr<geo_precipitation_vector> geo_precipitation_vector_;

    typedef std::vector<sa::RadiationSource> geo_radiation_vector;
    typedef std::shared_ptr<geo_radiation_vector> geo_radiation_vector_;

    typedef std::vector<sa::RelHumSource> geo_rel_hum_vector;
    typedef std::shared_ptr<geo_rel_hum_vector> geo_rel_hum_vector_;

    typedef std::vector<sa::WindSpeedSource> geo_wind_speed_vector;
    typedef std::shared_ptr<geo_wind_speed_vector> geo_wind_speed_vector_;

	template <typename VectorT>
	static std::shared_ptr<VectorT> make_dest_geo_ts(const geo_point_vector& points, sta::fixed_dt time_axis) {
		auto dst = std::make_shared<VectorT>();
		dst->reserve(points.size());
		double std_nan = std::numeric_limits<double>::quiet_NaN();
		for (const auto& gp : points)
			dst->emplace_back(gp, sa::apoint_ts(time_axis, std_nan, shyft::time_series::ts_point_fx::POINT_AVERAGE_VALUE));
		return dst;
	}

	template <typename VectorT>
	static void validate_parameters(const std::shared_ptr<VectorT> & src, const geo_point_vector& dst_points, sta::fixed_dt time_axis) {
		if (src==nullptr || src->size()==0 || dst_points.size()==0)
			throw std::runtime_error("the supplied src and dst_points should be non-null and have at least one time-series");
		if (time_axis.size()==0 || time_axis.delta()==0)
			throw std::runtime_error("the supplied destination time-axis should have more than 0 element, and a delta-t larger than 0");
	}

    ///< a local wrapper with api-typical checks on the input to support use from python
    static geo_temperature_vector_ bayesian_kriging_temperature(geo_temperature_vector_ src,const geo_point_vector& dst_points,shyft::time_axis::fixed_dt time_axis,btk::parameter btk_parameter) {
        using namespace std;
        typedef shyft::time_series::average_accessor<typename shyft::api::apoint_ts, shyft::time_axis::fixed_dt> btk_tsa_t;
        // 1. some minor checks to give the python user early warnings.
        validate_parameters(src, dst_points, time_axis);
        auto dst = make_dest_geo_ts<geo_temperature_vector>(dst_points, time_axis);
        // 2. then run btk to fill inn the results
        if(src->size()>1) {
            btk::btk_interpolation<btk_tsa_t>(begin(*src), end(*src), begin(*dst), end(*dst),time_axis, btk_parameter);
        } else {
            // just one temperature ts. just a a clean copy to destinations
            btk_tsa_t tsa((*src)[0].ts, time_axis);
            sa::apoint_ts temp_ts(time_axis, 0.0);
            for(size_t i=0;i<time_axis.size();++i) temp_ts.set(i, tsa.value(i));
            for(auto& d:*dst) d.ts=temp_ts;
        }
        return dst;
    }

    enum ok_covariance_type {
        GAUSSIAN =0,
        EXPONENTIAL=1
    };
    struct ok_parameter {
        double c;
        double a;
        double z_scale;
        ok_covariance_type cov_type;
        ok_parameter(double c=1.0,double a=10*1000.0,ok_covariance_type cov_type=ok_covariance_type::EXPONENTIAL, double z_scale=1.0):
            c(c),a(a),z_scale(z_scale),cov_type(cov_type) {}
    };

    static geo_ts_vector_ ordinary_kriging(geo_ts_vector_ src,const geo_point_vector& dst_points,shyft::time_axis::fixed_dt time_axis, ok_parameter p ) {
        validate_parameters(src,dst_points,time_axis);
        if(p.a <=0.0)
            throw std::runtime_error("the supplied parameter a, covariance practical range, must be >0.0");
        if(p.c<=0.0)
            throw std::runtime_error("the supplied parameter c, covariance sill, must be >0.0");
        if(p.z_scale <0.0)
            throw std::runtime_error("the supplied parameter z_scale used to scale vertical distance must be >= 0.0");

        auto dst = make_dest_geo_ts<geo_ts_vector>(dst_points,time_axis);
        typedef shyft::time_series::average_accessor<sa::apoint_ts, sc::timeaxis_t> avg_tsa_t;

        if(src->size()>1) {
            // make accessor for the observations
            // build and solve the ordinary::kriging Ax = b, invert A
            // compute the weights
            //
            shyft::core::kriging::covariance::exponential exp_cov(p.c,p.a);
            shyft::core::kriging::covariance::gaussian gss_cov(p.c,p.a);
            auto fx_cov =[p,exp_cov](const geo_ts_vector::value_type &o1,const geo_ts_vector::value_type &o2 )->double {
                double distance = shyft::core::geo_point::zscaled_distance(o1.mid_point_,o2.mid_point_,p.z_scale);
                return exp_cov(distance);
            };
            auto fg_cov =[p,gss_cov](const geo_ts_vector::value_type &o1,const geo_ts_vector::value_type &o2 )->double {
                double distance2 = shyft::core::geo_point::zscaled_distance2(o1.mid_point_,o2.mid_point_,p.z_scale);
                return gss_cov(distance2);
            };
            arma::mat A;
            arma::mat B;
            if(p.cov_type==ok_covariance_type::EXPONENTIAL) {
                A = shyft::core::kriging::ordinary::build(begin(*src),end(*src),fx_cov);
                B = shyft::core::kriging::ordinary::build(begin(*src),end(*src),begin(*dst),end(*dst),fx_cov);
            } else {
                A = shyft::core::kriging::ordinary::build(begin(*src),end(*src),fg_cov);
                B = shyft::core::kriging::ordinary::build(begin(*src),end(*src),begin(*dst),end(*dst),fg_cov);
            }
            auto X = (A.i()*B).eval();
            auto weights = X.head_rows(src->size());// skip sum w = 1.0 row at bottom
            std::vector<avg_tsa_t> obs_tsa;obs_tsa.reserve(src->size());
            for(auto&s:*src)
                obs_tsa.emplace_back(s.ts,time_axis);
            //TODO: make partition on destination cells and use multiple threads to exec
            //      just copy the obs_tsa (it's a shallow copy of the stuff that could have a cost)
            // spawn 1-4 threads pr. core available
            // join & wait
            arma::mat obs(1,src->size(),arma::fill::none);
            arma::mat dst_values(1,dst->size(),arma::fill::none);
            for(size_t p=0;p<time_axis.size();++p) {
                // make obs. vector
                for(size_t j=0;j<src->size();++j) obs.at(0,j) = obs_tsa[j].value(p);
                dst_values = obs*weights;// compute the destination values
                for(size_t j=0;j<dst->size();++j) (*dst)[j].ts.set(p,dst_values(0,j));
                // for each d
            }

        } else {
            avg_tsa_t tsa((*src)[0].ts,time_axis);
            sa::apoint_ts temp_ts(time_axis, 0.0);
            for(size_t i=0;i<time_axis.size();++i) temp_ts.set(i, tsa.value(i));
            for(auto& d:*dst) d.ts=temp_ts;
        }
        return dst;
    }
    static void ok_kriging() {
        enum_<ok_covariance_type>("OKCovarianceType")
            .value("GAUSSIAN",ok_covariance_type::GAUSSIAN)
            .value("EXPONENTIAL",ok_covariance_type::EXPONENTIAL)
            .export_values()
            ;

        class_<ok_parameter>("OKParameter","Ordinary Kriging Parameter, keeps parameters that controls the ordinary kriging calculation")
            .def(init<optional<double,double,ok_covariance_type,double>>(args("c","a","cov_type","z_scale")))
            .def_readwrite("c",&ok_parameter::c,"the c-constant, sill value in the covariance formula")
            .def_readwrite("a",&ok_parameter::a,"the a-constant, range or distance, value in the covariance formula")
            .def_readwrite("cov_type",&ok_parameter::cov_type,"covariance type EXPONENTIAL|GAUSSIAN to be used")
            .def_readwrite("z_scale", &ok_parameter::z_scale,"z_scale to be used for range|distance calculations")
            ;

        def("ordinary_kriging",ordinary_kriging,
            "Runs ordinary kriging for geo sources and project the source out to the destination geo-timeseries\n"
            "\n\n\tNotice that kriging is currently not very efficient for large grid inputs,\n"
            "\tusing only one thread, and considering all source-timeseries (entire grid) for all destinations\n"
            "\tFor few sources, spread out on a grid, it's quite efficient should work well\n"
            "\tAlso note that this function currently does not elicite observations with nan-data\n"
            "\tmost useful when you have control on the inputs, providing full set of data.\n"
            "\n"
            "Parameters\n"
            "----------\n"
            "src : GeoSourceVector\n"
            "\t input a geo-located list of time-series with filled in values\n\n"
            "dst : GeoPointVector\n"
            "\tthe GeoPoints,(x,y,z) locations to interpolate into\n"
            "time_axis : Timeaxis, - the destination time-axis, recall that the inputs can be any-time-axis, \n"
            "\tand they are transformed and interpolated into the destination-timeaxis\n"
            "parameter:OKParameter\n"
            "\t the parameters to be used during interpolation\n\n"
            "Returns\n"
            "-------\n"
            "GeoSourceVector, -with filled in ts-values according to their position, the parameters and time_axis\n"
            );
    }

    static void btk_interpolation() {
        typedef shyft::core::bayesian_kriging::parameter BTKParameter;

        class_<BTKParameter>("BTKParameter","BTKParameter class with time varying gradient based on day no")
            .def(init<double,double>(args("temperature_gradient","temperature_gradient_sd"),"specifying default temp.grad(not used) and std.dev[C/100m]"))
            .def(init<double,double,double,double,double,double>(args("temperature_gradient","temperature_gradient_sd", "sill", "nugget", "range", "zscale"),"full specification of all parameters"))
            .def("temperature_gradient",&BTKParameter::temperature_gradient,args("p"),"return default temp.gradient based on day of year calculated for midst of utcperiod p")
            .def("temperature_gradient_sd",&BTKParameter::temperature_gradient_sd,"returns Prior standard deviation of temperature gradient in [C/m]" )
            .def("sill",&BTKParameter::sill,"Value of semivariogram at range default=25.0")
            .def("nug",&BTKParameter::nug,"Nugget magnitude,default=0.5")
            .def("range",&BTKParameter::range,"Point where semivariogram flattens out,default=200000.0")
            .def("zscale",&BTKParameter::zscale,"Height scale used during distance computations,default=20.0")
            ;
        def("bayesian_kriging_temperature",bayesian_kriging_temperature,
            "Runs kriging for temperature sources and project the temperatures out to the destination geo-timeseries\n"
            "\n\n\tNotice that bayesian kriging is currently not very efficient for large grid inputs,\n"
            "\tusing only one thread, and considering all source-timeseries (entire grid) for all destinations\n"
            "\tFor few sources, spread out on a grid, it's quite efficient should work well\n"
            "\n"
            "Parameters\n"
            "----------\n"
            "src : TemperatureSourceVector\n"
            "\t input a geo-located list of temperature time-series with filled in values (some might be nan etc.)\n\n"
            "dst : GeoPointVector\n"
            "\tthe GeoPoints,(x,y,z) locations to interpolate into\n"
            "time_axis : Timeaxis, - the destination time-axis, recall that the inputs can be any-time-axis, \n"
            "\tand they are transformed and interpolated into the destination-timeaxis\n"
            "btk_parameter:BTKParameter\n"
            "\t the parameters to be used during interpolation\n\n"
            "Returns\n"
            "-------\n"
            "TemperatureSourceVector, -with filled in temperatures according to their position, the idw_parameters and time_axis\n"
            );
    }

	static geo_temperature_vector_ idw_temperature(geo_temperature_vector_ src, const geo_point_vector& dst_points, shyft::time_axis::fixed_dt ta, idw::temperature_parameter idw_temp_p) {
		typedef shyft::time_series::average_accessor<sa::apoint_ts, sc::timeaxis_t> avg_tsa_t;
		typedef sc::idw_compliant_geo_point_ts<sa::TemperatureSource, avg_tsa_t, sc::timeaxis_t> idw_gts_t;
		typedef idw::temperature_model<idw_gts_t, sa::TemperatureSource, idw::temperature_parameter, sc::geo_point, idw::temperature_gradient_scale_computer> idw_temperature_model_t;

		validate_parameters(src, dst_points, ta);
        auto dst = make_dest_geo_ts<geo_temperature_vector>(dst_points, ta);
		idw::run_interpolation<idw_temperature_model_t, idw_gts_t>(ta, *src, idw_temp_p, *dst,
			[](auto& d, size_t ix, double value) { d.set_value(ix, value); });

		return dst;
	}

	static geo_precipitation_vector_ idw_precipitation(geo_precipitation_vector_ src, const geo_point_vector& dst_points, shyft::time_axis::fixed_dt ta, idw::precipitation_parameter idw_p) {
		typedef shyft::time_series::average_accessor<sa::apoint_ts, sc::timeaxis_t> avg_tsa_t;
		typedef sc::idw_compliant_geo_point_ts<sa::PrecipitationSource, avg_tsa_t, sc::timeaxis_t> idw_gts_t;
		typedef idw::precipitation_model<idw_gts_t, sa::PrecipitationSource, idw::precipitation_parameter, sc::geo_point> idw_precipitation_model_t;

		validate_parameters(src, dst_points, ta);
		auto dst = make_dest_geo_ts<geo_precipitation_vector>(dst_points, ta);
		idw::run_interpolation<idw_precipitation_model_t, idw_gts_t>(ta, *src, idw_p, *dst,
			[](auto& d, size_t ix, double value) { d.set_value(ix, value); });

		return dst;
	}

    /** fake cell to support slope-factor pr. cell*/
    struct radiation_cell {
        size_t cell_ix;
        double cell_slope_factor;
        geo_radiation_vector_ dst;
        sc::geo_point mid_point() const { return (*dst)[cell_ix].mid_point();}
        void set(size_t ix, double value) {(*dst)[cell_ix].ts.set(ix, value);}
        double slope_factor() const { return cell_slope_factor; }
    };
    static geo_radiation_vector_ idw_radiation(geo_radiation_vector_ src, const geo_point_vector& dst_points, shyft::time_axis::fixed_dt ta, idw::parameter idw_p, const std::vector<double>& radiation_slope_factors) {
        typedef shyft::time_series::average_accessor<sa::apoint_ts, sc::timeaxis_t> avg_tsa_t;
        typedef sc::idw_compliant_geo_point_ts<sa::RadiationSource, avg_tsa_t, sc::timeaxis_t> idw_gts_t;
        typedef idw::radiation_model<idw_gts_t, radiation_cell, idw::parameter, sc::geo_point> idw_radiation_model_t;

        validate_parameters(src, dst_points, ta);
        if (dst_points.size()!=radiation_slope_factors.size())
            throw std::runtime_error("slope-factors needs to have same length as destination points");
        auto dst = make_dest_geo_ts<geo_radiation_vector>(dst_points, ta);
        std::vector<radiation_cell> rdst; rdst.reserve(dst->size());
        for (size_t i = 0; i<dst->size(); ++i)
            rdst.emplace_back(radiation_cell{ i, radiation_slope_factors[i], dst });

        idw::run_interpolation<idw_radiation_model_t, idw_gts_t>(ta, *src, idw_p, rdst,
                                                                     [](auto& d, size_t ix, double value) { d.set(ix, value); });

        return dst;
    }


    static geo_wind_speed_vector_ idw_wind_speed(geo_wind_speed_vector_ src, const geo_point_vector& dst_points, shyft::time_axis::fixed_dt ta, idw::parameter idw_p) {
        typedef shyft::time_series::average_accessor<sa::apoint_ts, sc::timeaxis_t> avg_tsa_t;
        typedef sc::idw_compliant_geo_point_ts<sa::WindSpeedSource, avg_tsa_t, sc::timeaxis_t> idw_gts_t;
        typedef idw::wind_speed_model<idw_gts_t, sa::WindSpeedSource, idw::parameter, sc::geo_point> idw_wind_speed_model_t;

        validate_parameters(src, dst_points, ta);
        auto dst = make_dest_geo_ts<geo_wind_speed_vector>(dst_points, ta);
        idw::run_interpolation<idw_wind_speed_model_t, idw_gts_t>(ta, *src, idw_p, *dst,
                                                                 [](auto& d, size_t ix, double value) { d.ts.set(ix, value); });

        return dst;
    }

    static geo_rel_hum_vector_ idw_rel_hum(geo_rel_hum_vector_ src, const geo_point_vector& dst_points, shyft::time_axis::fixed_dt ta, idw::parameter idw_p) {
        typedef shyft::time_series::average_accessor<sa::apoint_ts, sc::timeaxis_t> avg_tsa_t;
        typedef sc::idw_compliant_geo_point_ts<sa::RelHumSource, avg_tsa_t, sc::timeaxis_t> idw_gts_t;
        typedef idw::rel_hum_model<idw_gts_t, sa::RelHumSource, idw::parameter, sc::geo_point> idw_rel_hum_model_t;

        validate_parameters(src, dst_points, ta);
        auto dst = make_dest_geo_ts<geo_rel_hum_vector>(dst_points, ta);
        idw::run_interpolation<idw_rel_hum_model_t, idw_gts_t>(ta, *src, idw_p, *dst,
                                                                  [](auto& d, size_t ix, double value) { d.ts.set(ix, value); });

        return dst;
    }

    static void idw_interpolation() {
        typedef shyft::core::inverse_distance::parameter IDWParameter;

        class_<IDWParameter>("IDWParameter",
                    "IDWParameter is a simple place-holder for IDW parameters\n"
                    "The two most common:\n"
                    "  max_distance \n"
                    "  max_members \n"
                    "Is used for interpolation.\n"
                    "Additionally it keep distance measure-factor,\n"
                    "so that the IDW distance is computed as 1 over pow(euclid distance,distance_measure_factor)\n"
					"zscale is used to discriminate neighbors that are at different elevation than target point.\n"
			)
			.def(init<int,optional<double,double>>(args("max_members","max_distance","distance_measure_factor"),"create IDW from supplied parameters"))
            .def_readwrite("max_members",&IDWParameter::max_members,"maximum members|neighbors used to interpolate into a point,default=10")
            .def_readwrite("max_distance",&IDWParameter::max_distance,"[meter] only neighbours within max distance is used for each destination-cell,default= 200000.0")
			.def_readwrite("distance_measure_factor",&IDWParameter::distance_measure_factor,"IDW distance is computed as 1 over pow(euclid distance,distance_measure_factor), default=2.0")
			.def_readwrite("zscale",&IDWParameter::zscale,"Use to weight neighbors having same elevation, default=1.0")
            ;
		def("idw_temperature", idw_temperature,
			"Runs inverse distance interpolation to project temperature sources out to the destination geo-timeseries\n"
			"\n"
			"Parameters\n"
			"----------\n"
			"src : TemperatureSourceVector\n"
			"\t input a geo-located list of temperature time-series with filled in values (some might be nan etc.)\n\n"
			"dst : GeoPointVector\n"
			"\tthe GeoPoints,(x,y,z) locations to interpolate into\n"
			"time_axis : Timeaxis, - the destination time-axis, recall that the inputs can be any-time-axis, \n"
			"\tthey are transformed and interpolated into the destination-timeaxis\n"
			"idw_para : IDWTemperatureParameter\n"
			"\t the parameters to be used during interpolation\n\n"
			"Returns\n"
			"-------\n"
			"TemperatureSourceVector, -with filled in temperatures according to their position, the idw_parameters and time_axis\n"
		);
		def("idw_precipitation", idw_precipitation,
			"Runs inverse distance interpolation to project precipitation sources out to the destination geo-timeseries\n"
			"\n"
			"Parameters\n"
			"----------\n"
			"src : PrecipitationSourceVector\n"
			"\t input a geo-located list of precipitation time-series with filled in values (some might be nan etc.)\n\n"
			"dst : GeoPointVector\n"
			"\tthe GeoPoints,(x,y,z) locations to interpolate into\n"
			"time_axis : Timeaxis, - the destination time-axis, recall that the inputs can be any-time-axis, \n"
			"\tthey are transformed and interpolated into the destination-timeaxis\n"
			"idw_para : IDWPrecipitationParameter\n"
			"\t the parameters to be used during interpolation\n\n"
			"Returns\n"
			"-------\n"
			"PrecipitationSourceVector, -with filled in precipitations according to their position, the idw_parameters and time_axis\n"
		);

        typedef shyft::core::inverse_distance::temperature_parameter IDWTemperatureParameter;
        class_<IDWTemperatureParameter,bases<IDWParameter>> ("IDWTemperatureParameter",
                "For temperature inverse distance, also provide default temperature gradient to be used\n"
                "when the gradient can not be computed.\n"
                "note: if gradient_by_equation is set true, and number of points >3, the temperature gradient computer\n"
                "      will try to use the 4 closes points and determine the 3d gradient including the vertical gradient.\n"
                "      (in scenarios with constant gradients(vertical/horizontal), this is accurate) \n"
            )
            .def(init<double,optional<int,double,bool>>(args("default_gradient","max_members","max_distance","gradient_by_equation"),"construct IDW for temperature as specified by arguments"))
            .def_readwrite("default_temp_gradient",&IDWTemperatureParameter::default_temp_gradient,"[degC/m], default=-0.006")
            .def_readwrite("gradient_by_equation",&IDWTemperatureParameter::gradient_by_equation,"if true, gradient is computed using 4 closest neighbors, solving equations to find 3D temperature gradients.")
            ;

        typedef shyft::core::inverse_distance::precipitation_parameter IDWPrecipitationParameter;
        class_<IDWPrecipitationParameter,bases<IDWParameter>>("IDWPrecipitationParameter",
                    "For precipitation,the scaling model needs the scale_factor.\n"
                    "adjusted_precipitation = precipitation* (scale_factor)^(z-distance-in-meters/100.0)\n"
                    "Ref to IDWParameter for the other parameters\n"
            )
            .def(init<double,optional<int,double>>(args("scale_factor", "max_members","max_distance"),"create IDW from supplied parameters"))
            .def_readwrite("scale_factor",&IDWPrecipitationParameter::scale_factor," ref. formula for adjusted_precipitation,  default=1.02")
        ;
        //-- remaining exposure
        def("idw_radiation", idw_radiation,
            "Runs inverse distance interpolation to project radiation sources out to the destination geo-timeseries\n"
            "\n"
            "Parameters\n"
            "----------\n"
            "src : RadiationSourceVector\n"
            "\t input a geo-located list of precipitation time-series with filled in values (some might be nan etc.)\n\n"
            "dst : GeoPointVector\n"
            "\tthe GeoPoints,(x,y,z) locations to interpolate into\n"
            "time_axis : Timeaxis, - the destination time-axis, recall that the inputs can be any-time-axis, \n"
            "\tthey are transformed and interpolated into the destination-timeaxis\n"
            "idw_para : IDWParameter\n"
            "\t the parameters to be used during interpolation\n\n"
            "slope_factors: DoubleVector\n"
            "\t the slope-factor corresponding to geopoints, typical 0.9 \n"
            "Returns\n"
            "-------\n"
            "RadiationSourceVector, -with filled in radiation according to their position, the idw_parameters and time_axis\n"
        );
        def("idw_relative_humidity", idw_rel_hum,
            "Runs inverse distance interpolation to project relative humidity sources out to the destination geo-timeseries\n"
            "\n"
            "Parameters\n"
            "----------\n"
            "src : RelHumSourceVector\n"
            "\t input a geo-located list of precipitation time-series with filled in values (some might be nan etc.)\n\n"
            "dst : GeoPointVector\n"
            "\tthe GeoPoints,(x,y,z) locations to interpolate into\n"
            "time_axis : Timeaxis, - the destination time-axis, recall that the inputs can be any-time-axis, \n"
            "\tthey are transformed and interpolated into the destination-timeaxis\n"
            "idw_para : IDWParameter\n"
            "\t the parameters to be used during interpolation\n\n"
            "Returns\n"
            "-------\n"
            "RelHumSourceVector, -with filled in relative humidity according to their position, the idw_parameters and time_axis\n"
        );
        def("idw_wind_speed", idw_wind_speed,
            "Runs inverse distance interpolation to project precipitation sources out to the destination geo-timeseries\n"
            "\n"
            "Parameters\n"
            "----------\n"
            "src : RelHumSourceVector\n"
            "\t input a geo-located list of wind speed time-series with filled in values (some might be nan etc.)\n\n"
            "dst : GeoPointVector\n"
            "\tthe GeoPoints,(x,y,z) locations to interpolate into\n"
            "time_axis : Timeaxis, - the destination time-axis, recall that the inputs can be any-time-axis, \n"
            "\tthey are transformed and interpolated into the destination-timeaxis\n"
            "idw_para : IDWParameter\n"
            "\t the parameters to be used during interpolation\n\n"
            "Returns\n"
            "-------\n"
            "WindSpeedSourceVector, -with filled in wind speed according to their position, the idw_parameters and time_axis\n"
        );

    }

	static void interpolation_parameter() {
        typedef shyft::core::interpolation_parameter InterpolationParameter;
        namespace idw = shyft::core::inverse_distance;
        namespace btk = shyft::core::bayesian_kriging;
        class_<InterpolationParameter>("InterpolationParameter",
                 "The InterpolationParameter keep parameters needed to perform the\n"
                 "interpolation steps, IDW,BTK etc\n"
                 "It is used as parameter  in the model.run_interpolation() method\n"
            )
            .def(init<const btk::parameter&,const idw::precipitation_parameter&,const idw::parameter&,const idw::parameter&,const idw::parameter&>(args("temperature","precipitation","wind_speed","radiation","rel_hum"),"using BTK for temperature"))
            .def(init<const idw::temperature_parameter&,const idw::precipitation_parameter&,const idw::parameter&,const idw::parameter&,const idw::parameter&>(args("temperature","precipitation","wind_speed","radiation","rel_hum"),"using smart IDW for temperature, typically grid inputs"))
            .def_readwrite("use_idw_for_temperature",&InterpolationParameter::use_idw_for_temperature,"if true, the IDW temperature is used instead of BTK, useful for grid-input scenarios")
            .def_readwrite("temperature",&InterpolationParameter::temperature,"BTK for temperature (in case .use_idw_for_temperature is false)")
            .def_readwrite("temperature_idw",&InterpolationParameter::temperature_idw,"IDW for temperature(in case .use_idw_for_temperature is true)")
            .def_readwrite("precipitation",&InterpolationParameter::precipitation,"IDW parameters for precipitation")
            .def_readwrite("wind_speed", &InterpolationParameter::wind_speed,"IDW parameters for wind_speed")
            .def_readwrite("radiation", &InterpolationParameter::radiation,"IDW parameters for radiation")
            .def_readwrite("rel_hum",&InterpolationParameter::rel_hum,"IDW parameters for relative humidity")
            ;
    }

	void interpolation() {
        idw_interpolation();
        btk_interpolation();
        interpolation_parameter();
        ok_kriging();
    }
}
