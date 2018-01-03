#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wsign-compare"
#endif

#include "core_serialization.h"
#include "core_archive.h"

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/base_object.hpp>

//
// 1. first include std stuff and the headers for
// files with serializeation support
//

#include "utctime_utilities.h"
#include "time_axis.h"
#include "geo_cell_data.h"
#include "hbv_snow.h"
#include "hbv_soil.h"
#include "hbv_tank.h"
#include "gamma_snow.h"
#include "kirchner.h"
#include "skaugen.h"
#include "hbv_stack.h"
#include "pt_gs_k.h"
#include "pt_ss_k.h"
#include "pt_hs_k.h"

// then include stuff you need like vector,shared, base_obj,nvp etc.




//
// 2. Then implement each class serialization support
//


using namespace boost::serialization;
using namespace shyft::core;

//-- utctime_utilities.h

template<class Archive>
void shyft::core::utcperiod::serialize(Archive & ar, const unsigned int version) {
    ar
    & core_nvp("start", start)
    & core_nvp("end",end);
    ;
}

template<class Archive>
void shyft::core::time_zone::tz_table::serialize(Archive & ar, const unsigned int version) {
    ar
    & core_nvp("start_year", start_year)
    & core_nvp("tz_name",tz_name)
    & core_nvp("dst",dst)
    & core_nvp("dt",dt)
    ;
}

template<class Archive>
void shyft::core::time_zone::tz_info_t::serialize(Archive & ar, const unsigned int version) {
    ar
    & core_nvp("base_tz", base_tz)
    & core_nvp("tz",tz);
    ;
}

template <class Archive>
void shyft::core::calendar::serialize(Archive & ar, const unsigned int version) {
    ar
    & core_nvp("tz_info", tz_info)
    ;
}

//-- time_axis.h

template <class Archive>
void shyft::time_axis::fixed_dt::serialize(Archive & ar, const unsigned int version) {
    ar
    & core_nvp("t", t)
    & core_nvp("dt",dt)
    & core_nvp("n",n)
    ;
}

template <class Archive>
void shyft::time_axis::calendar_dt::serialize(Archive & ar, const unsigned int version) {
    ar
    & core_nvp("cal", cal)
    & core_nvp("t", t)
    & core_nvp("dt",dt)
    & core_nvp("n",n)
    ;
}

template <class Archive>
void shyft::time_axis::point_dt::serialize(Archive & ar, const unsigned int version) {
    ar
    & core_nvp("t", t)
    & core_nvp("dt",t_end)
    ;
}

template <class Archive>
void shyft::time_axis::generic_dt::serialize(Archive & ar,const unsigned int version) {
    ar
    & core_nvp("gt", gt)
    ;
    if (gt == shyft::time_axis::generic_dt::FIXED)
        ar & core_nvp("f", f);
    else if (gt == shyft::time_axis::generic_dt::CALENDAR)
        ar & core_nvp("c", c);
    else
        ar & core_nvp("p", p);
}


//-- basic geo stuff
template <class Archive>
void shyft::core::geo_point::serialize(Archive& ar, const unsigned int version) {
    ar
    & core_nvp("x",x)
    & core_nvp("y",y)
    & core_nvp("z",z)
    ;
}

template <class Archive>
void shyft::core::land_type_fractions::serialize(Archive& ar, const unsigned int version) {
    ar
    & core_nvp("glacier_",glacier_)
    & core_nvp("lake_",lake_)
    & core_nvp("reservoir_",reservoir_)
    & core_nvp("forest_",forest_)
    ;
}

template <class Archive>
void shyft::core::routing_info::serialize(Archive& ar, const unsigned int version) {
    ar
    & core_nvp("id", id)
    & core_nvp("distance", distance)
    ;
}

template <class Archive>
void shyft::core::geo_cell_data::serialize(Archive& ar, const unsigned int version) {
    ar
    & core_nvp("mid_point_",mid_point_)
    & core_nvp("area_m2",area_m2)
    & core_nvp("catchment_id_",catchment_id_)
    & core_nvp("radiation_slope_factor_",radiation_slope_factor_)
    & core_nvp("fractions",fractions)
    & core_nvp("routing",routing)
    ;
}
//-- state serialization
template <class Archive>
void shyft::core::hbv_snow::state::serialize(Archive & ar, const unsigned int file_version) {
    ar
    & core_nvp("swe",swe)
    & core_nvp("sca",sca)
    ;
}
template <class Archive>
void shyft::core::hbv_soil::state::serialize(Archive & ar, const unsigned int file_version) {
    ar
    & core_nvp("sm",sm)
    ;
}
template <class Archive>
void shyft::core::hbv_tank::state::serialize(Archive & ar, const unsigned int file_version) {
    ar
    & core_nvp("uz",uz)
    & core_nvp("lz",lz)
    ;
}
template <class Archive>
void shyft::core::kirchner::state::serialize(Archive & ar, const unsigned int file_version) {
    ar
        & core_nvp("q",q)
        ;
}
template <class Archive>
void shyft::core::gamma_snow::state::serialize(Archive & ar, const unsigned int file_version) {
    ar
        & core_nvp("albedo", albedo)
        & core_nvp("lwc", lwc)
        & core_nvp("surface_heat",surface_heat )
        & core_nvp("alpha", alpha)
        & core_nvp("sdc_melt_mean",sdc_melt_mean )
        & core_nvp("acc_melt",acc_melt )
        & core_nvp("iso_pot_energy", iso_pot_energy)
        & core_nvp("temp_swe",temp_swe )
        ;
}
template <class Archive>
void shyft::core::skaugen::state::serialize(Archive & ar, const unsigned int file_version) {
    ar
        & core_nvp("nu",nu)
        & core_nvp("alpha",alpha)
        & core_nvp("sca",sca)
        & core_nvp("swe",swe)
        & core_nvp("free_water",free_water)
        & core_nvp("residual",residual)
        & core_nvp("num_units",num_units)
        ;
}
template <class Archive>
void shyft::core::hbv_stack::state::serialize(Archive & ar, const unsigned int file_version) {
    ar
        & core_nvp("snow",snow)
        & core_nvp("soil",soil)
        & core_nvp("tank",tank)
        ;
}
template <class Archive>
void shyft::core::pt_gs_k::state::serialize(Archive & ar, const unsigned int file_version) {
    ar
        & core_nvp("gs",gs)
        & core_nvp("kirchner",kirchner)
        ;
}
template <class Archive>
void shyft::core::pt_ss_k::state::serialize(Archive & ar, const unsigned int file_version) {
    ar
        & core_nvp("snow",snow)
        & core_nvp("kirchner", kirchner)
        ;
}
template <class Archive>
void shyft::core::pt_hs_k::state::serialize(Archive & ar, const unsigned int file_version) {
    ar
        & core_nvp("snow",snow)
        & core_nvp("kirchner", kirchner)
        ;
}


//-- export geo stuff
x_serialize_implement(shyft::core::geo_point);
x_serialize_implement(shyft::core::land_type_fractions);
x_serialize_implement(shyft::core::routing_info);
x_serialize_implement(shyft::core::geo_cell_data);

//-- export utctime_utilities
x_serialize_implement(shyft::core::utcperiod);
x_serialize_implement(shyft::core::time_zone::tz_info_t);
x_serialize_implement(shyft::core::time_zone::tz_table);
x_serialize_implement(shyft::core::calendar);

//-- export time-axis
x_serialize_implement(shyft::time_axis::fixed_dt);
x_serialize_implement(shyft::time_axis::calendar_dt);
x_serialize_implement(shyft::time_axis::point_dt);
x_serialize_implement(shyft::time_axis::generic_dt);

//-- export core time-series (except binary-ops)


//-- export method and method-stack state
x_serialize_implement(shyft::core::hbv_snow::state);
x_serialize_implement(shyft::core::hbv_soil::state);
x_serialize_implement(shyft::core::hbv_tank::state);
x_serialize_implement(shyft::core::gamma_snow::state);
x_serialize_implement(shyft::core::skaugen::state);
x_serialize_implement(shyft::core::kirchner::state);

x_serialize_implement(shyft::core::pt_gs_k::state);
x_serialize_implement(shyft::core::pt_hs_k::state);
x_serialize_implement(shyft::core::pt_ss_k::state);
x_serialize_implement(shyft::core::hbv_stack::state);

//-- export predictors

//
// 4. Then include the archive supported
//
// repeat template instance for each archive class

x_arch(shyft::core::utcperiod);
x_arch(shyft::core::time_zone::tz_info_t);
x_arch(shyft::core::time_zone::tz_table);
x_arch(shyft::core::calendar);

x_arch(shyft::time_axis::fixed_dt);
x_arch(shyft::time_axis::calendar_dt);
x_arch(shyft::time_axis::point_dt);
x_arch(shyft::time_axis::generic_dt);


x_arch(shyft::core::geo_point);
x_arch(shyft::core::land_type_fractions);
x_arch(shyft::core::routing_info);
x_arch(shyft::core::geo_cell_data);

x_arch(shyft::core::hbv_snow::state);
x_arch(shyft::core::hbv_soil::state);
x_arch(shyft::core::hbv_tank::state);
x_arch(shyft::core::gamma_snow::state);
x_arch(shyft::core::skaugen::state);
x_arch(shyft::core::kirchner::state);

x_arch(shyft::core::pt_gs_k::state);
x_arch(shyft::core::pt_hs_k::state);
x_arch(shyft::core::pt_ss_k::state);
x_arch(shyft::core::hbv_stack::state);

