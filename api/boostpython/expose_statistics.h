#pragma once

namespace expose {
    namespace statistics {
        typedef shyft::api::result_ts_t_ rts_;
        typedef std::vector<double> vd_;
        typedef const std::vector<int>& cids_;
        typedef size_t ix_;
        using namespace boost::python;

        template<class cell>
        static void kirchner(const char *cell_name) {
            char state_name[200];sprintf(state_name,"%sKirchnerStateStatistics",cell_name);
            typedef typename shyft::api::kirchner_cell_state_statistics<cell>    sc_stat;

            rts_ (sc_stat::*discharge_ts)(cids_) const = &sc_stat::discharge;
            vd_  (sc_stat::*discharge_vd)(cids_,ix_) const =&sc_stat::discharge;
            class_<sc_stat>(state_name,"Kirchner response statistics",no_init)
                .def(init<std::shared_ptr<std::vector<cell>> >(args("cells"),"construct Kirchner cell response statistics object"))
                .def("discharge",discharge_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
                .def("discharge",discharge_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
            ;
        }

        template <class cell>
        static void priestley_taylor(const char *cell_name) {
            char response_name[200];sprintf(response_name,"%sPriestleyTaylorResponseStatistics",cell_name);
            typedef typename shyft::api::priestley_taylor_cell_response_statistics<cell> rc_stat;

            rts_ (rc_stat::*output_ts)(cids_) const = &rc_stat::output;
            vd_  (rc_stat::*output_vd)(cids_,ix_) const =&rc_stat::output;
            class_<rc_stat>(response_name,"PriestleyTaylor response statistics",no_init)
                .def(init<std::shared_ptr<std::vector<cell>> >(args("cells"),"construct PriestleyTaylor cell response statistics object"))
                .def("output",output_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
                .def("output",output_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
            ;
        }

        template <class cell>
        static void actual_evapotranspiration(const char *cell_name) {
            char response_name[200];sprintf(response_name,"%sActualEvapotranspirationResponseStatistics",cell_name);
            typedef typename shyft::api::actual_evapotranspiration_cell_response_statistics<cell> rc_stat;

            rts_ (rc_stat::*output_ts)(cids_) const = &rc_stat::output;
            vd_  (rc_stat::*output_vd)(cids_,ix_) const =&rc_stat::output;
            class_<rc_stat>(response_name,"ActualEvapotranspiration response statistics",no_init)
                .def(init<std::shared_ptr<std::vector<cell>> >(args("cells"),"construct ActualEvapotranspiration cell response statistics object"))
                .def("output",output_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
                .def("output",output_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
            ;
        }

        template <class cell>
        static void gamma_snow(const char *cell_name) {
            char state_name[200];sprintf(state_name,"%sGammaSnowStateStatistics",cell_name);
            char response_name[200];sprintf(response_name,"%sGammaSnowResponseStatistics",cell_name);
            typedef typename shyft::api::gamma_snow_cell_state_statistics<cell>    sc_stat;
            typedef typename shyft::api::gamma_snow_cell_response_statistics<cell> rc_stat;

            rts_ (sc_stat::*albedo_ts)(cids_) const = &sc_stat::albedo;
            vd_  (sc_stat::*albedo_vd)(cids_,ix_) const =&sc_stat::albedo;

            rts_ (sc_stat::*lwc_ts)(cids_) const = &sc_stat::lwc;
            vd_  (sc_stat::*lwc_vd)(cids_,ix_) const =&sc_stat::lwc;

            rts_ (sc_stat::*surface_heat_ts)(cids_) const = &sc_stat::surface_heat;
            vd_  (sc_stat::*surface_heat_vd)(cids_,ix_) const =&sc_stat::surface_heat;

            rts_ (sc_stat::*alpha_ts)(cids_) const = &sc_stat::alpha;
            vd_  (sc_stat::*alpha_vd)(cids_,ix_) const =&sc_stat::alpha;

            rts_ (sc_stat::*sdc_melt_mean_ts)(cids_) const = &sc_stat::sdc_melt_mean;
            vd_  (sc_stat::*sdc_melt_mean_vd)(cids_,ix_) const =&sc_stat::sdc_melt_mean;

            rts_ (sc_stat::*acc_melt_ts)(cids_) const = &sc_stat::acc_melt;
            vd_  (sc_stat::*acc_melt_vd)(cids_,ix_) const =&sc_stat::acc_melt;

            rts_ (sc_stat::*iso_pot_energy_ts)(cids_) const = &sc_stat::iso_pot_energy;
            vd_  (sc_stat::*iso_pot_energy_vd)(cids_,ix_) const =&sc_stat::iso_pot_energy;

            rts_ (sc_stat::*temp_swe_ts)(cids_) const = &sc_stat::temp_swe;
            vd_  (sc_stat::*temp_swe_vd)(cids_,ix_) const =&sc_stat::temp_swe;

            class_<sc_stat>(state_name,"GammaSnow state statistics",no_init)
                .def(init<std::shared_ptr<std::vector<cell>> >(args("cells"),"construct GammaSnow cell state statistics object"))
                .def("albedo",albedo_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
                .def("albedo",albedo_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
                .def("lwc",lwc_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
                .def("lwc",lwc_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
                .def("surface_heat",surface_heat_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
                .def("surface_heat",surface_heat_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
                .def("alpha",alpha_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
                .def("alpha",alpha_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
                .def("sdc_melt_mean",sdc_melt_mean_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
                .def("sdc_melt_mean",sdc_melt_mean_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
                .def("acc_melt",acc_melt_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
                .def("acc_melt",acc_melt_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
                .def("iso_pot_energy",iso_pot_energy_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
                .def("iso_pot_energy",iso_pot_energy_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
                .def("temp_swe",temp_swe_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
                .def("temp_swe",temp_swe_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
            ;


            rts_ (rc_stat::*sca_ts)(cids_) const = &rc_stat::sca;
            vd_  (rc_stat::*sca_vd)(cids_,ix_) const =&rc_stat::sca;

            rts_ (rc_stat::*swe_ts)(cids_) const = &rc_stat::swe;
            vd_  (rc_stat::*swe_vd)(cids_,ix_) const =&rc_stat::swe;

            rts_ (rc_stat::*outflow_ts)(cids_) const = &rc_stat::outflow;
            vd_  (rc_stat::*outflow_vd)(cids_,ix_) const =&rc_stat::outflow;

            class_<rc_stat>(response_name,"GammaSnow response statistics",no_init)
                .def(init<std::shared_ptr<std::vector<cell>> >(args("cells"),"construct GammaSnow cell response statistics object"))
                .def("outflow",outflow_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
                .def("outflow",outflow_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
                .def("swe",swe_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
                .def("swe",swe_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
                .def("sca",sca_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
                .def("sca",sca_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")

                ;

        }

        template <class cell>
        static void hbv_snow(const char *cell_name) {
            char state_name[200];sprintf(state_name,"%sHBVSnowStateStatistics",cell_name);
            char response_name[200];sprintf(response_name,"%sHBVSnowResponseStatistics",cell_name);
            typedef typename shyft::api::hbv_snow_cell_state_statistics<cell>    sc_stat;
            typedef typename shyft::api::hbv_snow_cell_response_statistics<cell> rc_stat;

            rts_ (sc_stat::*swe_ts)(cids_) const = &sc_stat::swe;
            vd_  (sc_stat::*swe_vd)(cids_,ix_) const =&sc_stat::swe;
            rts_ (sc_stat::*sca_ts)(cids_) const = &sc_stat::sca;
            vd_  (sc_stat::*sca_vd)(cids_,ix_) const =&sc_stat::sca;

            class_<sc_stat>(state_name,"HBVSnow state statistics",no_init)
                .def(init<std::shared_ptr<std::vector<cell>> >(args("cells"),"construct HBVSnow cell state statistics object"))
                .def("swe",swe_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
                .def("swe",swe_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
                .def("sca",sca_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
                .def("sca",sca_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
            ;


            rts_ (rc_stat::*outflow_ts)(cids_) const = &rc_stat::outflow;
            vd_  (rc_stat::*outflow_vd)(cids_,ix_) const =&rc_stat::outflow;

            class_<rc_stat>(response_name,"HBVSnow response statistics",no_init)
                .def(init<std::shared_ptr<std::vector<cell>> >(args("cells"),"construct HBVSnow cell response statistics object"))
                .def("outflow",outflow_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
                .def("outflow",outflow_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
            ;
        }


        template <class cell>
        static void basic_cell(const char *cell_name) {
            char base_name[200];sprintf(base_name,"%sStatistics",cell_name);
            typedef typename shyft::api::basic_cell_statistics<cell> bc_stat;

            rts_ (bc_stat::*discharge_ts)(cids_) const = &bc_stat::discharge;
            vd_  (bc_stat::*discharge_vd)(cids_,ix_) const =&bc_stat::discharge;

            rts_ (bc_stat::*temperature_ts)(cids_) const = &bc_stat::temperature;
            vd_  (bc_stat::*temperature_vd)(cids_,ix_) const =&bc_stat::temperature;

            rts_ (bc_stat::*radiation_ts)(cids_) const = &bc_stat::radiation;
            vd_  (bc_stat::*radiation_vd)(cids_,ix_) const =&bc_stat::radiation;

            rts_ (bc_stat::*wind_speed_ts)(cids_) const = &bc_stat::wind_speed;
            vd_  (bc_stat::*wind_speed_vd)(cids_,ix_) const =&bc_stat::wind_speed;

            rts_ (bc_stat::*rel_hum_ts)(cids_) const = &bc_stat::rel_hum;
            vd_  (bc_stat::*rel_hum_vd)(cids_,ix_) const =&bc_stat::rel_hum;

            rts_ (bc_stat::*precipitation_ts)(cids_) const = &bc_stat::precipitation;
            vd_  (bc_stat::*precipitation_vd)(cids_,ix_) const =&bc_stat::precipitation;



            class_<bc_stat>(base_name,"provides statistics for cell environment plus mandatory discharge",no_init)
                .def(init<std::shared_ptr<std::vector<cell>> >(args("cells"),"construct basic cell statistics object"))
                .def("discharge",discharge_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
                .def("discharge",discharge_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
                .def("temperature",temperature_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
                .def("temperature",temperature_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
                .def("precipitation",precipitation_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
                .def("precipitation",precipitation_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
                .def("radiation",radiation_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
                .def("radiation",radiation_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
                .def("wind_speed",wind_speed_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
                .def("wind_speed",wind_speed_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
                .def("rel_hum",rel_hum_ts,args("catchment_indexes"), "returns sum  for catcment_ids")
                .def("rel_hum",rel_hum_vd,args("catchment_indexes","i"),"returns  for cells matching catchments_ids at the i'th timestep")
                ;
        }
    }
}
