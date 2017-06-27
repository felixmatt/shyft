#pragma once
namespace shyft {
    namespace dtss {
        using shyft::core::utctime;
        using shyft::core::utcperiod;
        using shyft::core::utctimespan;
        using shyft::core::no_utctime;
        /**\brief ts_info gives some data for time-series
        *
        * ts_info is the return-type for the message_type::FIND_TS request,
        * that could be useful in some contexts.
        * We do not aim for covering up for  a time-series centric system,
        * - only for a model-driven system
        *
        * data-members in the struct are mentioned in the order of significance
        * alternative: just return json-formattet string??
        */
        struct ts_info {
            //--some stuff that need to be here for the python exposure
            ts_info() = default;
            ts_info(std::string name, time_series::ts_point_fx point_fx, utctimespan delta_t, std::string olson_tz_id, utcperiod data_period, utctime created, utctime modified)
                :name(name), point_fx(point_fx), delta_t(delta_t), olson_tz_id(olson_tz_id), data_period(data_period), created(created), modified(modified) {}
            ts_info(const ts_info&) = default;
            bool operator==(const ts_info&o) const { return name == o.name && point_fx == o.point_fx && delta_t == o.delta_t && olson_tz_id == o.olson_tz_id&& data_period == o.data_period && created == o.created && modified == o.modified; }
            bool operator!=(const ts_info&o) const { return !this->operator==(o); }
            //--
            std::string name; ///< the 'unique' name of the ts, url-formatted?
            time_series::ts_point_fx point_fx = time_series::ts_point_fx::POINT_AVERAGE_VALUE; ///< how to interpret points, stair-case/linear
            utctimespan delta_t = 0L;///< time-axis, fixed delta_t, or 0 if breakpoint
            std::string olson_tz_id; ///< time-axis, empty, or if delta_t g.t. hour, 
            utcperiod data_period; ///< stored data period, if that gives meaning(not for expressions)
                                   // we could have:
            utctime created = no_utctime; ///< when ts was created, IF supported by underlying driver else no_utctime
            utctime modified = no_utctime;///< when ts was modified, IF supported by underlying driver, else no_utctime
                                          // consider std::string json_info;
            x_serialize_decl();
        };
    }
}
//-- serialization support for dtss specific message types
x_serialize_export_key(shyft::dtss::ts_info);
