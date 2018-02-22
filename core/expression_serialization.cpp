#include "expression_serialization.h"
#include "core_archive.h"


/** from  google how to serialize tuple using a straight forward tuple expansion */
namespace boost { namespace serialization {

    /** generic recursive template version of tuple_serialize */
    template<int N>
    struct tuple_serialize {
        template<class Archive, typename ...tuple_types>
        static void serialize(Archive& ar, std::tuple<tuple_types...>& t, const unsigned version) {
            ar & std::get<N - 1>(t);
            tuple_serialize<N - 1>::serialize(ar, t, version); // recursive expand/iterate over members
        }
    };
    /** specialize recurisive template expansion termination at 0 args */
    template<>
    struct tuple_serialize<0> {
        template<class Archive, typename ...tuple_types>
        static void serialize(Archive&, std::tuple<tuple_types...>&, const unsigned) {
            ;// zero elements to serialize, noop, btw: should we instead stop at <1>??
        }
    };

    template<class Archive, typename ...tuple_types>
    void serialize(Archive& ar, std::tuple<tuple_types...>& t, const unsigned version) {
        tuple_serialize<sizeof ...(tuple_types)>::serialize(ar, t, version);
    }


}} // boost.serialization




template < class ...srep_types>
template <class Archive>
void shyft::time_series::dd::ts_expression<srep_types...>::serialize(Archive& ar, const unsigned int /*version*/) {
    ar & ts_reps & roots; // we *could* use for_each tuple here
                          // we could just do ar  & rts & gts;
                          // but at least for the rts, serializing only strings means 2-3 times faster
                          // serialization step.
    if (Archive::is_loading::value) {
        size_t n;
        ar & n;
        rts.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            auto ts = new aref_ts();
            bool has_real_ts = false;
            ar & ts->id &has_real_ts;
            if (has_real_ts) {
                auto gts = make_shared<gpoint_ts>();
                ar & gts->rep;
                ts->rep = gts;
            }
            rts.push_back(ts);
        }
        ar & n;
        gts.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            auto ts = new gpoint_ts();
            ar & ts->rep;
            gts.push_back(ts);
        }
    } else { // saving
        size_t n = rts.size();
        ar & n;
        for (size_t i = 0; i < n; ++i) {
            ar & rts[i]->id;
            bool has_real_ts = !rts[i]->needs_bind();
            ar & has_real_ts;
            if (has_real_ts) {
                ar & rts[i]->rep->rep;
            }
        }
        n = gts.size();
        ar & n;
        for (size_t i = 0; i < n; ++i) {
            ar & gts[i]->rep;
        }
    }
}

template<class Archive>
void shyft::time_series::dd::srep::saverage_ts::serialize(Archive &ar, const unsigned /*version*/) { ar & ts &ta; }

template<class Archive>
void shyft::time_series::dd::srep::sintegral_ts::serialize(Archive &ar, const unsigned /*version*/) { ar & ts &ta; }

template<class Archive>
void shyft::time_series::dd::srep::saccumulate_ts::serialize(Archive &ar, const unsigned /*version*/) { ar & ts &ta; }

template<class Archive>
void shyft::time_series::dd::srep::speriodic_ts::serialize(Archive &ar, const unsigned /*version*/) { ar & ts; }

template<class Archive>
void shyft::time_series::dd::srep::sconvolve_w_ts::serialize(Archive &ar, const unsigned /*version*/) { ar & ts & w & policy; }

template<class Archive>
void shyft::time_series::dd::srep::srating_curve_ts::serialize(Archive &ar, const unsigned /*version*/) { ar & ts & rc_param; }

template<class Archive>
void shyft::time_series::dd::srep::skrls_interpolation_ts::serialize(Archive &ar, const unsigned /*version*/) { ar & ts & predictor; }

x_serialize_implement(shyft::time_series::dd::srep::saverage_ts);
x_serialize_implement(shyft::time_series::dd::srep::sintegral_ts);
x_serialize_implement(shyft::time_series::dd::srep::saccumulate_ts);
x_serialize_implement(shyft::time_series::dd::srep::speriodic_ts);
x_serialize_implement(shyft::time_series::dd::srep::sconvolve_w_ts);
x_serialize_implement(shyft::time_series::dd::srep::srating_curve_ts);
x_serialize_implement(shyft::time_series::dd::srep::skrls_interpolation_ts);
x_serialize_implement(shyft::time_series::dd::compressed_ts_expression);

using shyft::core::core_oarchive;
using shyft::core::core_iarchive;

x_serialize_archive(shyft::time_series::dd::compressed_ts_expression, core_oarchive, core_iarchive);
x_serialize_archive(shyft::time_series::dd::srep::saverage_ts, core_oarchive, core_iarchive);
x_serialize_archive(shyft::time_series::dd::srep::sintegral_ts, core_oarchive, core_iarchive);
x_serialize_archive(shyft::time_series::dd::srep::saccumulate_ts, core_oarchive, core_iarchive);
x_serialize_archive(shyft::time_series::dd::srep::speriodic_ts, core_oarchive, core_iarchive);
x_serialize_archive(shyft::time_series::dd::srep::sconvolve_w_ts, core_oarchive, core_iarchive);
x_serialize_archive(shyft::time_series::dd::srep::srating_curve_ts, core_oarchive, core_iarchive);
x_serialize_archive(shyft::time_series::dd::srep::skrls_interpolation_ts, core_oarchive, core_iarchive);