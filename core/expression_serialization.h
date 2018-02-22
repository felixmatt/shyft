#pragma once
#include <unordered_map>
#include <future>
#include <utility>
#include <tuple>

#include <boost/variant.hpp>

#include "utctime_utilities.h"
#include "time_axis.h"
#include "time_series_dd.h"

#include "core_serialization.h"


//-- notice that boost serialization require us to
//   include shared_ptr/vector .. etc.. wherever it's needed

#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/variant.hpp>

namespace cxx_ext {
    /** from google, https://codereview.stackexchange.com/questions/51407/stdtuple-foreach-implementation
    * found tuple building blocks to enable for_each(tuple t,F&&fx)
    * Should be a part of std.lib or boost ??
    */

    template <typename Tuple, typename F, std::size_t ...Indices>
    void for_each_impl(Tuple&& tuple, F&& f, std::index_sequence<Indices...>) {
        using swallow = int[];
        (void)swallow {
            1,
                (f(std::get<Indices>(std::forward<Tuple>(tuple))), void(), int{})...
        };
    }

    template <typename Tuple, typename F>
    void for_each(Tuple&& tuple, F&& f) {
        constexpr std::size_t N = std::tuple_size<std::remove_reference_t<Tuple>>::value;
        for_each_impl(std::forward<Tuple>(tuple), std::forward<F>(f),
                      std::make_index_sequence<N>{});
    }
}



// experiment with expression serialization
// through alternate representation.

namespace shyft { namespace time_series { namespace dd {

    using std::vector;
    using std::size_t;
    using std::future;
    using std::unordered_map;
    using std::tuple;
    using ::boost::variant;
    using ::boost::apply_visitor;

    /** Steps to add a new ts to the expression serializer
    *  1. add it to a_index variant , o_index< new_ts_type> (also add x_binary_serializable(o_index<new_ts_type>) at end of sections
    *  2. add new  namespace srep member, struct snew_ts_type { meat here}; template<> struct _type<new_ts_type> {using rep_t= new_ts_type;}
    *     also either add x_binary_serializable (if possible),
    *     or use standard x_serialize_decl(), x_serialize_export_key(cls) in header; then in impl. x_serialize_implement() and x_arch(..)
    *
    *  3. ts_expr_converter, add if-then-else for your class (similar contents as for other, how to create srep::snew_ts_type from new_ts_type
    *
    *  4. in ts_expr_deserialize_visitor, add your make(...) method to construct new_ts_type from it serializable type snew_ts_type.
    */


    /**The o_index<T> is a typed index, an int for a specific type T
    * we utilize the type T :
    *  a) to have a binary serializable element representing a node in the expression tree
    *  b) to drive the restore of expression in it's serialized form to it's tree representation.
    */
    template<class E> // typed index, bitwise serializable
    struct o_index {
        using ts_t = E;///< provide the type of the real ts-type this o_index is representing, needed to keep reverse-lookup to type E
        size_t value;///< index value, used for lookup into the expression tables (along with type T that help us find the right table in the tuple)
        operator size_t() const { return value; }
        bool operator==(const o_index& o) const { return value == o.value; }
    };

    /** a_index is a typed variant of o_index<T> that is the
    * binary serializable 'alias' for reference/pointer in the expression tree.
    */
    using a_index = variant< // all member serializable, but is variant
        boost::blank, // represent null, a.which()==0
        o_index<abin_op_ts>,// binop
        o_index<abin_op_scalar_ts>,
        o_index<abin_op_ts_scalar>,
        o_index<gpoint_ts>,// terminal
        o_index<aref_ts>, // terminal, possibly unbound, or bound with a ref to gpoints_ts(terminal)
        o_index<abs_ts>,
        o_index<average_ts>,
        o_index<integral_ts>,
        o_index<accumulate_ts>,
        o_index<time_shift_ts>,
        o_index<periodic_ts>,
        o_index<convolve_w_ts>,
        o_index<extend_ts>,
        o_index<rating_curve_ts>,
        o_index<ice_packing_ts>,
        o_index<ice_packing_recession_ts>,
        o_index<krls_interpolation_ts>,
        o_index<qac_ts>
    >;

    namespace srep {
        //
        // srep=serialize_representation of something,
        // naming convention: s<dd::ts-type> for all internal types here.
        //
        // note that all types declared there is 'boiler-plate' code, 
        // that is fast binary serializable (if possible/practical)
        // we get one for each ipoint_ts type
        // except for the terminals
        // we do not serialize state that can
        // be computed, or is computed as part
        // of the constructor
        // remember: create a srep::_type<T> mapping if you add more elements
        // serialization: if it's binary serialized, remember to mark at such using x_binary_serializable(cls..)

        /** The srep::_type<T> class helps us with type-lookup
        * we need the type look up, given a dd:ts-type, give the type of the srep::s<ts-type>
        * For all s<dd::ts-type> there must be a _type<dd::ts-type>::rep_t
        */
        template <class T>
        struct _type { /*using rep_t=void;*/ }; // a static assert here ? like your type T lacks declarative mapping to its namespace srep type ?

        struct sbinop_op_ts {
            using ts_t = abin_op_ts;
            iop_t op; // + ..
            a_index lhs, rhs;
            bool operator==(const sbinop_op_ts& o) const { return op == o.op && lhs == o.lhs && rhs == o.rhs; }
        };
        template<> struct _type<abin_op_ts> { using rep_t = srep::sbinop_op_ts; };

        struct sbinop_ts_scalar {
            using ts_t = abin_op_ts_scalar;
            iop_t op; // + ..
            a_index lhs;
            double rhs;
            bool operator==(const sbinop_ts_scalar& o) const { return op == o.op && lhs == o.lhs && rhs == o.rhs; }
        };
        template<> struct _type<abin_op_ts_scalar> { using rep_t = srep::sbinop_ts_scalar; };

        struct sbin_op_scalar_ts {
            using ts_t = abin_op_scalar_ts;
            iop_t op; // + ..
            double lhs;
            a_index rhs;
            bool operator==(const sbin_op_scalar_ts& o) const { return op == o.op && lhs == o.lhs && rhs == o.rhs; }
        };
        template<> struct _type<abin_op_scalar_ts> { using rep_t = srep::sbin_op_scalar_ts; };

        struct sabs_ts {
            using ts_t = abs_ts;
            a_index ts;
            bool operator==(const sabs_ts& o) const { return ts == o.ts; }
        };
        template<> struct _type<abs_ts> { using rep_t = srep::sabs_ts; };

        struct saverage_ts {
            using ts_t = average_ts;
            a_index ts;
            gta_t ta;
            bool operator==(const saverage_ts& o) const { return ts == o.ts && ta == o.ta; }
            x_serialize_decl();// this class needs to serialize time-axis,
        };
        template<> struct _type<average_ts> { using rep_t = srep::saverage_ts; };

        struct sintegral_ts {
            using ts_t = integral_ts;
            a_index ts;
            gta_t ta;
            bool operator==(const sintegral_ts& o) const { return ts == o.ts && ta == o.ta; }
            x_serialize_decl();// this class needs to serialize time-axis,
        };
        template<> struct _type<integral_ts> { using rep_t = srep::sintegral_ts; };

        struct saccumulate_ts {
            using ts_t = accumulate_ts;
            a_index ts;
            gta_t ta;
            bool operator==(const saccumulate_ts& o) const { return ts == o.ts && ta == o.ta; }
            x_serialize_decl();// this class needs to serialize time-axis,
        };
        template<> struct _type<accumulate_ts> { using rep_t = srep::saccumulate_ts; };

        struct stime_shift_ts {
            using ts_t = time_shift_ts;
            a_index ts;
            utctimespan dt;
            bool operator==(const stime_shift_ts& o) const { return ts == o.ts && dt == o.dt; }
        };
        template<> struct _type<time_shift_ts> { using rep_t = srep::stime_shift_ts; };

        struct speriodic_ts {
            using ts_t = periodic_ts;
            periodic_ts::pts_t ts;
            bool operator==(const speriodic_ts& o) const { return ts == o.ts; }
            x_serialize_decl();// this class needs to serialize time-axis,
        };
        template<> struct _type<periodic_ts> { using rep_t = srep::speriodic_ts; };

        struct sconvolve_w_ts {
            using ts_t = convolve_w_ts;
            a_index ts;
            vector<double> w;
            time_series::convolve_policy policy;
            bool operator==(const sconvolve_w_ts& o) const { return ts == o.ts && w == o.w && policy == o.policy; }
            x_serialize_decl();// this class needs to serialize time-axis,
        };
        template<> struct _type<convolve_w_ts> { using rep_t = srep::sconvolve_w_ts; };

        struct sextend_ts {
            using ts_t = extend_ts;
            a_index lhs;
            a_index rhs;
            extend_ts_split_policy ets_split_p;
            utctime split_at;
            extend_ts_fill_policy ets_fill_p;
            double fill_value;
            bool operator==(const sextend_ts& o) const {
                return lhs == o.lhs && rhs == o.rhs && ets_split_p == o.ets_split_p && split_at == o.split_at && ets_fill_p == o.ets_fill_p
                    && ((std::isfinite(fill_value) && std::isfinite(o.fill_value)) || (fill_value == o.fill_value));
            }
        };
        template<> struct _type<extend_ts> { using rep_t = srep::sextend_ts; };

        struct srating_curve_ts {
            using ts_t = rating_curve_ts;
            a_index ts;
            rating_curve_parameters rc_param;
            bool operator==(const srating_curve_ts& o) const { return ts == o.ts; } //TODO rc_param.equal(o.rc_param)
            x_serialize_decl();// needed because of rc_param
        };
        template<> struct _type<rating_curve_ts> { using rep_t = srep::srating_curve_ts; };

        struct sice_packing_ts {
            using ts_t = ice_packing_ts;
            a_index ts;
            ice_packing_parameters ip_param;
            ice_packing_temperature_policy ipt_policy;
            bool operator==(const sice_packing_ts& o) const {
                return ts == o.ts && ip_param == o.ip_param && ipt_policy == o.ipt_policy;
            }
            x_serialize_decl();// needed because of ip_param
        };
        template<> struct _type<ice_packing_ts> { using rep_t = srep::sice_packing_ts; };

        struct sice_packing_recession_ts {
            using ts_t = ice_packing_recession_ts;
            a_index flow_ts;
            a_index ip_ts;
            ice_packing_recession_parameters ipr_param;
            bool operator==(const sice_packing_recession_ts& o) const {
                return flow_ts == o.flow_ts && ip_ts == o.ip_ts && ipr_param == o.ipr_param;
            }
            x_serialize_decl();// needed because of ipr_param
        };
        template<> struct _type<ice_packing_recession_ts> { using rep_t = srep::sice_packing_recession_ts; };

        struct skrls_interpolation_ts {
            using ts_t = krls_interpolation_ts;
            a_index ts;
            krls_interpolation_ts::krls_p predictor;
            bool operator==(const sconvolve_w_ts& o) const { return ts == o.ts; } //TODO predictor.equal(o.predictor)
            x_serialize_decl();// needed because of predictor
        };
        template<> struct _type<krls_interpolation_ts> { using rep_t = srep::skrls_interpolation_ts; };

        struct sqac_ts {
            using ts_t = qac_ts;
            a_index ts;
            a_index cts;
            qac_parameter p;
            bool operator==(const sqac_ts& o) const { return ts == o.ts && cts == o.cts && p.equal(o.p, 1e-10); } //
        };
        template<> struct _type<qac_ts> { using rep_t = srep::sqac_ts; };

    } // namespace srep


        /** This class represents one ore more expressions, as formulated by
        * vector<apoint_ts>, where
        * each item in the vector have a corresponding item in the
        * vector<a_index> roots.
        * The sole purposes of this class is to speedup boost serialization/deserialization of
        * large expressions. One of the hotspots there is the (needed) reference tracking of all
        * the internal nodes in the expression tree.
        * Current test shows 3x speedups
        */
    template <class ...srep_types> // of type srep::binop etc.
    struct ts_expression {
        tuple<vector<srep_types>...> ts_reps; // serialization, require tuple  supported.
        vector<gpoint_ts*> gts; //no ownership
        vector<aref_ts*> rts; // no ownership
        vector<a_index> roots;// serialize

                                // to ease using and building ts_expr_rep, resolve tuple
        template<class T>
        const auto &at(o_index<T> i) const {
            return get<vector<typename srep::_type<T>::rep_t>>(ts_reps)[i];
        }
        const auto& at(o_index<aref_ts> i) const { return rts[i]; }
        const auto& at(o_index<gpoint_ts> i)const { return gts[i]; }

        size_t append(gpoint_ts *g) { gts.push_back(g); return gts.size() - 1; }
        size_t append(aref_ts *r) { rts.push_back(r); return rts.size() - 1; }

        template<class T>
        size_t append(const T& o) {
            get<vector<T>>(ts_reps).push_back(o); // get<T> resolves or fails compiletime!
            return get<vector<T>>(ts_reps).size() - 1;
        }

        //--- serialization support, major goal is speed
        x_serialize_decl();
    };


    /** this class converts vector<apoint_ts> into a ts_expression<...>
    * that provides 3xfaster serialization (taking the conversion-overhead into account).
    *
    */
    template<class ...srep_types>
    struct ts_expression_compressor {
    private:
        tuple< unordered_map<typename srep_types::ts_t*, o_index<typename srep_types::ts_t>> ... > ts_maps;
        unordered_map<gpoint_ts*, o_index<gpoint_ts>> gts_map; // terminal gpoint_ts is specially handled
        unordered_map<aref_ts *, o_index<aref_ts>> rts_map;// terminal aref_ts is also specially handled
        ts_expression<srep_types...> expr;

        /** given type of ts T, return the corresponding unordered_map from the tuple ts_map*/
        template<class T>
        auto & ts_map(const T*) {
            // tuple type lookup, looking for map of T* to its o_index<srep<T>> type         .. just ordinary find
            return get< unordered_map<T*, o_index<T>>  >(ts_maps);
        }

        /**recursive converter, that descends apoint_ts and building up the
        * internal expr type, along with the temporary ts_maps that ensures
        * we reference same object with same o_index.
        * By using dynamic_cast and inspecting the type,
        *  and then replacing all internal nodes in the expression
        * into a few binary-serializable vectors (kept within the expr variable).
        *
        */
        a_index convert(const apoint_ts &ats) {
            // nned to handle null ts
            if (!ats.ts)
                return boost::blank();// only for qac_ts that have optional .cts (correction ts)
            #define _m_find_ts_map(ts) auto &m=ts_map(ts);auto f=m.find(ts);if(f!=end(m))return f->second
            if (auto ts = dynamic_cast<abin_op_ts*>(ats.ts.get())) {
                _m_find_ts_map(ts);
                return m[ts] = o_index<abin_op_ts>{ expr.append(srep::_type<abin_op_ts>::rep_t{ ts->op,convert(ts->lhs),convert(ts->rhs) }) };
            } else if (auto ts = dynamic_cast<abin_op_ts_scalar*>(ats.ts.get())) {
                _m_find_ts_map(ts);
                return m[ts] = o_index<abin_op_ts_scalar>{ expr.append(srep::_type<abin_op_ts_scalar>::rep_t{ ts->op,convert(ts->lhs),ts->rhs }) };
            } else if (auto ts = dynamic_cast<abin_op_scalar_ts*>(ats.ts.get())) {
                _m_find_ts_map(ts);
                return m[ts] = o_index<abin_op_scalar_ts>{ expr.append(srep::_type<abin_op_scalar_ts>::rep_t{ ts->op,ts->lhs,convert(ts->rhs) }) };
            } else if (auto ts = dynamic_cast<abs_ts*>(ats.ts.get())) {
                _m_find_ts_map(ts);
                return m[ts] = o_index<abs_ts>{ expr.append(srep::_type<abs_ts>::rep_t{ convert(apoint_ts(ts->ts)) }) };
            } else if (auto ts = dynamic_cast<average_ts*>(ats.ts.get())) {
                _m_find_ts_map(ts);
                return m[ts] = o_index<average_ts>{ expr.append(srep::_type<average_ts>::rep_t{ convert(apoint_ts(ts->ts)),ts->ta }) };
            } else if (auto ts = dynamic_cast<integral_ts*>(ats.ts.get())) {
                _m_find_ts_map(ts);
                return m[ts] = o_index<integral_ts>{ expr.append(srep::_type<integral_ts>::rep_t{ convert(apoint_ts(ts->ts)),ts->ta }) };
            } else if (auto ts = dynamic_cast<accumulate_ts*>(ats.ts.get())) {
                _m_find_ts_map(ts);
                return m[ts] = o_index<accumulate_ts>{ expr.append(srep::_type<accumulate_ts>::rep_t{ convert(apoint_ts(ts->ts)),ts->ta }) };
            } else if (auto ts = dynamic_cast<time_shift_ts*>(ats.ts.get())) {
                _m_find_ts_map(ts);
                return m[ts] = o_index<time_shift_ts>{ expr.append(srep::_type<time_shift_ts>::rep_t{ convert(apoint_ts(ts->ts)),ts->dt }) };
            } else if (auto ts = dynamic_cast<periodic_ts*>(ats.ts.get())) {
                _m_find_ts_map(ts);
                return m[ts] = o_index<periodic_ts>{ expr.append(srep::_type<periodic_ts>::rep_t{ ts->ts }) };
            } else if (auto ts = dynamic_cast<convolve_w_ts*>(ats.ts.get())) {
                _m_find_ts_map(ts);
                return m[ts] = o_index<convolve_w_ts>{ expr.append(srep::_type<convolve_w_ts>::rep_t{ convert(ts->ts_impl.ts),ts->ts_impl.w,ts->ts_impl.policy }) };
            } else if (auto ts = dynamic_cast<extend_ts*>(ats.ts.get())) {
                _m_find_ts_map(ts);
                return m[ts] = o_index<extend_ts>{ expr.append(srep::_type<extend_ts>::rep_t{ convert(ts->lhs),convert(ts->rhs),ts->ets_split_p,ts->split_at,ts->ets_fill_p,ts->fill_value }) };
            } else if (auto ts = dynamic_cast<rating_curve_ts*>(ats.ts.get())) {
                _m_find_ts_map(ts);
                return m[ts] = o_index<rating_curve_ts>{ expr.append(srep::_type<rating_curve_ts>::rep_t{ convert(ts->ts.level_ts),ts->ts.rc_param }) };
            } else if (auto ts = dynamic_cast<ice_packing_ts*>(ats.ts.get())) {
                _m_find_ts_map(ts);
                return m[ts] = o_index<ice_packing_ts>{ expr.append(srep::_type<ice_packing_ts>::rep_t{ convert(ts->ts.temp_ts), ts->ts.ip_param, ts->ts.ipt_policy }) };
            } else if (auto ts = dynamic_cast<ice_packing_recession_ts*>(ats.ts.get())) {
                _m_find_ts_map(ts);
                return m[ts] = o_index<ice_packing_recession_ts>{ expr.append(srep::_type<ice_packing_recession_ts>::rep_t{ convert(ts->flow_ts), convert(ts->ice_packing_ts), ts->ipr_param }) };
            } else if (auto ts = dynamic_cast<krls_interpolation_ts*>(ats.ts.get())) {
                _m_find_ts_map(ts);
                return m[ts] = o_index<krls_interpolation_ts>{ expr.append(srep::_type<krls_interpolation_ts>::rep_t{ convert(ts->ts),ts->predictor }) };
            } else  if (auto ts = dynamic_cast<qac_ts*>(ats.ts.get())) {
                _m_find_ts_map(ts); // NOTICE that qac_ts is so far the only ts that keeps optional time-series,
                return m[ts] = o_index<qac_ts>{ expr.append(srep::_type<qac_ts>::rep_t{ convert(apoint_ts(ts->ts)),convert(apoint_ts(ts->cts)), ts->p }) };
            } else if (auto gts = dynamic_cast<gpoint_ts*>(ats.ts.get())) {
                auto f = gts_map.find(gts);
                if (f != end(gts_map))
                    return f->second;
                expr.gts.emplace_back(gts);
                return gts_map[gts] = o_index<gpoint_ts>{ expr.gts.size() - 1 };
            } else if (auto aref = dynamic_cast<aref_ts*>(ats.ts.get())) {
                auto f = rts_map.find(aref);
                if (f != end(rts_map))
                    return f->second;
                expr.rts.emplace_back(aref);
                return rts_map[aref] = o_index<aref_ts>{ expr.rts.size() - 1 };
            } else {
                throw runtime_error("Not supported yet");
            }
        #undef _m_find_ts_map
        }

        ts_expression_compressor() = default;
        ts_expression_compressor(const ts_expression_compressor&) = default;
    public:

        /** Convert a vector<apoint_ts> or equivalent collection to  ts_expression<...>
        */
        template <class V>
        static ts_expression<srep_types...> compress(const V& atsv) {
            ts_expression_compressor ec;
            for (const auto &ats : atsv)
                ec.expr.roots.push_back(ec.convert(ats));
            return ec.expr;
        }
    };

    using cxx_ext::for_each;

    /** visitor that does the recursive job
    * of constructing a apoint_ts (expression) from
    * it's fast serializable format ts_expr_rep<...>.
    *
    * It's used as major engine by the convert_to_ts_vector function,
    * utilizing the boost::apply_visitor pattern on each root node of the expression.
    *
    */
    template<class ...srep_types> // srep_types, srep::xxx , each required to have a ::ts_t that represents underlying type
    struct ts_expression_decompressor {
        using return_type = shared_ptr<ipoint_ts>;// needed for the boost::apply_visitor pattern
    private:
        /** Helper class to ensure ts_type_map vectors have same initial size (with nullptr) as it's expr.ts_reps */
        template<class T> // srep_types, srep::xxx , each required to have a ::ts_t that represents underlying type
        struct fx_vector_init {
            T& ts_type_map; // T= tuple< vector<shared_ptr<ts_t>> ...>

                            // this simple one should work..
            template<class ts_srep_t>
            void operator()(const vector<ts_srep_t>&v) {
                get< vector< shared_ptr<typename ts_srep_t::ts_t>>  >(ts_type_map).resize(v.size(), nullptr);
            }
        };

        using ts_expr_t = ts_expression<srep_types...>; // just to make the life easier
        const ts_expr_t &expr; // a reference only (this is a on-stack/short-lived object, take no responsibilites)

                                //-- structures to hold the constructed types during build
                                //-- we could have dropped them, if we knew that there was no multiple references to nodes or terminals
                                //-- and just build up structure from leaf-terminals to target-node.
        tuple< vector<shared_ptr<typename srep_types::ts_t>>... > ts_type_map; // for each ts_t a vector<shared_ptr<ts_t>> , except for terminal types
        vector<shared_ptr<aref_ts>> rts;
        vector<shared_ptr<gpoint_ts>> gts;

        ts_expression_decompressor(const ts_expr_t& e)
            :expr(e),
            //TODO: better need to forward constructor arguments to tuple<...>, members, aligned with corresponding e.member.size() ,
            rts(e.rts.size(), nullptr), gts(e.gts.size(), nullptr) {
            fx_vector_init<decltype(ts_type_map)> resize_vectors{ ts_type_map };
            for_each(expr.ts_reps, resize_vectors);// this one ensures vector in the tuple have correct initial size

        }

        //-- section for constructors, called by the generic operator() visitor callback --
        //-- you only need to add the make method for new types of 'nodes'
        shared_ptr<ipoint_ts> make(boost::blank i) { // represent nil values
            return shared_ptr<ipoint_ts>();
        }
        shared_ptr<abin_op_ts> make(o_index<abin_op_ts> i) {
            const auto &r = expr.at(i);
            apoint_ts lhs{ boost::apply_visitor(*this,r.lhs) };
            apoint_ts rhs{ boost::apply_visitor(*this,r.rhs) };
            return make_shared<abin_op_ts>(move(lhs), r.op, move(rhs));
        }

        shared_ptr<abin_op_scalar_ts> make(o_index<abin_op_scalar_ts> i) {
            const auto &r = expr.at(i);
            apoint_ts rhs{ boost::apply_visitor(*this,r.rhs) };
            return make_shared<abin_op_scalar_ts>(r.lhs, r.op, move(rhs));
        }

        shared_ptr<abin_op_ts_scalar> make(o_index<abin_op_ts_scalar> i) {
            const auto& rx = expr.at(i);
            apoint_ts lhs{ boost::apply_visitor(*this,rx.lhs) };
            return make_shared<abin_op_ts_scalar>(move(lhs), rx.op, rx.rhs);
        }

        shared_ptr<abs_ts> make(o_index<abs_ts> i) {
            const auto& rx = expr.at(i);
            return make_shared<abs_ts>(boost::apply_visitor(*this, rx.ts));
        }

        shared_ptr<average_ts> make(o_index<average_ts> i) {
            const auto& rx = expr.at(i);
            return make_shared<average_ts>(rx.ta, boost::apply_visitor(*this, rx.ts));
        }

        shared_ptr<integral_ts> make(o_index<integral_ts> i) {
            const auto& rx = expr.at(i);
            return make_shared<integral_ts>(rx.ta, boost::apply_visitor(*this, rx.ts));
        }

        shared_ptr<accumulate_ts> make(o_index<accumulate_ts> i) {
            const auto& rx = expr.at(i);
            return make_shared<accumulate_ts>(rx.ta, boost::apply_visitor(*this, rx.ts));
        }

        shared_ptr<time_shift_ts> make(o_index<time_shift_ts> i) {
            const auto& rx = expr.at(i);
            return make_shared<time_shift_ts>(boost::apply_visitor(*this, rx.ts), rx.dt);
        }


        shared_ptr<periodic_ts> make(o_index<periodic_ts> i) {
            const auto& rx = expr.at(i);
            return make_shared<periodic_ts>(rx.ts);// a move if expr was not a const (could make sense for scoped local serialization purpose)
        }

        shared_ptr<convolve_w_ts> make(o_index<convolve_w_ts> i) {
            const auto& rx = expr.at(i);
            apoint_ts ts{ boost::apply_visitor(*this,rx.ts) };
            return make_shared<convolve_w_ts>(move(ts), rx.w, rx.policy);
        }

        shared_ptr<extend_ts> make(o_index<extend_ts> i) {
            const auto& rx = expr.at(i);
            apoint_ts lhs{ boost::apply_visitor(*this,rx.lhs) };
            apoint_ts rhs{ boost::apply_visitor(*this,rx.rhs) };
            return make_shared<extend_ts>(lhs, rhs, rx.ets_split_p, rx.ets_fill_p, rx.split_at, rx.fill_value);
        }

        shared_ptr<rating_curve_ts> make(o_index<rating_curve_ts> i) {
            const auto& rx = expr.at(i);
            apoint_ts lts{ boost::apply_visitor(*this,rx.ts) };
            return make_shared<rating_curve_ts>(move(lts), rx.rc_param);
        }

        shared_ptr<ice_packing_ts> make(o_index<ice_packing_ts> i) {
            const auto& rx = expr.at(i);
            apoint_ts tts{ boost::apply_visitor(*this, rx.ts) };
            return make_shared<ice_packing_ts>(std::move(tts), rx.ip_param, rx.ipt_policy);
        }

        shared_ptr<ice_packing_recession_ts> make(o_index<ice_packing_recession_ts> i) {
            const auto& rx = expr.at(i);
            apoint_ts flow_ts{ boost::apply_visitor(*this, rx.flow_ts) };
            apoint_ts ip_ts{ boost::apply_visitor(*this, rx.ip_ts) };
            return make_shared<ice_packing_recession_ts>(std::move(flow_ts), std::move(ip_ts), rx.ipr_param);
        }

        shared_ptr<krls_interpolation_ts> make(o_index<krls_interpolation_ts> i) {
            const auto& rx = expr.at(i);
            apoint_ts src_ts{ boost::apply_visitor(*this,rx.ts) };
            return make_shared<krls_interpolation_ts>(move(src_ts), rx.predictor);
        }

        shared_ptr<qac_ts> make(o_index<qac_ts> i) {
            const auto& rx = expr.at(i);
            apoint_ts src_ts{ boost::apply_visitor(*this,rx.ts) };
            apoint_ts cts;
            if (rx.cts.which() != 0) { // NOTICE could be nil/null ptr.
                cts = apoint_ts(boost::apply_visitor(*this, rx.cts));
            }
            return make_shared<qac_ts>(src_ts, rx.p, cts);
        }

    public: // required for the visitor callbacks
            /** generic callback called by visitor for any type
            * performs lookup in the table, then if missing
            * forward the construction to the corresponding make (typespecific)... method
            * that returns a shared_ptr to the constructed object.
            */
        template<class T>
        return_type operator()(o_index<T> i) {
            auto &V = get<vector<shared_ptr<T>>>(ts_type_map);
            if (!V[i]) // we keep a map of constructed object, to avoid duplication in this process
                V[i] = make(i);
            return V[i];
        }

        //-- section for terminal-nodes that is not part of the tuple-based generic handling above.
        //-- notice that c++ resolves to the 'easiest' best match, and selects these
        //-- to the template above.

        /** specific callback for aref_ts, a terminal handled separately */
        return_type operator()(o_index<aref_ts> i) {
            if (rts[i])
                return rts[i];
            return rts[i] = shared_ptr<aref_ts>(expr.rts[i]);
        }

        /** specific callback for gpoint_ts, a terminal handled separately */
        return_type operator()(o_index<gpoint_ts> i) {
            if (gts[i])
                return gts[i];
            return gts[i] = shared_ptr<gpoint_ts>(expr.gts[i]);
        }
        return_type operator()(boost::blank i) {
            return return_type{};
        }

    public: // interface to be used:
            /**given a ts_expr_rep<...> instance, usually created by
            * the complimentary function convert_to_ts_expression( vector<apoint_ts>..)
            * reconstruct the vector<apoint_ts> from the information in the
            * ts-expression.
            */
        static vector<apoint_ts> decompress(ts_expression<srep_types...> &ex) {
            ts_expression_decompressor<srep_types...> d{ ex };
            vector<apoint_ts> r; r.reserve(ex.roots.size());
            for (const auto& root : ex.roots) {
                r.push_back(apoint_ts{ boost::apply_visitor(d, root) });
            }
            return r;
        }

    };



    //-- finally, concrete types that uses the tuple/variant based template framework above
    //--
    /**convinient macro to use for all know types, use as parameter-pack to ts_exp_rep, etc.*/
#define all_srep_types  srep::sbinop_op_ts, srep::sbinop_ts_scalar, srep::sbin_op_scalar_ts, srep::sabs_ts, srep::saverage_ts, srep::sintegral_ts, srep::saccumulate_ts, \
            srep::stime_shift_ts, srep::speriodic_ts, srep::sconvolve_w_ts, srep::sextend_ts, srep::srating_curve_ts, srep::sice_packing_ts, srep::sice_packing_recession_ts, \
            srep::skrls_interpolation_ts, srep::sqac_ts

    typedef ts_expression<all_srep_types> compressed_ts_expression;
    typedef ts_expression_compressor<all_srep_types> expression_compressor;
    typedef ts_expression_decompressor<all_srep_types> expression_decompressor;

}}} // shyft.time_series::dd

  // stuff for boost serialization needing outer scope
x_serialize_binary(shyft::time_series::dd::a_index);
x_serialize_binary(shyft::time_series::dd::srep::sbinop_op_ts);
x_serialize_binary(shyft::time_series::dd::srep::sbinop_ts_scalar);
x_serialize_binary(shyft::time_series::dd::srep::sbin_op_scalar_ts);
x_serialize_binary(shyft::time_series::dd::srep::sabs_ts);
x_serialize_binary(shyft::time_series::dd::srep::stime_shift_ts);
x_serialize_binary(shyft::time_series::dd::srep::sextend_ts);
x_serialize_binary(shyft::time_series::dd::srep::sqac_ts);

x_serialize_export_key(shyft::time_series::dd::srep::saverage_ts);
x_serialize_export_key(shyft::time_series::dd::srep::sintegral_ts);
x_serialize_export_key(shyft::time_series::dd::srep::saccumulate_ts);
x_serialize_export_key(shyft::time_series::dd::srep::speriodic_ts);
x_serialize_export_key(shyft::time_series::dd::srep::sconvolve_w_ts);
x_serialize_export_key(shyft::time_series::dd::srep::srating_curve_ts);
x_serialize_export_key(shyft::time_series::dd::srep::sice_packing_ts);
x_serialize_export_key(shyft::time_series::dd::srep::sice_packing_recession_ts);
x_serialize_export_key(shyft::time_series::dd::srep::skrls_interpolation_ts);

// annoying.. (could we just say binary serializable for all o_index<T>)
x_serialize_binary(shyft::time_series::dd::o_index<shyft::time_series::dd::abin_op_ts>);
x_serialize_binary(shyft::time_series::dd::o_index<shyft::time_series::dd::abin_op_scalar_ts>);
x_serialize_binary(shyft::time_series::dd::o_index<shyft::time_series::dd::abin_op_ts_scalar>);
x_serialize_binary(shyft::time_series::dd::o_index<shyft::time_series::dd::gpoint_ts>);
x_serialize_binary(shyft::time_series::dd::o_index<shyft::time_series::dd::aref_ts>);
x_serialize_binary(shyft::time_series::dd::o_index<shyft::time_series::dd::abs_ts>);
x_serialize_binary(shyft::time_series::dd::o_index<shyft::time_series::dd::average_ts>);
x_serialize_binary(shyft::time_series::dd::o_index<shyft::time_series::dd::integral_ts>);
x_serialize_binary(shyft::time_series::dd::o_index<shyft::time_series::dd::accumulate_ts>);
x_serialize_binary(shyft::time_series::dd::o_index<shyft::time_series::dd::time_shift_ts>);
x_serialize_binary(shyft::time_series::dd::o_index<shyft::time_series::dd::periodic_ts>);
x_serialize_binary(shyft::time_series::dd::o_index<shyft::time_series::dd::convolve_w_ts>);
x_serialize_binary(shyft::time_series::dd::o_index<shyft::time_series::dd::extend_ts>);
x_serialize_binary(shyft::time_series::dd::o_index<shyft::time_series::dd::rating_curve_ts>);
x_serialize_binary(shyft::time_series::dd::o_index<shyft::time_series::dd::ice_packing_ts>);
x_serialize_binary(shyft::time_series::dd::o_index<shyft::time_series::dd::ice_packing_recession_ts>);
x_serialize_binary(shyft::time_series::dd::o_index<shyft::time_series::dd::krls_interpolation_ts>);
x_serialize_binary(shyft::time_series::dd::o_index<shyft::time_series::dd::qac_ts>);
x_serialize_binary(boost::blank);
