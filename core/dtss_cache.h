#pragma once


#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <memory>
#include <utility>
#include <mutex>
#include <stdexcept>


#include "utctime_utilities.h"
#include "time_series.h"
#include "time_series_merge.h"
#include "api/time_series.h"

namespace shyft {
    namespace dtss {
        using std::size_t;
        using std::vector;
        using std::map;
        using std::make_shared;
        using std::shared_ptr;
        using std::lower_bound;
        using std::upper_bound;
        using std::string;
        using std::mutex;
        //using std::scoped_lock;
        using std::lock_guard;
        using std::min;
        using std::max;
        using std::runtime_error;
        using std::dynamic_pointer_cast;

        using shyft::core::utcperiod;
        using shyft::core::utctime;
        using shyft::time_series::merge;
        using shyft::api::apoint_ts;
        using shyft::api::gta_t;
        using shyft::api::gpoint_ts;

        /** \brief lru-cache
         *
         *  Based on https://timday.bitbucket.io/lru.html#x1-8007r1 and other that
         *  have done similar minimal approaches.
         *
         *  There are even more advanced open-source caches available, with a lot of features
         *  that are interesting, but currently we keep a minimalistic approach and
         *  with features that is closely related to our needs.
         *
         *  In the dtss context we use it as *tool* for the thread-safe and more specific
         *  caching, including sparse ts cache, possibly update with merge.
         *
         */
        template <
            typename K,
            typename V,
            template<typename...> class MAP
        >
        struct lru_cache {
            using key_type = K;
            using value_type = V;
            using key_tracker_type = std::list<key_type>;///< Key access history, most recent at back

                                                         /** Key to value and key history iterator */
            using key_to_value_type = MAP<
                key_type,
                std::pair<value_type, typename key_tracker_type::iterator>
            >;

            /** Constructor specifies the cached function and
             *  the maximum number of records to be stored
             *  \param c maximum id-count in the cache, required to be > 0
             *
             */
            lru_cache(size_t c) :_capacity(c) {
                assert(_capacity != 0);
            }

            /** check if a item with key k exists in the cache
             * \param k key to check for
             * \return true if item with key k exists
             */
            bool item_exists(const key_type& k) const {
                return _key_to_value.find(k)!=_key_to_value.end();
            }

            /**\return a reference to item with key k, or throws */
            value_type& get_item(const key_type& k) {
                const auto it = _key_to_value.find(k);
                if (it == _key_to_value.end()) {// We don't have it:
                    throw runtime_error(string("attempt to get non-existing key:")+k);
                }
                else { // We do have it, rotate the element into front of list
                    _key_tracker.splice(_key_tracker.end(), _key_tracker, (*it).second.second);
                    return (*it).second.first;// Return the retrieved value
                }
            }

            /**  try to get a value of the cached function for k */
            bool try_get_item(const key_type& k, value_type&r) {
                if (item_exists(k)) {
                    r = get_item(k);
                    return true;
                }
                return false;
            }

            /** add an item(that does not exists) into cache, or if exist, update its value */
            void add_item(const key_type&k, const value_type&v) {
                auto it = _key_to_value.find(k);
                if (it == _key_to_value.end()) {
                    insert(k, v);
                }
                else {
                    (*it).second.first = v;
                    _key_tracker.splice(_key_tracker.end(), _key_tracker, (*it).second.second);// rotate into 1st position
                }
            }

            /** remove and item that *might* exist */
            void remove_item(const key_type&k) {
                const auto it = _key_to_value.find(k);
                if (it != _key_to_value.end()) { // We do have it:
                    _key_tracker.erase((*it).second.second);
                    _key_to_value.erase(it);
                }
            }

            /** Obtain the cached keys, most recently used element
            *  at head, least recently used at tail.
            *  This method is provided purely to support testing. */
            template <typename IT>
            void get_mru_keys(IT dst) const {
                auto src = _key_tracker.rbegin();
                while (src != _key_tracker.rend()) *dst++ = *src++;
            }

            /**adjust capacity, evict excessive items as needed */
            void set_capacity(size_t cap) {
                if(cap==0) throw runtime_error("cache capacity must be >0");
                if(cap<_capacity) {
                    while(_key_to_value.size()>cap)
                        evict();
                }
                _capacity=cap;
            }
            size_t get_capacity() const {return _capacity;}

            private:

                /** Record a fresh key-value pair in the cache */
                void insert(const key_type& k, const value_type& v) {
                    assert(_key_to_value.find(k) == _key_to_value.end());// Method is only called on cache misses
                    if (_key_to_value.size() >= _capacity) // Make space if necessary
                        evict();

                    // Record k as most-recently-used key
                    auto it = _key_tracker.insert(_key_tracker.end(), k);
                    // Create the key-value entry, linked to the usage record.
                    _key_to_value.insert(
                        std::make_pair(
                            k,
                            std::make_pair(v, it)
                        )
                    );
                }

                /** Purge the least-recently-used element in the cache */
                void evict() {
                    assert(!_key_tracker.empty());// Assert method is never called when cache is empty
                                                  // Identify least recently used key
                    const auto it = _key_to_value.find(_key_tracker.front());
                    assert(it != _key_to_value.end());
                    // Erase both elements to completely purge record
                    _key_to_value.erase(it);
                    _key_tracker.pop_front();
                }


                size_t _capacity;///< Maximum number of key-value pairs to be retained
                key_tracker_type _key_tracker;///< Key access history
                key_to_value_type _key_to_value; ///< Key-to-value lookup
        };


        /** \brief mini_frag provides a container that minimizes the set of time-series fragments
         *
         *  The purpose of this class is to provide a container that keeps
         *  a minimum set of ts-fragments, each with non-overlapping.total_period().
         *
         *  When adding a new fragment, the existing fragments are checked, and if it's
         *  possible to merge with already existing ts-fragments, this is done, and
         *  existing fragments is merged into the new one.
         *  The merge is assumed to put priority to the newly inserted fragment, so
         *  overlapping parts are replaced with the new value-fragment.
         *
         * \tparam ts_frag
         *     is a type that provides:
         *     #) .total_period() const ->utcperiod; total-period covered by the fragment
         *     #) .merge(const ts_frag&other_with_lower_pri)->ts_frag;  a new ts-fragment
         *     #) .size() const ->size_t; return
         *     and satisfies requirement for a vector<ts_frag>
         */
        template<typename ts_frag> // ts_frag, provide .total_period(),.size() and .merge()
        struct mini_frag {
            vector<ts_frag> f;///< fragments, ordered by .total_period().start, non-overlapping, disjoint periods

            /** returns index of ts-fragment that covers p, or string::npos if none */
            size_t get_ix(const utcperiod&p) const {
                auto r = lower_bound(begin(f), end(f), p.start, [](const ts_frag& x, utctime t) { return x.total_period().start <= t; });
                size_t i = static_cast<size_t>(r - begin(f)) - 1;
                if (i == string::npos)
                    return i;
                return f[i].total_period().contains(p) ? i : string::npos;
            }

            ts_frag& get_by_ix(size_t i) { return f[i]; }

            const ts_frag& get_by_ix(size_t i) const { return f[i]; }

            /** return number of fragments */
            size_t count_fragments() const { return f.size(); }

            /**\return the accumulated .size() for all fragments, x8 ~ approx. bytes */
            size_t estimate_size() const { size_t s = 0; for (const auto&x:f) s += x.size(); return s; }

            /**\brief add a new fragment to the container
            *
            * ensures that the internal container remains ordered by .total_period().start
            * and that all periods are disjoint (not overlapping, not touching ends)
            *
            * \param tsf
            *   the new time-series fragment to be added into the container.
            */
            void add(const ts_frag& tsf) {
                auto p = tsf.total_period();
                auto p1 = lower_bound(begin(f), end(f), p.start, [](const ts_frag&x, const utctime& t)->bool {return x.total_period().end<t; });
                if (p1 == end(f)) { // entirely after last (if any) elements
                    f.push_back(tsf);
                    return;
                }
                if (p.end < p1->total_period().start) { //entirely before first.
                    f.insert(p1, tsf);
                    return;
                }
                auto p2 = upper_bound(p1, end(f), p.end, [](const utctime& t, const ts_frag&x) {return t < x.total_period().end; });//figure out upper bound element
                if (p.start <= p1->total_period().start) { // p1 completely covered
                    if (p2 == end(f)) {//  p2  also completely covered
                        *p1 = tsf; f.erase(p1+1, p2);
                        return;
                    }
                    // parts of p2 must be merged
                    if ( p2->total_period().start <= p.end) { // overlap, - we merge with p2
                        *p1 = tsf.merge(*p2); f.erase(p1+1, p2+1);// and consumes p2
                    } else {
                        *p1 = tsf;f.erase(p1+1,p2);//p2 is above, so we merge until p2
                    }
                } else { // parts of p1 must be merged
                    if (p2 == end(f)) { // now look at p2
                        *p1 = tsf.merge(*p1); f.erase(p1+1, p2); // parts of p1 merged, p2 vanishes
                        return;
                    }
                    if(p2->total_period().start<=p.end) {
                        *p1 = tsf.merge(*p1).merge(*p2); f.erase(p1+1, p2+1);// both p1 and p2 merged.
                    } else {
                        *p1= tsf.merge(*p1);f.erase(p1+1,p2);// p2 is above p.end, p2 must remain
                    }
                }
            }
        };

        /** ts-fragment class for use in dtss_cache for apoint_ts type
        * to be located in the shyft::api section
        */
        struct apoint_ts_frag {

            apoint_ts ats;///< a concrete ts ref. to  gpoint_ts ref to point_ts<gta_t> for impl.

            apoint_ts& ts() { return ats; } ///<ref to ts used in cache
            const apoint_ts&  ts() const { return ats; } ///< const ref to ts used in cache

                                                         //
                                                         // the required template signature for use in mini_frag<ts_frag> class
                                                         //

                                                         /** total_period() for the underlying time-series */
            utcperiod total_period() const { return ats.total_period(); }

            /** the point size() of the underlying time-series */
            size_t size() const { return ats.size(); }

            /** merge returns a NEW apoint_ts_frag
            *
            * Performs a ts_merge of this fragment (high priority) with the other
            * such that the new fragment as minimum keeps this, plus extensions from
            * the other at none, one or both sides of this
            * \sa shyft::time_series::merge
            *
            * \param o the other ts-fragment
            * \return a new ts-fragment
            */
            apoint_ts_frag merge(const apoint_ts_frag& o) const {
                namespace ts = shyft::time_series;
                auto s_pts = dynamic_pointer_cast<gpoint_ts>(ats.ts);
                auto o_pts = dynamic_pointer_cast<gpoint_ts>(o.ats.ts);
                if (s_pts && o_pts) {
                    return apoint_ts_frag{
                        apoint_ts(
                            make_shared<gpoint_ts>(ts::merge(s_pts->rep,o_pts->rep))
                        )
                    };// move ct to shared
                }
                throw runtime_error("attempt to merge nullptr apoint_ts time-series");
            }
        };

        /** cache stats for performance measures */
        struct cache_stats {
            cache_stats() = default;
            size_t hits{ 0 };///< accumulated hits by id
            size_t misses{ 0 };///< accumulated misses by id
            size_t coverage_misses{ 0 };///< accumulated misses for period-coverage (the id exists, but not requested period)
            size_t id_count{ 0 };///< current count of disticnt ts-ids in the cache
            size_t point_count{ 0 };///< current estimate of ts-points in the cache, one point ~8 bytes
            size_t fragment_count{ 0 };///< current count of ts-fragments, equal or larger than id_count
        };

        /** \brief a dtss cache for id-based ts-fragments
         *
         * Provides thread-safe:
         *	 .cache( (id|ts) | (ids| tsv) | tsv<ref-ts>)
         *  .try_get( id | ids ) -> vector<id> (null or real)
         *  .remove( (id|ids))
         *  .statistics (..)->stat
         *  .size(#n ts, # max elems)
         *
         * key-value based lru cache
         *  key-type: string, url type as passed
         * 	value-type: mini_frag<ts_frag>
         *
         * maintain by lru, - where the value-type (ts_frag) is maintained as a
         *                    minimal set of non-overlapping disjoint ts-fragments.
         *
         * \sa lru_cache
         * \sa cache_stats
         * \sa apoints_ts_frag
         *
         */
        template<class ts_frag, class ts_t>
        struct cache {
            using value_type = mini_frag<ts_frag>;
            using internal_cache = lru_cache<string, value_type, map>;
        private:
            mutable mutex mx; ///< mutex to protect access to c and cs
            internal_cache c;///< internal cache implementation
            cache_stats cs;///< internal cache stats to collect misses/hits

                           /** get one single item from cache, if exists, and matches period, record hits/misses */
            bool internal_try_get(const string& id, const utcperiod& p, ts_t& ts) {
                if (!c.item_exists(id)) {
                    ++cs.misses;
                    return false;
                }
                ++cs.hits;
                const auto& mf = c.get_item(id);
                size_t ix = mf.get_ix(p);
                if (ix==string::npos) {
                    ++cs.coverage_misses;
                    return false;
                }
                ts = mf.get_by_ix(ix).ts();
                return true;
            }

            /** add one single item to cache, defrag if already there */
            void internal_add(const string &id, const ts_t &ts) {
                if (!c.item_exists(id)) {
                    value_type mf; mf.add(ts_frag{ ts });
                    c.add_item(id, mf);
                }
                else {
                    auto&mf = c.get_item(id);
                    mf.add(ts_frag{ ts });
                }
            }

        public:
            /** construct a cache with max ts-id count */
            cache(size_t id_max_count) :c(id_max_count) {}

            /** \brief adjust the cache capacity
            *
            * Set the capacity related to unique ts-ids to specified count.
            * If adjusting down, elements are evicted from cache in lru-order.
            *
            * \param id_max_count the new maximium number of unique ts-ids to keep
            *
            */
            void set_capacity(size_t id_max_count) {
                lock_guard<mutex> guard(mx);
                c.set_capacity(id_max_count);
            }
            size_t get_capacity() const {
                lock_guard<mutex> guard(mx);
                return c.get_capacity();
            }

            /** try get a ts that matches id and period.
            *
            * \sa get
            *
            * \param id any valid time-series id
            * \param p  period specification
            * \param ts a reference to ts, set if id is found in the cache with sufficient period
            * \return true if  found and matches period, with ts set to found ts-frag, otherwise false and untouched ts
            */
            bool try_get(const string& id, const utcperiod& p, ts_t& ts) {
                lock_guard<mutex> guard(mx);
                return internal_try_get(id, p, ts);
            }

            /** \brief get out a list of ts by specified id and period from cache
             *
             * thread-safe get out cache content by ts-id and period specification
             *
             * \sa try_get
             *
             * \param ids a vector of ts-ids to be fetched
             * \param p specifies the period requirement
             * \return a map<string,ts_t> with the time-series from cache that matches the criteria
             */
            map<string, ts_t> get(const vector<string>& ids, const utcperiod& p) {
                lock_guard<mutex> guard(mx);
                map<string, ts_t> r;
                for (const auto&id:ids) {
                    ts_t x;
                    if (internal_try_get(id, p, x)) {
                        r[id] = x;
                    }
                }
                return r;
            }

            /**\brief add (id,ts) to cache
             *
             * Adds or replaces id,ts pair into the cahce,
             * possibly merge fragments if that id have fragments already present in the
             * cache
             *
             * \sa add vector
             *
             * \param id any valid time-series id
             * \param ts a const ref to ts (fragment) to be added/replaced in the cache
             *
             */
            void add(const string &id, const ts_t &ts) {
                lock_guard<mutex> guard(mx);
                internal_add(id, ts);
            }

            /** \brief add a vector of time-series to cache
             *
             * Ensures size of ids and tss are equal, then iterate over the pairs
             * and add/replace those items to cache in the ascending index order.
             *
             * \param ids a vector of valid time-series identifiers
             * \param tss a vector with time-series corresponding by position to ids
             *
             */
            template<typename TSV>
            void add(const vector<string>& ids, const TSV& tss) {
                if (ids.size()!=tss.size())
                    throw runtime_error("attempt to add mismatched size for ts-ids and ts to cache");
                lock_guard<mutex> guard(mx);
                for (size_t i = 0; i<ids.size(); ++i) {
                    internal_add(ids[i], tss[i]);
                }
            }

            /** remove a specified ts-id from cache
             *
             * If the id does not exits, this call has null-effect
             *
             * \param id any valid time-series id
             */
            void remove(const string& id) {
                lock_guard<mutex> guard(mx);
                c.remove_item(id);
            }

            /** \brief remove specified ts-ids from cache
             *
             * If one or more of the  id does not exits, this is ignored(not an error)
             *
             * \param ids a list of valid time-series id
             */
            void remove(const vector<string>& ids) {
                lock_guard<mutex> guard(mx);
                for (const auto& id:ids)
                    c.remove_item(id);
            }

            /** \brief flushes the cache
             *
             * All elements are removed from cache, resources released
             *
             * \note the accumulated cache-statistics is not cleared
             */
            void flush() {
                lock_guard<mutex> guard(mx);
                vector<string> ids;
                c.get_mru_keys(back_inserter(ids));
                for (const auto&id:ids)
                    c.remove_item(id);
            }

            /** Provide cache-statistics
             *
             * \return cache_stats with accumulated hits/misses as well as current id-count and point-count
             */
            cache_stats get_cache_stats() {
                lock_guard<mutex> guard(mx);
                cache_stats r{ cs };
                vector<string> ids;
                c.get_mru_keys(back_inserter(ids));
                r.id_count = ids.size();
                for (const auto&id:ids) {
                    const auto& ci = c.get_item(id);
                    r.point_count += ci.estimate_size();
                    r.fragment_count += ci.count_fragments();
                }
                return r;
            }

            /** clear accumulated cache-stats */
            void clear_cache_stats() {
                lock_guard<mutex> guard(mx);
                cs = cache_stats{};
            }

        };
    }
}
