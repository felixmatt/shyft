#pragma once
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <memory>

#include <dlib/iosockstream.h>
#include <dlib/misc_api.h>

#include "core/time_series_dd.h"
#include "time_series_info.h"
#include "utctime_utilities.h"
#include "dtss_cache.h"

namespace shyft {
namespace dtss {

using std::vector;
using std::string;
using std::unique_ptr;
using std::make_unique;
using std::max;

using shyft::core::utctime;
using shyft::core::utcperiod;
using shyft::core::utctimespan;
using shyft::core::no_utctime;
using shyft::core::calendar;
using shyft::core::deltahours;

using gta_t = shyft::time_axis::generic_dt;
using gts_t = shyft::time_series::point_ts<gta_t>;

// TODO: move api::ts dependency into core
using shyft::time_series::dd::apoint_ts;
using shyft::time_series::dd::gpoint_ts;
using shyft::time_series::dd::gts_t;
using shyft::time_series::dd::aref_ts;

// ========================================

using ts_vector_t = shyft::time_series::dd::ats_vector;
using ts_info_vector_t = vector<ts_info>;
using id_vector_t = vector<string>;

struct srv_connection {
    unique_ptr<dlib::iosockstream> io;
    string host_port;
    int timeout_ms;
    void open(int timeout_ms=1000);
    void close(int timeout_ms=1000);
    void reopen(int timeout_ms=1000);
};

/** \brief a dtss client
 *
 * This class implements the client side functionality of the dtss client-server.
 *
 *
 */
struct client {
    /** A client can connect to 1..n dtss and distribute the calculations
     * among these
     */
    vector<srv_connection> srv_con;

	bool auto_connect{true}; ///< if enabled, connections are made as needed, and kept short, otherwise externally managed.

    bool compress_expressions{true};///< compress expressions to gain speed

	client (const string& host_port, bool auto_connect = true, int timeout_ms=1000);

    client(const vector<string>& host_ports,bool auto_connect,int timeout_ms);

	void reopen(int timeout_ms=1000);

	void close(int timeout_ms=1000);

	vector<apoint_ts> percentiles(ts_vector_t const& tsv, utcperiod p, gta_t const&ta, const vector<int64_t>& percentile_spec,bool use_ts_cached_read,bool update_ts_cache);

	vector<apoint_ts> evaluate(ts_vector_t const& tsv, utcperiod p,bool use_ts_cached_read,bool update_ts_cache) ;

	void store_ts(const ts_vector_t &tsv, bool overwrite_on_write, bool cache_on_write) ;
    
    void merge_store_ts(const ts_vector_t &tsv, bool cache_on_write) ;

	ts_info_vector_t find(const string& search_expression) ;

	void cache_flush() ;

	cache_stats get_cache_stats();

};

// ========================================

inline vector<apoint_ts> dtss_evaluate(const string& host_port, const ts_vector_t& tsv, utcperiod p, int timeout_ms = 10000,bool use_ts_cached_read=false,bool update_ts_cache=false) {
	return client(host_port).evaluate(tsv, p, use_ts_cached_read,update_ts_cache);
}

inline vector<apoint_ts> dtss_percentiles(const string& host_port, const ts_vector_t& tsv, utcperiod p, const gta_t& ta, vector<int64_t> percentile_spec, int timeout_ms = 10000,bool use_ts_cached_read=false,bool update_ts_cache=false) {
	return client(host_port).percentiles(tsv, p, ta, percentile_spec, use_ts_cached_read,update_ts_cache);
}

} // shyft::dtss
} // shyft
