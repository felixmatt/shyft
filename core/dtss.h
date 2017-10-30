#pragma once


#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <memory>
#include <utility>
#include <functional>
#include <cstring>

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include <regex>

#ifdef _WIN32
#include <io.h>
#else
#include <sys/io.h>
#define O_BINARY 0
#define O_SEQUENTIAL 0
#include <sys/stat.h>
#endif
#include <fcntl.h>

#include <dlib/server.h>
#include <dlib/iosockstream.h>
#include <dlib/logger.h>
#include <dlib/misc_api.h>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include "api/time_series.h"
#include "time_series_info.h"
#include "utctime_utilities.h"
#include "dtss_cache.h"

#include <dlib/server.h>
#include <dlib/iosockstream.h>
#include <dlib/logger.h>
#include <dlib/misc_api.h>

namespace shyft {
namespace dtss {

using shyft::core::utctime;
using shyft::core::utcperiod;
using shyft::core::utctimespan;
using shyft::core::no_utctime;
using shyft::core::calendar;
using shyft::core::deltahours;

using gta_t = shyft::time_axis::generic_dt;
using gts_t = shyft::time_series::point_ts<gta_t>;

// TODO: Remove API dependency from core...
using shyft::api::apoint_ts;
using shyft::api::gpoint_ts;
using shyft::api::gts_t;
using shyft::api::aref_ts;

// ========================================

enum class message_type : uint8_t {
    SERVER_EXCEPTION,
    EVALUATE_TS_VECTOR,
    EVALUATE_TS_VECTOR_PERCENTILES,
    FIND_TS,
    STORE_TS,
    // EVALUATE_TS_VECTOR_HISTOGRAM //-- tsv,period,ta,bin_min,bin_max -> ts_vector[n_bins]
};

// ========================================

namespace msg {

template <class T>
message_type read_type(T& in) {
    int32_t mtype;
    in.read((char*)&mtype, sizeof(mtype));
    return (message_type)mtype;
}

template <class T>
void write_type(message_type mt, T& out) {
    int32_t mtype = (int32_t)mt;
    out.write((const char *)&mtype, sizeof(mtype));
}

template <class T>
void write_string(const std::string& s, T& out) {
    int32_t sz = s.size();
    out.write((const char*)&sz, sizeof(sz));
    out.write(s.data(), sz);
}

template <class T>
std::string read_string(T& in) {
    std::int32_t sz;
    in.read((char*)&sz, sizeof(sz));
    std::string msg(sz, '\0');
    in.read((char*)msg.data(), sz);
    return msg;
}

template <class T>
void write_exception(const std::exception& e, T& out) {
    int32_t sz = strlen(e.what());
    out.write((const char*)&sz, sizeof(sz));
    out.write(e.what(), sz);
}

template <class T>
void send_exception(const std::exception& e, T& out) {
    write_type(message_type::SERVER_EXCEPTION, out);
    int32_t sz = strlen(e.what());
    out.write((const char*)&sz, sizeof(sz));
    out.write(e.what(), sz);
}

template <class T>
std::runtime_error read_exception(T& in) {
    int32_t sz;
    in.read((char*)&sz, sizeof(sz));
    std::string msg(sz, '\0');
    in.read((char*)msg.data(), sz);
    return std::runtime_error(msg);
}

}  // msg

// ========================================

using ts_vector_t = shyft::api::ats_vector;
using ts_info_vector_t = std::vector<ts_info>;
using id_vector_t = std::vector<std::string>;
using read_call_back_t = std::function<ts_vector_t(const id_vector_t& ts_ids, utcperiod p)>;
using store_call_back_t = std::function<void(const ts_vector_t&)>;
using find_call_back_t = std::function<ts_info_vector_t(std::string search_expression)>;

// TODO: inline when vs implements P0386R2: Inline variables
extern std::string shyft_prefix;//="shyft://";  ///< marks all internal handled urls

// ========================================

/**construct a shyft-url from container and ts-name */
inline std::string shyft_url(const std::string& container, const std::string& ts_name) {
    return shyft_prefix + container + "/" + ts_name;
}

/** match & extract fast the following 'shyft://<container>/'
* \param url like pattern above
* \return <container> or empty string if no match
*/
inline std::string extract_shyft_url_container(const std::string& url) {
    if ( (url.size() < shyft_prefix.size() + 2) || !std::equal(begin(shyft_prefix), end(shyft_prefix), begin(url)) )
        return std::string{};
    auto ce = url.find_first_of('/', shyft_prefix.size());
    if ( ce == std::string::npos )
        return std::string{};
    return url.substr(shyft_prefix.size(), ce - shyft_prefix.size());
}

// ========================================

/** \brief Storage layer header-record.
 *
 * The storage layer header-record contains enough
 * information to
 *  a) identify the format version (and thus the rest of the layout)
 *  b) provide  minimal quick summary information that could ease search
 *  c) vital features of the ts that is invariant
 *
 * \note This is a packed record, and should portable through small-endian cpu-arch.
 *
 * Format specification:
 * The binary format of the file is then defined as :
 *   <ts.db.file>   -> <header><time-axis><values>
 *       <header>   -> 'TS1' <point_fx> <ta_type> <n> <data_period>
 *     <point_fx>   -> ts_point_fx:uint8_t
 *      <ta_type>   -> time_axis::generic_dt::generic_type:uint8_t
 *            <n>   -> uint32_t // number points
 *  <data_period>   -> int64_t int64_t // from .. until
 *
 * <time-axis>      ->
 *   if ta_type == fixed_dt:
 *        <start>   -> int64_t
 *        <delta_t> -> int64_t
 *
 *   if ta_type == calendar_dt:
 *        <start>   -> int64_t
 *        <delta_t> -> int64_t
 *        <tz_sz>   -> uint32_t // the size of tz-info string bytes following
 *        <tz_name> -> uint8_t[<tz_sz>] // length given by  tz_sz above
 *
 *   if ta_type == point_dt:
 *        <t_end>   -> int64_t // the end of the last interval, aka t_end
 *        <t>       -> int64_t[<n>] // <n> from the header
 *
 * <values>         -> double[<n>] // <n> from the header
 *
 */
#pragma pack(push,1)
struct ts_db_header {
    char signature[4] = { 'T','S','1','\0' }; ///< signature header with version #
    time_series::ts_point_fx point_fx = time_series::POINT_AVERAGE_VALUE; ///< point_fx feature
    time_axis::generic_dt::generic_type ta_type = time_axis::generic_dt::FIXED; ///< time-axis type
    uint32_t n = 0; ///< number of points in the time-series (time-axis and values)
    utcperiod data_period; ///< [from..until> period range

    ts_db_header() = default;
    ts_db_header(time_series::ts_point_fx point_fx, time_axis::generic_dt::generic_type ta_type, uint32_t n, utcperiod data_period)
        : point_fx(point_fx), ta_type(ta_type), n(n), data_period(data_period) {}
};
#pragma pack(pop)


/** \brief A simple file-io based internal time-series storage for the dtss.
*
* Utilizing standard c++ libraries to store time-series
* to regular files, that resides in directory containers.
* Features are limited to simple write/replace, read, search and delete.
*
* The simple idea is just to store one time-series(fragment) pr. file.
*
*
* Using a simple client side url naming:
*
*  shyft://<container>/<container-relative-path>
*
* inside the ts_db at the server-side there is a map
*   <container> -> root_dir
* and thus the fullname of the ts is
*   <container>.root_dir/<container-relative-path>
*
*  e.g.:
*              (proto)  (container ) (   path within container     )
*  client url: 'shyft://measurements/hydmet_station/1/temperature_1'
*
*  server side:
*    ts_db_container['measurements']=ts_db('/srv/shyft/ts_db/measurements')
*
*   which then would resolve into the full-path for the stored ts-file:
*
*     '/srv/shyft/ts_db/measurements/hydmet_station/1/temperature_1'
*
* \note that ts-urls that do not match internal 'shyft' protocol are dispatched
*      to the external setup callback (if any). This way we support both
*      internally managed as well as externally mapped ts-db
*
*/
class ts_db {

private:
    struct close_write_handle {
        bool win_thread_close = false;

        close_write_handle() noexcept = default;
        close_write_handle(bool wtc) noexcept : win_thread_close{ wtc } { };
        close_write_handle(const close_write_handle &) noexcept = default;

        void operator()(std::FILE * fh) const {
#ifdef _WIN32
            if ( win_thread_close ) {
                std::thread(std::fclose, fh).detach();// allow time-consuming close to work in background
            } else {
                std::fclose(fh); // takes forever in windows, by design
            }
#else
            std::fclose(fh);
#endif
        }
    };

private:
    std::map<std::string, std::shared_ptr<core::calendar>> calendars;

public:
    std::string root_dir; ///< root_dir points to the top of the container

public:
    ts_db() = default;

    /** constructs a ts_db with specified container root */
    explicit ts_db(const std::string& root_dir) :root_dir(root_dir) {
        if ( !fs::is_directory(root_dir) ) {
            if ( !fs::exists(root_dir) ) {
                if ( !fs::create_directory(root_dir) ) {
                    throw std::runtime_error(std::string("ts_db: failed to create root directory :") + root_dir);
                }
            } else {
                throw std::runtime_error(std::string("ts_db: designated root directory is not a directory:") + root_dir);
            }
        }
        make_calendar_lookups();
    }
    //mutable double t_open = 0;
    //mutable double t_write = 0;
    //mutable double t_close = 0.0;

    /** \brief Save a time-series to a file, *overwrite* any existing file with contents.
     *
     * \note Windows NTFS is incredibly slow on close, so as a workaround using
     *       handles and separate thread for the close-job for now.
     *
     * \param fn  Pathname to save the time-series at.
     * \param ts  Time-series to save.
     * \param win_thread_close  Only meaningfull on the Windows platform.
     *                          Use a deatached background thread to close the file.
     *                          Defaults to true.
     */
    void save(const std::string& fn, const gts_t& ts, bool overwrite = true, bool win_thread_close = true) const {
        std::string ffp = make_full_path(fn, true);

        //std::chrono::steady_clock::time_point t0, t1, t2, t3;

        std::unique_ptr<std::FILE, close_write_handle> fh;  // zero-initializes deleter
        fh.get_deleter().win_thread_close = win_thread_close;
        ts_db_header old_header;

        //t0 = timing::now();

        bool do_merge = false;
        if ( ! overwrite && save_path_exists(fn) ) {
            fh.reset(std::fopen(ffp.c_str(), "r+b"));
            old_header = read_header(fh.get());
            if ( ts.total_period().contains(old_header.data_period) ) {
                // old data is completly contained in the new => start the file anew
                //  - reopen, as there is no simple way to truncate an open file...
                //std::fseek(fh.get(), 0, SEEK_SET);
                fh.reset(std::fopen(ffp.c_str(), "wb"));
            } else {
                do_merge = true;
            }
        } else {
            fh.reset(std::fopen(ffp.c_str(), "wb"));
        }
        //t1 = timing::now();
        if ( ! do_merge ) {
            write_ts(fh.get(), ts);
        } else {
            merge_ts(fh.get(), old_header, ts);
        }

        //auto t2 = timing::now();
        //close_write_handle(fh.get(), win_thread_close);

        //x f.close(); // actually, it's the close call on windows that is slow, we need to use the handle to easily transfer it to a close-thread
        //auto t3 = timing::now();
        //t_open += elapsed_us(t0, t1)/1000000.0;  // open + a possible header parsing
        //t_write += elapsed_us(t1, t2) / 1000000.0;
        //t_close += elapsed_us(t2, t3) / 1000000.0;
    }

    /** read a ts from specified file */
    gts_t read(const std::string& fn, core::utcperiod p) const {
        //std::ifstream f(make_full_path(fn), std::ios_base::in | std::ios_base::binary);
        std::string ffp = make_full_path(fn);
        std::unique_ptr<std::FILE, decltype(&std::fclose)> fh{ std::fopen(ffp.c_str(), "rb"), &std::fclose };
        return read_ts(fh.get(), p);
    }

    /** removes a ts from the container */
    void remove(const std::string& fn) const {
        auto fp = make_full_path(fn);
        for ( std::size_t retry = 0; retry < 10; ++retry ) {
            try {
                fs::remove(fp);
                return;
            } catch ( ... ) { // windows usually fails, due to delayed file-close/file-release so we retry 10 x 0.3 seconds
                std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(300));
            }
        }
        throw std::runtime_error("failed to remove file '" + fp + "' after 10 repeated attempts lasting for 3 seconds");
    }

    /** get minimal ts-information from specified fn */
    ts_info get_ts_info(const std::string& fn) const {
        auto ffp = make_full_path(fn);
        std::unique_ptr<std::FILE, decltype(&std::fclose)> fh{ std::fopen(ffp.c_str(), "rb"), &std::fclose };
        auto h = read_header(fh.get());
        ts_info i;
        i.name = fn;
        i.point_fx = h.point_fx;
        i.modified = utctime(fs::last_write_time(ffp));
        i.data_period = h.data_period;
        // consider time-axis type info, dt.. as well
        return i;
    }

    /** find all ts_info s that matches the specified re match string
    *
    * e.g.: match= 'hydmet_station/.*_id/temperature'
    *    would find all time-series /hydmet_station/xxx_id/temperature
    */
    std::vector<ts_info> find(const std::string& match) const {
        fs::path root(root_dir);
        std::vector<ts_info> r;
        std::regex r_match(match, std::regex_constants::ECMAScript | std::regex_constants::icase);
        for ( auto&& x : fs::recursive_directory_iterator(root) ) {
            if ( fs::is_regular(x.path()) ) {
                std::string fn = x.path().lexically_relative(root).generic_string(); // x.path() except root-part
                if ( std::regex_search(fn, r_match) ) {
                    r.push_back(get_ts_info(fn)); // TODO: maybe multi-core this into a job-queue
                }
            } else if ( fs::is_directory(x.path()) ) {
                // TODO: elide recursion into the x, calling
                // if x.path is not part of match
                //   x.no_push();
            }
        }
        return r;
    }

private:
    std::shared_ptr<core::calendar> lookup_calendar(const std::string& tz) const {
        auto it = calendars.find(tz);
        if ( it == calendars.end() )
            return std::make_shared<core::calendar>(tz);
        return it->second;
    }

    void make_calendar_lookups() {
        for ( int hour = -11; hour < 12; hour++ ) {
            auto c = std::make_shared<core::calendar>(deltahours(hour));
            calendars[c->tz_info->name()] = c;
        }
    }

    bool save_path_exists(const std::string & fn) const {
        fs::path fn_path{ fn }, root_path{ root_dir };
        if ( fn_path.is_relative() ) {
            fn_path = root_path / fn_path;
        } else {
            // questionable: should we allow outside container specs?
            // return false;
        }
        return fs::is_regular_file(fn_path);
    }
    std::string make_full_path(const std::string& fn, bool create_paths = false) const {
        fs::path fn_path{ fn }, root_path{ root_dir };
        // determine path type
        if ( fn_path.is_relative() ) {
            fn_path = root_path / fn_path;
        } else {  // fn_path.is_absolute()
            // questionable: should we allow outside container specs?
            //  - if determined to be fully allowed: remove this branch or throw
        }
        // not a directory and create missing path
        if ( fs::is_directory(fn_path) ) {
            throw std::runtime_error(fn_path.string()+" is a directory. Should be a file.");
        } else if ( ! fs::exists(fn_path) && create_paths ) {
            fs::path rp = fn_path.parent_path();
            if ( rp.compare(root_path) > 0 ) {  // if fn contains sub-directory, we have to check that it exits
                if ( ! fs::is_directory(rp) ) {
                    fs::create_directories(rp);
                }
            }
        }
        // -----
        return fn_path.string();
    }

    ts_db_header mk_header(const gts_t& ts) const {
        return ts_db_header{ ts.point_interpretation(),ts.time_axis().gt,uint32_t(ts.size()),ts.total_period() };
    }

    // ----------

    inline void write(std::FILE * fh, const void* d, std::size_t sz) const {
        if ( std::fwrite(d, sizeof(char), sz, fh) != sz )
            throw std::runtime_error("dtss_store: failed to write do disk");
    }
    void write_header(std::FILE * fh, const gts_t & ats) const {
        ts_db_header h = mk_header(ats);
        write(fh, static_cast<const void*>(&h), sizeof(h));
    }
    void write_time_axis(std::FILE * fh, const gta_t& ta) const {
        switch ( ta.gt ) {
        case time_axis::generic_dt::FIXED:
        {
            write(fh, static_cast<const void*>(&ta.f.t), sizeof(core::utctime));
            write(fh, static_cast<const void*>(&ta.f.dt), sizeof(core::utctimespan));
        } break;
        case time_axis::generic_dt::CALENDAR:
        {
            write(fh, static_cast<const void*>(&ta.c.t), sizeof(core::utctime));
            write(fh, static_cast<const void*>(&ta.c.dt), sizeof(core::utctimespan));
            std::string tz = ta.c.cal->tz_info->name();
            uint32_t sz = tz.size();
            write(fh, static_cast<const void*>(&sz), sizeof(uint32_t));
            write(fh, static_cast<const void*>(tz.c_str()), sz);
        } break;
        case time_axis::generic_dt::POINT:
        {
            write(fh, static_cast<const void*>(&ta.p.t_end), sizeof(core::utctime));
            write(fh, static_cast<const void*>(ta.p.t.data()), sizeof(core::utctime)*ta.p.t.size());
        } break;
        }
    }
    void write_values(std::FILE * fh, const std::vector<double>& v) const {
        write(fh, static_cast<const void*>(v.data()), sizeof(double)*v.size());
    }
    void write_ts(std::FILE * fh, const gts_t& ats) const {
        write_header(fh, ats);
        write_time_axis(fh, ats.ta);
        write_values(fh, ats.v);
    }

    // ----------

    void do_merge(std::FILE * fh, const ts_db_header & old_header, const time_axis::generic_dt & old_ta, const gts_t & new_ts) const {

        // assume the two time-axes have the same type and are aligned

        const core::utcperiod old_p = old_header.data_period,
                              new_p = new_ts.ta.total_period();

        switch ( old_header.ta_type ) {
        case time_axis::generic_dt::FIXED:
        {
            core::utctime t0, tn;          // first/last time point in the merged data
            std::size_t keep_old_v = 0,    // number of old values to keep
                        nan_v_before = 0,  // number of NaN values to insert between new and old
                        nan_v_after = 0;   // number of NaN values to insert between old and new
            std::vector<double> old_tail(0, 0.);
            // determine first time-axis
            if ( old_p.start <= new_p.start ) {  // old start before new
                t0 = old_p.start;

                // is there a gap between?
                if ( old_p.end < new_p.start ) {
                    nan_v_after = (new_p.start - old_p.end)/new_ts.ta.f.dt;
                }

                // determine number of old values to keep
                keep_old_v = (new_p.start - old_p.start)/new_ts.ta.f.dt - nan_v_after;
            } else {  // new start before old
                t0 = new_p.start;

                // is there a gap between?
                if ( new_p.end < old_p.start ) {
                    nan_v_before = (old_p.start - new_p.end)/new_ts.ta.f.dt;
                }

                // is there is a tail of old values
                if ( new_p.end < old_p.end ) {
                    std::size_t count = (old_p.end - new_p.end)/new_ts.ta.f.dt - nan_v_before;
                    old_tail.resize(count);
                    std::fseek(fh,
                        sizeof(ts_db_header)  // header
                            + 2*sizeof(int64_t)  // + fixed_dt time-axis
                            + (old_header.n - count)*sizeof(double),  // + values except count last
                        SEEK_SET);
                    read(fh, static_cast<void*>(old_tail.data()), count*sizeof(double));
                }
            }

            // determine last time-axis
            if ( old_p.end <= new_p.end ) {  // old end before new
                tn = new_p.end;
            } else {
                tn = old_p.end;
            }

            // write header
            ts_db_header new_header{
                old_header.point_fx, old_header.ta_type,
                static_cast<uint32_t>((tn - t0)/new_ts.ta.f.dt), core::utcperiod{ t0, tn }
            };
            // -----
            std::fseek(fh, 0, SEEK_SET);  // seek to begining
            write(fh, static_cast<const void*>(&new_header), sizeof(ts_db_header));

            // write time-axis
            write(fh, static_cast<const void*>(&t0), sizeof(int64_t));

            // write values
            //  - seek past old values to keep
            std::fseek(fh, sizeof(int64_t) + keep_old_v*sizeof(double), SEEK_CUR);
            //  - if gap after new -> write NaN
            if ( nan_v_after > 0 ) {
                std::vector<double> tmp(nan_v_after, shyft::nan);
                write(fh, static_cast<const void*>(tmp.data()), nan_v_after*sizeof(double));
            }
            //  - write new values
            write(fh, static_cast<const void*>(new_ts.v.data()), new_ts.v.size()*sizeof(double));
            //  - if gap before old -> write NaN
            if ( nan_v_before > 0 ) {
                std::vector<double> tmp(nan_v_before, shyft::nan);
                write(fh, static_cast<const void*>(tmp.data()), nan_v_before*sizeof(double));
            }
            //  - if old tail values -> write them back
            if ( old_tail.size() > 0 ) {
                write(fh, static_cast<const void*>(old_tail.data()), old_tail.size()*sizeof(double));
            }
        } break;
        case time_axis::generic_dt::CALENDAR:
        {
            core::utctime t0, tn;          // first/last time point in the merged data
            std::size_t keep_old_v = 0,    // number of old values to keep
                        nan_v_before = 0,  // number of NaN values to insert between new and old
                        nan_v_after = 0;   // number of NaN values to insert between old and new
            std::vector<double> old_tail(0, 0.);
            // determine first time-axis
            if ( old_p.start <= new_p.start ) {  // old start before new
                t0 = old_p.start;

                // is there a gap between?
                if ( old_p.end < new_p.start ) {
                    nan_v_after = new_ts.ta.c.cal->diff_units(old_p.end, new_p.start, new_ts.ta.c.dt);
                }

                // determine number of old values to keep
                keep_old_v = new_ts.ta.c.cal->diff_units(old_p.start, new_p.start, new_ts.ta.c.dt) - nan_v_after;
            } else {  // new start before old
                t0 = new_p.start;

                // is there a gap between?
                if ( new_p.end < old_p.start ) {
                    nan_v_before = new_ts.ta.c.cal->diff_units(new_p.end, old_p.start, new_ts.ta.c.dt);
                }

                // is there is a tail of old values
                if ( new_p.end < old_p.end ) {
                    std::size_t count = new_ts.ta.c.cal->diff_units(new_p.end, old_p.end, new_ts.ta.c.dt) - nan_v_before;
                    old_tail.resize(count);
                    std::fseek(fh,
                        sizeof(ts_db_header)  // header
                            + 2*sizeof(int64_t),  // + first part of calendar_dt time-axis
                        SEEK_SET);
                    uint32_t tz_sz{};
                    read(fh, static_cast<void *>(&tz_sz), sizeof(tz_sz));  // read size of tz_name
                    std::fseek(fh,
                        tz_sz*sizeof(uint8_t)  // + second part of calendar_dt time-axis
                            + (old_header.n - count)*sizeof(double),  // values to overwrite
                        SEEK_CUR);
                    read(fh, static_cast<void*>(old_tail.data()), count*sizeof(double));
                }
            }

            // determine last time-axis
            if ( old_p.end <= new_p.end ) {  // old end before new
                tn = new_p.end;
            } else {
                tn = old_p.end;
            }

            // write header
            ts_db_header new_header{
                old_header.point_fx, old_header.ta_type,
                static_cast<uint32_t>(new_ts.ta.c.cal->diff_units(t0, tn, new_ts.ta.c.dt)),
                core::utcperiod{ t0, tn }
            };
            // -----
            std::fseek(fh, 0, SEEK_SET);  // seek to begining
            write(fh, static_cast<const void*>(&new_header), sizeof(ts_db_header));

            // update time-axis
            write(fh, static_cast<const void*>(&t0), sizeof(int64_t));
            std::fseek(fh, sizeof(int64_t), SEEK_CUR);
            {
                uint32_t tz_sz{};
                read(fh, static_cast<void *>(&tz_sz), sizeof(tz_sz));  // read size of calendar str
                std::fseek(fh, tz_sz*sizeof(uint8_t), SEEK_CUR);  // seek past tz_name
            }

            // write values
            //  - seek past old values to keep
            std::fseek(fh, keep_old_v*sizeof(double), SEEK_CUR);
            //  - if gap after new -> write NaN
            if ( nan_v_after > 0 ) {
                std::vector<double> tmp(nan_v_after, shyft::nan);
                write(fh, static_cast<const void*>(tmp.data()), nan_v_after*sizeof(double));
            }
            //  - write new values
            write(fh, static_cast<const void*>(new_ts.v.data()), new_ts.v.size()*sizeof(double));
            //  - if gap before old -> write NaN
            if ( nan_v_before > 0 ) {
                std::vector<double> tmp(nan_v_before, shyft::nan);
                write(fh, static_cast<const void*>(tmp.data()), nan_v_before*sizeof(double));
            }
            //  - if old tail values -> write them back
            if ( old_tail.size() > 0 ) {
                write(fh, static_cast<const void*>(old_tail.data()), old_tail.size()*sizeof(double));
            }
        } break;
        case time_axis::generic_dt::POINT:
        {
            std::vector<core::utctime> merged_t;
            merged_t.reserve(old_ta.size() + new_ts.size() + 1);  // guaranteed large enough
            // -----
            std::vector<double> merged_v(0u, 0.);
            merged_v.reserve(old_ta.size() + new_ts.size() + 1);  // guaranteed large enough
            // -----
            const std::size_t old_v_offset = sizeof(ts_db_header) + (old_header.n + 1)*sizeof(int64_t);

            // new start AFTER  =>  start at old
            if ( old_p.start < new_p.start ) {

                // get iterator into old time-axis at first where p.start >= old.start
                //  - [old_ta.begin(), old_end) is the part of old we want to keep
                auto old_end = std::lower_bound(
                    old_ta.p.t.cbegin(), old_ta.p.t.cend(),
                    new_ts.total_period().start );

                // store portion of old time-points to keep
                merged_t.insert(merged_t.end(), old_ta.p.t.cbegin(), old_end);
                // store portion of old values to keep
                const std::size_t to_insert = std::distance(old_ta.p.t.cbegin(), old_end);
                auto it = merged_v.insert(merged_v.end(), to_insert, 0.);
                std::fseek(fh, old_v_offset, SEEK_SET);
                read(fh, static_cast<void*>(&(*it)), to_insert*sizeof(double));

                // if NEW start truly after OLD include the end point
                if ( old_end == old_ta.p.t.cend() && new_p.start > old_ta.p.t_end ) {
                    merged_t.emplace_back(old_ta.p.t_end);  // include the end point
                    merged_v.emplace_back(shyft::nan);      // nan for the gap
                }
            }

            // read new into merge_ts
            merged_t.insert(merged_t.end(), new_ts.ta.p.t.cbegin(), new_ts.ta.p.t.cend());
            merged_t.emplace_back(new_ts.ta.p.t_end);
            merged_v.insert(merged_v.end(), new_ts.v.cbegin(), new_ts.v.cend());
            // if new end BEFORE start of old  =>  insert NaN
            if ( new_p.end < old_p.start ) {
                merged_v.emplace_back(shyft::nan);
            }

            // new end BEFORE end of old  =>  read more from old
            if ( new_p.end < old_p.end ) {

                // determine first period in old NOT CONTAINING new.end
                auto old_begin = std::upper_bound(
                    old_ta.p.t.cbegin(), old_ta.p.t.cend(),
                    new_ts.ta.p.t_end );

                // store any trailing old time-points
                merged_t.insert(merged_t.end(), old_begin, old_ta.p.t.cend());
                merged_t.emplace_back(old_ta.p.t_end);
                // store portion of old values to keep
                std::size_t to_insert = std::distance(old_begin, old_ta.p.t.cend());
                // new end INSIDE of AT start of old  =>  insert value from old where new end
                if ( new_p.end >= old_p.start ) {
                    to_insert += 1;
                }
                auto it = merged_v.insert(merged_v.end(), to_insert, 0.);
                std::fseek(fh, old_v_offset + (old_header.n - to_insert)*sizeof(double), SEEK_SET);
                read(fh, static_cast<void*>(&(*it)), to_insert*sizeof(double));
            }

            // write header
            ts_db_header new_header{
                old_header.point_fx, old_header.ta_type,
                static_cast<uint32_t>(merged_t.size() - 1),
                core::utcperiod{ merged_t.at(0), merged_t.at(merged_t.size()-1) }
            };
            // -----
            std::fseek(fh, 0, SEEK_SET);  // seek to begining
            write(fh, static_cast<const void *>(&new_header), sizeof(ts_db_header));

            // write time-axis
            write(fh, static_cast<const void *>(&merged_t.at(merged_t.size() - 1)), sizeof(int64_t));
            write(fh, static_cast<const void *>(merged_t.data()), (merged_t.size() - 1)*sizeof(int64_t));

            // write values
            write(fh, static_cast<const void *>(merged_v.data()), merged_v.size()*sizeof(double));
        } break;
        }
    }
    void check_ta_alignment(std::FILE * fh, const ts_db_header & old_header, const time_axis::generic_dt & old_ta, const gts_t & ats) const {
        if ( ats.fx_policy != old_header.point_fx ) {
            throw std::runtime_error("dtss_store: cannot merge with different point interpretation");
        }
        // -----
        if ( ats.ta.gt != old_header.ta_type ) {
            throw std::runtime_error("dtss_store: cannot merge with different ta type");
        } else {
            // parse specific ta data to determine compatibility
            switch ( old_header.ta_type ) {
            case time_axis::generic_dt::FIXED:
            {
                if ( old_ta.f.dt != ats.ta.f.dt || (old_ta.f.t - ats.ta.f.t) % old_ta.f.dt != 0  ) {
                    throw std::runtime_error("dtss_store: cannot merge unaligned fixed_dt");
                }
            } break;
            case time_axis::generic_dt::CALENDAR:
            {
                if ( ats.ta.c.cal->tz_info->tz.tz_name == old_ta.c.cal->tz_info->tz.tz_name ) {
                    core::utctimespan remainder;
                    ats.ta.c.cal->diff_units(old_ta.c.t, ats.ta.c.t, old_ta.c.dt, remainder);
                    if ( old_ta.c.dt != ats.ta.c.dt || remainder != 0  ) {
                        throw std::runtime_error("dtss_store: cannot merge unaligned calendar_dt");
                    }
                } else {
                    throw std::runtime_error("dtss_store: cannot merge calendar_dt with different calendars");
                }
            } break;
            case time_axis::generic_dt::POINT: break;
            }
        }
    }
    void merge_ts(std::FILE * fh, const ts_db_header & old_header, const gts_t & ats) const {
        // read time-axis
        std::size_t ignored{};
        time_axis::generic_dt old_ta = read_time_axis(fh, old_header, old_header.data_period, ignored);

        check_ta_alignment(fh, old_header, old_ta, ats);
        do_merge(fh, old_header, old_ta, ats);
    }

    // ----------

    inline void read(std::FILE * fh, void* d, std::size_t sz) const {
        std::size_t rsz = std::fread(d, sizeof(char), sz, fh);
        if ( rsz != sz )
            throw std::runtime_error("dtss_store: failed to read from disk");
    }
    ts_db_header read_header(std::FILE * fh) const {
        ts_db_header h;
        std::fseek(fh, 0, SEEK_SET);
        read(fh, static_cast<void *>(&h), sizeof(ts_db_header));
        return h;
    }
    gta_t read_time_axis(std::FILE * fh, const ts_db_header& h, const utcperiod p, std::size_t& skip_n) const {

        // seek to beginning of time-axis
        std::fseek(fh, sizeof(ts_db_header), SEEK_SET);

        gta_t ta;
        ta.gt = h.ta_type;

        core::utctime t_start = p.start;
        core::utctime t_end = p.end;
        if ( t_start == core::no_utctime )
            t_start = core::min_utctime;
        if ( t_end == core::no_utctime )
            t_end = core::max_utctime;

        // no overlap?
        if ( h.data_period.end <= t_start || h.data_period.start >= t_end ) {
            return ta;
        }

        skip_n = 0;
        core::utctime t0 = core::no_utctime;
        switch ( h.ta_type ) {
        case time_axis::generic_dt::FIXED:
        {
            // read start & step
            read(fh, static_cast<void *>(&t0), sizeof(core::utctime));
            read(fh, static_cast<void *>(&ta.f.dt), sizeof(core::utctimespan));
            // handle various overlaping periods
            if ( t_start <= h.data_period.start && t_end >= h.data_period.end ) {
                // fully around or exact
                ta.f.t = t0;
                ta.f.n = h.n;
            } else {
                std::size_t drop_n = 0;
                if ( t_start > h.data_period.start )  // start inside
                    skip_n = (t_start - h.data_period.start) / ta.f.dt;
                if ( t_end < h.data_period.end )  // end inside
                    drop_n = (h.data_period.end - t_end) / ta.f.dt;
                // -----
                ta.f.t = t0 + ta.f.dt*skip_n;
                ta.f.n = h.n - skip_n - drop_n;
            }
        } break;
        case time_axis::generic_dt::CALENDAR:
        {
            // read start & step
            read(fh, static_cast<void *>(&t0), sizeof(core::utctime));
            read(fh, static_cast<void *>(&ta.c.dt), sizeof(core::utctimespan));
            // read tz_info
            uint32_t sz{ 0 };
            read(fh, static_cast<void *>(&sz), sizeof(uint32_t));
            std::string tz(sz, '\0');
            {
                std::unique_ptr<char[]> tmp_ptr = std::make_unique<char[]>(sz);
                read(fh, static_cast<void*>(tmp_ptr.get()), sz);
                tz.replace(0, sz, tmp_ptr.get(), sz);
            }
            ta.c.cal = lookup_calendar(tz);
            // handle various overlaping periods
            if ( t_start <= h.data_period.start && t_end >= h.data_period.end ) {
                // fully around or exact
                ta.c.t = t0;
                ta.c.n = h.n;
            } else {
                std::size_t drop_n = 0;
                if ( t_start > h.data_period.start )  // start inside
                    skip_n = ta.c.cal->diff_units(h.data_period.start, t_start, ta.c.dt);
                if ( t_end < h.data_period.end )  // end inside
                    drop_n = ta.c.cal->diff_units(t_end, h.data_period.end, ta.c.dt);
                // -----
                ta.c.t = ta.c.cal->add(t0, ta.c.dt, skip_n);
                ta.c.n = h.n - skip_n - drop_n;
            }
        } break;
        case time_axis::generic_dt::POINT:
        {
            if ( t_start <= h.data_period.start && t_end >= h.data_period.end ) {
                // fully around or exact
                ta.p.t.resize(h.n);
                read(fh, static_cast<void *>(&ta.p.t_end), sizeof(core::utctime));
                read(fh, static_cast<void *>(ta.p.t.data()), sizeof(core::utctime)*h.n);
            } else {
                core::utctime f_time = 0;
                std::vector<core::utctime> tmp( h.n, 0. );
                read(fh, static_cast<void *>(&f_time), sizeof(core::utctime));
                read(fh, static_cast<void *>(tmp.data()), sizeof(core::utctime)*h.n);
                // -----
                auto it_b = tmp.begin();
                if ( t_start > h.data_period.start ) {
                    it_b = std::upper_bound(tmp.begin(), tmp.end(), t_start, std::less<core::utctime>());
                    if ( it_b != tmp.begin() ) {
                        std::advance(it_b, -1);
                    }
                }
                // -----
                auto it_e = tmp.end();
                if ( t_end < h.data_period.end ) {
                    it_e = std::upper_bound(it_b, tmp.end(), t_end, std::less<core::utctime>());
                    if ( it_e != tmp.end() ) {
                        f_time = *it_e;
                    }
                }
                // -----
                skip_n = std::distance(tmp.begin(), it_b);
                ta.p.t.reserve(std::distance(it_b, it_e));
                ta.p.t.assign(it_b, it_e);
                ta.p.t_end = f_time;
            }
        } break;
        }
        return ta;
    }
    std::vector<double> read_values(std::FILE * fh, const ts_db_header& h, const gta_t& ta, const std::size_t skip_n) const {

        // seek to beginning of values
        std::fseek(fh, sizeof(ts_db_header), SEEK_SET);
        switch ( h.ta_type ) {
        case time_axis::generic_dt::FIXED:
        {
            std::fseek(fh, 2*sizeof(int64_t), SEEK_CUR);
        } break;
        case time_axis::generic_dt::CALENDAR:
        {
            std::fseek(fh, 2*sizeof(int64_t), SEEK_CUR);
            uint32_t sz{};
            read(fh, static_cast<void*>(&sz), sizeof(uint32_t));
            std::fseek(fh, sz*sizeof(uint8_t), SEEK_CUR);
        } break;
        case time_axis::generic_dt::POINT:
        {
            std::fseek(fh, (h.n + 1)*sizeof(int64_t), SEEK_CUR);
        } break;
        }

        const std::size_t points_n = ta.size();
        std::vector<double> val(points_n, 0.);
        std::fseek(fh, sizeof(double)*skip_n, SEEK_CUR);
        read(fh, static_cast<void *>(val.data()), sizeof(double)*points_n);
        return val;
    }
    gts_t read_ts(std::FILE * fh, const utcperiod p) const {
        std::size_t skip_n = 0u;
        ts_db_header h = read_header(fh);
        gta_t ta = read_time_axis(fh, h, p, skip_n);
        std::vector<double> v = read_values(fh, h, ta, skip_n);
        return gts_t{ std::move(ta),move(v),h.point_fx };
    }

};


/** \brief A dtss server with time-series server-side functions
*
* The dtss server listens on a port, receives messages, interpret them
* and ship the response back to the client.
*
* Callbacks are provided for extending/delegating find/read_ts/store_ts,
* as well as internal implementation of storing time-series
* using plain binary files stored in containers(directory).
*
* Time-series are named with url's, and all request involving 'shyft://'
* like
*   shyft://<container>/<local_ts_name>
* resolves to the internal implementation.
*
* TODO: container and thread-safety, given user changes the containers after
*       the server is started.
*
*/
struct server : dlib::server_iostream {
    using ts_cache_t = cache<apoint_ts_frag,apoint_ts>;
    // callbacks for extensions
    read_call_back_t bind_ts_cb; ///< called to read non shyft:// unbound ts
    find_call_back_t find_ts_cb; ///< called for all non shyft:// find operations
    store_call_back_t store_ts_cb;///< called for all non shyft:// store operations
    // shyft-internal implementation
    std::map<std::string, ts_db> container;///< mapping of internal shyft <container> -> ts_db
    ts_cache_t ts_cache{1000000};// default 1 mill ts in cache
    bool cache_all_reads{false};
    // constructors

    server()=default;

    template <class CB>
    explicit server(CB&& cb):bind_ts_cb(std::forward<CB>(cb)) {
    }

    template <class RCB,class FCB>
    server(RCB&& rcb, FCB && fcb ) :
        bind_ts_cb(std::forward<RCB>(rcb)),
        find_ts_cb(std::forward<FCB>(fcb)) {
    }
    template <class RCB,class FCB,class SCB>
    server(RCB&& rcb, FCB && fcb ,SCB&& scb) :
        bind_ts_cb(std::forward<RCB>(rcb)),
        find_ts_cb(std::forward<FCB>(fcb)),
        store_ts_cb(std::forward<SCB>(scb)) {
    }

    ~server() =default;

    //-- container management
    void add_container(const std::string &container_name,const std::string& root_dir) {
        container[container_name]=ts_db(root_dir);
    }

    const ts_db& internal(const std::string& container_name) const {
        auto f=container.find(container_name);
        if(f == end(container))
            throw runtime_error(std::string("Failed to find shyft container:")+container_name);
        return f->second;
    }
    //-- expose cache functions
    // or just leak them through ts_cache ?
    void add_to_cache(id_vector_t&ids, ts_vector_t& tss) { ts_cache.add(ids,tss);}

    void remove_from_cache(id_vector_t &ids) { ts_cache.remove(ids);}

    cache_stats get_cache_stats() { return ts_cache.get_cache_stats();}

    void clear_cache_stats() { ts_cache.clear_cache_stats();}

    void flush_cache() { return ts_cache.flush();}

    void set_cache_size(std::size_t max_size) { ts_cache.set_capacity(max_size);}

    void set_auto_cache(bool active) { cache_all_reads=active;}

    std::size_t get_cache_size() const {return ts_cache.get_capacity();}

    ts_info_vector_t do_find_ts(const std::string& search_expression) {
        //TODO:
        // 1. filter shyft://<container>/
        auto c=extract_shyft_url_container(search_expression);
        if(c.size()) {
            return internal(c).find(search_expression.substr(shyft_prefix.size()+c.size()+1));
        } else if (find_ts_cb) {
            return find_ts_cb(search_expression);
        } else {
            return ts_info_vector_t();
        }
    }

    std::string extract_url(const apoint_ts&ats) const {
        auto rts = dynamic_pointer_cast<aref_ts>(ats.ts);
        if(rts)
            return rts->id;
        throw runtime_error("dtss store.extract_url:supplied type must be of type ref_ts");
    }

    void do_cache_update_on_write(const ts_vector_t&tsv) {
        for (std::size_t i = 0; i<tsv.size(); ++i) {
            auto rts = dynamic_pointer_cast<aref_ts>(tsv[i].ts);
            ts_cache.add(rts->id, apoint_ts(rts->rep));
        }
    }
    void do_store_ts(const ts_vector_t & tsv, bool overwrite_on_write, bool cache_on_write) {
        if(tsv.size()==0) return;
        // 1. filter out all shyft://<container>/<ts-path> elements
        //    and route these to the internal storage controller (threaded)
        //    std::map<std::string, ts_db> shyft_internal;
        //
        std::vector<std::size_t> other;
        other.reserve(tsv.size());
        for(std::size_t i=0;i<tsv.size();++i) {
            auto rts = dynamic_pointer_cast<aref_ts>(tsv[i].ts);
            if(!rts) throw runtime_error("dtss store: require ts with url-references");
            auto c= extract_shyft_url_container(rts->id);
            if(c.size()) {
                internal(c).save(
                    rts->id.substr(shyft_prefix.size()+c.size()+1),  // path
                    rts->core_ts(),  // ts to save
                    overwrite_on_write  // should do overwrite instead of merge
                );
                if ( cache_on_write ) { // ok, this ends up in a copy, and lock for each item(can be optimized if many)
                    ts_cache.add(rts->id, apoint_ts(rts->rep));
                }
            } else {
                other.push_back(i); // keep idx of those we have not saved
            }
        }

        // 2. for all non shyft:// forward those to the
        //    store_ts_cb
        if(store_ts_cb && other.size()) {
            if(other.size()==tsv.size()) { //avoid copy/move if possible
                store_ts_cb(tsv);
                if (cache_on_write) do_cache_update_on_write(tsv);
            } else { // have to do a copy to new vector
                ts_vector_t r;
                for(auto i:other) r.push_back(tsv[i]);
                store_ts_cb(r);
                if (cache_on_write) do_cache_update_on_write(r);
            }
        }
    }

    ts_vector_t do_read(const id_vector_t& ts_ids,utcperiod p) {
        if(ts_ids.size()==0) return ts_vector_t{};
        // 0. filter out cached ts
        auto cc = ts_cache.get(ts_ids,p);
        ts_vector_t r(ts_ids.size());
        std::vector<std::size_t> other;other.reserve(ts_ids.size());
        // 1. filter out shyft://
        //    if all shyft: return internal read
        for(std::size_t i=0;i<ts_ids.size();++i) {
            if(cc.find(ts_ids[i])==cc.end()) {
                auto c=extract_shyft_url_container(ts_ids[i]);
                if(c.size()) {
                    r[i]=apoint_ts(make_shared<gpoint_ts>(internal(c).read(ts_ids[i].substr(shyft_prefix.size()+c.size()+1),p)));
                    if(cache_all_reads) ts_cache.add(ts_ids[i],r[i]);
                } else
                    other.push_back(i);
            } else {
                r[i]=cc[ts_ids[i]];
            }
        }
        // 2. if other/more than shyft
        //    get all those
        if(other.size()) {
            if(!bind_ts_cb)
                throw std::runtime_error("dtss: read-request to external ts, without external handler");
            if(other.size()==ts_ids.size()) {// only other series, just return result
                auto rts= bind_ts_cb(ts_ids,p);
                if(cache_all_reads) ts_cache.add(ts_ids,rts);
                return rts;
            }
            std::vector<std::string> o_ts_ids;o_ts_ids.reserve(other.size());
            for(auto i:other) o_ts_ids.push_back(ts_ids[i]);
            auto o=bind_ts_cb(o_ts_ids,p);
            if(cache_all_reads) ts_cache.add(o_ts_ids,o);
            // if 1 and 2, merge into one ordered result vector
            //
            for(std::size_t i=0;i<o.size();++i)
                r[other[i]]=o[i];
        }
        return r;
    }

    void
        do_bind_ts(utcperiod bind_period, ts_vector_t& atsv)  {
        std::map<std::string, std::vector<api::ts_bind_info>> ts_bind_map;
        std::vector<std::string> ts_id_list;
        // step 1: bind not yet bound time-series ( ts with only symbol, needs to be resolved using bind_cb)
        for (auto& ats : atsv) {
            auto ts_refs = ats.find_ts_bind_info();
            for (const auto& bi : ts_refs) {
                if (ts_bind_map.find(bi.reference) == ts_bind_map.end()) { // maintain unique set
                    ts_id_list.push_back(bi.reference);
                    ts_bind_map[bi.reference] = std::vector<api::ts_bind_info>();
                }
                ts_bind_map[bi.reference].push_back(bi);
            }
        }

        // step 2: (optional) bind_ts callback should resolve symbol time-series with content
        if (ts_bind_map.size()) {
            auto bts = do_read(ts_id_list, bind_period);
            if (bts.size() != ts_id_list.size())
                throw std::runtime_error(std::string("failed to bind all of ") + std::to_string(bts.size()) + std::string(" ts"));

            for ( std::size_t i = 0; i < ts_id_list.size(); ++i ) {
                for ( auto & bi : ts_bind_map[ts_id_list[i]] )
                    bi.ts.bind(bts[i]);
            }
        }
        // step 3: after the symbolic ts are read and bound, we iterate over the
        //         expression tree and calls .do_bind() so that
        //         the new information is taken into account and the expression tree are
        //         ready for evaluate with everything const so threading is safe.
        for (auto& ats : atsv)
            ats.do_bind();
    }

    ts_vector_t
        do_evaluate_ts_vector(utcperiod bind_period, ts_vector_t& atsv) {
        do_bind_ts(bind_period, atsv);
        return ts_vector_t(api::deflate_ts_vector<apoint_ts>(atsv));
    }

    ts_vector_t
        do_evaluate_percentiles(utcperiod bind_period, ts_vector_t& atsv, api::gta_t const&ta,std::vector<int> const& percentile_spec) {
        do_bind_ts(bind_period, atsv);
        return api::percentiles(atsv, ta, percentile_spec);// we can assume the result is trivial to serialize
    }

    void on_connect(
        std::istream& in,
        std::ostream& out,
        const std::string& foreign_ip,
        const std::string& local_ip,
        unsigned short foreign_port,
        unsigned short local_port,
        dlib::uint64 connection_id
    ) {
        while (in.peek() != EOF) {
            auto msg_type= msg::read_type(in);
            try {
                switch (msg_type) { // currently switch, later maybe table[msg_type]=msg_handler
                case message_type::EVALUATE_TS_VECTOR: {
                    utcperiod bind_period;
                    ts_vector_t rtsv; {
                        boost::archive::binary_iarchive ia(in);
                        ia>>bind_period>>rtsv;
                    }

                    auto result=do_evaluate_ts_vector(bind_period, rtsv); {// first get result
                        msg::write_type(message_type::EVALUATE_TS_VECTOR,out);// then send
                        boost::archive::binary_oarchive oa(out);
                        oa<<result;
                    }
                } break;
                case message_type::EVALUATE_TS_VECTOR_PERCENTILES: {
                    utcperiod bind_period;
                    ts_vector_t rtsv;
                    std::vector<int> percentile_spec;
                    api::gta_t ta; {
                        boost::archive::binary_iarchive ia(in);
                        ia >> bind_period >> rtsv>>ta>>percentile_spec;
                    }

                    auto result = do_evaluate_percentiles(bind_period, rtsv,ta,percentile_spec);{
                        msg::write_type(message_type::EVALUATE_TS_VECTOR_PERCENTILES, out);
                        boost::archive::binary_oarchive oa(out);
                        oa << result;
                    }
                } break;
                case message_type::FIND_TS: {
                    std::string search_expression; {
                        search_expression = msg::read_string(in);// >> search_expression;
                    }
                    auto find_result = do_find_ts(search_expression); {
                        msg::write_type(message_type::FIND_TS, out);
                        boost::archive::binary_oarchive oa(out);
                        oa << find_result;
                    }
                } break;
                case message_type::STORE_TS: {
                    ts_vector_t rtsv;
                    bool overwrite_on_write{ true };
                    bool cache_on_write{ false };
                    {
                        boost::archive::binary_iarchive ia(in);
                        ia >> rtsv >> overwrite_on_write >> cache_on_write;
                    }
                    do_store_ts(rtsv, overwrite_on_write, cache_on_write);
                    {
                        msg::write_type(message_type::STORE_TS, out);
                    }
                } break;
                default:
                    throw std::runtime_error(std::string("Got unknown message type:") + std::to_string((int)msg_type));
                }
            } catch (std::exception const& e) {
                msg::send_exception(e,out);
            }
        }
        //std::cout<<"of conn:"<<foreign_ip<<":"<<foreign_port<<", local_port ="<<local_port<<std::endl;

    }
};

struct client {
    dlib::iosockstream io;
    std::string host_port;
    explicit client(const std::string& host_port):io(host_port),host_port(host_port) {}

    void reopen() {
        io.close();
        io.open(host_port);
    }
    void close(int timeout_ms=1000) {io.close(timeout_ms);}

    std::vector<apoint_ts>
        percentiles(ts_vector_t const& tsv, utcperiod p,api::gta_t const&ta,const std::vector<int>& percentile_spec) {
        if (tsv.size()==0)
            throw std::runtime_error("percentiles requires a source ts-vector with more than 0 time-series");
        if (percentile_spec.size()==0)
            throw std::runtime_error("percentile function require more than 0 percentiles specified");
        if (!p.valid())
            throw std::runtime_error("percentiles require a valid period-specification");
        if (ta.size()==0)
            throw std::runtime_error("percentile function require a time-axis with more than 0 steps");

        msg::write_type(message_type::EVALUATE_TS_VECTOR_PERCENTILES, io);
        {
            boost::archive::binary_oarchive oa(io);
            oa << p << tsv<<ta<<percentile_spec;
        }
        auto response_type = msg::read_type(io);
        if (response_type == message_type::SERVER_EXCEPTION) {
            auto re = msg::read_exception(io);
            throw re;
        } else if (response_type == message_type::EVALUATE_TS_VECTOR_PERCENTILES) {
            ts_vector_t r;
            {
                boost::archive::binary_iarchive ia(io);
                ia >> r;
            }
            return r;
        }
        throw std::runtime_error(std::string("Got unexpected response:") + std::to_string((int)response_type));
    }

    std::vector<apoint_ts>
        evaluate(ts_vector_t const& tsv, utcperiod p) {
        if (tsv.size()==0)
            throw std::runtime_error("evaluate requires a source ts-vector with more than 0 time-series");
        if (!p.valid())
            throw std::runtime_error("percentiles require a valid period-specification");
        msg::write_type(message_type::EVALUATE_TS_VECTOR,io);{
            boost::archive::binary_oarchive oa(io);
            oa<<p<<tsv;
        }
        auto response_type= msg::read_type(io);
        if(response_type==message_type::SERVER_EXCEPTION) {
            auto re= msg::read_exception(io);
            throw re;
        } else if(response_type==message_type::EVALUATE_TS_VECTOR) {
            ts_vector_t r;{
                boost::archive::binary_iarchive ia(io);
                ia>>r;
            }
            return r;
        }
        throw std::runtime_error(std::string("Got unexpected response:")+std::to_string((int)response_type));
    }
    void store_ts(const ts_vector_t &tsv, bool overwrite_on_write, bool cache_on_write) {
        if (tsv.size()==0)
            return; //trivial and considered valid case
                    // verify that each member of tsv is a gpoint_ts
        for(auto const &ats:tsv) {
            auto rts = dynamic_cast<api::aref_ts*>(ats.ts.get());
            if(!rts) throw std::runtime_error(std::string("attempt to store a null ts"));
            if(rts->needs_bind()) throw std::runtime_error(std::string("attempt to store unbound ts:")+rts->id);
        }
        msg::write_type(message_type::STORE_TS, io);
        {
            boost::archive::binary_oarchive oa(io);
            oa << tsv << overwrite_on_write << cache_on_write;
        }
        auto response_type= msg::read_type(io);
        if(response_type==message_type::SERVER_EXCEPTION) {
            auto re= msg::read_exception(io);
            throw re;
        } else if(response_type==message_type::STORE_TS) {
            return ;
        }
        throw std::runtime_error(std::string("Got unexpected response:")+std::to_string((int)response_type));
    }

    ts_info_vector_t find(const std::string& search_expression) {
        msg::write_type(message_type::FIND_TS, io);
        {
            //boost::archive::binary_oarchive oa(io);
            //oa << search_expression;
            msg::write_string(search_expression,io);
        }
        auto response_type = msg::read_type(io);
        if (response_type == message_type::SERVER_EXCEPTION) {
            auto re = msg::read_exception(io);
            throw re;
        }
        else if (response_type == message_type::FIND_TS) {
            ts_info_vector_t r;
            {
                boost::archive::binary_iarchive ia(io);
                ia >> r;
            }
            return r;
        }
        throw std::runtime_error(std::string("Got unexpected response:") + std::to_string((int)response_type));
    }

};

// ========================================

inline std::vector<apoint_ts> dtss_evaluate(const std::string& host_port, const ts_vector_t& tsv, utcperiod p, int timeout_ms = 10000) {
    return client(host_port).evaluate(tsv, p);
}

inline std::vector<apoint_ts> dtss_percentiles(const std::string& host_port, const ts_vector_t& tsv, utcperiod p, const api::gta_t& ta, std::vector<int> percentile_spec, int timeout_ms = 10000) {
    return client(host_port).percentiles(tsv, p, ta, percentile_spec);
}

} // shyft::dtss
} // shyft
