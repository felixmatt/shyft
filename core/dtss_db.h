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

#include "api/time_series.h"
#include "time_series_info.h"
#include "utctime_utilities.h"

namespace shyft {
namespace dtss {
// TODO: Remove API dependency from core...
using shyft::api::apoint_ts;
using shyft::api::gpoint_ts;
using shyft::api::gts_t;
using shyft::api::aref_ts;
using shyft::core::utctime;
using shyft::core::utcperiod;
using shyft::core::utctimespan;
using shyft::core::no_utctime;
using shyft::core::calendar;
using shyft::core::deltahours;

using gta_t = shyft::time_axis::generic_dt;
using gts_t = shyft::time_series::point_ts<gta_t>;

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
		: point_fx(point_fx), ta_type(ta_type), n(n), data_period(data_period) {
	}
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
struct ts_db {
	std::string root_dir; ///< root_dir points to the top of the container
  private:

  	/** helper class needed for win compensating code */
	struct close_write_handle {
		bool win_thread_close = false;
		mutable ts_db * parent = nullptr;

		close_write_handle() noexcept = default;
		close_write_handle(bool wtc) noexcept : win_thread_close{ wtc } {};
		close_write_handle(const close_write_handle &) noexcept = default;

		void operator()(std::FILE * fh) const {
#ifdef _WIN32
			if (win_thread_close && parent) {
				parent->fclose_me(fh);
				//std::thread(std::fclose, fh).detach();// allow time-consuming close to work in background
			} else {
				std::fclose(fh); // takes forever in windows, by design
			}
#else
			std::fclose(fh);
#endif
		}
	};

	std::map<std::string, std::shared_ptr<core::calendar>> calendars;

	//--section dealing with windows and (postponing slow) closing files
#ifdef _WIN32
	mutable mutex fclose_mx;
	mutable std::vector<std::future<void>> fclose_windows;

	void fclose_me(std::FILE *fh) {
		lock_guard<decltype(fclose_mx)> scoped_lock(fclose_mx);
		fclose_windows.emplace_back(std::async(std::launch::async, [fh]() { std::fclose(fh); }));
	}

	void wait_for_close_fh() const noexcept {
		try {
			lock_guard<decltype(fclose_mx)> scope_lock(fclose_mx);
			for (auto& fc : fclose_windows)
				fc.get();
			fclose_windows.clear();
		} catch (...) {

		}
	}
#else
	void wait_for_close_fh() const noexcept {}
#endif
public:


	ts_db() = default;

	/** constructs a ts_db with specified container root */
	explicit ts_db(const std::string& root_dir) :root_dir(root_dir) {
		if (!fs::is_directory(root_dir)) {
			if (!fs::exists(root_dir)) {
				if (!fs::create_directory(root_dir)) {
					throw std::runtime_error(std::string("ts_db: failed to create root directory :") + root_dir);
				}
			} else {
				throw std::runtime_error(std::string("ts_db: designated root directory is not a directory:") + root_dir);
			}
		}
		make_calendar_lookups();
	}

	~ts_db() {
		wait_for_close_fh();
	}

	// note that we need special care(windows) for the operations below
	// basically we don't copy/move the fclose_windows, rather just wait it out before overwrite.
	ts_db(const ts_db&c) :root_dir(c.root_dir),calendars(c.calendars) {}
	ts_db(ts_db&&c) :root_dir(c.root_dir), calendars(c.calendars) {};
	ts_db & operator=(const ts_db&o) {
		if (&o != this) {
			wait_for_close_fh();
			root_dir = o.root_dir;
			calendars = o.calendars;
		}
		return *this;
	};
	ts_db & operator=( ts_db&&o) {
		if (&o != this) {
			wait_for_close_fh();
			root_dir = o.root_dir;
			calendars = o.calendars;
		}
		return *this;
	};

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

		std::unique_ptr<std::FILE, close_write_handle> fh;  // zero-initializes deleter
		fh.get_deleter().win_thread_close = win_thread_close;
		fh.get_deleter().parent = const_cast<ts_db*>(this);
		ts_db_header old_header;

		bool do_merge = false;
		if (!overwrite && save_path_exists(fn)) {
			fh.reset(std::fopen(ffp.c_str(), "r+b"));
			old_header = read_header(fh.get());
			if (ts.total_period().contains(old_header.data_period)) {
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
		if (!do_merge) {
			write_ts(fh.get(), ts);
		} else {
			merge_ts(fh.get(), old_header, ts);
		}
	}

	/** read a ts from specified file */
	gts_t read(const std::string& fn, core::utcperiod p) const {
		wait_for_close_fh();
		std::string ffp = make_full_path(fn);
		std::unique_ptr<std::FILE, decltype(&std::fclose)> fh{ std::fopen(ffp.c_str(), "rb"), &std::fclose };
		return read_ts(fh.get(), p);
	}

	/** removes a ts from the container */
	void remove(const std::string& fn) const {
		wait_for_close_fh();
		auto fp = make_full_path(fn);
		for (std::size_t retry = 0; retry < 10; ++retry) {
			try {
				fs::remove(fp);
				return;
			} catch (...) { // windows usually fails, due to delayed file-close/file-release so we retry 10 x 0.3 seconds
				std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(300));
			}
		}
		throw std::runtime_error("failed to remove file '" + fp + "' after 10 repeated attempts lasting for 3 seconds");
	}

	/** get minimal ts-information from specified fn */
	ts_info get_ts_info(const std::string& fn) const {
		wait_for_close_fh();
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
		wait_for_close_fh();
		fs::path root(root_dir);
		std::vector<ts_info> r;
		std::regex r_match(match, std::regex_constants::ECMAScript | std::regex_constants::icase);
		for (auto&& x : fs::recursive_directory_iterator(root)) {
			if (fs::is_regular(x.path())) {
				std::string fn = x.path().lexically_relative(root).generic_string(); // x.path() except root-part
				if (std::regex_search(fn, r_match)) {
					r.push_back(get_ts_info(fn)); // TODO: maybe multi-core this into a job-queue
				}
			} else if (fs::is_directory(x.path())) {
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
		if (it == calendars.end())
			return std::make_shared<core::calendar>(tz);
		return it->second;
	}

	void make_calendar_lookups() {
		for (int hour = -11; hour < 12; hour++) {
			auto c = std::make_shared<core::calendar>(deltahours(hour));
			calendars[c->tz_info->name()] = c;
		}
	}

	bool save_path_exists(const std::string & fn) const {
		fs::path fn_path{ fn }, root_path{ root_dir };
		if (fn_path.is_relative()) {
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
		if (fn_path.is_relative()) {
			fn_path = root_path / fn_path;
		} else {  // fn_path.is_absolute()
				  // questionable: should we allow outside container specs?
				  //  - if determined to be fully allowed: remove this branch or throw
		}
		// not a directory and create missing path
		if (fs::is_directory(fn_path)) {
			throw std::runtime_error(fn_path.string() + " is a directory. Should be a file.");
		} else if (!fs::exists(fn_path) && create_paths) {
			fs::path rp = fn_path.parent_path();
			if (rp.compare(root_path) > 0) {  // if fn contains sub-directory, we have to check that it exits
				if (!fs::is_directory(rp)) {
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
		if (std::fwrite(d, sizeof(char), sz, fh) != sz)
			throw std::runtime_error("dtss_store: failed to write do disk");
	}
	void write_header(std::FILE * fh, const gts_t & ats) const {
		ts_db_header h = mk_header(ats);
		write(fh, static_cast<const void*>(&h), sizeof(h));
	}
	void write_time_axis(std::FILE * fh, const gta_t& ta) const {
		switch (ta.gt) {
		case time_axis::generic_dt::FIXED: {
			write(fh, static_cast<const void*>(&ta.f.t), sizeof(core::utctime));
			write(fh, static_cast<const void*>(&ta.f.dt), sizeof(core::utctimespan));
		} break;
		case time_axis::generic_dt::CALENDAR: {
			write(fh, static_cast<const void*>(&ta.c.t), sizeof(core::utctime));
			write(fh, static_cast<const void*>(&ta.c.dt), sizeof(core::utctimespan));
			std::string tz = ta.c.cal->tz_info->name();
			uint32_t sz = tz.size();
			write(fh, static_cast<const void*>(&sz), sizeof(uint32_t));
			write(fh, static_cast<const void*>(tz.c_str()), sz);
		} break;
		case time_axis::generic_dt::POINT: {
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

		switch (old_header.ta_type) {
		case time_axis::generic_dt::FIXED: {
			core::utctime t0, tn;          // first/last time point in the merged data
			std::size_t keep_old_v = 0,    // number of old values to keep
				nan_v_before = 0,  // number of NaN values to insert between new and old
				nan_v_after = 0;   // number of NaN values to insert between old and new
			std::vector<double> old_tail(0, 0.);
			// determine first time-axis
			if (old_p.start <= new_p.start) {  // old start before new
				t0 = old_p.start;

				// is there a gap between?
				if (old_p.end < new_p.start) {
					nan_v_after = (new_p.start - old_p.end) / new_ts.ta.f.dt;
				}

				// determine number of old values to keep
				keep_old_v = (new_p.start - old_p.start) / new_ts.ta.f.dt - nan_v_after;
			} else {  // new start before old
				t0 = new_p.start;

				// is there a gap between?
				if (new_p.end < old_p.start) {
					nan_v_before = (old_p.start - new_p.end) / new_ts.ta.f.dt;
				}

				// is there is a tail of old values
				if (new_p.end < old_p.end) {
					std::size_t count = (old_p.end - new_p.end) / new_ts.ta.f.dt - nan_v_before;
					old_tail.resize(count);
					std::fseek(fh,
						sizeof(ts_db_header)  // header
						+ 2 * sizeof(int64_t)  // + fixed_dt time-axis
						+ (old_header.n - count) * sizeof(double),  // + values except count last
						SEEK_SET);
					read(fh, static_cast<void*>(old_tail.data()), count * sizeof(double));
				}
			}

			// determine last time-axis
			if (old_p.end <= new_p.end) {  // old end before new
				tn = new_p.end;
			} else {
				tn = old_p.end;
			}

			// write header
			ts_db_header new_header{
				old_header.point_fx, old_header.ta_type,
				static_cast<uint32_t>((tn - t0) / new_ts.ta.f.dt), core::utcperiod{ t0, tn }
			};
			// -----
			std::fseek(fh, 0, SEEK_SET);  // seek to begining
			write(fh, static_cast<const void*>(&new_header), sizeof(ts_db_header));

			// write time-axis
			write(fh, static_cast<const void*>(&t0), sizeof(int64_t));

			// write values
			//  - seek past old values to keep
			std::fseek(fh, sizeof(int64_t) + keep_old_v * sizeof(double), SEEK_CUR);
			//  - if gap after new -> write NaN
			if (nan_v_after > 0) {
				std::vector<double> tmp(nan_v_after, shyft::nan);
				write(fh, static_cast<const void*>(tmp.data()), nan_v_after * sizeof(double));
			}
			//  - write new values
			write(fh, static_cast<const void*>(new_ts.v.data()), new_ts.v.size() * sizeof(double));
			//  - if gap before old -> write NaN
			if (nan_v_before > 0) {
				std::vector<double> tmp(nan_v_before, shyft::nan);
				write(fh, static_cast<const void*>(tmp.data()), nan_v_before * sizeof(double));
			}
			//  - if old tail values -> write them back
			if (old_tail.size() > 0) {
				write(fh, static_cast<const void*>(old_tail.data()), old_tail.size() * sizeof(double));
			}
		} break;
		case time_axis::generic_dt::CALENDAR: {
			core::utctime t0, tn;          // first/last time point in the merged data
			std::size_t keep_old_v = 0,    // number of old values to keep
				nan_v_before = 0,  // number of NaN values to insert between new and old
				nan_v_after = 0;   // number of NaN values to insert between old and new
			std::vector<double> old_tail(0, 0.);
			// determine first time-axis
			if (old_p.start <= new_p.start) {  // old start before new
				t0 = old_p.start;

				// is there a gap between?
				if (old_p.end < new_p.start) {
					nan_v_after = new_ts.ta.c.cal->diff_units(old_p.end, new_p.start, new_ts.ta.c.dt);
				}

				// determine number of old values to keep
				keep_old_v = new_ts.ta.c.cal->diff_units(old_p.start, new_p.start, new_ts.ta.c.dt) - nan_v_after;
			} else {  // new start before old
				t0 = new_p.start;

				// is there a gap between?
				if (new_p.end < old_p.start) {
					nan_v_before = new_ts.ta.c.cal->diff_units(new_p.end, old_p.start, new_ts.ta.c.dt);
				}

				// is there is a tail of old values
				if (new_p.end < old_p.end) {
					std::size_t count = new_ts.ta.c.cal->diff_units(new_p.end, old_p.end, new_ts.ta.c.dt) - nan_v_before;
					old_tail.resize(count);
					std::fseek(fh,
						sizeof(ts_db_header)  // header
						+ 2 * sizeof(int64_t),  // + first part of calendar_dt time-axis
						SEEK_SET);
					uint32_t tz_sz{};
					read(fh, static_cast<void *>(&tz_sz), sizeof(tz_sz));  // read size of tz_name
					std::fseek(fh,
						tz_sz * sizeof(uint8_t)  // + second part of calendar_dt time-axis
						+ (old_header.n - count) * sizeof(double),  // values to overwrite
						SEEK_CUR);
					read(fh, static_cast<void*>(old_tail.data()), count * sizeof(double));
				}
			}

			// determine last time-axis
			if (old_p.end <= new_p.end) {  // old end before new
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
				std::fseek(fh, tz_sz * sizeof(uint8_t), SEEK_CUR);  // seek past tz_name
			}

			// write values
			//  - seek past old values to keep
			std::fseek(fh, keep_old_v * sizeof(double), SEEK_CUR);
			//  - if gap after new -> write NaN
			if (nan_v_after > 0) {
				std::vector<double> tmp(nan_v_after, shyft::nan);
				write(fh, static_cast<const void*>(tmp.data()), nan_v_after * sizeof(double));
			}
			//  - write new values
			write(fh, static_cast<const void*>(new_ts.v.data()), new_ts.v.size() * sizeof(double));
			//  - if gap before old -> write NaN
			if (nan_v_before > 0) {
				std::vector<double> tmp(nan_v_before, shyft::nan);
				write(fh, static_cast<const void*>(tmp.data()), nan_v_before * sizeof(double));
			}
			//  - if old tail values -> write them back
			if (old_tail.size() > 0) {
				write(fh, static_cast<const void*>(old_tail.data()), old_tail.size() * sizeof(double));
			}
		} break;
		case time_axis::generic_dt::POINT: {
			std::vector<core::utctime> merged_t;
			merged_t.reserve(old_ta.size() + new_ts.size() + 1);  // guaranteed large enough
																  // -----
			std::vector<double> merged_v(0u, 0.);
			merged_v.reserve(old_ta.size() + new_ts.size() + 1);  // guaranteed large enough
																  // -----
			const std::size_t old_v_offset = sizeof(ts_db_header) + (old_header.n + 1) * sizeof(int64_t);

			// new start AFTER  =>  start at old
			if (old_p.start < new_p.start) {

				// get iterator into old time-axis at first where p.start >= old.start
				//  - [old_ta.begin(), old_end) is the part of old we want to keep
				auto old_end = std::lower_bound(
					old_ta.p.t.cbegin(), old_ta.p.t.cend(),
					new_ts.total_period().start);

				// store portion of old time-points to keep
				merged_t.insert(merged_t.end(), old_ta.p.t.cbegin(), old_end);
				// store portion of old values to keep
				const std::size_t to_insert = std::distance(old_ta.p.t.cbegin(), old_end);
				auto it = merged_v.insert(merged_v.end(), to_insert, 0.);
				std::fseek(fh, old_v_offset, SEEK_SET);
				read(fh, static_cast<void*>(&(*it)), to_insert * sizeof(double));

				// if NEW start truly after OLD include the end point
				if (old_end == old_ta.p.t.cend() && new_p.start > old_ta.p.t_end) {
					merged_t.emplace_back(old_ta.p.t_end);  // include the end point
					merged_v.emplace_back(shyft::nan);      // nan for the gap
				}
			}

			// read new into merge_ts
			merged_t.insert(merged_t.end(), new_ts.ta.p.t.cbegin(), new_ts.ta.p.t.cend());
			merged_t.emplace_back(new_ts.ta.p.t_end);
			merged_v.insert(merged_v.end(), new_ts.v.cbegin(), new_ts.v.cend());
			// if new end BEFORE start of old  =>  insert NaN
			if (new_p.end < old_p.start) {
				merged_v.emplace_back(shyft::nan);
			}

			// new end BEFORE end of old  =>  read more from old
			if (new_p.end < old_p.end) {

				// determine first period in old NOT CONTAINING new.end
				auto old_begin = std::upper_bound(
					old_ta.p.t.cbegin(), old_ta.p.t.cend(),
					new_ts.ta.p.t_end);

				// store any trailing old time-points
				merged_t.insert(merged_t.end(), old_begin, old_ta.p.t.cend());
				merged_t.emplace_back(old_ta.p.t_end);
				// store portion of old values to keep
				std::size_t to_insert = std::distance(old_begin, old_ta.p.t.cend());
				// new end INSIDE of AT start of old  =>  insert value from old where new end
				if (new_p.end >= old_p.start) {
					to_insert += 1;
				}
				auto it = merged_v.insert(merged_v.end(), to_insert, 0.);
				std::fseek(fh, old_v_offset + (old_header.n - to_insert) * sizeof(double), SEEK_SET);
				read(fh, static_cast<void*>(&(*it)), to_insert * sizeof(double));
			}

			// write header
			ts_db_header new_header{
				old_header.point_fx, old_header.ta_type,
				static_cast<uint32_t>(merged_t.size() - 1),
				core::utcperiod{ merged_t.at(0), merged_t.at(merged_t.size() - 1) }
			};
			// -----
			std::fseek(fh, 0, SEEK_SET);  // seek to begining
			write(fh, static_cast<const void *>(&new_header), sizeof(ts_db_header));

			// write time-axis
			write(fh, static_cast<const void *>(&merged_t.at(merged_t.size() - 1)), sizeof(int64_t));
			write(fh, static_cast<const void *>(merged_t.data()), (merged_t.size() - 1) * sizeof(int64_t));

			// write values
			write(fh, static_cast<const void *>(merged_v.data()), merged_v.size() * sizeof(double));
		} break;
		}
	}
	void check_ta_alignment(std::FILE * fh, const ts_db_header & old_header, const time_axis::generic_dt & old_ta, const gts_t & ats) const {
		if (ats.fx_policy != old_header.point_fx) {
			throw std::runtime_error("dtss_store: cannot merge with different point interpretation");
		}
		// -----
		if (ats.ta.gt != old_header.ta_type) {
			throw std::runtime_error("dtss_store: cannot merge with different ta type");
		} else {
			// parse specific ta data to determine compatibility
			switch (old_header.ta_type) {
			case time_axis::generic_dt::FIXED:
			{
				if (old_ta.f.dt != ats.ta.f.dt || (old_ta.f.t - ats.ta.f.t) % old_ta.f.dt != 0) {
					throw std::runtime_error("dtss_store: cannot merge unaligned fixed_dt");
				}
			} break;
			case time_axis::generic_dt::CALENDAR:
			{
				if (ats.ta.c.cal->tz_info->tz.tz_name == old_ta.c.cal->tz_info->tz.tz_name) {
					core::utctimespan remainder;
					ats.ta.c.cal->diff_units(old_ta.c.t, ats.ta.c.t, old_ta.c.dt, remainder);
					if (old_ta.c.dt != ats.ta.c.dt || remainder != 0) {
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
		if (rsz != sz)
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
		if (t_start == core::no_utctime)
			t_start = core::min_utctime;
		if (t_end == core::no_utctime)
			t_end = core::max_utctime;

		// no overlap?
		if (h.data_period.end <= t_start || h.data_period.start >= t_end) {
			return ta;
		}

		skip_n = 0;
		core::utctime t0 = core::no_utctime;
		switch (h.ta_type) {
		case time_axis::generic_dt::FIXED: {
			// read start & step
			read(fh, static_cast<void *>(&t0), sizeof(core::utctime));
			read(fh, static_cast<void *>(&ta.f.dt), sizeof(core::utctimespan));
			// handle various overlaping periods
			if (t_start <= h.data_period.start && t_end >= h.data_period.end) {
				// fully around or exact
				ta.f.t = t0;
				ta.f.n = h.n;
			} else {
				std::size_t drop_n = 0;
				if (t_start > h.data_period.start)  // start inside
					skip_n = (t_start - h.data_period.start) / ta.f.dt;
				if (t_end < h.data_period.end)  // end inside
					drop_n = (h.data_period.end - t_end) / ta.f.dt;
				// -----
				ta.f.t = t0 + ta.f.dt*skip_n;
				ta.f.n = h.n - skip_n - drop_n;
			}
		} break;
		case time_axis::generic_dt::CALENDAR: {
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
			if (t_start <= h.data_period.start && t_end >= h.data_period.end) {
				// fully around or exact
				ta.c.t = t0;
				ta.c.n = h.n;
			} else {
				std::size_t drop_n = 0;
				if (t_start > h.data_period.start)  // start inside
					skip_n = ta.c.cal->diff_units(h.data_period.start, t_start, ta.c.dt);
				if (t_end < h.data_period.end)  // end inside
					drop_n = ta.c.cal->diff_units(t_end, h.data_period.end, ta.c.dt);
				// -----
				ta.c.t = ta.c.cal->add(t0, ta.c.dt, skip_n);
				ta.c.n = h.n - skip_n - drop_n;
			}
		} break;
		case time_axis::generic_dt::POINT: {
			if (t_start <= h.data_period.start && t_end >= h.data_period.end) {
				// fully around or exact
				ta.p.t.resize(h.n);
				read(fh, static_cast<void *>(&ta.p.t_end), sizeof(core::utctime));
				read(fh, static_cast<void *>(ta.p.t.data()), sizeof(core::utctime)*h.n);
			} else {
				core::utctime f_time = 0;
				std::vector<core::utctime> tmp(h.n, 0.);
				read(fh, static_cast<void *>(&f_time), sizeof(core::utctime));
				read(fh, static_cast<void *>(tmp.data()), sizeof(core::utctime)*h.n);
				// -----
				auto it_b = tmp.begin();
				if (t_start > h.data_period.start) {
					it_b = std::upper_bound(tmp.begin(), tmp.end(), t_start, std::less<core::utctime>());
					if (it_b != tmp.begin()) {
						std::advance(it_b, -1);
					}
				}
				// -----
				auto it_e = tmp.end();
				if (t_end < h.data_period.end) {
					it_e = std::upper_bound(it_b, tmp.end(), t_end, std::less<core::utctime>());
					if (it_e != tmp.end()) {
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
		switch (h.ta_type) {
		case time_axis::generic_dt::FIXED: {
			std::fseek(fh, 2 * sizeof(int64_t), SEEK_CUR);
		} break;
		case time_axis::generic_dt::CALENDAR: {
			std::fseek(fh, 2 * sizeof(int64_t), SEEK_CUR);
			uint32_t sz{};
			read(fh, static_cast<void*>(&sz), sizeof(uint32_t));
			std::fseek(fh, sz * sizeof(uint8_t), SEEK_CUR);
		} break;
		case time_axis::generic_dt::POINT: {
			std::fseek(fh, (h.n + 1) * sizeof(int64_t), SEEK_CUR);
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

}
}
