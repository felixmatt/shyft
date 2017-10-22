#pragma once
#ifdef SHYFT_NO_PCH
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <memory>
#include <utility>
#include <functional>
#include <cstring>

#include <boost/filesystem.hpp>
namespace fs=boost::filesystem;

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


#endif // SHYFT_NO_PCH
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
        using namespace std;
        using shyft::core::utctime;
        using shyft::core::utcperiod;
        using shyft::core::utctimespan;
        using shyft::core::no_utctime;
        using shyft::core::calendar;
        using shyft::core::deltahours;
        using shyft::api::apoint_ts;
        using shyft::api::gpoint_ts;
        using shyft::api::gts_t;
        using shyft::api::gta_t;
        using shyft::api::aref_ts;

        enum class message_type:uint8_t {
            SERVER_EXCEPTION,
            EVALUATE_TS_VECTOR,
            EVALUATE_TS_VECTOR_PERCENTILES,
            FIND_TS,
            STORE_TS,
            // EVALUATE_TS_VECTOR_HISTOGRAM //-- tsv,period,ta,bin_min,bin_max -> ts_vector[n_bins]
        };

        namespace msg {
            template <class T>
            message_type read_type(T&in) {
                int32_t mtype;
                in.read((char*)&mtype,sizeof(mtype));
                return (message_type) mtype;
            }

            template <class T>
            void write_type(message_type mt,T&out) {
                int32_t mtype=(int32_t)mt;
                out.write((char const *)&mtype,sizeof(mtype));
            }



            template <class T>
            void write_string(string const&s,T& out) {
                int32_t sz=s.size();
                out.write((char const*)&sz,sizeof(sz));
                out.write(s.data(),sz);
            }
            template <class T>
            string read_string(T& in) {
                int32_t sz;
                in.read((char*)&sz, sizeof(sz));
                string msg(sz, '\0');
                in.read((char*)msg.data(), sz);
                return msg;
            }

            template<class T>
            void write_exception(exception const & e,T& out) {
                int32_t sz=strlen(e.what());
                out.write((char const*)&sz,sizeof(sz));
                out.write(e.what(),sz);
            }

            template<class T>
            void send_exception(exception const & e,T& out) {
                write_type(message_type::SERVER_EXCEPTION,out);
                int32_t sz=strlen(e.what());
                out.write((char const*)&sz,sizeof(sz));
                out.write(e.what(),sz);
            }

            template<class T>
            std::runtime_error read_exception(T& in) {
                int32_t sz;
                in.read((char*)&sz,sizeof(sz));
                string msg(sz,'\0');
                in.read((char*)msg.data(),sz);
                return std::runtime_error(msg);
            }

        }

        using ts_vector_t = shyft::api::ats_vector;
        typedef vector<ts_info> ts_info_vector_t;
        typedef vector<string> id_vector_t;
        typedef std::function< ts_vector_t (id_vector_t const& ts_ids,utcperiod p)> read_call_back_t;
        typedef std::function< void (const ts_vector_t& )> store_call_back_t;
        typedef std::function< ts_info_vector_t(string search_expression)> find_call_back_t;

        extern string shyft_prefix;//="shyft://";///< marks all internal handled urls

        /**construct a shyft-url from container and ts-name */
        inline string shyft_url(const string&container,const string& ts_name) {
            return shyft_prefix+container+"/"+ts_name;
        }

        /** match & extract fast the following 'shyft://<container>/'
        * \param url like pattern above
        * \return <container> or empty string if no match
        */
        inline string extract_shyft_url_container(const string&url) {
            if( (url.size() < shyft_prefix.size()+2) || !std::equal(begin(shyft_prefix), end(shyft_prefix), begin(url)))
                return string{};
            auto ce=url.find_first_of('/',shyft_prefix.size());
            if(ce==string::npos)
                return string{};
            return url.substr(shyft_prefix.size(),ce-shyft_prefix.size());
        }


        /** \brief storage layer header-record
        *
        * The storage layer header-record contains enough
        * information to
        *  a) identify the format version (and thus the rest of the layout)
        *  b) provide  minimal quick summary information that could ease search
        *  c) vital features of the ts that is invariant
        *
        *
        * \note this is a packed record, should portable through small-endian cpu-arch.
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
            ts_db_header()=default;
            char signature[4]={'T','S','1','\0'}; ///< signature header with version #
            time_series::ts_point_fx point_fx=time_series::POINT_AVERAGE_VALUE; ///< point_fx feature
            time_axis::generic_dt::generic_type ta_type=time_axis::generic_dt::FIXED; ///< time-axis type
            uint32_t n=0; ///< number of points in the time-series (time-axis and values)
            utcperiod data_period; ///< [from..until> period range
            ts_db_header(time_series::ts_point_fx point_fx,time_axis::generic_dt::generic_type ta_type,uint32_t n,utcperiod data_period)
            :point_fx(point_fx),ta_type(ta_type),n(n),data_period(data_period) {}
        };
        #pragma pack(pop)

        /** \brief A simple file-io based internal time-series storage for the dtss
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
            string root_dir; ///< root_dir points to the top of the container
            ts_db()=default;

            /** constructs a ts_db with specified container root */
            explicit ts_db(const string&root_dir):root_dir(root_dir) {
                if(!fs::is_directory(root_dir)) {
                    if(!fs::exists(root_dir)) {
                        if(!fs::create_directory(root_dir)) {
                            throw runtime_error(string("ts_db: failed to create root directory :")+root_dir);
                        }
                    } else {
                        throw runtime_error(string("ts_db: designated root directory is not a directory:")+root_dir);
                    }
                }
                make_calendar_lookups();
            }
			//mutable double t_open=0;
			//mutable double t_write = 0;
			//mutable double t_close = 0.0;

            /** save a ts to file, *overwrite* any existing file with contents
            *
            * \note windows ntfs is incredibly slow on close, so as a workaround using
            *      handles and separate thread for the close-job for now.
            */
            void save(const string &fn,const gts_t& gts, bool win_thread_close=true) const {
                using namespace std;
				auto fp = make_full_path(fn, true);
				//auto t0 = timing::now();
				int f = open(fp.c_str(), O_WRONLY | O_BINARY | O_TRUNC | O_CREAT|O_SEQUENTIAL,S_IWRITE|S_IREAD);
				//auto t1 = timing::now();
                //x ofstream f(, ios_base::out|ios_base::binary|ios_base::trunc); // yes on linux this works ok, but windows ... slow
                write_ts(f,gts);
				//auto t2 = timing::now();
				#ifdef _WIN32
				if(win_thread_close) {
				    std::thread(close, f).detach();// allow time-consuming close to work in background
				} else {
				    close(f); // takes forever in windows, by design
				}
				#else
				    close(f);
				#endif
				//x f.close(); // actually, it's the close call on windows that is slow, we need to use the handle to easily transfer it to a close-thread
				//auto t3 = timing::now();
				//t_open += elapsed_us(t0, t1)/1000000.0;
				//t_write += elapsed_us(t1, t2) / 1000000.0;
				//t_close += elapsed_us(t2, t3) / 1000000.0;
            }

            /** read a ts from specified file */
            gts_t read(const string &fn, utcperiod) const {
                using namespace std;
                ifstream f(make_full_path(fn), ios_base::in|ios_base::binary);
                return read_ts(f);
            }

            /** removes a ts from the container */
            void remove(const string&fn) const {
                auto fp = make_full_path(fn);
                for (size_t retry = 0; retry < 10; ++retry) {
                    try {
                        fs::remove(fp);
                        return;
                    }
                    catch (...) { // windows usually fails, due to delayed file-close/file-release so we retry 10 x 0.3 seconds
                        this_thread::sleep_for(chrono::duration<int, std::milli>(300));
                    }
                }
                throw runtime_error("failed to remove file '" + fp + "' after 10 repeated attempts lasting for 3 seconds");
            }

            /** get minimal ts-information from specified fn */
            ts_info get_ts_info(const string &fn) const {
                auto fp=make_full_path(fn);
                ifstream f(fp,ios_base::in|ios_base::binary);
                auto h=read_header(f);
                ts_info i;
                i.name=fn;
                i.point_fx=h.point_fx;
                i.modified = utctime(fs::last_write_time(fp));
                i.data_period = h.data_period;
                // consider time-axis type info, dt.. as well
                return i;
            }

            /** find all ts_info s that matches the specified re match string
            *
            * e.g.: match= 'hydmet_station/.*_id/temperature'
            *    would find all time-series /hydmet_station/xxx_id/temperature
            */
            vector<ts_info> find(const string& match) const {
                fs::path root(root_dir);
                vector<ts_info> r;
                regex r_match(match,std::regex_constants::ECMAScript|std::regex_constants::icase);
                for(auto&& x : fs::recursive_directory_iterator(root)) {
                    if(fs::is_regular(x.path())) {
                        string fn = x.path().lexically_relative(root).generic_string(); // x.path() except root-part
                        if(regex_search(fn,r_match)) {
                            r.push_back(get_ts_info(fn)); // TODO: maybe multi-core this into a job-queue
                        }
                    } else if(fs::is_directory(x.path())) {
                        // TODO: elide recursion into the x, calling
                        // if x.path is not part of match
                        //   x.no_push();
                    }
                }
                return r;
            }
          private:
            shared_ptr<calendar> lookup_calendar(const string&tz) const {
                auto f= calendars.find(tz);
                if( f==calendars.end())
                    return make_shared<calendar>(tz);
                return (*f).second;
            }
            map<string,shared_ptr<calendar>> calendars;

            void make_calendar_lookups() {
                for(int hour=-11;hour<12;hour++) {
                    auto c = make_shared<calendar>(deltahours(hour));
                    calendars[c->tz_info->name()] = c;
                }
            }
            string make_full_path(const string &fn,bool create_paths=false) const {
                fs::path fn_path(fn);
                if (fn_path.is_absolute()) return fn_path.string(); // questionable: should we allow outside container specs?
				auto root = fs::path(root_dir);
                auto r =fs::path(root_dir)/fn_path;
                if(create_paths) {
                    auto rp =r.parent_path();
					if(rp.compare(root)>0) // if fn contains sub-directory, we have to check that it exits
						if(!fs::is_directory(rp))
							fs::create_directories(rp);
                }
                return r.string();
            }
            ts_db_header mk_header(const gts_t& ts) const {return ts_db_header{ts.point_interpretation(),ts.time_axis().gt,uint32_t(ts.size()),ts.total_period()};}

            #if 0
            // When we can fix the windows-slowness bug, we can use stream for all ops
            void write_header(ofstream&f,const gts_t&ats) const {auto h=mk_header(ats);f.write((const char*)&h,sizeof(h));}
            void write_time_axis(ofstream&f, const gta_t& ta) const {
                switch(ta.gt) {
                    case time_axis::generic_dt::FIXED: {
                        f.write((const char*)&ta.f.t,sizeof(utctime));
                        f.write((const char*)&ta.f.dt,sizeof(utctimespan));
                    } break;
                    case time_axis::generic_dt::CALENDAR: {
                        f.write((const char*)&ta.c.t,sizeof(utctime));
                        f.write((const char*)&ta.c.dt,sizeof(utctimespan));
                        string tz=ta.c.cal->tz_info->name();
                        uint32_t sz=tz.size();
                        f.write((const char*)&sz,sizeof(uint32_t));
                        f.write((const char*)tz.c_str(),sz);
                    } break;
                    case time_axis::generic_dt::POINT: {
                        f.write((const char*)&ta.p.t_end,sizeof(utctime));
                        f.write((const char*)ta.p.t.data(),sizeof(utctime)*ta.p.t.size());
                    }break;
                }
            }
            void write_values(ofstream&f, const vector<double>&v) const {f.write((const char*)v.data(),sizeof(double)*v.size());}
            void write_ts(ofstream &f, const gts_t &ats) const {
                write_header(f,ats);
                write_time_axis(f,ats.ta);
                write_values(f,ats.v);
            }
            #endif
            void write_header(int f, const gts_t&ats) const { auto h = mk_header(ats); write(f,(const char*)&h, sizeof(h)); }
            inline void write(int f,const void*d,size_t sz) const {if((size_t)::write(f,d,sz)!=sz) throw std::runtime_error("dtss_store:failed to write do disk");}
			void write_time_axis(int f, const gta_t& ta) const {
				switch (ta.gt) {
				case time_axis::generic_dt::FIXED: {
					write(f,(const char*)&ta.f.t, sizeof(utctime));
					write(f,(const char*)&ta.f.dt, sizeof(utctimespan));
				} break;
				case time_axis::generic_dt::CALENDAR: {
					write(f,(const char*)&ta.c.t, sizeof(utctime));
					write(f,(const char*)&ta.c.dt, sizeof(utctimespan));
					string tz = ta.c.cal->tz_info->name();
					uint32_t sz = tz.size();
					write(f,(const char*)&sz, sizeof(uint32_t));
					write(f,(const char*)tz.c_str(), sz);
				} break;
				case time_axis::generic_dt::POINT: {
					write(f,(const char*)&ta.p.t_end, sizeof(utctime));
					write(f,(const char*)ta.p.t.data(), sizeof(utctime)*ta.p.t.size());
				}break;
				}
			}


			void write_values(int f, const vector<double>&v) const { write(f,(const char*)v.data(), sizeof(double)*v.size()); }


			void write_ts( int f, const gts_t &ats) const {
				write_header(f, ats);
				write_time_axis(f, ats.ta);
				write_values(f, ats.v);
			}

            ts_db_header read_header(ifstream&f) const {
                ts_db_header h;
                f.read((char*)&h,sizeof(ts_db_header));
                return h;
            }

            gta_t read_time_axis(ifstream& f, const ts_db_header&h) const {
                gta_t ta;
                ta.gt=h.ta_type;
                switch(h.ta_type) {
                    case time_axis::generic_dt::FIXED: {
                        f.read((char*)&ta.f.t,sizeof(utctime));
                        f.read((char*)&ta.f.dt,sizeof(utctimespan));
                        ta.f.n = h.n;
                    } break;
                    case time_axis::generic_dt::CALENDAR: {
                        f.read((char*)&ta.c.t,sizeof(utctime));
                        f.read((char*)&ta.c.dt,sizeof(utctimespan));
                        ta.c.n=h.n;
                        uint32_t sz{0};
                        f.read((char*)&sz,sizeof(uint32_t));
                        string tz(sz,'\0');
                        f.read((char*)tz.data(),sz);
                        ta.c.cal=lookup_calendar(tz);

                    } break;
                    case time_axis::generic_dt::POINT: {
                        f.read((char*)&ta.p.t_end,sizeof(utctime));
                        ta.p.t.resize(h.n);
                        f.read((char*)ta.p.t.data(),sizeof(utctime)*h.n);
                    }break;
                }
                return ta;
            }
            vector<double> read_values(ifstream& f,const ts_db_header& h) const {
                vector<double> v(h.n,0.0);
                f.read((char*)v.data(),sizeof(double)*h.n);
                return v;
            }
            gts_t read_ts(ifstream& f) const {
                auto h=read_header(f);
                auto ta= read_time_axis(f,h);
                auto v = read_values(f,h);
                return gts_t{move(ta),move(v),h.point_fx};
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
            map<string,ts_db> container;///< mapping of internal shyft <container> -> ts_db
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
            void add_container(const string &container_name,const string& root_dir) {
                container[container_name]=ts_db(root_dir);
            }

            const ts_db& internal(const string& container_name) const {
                auto f=container.find(container_name);
                if(f == end(container))
                    throw runtime_error(string("Failed to find shyft container:")+container_name);
                return f->second;
            }
            //-- expose cache functions
            // or just leak them through ts_cache ?
            void add_to_cache(id_vector_t&ids, ts_vector_t& tss) { ts_cache.add(ids,tss);}

            void remove_from_cache(id_vector_t &ids) { ts_cache.remove(ids);}

            cache_stats get_cache_stats() { return ts_cache.get_cache_stats();}

            void clear_cache_stats() { ts_cache.clear_cache_stats();}

            void flush_cache() { return ts_cache.flush();}

            void set_cache_size(size_t max_size) { ts_cache.set_capacity(max_size);}

            void set_auto_cache(bool active) { cache_all_reads=active;}

            size_t get_cache_size() const {return ts_cache.get_capacity();}

            ts_info_vector_t do_find_ts(const string& search_expression) {
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

            string extract_url(const apoint_ts&ats) const {
                auto rts = dynamic_pointer_cast<aref_ts>(ats.ts);
                if(rts)
                    return rts->id;
                throw runtime_error("dtss store.extract_url:supplied type must be of type ref_ts");
            }

            void do_cache_update_on_write(const ts_vector_t&tsv) {
                for (size_t i = 0; i<tsv.size(); ++i) {
                    auto rts = dynamic_pointer_cast<aref_ts>(tsv[i].ts);
                    ts_cache.add(rts->id, apoint_ts(rts->rep));
                }
            }
            void do_store_ts(const ts_vector_t&tsv,bool cache_on_write) {
                if(tsv.size()==0) return;
                // 1. filter out all shyft://<container>/<ts-path> elements
                //    and route these to the internal storage controller (threaded)
                //    map<string,ts_db> shyft_internal;
                //
                vector<size_t> other;other.reserve(tsv.size());
                for(size_t i=0;i<tsv.size();++i) {
                    auto rts = dynamic_pointer_cast<aref_ts>(tsv[i].ts);
                    if(!rts) throw runtime_error("dtss store: require ts with url-references");
                    auto c= extract_shyft_url_container(rts->id);
                    if(c.size()) {
                        internal(c).save(rts->id.substr(shyft_prefix.size()+c.size()+1),rts->core_ts());
                        if (cache_on_write) { // ok, this ends up in a copy, and lock for each item(can be optimized if many)
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
                vector<size_t> other;other.reserve(ts_ids.size());
                // 1. filter out shyft://
                //    if all shyft: return internal read
                for(size_t i=0;i<ts_ids.size();++i) {
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
                        throw runtime_error("dtss: read-request to external ts, without external handler");
                    if(other.size()==ts_ids.size()) {// only other series, just return result
                        auto rts= bind_ts_cb(ts_ids,p);
                        if(cache_all_reads) ts_cache.add(ts_ids,rts);
                        return rts;
                    }
                    vector<string> o_ts_ids;o_ts_ids.reserve(other.size());
                    for(auto i:other) o_ts_ids.push_back(ts_ids[i]);
                    auto o=bind_ts_cb(o_ts_ids,p);
                    if(cache_all_reads) ts_cache.add(o_ts_ids,o);
                    // if 1 and 2, merge into one ordered result vector
                    //
                    for(size_t i=0;i<o.size();++i)
                        r[other[i]]=o[i];
                }
                return r;
            }

            void
            do_bind_ts(utcperiod bind_period, ts_vector_t& atsv)  {
                std::map<string, vector<api::ts_bind_info>> ts_bind_map;
                vector<string> ts_id_list;
                // step 1: bind not yet bound time-series ( ts with only symbol, needs to be resolved using bind_cb)
                for (auto& ats : atsv) {
                    auto ts_refs = ats.find_ts_bind_info();
                    for (const auto& bi : ts_refs) {
                        if (ts_bind_map.find(bi.reference) == ts_bind_map.end()) { // maintain unique set
                            ts_id_list.push_back(bi.reference);
                            ts_bind_map[bi.reference] = vector<api::ts_bind_info>();
                        }
                        ts_bind_map[bi.reference].push_back(bi);
                    }
                }

                // step 2: (optional) bind_ts callback should resolve symbol time-series with content
                if (ts_bind_map.size()) {
                    auto bts = do_read(ts_id_list, bind_period);
                    if (bts.size() != ts_id_list.size())
                        throw std::runtime_error(string("failed to bind all of ") + std::to_string(bts.size()) + string(" ts"));

                    for (size_t i = 0;i < ts_id_list.size();++i) {
                        for (auto &bi : ts_bind_map[ts_id_list[i]])
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
            do_evaluate_percentiles(utcperiod bind_period, ts_vector_t& atsv, api::gta_t const&ta,vector<int> const& percentile_spec) {
                do_bind_ts(bind_period, atsv);
                return api::percentiles(atsv, ta, percentile_spec);// we can assume the result is trivial to serialize
            }

            void on_connect(
                std::istream& in,
                std::ostream& out,
                const string& foreign_ip,
                const string& local_ip,
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
                                vector<int> percentile_spec;
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
                                string search_expression; {
                                    search_expression = msg::read_string(in);// >> search_expression;
                                }
                                auto find_result = do_find_ts(search_expression); {
                                    msg::write_type(message_type::FIND_TS, out);
                                    boost::archive::binary_oarchive oa(out);
                                    oa << find_result;
                                }
                            } break;
                            case message_type::STORE_TS: {
								ts_vector_t rtsv; bool cache_on_write{false}; {
                                    boost::archive::binary_iarchive ia(in);
                                    ia>>rtsv>>cache_on_write;
                                }
                                do_store_ts(rtsv,cache_on_write); {
                                    msg::write_type(message_type::STORE_TS,out);
                                }
                            } break;
                            default:
                                throw std::runtime_error(string("Got unknown message type:") + std::to_string((int)msg_type));
                        }
                    } catch (exception const& e) {
                        msg::send_exception(e,out);
                    }
                }
                //std::cout<<"of conn:"<<foreign_ip<<":"<<foreign_port<<", local_port ="<<local_port<<std::endl;

            }
        };

        struct client {
            dlib::iosockstream io;
            string host_port;
            explicit client(const string& host_port):io(host_port),host_port(host_port) {}

            void reopen() {
                io.close();
                io.open(host_port);
            }
            void close(int timeout_ms=1000) {io.close(timeout_ms);}

            vector<apoint_ts>
            percentiles(ts_vector_t const& tsv, utcperiod p,api::gta_t const&ta,const vector<int>& percentile_spec) {
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
                throw std::runtime_error(string("Got unexpected response:") + std::to_string((int)response_type));
            }

            vector<apoint_ts>
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
                throw std::runtime_error(string("Got unexpected response:")+std::to_string((int)response_type));
            }
            void store_ts(const ts_vector_t &tsv,bool cache_on_write) {
                if (tsv.size()==0)
                    return; //trivial and considered valid case
                // verify that each member of tsv is a gpoint_ts
                for(auto const &ats:tsv) {
                    auto rts = dynamic_cast<api::aref_ts*>(ats.ts.get());
                    if(!rts) throw std::runtime_error(string("attempt to store a null ts"));
                    if(rts->needs_bind()) throw std::runtime_error(string("attempt to store unbound ts:")+rts->id);
                }
                msg::write_type(message_type::STORE_TS,io);{
                    boost::archive::binary_oarchive oa(io);
                    oa<<tsv<<cache_on_write;
                }
                auto response_type= msg::read_type(io);
                if(response_type==message_type::SERVER_EXCEPTION) {
                    auto re= msg::read_exception(io);
                    throw re;
                } else if(response_type==message_type::STORE_TS) {
                    return ;
                }
                throw std::runtime_error(string("Got unexpected response:")+std::to_string((int)response_type));
            }

            ts_info_vector_t find(const string& search_expression) {
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
                throw std::runtime_error(string("Got unexpected response:") + std::to_string((int)response_type));
            }

        };

        inline vector<apoint_ts> dtss_evaluate(const string& host_port, ts_vector_t const& tsv, utcperiod p,int timeout_ms=10000) {
            return client(host_port).evaluate(tsv,p);
        }
        inline vector<apoint_ts> dtss_percentiles(const string& host_port, ts_vector_t const& tsv, utcperiod p,api::gta_t const&ta,vector<int> percentile_spec,int timeout_ms=10000) {
            return client(host_port).percentiles(tsv,p,ta,percentile_spec);
        }


    }
}
