#include "core_pch.h"
#include "utctime_utilities.h"
#include <ostream>
#include <cstring>
#include <boost/date_time/local_time/local_time.hpp>

namespace shyft {
    namespace core {
        // python exposure needs definition here (even if's a constant)
			const utctimespan calendar::YEAR;//=365*24*3600L;
			const utctimespan calendar::MONTH;//=30*24*3600L;
			const utctimespan calendar::QUARTER;
			const utctimespan calendar::WEEK;// = 7*24*3600L;
			const utctimespan calendar::DAY;// =  1*24*3600L;
			// these are just timespan constants with no calendar semantics
			const utctimespan calendar::HOUR;// = 3600L;
			const utctimespan calendar::MINUTE;// = 60L;
			const utctimespan calendar::SECOND;// = 1L;
        using namespace std;
        std::ostream& operator<<(std::ostream& os, const utcperiod& p) {
            os << p.to_string();
            return os;
        }

        string utcperiod::to_string() const {
            calendar utc;
            return utc.to_string(*this);
        }

        string calendar::to_string(utctime t) const {
            char s[100];
            switch(t) {
                case no_utctime:sprintf(s, "no_utctime"); break;
                case min_utctime:sprintf(s, "-oo"); break;
                case max_utctime:sprintf(s, "+oo"); break;
                default: {
                    auto c = calendar_units(t);
                    auto tz= tz_info->utc_offset(t);
                    auto tz_hours= int(tz/deltahours(1));
                    auto tz_minutes= int(abs(tz-tz_hours*deltahours(1))/deltaminutes(1));
                    char tzs[100];
                    if(tz) {
                        if(tz_minutes)
                            sprintf(tzs,"%+03d:%02d",tz_hours,tz_minutes);
                        else
                            sprintf(tzs,"%+03d",tz_hours);
                    } else {
                        strcpy(tzs,"Z");
                    }
                    sprintf(s, "%04d-%02d-%02dT%02d:%02d:%02d%s", c.year, c.month, c.day, c.hour, c.minute, c.second,tzs);
                }break;
            }
            return string(s);
        }

        string calendar::to_string(utcperiod p) const {
            if(p.valid()) {
                return string("[") + to_string(p.start) + "," + to_string(p.end) + ">";
            }
            return string("[not-valid-period>");
        }

        // returns sun=0, mon=1 etc.. based on code from boost greg.date
        static inline int day_of_week_idx(YMDhms const&c) {
            unsigned short a = static_cast<unsigned short>((14 - c.month) / 12);
            unsigned short y = static_cast<unsigned short>(c.year - a);
            unsigned short m = static_cast<unsigned short>(c.month + 12 * a - 2);
            return static_cast<int>((c.day + y + (y / 4) - (y / 100) + (y / 400) + (31 * m) / 12) % 7);
        }

        // return iso weekday, mon=1, sun=7;
        static inline int iso_week_day(YMDhms const&c) {
            auto wd = day_of_week_idx(c);
            return wd == 0 ? 7 : wd;
        }

        // returns the day-number trimmed to start of iso-week (trunc,round downwards)
        static inline unsigned long trim_day_number_to_week(unsigned long jdn) {
            return 7 * (jdn / 7);// jd starts on monday.
        }

        utctime calendar::time(YMDhms c) const {
            if(c.is_null()) return no_utctime;
            if(c==YMDhms::max()) return max_utctime;
            if(c==YMDhms::min()) return min_utctime;
            if (!c.is_valid_coordinates())
                throw std::runtime_error("calendar.time with invalid YMDhms coordinates attempted");

            utctime r= ((int(day_number(c)) - UnixDay)*DAY) + seconds(c.hour, c.minute, c.second);
            auto utc_diff_1= tz_info->utc_offset(r);// detect if we are in the dst-shift hour
            auto utc_diff_2= tz_info->utc_offset(r-utc_diff_1);
            return (utc_diff_1==utc_diff_2)?r-utc_diff_1: r-utc_diff_2;
        }

        utctime calendar::time_from_week(int Y, int W, int wd, int h, int m, int s) const {
            return time(YWdhms(Y, W, wd, h, m, s));
        }
        utctime calendar::time(YWdhms c) const {
            if (c.is_null()) return no_utctime;
            if (c == YWdhms::max()) return max_utctime;
            if (c == YWdhms::min()) return min_utctime;
            if (!c.is_valid_coordinates())
                throw std::runtime_error("calendar.time with invalid YWdhms coordinates attempted");

            // figure out utc-time of week 1:
            utctime t = int(day_number(YMDhms(c.iso_year, 1, 14)) - UnixDay)*DAY;
            auto jdn = day_number(t);
            auto cw = from_day_number(jdn);
            cw.month = 1;cw.day = 1;// round/trim down to year.1.1
            auto w1_daynumber = trim_day_number_to_week(day_number(cw));
            auto cy = from_day_number(w1_daynumber);
            if (cy.month == 12 && cy.day < 29)
                    w1_daynumber += 7;
            // then just add relative weeks days from that
            auto w_daynumber = w1_daynumber + 7 *(c.iso_week-1) + (c.week_day - 1);
            utctime r = (int(w_daynumber) - UnixDay)*DAY + seconds(c.hour, c.minute, c.second);// make it local utc-time
            // map it back to true utc-time (subtract the tz-offset at the time)
            auto utc_diff_1 = tz_info->utc_offset(r);// detect if we are in the dst-shift hour
            auto utc_diff_2 = tz_info->utc_offset(r - utc_diff_1);
            return (utc_diff_1 == utc_diff_2) ? r - utc_diff_1 : r - utc_diff_2;

        }

        template <class C>
        static inline C& fill_in_hms_from_t(utctime t, C& r) {
            utctime tj = calendar::UnixSecond + t;
            utctime td = calendar::DAY*(tj / calendar::DAY);
            utctimespan dx = tj - td;// n seconds this day.
            r.hour = int(dx / calendar::HOUR);
            dx -= r.hour*calendar::HOUR;
            r.minute = int(dx / calendar::MINUTE);
            r.second = int(dx % calendar::MINUTE);
            return r;
        }

        YMDhms  calendar::calendar_units(utctime t) const {
            if (t == no_utctime)  return YMDhms();
            if (t == max_utctime) return YMDhms::max();
            if (t == min_utctime) return YMDhms::min();
            auto tz_dt=tz_info->utc_offset(t);
            t += tz_dt;
            auto jdn = day_number(t);
            auto x = from_day_number(jdn);
            YMDhms r;
            r.year = x.year; r.month = x.month; r.day = x.day;
            return fill_in_hms_from_t(t, r);
        }

        YWdhms calendar::calendar_week_units(utctime t) const {
            if (t == no_utctime)  return YWdhms();
            if (t == max_utctime) return YWdhms::max();
            if (t == min_utctime) return YWdhms::min();
            auto tz_dt = tz_info->utc_offset(t);
            t += tz_dt;
            auto jdn = day_number(t);
            auto c = from_day_number(jdn);
            YWdhms r;
            r.week_day = iso_week_day(c);
            auto jdnw = trim_day_number_to_week(jdn);
            c = from_day_number(jdnw);
            if (c.month == 12 && c.day >= 29) { // we are lucky: it has to be week one
                r.iso_year = c.year + 1; // but iso-year start the year before, so plus one
                r.iso_week = 1;
            } else if (c.month == 1 && c.day <= 4) { // still lucky, it's week one
                r.iso_year = c.year;
                r.iso_week = 1;
            } else { // since week one is sorted out above, this is week 2 and up.
                c.month = 1;c.day = 1; // round/trim down to year.1.1 and we are sure we are on the same (iso) year
                auto w1_daynumber = trim_day_number_to_week(day_number(c));
                auto cy = from_day_number(w1_daynumber); // now figure out correct iso-week start
                if (cy.month == 12&& cy.day < 29) //iso week 1 start next week
                    w1_daynumber += 7; // step up.
                //c = from_day_number(w1_daynumber + 14);// just pick year for week 2
                r.iso_year = c.year; //we are on week 2 or more so safely use c.year
                r.iso_week = 1 + (jdn - w1_daynumber) / 7;
            }
            return fill_in_hms_from_t(t,r);
        }

        int calendar::day_of_week(utctime t) const {
            if (t == no_utctime || t == min_utctime || t == max_utctime)
                return -1;
            return day_of_week_idx(calendar_units(t));
        }
        ///< returns day_of year, 1..365..
        size_t calendar::day_of_year(utctime t) const {
            if (t == no_utctime || t == max_utctime || t == min_utctime) return -1;
            auto x = calendar_units(t );
            YMDhms y(x.year, 1, 1);
            return 1 + day_number(x) - day_number(y);
        }
        ///< returns the month of t, 1..12, -1 of not valid time
        int calendar::month(utctime t) const {
            if (t == no_utctime || t == max_utctime || t == min_utctime) return -1;
            return calendar_units(t ).month;
        }
        ///< month to quarter for the trim function
        static int mq[12] = { 1, 1, 1, 4, 4, 4, 7, 7, 7, 10, 10, 10 };
        int calendar::quarter(utctime t) const {
            if (t == no_utctime || t == max_utctime || t == min_utctime) return -1;
            auto cu = calendar_units(t);
            return 1+ mq[cu.month - 1]/3;
        }
        utctime calendar::trim(utctime t, utctimespan deltaT) const {
            if (t == no_utctime || t == min_utctime || t == max_utctime || deltaT==utctimespan(0))
                return t;
            switch (deltaT) {
            case YEAR: {
                auto c = calendar_units(t);
                c.month = c.day = 1; c.hour = c.minute = c.second = 0;
                return time(c);
            }break;
            case QUARTER: {
                auto c = calendar_units(t);
                return time(YMDhms(c.year, mq[c.month - 1], 1));
            }break;
            case MONTH: {
                auto c = calendar_units(t);
                c.day = 1; c.hour = c.minute = c.second = 0;
                return time(c);
            }break;
            case DAY: {
                auto c = calendar_units(t);
                c.hour = c.minute = c.second = 0;
                return time(c);
            }break;
            }
            auto tz_offset=tz_info->utc_offset(t);
            const utctime t0 = utctime(+3LL * DAY) + utctime(WEEK * 52L * 2000LL);// + 3 days to get to a isoweek monday, then add 2000 years to get a positive number
            t += t0 + tz_offset;
            t= deltaT*(t / deltaT) - t0 ;
            return  t - tz_info->utc_offset(t);
        }

        utctime calendar::add(utctime t, utctimespan deltaT, long n) const {
            auto dt=n*deltaT;
            switch (deltaT) {
                case YEAR: {
                    auto c=calendar_units(t);
                    c.year += int(dt/YEAR);// gives signed number of units.
                    return time(c);
                } break;
                case QUARTER://just let it appear as 3xmonth
                    deltaT = MONTH;
                    n = 3 * n;
                    // fall through to month
                case MONTH:{
                    auto c=calendar_units(t);
                    // calculate number of years..
                    int nyears= int(dt/MONTH/12); // with correct sign
                    c.year +=  nyears; // done with years, now single step remaining months.
                    int nmonths= n-nyears*12;// remaining months to deal with
                    c.month += nmonths;// just add them
                    if(c.month <1) {c.month+=12; c.year--;}// then repair underflow
                    else if(c.month >12 ) {c.month -=12;c.year++;}// or overflow
                    return time(c);
                } break;
                default: break;
            }
            utctime r = t + dt; //naive first estimate
            auto utc_diff_1=tz_info->utc_offset(t);
            auto utc_diff_2=tz_info->utc_offset(r);
            return r + (utc_diff_1-utc_diff_2);
            // explanation: if t and r are in different dst, compensate for difference
            // e.g.: t+ calendar day, in dst, will be 23, 24 or 25 hours long, this way
            //  25: t= mm:dd   00:00, utcdiff(t) is +2h, add 24h
            //      r= mm:dd+1 00:00, utcdiff(r) is +1h
            //      ret r +( 2h - 1h), e.g. add one hour and we get 25h day.
            //  24: trivial utcdiff before and after are same, 0.
            //
            //  23: t= mm:dd   00:00 utcdiff(t) is +1h, then we add 24h
            //      r= mm:dd+1 00:00 utcdiff(r) is +2h (since we are in summer time)
            //      ret r +( 1h - 2h), eg. sub one hour, and we get a 23h day.
        };

        utctimespan calendar::diff_units(utctime t1, utctime t2, utctimespan deltaT, utctimespan &remainder) const {
            if (t1 == no_utctime || t2 == no_utctime || deltaT == 0L) {
                remainder = 0L;
                return 0L;
            }
            int sgn=1;
            if(t1>t2) {sgn=-1;swap(t1,t2);}
            utctimespan n_units= (t2-t1)/deltaT;// a *rough* estimate in case of dst or month/year
            // recall that this should give n_unit=1, remainder=0 for dst days 23h | 25h etc.
            // and that it should be complementary to the add(t,dt,n) function.
            if (deltaT < DAY) {
                if (deltaT > HOUR) {// a bit tricky, tz might jeopardize interval size by 1h (again assuming dst=1h)
                    auto utc_diff_1 = tz_info->utc_offset(t1);//assuming dst=1h!
                    auto utc_diff_2 = tz_info->utc_offset(t2);//assuming dst=1h!
                    n_units = (t2 - t1 - (utc_diff_1 - utc_diff_2)) / deltaT;
                    remainder = t2 - add(t1, deltaT, n_units);
                } else { // simple case, hour< 15 min etc.
                    remainder = t2 - (t1 + n_units*deltaT);
                }
            } else {
                if (deltaT == MONTH) {
                    n_units -= n_units / 72; // MONTH is 30 days, a real one  is ~ 30.4375, after 72 months this adds up to ~ 1 month error
                } else if (deltaT == QUARTER) {
                    n_units -=  n_units /(3* 72);// 3x month compensation
                } else if (deltaT == YEAR) {
                    n_units -= n_units / (4 * 365 * 365);// YEAR is 365, real is ~365.25, takes 5332900 years before it adds up to 1 year
                }
                auto t2x=add(t1,deltaT,long(n_units));
                if(t2x>t2) { // got one step to far
                    --n_units;// reduce n_units, and calculate remainder
                    remainder= t2 - add(t1,deltaT,long(n_units));
                } else if(t2x<t2) {// got one step short, there might be a hope for one more
                    ++n_units;// try adding one unit
                    auto t2xx= add(t1,deltaT,long(n_units));//calc new t2x
                    if(t2xx>t2) { // to much ?
                        --n_units;// reduce, and
                        remainder = t2 - t2x;// remainder is as with previous t2x
                    } else {// success, we could increase with one
                        remainder = t2 - t2xx;//new remainder based on t2xx
                    }
                } else {
                    remainder=utctimespan(0);// perfect hit, zero remainder
                }
            }
            return n_units*sgn;
        };

        namespace time_zone {
            using namespace boost::gregorian;
            using namespace boost::local_time;
            using namespace boost::posix_time;
            using namespace std;

            ///< just a local adapter to the tz_table generator
            struct boost_tz_info { // limitation is posix time, def from 1901 and onward
                ptime posix_t1970;///< epoch reference in ptime, needed to convert to utctime
                time_zone_ptr tzinfo;///< we extract boost info from this one
                string tz_region_name;///< the timezone region name, olson standard
                /**\brief convert a ptime to a utctime */
                utctime to_utctime(ptime p) const {
                    time_duration diff(p - posix_t1970);
                    return (diff.ticks() / diff.ticks_per_second());
                }

                boost_tz_info(string tz_region_name,time_zone_ptr tzinfo):posix_t1970((date(1970,1,1))),tzinfo(tzinfo),tz_region_name(tz_region_name) {}


                utctime dst_start(int year) const {
                    return to_utctime(tzinfo->dst_local_start_time(year)) - tzinfo->base_utc_offset().total_seconds();
                }
                utctime dst_end(int year) const {
                    return to_utctime(tzinfo->dst_local_end_time(year)) - tzinfo->base_utc_offset().total_seconds() - tzinfo->dst_offset().total_seconds();
                }
                utctimespan base_offset() const {
                    return utctimespan(tzinfo->base_utc_offset().total_seconds());
                }
                utctimespan dst_offset(int y) const {
                    return utctimespan(tzinfo->dst_offset().total_seconds());
                }
                string name() const {return tz_region_name;}
            };

            static shared_ptr<tz_info_t> create_from_posix_definition(string tz_region_name,const string& posix_tz_string) {
                time_zone_ptr tz=time_zone_ptr(new posix_time_zone(posix_tz_string));
                boost_tz_info btz(tz_region_name,tz);
                return make_shared<tz_info_t>(tz_info_t(btz.base_offset(),tz_table(btz)));
            }

            void tz_info_database::load_from_file(const string& filename) {
                tz_database tzdb;
                tzdb.load_from_file(filename);
                region_tz_map.clear(); name_tz_map.clear();
                for(const auto&id:tzdb.region_list()) {
                    auto tz=tzdb.time_zone_from_region(id);
                    boost_tz_info btz(id,tz);
                    auto tzinfo= make_shared<tz_info_t>(tz_info_t(btz.base_offset(),tz_table(btz)));
                    region_tz_map[id]=tzinfo;
                    name_tz_map[tzinfo->name()]=tzinfo;
                }
            }
            void tz_info_database::add_tz_info(string region_name,string posix_tz_string) {
                auto tzinfo= create_from_posix_definition(region_name,posix_tz_string);
                region_tz_map[region_name]=tzinfo;
                name_tz_map[tzinfo->name()]=tzinfo;
            }
            struct tz_region_def {
                const char *region;const char *posix_definition;
            };
            static tz_region_def tzdef[]={ // as defined in boost 1.60 date_time_zonespec.csv 2016.01.30
                {"Africa/Abidjan","GMT+00"},
                {"Africa/Accra","GMT+00"},
                {"Africa/Addis_Ababa","EAT+03"},
                {"Africa/Algiers","CET+01"},
                {"Africa/Asmera","EAT+03"},
                {"Africa/Bamako","GMT+00"},
                {"Africa/Bangui","WAT+01"},
                {"Africa/Banjul","GMT+00"},
                {"Africa/Bissau","GMT+00"},
                {"Africa/Blantyre","CAT+02"},
                {"Africa/Brazzaville","WAT+01"},
                {"Africa/Bujumbura","CAT+02"},
                {"Africa/Cairo","EET+02EEST+01,M4.5.5/00:00,M9.5.5/00:00"},
                {"Africa/Casablanca","WET+00"},
                {"Africa/Ceuta","CET+01CEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Africa/Conakry","GMT+00"},
                {"Africa/Dakar","GMT+00"},
                {"Africa/Dar_es_Salaam","EAT+03"},
                {"Africa/Djibouti","EAT+03"},
                {"Africa/Douala","WAT+01"},
                {"Africa/El_Aaiun","WET+00"},
                {"Africa/Freetown","GMT+00"},
                {"Africa/Gaborone","CAT+02"},
                {"Africa/Harare","CAT+02"},
                {"Africa/Johannesburg","SAST+02"},
                {"Africa/Kampala","EAT+03"},
                {"Africa/Khartoum","EAT+03"},
                {"Africa/Kigali","CAT+02"},
                {"Africa/Kinshasa","WAT+01"},
                {"Africa/Lagos","WAT+01"},
                {"Africa/Libreville","WAT+01"},
                {"Africa/Lome","GMT+00"},
                {"Africa/Luanda","WAT+01"},
                {"Africa/Lubumbashi","CAT+02"},
                {"Africa/Lusaka","CAT+02"},
                {"Africa/Malabo","WAT+01"},
                {"Africa/Maputo","CAT+02"},
                {"Africa/Maseru","SAST+02"},
                {"Africa/Mbabane","SAST+02"},
                {"Africa/Mogadishu","EAT+03"},
                {"Africa/Monrovia","GMT+00"},
                {"Africa/Nairobi","EAT+03"},
                {"Africa/Ndjamena","WAT+01"},
                {"Africa/Niamey","WAT+01"},
                {"Africa/Nouakchott","GMT+00"},
                {"Africa/Ouagadougou","GMT+00"},
                {"Africa/Porto-Novo","WAT+01"},
                {"Africa/Sao_Tome","GMT+00"},
                {"Africa/Timbuktu","GMT+00"},
                {"Africa/Tripoli","EET+02"},
                {"Africa/Tunis","CET+01"},
                {"Africa/Windhoek","WAT+01WAST+01,M9.1.0/02:00,M4.1.0/02:00"},
                {"America/Adak","HAST-10HADT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/Anchorage","AKST-09AKDT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/Anguilla","AST-04"},
                {"America/Antigua","AST-04"},
                {"America/Araguaina","BRT-03BRST+01,M10.2.0/00:00,M2.3.0/00:00"},
                {"America/Aruba","AST-04"},
                {"America/Asuncion","PYT-04PYST+01,M10.1.0/00:00,M3.1.0/00:00"},
                {"America/Barbados","AST-04"},
                {"America/Belem","BRT-03"},
                {"America/Belize","CST-06"},
                {"America/Boa_Vista","AMT-04"},
                {"America/Bogota","COT-05"},
                {"America/Boise","MST-07MDT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/Buenos_Aires","ART-03"},
                {"America/Cambridge_Bay","MST-07MDT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/Cancun","CST-06CDT+01,M4.1.0/02:00,M10.5.0/02:00"},
                {"America/Caracas","VET-04"},
                {"America/Catamarca","ART-03"},
                {"America/Cayenne","GFT-03"},
                {"America/Cayman","EST-05"},
                {"America/Chicago","CST-06CDT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/Chihuahua","MST-07MDT+01,M4.1.0/02:00,M10.5.0/02:00"},
                {"America/Cordoba","ART-03"},
                {"America/Costa_Rica","CST-06"},
                {"America/Cuiaba","AMT-04AMST+01,M10.2.0/00:00,M2.3.0/00:00"},
                {"America/Curacao","AST-04"},
                {"America/Danmarkshavn","GMT+00"},
                {"America/Dawson","PST-08PDT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/Dawson_Creek","MST-07"},
                {"America/Denver","MST-07MDT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/Detroit","EST-05EDT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/Dominica","AST-04"},
                {"America/Edmonton","MST-07MDT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/Eirunepe","ACT-05"},
                {"America/El_Salvador","CST-06"},
                {"America/Fortaleza","BRT-03BRST+01,M10.2.0/00:00,M2.3.0/00:00"},
                {"America/Glace_Bay","AST-04ADT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/Godthab","WGT-03WGST+01,M3.5.6/22:00,M10.5.6/23:00"},
                {"America/Goose_Bay","AST-04ADT+01,M4.1.0/00:01,M10.5.0/00:01"},
                {"America/Grand_Turk","EST-05EDT+01,M4.1.0/00:00,M10.5.0/00:00"},
                {"America/Grenada","AST-04"},
                {"America/Guadeloupe","AST-04"},
                {"America/Guatemala","CST-06"},
                {"America/Guayaquil","ECT-05"},
                {"America/Guyana","GYT-04"},
                {"America/Halifax","AST-04ADT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/Havana","CST-05CDT+01,M4.1.0/00:00,M10.5.0/01:00"},
                {"America/Hermosillo","MST-07"},
                {"America/Indiana/Indianapolis","EST-05"},
                {"America/Indiana/Knox","EST-05"},
                {"America/Indiana/Marengo","EST-05"},
                {"America/Indiana/Vevay","EST-05"},
                {"America/Indianapolis","EST-05"},
                {"America/Inuvik","MST-07MDT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/Iqaluit","EST-05EDT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/Jamaica","EST-05"},
                {"America/Jujuy","ART-03"},
                {"America/Juneau","AKST-09AKDT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/Kentucky/Louisville","EST-05EDT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/Kentucky/Monticello","EST-05EDT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/La_Paz","BOT-04"},
                {"America/Lima","PET-05"},
                {"America/Los_Angeles","PST-08PDT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/Louisville","EST-05EDT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/Maceio","BRT-03BRST+01,M10.2.0/00:00,M2.3.0/00:00"},
                {"America/Managua","CST-06"},
                {"America/Manaus","AMT-04"},
                {"America/Martinique","AST-04"},
                {"America/Mazatlan","MST-07MDT+01,M4.1.0/02:00,M10.5.0/02:00"},
                {"America/Mendoza","ART-03"},
                {"America/Menominee","CST-06CDT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/Merida","CST-06CDT+01,M4.1.0/02:00,M10.5.0/02:00"},
                {"America/Mexico_City","CST-06CDT+01,M4.1.0/02:00,M10.5.0/02:00"},
                {"America/Miquelon","PMST-03PMDT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/Monterrey","CST-06CDT+01,M4.1.0/02:00,M10.5.0/02:00"},
                {"America/Montevideo","UYT-03"},
                {"America/Montreal","EST-05EDT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/Montserrat","AST-04"},
                {"America/Nassau","EST-05EDT+01,M4.1.0/02:00,M10.5.0/02:00"},
                {"America/New_York","EST-05EDT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/Nipigon","EST-05EDT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/Nome","AKST-09AKDT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/Noronha","FNT-02"},
                {"America/North_Dakota/Center","CST-06CDT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/Panama","EST-05"},
                {"America/Pangnirtung","EST-05EDT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/Paramaribo","SRT-03"},
                {"America/Phoenix","MST-07"},
                {"America/Port-au-Prince","EST-05"},
                {"America/Port_of_Spain","AST-04"},
                {"America/Porto_Velho","AMT-04"},
                {"America/Puerto_Rico","AST-04"},
                {"America/Rainy_River","CST-06CDT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/Rankin_Inlet","CST-06CDT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/Recife","BRT-03BRST+01,M10.2.0/00:00,M2.3.0/00:00"},
                {"America/Regina","CST-06"},
                {"America/Rio_Branco","ACT-05"},
                {"America/Rosario","ART-03"},
                {"America/Santiago","CLT-04CLST+01,M10.2.0/00:00,M3.2.0/00:00"},
                {"America/Santo_Domingo","AST-04"},
                {"America/Sao_Paulo","BRT-03BRST+01,M10.2.0/00:00,M2.3.0/00:00"},
                {"America/Scoresbysund","EGT-01EGST+01,M3.5.0/00:00,M10.5.0/01:00"},
                {"America/Shiprock","MST-07MDT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/St_Johns","NST-03:-30NDT+01,M4.1.0/00:01,M10.5.0/00:01"},
                {"America/St_Kitts","AST-04"},
                {"America/St_Lucia","AST-04"},
                {"America/St_Thomas","AST-04"},
                {"America/St_Vincent","AST-04"},
                {"America/Swift_Current","CST-06"},
                {"America/Tegucigalpa","CST-06"},
                {"America/Thule","AST-04"},
                {"America/Thunder_Bay","EST-05EDT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/Tijuana","PST-08PDT+01,M4.1.0/02:00,M10.5.0/02:00"},
                {"America/Tortola","AST-04"},
                {"America/Vancouver","PST-08PDT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/Whitehorse","PST-08PDT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/Winnipeg","CST-06CDT+01,M3.2.0/02:00,M11.1.0/03:00"},
                {"America/Yakutat","AKST-09AKDT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"America/Yellowknife","MST-07MDT+01,M3.2.0/02:00,M11.1.0/02:00"},
                {"Antarctica/Casey","WST+08"},
                {"Antarctica/Davis","DAVT+07"},
                {"Antarctica/DumontDUrville","DDUT+10"},
                {"Antarctica/Mawson","MAWT+06"},
                {"Antarctica/McMurdo","NZST+12NZDT+01,M10.1.0/02:00,M3.3.0/03:00"},
                {"Antarctica/Palmer","CLT-04CLST+01,M10.2.0/00:00,M3.2.0/00:00"},
                {"Antarctica/South_Pole","NZST+12NZDT+01,M10.1.0/02:00,M3.3.0/03:00"},
                {"Antarctica/Syowa","SYOT+03"},
                {"Antarctica/Vostok","VOST+06"},
                {"Arctic/Longyearbyen","CET+01CEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Asia/Aden","AST+03"},
                {"Asia/Almaty","ALMT+06ALMST+01,M3.5.0/00:00,M10.5.0/00:00"},
                {"Asia/Amman","EET+02EEST+01,M3.5.4/00:00,M9.5.4/01:00"},
                {"Asia/Anadyr","ANAT+12ANAST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Asia/Aqtau","AQTT+04AQTST+01,M3.5.0/00:00,M10.5.0/00:00"},
                {"Asia/Aqtobe","AQTT+05AQTST+01,M3.5.0/00:00,M10.5.0/00:00"},
                {"Asia/Ashgabat","TMT+05"},
                {"Asia/Baghdad","AST+03ADT+01,M4.1.0/03:00,M10.1.0/04:00"},
                {"Asia/Bahrain","AST+03"},
                {"Asia/Baku","AZT+04AZST+01,M3.5.0/01:00,M10.5.0/01:00"},
                {"Asia/Bangkok","ICT+07"},
                {"Asia/Beirut","EET+02EEST+01,M3.5.0/00:00,M10.5.0/00:00"},
                {"Asia/Bishkek","KGT+05KGST+01,M3.5.0/02:30,M10.5.0/02:30"},
                {"Asia/Brunei","BNT+08"},
                {"Asia/Calcutta","IST+05:30"},
                {"Asia/Choibalsan","CHOT+09"},
                {"Asia/Chongqing","CST+08"},
                {"Asia/Colombo","LKT+06"},
                {"Asia/Damascus","EET+02EEST+01,M4.1.0/00:00,M10.1.0/00:00"},
                {"Asia/Dhaka","BDT+06"},
                {"Asia/Dili","TPT+09"},
                {"Asia/Dubai","GST+04"},
                {"Asia/Dushanbe","TJT+05"},
                {"Asia/Gaza","EET+02EEST+01,M4.3.5/00:00,M10.3.5/00:00"},
                {"Asia/Harbin","CST+08"},
                {"Asia/Hong_Kong","HKT+08"},
                {"Asia/Hovd","HOVT+07"},
                {"Asia/Irkutsk","IRKT+08IRKST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Asia/Istanbul","EET+02EEST+01,M3.5.0/03:00,M10.5.0/04:00"},
                {"Asia/Jakarta","WIT+07"},
                {"Asia/Jayapura","EIT+09"},
                {"Asia/Jerusalem","IST+02IDT+01,M4.1.0/01:00,M10.1.0/01:00"},
                {"Asia/Kabul","AFT+04:30"},
                {"Asia/Kamchatka","PETT+12PETST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Asia/Karachi","PKT+05"},
                {"Asia/Kashgar","CST+08"},
                {"Asia/Katmandu","NPT+05:45"},
                {"Asia/Krasnoyarsk","KRAT+07KRAST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Asia/Kuala_Lumpur","MYT+08"},
                {"Asia/Kuching","MYT+08"},
                {"Asia/Kuwait","AST+03"},
                {"Asia/Macao","CST+08"},
                {"Asia/Macau","CST+08"},
                {"Asia/Magadan","MAGT+11MAGST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Asia/Makassar","CIT+08"},
                {"Asia/Manila","PHT+08"},
                {"Asia/Muscat","GST+04"},
                {"Asia/Nicosia","EET+02EEST+01,M3.5.0/03:00,M10.5.0/04:00"},
                {"Asia/Novosibirsk","NOVT+06NOVST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Asia/Omsk","OMST+06OMSST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Asia/Oral","WST+05"},
                {"Asia/Phnom_Penh","ICT+07"},
                {"Asia/Pontianak","WIT+07"},
                {"Asia/Pyongyang","KST+09"},
                {"Asia/Qatar","AST+03"},
                {"Asia/Qyzylorda","KST+06"},
                {"Asia/Rangoon","MMT+06:30"},
                {"Asia/Riyadh","AST+03"},
                {"Asia/Saigon","ICT+07"},
                {"Asia/Sakhalin","SAKT+10SAKST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Asia/Samarkand","UZT+05"},
                {"Asia/Seoul","KST+09"},
                {"Asia/Shanghai","CST+08"},
                {"Asia/Singapore","SGT+08"},
                {"Asia/Taipei","CST+08"},
                {"Asia/Tashkent","UZT+05"},
                {"Asia/Tbilisi","GET+04GEST+01,M3.5.0/00:00,M10.5.0/00:00"},
                {"Asia/Tehran","IRT+03:30"},
                {"Asia/Thimphu","BTT+06"},
                {"Asia/Tokyo","JST+09"},
                {"Asia/Ujung_Pandang","CIT+08"},
                {"Asia/Ulaanbaatar","ULAT+08"},
                {"Asia/Urumqi","CST+08"},
                {"Asia/Vientiane","ICT+07"},
                {"Asia/Vladivostok","VLAT+10VLAST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Asia/Yakutsk","YAKT+09YAKST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Asia/Yekaterinburg","YEKT+05YEKST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Asia/Yerevan","AMT+04AMST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Atlantic/Azores","AZOT-01AZOST+01,M3.5.0/00:00,M10.5.0/01:00"},
                {"Atlantic/Bermuda","AST-04ADT+01,M4.1.0/02:00,M10.5.0/02:00"},
                {"Atlantic/Canary","WET+00WEST+01,M3.5.0/01:00,M10.5.0/02:00"},
                {"Atlantic/Cape_Verde","CVT-01"},
                {"Atlantic/Faeroe","WET+00WEST+01,M3.5.0/01:00,M10.5.0/02:00"},
                {"Atlantic/Jan_Mayen","CET+01CEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Atlantic/Madeira","WET+00WEST+01,M3.5.0/01:00,M10.5.0/02:00"},
                {"Atlantic/Reykjavik","GMT+00"},
                {"Atlantic/South_Georgia","GST-02"},
                {"Atlantic/St_Helena","GMT+00"},
                {"Atlantic/Stanley","FKT-04FKST+01,M9.1.0/02:00,M4.3.0/02:00"},
                {"Australia/Adelaide","CST+09:30CST+01,M10.1.0/02:00,M4.1.0/03:00"},
                {"Australia/Brisbane","EST+10"},
                {"Australia/Broken_Hill","CST+09:30CST+01,M10.1.0/02:00,M4.1.0/03:00"},
                {"Australia/Darwin","CST+09:30"},
                {"Australia/Eucla","CWST+08:45"},
                {"Australia/Hobart","EST+10EST+01,M10.1.0/02:00,M4.1.0/03:00"},
                {"Australia/Lindeman","EST+10"},
                {"Australia/Lord_Howe","LHST+10:30LHST+00:30,M10.1.0/02:00,M4.1.0/02:30"},
                {"Australia/Melbourne","EST+10EST+01,M10.1.0/02:00,M4.1.0/03:00"},
                {"Australia/Perth","WST+08"},
                {"Australia/Sydney","EST+10EST+01,M10.1.0/02:00,M4.1.0/03:00"},
                {"Europe/Amsterdam","CET+01CEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Europe/Andorra","CET+01CEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Europe/Athens","EET+02EEST+01,M3.5.0/03:00,M10.5.0/04:00"},
                {"Europe/Belfast","GMT+00BST+01,M3.5.0/01:00,M10.5.0/02:00"},
                {"Europe/Belgrade","CET+01CEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Europe/Berlin","CET+01CEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Europe/Bratislava","CET+01CEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Europe/Brussels","CET+01CEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Europe/Bucharest","EET+02EEST+01,M3.5.0/03:00,M10.5.0/04:00"},
                {"Europe/Budapest","CET+01CEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Europe/Chisinau","EET+02EEST+01,M3.5.0/03:00,M10.5.0/04:00"},
                {"Europe/Copenhagen","CET+01CEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Europe/Dublin","GMT+00IST+01,M3.5.0/01:00,M10.5.0/02:00"},
                {"Europe/Gibraltar","CET+01CEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Europe/Helsinki","EET+02EEST+01,M3.5.0/03:00,M10.5.0/04:00"},
                {"Europe/Istanbul","EET+02EEST+01,M3.5.0/03:00,M10.5.0/04:00"},
                {"Europe/Kaliningrad","EET+02EEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Europe/Kiev","EET+02EEST+01,M3.5.0/03:00,M10.5.0/04:00"},
                {"Europe/Lisbon","WET+00WEST+01,M3.5.0/01:00,M10.5.0/02:00"},
                {"Europe/Ljubljana","CET+01CEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Europe/London","GMT+00BST+01,M3.5.0/01:00,M10.5.0/02:00"},
                {"Europe/Luxembourg","CET+01CEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Europe/Madrid","CET+01CEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Europe/Malta","CET+01CEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Europe/Minsk","EET+02EEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Europe/Monaco","CET+01CEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Europe/Moscow","MSK+03MSD+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Europe/Nicosia","EET+02EEST+01,M3.5.0/03:00,M10.5.0/04:00"},
                {"Europe/Oslo","CET+01CEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Europe/Paris","CET+01CEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Europe/Prague","CET+01CEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Europe/Riga","EET+02EEST+01,M3.5.0/03:00,M10.5.0/04:00"},
                {"Europe/Rome","CET+01CEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Europe/Samara","SAMT+04SAMST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Europe/San_Marino","CET+01CEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Europe/Sarajevo","CET+01CEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Europe/Simferopol","EET+02EEST+01,M3.5.0/03:00,M10.5.0/04:00"},
                {"Europe/Skopje","CET+01CEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Europe/Sofia","EET+02EEST+01,M3.5.0/03:00,M10.5.0/04:00"},
                {"Europe/Stockholm","CET+01CEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Europe/Tallinn","EET+02"},
                {"Europe/Tirane","CET+01CEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Europe/Uzhgorod","EET+02EEST+01,M3.5.0/03:00,M10.5.0/04:00"},
                {"Europe/Vaduz","CET+01CEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Europe/Vatican","CET+01CEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Europe/Vienna","CET+01CEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Europe/Vilnius","EET+02"},
                {"Europe/Warsaw","CET+01CEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Europe/Zagreb","CET+01CEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Europe/Zaporozhye","EET+02EEST+01,M3.5.0/03:00,M10.5.0/04:00"},
                {"Europe/Zurich","CET+01CEST+01,M3.5.0/02:00,M10.5.0/03:00"},
                {"Indian/Antananarivo","EAT+03"},
                {"Indian/Chagos","IOT+06"},
                {"Indian/Christmas","CXT+07"},
                {"Indian/Cocos","CCT+06:30"},
                {"Indian/Comoro","EAT+03"},
                {"Indian/Kerguelen","TFT+05"},
                {"Indian/Mahe","SCT+04"},
                {"Indian/Maldives","MVT+05"},
                {"Indian/Mauritius","MUT+04"},
                {"Indian/Mayotte","EAT+03"},
                {"Indian/Reunion","RET+04"},
                {"Pacific/Apia","WST-11"},
                {"Pacific/Auckland","NZST+12NZDT+01,M10.1.0/02:00,M3.3.0/03:00"},
                {"Pacific/Chatham","CHAST+12:45CHADT+01,M10.1.0/02:45,M3.3.0/03:45"},
                {"Pacific/Easter","EAST-06EASST+01,M10.2.6/22:00,M3.2.6/22:00"},
                {"Pacific/Efate","VUT+11"},
                {"Pacific/Enderbury","PHOT+13"},
                {"Pacific/Fakaofo","TKT-10"},
                {"Pacific/Fiji","FJT+12"},
                {"Pacific/Funafuti","TVT+12"},
                {"Pacific/Galapagos","GALT-06"},
                {"Pacific/Gambier","GAMT-09"},
                {"Pacific/Guadalcanal","SBT+11"},
                {"Pacific/Guam","ChST+10"},
                {"Pacific/Honolulu","HST-10"},
                {"Pacific/Johnston","HST-10"},
                {"Pacific/Kiritimati","LINT+14"},
                {"Pacific/Kosrae","KOST+11"},
                {"Pacific/Kwajalein","MHT+12"},
                {"Pacific/Majuro","MHT+12"},
                {"Pacific/Marquesas","MART-09:-30"},
                {"Pacific/Midway","SST-11"},
                {"Pacific/Nauru","NRT+12"},
                {"Pacific/Niue","NUT-11"},
                {"Pacific/Norfolk","NFT+11:30"},
                {"Pacific/Noumea","NCT+11"},
                {"Pacific/Pago_Pago","SST-11"},
                {"Pacific/Palau","PWT+09"},
                {"Pacific/Pitcairn","PST-08"},
                {"Pacific/Ponape","PONT+11"},
                {"Pacific/Port_Moresby","PGT+10"},
                {"Pacific/Rarotonga","CKT-10"},
                {"Pacific/Saipan","ChST+10"},
                {"Pacific/Tahiti","TAHT-10"},
                {"Pacific/Tarawa","GILT+12"},
                {"Pacific/Tongatapu","TOT+13"},
                {"Pacific/Truk","TRUT+10"},
                {"Pacific/Wake","WAKT+12"},
                {"Pacific/Wallis","WFT+12"},
                {"Pacific/Yap","YAPT+10"}
            };
            const size_t n_tzdef= sizeof(tzdef)/sizeof(tz_region_def);
            void tz_info_database::load_from_iso_db() {

                region_tz_map.clear(); name_tz_map.clear();
                for(size_t i=0;i<n_tzdef;++i) {
                    add_tz_info(tzdef[i].region,tzdef[i].posix_definition);
                }
            }
        }

        calendar::calendar(std::string region_id) {
            for(size_t i=0;i<time_zone::n_tzdef;++i) {
                if(region_id==time_zone::tzdef[i].region) {
                    tz_info=time_zone::create_from_posix_definition(region_id,time_zone::tzdef[i].posix_definition);
                }
            }
            if(!tz_info)
                throw runtime_error(string("time zone region id '")+region_id+ string("' not found, use .region_id_list() to get configured time zones"));
        }

        vector<string> calendar::region_id_list() {
            vector<string> r;
            for(size_t i=0;i<time_zone::n_tzdef;++i) r.push_back(time_zone::tzdef[i].region);
            return r;
        }

    } // core
} // shyft
