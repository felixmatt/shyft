#pragma once
#include <string>
#include <time.h>
#include <cmath>

namespace shyft {
	namespace core {
        /** \brief utctime
         * basic types for time handling
		 * we use linear time, i.e. time is just a number
		 * on the timeaxis, utc. Timeaxis zero is at 1970-01-01 00:00:00 (unix time).
		 * resolution is 1 second, integer.
		 * also define min max and no_utctime
		 * The advantage with the definition well defined and commonly known in all C++ platforms
         */

		#ifdef WIN32
		typedef long long utctime;          /// time_t is typedef'd as a __time64_t, which is an __int64., happens to be equal to EnkiTime
		typedef long long utctimespan;      /// utctimespan is typdedef'd as a utctime (thus __int64)

		#else
				typedef long utctime;       /// time_t is typedef'd as a __time64_t, which is an __int64., happens to be equal to EnkiTime
				typedef long utctimespan;   /// utctimespan is typdedef'd as a utctime (thus __int64)
		#endif


		namespace convert {
			const double COleDateTime1970Offset = 25569.0;
			static inline double to_COleDateTime(utctime t) {
				return COleDateTime1970Offset + t/(86400.0);
			}
			static inline utctime to_utctime(double t)
            {
				return (utctime)floor((t - COleDateTime1970Offset)*86400.0 + 0.5);
			}
		}
        /** \brief deltahours
         * \param n number of hours
         * \return utctimespan representing number of hours specified
         */
		inline utctimespan deltahours(int n) { return n*3600; }

        /** \brief deltaminutes
         * \param n number of minutes
         * \return utctimespan representing number of minutes specified
         */
		inline utctimespan deltaminutes(int n) { return n*60; }

        /** \brief max_utctime represent the maximum utctime
         */
		const		utctime max_utctime	= (utctime) (0x7FFFFFFFFFFFFFFFL);	/// max 64bit int
        /** \brief min_utctime represent the minimum utctime (equal to -max_utctime)
         */

		const		utctime min_utctime	= - max_utctime;  /// min 64bit int

        /** \brief no_utctime represents 'NaN' time, null time, not valid time
         */
		const		utctime no_utctime	= min_utctime - 1L;/// Nonvalid

        /** \brief utctimeNow, utctime_now()
         *  \return current systemclock utctime
         */
		inline utctime utctime_now() {return (utctime)time(0); }



        /** \brief utcperiod is defined
         *  as period on the utctime space, like
         * [start..end>, where end >=start to be valid
         *
         */
		struct utcperiod {
			utcperiod(utctime start, utctime end): start(start), end(end) {}
			utcperiod(): start(no_utctime), end(no_utctime) {}
			utctimespan timespan() const {	return end - start; }
			bool valid() const { return start != no_utctime && end != no_utctime && start <= end; }
			bool operator==(const utcperiod &p) const { return start == p.start &&  end == p.end; }
			utctime start;
			utctime end;
			std::string to_string() const;
#ifndef SWIG
            friend std::ostream& operator<<(std::ostream& os, const utcperiod& p);
#endif
		};

        /** \brief YMDhms, simple structure that contains calendar coordinates.
         * Contains year,month,day,hour,minute, second,
         * for ease of constructing utctime.
         *
         */
		struct YMDhms {
			YMDhms():year(0), month(0), day(0), hour(0), minute(0), second(0) {}
			YMDhms(int Y, int M=1, int D=1, int h=0, int m=0, int s=0): year(Y), month(M), day(D), hour(h), minute(m), second(s) {}

			int year; int month; int day; int hour; int minute; int second;
			bool is_null() const { return year == 0 && month == 0 && day == 0 && hour == 0 && minute == 0 && second == 0; }
			bool operator==(const YMDhms& x) const
            {
                return x.year == year && x.month == month && x.day == day && x.hour == hour
                       && x.minute == minute && x.second == second;
            }
			static YMDhms max() {return YMDhms(9999,12,31,23,59,59);}
			static YMDhms min() {return YMDhms(-9999,12,31,23,59,59);}
		};

        /** \brief Calendar deals with the concept of human calendar.
         *
         * Please notice that although the calendar concept is complete,
         * we only implement features as needed in the core and interfaces.
         *
         * including:
         * -# Conversion between the calendar coordinates YMDhms and utctime, taking  any timezone and DST into account
         * -# Calendar constants, utctimespan like values for Year,Month,Week,Day,Hour,Minute,Second
         * -# Calendar arithmetic, like adding calendar units, e.g. day,month,year etc.
         * -# Calendar arithmetic, like trim/truncate a utctime down to nearest timespan/calendar unit. eg. day
         * -# Calendar arithmetic, like calculate difference in calendar units(e.g days) between two utctime points
         * -# Calendar Timezone and DST handling
         * -# Converting utctime to string and vice-versa
         */
		struct calendar {
			static const utctimespan YEAR=365*24*3600L;
			static const utctimespan MONTH=30*24*3600L;
			static const utctimespan WEEK = 7*24*3600L;
			static const utctimespan DAY =  1*24*3600L;
			static const utctimespan HOUR = 3600L;
			static const utctimespan MINUTE = 60L;
			static const utctimespan SECOND = 1L;
			static const int UnixDay = 2440588;//Calc::julian_day_number(ymd(1970,01,01));
			static const utctime UnixSecond = 86400L * (utctime)UnixDay;//Calc::julian_day_number(ymd(1970,01,01));

			// Snapped from boost gregorian_calendar.ipp
			static unsigned long day_number(YMDhms ymd) {
				unsigned short a = static_cast<unsigned short>((14 - ymd.month) / 12);
				unsigned short y = static_cast<unsigned short>(ymd.year + 4800 - a);
				unsigned short m = static_cast<unsigned short>(ymd.month + 12 * a - 3);
				unsigned long  d = ymd.day + ((153 * m + 2) / 5) + 365 * y + (y / 4) - (y / 100) + (y / 400) - 32045;
				return d;
			}
			static YMDhms from_day_number(unsigned long dayNumber) {
				int a = dayNumber + 32044;
				int b = (4 * a + 3) / 146097;
				int c = a - ((146097 * b) / 4);
				int d = (4 * c + 3) / 1461;
				int e = c - (1461 * d) / 4;
				int m = (5 * e + 2) / 153;
				unsigned short day = static_cast<unsigned short>(e - ((153 * m + 2) / 5) + 1);
				unsigned short month = static_cast<unsigned short>(m + 3 - 12 * (m / 10));
				int year = static_cast<unsigned short>(100 * b + d - 4800 + (m / 10));
				//std::cout << year << "-" << month << "-" << day << "\n";

				return YMDhms(year, month, day);
			}
			static int day_number(utctime t) {
				return (int)(int)((UnixSecond + t) / DAY);
			}
			static inline utctimespan seconds(int h, int m, int s) { return h*HOUR + m*MINUTE + s*SECOND; }


			utctimespan tz_offset;
			calendar(utctimespan tz=0): tz_offset(tz) {}
			utctime time(YMDhms c) const {
				if (!((1 <= c.month && c.month <= 12) && (1 <= c.day && c.day <= 31) && (0 <= c.hour&&c.hour <= 23) && (0 <= c.minute&&c.minute <= 59) && (0 <= c.second&&c.second <= 59)))
					return no_utctime;
				return ((day_number(c) - UnixDay)*DAY) + seconds(c.hour, c.minute, c.second) - tz_offset;
			}
			YMDhms  calendar_units(utctime t) const {
				if (t == no_utctime) {
					return YMDhms();
				}
				if (t == max_utctime) {
					return YMDhms::max();
				}
				if (t == min_utctime) {
					return YMDhms::min();
				}
				t += tz_offset;
				auto jdn = day_number(t);
				auto x = from_day_number(jdn);
				YMDhms r;
				r.year = x.year; r.month = x.month; r.day = x.day;
				utctime tj = UnixSecond + t;
				utctime td = DAY*(tj / DAY);
				utctimespan dx = tj - td;// n seconds this day.
				r.hour = int(dx / HOUR);
				dx -= r.hour*HOUR;
				r.minute = int(dx / MINUTE);
				r.second = int(dx % MINUTE);
				return r;
			}

			///< returns  0=sunday, 1= monday ... 6=sat.
			int day_of_week(utctime t) const {
				if (t == no_utctime || t == min_utctime || t == max_utctime)
					return -1;
				auto ymd = calendar_units(t);
				unsigned short a = static_cast<unsigned short>((14 - ymd.month) / 12);
				unsigned short y = static_cast<unsigned short>(ymd.year - a);
				unsigned short m = static_cast<unsigned short>(ymd.month + 12 * a - 2);
				int d = static_cast<int>((ymd.day + y + (y / 4) - (y / 100) + (y / 400) + (31 * m) / 12) % 7);
				return d;
			}
			///< returns day_of year, 1..365..
			size_t day_of_year(utctime t) const {
				if (t == no_utctime || t == max_utctime || t == min_utctime) return -1;
				auto x = calendar_units(t );
				YMDhms y(x.year, 1, 1);
				return 1 + day_number(x) - day_number(y);
			}
			///< returns the month of t, 1..12
			int month(utctime t) const {
				if (t == no_utctime || t == max_utctime || t == min_utctime) return -1;
				return calendar_units(t ).month;
			}

			std::string to_string(utctime t) const;
			std::string to_string(utcperiod p) const;
			//utctime fromString(std::string datetime) const;
			// calendar arithmetic
			utctime trim(utctime t, utctimespan deltaT) const {
				if (t == no_utctime || t == min_utctime || t == max_utctime)
					return t;
				switch (deltaT) {
				case YEAR: {
					auto c = calendar_units(t);
					c.month = c.day = 1; c.hour = c.minute = c.second = 0;
					return time(c);
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
				const utctime t0 = utctime(3LL * DAY) + utctime(WEEK * 52L * 200LL);// + 3 days to get to a monday, then add 100 years, the point is, align to a week start.
				t += t0 + tz_offset;
				return deltaT*(t / deltaT) - t0 - tz_offset;
			}
			utctime add(utctime t, utctimespan deltaT, long n) const {
				switch (deltaT) {
				case YEAR:
				case MONTH:
					throw std::runtime_error("Not yet implemented");

				}
				return t + deltaT*n;// assume no dst yet
			};
			/// returns (t2-t1)/deltaT, and remainder, where deltaT could be calendar units Day,Week,Month,Year.
			utctimespan diff_units(utctime t1, utctime t2, utctimespan deltaT, utctimespan &remainder) const {
				if (t1 == no_utctime || t2 == no_utctime) {
					remainder = 0L;
					return 0L;
				}
				switch (deltaT) {
					case calendar::MONTH:{
						throw std::runtime_error("calendar::diff MONTH not supported");
					}break;
					case calendar::YEAR: {
						throw std::runtime_error("calendar::diff YEAR not supported");
					}break;

				}

				utctimespan dt = t2 - t1;
				utctimespan r = dt / deltaT;
				remainder = (dt - r*deltaT);
				return r;
			};
		};

		//utctime utctimeFromYMDhms(YMDhms c);




	}
}
