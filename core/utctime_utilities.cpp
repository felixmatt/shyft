#include "core_pch.h"
#include "utctime_utilities.h"
#include <ostream>

namespace shyft {
    namespace core {
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
                    sprintf(s, "%04d.%02d.%02dT%02d:%02d:%02d", c.year, c.month, c.day, c.hour, c.minute, c.second);
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
    } // core
} // shyft
