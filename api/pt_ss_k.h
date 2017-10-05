#pragma once
#ifdef SHYFT_NO_PCH
#include <string>
#include <vector>
#endif // SHYFT_NO_PCH
#include "core/pt_ss_k.h"

namespace shyft {
  namespace api {
    typedef shyft::core::pt_ss_k::state_t pt_ss_k_state_t;

    struct pt_ss_k_state_io {
        bool from_string(const std::string &str, pt_ss_k_state_t &s) const {
            return from_raw_string(str.c_str(), s);
        }

        bool from_raw_string(const char* str, pt_ss_k_state_t& s) const {
            if (str && *str) {
                if (sscanf(str, "ptssk:%lf %lf %lf %lf %lf %lf %ld %lf",
                    &s.snow.nu, &s.snow.alpha, &s.snow.sca,&s.snow.swe,
                    &s.snow.free_water, &s.snow.residual, &s.snow.num_units,
					&s.kirchner.q) == 8)
                    return true;
            }
            return false;
        }

        std::string to_string(const pt_ss_k_state_t& s) const {
            char r[500];
            sprintf(r, "ptssk:%f %f %f %f %f %f %ld %f\n",
				s.snow.nu, s.snow.alpha, s.snow.sca,s.snow.swe,
				s.snow.free_water, s.snow.residual, s.snow.num_units,
				s.kirchner.q);
            return r;
        }

        std::string to_string(const std::vector<pt_ss_k_state_t> &sv) const {
            std::string r; r.reserve(200*200*50);
            for (size_t i = 0; i<sv.size(); ++i) {
                r.append(to_string(sv[i]));
            }
            return r;
        }

        std::vector<pt_ss_k_state_t> vector_from_string(const std::string &s) const {
            std::vector<pt_ss_k_state_t> r;
            if (s.size() > 0) {
                r.reserve(200*200);
                const char *l = s.c_str();
                const char *h;
                pt_ss_k_state_t e;
                while (*l && (h = strstr(l, "ptssk:"))) {
                    if (!from_raw_string(h, e))
                        break;
                    r.emplace_back(e);
                    l = h + 6;  // advance after ptgsk marker
                }
            }
            return r;
        }
    };
  }
}
