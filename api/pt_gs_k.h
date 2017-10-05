#pragma once
#ifdef SHYFT_NO_PCH
#include <string>
#include <vector>
#endif // SHYFT_NO_PCH

#include "core/pt_gs_k.h"

namespace shyft {
  namespace api {
    typedef shyft::core::pt_gs_k::state pt_gs_k_state_t;

    struct pt_gs_k_state_io {
        bool from_string(const std::string &str, pt_gs_k_state_t &s) const {
            return from_raw_string(str.c_str(), s);
        }

        bool from_raw_string(const char* str, pt_gs_k_state_t& s) const {
            if (str && *str) {
                if (sscanf(str, "ptgsk:%lf %lf %lf %lf %lf %lf %lf %lf %lf",
                    &s.gs.albedo, &s.gs.alpha, &s.gs.sdc_melt_mean,
                    &s.gs.acc_melt, &s.gs.iso_pot_energy, &s.gs.temp_swe, &s.gs.surface_heat, &s.gs.lwc,
                    &s.kirchner.q) == 9)
                    return true;

                // support old 7 string state variable format
                if (sscanf(str, "ptgsk:%lf %lf %lf %lf %lf %lf %lf",
                    &s.gs.albedo, &s.gs.alpha, &s.gs.sdc_melt_mean,
                    &s.gs.acc_melt, &s.gs.iso_pot_energy, &s.gs.temp_swe,
                    &s.kirchner.q) == 7)
                    return true;
            }
            return false;
        }

        std::string to_string(const pt_gs_k_state_t& s) const {
            char r[500];
            sprintf(r, "ptgsk:%f %f %f %f %f %f %f %f %f\n",
                s.gs.albedo, s.gs.alpha, s.gs.sdc_melt_mean,
                s.gs.acc_melt, s.gs.iso_pot_energy, s.gs.temp_swe, s.gs.surface_heat, s.gs.lwc,
                s.kirchner.q);
            return r;
        }

        std::string to_string(const std::vector<pt_gs_k_state_t> &sv) const {
            std::string r; r.reserve(200*200*50);
            for (size_t i = 0; i<sv.size(); ++i) {
                r.append(to_string(sv[i]));
            }
            return r;
        }

        std::vector<pt_gs_k_state_t> vector_from_string(const std::string &s) const {
            std::vector<pt_gs_k_state_t> r;
            if (s.size() > 0) {
                r.reserve(200*200);
                const char *l = s.c_str();
                const char *h;
                pt_gs_k_state_t e;
                while (*l && (h = strstr(l, "ptgsk:"))) {
                    if (!from_raw_string(h, e))
                        break;
                    r.emplace_back(e);
                    l = h + 6;// advance after ptgsk marker
                }
            }
            return r;
        }
    };
  }
}
