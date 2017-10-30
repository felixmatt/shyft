#pragma once
#include <string>
#include <vector>
#include "core/hbv_stack.h"

namespace shyft {
	namespace api {
		typedef shyft::core::hbv_stack::state hbv_stack_state_t;

		struct hbv_stack_state_io {
			bool from_string(const std::string &str, hbv_stack_state_t &s) const {
				return from_raw_string(str.c_str(), s);
			}

			bool from_raw_string(const char* str, hbv_stack_state_t& s) const {
				if (str && *str) {
					if (sscanf(str, "hbv_stack:%lf %lf %lf %lf %lf",	// 5 states, 5 times %lf what %lf?
						&s.snow.sca, &s.snow.swe,						// why with &s
						&s.soil.sm,
						&s.tank.uz, &s.tank.lz) == 5)		//Why ==5
						return true;
				}
				return false;
			}

			std::string to_string(const hbv_stack_state_t& s) const {
				char r[500];
				sprintf(r, "hbv_stack:%f %f %f %f %f \n",				// 5 states, 5 times %lf what %lf?
					s.snow.sca, s.snow.swe,								// why not &s only s
					s.soil.sm,
					s.tank.uz, s.tank.lz);
				return r;
			}

			std::string to_string(const std::vector<hbv_stack_state_t> &sv) const {
				std::string r; r.reserve(200 * 200 * 50);
				for (size_t i = 0; i<sv.size(); ++i) {
					r.append(to_string(sv[i]));
				}
				return r;
			}

			std::vector<hbv_stack_state_t> vector_from_string(const std::string &s) const {
				std::vector<hbv_stack_state_t> r;
				if (s.size() > 0) {
					r.reserve(200 * 200);
					const char *l = s.c_str();
					const char *h;
					hbv_stack_state_t e;
					while (*l && (h = strstr(l, "hbv_stack:"))) {
						if (!from_raw_string(h, e))
							break;
						r.emplace_back(e);
						l = h + 10;  // advance after hbv_stack marker
					}
				}
				return r;
			}
		};
	}
}
