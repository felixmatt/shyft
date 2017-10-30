#pragma once

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include "core/core_pch.h"

#include "core/geo_cell_data.h"
#include "core/cell_model.h"
//-include stacks here, we need to make cell_state_with_id serializable
#include "core/hbv_stack.h"
#include "core/pt_gs_k.h"
#include "core/pt_hs_k.h"
#include "core/pt_ss_k.h"

namespace shyft {
    namespace api {
        /** \brief unique pseudo identifier of cell state
        *
        * A lot of Shyft models can appear in different geometries and
        * resolutions. In a few rare cases we would like to store the *state* of
        * a model. This would be typical for example for hydrological new-year 1-sept in Norway.
        * To ensure that each cell-state have a unique identifier so that we never risk
        * mixing state from different cells or different geometries, we create a pseudo unique
        * id that helps identifying unique cell-states given this usage and context.
        *
        * The primary usage is to identify which cell a specific identified state belongs to.
        *
        */
        struct cell_state_id {
            int cid;///< the catchment id, if entirely different model, this might change
            int x;///< the cell mid-point x (west-east), truncated to integer [meter]
            int y;///< the cell mid-point y (south-north), truncated to integer [meter]
            int area;///< the area in m[m2], - if different cell geometries, this changes
            cell_state_id() = default; // python exposure
            cell_state_id(int cid, int x, int y, int area) :cid(cid), x(x), y(y), area(area) {}
            bool operator==(const cell_state_id & o) const {
                return cid == o.cid && x == o.x && y == o.y && area == o.area;
            }
            bool operator!=(cell_state_id const&o) const { return !operator==(o); }
            bool operator<(const cell_state_id& o) const {
                if (cid < o.cid) return true;
                if (cid > o.cid) return false;
                if (x < o.x)return true;
                if (x > o.x)return false;
                if (y < o.y) return true;
                if (y > o.y) return false;
                return area < o.area;
            }
            x_serialize_decl();
        };
        /** create the cell_state_id based on specified cell.geo part*/
        inline cell_state_id cell_state_id_of(const shyft::core::geo_cell_data&c_geo) {
            return cell_state_id(c_geo.catchment_id(), (int)c_geo.mid_point().x, (int)c_geo.mid_point().y, (int)c_geo.area());
        }

        /** A cell state with a cell_state identifier */
        template <class CS>
        struct cell_state_with_id {
            typedef CS cell_state_t;
            cell_state_id id;
            cell_state_t state;
            cell_state_with_id() {};
            bool operator==(const cell_state_with_id o) const {
                return id == o.id; // only id equality, for vector support in boost python
            }
            /** do the magic given a cell, create the id, stash away the id:state*/
            template<class C>
            explicit cell_state_with_id(const C& c) :id(cell_state_id_of(c.geo)), state(c.state) {}
            x_serialize_decl();
        };

        template <class CS> std::vector<char> serialize_to_bytes(const std::shared_ptr<std::vector<CS>>& states);
          extern template std::vector<char> serialize_to_bytes(const std::shared_ptr<std::vector<cell_state_with_id<shyft::core::hbv_stack::state>>>& states);
          extern template std::vector<char> serialize_to_bytes(const std::shared_ptr<std::vector<cell_state_with_id<shyft::core::pt_gs_k::state>>>& states);
          extern template std::vector<char> serialize_to_bytes(const std::shared_ptr<std::vector<cell_state_with_id<shyft::core::pt_ss_k::state>>>& states);
          extern template std::vector<char> serialize_to_bytes(const std::shared_ptr<std::vector<cell_state_with_id<shyft::core::pt_hs_k::state>>>& states);

        template <class CS> void deserialize_from_bytes(const std::vector<char>& bytes, std::shared_ptr<std::vector<CS>>&states);
          extern  template void deserialize_from_bytes(const std::vector<char>& bytes, std::shared_ptr<std::vector<cell_state_with_id<shyft::core::hbv_stack::state>>>&states);
          extern  template void deserialize_from_bytes(const std::vector<char>& bytes, std::shared_ptr<std::vector<cell_state_with_id<shyft::core::pt_gs_k::state>>>&states);
          extern  template void deserialize_from_bytes(const std::vector<char>& bytes, std::shared_ptr<std::vector<cell_state_with_id<shyft::core::pt_hs_k::state>>>&states);
          extern  template void deserialize_from_bytes(const std::vector<char>& bytes, std::shared_ptr<std::vector<cell_state_with_id<shyft::core::pt_ss_k::state>>>&states);
        /** \brief state_io_handler for efficient handling of cell-identified states
        *
        * This class provides functionality to extract/apply state based on a
        * pseudo unique id of each cell.
        * A cell is identified by the integer portions of:
        * catchment_id, mid_point.x,mid_point.y,area
        */
        template <class C>
        struct state_io_handler {
            typedef typename C::state_t state_t;
            typedef cell_state_with_id<state_t> cell_state_id_t;
            std::shared_ptr<std::vector<C>> cells;// shared alias region model cells.

            state_io_handler() {}
            /** construct a handler for the cells provided*/
            explicit state_io_handler(const std::shared_ptr<std::vector<C>>& cellx) :cells(cellx) {}

            /** Extract cell identified state
            *\return the state for the cells, optionally filtered by the supplied catchment ids (cids)
            */
            std::shared_ptr<std::vector<cell_state_id_t>> extract_state(const std::vector<int>& cids) const {
                if (!cells)
                    throw std::runtime_error("No cells to extract state from");
                auto r = std::make_shared<std::vector<cell_state_id_t>>();
                r->reserve(cells->size());
                for (const auto &c : *cells)
                    if (cids.size() == 0 || std::find(cids.begin(), cids.end(), c.geo.catchment_id()) != cids.end())
                        r->emplace_back(c);// creates a a cell_state_with_id based on cell
                return r;
            }

            /** Restore cell identified state, filtered by cids.
            * \return a list identifying states that where not applied to cells(filtering out all that is related to non-matching cids)
            */
            std::vector<int> apply_state(const std::shared_ptr < std::vector<cell_state_id_t> >& s, const std::vector<int>& cids) {
                if (!cells)
                    throw std::runtime_error("No cells to apply state into");
                std::map<cell_state_id, C*> cmap;// yes store pointers, we know the scope is this routine
                for (auto& c : *cells) {
                    if (cids.size() == 0 || std::find(cids.begin(), cids.end(), c.geo.catchment_id()) != cids.end())
                        cmap[cell_state_id_of(c.geo)] = &c;// fix the map
                }
                std::vector<int> missing;
                for (size_t i = 0;i < s->size();++i) {
                    if (cids.size() == 0 || std::find(cids.begin(), cids.end(), (*s)[i].id.cid) != cids.end()) {
                        auto f = cmap.find((*s)[i].id);// log(n)
                        if (f != cmap.end())
                            f->second->state = (*s)[i].state;
                        else
                            missing.push_back(i);
                    }
                }
                return missing;
            }
        };
    }
}
//-- serialization support shyft
x_serialize_export_key(shyft::api::cell_state_id);
x_serialize_export_key(shyft::api::cell_state_with_id<shyft::core::hbv_stack::state>);
x_serialize_export_key(shyft::api::cell_state_with_id<shyft::core::pt_gs_k::state>);
x_serialize_export_key(shyft::api::cell_state_with_id<shyft::core::pt_ss_k::state>);
x_serialize_export_key(shyft::api::cell_state_with_id<shyft::core::pt_hs_k::state>);
