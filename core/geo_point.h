#pragma once
#ifdef SHYFT_NO_PCH
#include <cmath>
#include "core_pch.h"
#endif // SHYFT_NO_PCH
namespace shyft {
    namespace core {

        /** \brief GeoPoint commonly used in the shyft::core for
         *  representing a 3D point in the terrain model.
         *  Primary usage is in the interpolation routines
         *
         *  Absolutely a primitive point model aiming for efficiency and simplicity.
         *
         *  Units of x,y,z are metric, z positive upwards to sky, represents elevation
         *  x is east-west axis
         *  y is south-north axis
         */
        struct geo_point {
            double x, y, z;
            geo_point() : x(0), y(0), z(0) {}
            geo_point(double x, double y=0.0, double z=0.0) : x(x), y(y), z(z) {}

            bool operator==(const geo_point &o) const {
                const double accuracy=1e-3;//mm
                return distance2(*this,o)<accuracy;
            }

            /** \brief Squared eucledian distance
             *
             * \return the 3D distance^2 between supplied arguments a and b
             */
            static inline double distance2(const geo_point& a, const geo_point& b) {
                return (a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y) + (a.z - b.z)*(a.z - b.z);
            }

            /** \brief Distance measure on the form sum(abs(a.x - b.x)^p + abs(a.y - b.y)^p + (zscale * abs(a.z - b.z))^p)
             *
             * \return the 3D distance measure between supplied arguments a and b, by using an optional z scale factor
             */
            static inline double distance_measure(const geo_point& a, const geo_point& b, double p, double zscale) {
                return std::pow((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y) + (a.z - b.z)*(a.z - b.z)*zscale*zscale, p/2.0);
            }

            /** \brief Z scaled, non-eucledian distance between points a and b.
             *
             * \return sqrt( (a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y) + (a.z - b.z)*(a.z - b.z)*zscale*zscale)
             */
            static inline double zscaled_distance(const geo_point& a, const geo_point& b, double zscale) {
                return std::sqrt( (a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y) + (a.z - b.z)*(a.z - b.z)*zscale*zscale);
            }

                        /** \brief Z scaled, non-eucledian distance between points a and b.
             *
             * \return  (a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y) + (a.z - b.z)*(a.z - b.z)*zscale*zscale
             */
            static inline double zscaled_distance2(const geo_point& a, const geo_point& b, double zscale) {
                return (a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y) + (a.z - b.z)*(a.z - b.z)*zscale*zscale;
            }

            /** \brief Euclidian distance between points a and b, first projected onto x-y plane.
             *
             * return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y))
             */
            static inline double xy_distance(const geo_point& a, const geo_point& b) {
                return std::sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
            }
            /** \brief Euclidian distance^2 between points a and b, first projected onto x-y plane.
            *
            * return (a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y)
            */
            static inline double xy_distance2(const geo_point& a, const geo_point& b) {
                return (a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y);
            }
            static inline double z_distance2(const geo_point& a, const geo_point& b) {
                return (a.z - b.z)*(a.z - b.z);
            }

            /** \brief Difference between points a and b, i.e. a - b. Should perhaps return a
             * GeoDifference instead for mathematical correctness.
             *
             * \return GeoPoint(a.x - b.x, a.y - b.y, a.z - b.z)
             */
            static inline geo_point difference(const geo_point& a, const geo_point& b) {
                return geo_point(a.x - b.x, a.y - b.y, a.z - b.z);
            }
            x_serialize_decl();
        };
    }
}
//-- serialization support shyft
x_serialize_export_key(shyft::core::geo_point);
