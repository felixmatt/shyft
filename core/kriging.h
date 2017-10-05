#pragma once
#ifdef SHYFT_NO_PCH
#include <cmath>
#include <stdexcept>
#include <armadillo>

#endif // SHYFT_NO_PCH

namespace shyft {
    namespace core {

        namespace kriging {

            /** This minimal implementation is
             * inspired by the approach and analysis found in
             * <a href="http://gstl.sourceforge.net">GsTL by Nicolas Remy 2001</a>
             *
             *  Reading the master thesis, especially the way he analyses and decomposes kriging and the application of it, will help
             *  understand the background, both theoretical and the mapping into software components as done here.
             */

            namespace covariance {

                /** exponential covariance */
                struct exponential {
                    double c;///< c = sill, or (sill-nug)
                    double minus3_over_a;///< -3/a, a is defined as range

                    /** construct a exponential  type callable covariance function
                     * \param c the height, sill of the covariance function, and corresponding semi-variogram
                     * \param a the practical range of the covariance, should be comparable to h in the h2 param of the call operator
                     */

                    exponential(double c,double a):c(c),minus3_over_a(-3.0/a) {}
                    /** computes the covariance for the specified distance h between two locations
                     * \param h2 specifies the h^2 where h is the distance measure
                     * \return exponential type covariance using specified h, a and c from constructor
                     */
                    inline double operator()(double h) const {
                        return c*std::exp(h*minus3_over_a);
                    }
                };

                /** gaussian covariance */
                struct gaussian {
                    double c;///< c = sill, or (sill-nug)
                    double minus3_over_a2;///< a is defined as range, we pre-compute to possibly speed up

                    /** construct a gaussian callable covariance function
                     * \param c the height, sill of the covariance function, and corresponding semi-variogram
                     * \param a the practical range of the covariance, should be comparable to h in the h2 param of the call operator
                     */
                    gaussian(double c, double a):c(c),minus3_over_a2(-3.0/(a*a)){}

                    /** computes the covariance for the specified h^2, where h is the distance between two locations
                     * \param h2 specifies the h^2 where h is the distance measure
                     * \return gaussian covariance using specified h2, a and c from constructor
                     */
                    inline double operator()(double h2) const {
                        return c*std::exp(h2*minus3_over_a2);
                    }

                };
            }

            namespace ordinary {

               /** build covariance matrix between pair of the observations [s..s_end>
                * plus one extra row/column to provide requirement to
                * the sum of weights equal to 1,
                * using the specified covariance function.
                *
                * The returned matrix is the A in the kriging equation(set)
                *
                *  Ax = b
                *
                * \tparam S source type iterators
                * \tparam F_cov the type of a covariance callable function
                * \param s the begin of the source range
                * \param s_end the end of the source range
                * \param f_cov a callable(S p1,S p2)->double that computes the covariance between p1 and p2
                * \return the covariance matrix(n+1 x n+1), n=sources.size(), symmetric, diagonal= f_cov(px,px), except last elem=0.0
                */
                template <class S, class F_cov>
                arma::mat build(S s, S s_end, F_cov && f_cov) {
                    using int_t = arma::uword;
                    const int_t n = (int_t)std::distance(s, s_end);
                    arma::mat c(n+1, n+1, arma::fill::eye);
                    for (int_t i = 0;i < n; ++i)
                        for (int_t j = i ;j < n; ++j)
                            c.at(j, i) = c.at(i, j) = f_cov(s[i], s[j]);

                    for(int_t j=0;j<n;++j)
                        c.at(j,n)=c.at(n,j)=1.0;// last row|col are all ones
                    c.at(n,n)=0.0; // zero out the end of diagonal element
                    return c;
                }

                /** build covariance matrix between
                * observations [s..s_end> and
                * destinations(grid) [d..d_end>
                * plus one row for the weigth sum requirement to 1.0
                * given the supplied covariance function.
                *
                * The returned matrix, represents the 'b' in all destination (typically grid) points
                * in the kriging equation set
                *
                *  Ax = b, and we compute the inverse of A, to compute the kriging weights x for all b's
                *
                * \tparam S source type iterators
                * \tparam D destination type iterators
                * \tparam F_cov the type of a covariance callable function
                * \param s the begin of the source range
                * \param s_end the end of the source range
                * \param d the begin of the destination range
                * \param d_end the end of the destination range
                * \param f_cov a callable(S p1,D p2)->double that computes the covariance between p1 and p2
                * \return the covariance matrix(n+1 x m),n=sources(obs),m=destinations(grid)
                */
                template <class S, class D,class F_cov>
                arma::mat build(S s, S s_end,D d,D d_end, F_cov && f_cov) {
                    using int_t = arma::uword;
                    const int_t n = (int_t)std::distance(s, s_end);
                    const int_t m = (int_t)std::distance(d, d_end);
                    arma::mat c(n+1, m, arma::fill::none);
                    for (int_t i = 0;i < n; ++i)
                        for (int_t j = 0;j < m; ++j)
                            c.at(i, j) = f_cov( s[i], d[j]);
                    for(int_t j=0;j<m;++j)
                        c.at(n,j) = 1.0;// fill in bottom row for the sum w to 1.0
                    return c;
                }

            }
            // later maybe: namespace simple {}
            // later maybe: namespace trend {}
        }
    }
}
