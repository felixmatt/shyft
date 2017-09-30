#pragma once
///	Copyright 2012 Statkraft Energi A/S
///
///	This file is part of SHyFT.
///
///	SHyFT is free software: you can redistribute it and/or modify it under the terms of
/// the GNU Lesser General Public License as published by the Free Software Foundation,
/// either version 3 of the License, or (at your option) any later version.
///
///	SHyFT is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
/// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
/// PURPOSE. See the GNU Lesser General Public License for more details.
///
///	You should have received a copy of the GNU Lesser General Public License along with
/// SHyFT, usually located under the SHyFT root directory in two files named COPYING.txt
/// and COPYING_LESSER.txt.	If not, see <http://www.gnu.org/licenses/>.
///
///  Theory is found in: Vrugt, J. et al: Accelerating Markov Chain Monte Carlo
///  simulations by Differential Evolution with Self-Adaptive Randomized Subspace
///  Sampling. Int. J. of Nonlinear Sciences and Numerical Simulation 10(3) 2009.
///
/// Thanks to Powel for contributing with the reimplementation
#ifdef SHYFT_NO_PCH
#include <vector>
#include <cstring>
#endif // SHYFT_NO_PCH

namespace shyft {
    namespace core {
        namespace optimizer {
            using namespace std;
            ///< just temporary simple abstract interface for the target function, later just a callable
            struct ifx {
                virtual double evaluate(const vector<double>& x)=0;
                double evaluate(size_t n,const double *x) {
                    vector<double> xx(x,x+n);
                    return evaluate(xx);
                }
            };

            /// \brief  __autoalloc__ uses alloca and typecast to allocate an array on stack,
            /// that are automatically deallocated when the calling function returns.
            /// NB: only useful if a lot of (small) repetitive allocations (speed/memory)
            /// NB: Too large arrays will result in stack overflow (as a rule of thumb sizes up to 1024 should be safe).
            /// NB: To make sure the allocation is in the correct scope, this needs be a macro.
            #define __autoalloc__(TP,n)((TP*)alloca(n*sizeof(TP)))

            ///< fastcopy is a template wrapper for memcpy for copying arrays.
            template<class T>
            static inline void fastcopy(T*dst,const T*src,size_t n) {
                memcpy(dst,src,n*sizeof(T));
            }
        }
    }
}
