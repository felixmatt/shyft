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
///  of first SCEUA implemented in
///  the Enki project by Sjur Kolberg, Sintef
///
#include "optimizer_utils.h"

namespace shyft {
    namespace core {
        namespace optimizer {
            using namespace std;

            template <class vec>
            struct sort_by_value_asc {
                const vec& v;
                sort_by_value_asc(const vec& v_):v(v_) {}
                bool operator()(size_t a,size_t b) const {
                    return v[a]<v[b];
                }
            };

            inline void construct_sorted_pivot_table(const double* v, size_t* ix,size_t n) {
                for(size_t i=0;i<n;i++) ix[i]=i;
                sort(ix,ix+n,sort_by_value_asc<const double*>(v));
            }

            enum OptimizerState { NotStarted=-1, Searching, FinishedFxConvergence, FinishedXconvergence, FinishedMaxIterations, FinishedUserRequest, FinishedMaxTime, SavingTimeSeries, ProcessTerminated };

            /** \brief The sceua implements the Shuffle Complex Evolution University of Arizona variant of
             *  sce published by Duan et. al (1993)
             *
             */
            class sceua  {
                #ifdef WIN32
				mutable std::mt19937 generator;
#else
                mutable default_random_engine generator;
#endif
				mutable uniform_real_distribution<double> distribution;//(0.0,1.0)

            public:
                sceua():distribution(0.0,1.0) {}
                OptimizerState find_min(
                    const size_t n,			// Number of active parameters
                    const double Xmin[],	// Lower limit of all n parameters
                    const double Xmax[],	// Upper limit of all n parameters
                    double x[],				// Xmin < x < Xmax. The [input]initial/[output]current/optimal n parameter values
                    double& FxOptimumFound,	// The optimal value found
                    ifx& Fx,// Fx(x1..xn) The function that takes x[n] parameters
                    // Stop criteria goes here: They are important, since evaluating Fx takes time.
                    double FxEpsilon,		// 1. Stop when diff of 5 last samples: 2x(Fmax-Fmin)/(Fmax+Fmin)<FxEpsilon
                    double FxSolutionMin,	// 2. Stop when Fx is within FxSolutionMin..FxSolutionMax
                    double FxSolutionMax,	//   (to disable, set FxSolutionMin > FxSolutionMax)
                    const double xEpsilon[],// 3. Stop when all x[] are just moving within xEpsilon range
                    size_t maxIterations	// 4. Stop when maxiterations/invocations are reached
                    )
                    const;
            private:
                /// Function evolve corresponds to the competitive complex ecolution (CCE),
                /// fig. 2 in Duan et. al (1993). It is called from sceua.
                void evolve(double *ax[],		// ax[m][n]  are the parameter values of complex to be evovled.
                    double af[],				// af[m]     is the objective function value of the subcomplex.
                    size_t m,					// m         is the size of the complex.
                    size_t n,					// n         is number of parameters to be optimized.
                    ifx& fn,	// The function (hydrological model) to be evaluated
                    const double Xmin[],		// Xmin[n]   are the minimum values of the parameters to be optimized.
                    const double Xmax[],		// maxi[n]   are the maximum values of the parameters to be optimized.
                    double x[],					// x[n]  is all the parameter vector required by the model.
                                                // The actual length of this vector is not required in this subroutine.
                    size_t& evaluations,		// evaluations is a counter for how many times the model is called.
                    size_t complexno			// For diagnostics: The number of this complex.
                    )
                    const;

                void mutate(double *Xset[], double newX[], int na, int nprm) const;
                void random_generate_x(size_t n, double newX[], const double Xmin[], const double Xmax[]) const;
                double random01() const { return distribution(generator); }
            };
        }
    }
}
