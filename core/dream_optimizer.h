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
///  of first DREAM implemented in
///  the Enki project by Sjur Kolberg, Sintef
///
#ifdef SHYFT_NO_PCH
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <random>

#endif // SHYFT_NO_PCH

#include "optimizer_utils.h"

namespace shyft {
    namespace core {
        namespace optimizer {
            using namespace std;
            /** \brief dream optimizer implementing the
              *
              *  Originally MCsetup::RunDREAM in ENKI source, invokes
              *  the DiffeRential Evolution Adaptive Metropolis MCMC sampler to estimate the
              *  posterior distribution of parameters.
              *
              * The number of chains (n_chains; N in Vrugt et al, 2009) is often chosen as d (n_parameters) or 2*d.
              *
              * Theory is found in: Vrugt, J. et al: Accelerating Markov Chain Monte Carlo
              * simulations by Differential Evolution with Self-Adaptive Randomized Subspace
              * Sampling. Int. J. of Nonlinear Sciences and Numerical Simulation 10(3) 2009.
              */
            class dream  {
                // TODO: get rid of these, use std::random for all needed functionality
                mutable bool super_hack_stored=false;
                mutable double stored_std_norm_super_hack=0.0;
#ifdef WIN32
				mutable std::mt19937 generator;
#else
                mutable default_random_engine generator;
#endif
				mutable uniform_real_distribution<double> distribution;//(0.0,1.0)

            public:
                dream():super_hack_stored(false),distribution(0.0,1.0) {}
                /** \brief find x so that fx is at maximum.
                 *
                 *
                 *  \param fx callable of type ifx
                 *  \param  x (in,out) [0..1] normalized starting point for the x-vector
                 *  \param  max_iterations stop at max iterations
                 *
                 *  \return the found maximum value
                 *  \throw runtime_exception if wrong parameters, or no convergence
                 */
                double find_max(ifx &fx,vector<double>& x,size_t max_iterations) const;

            private:
                void generate_candidate_parameters(vector<double> &cand,// Returned vector of length d
                                                 const size_t I,		// Current chain number
                                                 const size_t N,		// Number of chains
                                                 const size_t d,		// Number of parameters
                                                 const double cr,		// Cross-over probability for individual parameters
                                                 size_t &d_eff,			// Number of actually changed parameter components
                                                 const vector<vector<double>>& states)	// Current states (parameter values) for all chains
                                                 const;
                bool check_for_outlier_chain(const vector<vector<double>>& chainProbabilities,
                                          size_t reset, vector<double>& omega, double &lPlim) const;
                void update_cr_dist(vector<double> &cr_m, const vector<int> &cr_l, const vector<double>& cr_d) const;
                double get_gr_convergence(const vector<vector<vector<double>>>&/* double*** */ states,
                                        size_t nIterations, size_t nChains, size_t nParams, size_t reset, size_t iParam) const;
                double std_norm() const; // returns a standard Normal random number
                double normal(double mean, double sd) const { return std_norm()*sd + mean; }; // Normal random number generator taking mean and standard deviation as input and returning N(mean,sd^2) random number
                double random01() const { return distribution(generator); } // Helper function for accessing the random number generator.
                double random11() const { return random01()*2.0-1.0; } // Helper function for accessing the random number generator and scaling the result.
                double random(double minv, double maxv) const { return minv + random01()*(maxv-minv); } // Helper function for accessing the random number generator and scaling the result.
            };
        }
    }
}
