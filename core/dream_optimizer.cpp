#include "core_pch.h"
#include "dream_optimizer.h"

using namespace std;
namespace shyft {
namespace core {
namespace optimizer {

    double dream::find_max(
            ifx &fx,
            vector<double>& x,	// 0 < x < 1 The [input]initial/[output]current/optimal n parameter values
            size_t max_iterations) const {

        const size_t n_parameters = x.size();
        double fx_optimal = -numeric_limits<double>::max();
        if (n_parameters < 1)
            throw runtime_error("dream::find_max(): n_parameters must be >0 ");
        // Definition of configuration parameters
        const size_t n_chains = n_parameters; // TODO: Relevant values are n_parameters or 2*n_parameters. Could also make it entirely user-configurable?

        const size_t n_cr				= 4;								// Number of different cr values in the Dream algorithm
        size_t n_outliers				= 0;								// Number of detected and re-initiated outlier chains
        size_t n_pri_rejected			= 0;								// Number of proposals rejected on prior alone (outside box)
        size_t n_post_rejected			= 0;								// Number of proposals rejected after Metropolis step
        size_t n_accepted				= 0;								// Number of proposals accepted by the Metropolis step
        size_t i_last_outlier_reset		= 0;								// Index of last outlier-reset (gives which iteration we last had to replace an outlier chain with best results)

        const size_t min_burnins		= 100;								// Minimum number of iterations before burnin period is ended
        const size_t burnin_increments	= 10;								// Increment number of iterations when burnin period needs to be extended
        size_t n_burnins				= min_burnins;						// Current number of burnin iterations to perform (initially min_burnins, but may be incremented using burnin_increments)

        // Preparing data structures (using stack allocation for small arrays, heap allocation with vector for large arrays,
        // and currently for easy copying using vector for chainstates and curprob variables which must be copied into
        // all_chain_states and all_chain_prob variables for each iteration).
        vector<vector<vector<double>>> all_chain_states;					// The history of chain/parameter states [nIters*n_chains*n_parameters]
        vector<vector<double>> all_chain_prob;							// The current-state posterior probability for all chains in all iterations [nIters*n_chains]

        vector<vector<double>> chain_states;								// The states (values of all parameters) for all chains in the current iteration [n_chains*n_parameters]
        vector<double> chain_prob;										// The posterior log-postdensities (unscaled) for all chains in the current iteration [n_chains]
        chain_prob.resize(n_chains);
        chain_states.resize(n_chains);
        for (size_t chain = 0; chain < n_chains; ++chain) {
            chain_states[chain].resize(n_parameters);
        }

        vector<double> x_candidates(n_parameters,0.0);	// Temporary vector containing parameter proposals to be evaluated
        vector<double> omega(n_chains,0.0);	// Last-half average log-post-densities for each chain
        vector<int> cr_L(n_cr,0);			// Number of candidates generated for each cr value
        vector<double> cr_D(n_cr,0.0);		// Sum of sq.norm.dist achieved for each cr value
        vector<double> cr_M(n_cr,0.0);		// Probability of each cr value (sum=1)
        vector<double> x_variance(n_parameters,0.0);// Variance over chains for each par (used to calculate jump_distance)
        // Initialize some local variables
        size_t iteration_count = 0;
        double log_x_limit;

        for (size_t i = 0; i < n_chains; ++i) {
            // Generate random parameter values in their normalized range
            for (size_t p = 0; p < n_parameters; ++p) {
                chain_states[i][p] = random01();
            }
            chain_prob[i] = fx.evaluate(chain_states[i]);

            // Store the best parameter set achieved so far, to serve as diagnostic output,
            // and to replace outlier states during burn-in with the HPD parameter set achieved so far.
            if (chain_prob[i] > fx_optimal) {
                fx_optimal = chain_prob[i];
                x=chain_states[i];
            }
        }

        // Add information from the initial iteration into vectors keeping record of the information for the entire run,
        // but keep current values because it is used in the next iteration when requiring previous values.
        all_chain_states.push_back(chain_states);
        all_chain_prob.push_back(chain_prob);

        update_cr_dist( cr_M, cr_L, cr_D);

        // Start the sampling, with the burn-in adjustments placed within conditions, not in a separate loop.
        bool doing_burnin = true;
        bool converged = false;
        while (!converged) {

            // Continue with next iteration.
            // Done with one iteration (the initial iteration, iteration 0, is handled separately before
            // this loop, so the first execution of this loop is when entering the second iteration, iteration 1).
            // Starting the iteration by checking if the optimization has converged, and if so the loop is
            // aborted immediately (break).
            ++iteration_count;

            // During burn-in, detect and reject outliers using the last 50% of
            // the chains to estimate each chain's average log-likelihood.
            bool outliers = false;
            if (doing_burnin && check_for_outlier_chain(all_chain_prob, i_last_outlier_reset, omega, log_x_limit)) {
                outliers = true; // One or more outliers will be replaced.
                i_last_outlier_reset = iteration_count; // New starting point for outliers and GRC.
            } else {
                // Compute the Gelman-Rubin convergence criterion (if not an outlier was just detected).
                converged = true; // Initialize termination check
                double avg_grc = 0; // Par-averaged Gelman-Rubin convergence criterion
                for (size_t i = 0; i < n_parameters; ++i) {
                    // Gelman-Rubin convergence criterion for param i, estimated on last 50% of the chain since last reset.
                    double grc = get_gr_convergence(all_chain_states, all_chain_states.size(), n_chains, n_parameters, i_last_outlier_reset, i);
                    if (grc > 1.2 || grc < 0) // what ?!?
                        converged = false;
                    if (grc < 0) {
                        // grc is -1 if number of iterations since start or last outlier reset is insufficient.
                        // grc is -2 for a parameter which has no variance for at least one of the chains.
                        avg_grc = grc; // A negative grc for one parameter sets avg_grc.
                        break;
                    }
                    avg_grc += grc / n_parameters;
                }

                // Calculate and update per parameter summaries, and evaluate the convergence criterion given in
                if (doing_burnin && (avg_grc > 1.5 || avg_grc < 0)) { // Gelman and Rubin (1992). .. needs more burnin
                    n_burnins = max(n_burnins, iteration_count + burnin_increments); // Keep increasing nBurnin as long as GRC > 1.5.
                }
                if (doing_burnin)
                    converged = false;
                if (converged) {
                    break; // Terminate sampling due to convergence.
                } else {
                    ;//No convergence: At least one parameter's GRC is above 1.2
                }
            }


            // Update the inter-chain variance
            for (size_t parameter = 0; parameter < n_parameters; ++parameter) {
                double mean = 0.0;
                x_variance[parameter] = 0.0;
                for (size_t chain = 0; chain < n_chains; ++chain) {
                    mean += chain_states[chain][parameter]; // Sum of par[i] over chains
                    x_variance[parameter] += pow(chain_states[chain][parameter], 2); // SumSQ of par[i] over chains
                }
                mean /= n_chains; // Mean of par[i] over chains
                x_variance[parameter] /= n_chains;
                x_variance[parameter] -= pow(mean, 2); // Variance of par[j] over chains
            }

            // Develop the MCMC chain with index 'chain'
            for (size_t chain = 0; chain < n_chains; ++chain) {
                double jump_distance = 0; // The distance between previous (existing) parameter set and the new parameter set (accepted candidate) - squared and scaled by the previous-iter inter-chain posterior variance.
                if (doing_burnin && outliers && omega[chain] < log_x_limit) {
                    // This chain is an outlier; replace the chain states with the currently best parameter set and prob.
                    // Chains with both current and average llh (log-likelihood) lower than Q1-2*(Q3-Q1) will be aborted and
                    // restarted at the highest-probability state found so far in any chain.
                    for (size_t i = 0; i < n_parameters; ++i) {
                        // First calculate jump distance between the new candidates (the currently best
                        // parameter set) and the existing parameter set for the chain.
                        jump_distance += pow(chain_states[chain][i] - x[i], 2) / x_variance[i];
                        // Then set the new candidate (current best) as the current values.
                        chain_states[chain][i] = x[i];
                    }
                    chain_prob[chain] = fx_optimal;
                    n_burnins = iteration_count + min_burnins;
                    ++n_outliers;
                } else {
                    // No outlier, perform as normal.

                    // Draw a random cross-over probability (cr).
                    // First draw a random index between 1 and n_cr, inclusive.
                    // The multinomial probability distribution is provided in the cr_M.
                    // The corresponding cr value is found by dividing the index by n_cr.
                    // (From MCsetup::DrawCRindx in ENKI source)
                    double p = random01();
                    double pSum = 0;
                    size_t i_cr = 0; // Same as m in formulas
                    while (i_cr < n_cr) {
                        pSum += cr_M[i_cr];
                        if (pSum > p || i_cr == n_cr - 1) // Stop when pSum>p (or at last index), and keep the i_cr at that point (avoid the increment below)
                            break;
                        ++i_cr;
                    }
                    double cr = (double) (i_cr + 1) / (double) n_cr; // Individual-parameter cross-over probability (the cr value)

                    // Generate a candidate point for chain chain by adding up k difference-vectors
                    // between randomly selected pairs of the other chains
                    size_t d_eff = 0; // Number of actually changed parameter components
                    generate_candidate_parameters(x_candidates, chain, n_chains, n_parameters, cr, d_eff, chain_states);

                    // Check if the candidate point is legal or must be rejected
                    bool reject = false;
                    for (size_t i = 0; i < n_parameters; ++i) {
                        if (x_candidates[i] < 0 || x_candidates[i] > 1) {
                            // Parameter value is outside it's limits
                            reject = true;
                            break;
                        }
                    }

                    // A proposed parameter value was outside range.
                    // Probability is 0 without evaluation, but the rejected parameter
                    // vector is still written out to the text file for diagnostics.
                    if (reject) {
                        ++n_pri_rejected;
                    } else {
                        // The proposal is inside the allowed box, we must run the model to evaluate likelihood
                        // Evaluate the posterior log-density for the candidate parameters, EvalModel provides the likelihood part.
                        double cand_prob = fx.evaluate(x_candidates);

                        // If the new candidate parameters results in better results (higher posterior log-density) than
                        // the previous results for the same chain, the candidates will be accepted and become the
                        // next item in the sequence. But if it has worse results (lower posterior log-density) it
                        // can still be accepted - with a probability given by the ratio between the density of the
                        // new (candidate) and the existing parameter set. This ratio between the density of the
                        // candidate parameter set and the existing parameter set is calculated by using the exp function
                        // on the log densities that the fx evaluation returns. Since exp is the the inverse of log, this
                        // converts the log-density to density values. And exp(a-b) is the same as exp(a)/exp(b), so this
                        // gives the ratio. To keep all candidates with better density and a randomized rejection of worse,
                        // we compare the ratio to a random [0,1) value. If exp(a), the density of candidate, is larger
                        // than exp(b), the density of existing, the result is larger than 1 and it will never be less
                        // than a random [0,1) value. But if the density of the candidate is less than the density of
                        // the existing it will be in the range [0,1], and therefore it may be rejected depending on
                        // how it compares to a random [0,1) value!
                        //    If the candidate is not accepted we call it post-rejected (rejected after evaluation),
                        // and the existing parameter vector and results will be repeated for this chain (same as when pri-rejected).
                        if (exp(cand_prob - chain_prob[chain]) < random01()) { // Using the ratio between the density of the candidate parameter vector and the existing compared to random [0,1) number to randomly reject candidates that are not better than the existing.
                            // The proposal is a posteriori rejected chainstates[chain] already holds the values.
                            n_post_rejected++;
                        } else {
                            // The proposal is accepted. This will also happen when d_eff==0, since then candp==curprob[chain].
                            ++n_accepted;
                            chain_prob[chain] = cand_prob;
                            for (size_t i = 0; i < n_parameters; ++i) {
                                // Calculate jump distance between the new (now accepted) candidate parameter set
                                // and the existing (previous) parameter set for the chain.
                                jump_distance += pow(chain_states[chain][i] - x_candidates[i], 2) / x_variance[i];
                                // Make the accepted candidate parameter set the current parameter set for the chain
                                chain_states[chain][i] = x_candidates[i];
                            }

                            // Store the best parameter set achieved so far, to serve as diagnostic output,
                            // and to replace outlier states during burn-in with the HPD parameter set achieved so far.
                            if (chain_prob[chain] > fx_optimal) {
                                fx_optimal = chain_prob[chain];
                                x=chain_states[chain];
                            }
                        } // accepted (exp(candp-curprob[chain]) >= unif01())
                    } // A posteriori evaluated block

                    // Update the count and dist for cr value m.
                    if (doing_burnin) {
                        cr_L[i_cr]++; // Number of candidates generated for cr value m (i_cr)
                        cr_D[i_cr] += jump_distance; // If accepted (only then jump_distance is > 0), alter sum of sq.norm.dist achieved for cr value m (i_cr)
                    }

                } // Non-outlier block
            } // for (int chain=0; chain<n_chains; chain++) // End single-chain development.


            if (doing_burnin) {
                if (iteration_count >= n_burnins) {
                    if (n_accepted > n_chains * 20) { // At least 20 proposals accepted until we end burnin
                        doing_burnin = false;
                    } else {
                        n_burnins = n_burnins + burnin_increments;
                    }
                } else {
                }
                update_cr_dist( cr_M, cr_L, cr_D); // The last i iter set the cr_D and cr_L arrays
            }
            // Add information from the initial iteration into vectors keeping record of the information for the entire run,
            // but keep current values because it is used in the next iteration when requiring previous values.
            all_chain_states.push_back(chain_states);
            all_chain_prob.push_back(chain_prob);
        } // while (converged)



        if (!converged)
                throw runtime_error("dream::find_max: did not converge");

        //size_t numFx = iteration_count * n_chains - n_pri_rejected;
        // Log message to event log
        //setting the return values to an array for correct logging
        //returnSizes[0] = numFx;
        //returnSizes[1] = n_chains;
        //returnSizes[2] = iteration_count;
        //returnSizes[3] = n_burnins;
        //returnSizes[4] = n_outliers;
        //returnSizes[5] = n_accepted;
        //returnSizes[6] = n_pri_rejected;
        //returnSizes[7] = n_post_rejected;

        return fx_optimal;
    }


/// MCsetup::update_cr_dist is an internal function used by the Dream algorithm
/// to alter the probability of the different cr values according to the average
/// jump length achieved by the different values. Used only during burn-in.
/// m is the index in the cr array, m = 1..n_cr
/// cr_d[j] is the sum of squared distance achieved with cr[j]
/// cr_l[j] is the number of attempts with cr[j]
/// cr_m[j] is the average squared jumping distance with cr[j]
void dream::update_cr_dist(vector<double>& cr_m, const vector<int>& cr_l, const vector<double>& cr_d) const {
    double sum = 0;
    const size_t n_cr=cr_l.size();//TODO:We could assert on all equal.
    // Check if all m have had at least one accept
    bool allpos = true; // All cr values have at least one accept
    for (size_t i = 0; allpos && i < n_cr; ++i) {
        sum += cr_d[i]; // Sum of cr_D over all m values
        if (cr_d[i] == 0) ///TODO: 0 ?
            allpos = false;
    }
    // Only in the case that all m have at least one accept, we update the cr probability distribution.
    if (allpos) {
        sum = 0;
        for (size_t i = 0; i < n_cr; ++i) {
            cr_m[i] = cr_d[i] / cr_l[i];
            sum += cr_m[i];
        }
        for (size_t i = 0; i < n_cr; ++i)
            cr_m[i] /= sum;
    } else {
        for (size_t i = 0; i < n_cr; ++i)
            cr_m[i] = 1.0 / n_cr;
    }
}

/// MCsetup::std_norm is a standard normal distribution random number generator
double dream::std_norm() const {
    double u1, u2, tmp;
    if (super_hack_stored) {
        // An extra random number was generated the last time std_norm() was called, simply return it and delete it.
        super_hack_stored = false;
        return stored_std_norm_super_hack;
    } else {
        // Draw two standard normal random numbers.
        // rng_unif11() returns a uniform random between -1 and 1, non-inclusive, and is better than
        // the system-supplied rand() due to utilization of 32-bit maxima (opposed to 16-bit).
        // The polar-coordinate transformation to Normal avoids the time-consuming calls to trigonometric
        // functions more frequently employed.
        // Routines from Numerical Recepies in C.
        do {
            u1 = random11();
            u2 = random11();
            tmp = u1 * u1 + u2 * u2;
        } while (tmp >= 1.0 || tmp == 0.0);
    }
    tmp = sqrt(-2.0 * log(tmp) / tmp);
    stored_std_norm_super_hack = u1 * tmp;
    super_hack_stored=true;
    return u2 * tmp;
}

/// GenerateCandidatePoint (originally MCsetup::DreamNewPoint in ENKI source)
/// generates a new candidate point for one MCMC chain in DREAM.
/// The current state of the other chains is used to generate the proposal point
/// Single-parameter entries in each point may be rejected with probability 1-cr,
/// so that the candidate differs from current only for a subset of dimensions.
/// Theory: See Vrugt et al (2009) specified above the RunDream function.
void dream::generate_candidate_parameters(vector<double> &x_candidate,							// Returned vector of length d
        const size_t I,						// Current chain number
        const size_t N,						// Number of chains
        const size_t d,						// Number of parameters
        const double cr,						// cross-over probability
        size_t &d_eff,							// # actually changed params
        const vector<vector<double>>& states)	// Current states (parameter values) for all chains
const {
    size_t delta = 1 + (size_t)floor(3 * random01());	// 1, 2, or 3 pairs used to generate a proposal
    delta = min(delta, (N - 1) / 2);							// If N<3, delta is 0, since it is not possible
    delta = min(delta, (size_t)3);						// to generate a pair of chains other than I.
    size_t R[6];											// In that case, only b^* will affect proposal deviation.
    for (size_t k = 0; k < 2 * delta; ++k) {					// With delta equal to 1, 2, or 3,
        R[k] = I;										// draw two, four or six different R values
        while (R[k] == I) {								// less than N and different from I.
            R[k] = min(N - 1, (size_t) floor(random01() * N));
            for (size_t j = 0; j < k; ++j)					// Check formerly drawn values of R
                if (R[k] == R[j])							// Value is used before,
                    R[k] = I;								// set to I to ensure redraw.
        }
    }

    for (size_t i = 0; i < d; ++i)
        x_candidate[i] = 0;

    // Generate a candidate point for chain i by adding up k difference-vectors
    // between randomly selected pairs of the other chains.
    for (size_t k = 0; k < delta; ++k) {
        for (size_t i = 0; i < d; ++i)
            x_candidate[i] += states[R[2 * k]][i]	// For each dim, accumulate the pair diff over delta pairs.
                              - states[R[2 * k + 1]][i];	// If the chains are wide apart, the proposed jumps will
    }										// tend to be larger than when the chains agree.

    // Replace each element j=1,...,d in the proposal with the current coordinate (parameter value) with a cross-over probability cr.
    d_eff = d; // Number of dimensions in which candidate differs from current
    for (size_t i = 0; i < d; ++i) {
        if (random01() >= cr) {
            // Keep the current value for this parameter
            --d_eff;// d_eff holds the number of altered parameters where the candidate differs from the current state #i.
            x_candidate[i] = -numeric_limits<double>::max();
        }
    }

    double gamma; // Depends on num dims, num altered dims, num pairs
    if (random01() < 0.2)
        gamma = 1; // Facilitate mode jumping in 20% of the generations.
    else
        gamma = 2.38 / sqrt(2.0 * delta * d_eff); // If d_eff==0, gamma is undefined, but is never used, since x_candidate[j]will be -9999 for all j.

    double b = 0.05; // Vrugt et al page 278, Case Studies.
    double sd = 0.001; // sd is the square root of Vrugt et al's b^*.

    for (size_t i = 0; i < d; ++i) {
        if (x_candidate[i] == -numeric_limits<double>::max())					// The j'th dimension was selected to ignore update proposal
            x_candidate[i] = states[I][i];					// Candpoint stays at current value for this parameter
        else {										// Candpoint adopts the proposed value for this parameter
            double e = (double) random11() * b;	// Randomness in the increment direction
            x_candidate[i] = states[I][i]
                             + (1 + e) * gamma * x_candidate[i]
                             + (double) normal(0, sd);	// Additive random vector component, epsilon in Vrugt et al.
        }
    }
}

/// MCsetup::check_for_outlier_chain (originally MCsetup::DreamOutlier in ENKI source)
/// is an internal function used by RunDREAM autocalibration.
/// DreamOutlier identifies chains for which the average log posterior density
/// (LPD) over the last 50% of the iterations is much lower than from the other chains.
/// The limit is based on the InterQuartile Range, which identifies the 25% and the 75%
/// quantiles (Q1 and Q3) of chain-average densities, and the difference between the two (IQR).
/// Return value is true if any of the chains is an outlier, pNcChains points to the pChainProb
/// NcVar containing all iterations' LPD values for all chains. lPlim reports the limit LPD for
/// being an outlier, and omega reports the last-50%-average-LPD for all chains. The actual
/// decision for each chain, and the replacement, is left to the calling site.
/// Removing an outlier invalidates a lot of iterations with very bad LPD, which affects not
/// only the outlier chain, but also the computation of the IQR. It is therefore necessary to
/// to reset the burn-in period after an outlier removal.
/// TODO 1: Call DreamOutlier once at the start of each iteration, mark the outlier chains,
/// and let the removal of these take place instead of an ordinary MCMC step.
/// TODO 2: Use a lower limit instead of the n/2, where any removal of an outlier sets the
/// value of this lower limit to the current iter, and don't call DreamOutlier unless
/// iter-lowlimit is at least 10.
bool dream::check_for_outlier_chain(const vector<vector<double>>& chain_probabilities, size_t reset, vector<double>& omega, double &lPlim) const {
    const size_t n_iterations=chain_probabilities.size();
    const size_t n_chains=omega.size();
    size_t start = reset + (n_iterations - reset) / 2; //The first iteration used in calculating omega
    if (start > n_iterations) return false;
    size_t length = n_iterations - start; // Number of iterations used in calculating omega (last 50% of the period since last replacement)
    if (length < 5) return false;

    // Initialize the omega[] array
    fill(begin(omega),end(omega),0.0);
    // Calculate the omega array by first summing up log posterior likelihood (LPD)
    // for each chain, and over the last [length] iterations.
    for (size_t i = start; i < n_iterations; ++i)
        for (size_t j = 0; j < n_chains; ++j)
            omega[j] += chain_probabilities[i][j];
    for (size_t i = 0; i < n_chains; ++i)
        omega[i] /= (length); // Average posterior log-likelihood for each chain

    // The array omega[] is the unsorted chain ordered table.
    // Now we must create array sorted[] should contain the same values but sorted.
    auto sorted(omega);
    sort(begin(sorted),end(sorted));

    // Calculate the LPD limit value to be used to decide from the omega values if a chain is an outlier
    lPlim = sorted[n_chains / 4]			// The 25% quantile Q1
            - 2 *
            (sorted[3 * n_chains / 4]	// The 75% quantile Q3
             - sorted[n_chains / 4]);	// The 25% quantile Q1; Q3-Q1=IQR

    // Search all chains and check if at least one of them has an LPD lower than the limit
    for (size_t i = 0; i < n_chains; ++i)
        if (omega[i] < lPlim)
            return true; // Chain i has been detected as an outlier, so we have at least one outlier chain!

    // If not returned, then we have no outlier chains.
    return false;
}

/// get_gr_convergence (originally MCsetup::GRConvergence in ENKI source)
/// is an internal function used by the Dream algorithm.
/// GRConvergence evaluates and returns the convergence criterion given in Gelman,
/// A. and D. B. Rubin (1992): Inference from multiple iterative simulation
/// using sequences. Stat. Science, vol 7, no 4, p. 457-472.
/// The function does not access any member attributes of MCsetup.
double dream::get_gr_convergence(const vector<vector<vector<double>>>&/* double*** */ states, size_t n_iterations, size_t n_chains, size_t n_parameters, size_t reset, size_t parameter) const {
    // Argument states contains the chain states for all iterations as a 3D array: Iterations, chain and parameter.
    // Arguments n_iterations, n_chains and n_parameters gives us the size of the three dimensions.
    // Argument reset is index of last outlier-reset.
    // Argument parameter is the index of the current parameter, referring to the last dimension of input array states, to calculate GRC for.

    if (/*!states || */n_iterations < 1 || n_chains < 1 || n_parameters < 1) return -1.0;
    size_t start = reset + (n_iterations - reset) / 2; //The first iteration used in calculating omega
    if (start > n_iterations) return -1.0; // Return dummy GRC if too few data to estimate
    size_t length = n_iterations - start; // Number of iterations used in calculating omega
    if (length < 5) return -1.0; // Return dummy GRC if too few data to estimate

    double* chainmean = __autoalloc__(double, n_chains);
    double* chainvar = __autoalloc__(double, n_chains);

    // Follow Gelman & Rubin's symbolism, and convert from int to double to be used in calculations
    double m = (double) n_chains;
    double n = (double) length;

    double meanchainmean = 0; // Mean of chain averages
    double meanchainvar = 0; // Mean of chain variances
    double varchainmean = 0; // Variance of chain averages
    double varchainvar = 0; // Variance of chain variances
    double meansqchainmean = 0; // Mean of squared chain averages

    for (size_t i = 0; i < n_chains; ++i) {
        chainmean[i] = chainvar[i] = 0.0;
        for (size_t j = start; j < n_iterations; ++j) {
            double v = states[j][i][parameter];
            chainmean[i] += v;
            chainvar[i] += pow(v, 2);
        }
        chainmean[i] /= n;
        chainvar[i] /= n;
        chainvar[i] -= pow(chainmean[i], 2);
        if (chainvar[i] < 0) return -2.0;
        chainvar[i] *= n / (n - 1); // Unbiased population variance estimate
        meanchainmean += chainmean[i];
        meanchainvar += chainvar[i];
        varchainmean += pow(chainmean[i], 2);
        varchainvar += pow(chainvar[i], 2);
        meansqchainmean += pow(chainmean[i], 2);
    }

    meanchainmean /= m; // Mean of chain means
    meansqchainmean /= m; // Mean of squared chain means
    meanchainvar /= m; // Mean of chain variances, W in Gelman & Rubin
    varchainmean /= m; // Var of chain means, B/n in Gelman & Rubin where Gelman & Rubin's n is our length.
    varchainmean -= pow(meanchainmean, 2);
    if (varchainmean < 0) return -2.0;
    varchainmean *= m / (m - 1);
    varchainvar /= m;
    varchainvar -= pow(meanchainvar, 2);
    if (varchainvar < 0) return -2.0;
    varchainvar *= m / (m - 1);
    double parestvar = meanchainvar * (n - 1) / n + varchainmean; // Estimated target variance Eq 3 in Gelman & Rubin
    double v_hat = parestvar + varchainmean / m; // Squared scale in Student's t-distrib of a param.

    double cov_var_mean = 0; // Covariance of var and mean over chains
    double cov_var_sqmean = 0; // Covariance of var and squared mean over chains
    for (size_t i = 0; i < m; ++i) {
        cov_var_mean += (chainvar[i] - meanchainvar) * (chainmean[i] - meanchainmean);
        cov_var_sqmean += (chainvar[i] - meanchainvar) * (pow(chainmean[i], 2) - meansqchainmean);
    }

    cov_var_mean /= m;
    cov_var_sqmean /= m;

    double 								// Eq. 4 in Gelman & Rubin
    var_hat_v_hat = pow((n - 1.0) / n, 2.0) * varchainvar / m	// ((n-1)/n)^2 * var(s_i^2)/m
                    + pow((m + 1.0) / (m * n), 2.0)					// ((m+1)/mn)^2 (n_chains=m, length=n)
                    * 2 * pow(varchainmean * n, 2) / (m - 1.0)			// * 2B^2/(m-1) (varchainmean = B/n)
                    + 2 * (m + 1.0) * (n - 1.0) / (m * pow(n, 2.0))		// 2(m+1)(n-1)/(mn^2)
                    * n / m										// n/m
                    * (cov_var_sqmean
                       - 2 * meanchainmean * cov_var_mean);				// (cov(s_i^2,xbar_i^2)-2xbar*cov(s_i^2,xbar_i))

    double df = 2 * pow(v_hat, 2) / var_hat_v_hat; // Degrees of freedom (equation in text just above eq. 4 in Gelman&Rubin)
    if (df <= 2.0)
        return -2.0;

    double sqrtR = sqrt(v_hat / meanchainvar * df / (df - 2.0)); // Convergence criterion, equation in text below eq. 4 in Gelman&Rubin

    if (!isfinite(sqrtR))
        return -2.0;
    return sqrtR;
}
}
}
}
