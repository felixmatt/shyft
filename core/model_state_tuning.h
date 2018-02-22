#pragma once
#include <vector>
#include <string>
#include <dlib/optimization.h>
#include "cell_model.h"

namespace shyft {namespace core {
using std::string;
/**The result of region_model state-adjustemt */
struct q_adjust_result {
    double q_0{0.0}; ///< m3/s  for the timestep before adjustemnt
    double q_r{0.0}; ///< m3/s for the timestep after adjustemnt
    string diagnostics;///< empty if ok, otherwise diagnostics
    q_adjust_result()=default;
};

/** The adjust_state_model is an algorithm that takes
 * a region-model and optional list of catchments
 * represented as catchment-ids.
 *
 * The tune_flow functions provides the needed function
 * to adjust state so that the wanted, usually observed flow is reached.
 *
 * \tparam RM a region model type that supports
 *          .set_catchment_calculation_filter(..)
 *          .revert_to_initial_state()
 *          .adjust_q(...)
 *          .run_cells(n_core,start_step,n_steps)
 *
 *
 *
 * \note that a side-effect of using this class is that the catchment-calculation-filter is set.
 *
 */
template <class RM>
struct adjust_state_model {
    RM &rm;///< Just a reference to the model, no copy semantics
	typedef  std::vector<typename RM::state_t>state_vector_t;
    std::vector<int> cids;///< selected catchments, empty means all.
	size_t i0{ 0u };
    adjust_state_model()=delete;
	state_vector_t  s0;///< the snap-shot of current state when starting the model
	void revert_to_state_0() { rm.set_states(s0); }
    /**Construct a wrapper around the RM so that we can use multiple tune to flow*/
    adjust_state_model( RM&rm,const std::vector<int> &cids,size_t i0=0
		) :rm(rm), cids(cids),i0(i0) {
        rm.set_catchment_calculation_filter(cids);// only calc for cells we are working on.
		rm.get_states(s0);// important: get the state 0 snap-shot from the model as it is now
    }

    /** calculate the model response discharge, given a specified scale-factor
     * relative the region model initial-state variable
     * \param q_scale the scale factor to be applied to the initial-state
     * \return discharge in m3/s for the first time-step, using the selected catchment-filters.
     */
    double discharge(double q_scale) {
		revert_to_state_0();
		rm.adjust_q(q_scale,cids);
		rm.run_cells(0,i0,1);// run one step
		double q_avg= cell_statistics::sum_catchment_feature_value(
                *rm.get_cells(),cids,
                [](const typename RM::cell_t&c) {return c.rc.avg_discharge; },
                i0
        );
        return q_avg;
    }
    
    /** \brief Adjust state to tune the flow of region-model and selected catchments to a target value.
     *
     * Adjust region-model state, using initial state as starting point,
     * so that the wanted flow is reached for the first time-step.
     *
     * When the function returns, the current state is adjusted, so
     * that starting the run will give the tune flow as result.
     *
     * region-model initial state is not changed during the process
     *
     * \param q_wanted flow in m3/s
     * \param scale_range (default 3.0) ratio based scaling is bounded between scale_0/scale_range .. scale_0*scale_range
     * \param scale_eps (default 0.001) terminate iteration when the scale-factor search-region is less than scale_0*scale_eps
     * \param max_iter  (default 300) terminate search for minimum if iterations exceeds this number
     * \return q_0,q_r, diagnostics , basically tuned flow in m3/s achieved
     */
    q_adjust_result tune_flow(double q_wanted, double scale_range=3.0,double scale_eps=1e-3,size_t max_iter=300) {
        q_adjust_result r;
        r.q_0= discharge(1.0);// starting out with scale=1.0, establish q_0
        double scale= q_wanted/r.q_0;// approximate guess for scale storage factor to start with
        try {
        dlib::find_min_single_variable(
                    [this,q_wanted](double x)->double{
                        double q_diff= this->discharge(x) - q_wanted;
                        return q_diff*q_diff;
                    },
                    scale,              // ref initial guess and result variable
                    scale/scale_range,  // scale minimum value default 1/3 of initial quess
                    scale*scale_range,  // scale max value, default 3x initial guess
                    scale*scale_eps,    // iteration stop when scale region is less than scale_eps
                    max_iter            // or we reach max-iterations
        );
        } catch( const exception &e) {
            r.diagnostics = string("failed to find solution within ")+to_string(max_iter)+string(", exception was:")+ e.what();
        }
        r.q_r =discharge(scale); // get back the resulting discharge after optimize
		revert_to_state_0();       // get back initial state
        rm.adjust_q(scale,cids);            // adjust it to the factor
        return r;
    }

};

}}
