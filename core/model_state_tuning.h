#pragma once
#include <vector>
#include <dlib/optimization.h>
#include "cell_model.h"

namespace shyft {
namespace core {

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
    std::vector<int> cids;///< selected catchments, empty means all.
    adjust_state_model()=delete;

    /**Construct a wrapper around the RM so that we can use multiple tune to flow*/
    adjust_state_model( RM&rm,const std::vector<int> &cids
    ) :rm(rm), cids(cids) {
        rm.set_catchment_calculation_filter(cids);
    }

    /** calculate the model response discharge, given a specified scale-factor
     * relative the region model initial-state variable
     * \param q_scale the scale factor to be applied to the initial-state
     * \return discharge in m3/s for the first time-step, using the selected catchment-filters.
     */
    double discharge(double q_scale) {
		rm.revert_to_initial_state();
		rm.adjust_q(q_scale,cids);
		rm.run_cells(0,0,1);// run one step
		double q_avg= cell_statistics::sum_catchment_feature_value(
                *rm.get_cells(),cids,
                [](const typename RM::cell_t&c) {return c.rc.avg_discharge; },
                0
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
     * \return tuned flow in m3/s achieved
     */
    double tune_flow(double q_wanted, double scale_range=3.0,double scale_eps=1e-3,size_t max_iter=300) {
        double q_0= discharge(1.0);// starting out with scale=1.0, establish q_0
        double scale= q_wanted/q_0;// approximate guess for scale storage factor to start with
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
        double q_result =discharge(scale); // get back the resulting discharge after optimize
        rm.revert_to_initial_state();       // get back initial state
        rm.adjust_q(scale,cids);            // adjust it to the factor
        return q_result;
    }

};

}
}
