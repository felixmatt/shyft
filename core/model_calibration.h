#include <dlib/optimization.h>
#include <dlib/statistics.h>
#include "cell_model.h"
#include "region_model.h"
#include "dream_optimizer.h"
#include "sceua_optimizer.h"
#pragma once

namespace shyft{
	namespace core {
		namespace model_calibration {
			/** \brief Minimize the functional defined by model using the BOBYQA method
			 *
			 * Solve the optimization problem defined by the model M using the Bounded Optimization by Quadratic Approximation method
			 * by Michael J. D. Powell. See the "The BOBYQA Algorithm for Bound Constrained Optimization Without Derivatives",
			 * Technical report, Department of Applied Mathematics and Theoretical Physics, University of Cambridge, 2009.
			 *
			 * We use the dlib implementation of this algorithm, see www.dlib.net/optimization.html#find_min_bobyqa.
			 *
			 * \tparam M Model to be optimized, implementing the interface:
			 *   - double operator()(const dlib::matrix<double, 0, 1> params) --> Evaluate the functional for parameter set params.
			 *   -vector<double> to_scaled(std::vector<double> params) --> Scale original parameters to [0,1]
			 *   -vector<double> from_scaled(const dlib::matrix<double, 0, 1> scaled_params) --> Unscale parameters
			 *  \param x --> Initial guess
			 *  \param max_n_evaluations --> stop/throw if not converging after n evaluations
			 *  \param tr_start --> Initial trust region radius
			 *  \param tr_stop --> Stopping trust region radius
			 */
			template  <class M>
			double min_bobyqa(M& model,vector<double>& x,int max_n_evaluations, double tr_start , double tr_stop ) {

				typedef dlib::matrix<double, 0, 1> column_vector;

				// Scale all parameter ranges to [0, 1] for better initial trust region radius balance.
				std::vector<double> x_s = model.to_scaled(x);

				column_vector _x = dlib::mat(x_s);

				// Scaled region min and max
				column_vector x_l(_x.nr()); x_l = 0.0;
				column_vector x_u(_x.nr()); x_u = 1.0;

				double res = find_min_bobyqa([&model](column_vector x) { return model(x); },
					_x,
					2 * _x.nr() + 1,    // Recommended number of interpolation points
					x_l,
					x_u,
					tr_start,   // initial trust region radius
					tr_stop,    // stopping trust region radius
					max_n_evaluations         // max number of objective function evaluations
					);

				// Convert back to real parameter range
				x = model.from_scaled(_x);
				return res;
			}
			///<Template class to transform model evaluation into something that dream can run
            template<class M>
            struct dream_fx : public shyft::core::optimizer::ifx {
                M& m;
                dream_fx(M & m) : m(m) {}
                double evaluate(const vector<double> &x) {
                    return -m(x); // notice that dream find maximumvalue, so we need to negate the goal function, effectively finding the minimum value.
                }
            };

            /** \brief template function that find the x that minimizes the evaluated value of model M using DREAM algorithm
             * \tparam M the model that need to have .to_scaled(x) and .from_scaled(x) to normalize the supplied parameters to range 0..1
             * \param model a reference to the model to be evaluated
             * \param x     the initial x parameters to use (actually not, dream is completely random driven), filled in with the optimal values on return
             * \param max_n_evaluations is the maximum number of iterations, currently not used, the routine returns when convergence is reached
             * \return the goal function of m value, corresponding to the found x-vector
             */
			template  <class M>
			double min_dream( M& model, vector<double>& x, int max_n_evaluations) {
				// Scale all parameter ranges to [0, 1]
				std::vector<double> x_s = model.to_scaled(x);
                dream_fx<M> fx_m(model);
                shyft::core::optimizer::dream dr;
				double res = dr.find_max(fx_m, x_s, max_n_evaluations);
				// Convert back to real parameter range
				x = model.from_scaled(x_s);
				return res;
			}

			///<Template class to transform model evaluation into something that dream can run
            template<class M>
            struct sceua_fx : public shyft::core::optimizer::ifx {
                M& m;
                sceua_fx(M & m) : m(m) {}
                double evaluate(const vector<double> &x) {
                    return m(x);
                }
            };

            /** \brief template for the function that finds the x that minimizes the evaluated value of model M using SCEUA algorithm
             * \tparam M the model, that provide .to_scaled(x) and .from_scaled(x) to normalize the parameters to 0..1 range
             * \param model a reference to the model evaluated
             * \param max_n_iterations stop after max_n_interations reached(keep best x until then)
             * \param x_eps stop when all x's changes less than x_eps(recall range 0..1), convergence in x
             * \param y_eps stop when last y-values (model goal functions) seems to have converged (no improvements)
             * \return the goal function of m value, and x is the corresponding parameter-set.
             * \throw runtime_error with text sceua: max_iterations reached before convergence
             */
            template <class M>
            double min_sceua(M& model, vector<double>& x, size_t max_n_evaluations, double x_eps=0.0001, double y_eps=0.0001) {
				// Scale all parameter ranges to [0, 1]
				vector<double> x_s = model.to_scaled(x);
				vector<double> x_min(x_s.size(), 0.0);///normalized range is 0..1 so is min..max
				vector<double> x_max(x_s.size(), 1.0);
				vector<double> x_epsv(x_s.size(),x_eps);
				// notice that there is some adapter code here, for now, just to keep
				// the sceua as close to origin as possible(it is fast&efficient, with stack-based mem.allocs)
				double *xv=__autoalloc__(double, x_s.size()); shyft::core::optimizer::fastcopy(xv, x_s.data(), x_s.size());
                sceua_fx<M> fx_m(model);
                shyft::core::optimizer::sceua opt;
                double y_result = 0;
                // optimize with no specific range for y-exit (max-less than min):
                auto opt_state = opt.find_min(x_s.size(), x_min.data(), x_max.data(), xv, y_result, fx_m, y_eps, -1.0, -2.0, x_epsv.data(), max_n_evaluations);
                for(size_t i=0; i < x_s.size(); ++i) x_s[i]=xv[i];//copy from raw vector
				// Convert back to real parameter range
				x = model.from_scaled(x_s);
				if( !(opt_state == shyft::core::optimizer::OptimizerState::FinishedFxConvergence || opt_state == shyft::core::optimizer::OptimizerState::FinishedXconvergence))
                    throw runtime_error("sceua: max-iterations reached before convergence"); //FinishedMaxIterations)
				return y_result;

            }

		/**\brief utility class to help transfrom time-series into suitable resolution
		*/
		struct ts_transform {
			/**\brief transform the supplied time-series, f(t) interpreted according to its point_interpretation() policy
			* into a new time-series,
			* that represents the true average for each of the n intervals of length dt, starting at start.
			* the result ts will have the policy is set to POINT_AVERAGE_VALUE
			* \note that the resulting ts is a fresh new ts, not connected to the source
			*/
			template < class TS, class TSS>
			shared_ptr<TS> to_average(utctime start, utctimespan dt, size_t n,const TSS& src) {
				shyft::timeseries::timeaxis time_axis(start, dt, n);
				shyft::timeseries::average_accessor< TSS, shyft::timeseries::timeaxis> avg(src,time_axis);
				auto r =make_shared< TS>(time_axis, 0.0);
				r->set_point_interpretation(shyft::timeseries::POINT_AVERAGE_VALUE);
				for (size_t i = 0; i < avg.size(); ++i) r->set(i, avg.value(i));
				return r;
			}
			template < class TS, class TSS>
			shared_ptr<TS> to_average(utctime start, utctimespan dt, size_t n, shared_ptr<TSS> src) {
				return to_average<TS, TSS>(start, dt, n, *src);
			}
		};
		/** \brief calc_type to provide simple start of more than NS critera, first extension is diff of sum 2 */
		enum target_spec_calc_type {
			NASH_SUTCLIFFE ,
			KLING_GUPTA, // ref. Gupta09, Journal of Hydrology 377(2009) 80-91
		};

		/** \brief property_type for target specification */
		enum catchment_property_type {
            DISCHARGE,
            SNOW_COVERED_AREA,
            SNOW_WATER_EQUIVALENT
		};

		/** \brief The target specification contains:
		* -# a target ts (the observed quantity)
		* -# a list of catchment ids (zero-based), that denotes the catchment.discharges that should equal the target ts.
		* -# a scale_factor that is used to construct the final goal function as
		*
		* goal_function = sum of all: scale_factor*(1 - nash-sutcliff factor) or KLING-GUPTA
		*
		* \tparam PS type of the target time-series, any type that is time-series compatible will work. Usually a point-based series.
		*/
		template<class PS>
		struct target_specification {
			typedef PS target_time_series_t;
			target_specification()
              : scale_factor(1.0), calc_mode(NASH_SUTCLIFFE), catchment_property(DISCHARGE), s_r(1.0), s_a(1.0), s_b(1.0) {}
#ifndef SWIG
			target_specification(const target_specification& c)
              : ts(c.ts), catchment_indexes(c.catchment_indexes), scale_factor(c.scale_factor),
                calc_mode(c.calc_mode), s_r(c.s_r), s_a(c.s_a), s_b(c.s_b) {}
			target_specification(target_specification&&c)
              : ts(std::move(c.ts)),
                catchment_indexes(std::move(c.catchment_indexes)),
                scale_factor(c.scale_factor), calc_mode(c.calc_mode),
                s_r(c.s_r), s_a(c.s_a), s_b(c.s_b) {}
			target_specification& operator=(target_specification&& c) {
				ts = std::move(c.ts);
				catchment_indexes = move(c.catchment_indexes);
				scale_factor = c.scale_factor;
				calc_mode = c.calc_mode;
				s_r = c.s_r; s_a = c.s_a; s_b = c.s_b;
				return *this;
			}
			target_specification& operator=(const target_specification& c) {
                if(this == &c) return *this;
                ts = c.ts;
                catchment_indexes = c.catchment_indexes;
                scale_factor = c.scale_factor;
                calc_mode = c.calc_mode;
				s_r = c.s_r; s_a = c.s_a; s_b = c.s_b;
                return *this;
			}
#endif
            /** \brief Constructs a target specification element for calibration, specifying all neede parameters
             *
             * \param ts; the target time-series that contain the target/observed discharge values
             * \param cids;  a vector of the catchment ids (zero-based) in the model that together should add up to the target time-series
             * \param scale_factor; the weight that this target_specification should have relative to the possible other target_specs.
             */
			target_specification(const target_time_series_t& ts, vector<int> cids, double scale_factor,
                                 target_spec_calc_type calc_mode = NASH_SUTCLIFFE, double s_r=1.0,
                                 double s_a=1.0, double s_b=1.0, catchment_property_type catchment_property_ = DISCHARGE)
              : ts(ts), catchment_indexes(cids), scale_factor(scale_factor),
                calc_mode(calc_mode), catchment_property(catchment_property_), s_r(s_r), s_a(s_a), s_b(s_b) {}
			target_time_series_t ts; ///< The target ts, - any type that is time-series compatible
			std::vector<int> catchment_indexes; ///< the catchment_indexes (zero based) that denotes the catchments in the model that together should match the target ts
			double scale_factor; ///<< the scale factor to be used when considering multiple target_specifications.
			target_spec_calc_type calc_mode;///< *NASH_SUTCLIFFE, KLING_GUPTA
			catchment_property_type catchment_property;///<  *DISCHARGE,SNOW_COVERED_AREA, SNOW_WATER_EQUIVALENT
			double s_r; ///< KG-scalefactor for correlation
			double s_a; ///< KG-scalefactor for alpha (variance)
			double s_b; ///< KG-scalefactor for beta (bias)
		};


        #ifndef SWIG
        ///< template helper classes to be used in enable_if_t in the optimizer for snow swe/sca:
        template< bool B, class T = void >
        using enable_if_tx = typename enable_if<B,T>::type;

            #pragma GCC diagnostic push
            #pragma GCC diagnostic ignored "-Wunused-value"
            template<class T,class=void>            // we only want compute_sca_sum IF the response-collector do have snow_sca attribute
            struct detect_snow_sca:false_type{};    // detect_snow_sca, default it to false,

            template<class T>                       // then specialize it for T that have snow_sca
            struct detect_snow_sca<T,decltype(T::snow_sca,void())>:true_type{}; // to true.

            template<class T,class=void> // ref similar pattern for template specific generation of snow_sca above.
            struct detect_snow_swe:false_type{};

            template<class T>
            struct detect_snow_swe<T,decltype(T::snow_swe,void())>:true_type{};
            #pragma GCC diagnostic pop
            /** when doing catchment area related optimization, e.g. snow sca/swe
             *  we need to
             *  keep track of the area of each catchment so that we get
             *  true average values
             */
            struct area_ts {
                double area;///< area in m^2
                pts_t ts; ///< ts representing the property for the area, e.g. sca, swe
                area_ts(double area_m2,pts_t ts):area(area_m2),ts(move(ts)){}
                area_ts():area(0.0) {}                              // maybe we could drop this and rely on defaults ?
                area_ts(area_ts&&c):area(c.area),ts(move(ts)) {}
                area_ts(const area_ts&c):area(c.area),ts(c.ts) {}
                area_ts& operator=(const area_ts&c ) {
                    if(&c != this) {
                        area=c.area;
                        ts=c.ts;
                    }
                    return *this;
                }
                area_ts& operator=(area_ts && c) {
                    if(&c != this) {
                        ts=move(c.ts);
                        area=c.area;
                    }
                    return *this;
                }
            };

        #endif // SWIG

		/** \brief The optimizer for parameters in a \ref shyft::core::region_model
		 * provides needed functionality to orchestrate a search for the optimal parameters so that the goal function
		 * specified by the target_specifications are minmized.
		 * The user can specify which parameters (model specific) to optimize, giving range min..max for each of the
		 * parameters. Only parameters with min != max are used, thus minimizing the parameter space.
		 *
		 * Target specification \ref target_specification allows a lot of flexiblity when it comes to what
		 * goes into the \ref nash_sutcliffe goal function.
		 *
		 * The search for optimium starts with the current parameter-set, the current start state, over the specified model time-axis.
		 * After a run, the goal function is calculated and returned back to the minbobyqa algorithm that continue searching for the minimum
		 * value until tolerances/iterations area reached.
		 *
		 * \tparam M the region model, and we use that supports the following member:
		 *  -# .time_axis, a time-axis value and type used by the model, the optimizer needs this during goal function evaluation.
		 *  -# .run(), the region model from  state (s0) over .time_axis, using current values of parameters.
		 *
		 * \tparam PA a template parameter for the parameters, supporting vector form access
		 * \tparam PS type of the target time-series, inside. the target specification..(SiH: can we turn it the other way around ??)
		 *
		 */
        template< class M, class PA, class PS>
        class optimizer {
          public:
            typedef dlib::matrix<double, 0, 1> column_vector; ///< dlib optimizer enjoys dlib matrix types.
            typedef PS target_time_series_t; ///< target_time_series_t. could that be of same type as model calc, always, or is it supplied from outside?

			typedef target_specification<PS> target_specification_t;///< describes how to calculate the goal function, maybe better 'retemplate' on this
			typedef M region_model_t; ///< obvious and Ok template, the region_model, and there is just a few things that we need from this template.
            typedef typename M::state_t state_t;
            typedef typename M::parameter_t parameter_t;
            typedef typename M::cell_t cell_t;
            typedef typename cell_t::response_collector_t response_collector_t;
          private:
#ifndef SWIG
		public:
            PA& parameter_accessor; ///<  a *reference* to the model parameters in the target  model, all cells share this!
            region_model_t& model; ///< a reference to the region model that we optimize
		private:
            vector<target_specification_t> targets; ///<  list of targets ts& catchments indexes to be optimized, used to calculate goal function
			// internal parameter vectors
			vector<double> p_expanded;
            vector<double> p_min; // min==max, no- optimization..
            vector<double> p_max;
            vector<state_t> initial_state;
			int print_progress_level;
			size_t n_catchments;///< optimized counted number of model.catchments available
			//Need to handle expanded/reduced parameter vector based on min..max range to optimize speed for bobyqa
			bool is_active_parameter(size_t i) const { return fabs(p_max[i] - p_min[i]) > 0.000001; }
			vector<double> reduce_p_vector(const vector<double>& fp) const {
				std::vector<double> r; r.reserve(fp.size());
				for (size_t i = 0; i < fp.size(); ++i) {
					if (is_active_parameter(i))
						r.push_back(fp[i]);// only pick values that are active in optimization
				}
				return r;
			}
			vector<double> expand_p_vector(const vector<double>& rp) const {
				std::vector<double> r; r.reserve(p_expanded.size());
				size_t j = 0;
				for (size_t i = 0; i < p_expanded.size(); ++i) {
					if (is_active_parameter(i))
						r.push_back(rp[j++]);// pick from reduced vector
					else
						r.push_back(p_expanded[i]); // just use already set class global parameter
				}
				return r;
			}

#endif
		public:
			/**\brief construct an opt model for ptgsk, use p_min=p_max to disable optimization for a parameter
			* \param model reference to the model to be optimized, the model should be initialized, i.e. the interpolation step done.
			* \param vector<target_specification_t> specifies how to calculate the goal-function, \ref shyft::core::model_calibration::target_specification
			* \param p_min minimum values for the parameters to be optimized
			* \param p_max maximum values for the parameters to be  optimized
			*/
			optimizer(region_model_t& model, const vector<target_specification_t>& targetsA,
                      const vector<double>& p_min, const vector<double>& p_max)
              : parameter_accessor(model.get_region_parameter()),
                model(model),
                targets(targetsA),
                p_min(p_min),
				p_max(p_max), print_progress_level(0) {
                // 0. figure out n_catchments, asking the model
                n_catchments=model.number_of_catchments();
				// 1. figure out the catchment indexes to evaluate.
				vector<int> catchment_indexes;
				for (const auto&t : targets) {
					catchment_indexes.insert(catchment_indexes.end(), begin(t.catchment_indexes), end(t.catchment_indexes));
				}
				if (catchment_indexes.size() > 1) {
					sort(begin(catchment_indexes), end(catchment_indexes));
					auto unique_end = unique(begin(catchment_indexes), end(catchment_indexes));
					catchment_indexes.resize(distance(begin(catchment_indexes), unique_end));
				}
                for (auto i: catchment_indexes) {
                    if (model.has_catchment_parameter(i))
                        throw runtime_error("Cannot calibrate on local parameters.");
                }
				model.set_catchment_calculation_filter(catchment_indexes); //Only calculate the catchments that we optimize
				// 2. fetch the initial state (s0) from the supplied model, and store it so that we start each run with same state s0
                auto cells = model.get_cells();
                initial_state.reserve((*cells).size());
                for(const auto& cell:*cells) initial_state.emplace_back(cell.state);
           }

            state_t get_initial_state(size_t idx) {
                return initial_state[idx];
            }

			/**\brief Call to optimize model, starting with p parameter set, using p_min..p_max as boundaries.
			 * where p is the full parameter vector.
			 * the p_min,p_max specified in constructor is used to reduce the parameterspace for the optimizer
			 * down to a minimum number to facilitate fast run.
			 * \param p contains the starting point for the parameters
			 * \param max_n_evaluations stop after n calls of the objective functions, i.e. simulations.
			 * \param tr_start is the trust region start , default 0.1, ref bobyqa
			 * \param tr_stop is the trust region stop, default 1e-5, ref bobyqa
			 * \return the optimized parameter vector
			 */
           vector<double> optimize(vector<double> p, size_t max_n_evaluations=1500, double tr_start=0.1, double tr_stop=1.0e-5) {
				// reduce using min..max the parameter space,
				p_expanded = p;//put all parameters into class scope so that we can reduce/expand as needed during optimization
				auto rp = reduce_p_vector(p);
				min_bobyqa(*this, rp, max_n_evaluations, tr_start, tr_stop);
				// expand,put inplace p to return vector.
				p = expand_p_vector(rp);
                return p;
            }

			/**\brief Call to optimize model, using DREAM alg., find p, using p_min..p_max as boundaries.
			 * where p is the full parameter vector.
			 * the p_min,p_max specified in constructor is used to reduce the parameterspace for the optimizer
			 * down to a minimum number to facilitate fast run.
			 * \param p is used as start point (not really, DREAM use random, but we should be able to pass u and q....
			 * \param max_n_evaluations stop after n calls of the objective functions, i.e. simulations.
			 * \return the optimized parameter vector
			 */
            vector<double> optimize_dream(vector<double> p,size_t max_n_evaluations=1500) {
				// reduce using min..max the parameter space,
				p_expanded = p;//put all parameters into class scope so that we can reduce/expand as needed during optimization
				auto rp = reduce_p_vector(p);
				min_dream<optimizer>(*this, rp, max_n_evaluations);
				p = expand_p_vector(rp);// expand,put inplace p to return vector.
                return p;
            }

			/**\brief Call to optimize model, using SCE UA, using p as startpoint, find p, using p_min..p_max as boundaries.
			 * where p is the full parameter vector.
			 * the p_min,p_max specified in constructor is used to reduce the parameterspace for the optimizer
			 * down to a minimum number to facilitate fast run.
			 * \param p is used as start point and is updated with the found optimal points
			 * \param max_n_evaluations stop after n calls of the objective functions, i.e. simulations.
			 * \param x_eps is stop condition when all changes in x's are within this range
			 * \param y_eps is stop condition, and search is stopped when goal function does not improve anymore within this range
			 * \return the optimized parameter vector
			 */
            vector<double> optimize_sceua(vector<double> p,size_t max_n_evaluations=1500, double x_eps=0.0001, double y_eps=1.0e-5) {
				// reduce using min..max the parameter space,
				p_expanded = p;//put all parameters into class scope so that we can reduce/expand as needed during optimization
				auto rp = reduce_p_vector(p);
				min_sceua(*this, rp, max_n_evaluations, x_eps, y_eps);
				p = expand_p_vector(rp);				// expand,put inplace p to return vector.
                return p;
            }


			/**\brief reset the state of the model to the initial state before starting the run/optimize*/
            void reset_states() {
                auto cells = model.get_cells();
                size_t i = 0;
                for_each(begin(*cells), end(*cells), [this, &i] (cell_t& cell) { cell.set_state(initial_state[i++]); });
            }
			/**\brief set the parameter ranges, set min=max=wanted parameter value for those not subject to change during optimization */
			void set_parameter_ranges(vector<double> p_min, vector<double> p_max) { this->p_min = p_min; this->p_max = p_max; }
			void set_verbose_level(int level) { print_progress_level = level; }
			/**\brief calculate the goal_function as used by minbobyqa,
			 *   using the full set of  parameters vectors (as passed to optimize())
			 *   and also ensures that the shyft state/cell/catchment result is consistent
			 *   with the passed parameters passed
			 * \param full_vector_of_parameters contains all parameters that will be applied to the run.
			 *  \returns the goal-function, weigthed nash_sutcliffe sum
			 */
			double calculate_goal_function(std::vector<double> full_vector_of_parameters) {
				p_expanded = full_vector_of_parameters;// ensure all parameters are  according to full_vector..
				return run(reduce_p_vector(full_vector_of_parameters));// then run with parameters as if from optimize
			}
#ifndef SWIG
            friend class calibration_test;// to enable testing of individual methods
			/** called by bobyqua: */
            double operator() (const column_vector& p_s) { return run(from_scaled(p_s)); }
            double operator() (const vector<double>&p_s) { return run(from_scaled(p_s)); }

			/** called by bobyqua:reduced parameter space p */
           vector<double> to_scaled(const vector<double>& rp) const {
                if (p_min.size() == 0) throw runtime_error("Parameter ranges are not set");
                vector<double> p_s;
                auto rp_min = reduce_p_vector(p_min);
                auto rp_max = reduce_p_vector(p_max);
                const size_t n_params = rp.size();
                p_s.reserve(n_params);
                for (size_t i = 0; i < n_params; ++i)
                    p_s.emplace_back((rp[i] - rp_min[i])/(rp_max[i] - rp_min[i]));
                return p_s;
            }
			/** called by bobyqua: reduced parameter space p */
           vector<double> from_scaled(column_vector p_s) const {
                if (p_min.size() == 0) throw runtime_error("Parameter ranges are not set");
                vector<double> p;
                auto rp_min = reduce_p_vector(p_min);
                auto rp_max = reduce_p_vector(p_max);
                p.reserve(p_s.nr());
                for (int i = 0; i < p_s.nr(); ++i)
                    p.emplace_back((rp_max[i] - rp_min[i])*p_s(i) + rp_min[i]);
                return p;
            }
			/** called by bobyqua: reduced parameter space p */
           vector<double> from_scaled(vector<double> p_s) const {
                if (p_min.size() == 0) throw runtime_error("Parameter ranges are not set");
                vector<double> p;
                auto rp_min = reduce_p_vector(p_min);
                auto rp_max = reduce_p_vector(p_max);
                p.reserve(p_s.size());
                for (size_t i = 0; i < p_s.size(); ++i)
                    p.emplace_back((rp_max[i] - rp_min[i])*p_s[i] + rp_min[i]);
                return p;
            }
		  private:

            pts_t compute_discharge_sum(const target_specification_t& t,vector<pts_t>& catchment_d) const {
                if(catchment_d.empty())
                    model.catchment_discharges(catchment_d);
                pts_t discharge_sum(model.time_axis,0.0,shyft::timeseries::POINT_AVERAGE_VALUE);
                for(auto i : t.catchment_indexes)
                    discharge_sum.add(catchment_d[i]);
                return discharge_sum;
            }

            /** \brief extracts vector of area_ts for all calculated catchments using the
             * given property function tsf that should have signature pts_t (const cell& c)
             * \note that this function sum together contributions at cell-level.
             * TODO: Avoid duplicate code, - use average_catchment_feature(*model.cells, catchment_index, []() return c.rc.snow_sca) but it returns a shared_ptr..
             */
            template<class property_ts_function>
            vector<area_ts> extract_area_ts_property( property_ts_function && tsf) const {
                vector<area_ts> r(n_catchments,area_ts(0.0,pts_t(model.time_axis,0.0,shyft::timeseries::POINT_AVERAGE_VALUE)));
                for(const auto& c: *model.get_cells()) {
                    if (model.is_calculated(c.geo.catchment_id())){
                        r[c.geo.catchment_id()].ts.add_scale(tsf(c),c.geo.area());//the only ref. to snow_sca
                        r[c.geo.catchment_id()].area += c.geo.area(); //using entire cell geo area for now.
                    }
                }
                for(size_t i=0;i<n_catchments;++i)
                    if(model.is_calculated(i))
                        r[i].ts.scale_by(1/r[i].area);
                return r;
            }
            /** \brief returns the area weighted sum of vector<area_ts> according to t.catchment_indexes
            */
            pts_t compute_weighted_area_ts_average(const target_specification_t& t, const vector<area_ts>& ats) const {
                pts_t ts_sum(model.time_axis,0.0,shyft::timeseries::POINT_AVERAGE_VALUE);
                double a_sum=0.0;
                for(auto i:t.catchment_indexes) {
                    ts_sum.add_scale(ats[i].ts,ats[i].area);
                    a_sum += ats[i].area;
                }
                ts_sum.scale_by(1/a_sum);
                return ts_sum;
            }


            template<class rc_t = response_collector_t> // finally,
              enable_if_tx<detect_snow_sca<rc_t>::value,pts_t> // use enable_if_t  detect_snow_sca to enable this type
            compute_sca_sum(const target_specification_t& t,vector<area_ts>& catchment_sca) const {
                if(catchment_sca.empty())
                    catchment_sca=extract_area_ts_property([](const cell_t&c) {return c.rc.snow_sca;});
                return compute_weighted_area_ts_average(t,catchment_sca);
            }

            template<class rc_t = response_collector_t>
              enable_if_tx<!detect_snow_sca<rc_t>::value,pts_t>
            compute_sca_sum(const target_specification_t& t,vector<area_ts>& catchment_d) const {
                // To support dynamic typing and python: If a cell.rc do not have snow_sca, but we pass in a criteria/target
                // function that do specify snow_sca, we throw a runtime error.
                // TODO: verify this in the constructor and throw as early as possible
                throw runtime_error("resource collector doesn't have snow_sca");
            }



            template<class rc_t = response_collector_t>
            enable_if_tx<detect_snow_swe<rc_t>::value,pts_t> compute_swe_sum(const target_specification_t& t,vector<area_ts>& catchment_swe) const {
                if(catchment_swe.empty())
                    catchment_swe=extract_area_ts_property([](const cell_t&c){return c.rc.snow_swe;});
                return compute_weighted_area_ts_average(t,catchment_swe);
            }
            template<class rc_t = response_collector_t>
            enable_if_tx<!detect_snow_swe<rc_t>::value,pts_t> compute_swe_sum(const target_specification_t& t,vector<area_ts>& catchment_d) const {
                throw runtime_error("resource collector doesn't have snow_swe");
            }

			/**\brief from operator(), called by min_bobyqa, for each iteration, so p is bobyqa parameter vector
			* notice that the function returns the value of the goal function,
			* as specified by the target specification. The flexibility is rather large:
			*  It's a set of target-specifications, weighted, where each of them can
			*  be a KG or NS, for a specified period/resolution.
			*  E.g. we can specify targets that apply to specific catchments, and specified periods/resolutions.
			*/
			double run(const vector<double>& rp) {
				auto p = expand_p_vector(rp);// expand to full vector, then:
				parameter_accessor.set(p); // Sets global parameters, all cells share a common pointer.
				reset_states();
				model.run_cells();
				double goal_function_value = 0.0;// overall goal-function, intially zero
				double scale_factor_sum = 0.0; // each target-spec have a weight, -use this to sum up weights
				vector<pts_t> catchment_d;
				vector<area_ts> catchment_sca,catchment_swe;// "catchment" level simulated discharge,sca,swe
				for (const auto& t : targets) {
					shyft::timeseries::direct_accessor<pts_t, timeaxis_t> target_accessor(t.ts, t.ts.get_time_axis());
					pts_t property_sum;
                    switch(t.catchment_property){
                    case DISCHARGE:
                        property_sum=compute_discharge_sum(t,catchment_d);
                        break;
                    case SNOW_COVERED_AREA:
                        property_sum=compute_sca_sum(t,catchment_sca);
                        break;
                    case SNOW_WATER_EQUIVALENT:
                        property_sum=compute_swe_sum(t,catchment_swe);
                        break;
                    }
					shyft::timeseries::average_accessor<pts_t, timeaxis_t> property_sum_accessor(property_sum, t.ts.get_time_axis());

					/*pts_ts
                        discharge_sum(model.time_axis, 0, shyft::timeseries::POINT_AVERAGE_VALUE);

					// now calculate the discharge_sum for the target ts resolution
*/
					if (t.calc_mode == target_spec_calc_type::NASH_SUTCLIFFE) {
						double partial_nash_sutcliffe_gf = nash_sutcliffe_goal_function(target_accessor, property_sum_accessor);
						goal_function_value += partial_nash_sutcliffe_gf* t.scale_factor;// add scaled contribution from each target
					} else {
						// ref. KLING-GUPTA Journal of Hydrology 377 (2009) 80–91, page 83, formula (10):
                        // a=alpha, b=betha, q =sigma, u=my, s=simulated, o=observed
                        double EDs = kling_gupta_goal_function<dlib::running_scalar_covariance<double>>(target_accessor,
                                                                                                        property_sum_accessor,
                                                                                                        t.s_r,
                                                                                                        t.s_a,
                                                                                                        t.s_b);
                        goal_function_value += t.scale_factor*EDs;
					}
					scale_factor_sum += t.scale_factor;
				}
				goal_function_value /= scale_factor_sum;
				if (print_progress_level > 0) {
					std::cout << "ParameterVector(";
					for (size_t i = 0; i < parameter_accessor.size(); ++i) {
						std::cout << parameter_accessor.get(i);
						if (i < parameter_accessor.size() - 1)cout << ", ";
					}
					std::cout << ") = " << goal_function_value << " (NS or KG)" <<endl;
				}
				return goal_function_value;
			}
#endif
        };


		} // model_calibration
	} // core
} // shyft
