#include "core_pch.h"
#include "sceua_optimizer.h"

using namespace std;
namespace shyft {
    namespace core {
        namespace optimizer {

           OptimizerState
            sceua::find_min(
                const size_t n,			// Number of active parameters
                const double x_min[],	// Lower limit of all n parameters
                const double x_max[],	// Upper limit of all n parameters
                double x[],				// x_min < x < x_max. The [input]initial/[output]current/optimal n parameter values
                double& fx_minimum_found,	// The optimal value found
                ifx& fx,// fx(x1..xn) The function that takes x[n] parameters
                // Stop criteria goes here: They are important, since evaluating fx takes time.
                double fx_epsilon,		// 1. Stop when diff of 5 last samples: 2x(Fmax-Fmin)/(Fmax+Fmin)<fx_epsilon
                double fx_solution_min,	// 2. Stop when fx is within fx_solution_min..fx_solution_max
                double fx_solution_max,	//   (to disable, set fx_solution_min > fx_solution_max)
                const double x_epsilon[],// 3. Stop when all x[] are just moving within x_epsilon range
                size_t max_iterations	// 4. Stop when maxiterations/invocations are reached
                )
                const
            {
                const double eps=1e-10;
                size_t i,j,jj,ij,evaluations=0;
                OptimizerState optimizerState=Searching;
                const size_t m		=	2*n+1;	// The number of points in each complex
                const size_t p		=	5;		// The number of complexes
                const size_t n_live_points	=	p*m;	// The number of "live" points in the parameter space

                double **ax		= __autoalloc__(double*,m);
                double **sample = __autoalloc__(double*,n_live_points);
                double **ssample= __autoalloc__(double*,n_live_points);

                for (i=0; i<m; i++)
                    ax[i]=new double[n];
                for (i=0;i<n_live_points;i++) {// While the x table contains the full parameter vector,
                    sample[i] =__autoalloc__(double,n);	// sample does not contain any fixed parameters.
                    ssample[i]=__autoalloc__(double,n);	// Neither does ssample.
                }
                size_t*		ind=__autoalloc__(size_t,n_live_points);
                double*		af =__autoalloc__(double,m);
                double*		f  =__autoalloc__(double,n_live_points);
                double*		sf =__autoalloc__(double,n_live_points);
                bool terminateRequested=false;
                //  Step 1:	Draw a random sample of parameters and evaluate objective function value for each point
                fastcopy(sample[0],x,n); // first sample is initial values
                f[0] = fx.evaluate(n,x);++evaluations;
                for (i=1;i<n_live_points;i++) {
                    random_generate_x(n,x,x_min,x_max);
                    fastcopy(sample[i],x,n);
                    f[i]= fx.evaluate(n,x);++evaluations;
                }
                construct_sorted_pivot_table(f,ind,n_live_points);//	Step 2:	Sort the points according to value of objective function
                for (i=0;i<n_live_points;i++) {
                    sf[i]=f[ind[i]];
                    fastcopy(ssample[i],sample[ind[i]],n);
                }

                //**************** Start the optimisation *********************
                while(optimizerState==Searching && !terminateRequested) {// This loop performes one "shuffling", that is, sorting the m*p points and distributing them among complexes
                    for (ij=0; ij<p; ij++) {//	Step 3:	Partion the sample into p complexes of m points
                        for (j=0; j<m; j++) {
                            jj=(j)*p + ij;
                            fastcopy(ax[j],ssample[jj],n);
                            af[j]=sf[jj];
                        }
                        //	Step 4: Evolve each complex
                        evolve(ax,af,m,n,fx,x_min,x_max,x,evaluations,ij);
                        //	Step 5:	Replace the evolved complexes
                        for (j=0; j<m; j++) {
                            jj=(j)*p+ij;
                            fastcopy(sample[jj],ax[j],n);
                            f[jj]=af[j];
                        }
                    }
                    construct_sorted_pivot_table(f,ind,n_live_points);// Sort the points according to value of objective function
                    for (i=0;i<n_live_points;i++) {
                        sf[i]=f[ind[i]];
                        fastcopy(ssample[i],sample[ind[i]],n);
                    }
                    fastcopy(x,sample[0],n);
                    fx_minimum_found=sf[0];
                    //	Step 6:	 Evaluate convergence, or stop criteria.
                    if(fx_solution_min <= fx_minimum_found && fx_minimum_found < fx_solution_max) {
                        optimizerState=FinishedFxConvergence;
                    } else if (2.0 * fabs(sf[0]-sf[n_live_points-1]) / (fabs(sf[0])+fabs(sf[n_live_points-1])+eps) < fx_epsilon) {
                        optimizerState=FinishedFxConvergence;
                    } else {
                        size_t nFrozen=0;
                        for (i=0;i<n;i++) {
                            if(fabs(ssample[0][i]-ssample[n_live_points-1][i]) < x_epsilon[i])
                                ++nFrozen;
                        }
                        if(nFrozen==n)
                            optimizerState=FinishedXconvergence;// all params within very small range..
                        if(evaluations > max_iterations) {
                            optimizerState=FinishedMaxIterations;  //   file1 << "Maximum evaluations " <<evaluations << "  " << max_iterations << "/n";  */
                        }
                    }
                }
                if(terminateRequested)
                    optimizerState=FinishedUserRequest;
                return optimizerState;		// The reason for terminating sceua.
            }
            ///	Function evolve corresponds to the competitive complex ecolution (CCE),
            /// fig. 2 in Duan et. al (1993). It is called from sceua.
            void
            sceua::evolve(double *ax[],	// ax[m][n]  are the parameter values of complex to be evovled.
                double af[],						// af[m]     is the objective function value of the subcomplex.
                size_t m,							// m         is the n_live_points of the complex.
                size_t n,							// n         is number of parameters to be optimized.
                ifx& fn,			// The function (hydrological model) to be evaluated
                const double x_min[],				// x_min[n]   are the minimum values of the parameters to be optimized.
                const double x_max[],				// maxi[n]   are the maximum values of the parameters to be optimized.
                double x[],							// x[n]  is all the parameter vector required by the model.
                                                    // The actual length of this vector is not required in this subroutine.
                size_t& evaluations,				// evaluations is a counter for how many times the model is called.
                size_t complexno					// For diagnostics: The number of this complex.
                ) const
            {
                size_t		i, j, ii, nsel;
                double	ff, objf;
                size_t		kk, sel, mutation;

                int *selected	= __autoalloc__(int,m);
                double *gg		= __autoalloc__(double,n);		// The center of grativity for the q-1 best points in bf.
                double *pp		= __autoalloc__(double,m);		// The probability density of ax
                double *cp		= __autoalloc__(double,m);		// The probability function for ax used to elect points to bf.


                double **sax = __autoalloc__(double*,m);		// ax sorted according to af
                for (i=0; i<m; i++)
                    sax[i]=__autoalloc__(double,n);
                // STEP 0: initialization of optimization parameters.
                size_t q				= n+1;					// The n_live_points of a subcomplex
                size_t alfa				= 1;					// The values of q, alfa and beta are selected according
                size_t beta				= 2 * n + 1;			// to recommandations in ????? */

                size_t* inda=__autoalloc__(size_t,m)	;		// The index array sorting bf in increasing order
                double* saf =__autoalloc__(double,m);

                size_t* indb=__autoalloc__(size_t,q);			// The index array sorting af in increasing order
                double* sbf =__autoalloc__(double,q);
                double* bf  =__autoalloc__(double,q);			// The function values for the selected complex

                size_t*  ll	= __autoalloc__(size_t,q);			// The location in ax to which bx belongs
                size_t* sll	= __autoalloc__(size_t,q);			// ll sorted according to bf.
                double **bx	= __autoalloc__(double*,q);			// The parameter values for the selected sub-complex
                double **sbx= __autoalloc__(double*,q);			// bx sorted according to bf

                for (i=0; i<q; i++) {
                    bx[i]	= __autoalloc__(double,n);
                    sbx[i]	= __autoalloc__(double,n);
                }

                for (i=0; i<m; i++){					// STEP 1 : Calculate the probability distribution for the points
                    // The first point (with the lowest function value)
                    pp[i]	= (2.0*(m+1.0-(i+1.0)))		// has the highest probability.
                        /	(m*(m+1.0));				// pp is the probability density,
                    if (i > 0)	cp[i]	= cp[i-1]+pp[i];// cp is the cumulative probability distribution
                    else				cp[i]	= pp[i];
                }
                // Step 5 Iterate: Repeat step 2 through 4 beta times
                for (kk=0; kk<beta; kk++) {
                    // Step 2 :		Create the subcomplex bf; bx fom af; ax by
                    nsel=0;					// selecting q of m points from af;ax according to the
                    for (i=0; i<m; i++)		// distribution specified above. The index for the
                        selected[i]=0;		// location in the original array af;ax is stored in ll.*/

                    while (nsel < q) {
                        ff = random01();	// Formerly:		ff = OptUtil::ran1(idum) ;, which is now called from unif01.
                        i = sel = 0 ;
                        while (sel == 0 && i < m) {	// Continue until a new point is selected
                            // or the n_live_points of af;ax is reached
                            if (ff <= cp[i]) {
                                if (selected[i] == 0) {	// Check whether the point already has been selected
                                    fastcopy(bx[nsel],ax[i],n);
                                    bf[nsel]	= af[i];
                                    ll[nsel]	= i;
                                    selected[i]	= 1;
                                    sel			= 1;
                                    nsel++;
                                }
                            }
                            i++;
                        }
                    }

                    for (ii=0; ii<alfa; ii++) {		// step 3f : Repeat steps 3a trough 3e alfa times
                        construct_sorted_pivot_table(bf,indb,q);		// Step 3a: Sort the selected points.
                        for(i=0; i<q; i++) {		// indb is an index array that sorts bf in increasing order,
                            // sbx is the sorted sub-complex, sll is the sortet index array
                            fastcopy(sbx[i],bx[indb[i]],n);
                            sll[i] = ll[indb[i]];
                            sbf[i] = bf[indb[i]];
                        }

                        for (i=0 ;i<n ;i++)			// Step 3a continues + step 3b:  Compute the centroid
                            gg[i] = 0.0;			// of the q-1 best points in bf;bx
                        mutation = 0;				// gg is the ceontroide, newp is the new parameter vector

                        for (i=0; i<n; i++) {
                            for (j=0; j<q-1; j++)
                                gg[i] = gg[i] + sbx[j][i]/(q-1);
                            x[i]= 2.0*gg[i] - sbx[q-1][i]; // here we might break integral rules for x[]

                            if(x[i]<x_min[i] || x[i]>x_max[i])	// Step 3c: Check if new point is within inital parameter space
                                mutation=1;							// If outside, select the new point by a mutation step
                            // mutate is the subroutine that performs the mutation step
                        }

                        if (mutation == 1) {
                            mutate(ax,x, m,n);
                        }
                        objf = fn.evaluate(n, x);++evaluations;	// model(x,objf);

                        if(objf < bf[q-1]) {			// Step 3d : Check whether step 3c step gives a better objective function.
                            ;// If this is the case, continue to step 3f,
                            //	( go to the end of the alfa-loop )
                        } else {// otherwise a contraction step is performed.
                            for(i=0;i<n;i++) {
                                x[i]= (gg[i]+sbx[q-1][i]) / 2.0; // here we might break rules for the x[] integrity
                            }
                            objf = fn.evaluate(n,x);++evaluations;
                            if ( objf < bf[q-1] ) {	// Step 3e: Check whether the contraction step gives a better objective function
                                ;// Step 3d OK
                            } else {				// Mutation step
                                mutate(ax, x, m, n);
                                objf = fn.evaluate(n, x);++evaluations;
                            }
                        }	// End contraction step
                        sbf[q-1]=objf;//store result generated in above if-else-if-else
                        fastcopy(sbx[q-1],x,n);

                        fastcopy(ll,sll,q);
                        fastcopy(bf,sbf,q);
                        for(i=0;i<q;i++) {		// Book-keeping
                            fastcopy(bx[i],sbx[i],n);
                        }
                    }   /*alfa  loop */

                    for(i=0;i<q;i++) {			    // Step 4 :  Replace bx into ax (and bf into af)
                        fastcopy(ax[ll[i]],bx[i],n);// using the original locations stored in ll
                        af[ll[i]] = bf[i];	 	    // Sorts then the parent complex af  and ax.
                    }
                    construct_sorted_pivot_table(af,inda,m);// sort the 'parent complex' af and ax
                    for (i=0; i<m; i++) {
                        saf[i] = af[inda[i]];
                        fastcopy(sax[i],ax[inda[i]],n);
                    }
                    fastcopy(af,saf,m);
                    for (i=0; i<m; i++) {
                        fastcopy(ax[i],sax[i],n);
                    }
                }   /*  beta loop */
            }

            void
            sceua::mutate(double *x_alternatives[], double x_new[], int na, int nprm) const {
                double *x_min = __autoalloc__(double,nprm);
                double *x_max = __autoalloc__(double,nprm);
                fastcopy(x_min,x_alternatives[0],nprm);
                fastcopy(x_max,x_alternatives[0],nprm);
                for (int i=1;i<na;i++) {
                    for (int j=1;j<nprm;j++) {
                        if (x_alternatives[i][j] < x_min[j])
                            x_min[j] = x_alternatives[i][j];
                        if (x_alternatives[i][j] > x_max[j])
                            x_max[j] = x_alternatives[i][j];
                    }
                }
                random_generate_x(nprm, x_new, x_min, x_max);
            }

            void
            sceua::random_generate_x(size_t n, double x_new[], const double x_min[], const double x_max[]) const {
                for (size_t i=0; i<n; ++i)
                    x_new[i] = x_min[i] + random01()*(x_max[i]-x_min[i]);
            }
        }
    }
}
