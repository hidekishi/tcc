/* ***********************************************************************
  This program is part of the
        OpenMP Source Code Repository

        http://www.pcg.ull.es/ompscr/
        e-mail: ompscr@etsii.ull.es

   Copyright (c) 2004, OmpSCR Group
   All rights reserved.

   Redistribution and use in source and binary forms, with or without modification, 
   are permitted provided that the following conditions are met:
     * Redistributions of source code must retain the above copyright notice, 
       this list of conditions and the following disclaimer. 
     * Redistributions in binary form must reproduce the above copyright notice, 
       this list of conditions and the following disclaimer in the documentation 
       and/or other materials provided with the distribution. 
     * Neither the name of the University of La Laguna nor the names of its contributors 
       may be used to endorse or promote products derived from this software without 
       specific prior written permission. 

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
   WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
   IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
   BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
   OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY 
   OF SUCH DAMAGE.

  FILE:              c_jacobi_coarse.c
  VERSION:           1.1 (Coarse-Grained Granularity Variant)
  DATE:              Nov 2024
  DESCRIPTION:       Coarse-grained version of Jacobi iterative solver.
                     Uses static scheduling with large chunk sizes to minimize overhead
                     at the cost of potential load imbalance.
  GRANULARITY:       COARSE - Uses schedule(static, chunk) where chunk = m/threads.
                     Large contiguous chunks reduce synchronization overhead but may
                     cause imbalance in irregular convergence scenarios.
  COMMENTS:          Modified from original to use static scheduling with large chunks
                     for coarse-grained parallelism.
  REFERENCES:        http://www.rz.rwth-aachen.de/computing/hpc/prog/par/openmp/jacobi.html
  BASIC PRAGMAS:     parallel for with schedule(static, chunk)
  USAGE:             ./c_jacobi_coarse.par 5000 5000 0.8 1.0 1e-7 1000
  INPUT:             n - grid dimension in x direction
                     m - grid dimension in y direction
                     alpha - Helmholtz constant (always greater than 0.0)
                     tol   - error tolerance for iterative solver
                     relax - Successive over relaxation parameter
                     mits  - Maximum iterations for iterative solver
**************************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "OmpSCR.h"

#define U(i,j) u[(i)*n+(j)]
#define F(i,j) f[(i)*n+(j)]
#define UOLD(i,j) uold[(i)*n+(j)]
#define NUM_ARGS  6
#define NUM_TIMERS 1

int n, m, mits;
double tol, relax, alpha;

void jacobi (int n, int m, double dx, double dy, 
             double alpha, double omega, 
             double *u, double *f, 
             double tol, int maxit );

/******************************************************
* Initializes data 
* Assumes exact solution is u(x,y) = (1-x^2)*(1-y^2)
*
******************************************************/
void initialize(  
                int n,    
                int m,
                double alpha,
                double *dx,
                double *dy,
                double *u,
                double *f)
{
  int i,j,xx,yy;

  *dx = 2.0 / (n-1);
  *dy = 2.0 / (m-1);

  /* Initilize initial condition and RHS */
  for (j=0; j<m; j++){
    for (i=0; i<n; i++){
      xx = -1.0 + *dx * (i-1);
      yy = -1.0 + *dy * (j-1);
      U(j,i) = 0.0;
      F(j,i) = -alpha * (1.0 - xx*xx) * (1.0 - yy*yy)
                - 2.0 * (1.0 - xx*xx) - 2.0 * (1.0 - yy*yy);
    }
  }
      
}


/************************************************************
* Checks error between numerical and exact solution 
*
************************************************************/
void error_check(
                 int n,
                 int m,
                 double alpha,
                 double dx,
                 double dy,
                 double *u,
                 double *f)
{
  int i,j;
  double xx, yy, temp, error;

  dx = 2.0 / (n-1);
  dy = 2.0 / (n-2);
  error = 0.0;

  for (j=0; j<m; j++){
    for (i=0; i<n; i++){
      xx = -1.0 + dx * (i-1);
      yy = -1.0 + dy * (j-1);
      temp = U(j,i) - (1.0 - xx*xx) * (1.0 - yy*yy);
      error += temp*temp;
    }
  }

 error = sqrt(error)/(n*m);

  printf("Solution Error : %g\n", error);

}




int main(int argc, char **argv){
    double *u, *f, dx, dy;
    double dt, mflops;
    int NUMTHREADS;
    char *PARAM_NAMES[NUM_ARGS] = {"Grid dimension: X dir =", "Grid dimension: Y dir =", "Helmhotlz constant =", 
                                   "Successive over-relaxation parameter =", 
                                   "error tolerance for iterative solver =", "Maximum iterations for solver ="}; 
    char *TIMERS_NAMES[NUM_TIMERS] = {"Total_time"};
    char *DEFAULT_VALUES[NUM_ARGS] = {"5000", "5000", "0.8", "1.0", "1e-7", "1000"};



   NUMTHREADS = omp_get_max_threads();
   OSCR_init (NUMTHREADS, "Jacobi Solver v1 (Coarse-Grained)", "Use 'jacobi_coarse' <n> <m> <alpha> <relax> <tol> <mits>", NUM_ARGS,
                PARAM_NAMES, DEFAULT_VALUES , NUM_TIMERS, NUM_TIMERS, TIMERS_NAMES,
                argc, argv);

    n = OSCR_getarg_int(1);
    m = OSCR_getarg_int(2);
    alpha = OSCR_getarg_double(3);
    relax = OSCR_getarg_double(4);
    tol = OSCR_getarg_double(5);
    mits = OSCR_getarg_int(6);

    printf("-> %d, %d, %g, %g, %g, %d\n",
           n, m, alpha, relax, tol, mits);
    
    u = (double *) OSCR_calloc(n*m, sizeof(double));
    f = (double *) OSCR_calloc(n*m, sizeof(double));


    /* arrays are allocated and initialzed */
    initialize(n, m, alpha, &dx, &dy, u, f);
    

    /* Solve Helmholtz eqiation */
    OSCR_timer_start(0);
    jacobi(n, m, dx, dy, alpha, relax, u,f, tol, mits);

    OSCR_timer_stop(0);
    dt = OSCR_timer_read(0);

    printf(" elapsed time : %12.6f\n", dt);
    mflops = (0.000001*mits*(m-2)*(n-2)*13) / dt;
    printf(" MFlops       : %12.6g (%d, %d, %d, %g)\n",mflops, mits, m, n, dt);

    error_check(n, m, alpha, dx, dy, u, f);

    OSCR_report();
   
    return 0;
}



/*************************************************************
* Subroutine jacobi - COARSE-GRAINED VERSION
* Solves poisson equation on rectangular grid assuming : 
* (1) Uniform discretization in each direction, and 
* (2) Dirichlect boundary conditions 
* 
* Jacobi method uses two arrays to allow read data to be 
* from the old values, and writes are to the new values
*
* Inputs : n,m number of grid points in the x/y directions 
*          dx,dy grid spacing in x/y directions 
*          alpha - Helmholtz eqn. coefficient 
*          omega - relaxation factor 
*          f(n,m) - right hand side function 
*          u(n,m) - dependent variable/solution
*          tol    - error tolerance for iterative solver 
*          maxit  - maximum number of iterations 
*
* Outputs : u(n,m) - updated solution 
*
* COARSE-GRAINED: Uses schedule(static, chunk) where chunk = m / num_threads.
*                 Minimizes synchronization overhead by assigning large contiguous blocks.
*************************************************************/
void jacobi (int n, int m, double dx, double dy, double alpha, 
             double omega, double *u, double *f, double tol, int maxit )
{
  int i,j,k;
  double  error, resid, ax, ay, b;
  double  *uold;
  int chunk_size;
  int num_threads;

  uold = (double *) OSCR_calloc(n*m, sizeof(double));

  ax = 1.0/(dx*dx); /* X-direction coef */
  ay = 1.0/(dy*dy); /* Y-direction coef */
  b  = -2.0/(dx*dx)-2.0/(dy*dy) - alpha; /* Central coeff */ 

  /* Calculate chunk size for coarse-grained scheduling */
  #pragma omp parallel
  {
    #pragma omp single
    {
      num_threads = omp_get_num_threads();
      chunk_size = m / num_threads;
      if (chunk_size < 1) chunk_size = 1;
    }
  }

  error = 10.0 * tol;
  k = 1;

  while (k<=maxit && error>tol) {
    error = 0.0;

    /* Copy new solution into old - Coarse-grained with static scheduling */
    #pragma omp parallel for private(i) schedule(static, chunk_size)
    for (j=0; j<m; j++)
      for (i=0; i<n; i++)
        UOLD(j,i) = U(j,i);

    /* Compute stencil, residual and update - Coarse-grained with static scheduling */
    #pragma omp parallel for reduction(+:error) private(i,resid) schedule(static, chunk_size)
    for (j=1; j<m-1; j++)
      for (i=1; i<n-1; i++){
        resid = (ax*(UOLD(j,i-1) + UOLD(j,i+1))
               + ay*(UOLD(j-1,i) + UOLD(j+1,i))+ b * UOLD(j,i) - F(j,i))/b;

        U(j,i) = UOLD(j,i) - omega * resid;
        error += resid*resid ;
      }

    /* Error check */
    k++;
    error = sqrt(error)/(n*m);

  }  /*  End iteration loop */

  printf("Total Number of Iterations:%d\n", k); 
  printf("Residual:%E\n", error); 

  OSCR_free(uold);
}
