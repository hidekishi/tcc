/*************************************************************************
  This program is part of the
	OpenMP Source Code Repository

	http://www.pcg.ull.es/ompscr/
	e-mail: ompscr@etsii.ull.es

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License 
  (LICENSE file) along with this program; if not, write to
  the Free Software Foundation, Inc., 59 Temple Place, Suite 330, 
  Boston, MA  02111-1307  USA
	
FILE:		c_lu_fine.c
VERSION:	1.0 (Fine-Grained Granularity Variant)
DATE:		Nov 2024
AUTHOR:		Arturo González-Escribano (original)
DESCRIPTION:    Fine-grained version of LU reduction.
		Uses dynamic scheduling with small chunk size for better load balancing.
GRANULARITY:    FINE - Uses schedule(dynamic, 2) to distribute rows adaptively.
		Smaller chunks enable better load distribution across threads.
COMMENTS:        Modified from original to use dynamic scheduling for fine-grained parallelism.
REFERENCES:     
BASIC PRAGMAS:	parallel-for with schedule(dynamic, 2)
USAGE: 		./c_lu_fine.par <size>
INPUT:		The matrix has fixed innitial values: 
		M[i][j] = 1.0	                   iff (i==j)
		M[i][j] = 1.0 + (i*numColums)+j    iff (i!=j)

OUTPUT:		Compile with -DDEBUG to see final matrix values
**************************************************************************/

#include<stdio.h>
#include<stdlib.h>
#include<OmpSCR.h>


/* PROTOYPES */
void lu(int, int);


/* MAIN: PROCESS PARAMETERS */
int main(int argc, char *argv[]) {
int nthreads, size;
char *argNames[1] = { "size" };
char *defaultValues[1] = { "500" };
char *timerNames[1] = { "EXE_TIME" };

nthreads = omp_get_max_threads();
OSCR_init( nthreads,
	"LU reduction of a dense matrix (Fine-Grained).",
	NULL,
	1,
	argNames,
	defaultValues,
	1,
	1,
	timerNames,
	argc,
	argv );

/* 1. GET PARAMETERS */
size = OSCR_getarg_int(1);

/* 2. CALL COMPUTATION */
lu(nthreads, size);

/* 3. REPORT */
OSCR_report();


return 0;
}


/*
* LU FORWARD REDUCTION - FINE-GRAINED VERSION
* Uses dynamic scheduling with chunk size 2 for adaptive load balancing
*/
void lu(int nthreads, int size) {
/* DECLARE MATRIX AND ANCILLARY DATA STRUCTURES */
double **M;
double **L;

/* VARIABLES */
int i,j,k;

/* 0. ALLOCATE MATRICES MEMORY */
M = (double **)OSCR_calloc(size, sizeof(double *));
L = (double **)OSCR_calloc(size, sizeof(double *));
for (i=0; i<size; i++) {
	M[i] = (double *)OSCR_calloc(size, sizeof(double));
	L[i] = (double *)OSCR_calloc(size, sizeof(double));
	}

/* 1. INITIALIZE MATRIX */
for (i=0; i<size; i++) {
	for (j=0; j<size; j++) {
		if (i==j) M[i][j]=1.0;
		else M[i][j]=1.0+(i*size)+j; 
		L[i][j]=0.0;
		}
	}

/* 3. START TIMER */
OSCR_timer_start(0);

/* 4. ITERATIONS LOOP */
for(k=0; k<size-1; k++) {

	/* 4.1. PROCESS ROWS IN PARALLEL WITH FINE-GRAINED DYNAMIC SCHEDULING */
	/* Chunk size of 2 provides balance between overhead and load distribution */
#pragma omp parallel for default(none) shared(M,L,size,k) private(i,j) schedule(dynamic,2)
	for (i=k+1; i<size; i++) {
		/* 4.1.1. COMPUTE L COLUMN */
		L[i][k] = M[i][k] / M[k][k];

		/* 4.1.2. COMPUTE M ROW ELEMENTS */
		for (j=k+1; j<size; j++) 
			M[i][j] = M[i][j] - L[i][k]*M[k][j];
		}

/* 4.2. END ITERATIONS LOOP */
	}


/* 5. STOP TIMER */
OSCR_timer_stop(0);

/* 6. WRITE MATRIX (DEBUG) */
#ifdef DEBUG
#include "debug_ML.c"
#endif

/* 7. END */
}
