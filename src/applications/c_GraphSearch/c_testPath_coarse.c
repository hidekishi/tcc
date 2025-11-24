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
	
FILE:		c_testPath_coarse.c
VERSION:	1.0 (Coarse-Grained Granularity Variant)
DATE:		Nov 2024
AUTHOR:		Arturo González-Escribano (original)
DESCRIPTION:	Coarse-grained version of graph path search.
		Uses larger batches of work per thread access to the pool.
GRANULARITY:    COARSE - Workers process multiple nodes in batches before pool access.
		Each worker retrieves a batch of nodes (up to 10) from the pool at once,
		reducing critical section frequency at cost of potential load imbalance.
COMMENTS:       Modified from original to use coarse-grained work distribution.
REFERENCES:     
BASIC PRAGMAS:	parallel, critical
USAGE: 		./c_testPath_coarse.par <source_node_num> <target_node_num> <graph_file>
INPUT:		A graph stored in a file (.gph format)
OUTPUT:		True or False if path exists between source and target nodes
**************************************************************************/

#include<stdio.h>
#include<stdlib.h>
#include<OmpSCR.h>
#include "AStack.h"
#include "tg.h"

#define BATCH_SIZE 10  /* Coarse-grained: process up to 10 nodes per batch */

/* PROTOYPES */
void testPath(int, int, int, tg);


/* MAIN: PROCESS PARAMETERS */
int main(int argc, char *argv[]) {
int nthreads, source, target;
char *graphFileName;
FILE *graphFile;
tg graph;

char *argNames[3] = { "source_node_num", "target_node_num", "graph_file" };
char *defaultValues[3] = { "1", "29", "exampleGraph_01.gph"  };
char *timerNames[1] = { "EXE_TIME" };

nthreads = omp_get_max_threads();
OSCR_init( nthreads,
	"Check if there is a path on a directed graph (Coarse-Grained).",
	NULL,
	3,
	argNames,
	defaultValues,
	1,
	1,
	timerNames,
	argc,
	argv );

/* 1. GET PARAMETERS */
source = OSCR_getarg_int(1);
target = OSCR_getarg_int(2);
graphFileName = OSCR_getarg_string(3);

/* 2. READ GRAPH */
graphFile = fopen(graphFileName,"r");
if (graphFile==NULL) {
	fprintf(stderr,"Imposible to open graph file: %s\n",graphFileName);
	exit(-1);
	}
graph = tg_read(graphFile);
fclose(graphFile);

/* 3. CALL COMPUTATION */
testPath(nthreads, source, target, graph);

/* 4. REPORT */
OSCR_report();

return 0;
}

/*
*
* Graph search - COARSE-GRAINED VERSION
* Test if exists a path from a source node to a target node
*
* Parallelization method: Shared-Memory workers-farm with coarse-grained work distribution
* Each worker processes batches of nodes, minimizing critical sections at cost of load balance
*
*/
void testPath(int nthreads, int source, int target, tg graph) {
/* SHARED STRUCTURES */
Bool	*searched=NULL;
Astack	pool;
Bool	found = FALSE;
int	ind;

/* ENDING CONTROL */
int		num_waiting=0;

/* 1. ALLOCATE MEMORY FOR ANCILLARY STRUCTURES */
pool = Ast_init();
searched = OSCR_calloc(tg_nodes(graph), sizeof(Bool));

for (ind=0; ind<tg_nodes(graph); ind++) { searched[ind]=FALSE; }

/* 2. INIT "nodes to explore" POOL WITH THE source ID */
Ast_push(pool, source);


/* 3. START TIMER */
OSCR_timer_start(0);


/* 4. SPAWN WORKERS - COARSE-GRAINED: Process batches of nodes per critical section */
#pragma omp parallel default(none) 					\
	shared(nthreads,num_waiting,graph,searched,pool,target,found) 	
{
Bool		waiting = FALSE;
tg_task		batch[BATCH_SIZE];
int		batch_count = 0;
task_list	succs;
int		num_succs;
int		ind, b;
#ifdef DEBUG
int		numPops=0;
int		numNoPops=0;
int		thread = omp_get_thread_num();
#endif

/* WORKER WORKS UNTIL:
 * 	ALL WORKERS ARE WAITING (TARGET NOT FOUND) 
 * 	OR SOMEONE FINDS THE TARGET
 */
while ( num_waiting != nthreads && !found ) {

	/* 1. GET BATCH OF ELEMENTS TO PROCESS (OR WAIT UNTIL MORE ELEMENTS) */
	/* COARSE-GRAINED: Retrieve multiple nodes at once to reduce critical section frequency */
	batch_count = 0;
	while( batch_count == 0 && num_waiting != nthreads && !found) {

		/* ALL POOL OPERATIONS ARE MONITORIZED */
		#pragma omp critical
			{
			/* 1.1. CHECK THE POOL AND GET A BATCH */
			while ( Ast_more(pool) && batch_count < BATCH_SIZE ) {
				/* 1.1.1. ELEMENTS IN THE POOL: GET NEXT BATCH */
				batch[batch_count] = Ast_pop(pool);
				batch_count++;
#ifdef DEBUG
numPops++;
#endif
				}
			
			/* 1.1.2. IF WAITING AND GOT ELEMENTS, CHANGE STATE */
			if ( batch_count > 0 && waiting ) { 
				waiting = FALSE; 
				num_waiting--; 
			}
			/* 1.1.3. EMPTY POOL: IF NOT WAITING, CHANGE STATE */
			else if ( batch_count == 0 && !waiting ) { 
#ifdef DEBUG
numNoPops++;
#endif
				waiting = TRUE; 
				num_waiting++; 
			}
			/* OMP END CRITICAL: MONITORIZED OPERATION */
			}

		} /* END GET BATCH FROM THE POOL */


	/* 2. PROCESS BATCH OF ELEMENTS - COARSE-GRAINED: Multiple nodes per critical section access */
	for (b = 0; b < batch_count && !found; b++) {
		tg_task next = batch[b];

		/* 2.1. TARGET FOUND: END ALL */
		if (next == target) { 
			found = TRUE; 
			break;
		}

		/* 2.2. HAS SUCCESORS: GET AND PUSH THEM */
		if ( tg_succ_num(graph, next) > 0 ) {
			/* 2.3.1. GET SUCCS LIST */
			num_succs = tg_succ_num(graph, next);
			succs = tg_succ(graph, next);

			/* 2.3.2. PUSH SUCCS TO POOL: MONITORIZED OPERATION */
			/* COARSE-GRAINED: Batch processing reduces pool access frequency */
			#pragma omp critical
			{
				for(ind=0; ind<num_succs; ind++) {
					tg_task	vp = succs[ind];

					/* PUSH ONLY NON-EXPLORED NODES */
					if ( ! searched[ vp ] ) {
						searched[ vp ] = TRUE;
						Ast_push(pool, vp);
					}
				}
			/* END OMP CRITICAL: MONITORIZED OPERATION */
			}
		}
	} /* END PROCESSING BATCH */

	} /* END PROCESSING */

#ifdef DEBUG
printf("#DEBUG Thread %d ENDING ----> Pops: %d, NoPops: %d\n",thread,numPops,numNoPops);
#endif

/* WORKERS END: PARALLEL REGION */
}

/* 5. STOP TIMER */
OSCR_timer_stop(0);

/* 6. WRITE RESULT */
printf("\nPath(%d,%d) = %d\n\n", source, target, found);

/* 7. END */
}
