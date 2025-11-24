/* ***********************************************************************
  QuickSort - Fine-Grained Version
  
  This version parallelizes at a deeper recursion level, creating more
  parallel tasks for better load balancing with many threads.
  Uses task-based parallelism with smaller cutoff threshold.
**************************************************************************/
#include "OmpSCR.h"

#define NUM_ARGS	1
#define NUM_TIMERS	1
#define KILO (1024)
#define MEGA (1024 * 1024)
#define DEFAULT_SIZE  (2 * MEGA)
#define MAXSIZE  (9 * MEGA)
#define NUM_STEPS 10
#define FINE_CUTOFF 1000  /* Create tasks for partitions > 1000 elements */

char USAGE_STR[] = "<size_in_Kb>";
int SIZE;
int array[MAXSIZE];      

void initialize(int *v, int seed);
void testit(int *v);
void qs(int *v, int first, int last);

void initialize(int *v, int seed) {
  unsigned i;
   srandom(seed);
   for(i = 0; i < SIZE; i++)
     v[i] = (int)random();
}

void testit(int *v) {
  register int k;
	int not_sorted = 0;
  for (k = 0; k < SIZE - 1; k++)
    if (v[k] > v[k + 1]) {
      not_sorted = 1;
      break;
    }
  if (not_sorted)
    printf("Array NOT sorted.\n");
	else
    printf("Array sorted.\n");
}

void qs(int *v, int first, int last) {
  int start[2], end[2], pivot, i, temp;
  int size = last - first + 1;

  if (first < last) {
     start[1] = first;
     end[0] = last;
     pivot = v[(first + last) / 2];
     while (start[1] <= end[0]) {
       while (v[start[1]] < pivot)
         start[1]++;
       while (pivot < v[end[0]])
         end[0]--;
       if (start[1] <= end[0]) {
         temp = v[start[1]];
         v[start[1]] = v[end[0]];
         v[end[0]] = temp;
         start[1]++;
         end[0]--;
       }
     }
     start[0] = first; 
     end[1]   = last; 

     /* Fine-grained: create tasks even for smaller partitions */
     if (size > FINE_CUTOFF) {
#pragma omp task firstprivate(v, start, end) shared(SIZE) if(size > FINE_CUTOFF)
       qs(v, start[0], end[0]);
#pragma omp task firstprivate(v, start, end) shared(SIZE) if(size > FINE_CUTOFF)
       qs(v, start[1], end[1]);
#pragma omp taskwait
     } else {
       qs(v, start[0], end[0]);
       qs(v, start[1], end[1]);
     }
   }
}

int main(int argc, char *argv[]) {
  int STEP, NUMTHREADS;
  double total_time;
  char *PARAM_NAMES[NUM_ARGS] = {"Size (in K)"};
  char *TIMERS_NAMES[NUM_TIMERS] = {"Total_time" };
  char *DEFAULT_VALUES[NUM_ARGS] = {"2048 K"};

  NUMTHREADS = omp_get_max_threads();
  OSCR_init (NUMTHREADS, "Quicksort (Fine-Grained)", "Use 'qsort' <size (in K)>", NUM_ARGS,
    PARAM_NAMES, DEFAULT_VALUES , NUM_TIMERS, NUM_TIMERS, TIMERS_NAMES,
    argc, argv);

  SIZE = OSCR_getarg_int(1);
  if (SIZE > MAXSIZE) {
    printf("Size: %d Maximum size: %d\n", SIZE, MAXSIZE);
    exit(-1);
  }
  
  for (STEP = 0; STEP < NUM_STEPS; STEP++) {
    initialize(array, STEP);
	  OSCR_timer_start(0);
#pragma omp parallel
{
#pragma omp single
    qs(array, 0, SIZE-1);
}
		OSCR_timer_stop(0);
    testit(array);
  }
	total_time = OSCR_timer_read(0);
	OSCR_report(1, TIMERS_NAMES);
	printf("\n \t# THREADS \tSIZE \tSTEPS \tTIME (secs.) \n");
	printf("\t%d \t\t%d \t%d \t%14.6lf \n", NUMTHREADS, SIZE, NUM_STEPS, total_time);
}
