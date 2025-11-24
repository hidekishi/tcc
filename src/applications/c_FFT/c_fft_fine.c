/* ***********************************************************************
  FFT - Fine-Grained Version
  
  This version uses deeper nested parallelism and smaller task granularity.
  Parallelizes more levels of recursion for better scalability with many threads.
**************************************************************************/
#include "OmpSCR.h"
#include <math.h>

#define KILO	  (1024)
#define DEFAULT_SIZE_IN_KB (64)
#define	NUM_ARGS	1
#define	NUM_TIMERS	1
#define FINE_CUTOFF 64  /* Create parallel tasks for N >= 64 */

typedef double doubleType;
typedef struct {
  doubleType re;
  doubleType im;
} Complex;

void initialize(unsigned Size, Complex *a);
void write_array(unsigned Size, Complex *a);
int test_array(unsigned Size, Complex *a);
void FFT(Complex *A, Complex *a, Complex *W, unsigned N, unsigned stride, Complex *D);
void Roots(unsigned Size, Complex *W);
unsigned get_params(int argc, char *argv[]);

void initialize(unsigned Size, Complex *a) {
  unsigned i;
  for(i = 0; i < Size; i++) {
    a[i].re = 1.0;
    a[i].im = 0.0;
  }
}

void write_array(unsigned Size, Complex *a) {
  unsigned i;
  for(i = 0; i < Size; i++)
    printf("a[%2u] = [%.8lf,%.8lf]\n", i, a[i].re, a[i].im);
}

int test_array(unsigned Size, Complex *a) {
  register unsigned i;
	unsigned OK = 1;

  if((a[0].re == Size) && (a[0].im == 0)) {
    for(i = 1; i < Size; i++)
      if (a[i].re != 0.0 || a[i].im != 0.0) {
        OK = 0;
        break;
      }
  }
  else OK = 0;
  return OK;
}

void Roots(unsigned Size, Complex *W) {
  register unsigned i;
  double phi;
  Complex Omega;
  
  phi = 4 * atan(1.0) / (double)Size;
  Omega.re = cos(phi);
  Omega.im = sin(phi);
  W[0].re = 1.0;
  W[0].im = 0.0;
  for(i = 1; i < Size; i++) {
    W[i].re = W[i-1].re * Omega.re - W[i-1].im * Omega.im;
    W[i].im = W[i-1].re * Omega.im + W[i-1].im * Omega.re;
  }
}

/* Fine-grained FFT: Parallelize deeper recursion levels */
void FFT(Complex *A, Complex *a, Complex *W, unsigned N,           
            unsigned stride, Complex *D) {
  Complex *B, *C;
  Complex Aux, *pW;
  unsigned n;
	int i;

  if (N == 1) {
    A[0].re = a[0].re;
    A[0].im = a[0].im;
  }
  else {
    n = (N >> 1);

    /* Fine-grained: Parallelize even for smaller N */
    if (N >= FINE_CUTOFF) {
#pragma omp parallel for if(N >= FINE_CUTOFF)
      for(i = 0; i <= 1; i++) {
        FFT(D + i * n, a + i * stride, W, n, stride << 1, A + i * n);
      }
    } else {
      for(i = 0; i <= 1; i++) {
        FFT(D + i * n, a + i * stride, W, n, stride << 1, A + i * n);
      }
    }
    
    /* Combination with dynamic scheduling for better load balance */
    B = D;
    C = D + n;
#pragma omp parallel for default(none) private(i, Aux, pW) shared(stride, n, A, B, C, W) schedule(dynamic, 8)
    for(i = 0; i <= n - 1; i++) {
      pW = W + i * stride;
      Aux.re = pW->re * C[i].re - pW->im * C[i].im;
      Aux.im = pW->re * C[i].im + pW->im * C[i].re;
      
      /* Loop unrolling opportunity */
      A[i].re = B[i].re + Aux.re;
      A[i].im = B[i].im + Aux.im;
      A[i+n].re = B[i].re - Aux.re;
      A[i+n].im = B[i].im - Aux.im;
    }
  }
}

unsigned get_params(int argc, char *argv[]) {
	char usage_str[] = "<size_in_Kb>";
  unsigned sizeInKb;

  if (argc == 2)
    sizeInKb = atoi(argv[1]);
	else
		if (argc == 1)
			sizeInKb = DEFAULT_SIZE_IN_KB;
    else {
      printf("\nUse: %s %s\n", argv[0], usage_str);
      exit(-1);
    }
  printf("\nUse: %s %s\n", argv[0], usage_str);
  printf("Running with Size: %d K\n", sizeInKb);
	return sizeInKb;
}

int main(int argc, char *argv[]) {
  unsigned N;
  Complex *a, *A, *W, *D;
	int NUMTHREADS;
	char *PARAM_NAMES[NUM_ARGS] = {"Size of the input signal (in Kb)"};
	char *TIMERS_NAMES[NUM_TIMERS] = {"Total_time" };
	char *DEFAULT_VALUES[NUM_ARGS] = {"64"};
 	
 	NUMTHREADS = omp_get_max_threads();
	OSCR_init (NUMTHREADS, "FFT (Fine-Grained)", "Use 'fft' <size (in K)>", NUM_ARGS, 
		PARAM_NAMES, DEFAULT_VALUES , NUM_TIMERS, NUM_TIMERS, TIMERS_NAMES, 
		argc, argv);

	N = KILO * OSCR_getarg_int(1);
  
  a = (Complex*)calloc(N, sizeof(Complex));
  A = (Complex*)calloc(N, sizeof(Complex));
  D = (Complex*)calloc(N, sizeof(Complex));
  W = (Complex*)calloc(N>>1, sizeof(Complex));
  if((a==NULL) || (A==NULL) || (D==NULL) || (W==NULL)) {
		printf("Not enough memory initializing arrays\n");
		exit(1);
	}
  initialize(N, a);
  Roots(N >> 1, W);
  OSCR_timer_start(0);
  FFT(A, a, W, N, 1, D);
  OSCR_timer_stop(0);

	printf("Test array: ");
	if (test_array(N, A))
		printf("Ok\n");
	else
		printf("Fails\n");
	OSCR_report(1, TIMERS_NAMES);
  free(W);
  free(D);
  free(A);
  free(a);

  return 0;
}
