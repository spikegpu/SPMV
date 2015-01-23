#include <algorithm>
#include <fstream>
#include <cmath>
#include <map>
#include <stdio.h>
#include <stdlib.h>

#include <cusp/io/matrix_market.h>
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>
#include <cusp/blas/blas.h>

#include <src/timer.h>


// -----------------------------------------------------------------------------
// Macro to obtain a random number between two specified values
// -----------------------------------------------------------------------------
#define RAND(L,H)  ((L) + ((H)-(L)) * (float)rand()/(float)RAND_MAX)


// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
#ifdef WIN32
#   define ISNAN(A)  (_isnan(A))
#else
#   define ISNAN(A)  (isnan(A))
#endif


// -----------------------------------------------------------------------------
// Typedefs
// -----------------------------------------------------------------------------
typedef double REAL;
typedef double PREC_REAL;

typedef typename cusp::csr_matrix<int, REAL, cusp::device_memory> Matrix;
typedef typename cusp::array1d<REAL, cusp::device_memory>         Vector;
typedef typename cusp::array1d<REAL, cusp::host_memory>           VectorH;
typedef typename cusp::array1d<PREC_REAL, cusp::device_memory>    PrecVector;


// -----------------------------------------------------------------------------
using std::cout;
using std::cerr;
using std::cin;
using std::endl;
using std::string;
using std::vector;

// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------
int main(int argc, char** argv) 
{
	// Set up the problem to be solved.
	string         fileMat = "/home/ali/CUDA_project/reordering/matrices/ancfBigDan.mtx";
	if (argc > 1)
		fileMat = argv[1];

	cout << fileMat << endl;

	// Get matrix and rhs.
	Matrix A;
	Vector b;
	Vector x;

	cusp::io::read_matrix_market_file(A, fileMat);

	b.resize(A.num_rows);

	{
		VectorH x_h(A.num_rows);

		for (int i = 0; i < A.num_rows; i++)
			x_h[i] = RAND(2,10) / 2;

		x = x_h;
	}

	CUDATimer timer;
	int counter = 0;
	double elapsed = 0.0;
	for (int i = 0; i < 10; i++) {
		timer.Start();
		cusp::multiply(A, x, b);
		timer.Stop();
		if (i > 0) {
			counter ++;
			elapsed += timer.getElapsed();
		}
	}
	elapsed /= counter;
	cout << "CUSP CSR: " << elapsed << endl;

	return 0;
}
