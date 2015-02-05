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
#include <src/cusparse_wrapper.h>


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
typedef typename cusp::csr_matrix<int, REAL, cusp::host_memory>   MatrixH;
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
	string         fileMat  = "/home/ali/CUDA_project/reordering/matrices/M_invD_";
	string         fileMat2 = "/home/ali/CUDA_project/reordering/matrices/D_T_";
	string         fileVec  = "/home/ali/CUDA_project/reordering/matrices/gamma_";

	if (argc < 2) return 1;

	fileMat.append(argv[1]);
	fileMat.append(".mtx");
	fileMat2.append(argv[1]);
	fileMat2.append(".mtx");
	fileVec.append(argv[1]);
	fileVec.append(".mtx");

	// Get matrix and rhs.
	MatrixH M_invD_h;
	MatrixH D_T_h;

	Vector  gamma;

	cusp::io::read_matrix_market_file(M_invD_h, fileMat);
	cusp::io::read_matrix_market_file(D_T_h,    fileMat2);
	cusp::io::read_matrix_market_file(gamma,    fileVec);

	Vector t1(gamma.size()), t2(gamma.size());

	cusparse::CuSparseCsrMatrixD M_invD(M_invD_h.row_offsets, M_invD_h.column_indices, M_invD_h.values);
	cusparse::CuSparseCsrMatrixD D_T   (D_T_h.row_offsets,    D_T_h.column_indices,    D_T_h.values);

	CUDATimer timer;
	int counter = 0;
	double elapsed = 0.0;
	for (int i = 0; i < 10; i++) {
		timer.Start();
		M_invD.spmv(gamma, t1);
		D_T.spmv(t1, t2);
		timer.Stop();

		if (i > 0) {
			counter ++;
			elapsed += timer.getElapsed();
		}
	}
	elapsed /= counter;
	cout << "cuSparse: " << elapsed << endl;

	return 0;
}
