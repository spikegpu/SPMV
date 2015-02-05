#include <iostream>
#include <cstdio>
#include <string>
#include <stdlib.h>
#include <cstring>
#include <cmath>
#include <src/timer.h>
#include <algorithm>

#include <cusp/io/matrix_market.h>
#include <cusp/csr_matrix.h>
#include <cusp/blas/blas.h>

using std::string;
using std::cout;
using std::cerr;
using std::endl;


typedef double REAL;

typedef typename cusp::csr_matrix<int, REAL, cusp::host_memory>   CsrMatrixH;
typedef typename cusp::array1d<REAL, cusp::device_memory>         Vector;
typedef typename cusp::array1d<REAL, cusp::host_memory>           VectorH;


int main (int argc, char **argv)
{
	int N, NNZ;

	if (argc < 2) return 1;

	string mat_file  = argv[1];

	cout << argv[1] << " ";

	CsrMatrixH A;
	cusp::io::read_matrix_market_file(A, mat_file);

	N   = A.num_rows;
	NNZ = A.num_entries;

	double ave    = (double)NNZ / N;
	double stddev = 0.0;

	for (int i = 0; i < N; i++) {
		double delta = (A.row_offsets[i+1] - A.row_offsets[i] - ave);
		stddev += delta * delta;
	}
	stddev = sqrt(stddev);
	cout << ave << " " << stddev << endl;
	return 0;
}
