#include <iostream>
#include <cstdio>
#include <string>
#include <stdlib.h>
#include <cstring>
#include <cmath>
#include <src/timer.h>
#include <algorithm>

using std::string;
using std::cout;
using std::cerr;
using std::endl;

#define VIENNACL_WITH_UBLAS
#define VIENNACL_WITH_OPENCL

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include "CL/cl.h"
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>



// ViennaCL includes
//
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/matrix_proxy.hpp"
#include "viennacl/linalg/qr.hpp"
#include "viennacl/linalg/lu.hpp"
#include "viennacl/ocl/command_queue.hpp"

#include "mm_io/mm_io.h"

typedef double REAL;
typedef viennacl::vector<REAL>   Vector;
typedef viennacl::compressed_matrix<REAL, 8>   CsrMatrix;


int main (int argc, char **argv)
{
	viennacl::scalar<double> gpu_double = 2.71828;
	double cpu_double = gpu_double;

	int N, NNZ;

	REAL    *h_vals = NULL;
	cl_uint *h_cols = NULL, *h_rowDelimiters = NULL;

	string mat_file = "/home/ali/CUDA_project/reordering/matrices/ancfBigDan.mtx";
	if (argc > 1)
		mat_file = argv[1];

	cout << mat_file << endl;

	read_mm_matrix_csr(mat_file, N, NNZ, h_rowDelimiters, h_cols, h_vals);

	CsrMatrix A(N, N, NNZ);
	A.set(h_rowDelimiters, h_cols, h_vals, N, N, NNZ);

	std::vector<REAL> x_h(N);
	for (int i = 0; i < N; i++)
		x_h[i] = 0.1;

	Vector b(N);
	Vector x(N);

	std::vector<REAL> ref_out(N);

	for (int i = 0; i < N; i++) {
		double sum = 0.0;
		int start_idx = h_rowDelimiters[i], end_idx = h_rowDelimiters[i+1];
		for (int j = start_idx; j < end_idx; j++)
			sum += h_vals[j] * x_h[h_cols[j]];
		ref_out[i] = sum;
	}

	viennacl::copy(x_h, x); 
	

	CPUTimer loc_timer, timer2;
	double elapsed = 0.0;
	int counter = 10 - 1;
	loc_timer.Start();
	for (int i = 0; i < 10; i++) {
		if (i == 0)
			timer2.Start();
		b = viennacl::linalg::prod(A, x);
		viennacl::ocl::get_queue().finish();
		if (i == 0)
			timer2.Stop();
	}
	loc_timer.Stop();
	elapsed = (loc_timer.getElapsed() - timer2.getElapsed())/ counter;
	cout << "ViennaCL CSR: " << elapsed << endl;

	delete [] h_vals;
	delete [] h_cols;
	delete [] h_rowDelimiters;
	

	return 0;
}
