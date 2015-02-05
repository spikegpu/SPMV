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
#include "viennacl/hyb_matrix.hpp"
#include "viennacl/ell_matrix.hpp"
#include "viennacl/coordinate_matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/matrix_proxy.hpp"
#include "viennacl/linalg/qr.hpp"
#include "viennacl/linalg/lu.hpp"
#include "viennacl/ocl/command_queue.hpp"
#include "viennacl/io/matrix_market.hpp"

#include "mm_io/mm_io.h"

typedef double REAL;
typedef viennacl::vector<REAL>          Vector;
typedef viennacl::hyb_matrix<REAL, 8>   HybMatrix;
typedef viennacl::ell_matrix<REAL, 8>   EllMatrix;
typedef viennacl::compressed_matrix<REAL, 8>   CsrMatrix;
typedef viennacl::coordinate_matrix<REAL, 8>   CooMatrix;


int main (int argc, char **argv)
{
	viennacl::scalar<double> gpu_double = 2.71828;
	double cpu_double = gpu_double;

	int N, NNZ, NNZ2;

	REAL    *h_vals = NULL;
	cl_uint *h_cols = NULL, *h_rows = NULL;

	string mat_file  = "/home/ali/CUDA_project/reordering/matrices/M_invD_9000.mtx";
	string mat_file2 = "/home/ali/CUDA_project/reordering/matrices/D_T_9000.mtx";
	string vec_file  = "/home/ali/CUDA_project/reordering/matrices/gamma_9000.mtx";

	cout << mat_file  << endl;
	cout << mat_file2 << endl;

	read_mm_matrix_coo(mat_file, N, NNZ, h_rows, h_cols, h_vals);


	boost::numeric::ublas::coordinate_matrix<REAL> M_invD_h(N, N, NNZ);
	for (int i = 0; i < NNZ; i++)
		M_invD_h.insert_element(h_rows[i], h_cols[i], h_vals[i]);

	delete [] h_rows;
	delete [] h_cols;
	delete [] h_vals;

	read_mm_matrix_coo(mat_file2, N, NNZ2, h_rows, h_cols, h_vals);
	boost::numeric::ublas::coordinate_matrix<REAL> D_T_h(N, N, NNZ2);
	for (int i = 0; i < NNZ2; i++)
		D_T_h.insert_element(h_rows[i], h_cols[i], h_vals[i]);

	delete [] h_rows;
	delete [] h_cols;
	delete [] h_vals;

	std::vector<REAL> x_h(N);

	read_mm_vector(vec_file, h_vals);
	for (int i = 0; i < N; i++)
		x_h[i] = h_vals[i];

	delete [] h_vals;

	Vector gamma(N);
	Vector t1(N);
	Vector t2(N);

	viennacl::copy(x_h, gamma); 

	{
		CsrMatrix M_invD, D_T;
		viennacl::copy(M_invD_h, M_invD);
		viennacl::copy(D_T_h, D_T);

		CPUTimer loc_timer, timer2;
		double elapsed = 0.0;
		int counter = 10 - 1;
		loc_timer.Start();
		for (int i = 0; i < 10; i++) {
			if (i == 0)
				timer2.Start();
			t1 = viennacl::linalg::prod(M_invD, gamma);
			viennacl::ocl::get_queue().finish();
			t2 = viennacl::linalg::prod(D_T, t1);
			viennacl::ocl::get_queue().finish();
			if (i == 0)
				timer2.Stop();
		}
		loc_timer.Stop();
		elapsed = (loc_timer.getElapsed() - timer2.getElapsed())/ counter;
		cout << "ViennaCL CSR: " << elapsed << " " << 2.0 * (NNZ + NNZ2) / elapsed << endl;
	}

	{
		CooMatrix M_invD, D_T;
		viennacl::copy(M_invD_h, M_invD);
		viennacl::copy(D_T_h, D_T);

		CPUTimer loc_timer, timer2;
		double elapsed = 0.0;
		int counter = 10 - 1;
		loc_timer.Start();
		for (int i = 0; i < 10; i++) {
			if (i == 0)
				timer2.Start();
			t1 = viennacl::linalg::prod(M_invD, gamma);
			viennacl::ocl::get_queue().finish();
			t2 = viennacl::linalg::prod(D_T, t1);
			viennacl::ocl::get_queue().finish();
			if (i == 0)
				timer2.Stop();
		}
		loc_timer.Stop();
		elapsed = (loc_timer.getElapsed() - timer2.getElapsed())/ counter;
		cout << "ViennaCL COO: " << elapsed << " " << 2.0 * (NNZ + NNZ2) / elapsed << endl;
	}

	{
		EllMatrix M_invD, D_T;
		viennacl::copy(M_invD_h, M_invD);
		viennacl::copy(D_T_h, D_T);

		CPUTimer loc_timer, timer2;
		double elapsed = 0.0;
		int counter = 10 - 1;
		loc_timer.Start();
		for (int i = 0; i < 10; i++) {
			if (i == 0)
				timer2.Start();
			t1 = viennacl::linalg::prod(M_invD, gamma);
			viennacl::ocl::get_queue().finish();
			t2 = viennacl::linalg::prod(D_T, t1);
			viennacl::ocl::get_queue().finish();
			if (i == 0)
				timer2.Stop();
		}
		loc_timer.Stop();
		elapsed = (loc_timer.getElapsed() - timer2.getElapsed())/ counter;
		cout << "ViennaCL ELL: " << elapsed << " " << 2.0 * (NNZ + NNZ2) / elapsed << endl;
	}

	{

		HybMatrix M_invD, D_T;
		viennacl::copy(M_invD_h, M_invD);
		viennacl::copy(D_T_h, D_T);

		CPUTimer loc_timer, timer2;
		double elapsed = 0.0;
		int counter = 10 - 1;
		loc_timer.Start();
		for (int i = 0; i < 10; i++) {
			if (i == 0)
				timer2.Start();
			t1 = viennacl::linalg::prod(M_invD, gamma);
			viennacl::ocl::get_queue().finish();
			t2 = viennacl::linalg::prod(D_T, t1);
			viennacl::ocl::get_queue().finish();
			if (i == 0)
				timer2.Stop();
		}
		loc_timer.Stop();
		elapsed = (loc_timer.getElapsed() - timer2.getElapsed())/ counter;
		cout << "ViennaCL HYB: " << elapsed << " " << 2.0 * (NNZ + NNZ2) / elapsed << endl;
	}
	return 0;
}
