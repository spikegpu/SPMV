#include <iostream>
#include <cstdio>
#include <string>
#include <stdlib.h>
#include <cstring>
#include <cmath>
#include <src/timer.h>
#include <algorithm>

#include <cusp/io/matrix_market.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>
#include <cusp/blas/blas.h>

using std::string;
using std::cout;
using std::cerr;
using std::endl;


typedef double REAL;

typedef typename cusp::coo_matrix<int, REAL, cusp::device_memory> CooMatrix;
typedef typename cusp::csr_matrix<int, REAL, cusp::device_memory> CsrMatrix;
typedef typename cusp::csr_matrix<int, REAL, cusp::host_memory>   CsrMatrixH;
typedef typename cusp::ell_matrix<int, REAL, cusp::device_memory> EllMatrix;
typedef typename cusp::hyb_matrix<int, REAL, cusp::device_memory> HybMatrix;
typedef typename cusp::array1d<REAL, cusp::device_memory>         Vector;
typedef typename cusp::array1d<REAL, cusp::host_memory>           VectorH;


int main (int argc, char **argv)
{
	int N, NNZ, NNZ2;

	if (argc < 2) return 1;

	string mat_file  = "/home/ali/CUDA_project/reordering/matrices/M_invD_";
	string mat_file2 = "/home/ali/CUDA_project/reordering/matrices/D_T_";
	string vec_file  = "/home/ali/CUDA_project/reordering/matrices/gamma_";

	mat_file.append(argv[1]);
	mat_file.append(".mtx");
	mat_file2.append(argv[1]);
	mat_file2.append(".mtx");
	vec_file.append(argv[1]);
	vec_file.append(".mtx");

	cout << argv[1] << endl;

	CsrMatrixH M_invD_h;
	CsrMatrixH D_T_h;

	cusp::io::read_matrix_market_file(M_invD_h, mat_file);
	cusp::io::read_matrix_market_file(D_T_h,    mat_file2);

	N    = M_invD_h.num_rows;
	NNZ  = M_invD_h.num_entries;
	NNZ2 = D_T_h.num_entries;

	Vector gamma;
	cusp::io::read_matrix_market_file(gamma, vec_file);

	Vector t1(N);
	Vector t2(N);

	{
		CsrMatrix M_invD, D_T;

		M_invD = M_invD_h;
		D_T    = D_T_h;

		CPUTimer loc_timer, timer2;
		double elapsed = 0.0;
		int counter = 10 - 1;
		loc_timer.Start();
		for (int i = 0; i < 10; i++) {
			if (i == 0)
				timer2.Start();

			cusp::multiply(M_invD, gamma, t1);
			cusp::multiply(D_T,    t1,    t2);
			cudaDeviceSynchronize();

			if (i == 0)
				timer2.Stop();
		}
		loc_timer.Stop();
		elapsed = (loc_timer.getElapsed() - timer2.getElapsed())/ counter;
		cout << "CUSP CSR: " << elapsed << " " << 2.0 * (NNZ + NNZ2) / elapsed << endl;
	}

	{
		CooMatrix M_invD, D_T;

		M_invD = M_invD_h;
		D_T    = D_T_h;

		CPUTimer loc_timer, timer2;
		double elapsed = 0.0;
		int counter = 10 - 1;
		loc_timer.Start();
		for (int i = 0; i < 10; i++) {
			if (i == 0)
				timer2.Start();

			cusp::multiply(M_invD, gamma, t1);
			cusp::multiply(D_T,    t1,    t2);
			cudaDeviceSynchronize();

			if (i == 0)
				timer2.Stop();
		}
		loc_timer.Stop();
		elapsed = (loc_timer.getElapsed() - timer2.getElapsed())/ counter;
		cout << "CUSP COO: " << elapsed << " " << 2.0 * (NNZ + NNZ2) / elapsed << endl;
	}

	{
		EllMatrix M_invD, D_T;

		M_invD = M_invD_h;
		D_T    = D_T_h;

		CPUTimer loc_timer, timer2;
		double elapsed = 0.0;
		int counter = 10 - 1;
		loc_timer.Start();
		for (int i = 0; i < 10; i++) {
			if (i == 0)
				timer2.Start();

			cusp::multiply(M_invD, gamma, t1);
			cusp::multiply(D_T,    t1,    t2);
			cudaDeviceSynchronize();

			if (i == 0)
				timer2.Stop();
		}
		loc_timer.Stop();
		elapsed = (loc_timer.getElapsed() - timer2.getElapsed())/ counter;
		cout << "CUSP ELL: " << elapsed << " " << 2.0 * (NNZ + NNZ2) / elapsed << endl;
	}

	{
		HybMatrix M_invD, D_T;

		M_invD = M_invD_h;
		D_T    = D_T_h;

		CPUTimer loc_timer, timer2;
		double elapsed = 0.0;
		int counter = 10 - 1;
		loc_timer.Start();
		for (int i = 0; i < 10; i++) {
			if (i == 0)
				timer2.Start();

			cusp::multiply(M_invD, gamma, t1);
			cusp::multiply(D_T,    t1,    t2);
			cudaDeviceSynchronize();

			if (i == 0)
				timer2.Stop();
		}
		loc_timer.Stop();
		elapsed = (loc_timer.getElapsed() - timer2.getElapsed())/ counter;
		cout << "CUSP HYB: " << elapsed << " " << 2.0 * (NNZ + NNZ2) / elapsed << endl;
	}
	return 0;
}
