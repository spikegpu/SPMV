#include "mm_io.h"

#include <string>

#include <cusp/io/matrix_market.h>
#include <cusp/csr_matrix.h>
#include <cusp/coo_matrix.h>

void read_mm_matrix_csr(const std::string& fileMat, int &N, int &NNZ, unsigned* &row_offsets, unsigned* &column_indices, double *&values)
{
	cusp::csr_matrix<unsigned, double, cusp::host_memory> A;
	cusp::io::read_matrix_market_file(A, fileMat);

	row_offsets = new unsigned [A.num_rows + 1];
	column_indices = new unsigned [A.num_entries];
	values = new double [A.num_entries];

	for (int i = 0; i <= A.num_rows; i++)
		row_offsets[i] = A.row_offsets[i];

	for (int i = 0; i < A.num_entries; i++) {
		column_indices[i] = A.column_indices[i];
		values[i]         = A.values[i];
	}

	N   = A.num_rows;
	NNZ = A.num_entries;
}

void read_mm_matrix_coo(const std::string& fileMat, int &N, int &NNZ, unsigned* &row_indices, unsigned *&column_indices, double *&values) 
{
	cusp::coo_matrix<unsigned, double, cusp::host_memory> A;
	cusp::io::read_matrix_market_file(A, fileMat);

	row_indices = new unsigned [A.num_entries];
	column_indices = new unsigned [A.num_entries];
	values = new double [A.num_entries];

	for (int i = 0; i < A.num_entries; i++) {
		row_indices[i]    = A.row_indices[i];
		column_indices[i] = A.column_indices[i];
		values[i]         = A.values[i];
	}

	N   = A.num_rows;
	NNZ = A.num_entries;
}

void read_mm_vector(const std::string& fileVec, double *&values)
{
	cusp::array1d<double, cusp::host_memory> v;
	cusp::io::read_matrix_market_file(v, fileVec);

	size_t v_size = v.size();
	values = new double [v_size];

	for (int i = 0; i < v_size; i++)
		values[i] = v[i];
}
