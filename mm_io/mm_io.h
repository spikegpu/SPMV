# ifndef MM_IO_H
# define MM_IO_H

#include <string>

void read_mm_matrix_csr(const std::string& fileMat, int &N, int &NNZ, unsigned* &row_offsets, unsigned *&column_indices, double *&values);

void read_mm_matrix_coo(const std::string& fileMat, int &N, int &NNZ, unsigned* &row_indices, unsigned *&column_indices, double *&values);

#endif // MM_IO_H
