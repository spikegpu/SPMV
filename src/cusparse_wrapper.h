#ifndef COMMON_H
#define COMMON_H



#define ALWAYS_ASSERT

#ifdef WIN32
typedef long long int64_t;
#endif



// ----------------------------------------------------------------------------
// If ALWAYS_ASSERT is defined, we make sure that  assertions are triggered 
// even if NDEBUG is defined.
// ----------------------------------------------------------------------------
#ifdef ALWAYS_ASSERT
// If NDEBUG is actually defined, remember this so
// we can restore it.
#  ifdef NDEBUG
#    define NDEBUG_ACTIVE
#    undef NDEBUG
#  endif
// Include the assert.h header file here so that it can
// do its stuff while NDEBUG is guaranteed to be disabled.
#  include <assert.h>
// Restore NDEBUG mode if it was active.
#  ifdef NDEBUG_ACTIVE
#    define NDEBUG
#    undef NDEBUG_ACTIVE
#  endif
#else
// Include the assert.h header file using whatever the
// current definition of NDEBUG is.
#  include <assert.h>
#endif

# include <memory.h>
# include <cstdio>
# include <iostream>

# include <thrust/scan.h>
# include <thrust/functional.h>
# include <thrust/sequence.h>
# include <thrust/iterator/zip_iterator.h>
# include <thrust/gather.h>
# include <thrust/binary_search.h>
# include <thrust/system/cuda/execution_policy.h>
# include <thrust/logical.h>
# include <thrust/host_vector.h>
# include <thrust/device_vector.h>
# include <thrust/device_ptr.h>
# include <thrust/adjacent_difference.h>
# include <thrust/inner_product.h>
# include <thrust/extrema.h>

# include "cusparse.h"
# include "cusolver.h"

# define TEMP_TOL 1e-10


// ----------------------------------------------------------------------------


namespace cusparse {

const unsigned int BLOCK_SIZE = 512;

const unsigned int MAX_GRID_DIMENSION = 32768;

inline
void kernelConfigAdjust(int &numThreads, int &numBlockX, const int numThreadsMax) {
	if (numThreads > numThreadsMax) {
		numBlockX = (numThreads + numThreadsMax - 1) / numThreadsMax;
		numThreads = numThreadsMax;
	}
}

inline
void kernelConfigAdjust(int &numThreads, int &numBlockX, int &numBlockY, const int numThreadsMax, const int numBlockXMax) {
	kernelConfigAdjust(numThreads, numBlockX, numThreadsMax);
	if (numBlockX > numBlockXMax) {
		numBlockY = (numBlockX + numBlockXMax - 1) / numBlockXMax;
		numBlockX = numBlockXMax;
	}
}

class CuSparseCsrMatrix_base
{
public:
	int    m_n;
	int    m_nnz;
	int    *m_row_offsets;
	int    *m_column_indices;
	int    *m_perm;
	int    *m_reordering;
	double *m_values;

	virtual ~CuSparseCsrMatrix_base() {}

	struct empty_row_functor
	{
		typedef bool result_type;
		typedef typename thrust::tuple<int, int>       IntTuple;
			__host__ __device__
			bool operator()(const IntTuple& t) const
			{
				const int a = thrust::get<0>(t);
				const int b = thrust::get<1>(t);

				return a != b;
			}
	};

	template <typename IVector>
	static void offsets_to_indices(const IVector& offsets, IVector& indices)
	{
		// convert compressed row offsets into uncompressed row indices
		thrust::fill(indices.begin(), indices.end(), 0);
		thrust::scatter_if( thrust::counting_iterator<int>(0),
				thrust::counting_iterator<int>(offsets.size()-1),
				offsets.begin(),
                    	thrust::make_transform_iterator(
                                thrust::make_zip_iterator( thrust::make_tuple( offsets.begin(), offsets.begin()+1 ) ),
                                empty_row_functor()),
				indices.begin());
		thrust::inclusive_scan(indices.begin(), indices.end(), indices.begin(), thrust::maximum<int>());
	}

	template <typename IntIterator, typename IntIterator2>
	static void offsets_to_indices(const IntIterator offsets_begin, const IntIterator  offsets_end, IntIterator2 indices_begin, IntIterator2 indices_end)
	{
		// convert compressed row offsets into uncompressed row indices
		thrust::fill(indices_begin, indices_end, 0);
		thrust::scatter_if( thrust::counting_iterator<int>(0),
				thrust::counting_iterator<int>((offsets_end - offsets_begin) - 1),
				offsets_begin,
                    	thrust::make_transform_iterator(
                                thrust::make_zip_iterator( thrust::make_tuple( offsets_begin, offsets_begin+1 ) ),
                                empty_row_functor()),
				indices_begin);
		thrust::inclusive_scan(indices_begin, indices_end, indices_begin, thrust::maximum<int>());
	}

	template <typename IVector>
	static void indices_to_offsets(const IVector& indices, IVector& offsets)
	{
		// convert uncompressed row indices into compressed row offsets
		thrust::lower_bound(indices.begin(),
				indices.end(),
				thrust::counting_iterator<int>(0),
				thrust::counting_iterator<int>(offsets.size()),
				offsets.begin());
	}

	template <typename IntIterator, typename IntIterator2>
	static void indices_to_offsets(const IntIterator indices_begin, const IntIterator indices_end, IntIterator2 offsets_begin, IntIterator2 offsets_end)
	{
		// convert uncompressed row indices into compressed row offsets
		thrust::lower_bound(indices_begin,
				indices_end,
				thrust::counting_iterator<int>(0),
				thrust::counting_iterator<int>((int)(offsets_end - offsets_begin)),
				offsets_begin);
	}

	template <typename IVector>
	void
	matrix_transform(int *perm)
	{
		IVector row_indices(m_nnz);
		offsets_to_indices(m_row_offsets, m_row_offsets + (m_n + 1), row_indices.begin(), row_indices.end());
		thrust::gather(row_indices.begin(), row_indices.end(), perm, row_indices.begin());
		thrust::gather(m_column_indices, m_column_indices + m_nnz, perm, m_column_indices);
		thrust::sort_by_key(thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), m_column_indices)), 
						    thrust::make_zip_iterator(thrust::make_tuple(row_indices.end(),   m_column_indices + m_nnz)), 
							m_values
				);
		indices_to_offsets(row_indices.begin(), row_indices.end(), m_row_offsets, m_row_offsets + (m_n + 1));
	}

	virtual bool symmetricRCM() = 0;

protected:
	bool   m_l_analyzed;
	bool   m_u_analyzed;
	cusparseSolveAnalysisInfo_t m_infoL;
	cusparseSolveAnalysisInfo_t m_infoU;
	cusparseMatDescr_t          m_descrL;
	cusparseMatDescr_t          m_descrU;

	double m_tolerance;

	static cusparseHandle_t m_handle;
	static cusolverHandle_t m_solver_handle;
	static bool             m_handle_initialized;
	static bool             m_solver_handle_initialized;

};

class CuSparseCsrMatrixH: public CuSparseCsrMatrix_base
{
public:
	CuSparseCsrMatrixH(int N, int nnz) {
		this -> m_n          = N;
		this -> m_nnz        = nnz;
		this -> m_l_analyzed = false;
		this -> m_u_analyzed = false;
		this -> m_tolerance  = TEMP_TOL;

		m_row_offsets    = (int *)malloc(sizeof(int) * (N + 1));
		m_perm           = (int *)malloc(sizeof(int) * N);
		m_reordering     = (int *)malloc(sizeof(int) * N);
		m_column_indices = (int *)malloc(sizeof(int) * nnz);
		m_values         = (double *)malloc(sizeof(double) * nnz);

		thrust::sequence(m_perm, m_perm + N);
		thrust::sequence(m_reordering, m_reordering + N);

		if (!(this->m_handle_initialized)) {
			cusparseCreate(&m_handle);
			this->m_handle_initialized = true;
		}

		if (!(this->m_solver_handle_initialized)) {
			cusolverCreate(&m_solver_handle);
			this->m_solver_handle_initialized = true;
		}

		cusparseCreateSolveAnalysisInfo(&(this->m_infoL));
		cusparseCreateSolveAnalysisInfo(&(this->m_infoU));
		cusparseCreateMatDescr(&(this->m_descrL));
		cusparseCreateMatDescr(&(this->m_descrU));
	}

	CuSparseCsrMatrixH(int N, int nnz, int *row_offsets, int *column_indices, double *values){
		this -> m_n          = N;
		this -> m_nnz        = nnz;
		this -> m_l_analyzed = false;
		this -> m_u_analyzed = false;
		this -> m_tolerance  = TEMP_TOL;

		m_row_offsets    = (int *)malloc(sizeof(int) * (N + 1));
		memcpy(m_row_offsets, row_offsets, sizeof(int) * (N + 1));

		m_column_indices = (int *)malloc(sizeof(int) * nnz);
		memcpy(m_column_indices, column_indices, sizeof(int) * nnz);

		m_values         = (double *)malloc(sizeof(double) * nnz);
		memcpy(m_values, values, sizeof(double) * nnz);

		m_perm           = (int *)malloc(sizeof(int) * N);
		m_reordering     = (int *)malloc(sizeof(int) * N);
		thrust::sequence(m_perm, m_perm + N);
		thrust::sequence(m_reordering, m_reordering + N);

		if (!m_handle_initialized) {
			cusparseCreate(&m_handle);
			m_handle_initialized = true;
		}

		if (!m_solver_handle_initialized) {
			cusolverCreate(&m_solver_handle);
			m_solver_handle_initialized = true;
		}

		cusparseCreateSolveAnalysisInfo(&m_infoL);
		cusparseCreateSolveAnalysisInfo(&m_infoU);
		cusparseCreateMatDescr(&m_descrL);
		cusparseCreateMatDescr(&m_descrU);
	}

	template<typename IntVector, typename DoubleVector>
	CuSparseCsrMatrixH(const IntVector    &row_offsets,
			           const IntVector    &column_indices,
					   const DoubleVector &values) {
		int N   = row_offsets.size() - 1;
		m_n     = N;
		int nnz = column_indices.size();
		m_nnz   = nnz;
		const int *p_row_offsets   = thrust::raw_pointer_cast(&row_offsets[0]);
		const int *p_column_indices = thrust::raw_pointer_cast(&column_indices[0]);
		const double *p_values         = thrust::raw_pointer_cast(&values[0]);

		this->m_l_analyzed = false;
		this->m_u_analyzed = false;
		m_tolerance        = TEMP_TOL;

		if (!m_handle_initialized) {
			cusparseCreate(&m_handle);
			m_handle_initialized = true;
		}

		if (!m_solver_handle_initialized) {
			cusolverCreate(&m_solver_handle);
			m_solver_handle_initialized = true;
		}

		m_row_offsets    = (int *)malloc(sizeof(int) * (N + 1));
		memcpy(m_row_offsets, p_row_offsets, sizeof(int) * (N + 1));

		m_column_indices = (int *)malloc(sizeof(int) * nnz);
		memcpy(m_column_indices, p_column_indices, sizeof(int) * nnz);

		m_values         = (double *)malloc(sizeof(double) * nnz);
		memcpy(m_values, p_values, sizeof(double) * nnz);

		m_perm           = (int *)malloc(sizeof(int) * N);
		m_reordering     = (int *)malloc(sizeof(int) * N);
		thrust::sequence(m_perm, m_perm + N);
		thrust::sequence(m_reordering, m_reordering + N);

		cusparseCreateSolveAnalysisInfo(&m_infoL);
		cusparseCreateSolveAnalysisInfo(&m_infoU);
		cusparseCreateMatDescr(&m_descrL);
		cusparseCreateMatDescr(&m_descrU);
	}

	virtual ~CuSparseCsrMatrixH() {
		if (m_row_offsets)    {free(m_row_offsets); m_row_offsets = 0;}
		if (m_column_indices) {free(m_column_indices); m_column_indices = 0;}
		if (m_values)         {free(m_values); m_values = 0;}
		if (m_perm)           {free(m_perm); m_perm = 0;}
		if (m_reordering)     {free(m_reordering); m_reordering = 0;}

		cusparseDestroySolveAnalysisInfo(m_infoL);
		cusparseDestroySolveAnalysisInfo(m_infoU);
		cusparseDestroyMatDescr(m_descrL);
		cusparseDestroyMatDescr(m_descrU);
		m_infoL = 0;
		m_infoU = 0;
		m_descrL = 0;
		m_descrU = 0;
	}

	virtual bool symmetricRCM();
	
	template<typename DVector>
	cusolverStatus_t QRSolve(const DVector &x, DVector &y)
	{
		if (y.size() != x.size())
			y.resize(x.size());

		const double *p_x = thrust::raw_pointer_cast(&x[0]);
		double *p_y = thrust::raw_pointer_cast(&y[0]);

		cusparseSetMatType(m_descrL,CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatDiagType(m_descrL,CUSPARSE_DIAG_TYPE_NON_UNIT);

		cusparseSetMatIndexBase(m_descrL,CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatFillMode(m_descrL, CUSPARSE_FILL_MODE_LOWER);

		cusolverStatus_t status;

		int singularity;

		status = hsolverDcsrlsvqr(m_solver_handle, m_n, m_nnz, m_descrL, m_values, m_row_offsets, m_column_indices, p_x, m_tolerance, p_y, &singularity);

		return status;
	}
};

bool
CuSparseCsrMatrixH::symmetricRCM()
{
	int is_sym = 0;
	cusparseSetMatType(m_descrL,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatDiagType(m_descrL,CUSPARSE_DIAG_TYPE_NON_UNIT);

	cusparseSetMatIndexBase(m_descrL,CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatFillMode(m_descrL, CUSPARSE_FILL_MODE_LOWER);

	cusolverStatus_t status = hsolverXcsrissym(m_solver_handle, m_n, m_nnz, m_descrL, m_row_offsets, m_row_offsets + 1, m_column_indices, &is_sym);
	if (status != CUSOLVER_STATUS_SUCCESS)
		return false;

	if (!is_sym) {
		// TODO
		return false;
	}

	size_t bufferSizeInBytes;
	hsolverXcsrsymrcm_bufferSize(m_solver_handle, m_n, &bufferSizeInBytes);
	char *pBuffer = 0;
	pBuffer = (char *)malloc(bufferSizeInBytes);

	if (pBuffer == NULL) {
		return false;
	}

	if (is_sym) {
		status = hsolverXcsrsymrcm(m_solver_handle, m_n, m_nnz, m_descrL, m_row_offsets, m_row_offsets + 1, m_column_indices, m_reordering, (void *)pBuffer);
	}
	else {
		// TODO
	}
	free(pBuffer);

	thrust::scatter(thrust::make_counting_iterator<int>(0),
					thrust::make_counting_iterator<int>(m_n),
					m_reordering,
					m_perm);

	if (status != CUSOLVER_STATUS_SUCCESS)
		return false;

	if (is_sym) {
		matrix_transform<typename thrust::host_vector<int> >(m_perm);
	} else {
		// TODO
	}
	return true;
}

class CuSparseCsrMatrixD: public CuSparseCsrMatrix_base
{
public:
	CuSparseCsrMatrixD(int N, int nnz) {
		this -> m_n          = N;
		this -> m_nnz        = nnz;
		this -> m_l_analyzed = false;
		this -> m_u_analyzed = false;
		this -> m_tolerance  = TEMP_TOL;

		cudaMalloc(&m_row_offsets,    sizeof(int) * (N + 1));
		cudaMalloc(&m_column_indices, sizeof(int) * nnz);
		cudaMalloc(&m_values,         sizeof(double) * nnz);
		cudaMalloc(&m_perm,           sizeof(int) * N);
		cudaMalloc(&m_reordering,     sizeof(int) * N);

		if (!(this->m_handle_initialized)) {
			cusparseCreate(&m_handle);
			this->m_handle_initialized = true;
		}

		if (!(this->m_solver_handle_initialized)) {
			cusolverCreate(&m_solver_handle);
			this->m_solver_handle_initialized = true;
		}

		cusparseCreateSolveAnalysisInfo(&(this->m_infoL));
		cusparseCreateSolveAnalysisInfo(&(this->m_infoU));
		cusparseCreateMatDescr(&(this->m_descrL));
		cusparseCreateMatDescr(&(this->m_descrU));
	}

	CuSparseCsrMatrixD(int N, int nnz, int *row_offsets, int *column_indices, double *values){
		this -> m_n          = N;
		this -> m_nnz        = nnz;
		this -> m_l_analyzed = false;
		this -> m_u_analyzed = false;
		this -> m_tolerance  = TEMP_TOL;

		cudaMalloc(&m_row_offsets,   sizeof(int) * (N + 1));
		cudaMemcpy(m_row_offsets, row_offsets, sizeof(int) * (N + 1), cudaMemcpyDeviceToDevice);

		cudaMalloc(&m_column_indices, sizeof(int) * nnz);
		cudaMemcpy(m_column_indices, column_indices, sizeof(int) * nnz, cudaMemcpyDeviceToDevice);

		cudaMalloc(&m_values,         sizeof(double) * nnz);
		cudaMemcpy(m_values, values, sizeof(double) * nnz, cudaMemcpyDeviceToDevice);

		cudaMalloc(&m_perm,           sizeof(int) * N);
		cudaMalloc(&m_reordering,     sizeof(int) * N);

		if (!m_handle_initialized) {
			cusparseCreate(&m_handle);
			m_handle_initialized = true;
		}

		if (!m_solver_handle_initialized) {
			cusolverCreate(&m_solver_handle);
			m_solver_handle_initialized = true;
		}

		cusparseCreateSolveAnalysisInfo(&m_infoL);
		cusparseCreateSolveAnalysisInfo(&m_infoU);
		cusparseCreateMatDescr(&m_descrL);
		cusparseCreateMatDescr(&m_descrU);
	}

	template<typename IntVector, typename DoubleVector>
	CuSparseCsrMatrixD(const IntVector    &row_offsets,
			           const IntVector    &column_indices,
					   const DoubleVector &values) {
		int N   = row_offsets.size() - 1;
		m_n     = N;
		int nnz = column_indices.size();
		m_nnz   = nnz;
		const int *p_row_offsets   = thrust::raw_pointer_cast(&row_offsets[0]);
		const int *p_column_indices = thrust::raw_pointer_cast(&column_indices[0]);
		const double *p_values         = thrust::raw_pointer_cast(&values[0]);

		this->m_l_analyzed = false;
		this->m_u_analyzed = false;
		m_tolerance        = TEMP_TOL;

		if (!m_handle_initialized) {
			cusparseCreate(&m_handle);
			m_handle_initialized = true;
		}

		if (!m_solver_handle_initialized) {
			cusolverCreate(&m_solver_handle);
			m_solver_handle_initialized = true;
		}

		cudaMalloc(&m_row_offsets,   sizeof(int) * (N + 1));
		cudaMemcpy(m_row_offsets, p_row_offsets, sizeof(int) * (N + 1), cudaMemcpyDeviceToDevice);

		cudaMalloc(&m_column_indices, sizeof(int) * nnz);
		cudaMemcpy(m_column_indices, p_column_indices, sizeof(int) * nnz, cudaMemcpyDeviceToDevice);

		cudaMalloc(&m_values,         sizeof(double) * nnz);
		cudaMemcpy(m_values, p_values, sizeof(double) * nnz, cudaMemcpyDeviceToDevice);

		cudaMalloc(&m_perm,           sizeof(int) * N);
		cudaMalloc(&m_reordering,     sizeof(int) * N);

		cusparseCreateSolveAnalysisInfo(&m_infoL);
		cusparseCreateSolveAnalysisInfo(&m_infoU);
		cusparseCreateMatDescr(&m_descrL);
		cusparseCreateMatDescr(&m_descrU);
	}

	virtual ~CuSparseCsrMatrixD() {
		if (m_row_offsets)    {cudaFree(m_row_offsets); m_row_offsets = 0;}
		if (m_column_indices) {cudaFree(m_column_indices); m_column_indices = 0;}
		if (m_values)         {cudaFree(m_values); m_values = 0;}
		if (m_reordering)     {cudaFree(m_reordering); m_reordering = 0;}
		if (m_perm)           {cudaFree(m_perm); m_perm = 0;}

		cusparseDestroySolveAnalysisInfo(m_infoL);
		cusparseDestroySolveAnalysisInfo(m_infoU);
		cusparseDestroyMatDescr(m_descrL);
		cusparseDestroyMatDescr(m_descrU);
		m_infoL = 0;
		m_infoU = 0;
		m_descrL = 0;
		m_descrU = 0;
	}

	template<typename DVector>
	cusparseStatus_t spmv(const DVector &x, DVector &y)
	{
		if (y.size() != x.size())
			y.resize(x.size());

		double one = 1.0, zero = 0.0;
		const double *p_x = thrust::raw_pointer_cast(&x[0]);
		double *p_y = thrust::raw_pointer_cast(&y[0]);

		cusparseSetMatType(m_descrL,CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatDiagType(m_descrL,CUSPARSE_DIAG_TYPE_NON_UNIT);
		cusparseSetMatIndexBase(m_descrL,CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatFillMode(m_descrL, CUSPARSE_FILL_MODE_LOWER);

		return cusparseDcsrmv(m_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m_n, m_n, m_nnz, &one, m_descrL, m_values, m_row_offsets, m_column_indices, p_x, &zero, p_y);
	}

	template<typename DVector>
	cusparseStatus_t forwardSolve(const DVector &x, DVector &y, bool unit = true)
	{
		if (y.size() != x.size())
			y.resize(x.size());

		double one = 1.0;
		const double *p_x = thrust::raw_pointer_cast(&x[0]);
		double *p_y = thrust::raw_pointer_cast(&y[0]);

		cusparseSetMatType(m_descrL,CUSPARSE_MATRIX_TYPE_GENERAL);
		if (unit)
			cusparseSetMatDiagType(m_descrL,CUSPARSE_DIAG_TYPE_UNIT);
		else
			cusparseSetMatDiagType(m_descrL,CUSPARSE_DIAG_TYPE_NON_UNIT);

		cusparseSetMatIndexBase(m_descrL,CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatFillMode(m_descrL, CUSPARSE_FILL_MODE_LOWER);

		cusparseStatus_t status;

		if (!m_l_analyzed) {
			status = cusparseDcsrsv_analysis(m_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m_n, m_nnz, m_descrL, m_values, m_row_offsets, m_column_indices, m_infoL);
			if (status != CUSPARSE_STATUS_SUCCESS)
				return status;
		}

		status = cusparseDcsrsv_solve(m_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m_n, &one, m_descrL, m_values, m_row_offsets, m_column_indices, m_infoL, p_x, p_y);

		return status;
	}

	template<typename DVector>
	cusparseStatus_t backwardSolve(const DVector &x, DVector &y, bool unit = false)
	{
		if (y.size() != x.size())
			y.resize(x.size());

		double one = 1.0;
		const double *p_x = thrust::raw_pointer_cast(&x[0]);
		double *p_y = thrust::raw_pointer_cast(&y[0]);

		cusparseSetMatType(m_descrU,CUSPARSE_MATRIX_TYPE_GENERAL);
		if (unit)
			cusparseSetMatDiagType(m_descrU,CUSPARSE_DIAG_TYPE_UNIT);
		else
			cusparseSetMatDiagType(m_descrU,CUSPARSE_DIAG_TYPE_NON_UNIT);

		cusparseSetMatIndexBase(m_descrU,CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatFillMode(m_descrU, CUSPARSE_FILL_MODE_UPPER);

		cusparseStatus_t status;

		if (!m_u_analyzed) {
			status = cusparseDcsrsv_analysis(m_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m_n, m_nnz, m_descrU, m_values, m_row_offsets, m_column_indices, m_infoU);
			if (status != CUSPARSE_STATUS_SUCCESS)
				return status;
		}

		status = cusparseDcsrsv_solve(m_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m_n, &one, m_descrU, m_values, m_row_offsets, m_column_indices, m_infoU, p_x, p_y);

		return status;
	}

	template<typename DVector>
	cusolverStatus_t QRSolve(const DVector &x, DVector &y)
	{
		if (y.size() != x.size())
			y.resize(x.size());

		const double *p_x = thrust::raw_pointer_cast(&x[0]);
		double *p_y = thrust::raw_pointer_cast(&y[0]);

		cusparseSetMatType(m_descrL,CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatDiagType(m_descrL,CUSPARSE_DIAG_TYPE_NON_UNIT);

		cusparseSetMatIndexBase(m_descrL,CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatFillMode(m_descrL, CUSPARSE_FILL_MODE_LOWER);

		cusolverStatus_t status;

		int singularity;

		status = cusolverDcsrlsvqr(m_solver_handle, m_n, m_nnz, m_descrL, m_values, m_row_offsets, m_column_indices, p_x, m_tolerance, p_y, &singularity);

		return status;
	}

	virtual bool symmetricRCM();
};

bool
CuSparseCsrMatrixD::symmetricRCM()
{
	int is_sym = 0;
	cusparseSetMatType(m_descrL,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatDiagType(m_descrL,CUSPARSE_DIAG_TYPE_NON_UNIT);

	cusparseSetMatIndexBase(m_descrL,CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatFillMode(m_descrL, CUSPARSE_FILL_MODE_LOWER);

	// Transfer to host to do RCM
	thrust::host_vector<int> h_row_offsets(m_n + 1);
	thrust::host_vector<int> h_column_indices(m_nnz);

	int *p_row_offsets    = thrust::raw_pointer_cast(&h_row_offsets[0]);
	int *p_column_indices = thrust::raw_pointer_cast(&h_column_indices[0]);

	cudaMemcpy(p_row_offsets, m_row_offsets, sizeof(int) * (m_n + 1), cudaMemcpyDeviceToHost);
	cudaMemcpy(p_column_indices, m_column_indices, sizeof(int) * (m_nnz), cudaMemcpyDeviceToHost);

	cusolverStatus_t status = hsolverXcsrissym(m_solver_handle, m_n, m_nnz, m_descrL, p_row_offsets, p_row_offsets + 1, p_column_indices, &is_sym);
	if (status != CUSOLVER_STATUS_SUCCESS)
		return false;

	if (!is_sym) {
		// TODO
		return false;
	}
	thrust::host_vector<double> h_values(m_nnz);
	thrust::host_vector<int> h_reordering(m_n);
	thrust::host_vector<int> h_perm(m_n);
	int *p_reordering     = thrust::raw_pointer_cast(&h_reordering[0]);
	int *p_perm           = thrust::raw_pointer_cast(&h_perm[0]);
	double *p_values      = thrust::raw_pointer_cast(&h_values[0]);
	cudaMemcpy(p_values, m_values, sizeof(double) * (m_nnz), cudaMemcpyDeviceToHost);

	size_t bufferSizeInBytes;
	hsolverXcsrsymrcm_bufferSize(m_solver_handle, m_n, &bufferSizeInBytes);
	char *pBuffer = 0;
	pBuffer = (char *)malloc(bufferSizeInBytes);

	if (pBuffer == NULL) {
		return false;
	}

	if (is_sym) {
		status = hsolverXcsrsymrcm(m_solver_handle, m_n, m_nnz, m_descrL, p_row_offsets, p_row_offsets + 1, p_column_indices, p_reordering, (void *)pBuffer);
	}
	else {
		// TODO
	}
	free(pBuffer);

	cudaMemcpy(m_reordering, p_reordering, sizeof(int) * m_n, cudaMemcpyHostToDevice);

	thrust::scatter(thrust::make_counting_iterator<int>(0),
					thrust::make_counting_iterator<int>(m_n),
					p_reordering,
					p_perm);
	
	cudaMemcpy(m_perm, p_perm, sizeof(int) * m_n, cudaMemcpyHostToDevice);

	if (status != CUSOLVER_STATUS_SUCCESS)
		return false;

	if (is_sym) {
		thrust::host_vector<int> row_indices(m_nnz);
		offsets_to_indices(p_row_offsets, p_row_offsets + (m_n + 1), row_indices.begin(), row_indices.end());
		thrust::gather(row_indices.begin(), row_indices.end(), p_perm, row_indices.begin());
		thrust::gather(p_column_indices, p_column_indices + m_nnz, p_perm, p_column_indices);
		thrust::sort_by_key(thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), p_column_indices)), 
						    thrust::make_zip_iterator(thrust::make_tuple(row_indices.end(),   p_column_indices + m_nnz)), 
							h_values.begin()
				);
		indices_to_offsets(row_indices.begin(), row_indices.end(), p_row_offsets, p_row_offsets + (m_n + 1));

		cudaMemcpy(m_row_offsets, p_row_offsets, sizeof(int) * (m_n + 1), cudaMemcpyHostToDevice);
		cudaMemcpy(m_column_indices, p_column_indices, sizeof(int) * (m_nnz), cudaMemcpyHostToDevice);
		cudaMemcpy(m_values, p_values, sizeof(double) * (m_nnz), cudaMemcpyHostToDevice);
	} else {
		// TODO
	}
	return true;
}

cusparseHandle_t CuSparseCsrMatrix_base::m_handle = 0;
bool             CuSparseCsrMatrix_base::m_handle_initialized = false;
cusolverHandle_t CuSparseCsrMatrix_base::m_solver_handle = 0;
bool             CuSparseCsrMatrix_base::m_solver_handle_initialized = false;


} // namespace cusparse


#endif
