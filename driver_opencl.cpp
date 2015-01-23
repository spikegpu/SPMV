#include <iostream>
#include <cstdio>
#include <string>
#include <stdlib.h>
#include <cstring>
#include <cmath>
#include <src/timer.h>

extern "C" {
#include "mm_io/mm_io.h"
}

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include "CL/cl.h"

#define DATA_SIZE 10

static const int BLOCK_SIZE = 128;

typedef double REAL;

using namespace std;

const char *KernelSource = 
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"\
"__kernel void hello(__global double *input, __global double* output)\n"\
"{\n"\
"    size_t id = get_global_id(0);\n"\
"    output[id] = input[id] * input[id];\n"\
"}\n"\
"\n";

const char *cl_source_spmv = 
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"\
"__kernel void\n"\
"spmv_csr_scalar_kernel( __global const double * restrict val,\n"\
"#ifdef USE_TEXTURE\n"\
"	                        image2d_t vec,\n"\
"#else\n"\
"       __global const double * restrict vec,\n"\
"#endif\n"\
"       __global const int * restrict cols,\n"\
"       __global const int * restrict rowDelimiters,\n"\
"       const int dim, __global double * restrict out)\n"\
"{\n"\
"	int myRow = get_global_id(0);\n"\
"	if (myRow < dim)\n"\
"   {\n"\
"       double t=0;\n"\
"		int start = rowDelimiters[myRow];\n"\
"		int end = rowDelimiters[myRow+1];\n"\
"		for (int j = start; j < end; j++)\n"\
"		{\n"\
"			int col = cols[j];\n"\
"#ifdef USE_TEXTURE\n"\
"	        t += val[j] * texFetch(vec,col);\n"\
"#else\n"\
"			t += val[j] * vec[col];\n"\
"#endif\n"\
"		}\n"\
"		out[myRow] = t;\n"\
"   }\n"\
"}\n"\
"\n";

template <typename floatType, typename clFloatType, bool devSupportsImages>
void csrTest(cl_device_id dev, cl_context ctx, string compileFlags,
		cl_command_queue queue,
		floatType* h_val, int* h_cols, int* h_rowDelimiters,
		floatType* h_vec, floatType* h_out, int numRows, int numNonZeroes,
		floatType* refOut, bool padded, const size_t maxImgWidth)
{
	// Set up OpenCL Program Object
	int err = 0;

	if (devSupportsImages)
	{
		char texflags[64] = {0};
		sprintf(texflags," -DUSE_TEXTURE -DMAX_IMG_WIDTH=%ld", maxImgWidth);
		compileFlags+=string(texflags);
	}

	cl_program prog = clCreateProgramWithSource(ctx, 1, &cl_source_spmv, NULL,
			&err);

	if (err != CL_SUCCESS) {
		return;
	}

	// Build the openCL kernels
	err = clBuildProgram(prog, 1, &dev, compileFlags.c_str(), NULL, NULL);
	// If there is a build error, print the output and return
	if (err != CL_SUCCESS)
	{
		char log[5000];
		size_t retsize = 0;
		err = clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 5000
				* sizeof(char), log, &retsize);
		cerr << "Retsize: " << retsize << endl;
		cerr << "Log: " << log << endl;
		return;
	}

	// If there is a build error, print the output and return
	if (err != CL_SUCCESS)
	{
		char log[5000];
		size_t retsize = 0;
		err = clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 5000
				* sizeof(char), log, &retsize);
		cerr << "Retsize: " << retsize << endl;
		cerr << "Log: " << log << endl;
		return;
	}

	// Device data structures
	cl_mem d_val, d_vec, d_out;
	cl_mem d_cols, d_rowDelimiters;
	// Allocate device memory
	d_val = clCreateBuffer(ctx, CL_MEM_READ_WRITE, numNonZeroes *
			sizeof(clFloatType), NULL, &err);
	d_cols = clCreateBuffer(ctx, CL_MEM_READ_WRITE, numNonZeroes *
			sizeof(cl_int), NULL, &err);
	int imgHeight = 0;

	if (devSupportsImages)
	{
		imgHeight=(numRows+maxImgWidth-1)/maxImgWidth;
		cl_image_format fmt; 
		if(sizeof(floatType)==4) {
			fmt.image_channel_order=CL_R;
			fmt.image_channel_data_type = CL_FLOAT;
		}
		else {
			fmt.image_channel_order=CL_RG;
			fmt.image_channel_data_type = CL_FLOAT;
		}
		d_vec = clCreateImage2D( ctx, CL_MEM_READ_ONLY, &fmt, maxImgWidth,
				imgHeight, 0, NULL, &err);
	} else {
		d_vec = clCreateBuffer(ctx, CL_MEM_READ_WRITE, numRows *
				sizeof(clFloatType), NULL, &err);
	}
	
	d_out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, numRows *
			sizeof(clFloatType), NULL, &err);
	d_rowDelimiters = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (numRows+1) *
			sizeof(cl_int), NULL, &err);

	// Transfer data to device
	err = clEnqueueWriteBuffer(queue, d_val, true, 0, numNonZeroes *
			sizeof(floatType), h_val, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		return ;
	}
	err = clEnqueueWriteBuffer(queue, d_cols, true, 0, numNonZeroes *
			sizeof(int), h_cols, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		return ;
	}

	//// err = clEnqueueWriteBuffer(queue, d_vec, true, 0, numRows *
			//// sizeof(floatType), h_vec, 0, NULL, NULL);
	if (devSupportsImages)
	{
		size_t offset[3]={0};
		size_t size[3]={maxImgWidth,(size_t)imgHeight,1};
		err = clEnqueueWriteImage(queue,d_vec, true, offset, size,
				0, 0, h_vec, 0, NULL, NULL);
	} else
	{
		err = clEnqueueWriteBuffer(queue, d_vec, true, 0, numRows *
				sizeof(floatType), h_vec, 0, NULL, NULL);
	}
	if (err != CL_SUCCESS) {
		return ;
	}

	err = clEnqueueWriteBuffer(queue, d_rowDelimiters, true, 0, (numRows+1) *
			sizeof(int), h_rowDelimiters, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		return ;
	}
	err = clFinish(queue);
	if (err != CL_SUCCESS) {
		return ;
	}

	int passes = 2;
	int iters  = 5;

	// Set up CSR Kernels
	cl_kernel csrScalar, csrVector;
	csrScalar  = clCreateKernel(prog, "spmv_csr_scalar_kernel", &err);
	if (err != CL_SUCCESS) {
		return;
	}
	err = clSetKernelArg(csrScalar, 0, sizeof(cl_mem), (void*) &d_val);
	if (err != CL_SUCCESS) {
		return;
	}
	err = clSetKernelArg(csrScalar, 1, sizeof(cl_mem), (void*) &d_vec);
	if (err != CL_SUCCESS) {
		return;
	}
	err = clSetKernelArg(csrScalar, 2, sizeof(cl_mem), (void*) &d_cols);
	if (err != CL_SUCCESS) {
		return;
	}
	err = clSetKernelArg(csrScalar, 3, sizeof(cl_mem),
			(void*) &d_rowDelimiters);
	if (err != CL_SUCCESS) {
		return;
	}
	err = clSetKernelArg(csrScalar, 4, sizeof(cl_int), (void*) &numRows);
	if (err != CL_SUCCESS) {
		return;
	}
	err = clSetKernelArg(csrScalar, 5, sizeof(cl_mem), (void*) &d_out);
	if (err != CL_SUCCESS) {
		return;
	}

	// Append correct suffix to resultsDB entry
	string suffix;
	if (sizeof(floatType) == sizeof(float))
	{
		suffix = "-SP";
	}
	else
	{
		suffix = "-DP";
	}

	const size_t scalarGlobalWSize = numRows;
	size_t localWorkSize = BLOCK_SIZE;

	cl_event event;

	cl_ulong timeStart;
	cl_ulong timeEnd;

	OpenCLTimer timer;

	int counter = 0;
	double elapsed = 0.0;
	for (int k = 0; k < passes; k++)
	{
		// Run Scalar Kernel
		for (int j = 0; j < iters; j++)
		{
			err = clEnqueueNDRangeKernel(queue, csrScalar, 1, NULL,
					&scalarGlobalWSize, NULL, 0, NULL,
					timer.getEvent());
			timer.Start();
			timer.Stop();
			if (err != CL_SUCCESS) {
				exit(-1);
			}

			if (!(k == 0 && j == 0)) {
				counter ++;
				elapsed += timer.getElapsed();
			}

			err = clFinish(queue);
		}

		// Transfer data back to host
		err = clEnqueueReadBuffer(queue, d_out, true, 0, numRows *
				sizeof(floatType), h_out, 0, NULL, NULL);
		err = clFinish(queue);
	}

	elapsed /= counter;
	cout << "OpenCL: " << elapsed << endl;

	int diff_cnt = 0;
	for (int i = 0; i < numRows; i++) {
		if (refOut[i] != 0) {
			if (! (fabs(h_out[i] - refOut[i]) / fabs(refOut[i]) < 1e-10 ))
				diff_cnt ++;
		} else {
			if (! (fabs(h_out[i]) < 1e-10))
				diff_cnt ++;
		}
	}

	// Clobber correct answer, so we can be sure the vector kernel is correct
	err = clEnqueueWriteBuffer(queue, d_out, true, 0, numRows *
			sizeof(floatType), h_vec, 0, NULL, NULL);


	// Free device memory
	err = clReleaseMemObject(d_rowDelimiters);
	err = clReleaseMemObject(d_vec);
	err = clReleaseMemObject(d_out);
	err = clReleaseMemObject(d_val);
	err = clReleaseMemObject(d_cols);
	err = clReleaseKernel(csrScalar);
	err = clReleaseProgram(prog);
}


int main(int argc, char **argv)
{
	cl_context            context;
	cl_context_properties properties[3];
	cl_kernel             kernel;
	cl_command_queue      command_queue;
	cl_program            program;
	cl_int                err;
	cl_uint               num_of_platforms = 0;
	cl_platform_id        platform_id;
	cl_device_id          device_id;
	cl_uint               num_of_devices = 0;
	cl_mem                input, output;
	size_t                global;

	double                inputData[DATA_SIZE] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f};
	double                results[DATA_SIZE]   = {0.f};

	int                   i;

	if (clGetPlatformIDs(1, &platform_id, &num_of_platforms) != CL_SUCCESS) {
		cerr << "Unable to get platform ID" << endl;
		return 1;
	}

	if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_of_devices) != CL_SUCCESS)
	{
		cerr << "Unable to get device ID" << endl;
		return 1;
	}

	properties[0] = CL_CONTEXT_PLATFORM;
	properties[1] = (cl_context_properties) platform_id;
	properties[2] = 0;

	context = clCreateContext(properties, 1, &device_id, NULL, NULL, &err);

	command_queue = clCreateCommandQueue(context, device_id, 0, &err);

	program = clCreateProgramWithSource(context, 1, (const char **)&KernelSource, NULL, &err);

	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	if (err != CL_SUCCESS) {
		cerr << "Error building program" << endl;
		cerr << "Error code: " << err << endl;
		return 1;
	}

	kernel = clCreateKernel(program, "hello", &err);

	input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * DATA_SIZE, NULL, NULL);
	output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * DATA_SIZE, NULL, NULL);

	clEnqueueWriteBuffer(command_queue, input, CL_TRUE, 0, sizeof(double) * DATA_SIZE, inputData, 0, NULL, NULL);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
	global = DATA_SIZE;

	clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
	clFinish(command_queue);

	clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0, sizeof(double) * DATA_SIZE, results, 0, NULL, NULL);

	int M, N, NNZ;

	double *h_vals_tmp = NULL;
	int    *h_cols_tmp = NULL, *h_rows_tmp = NULL;

	REAL   *h_vals = NULL;
	int    *h_cols = NULL, *h_rowDelimiters = NULL;
	MM_typecode matcode;

	string mat_file = "/home/ali/CUDA_project/reordering/matrices/ancfBigDan.mtx";
	if (argc > 1)
		mat_file = argv[1];

	cout << mat_file << endl;

	char mat_file_name[303] = {0};
	strncpy(mat_file_name, mat_file.c_str(), mat_file.size());

	err = mm_read_mtx_crd(mat_file_name, &M, &N, &NNZ, &h_rows_tmp, &h_cols_tmp, &h_vals_tmp, &matcode);

	if (err != 0) {
		cerr << "error: " << err << endl;
		return 1;
	}

	for (int i = 0; i < NNZ; i++) {
		h_rows_tmp[i] --;
		h_cols_tmp[i] --;
	}

	h_rowDelimiters = new int [N+1];
	h_cols          = new int [NNZ];
	h_vals          = new REAL [NNZ];

	{
		for (int i = 0; i <= N; i++)
			h_rowDelimiters[i] = 0;

		for (int i = 0; i < NNZ; i++)
			h_rowDelimiters[h_rows_tmp[i]] ++;

		for (int i = 0; i < N; i++)
			h_rowDelimiters[i + 1] += h_rowDelimiters[i];

		if (h_vals_tmp != NULL) {
			for (int i = NNZ - 1; i >= 0; i--) {
				int row  = h_rows_tmp[i];
				int dest = h_rowDelimiters[row];

				h_cols[dest] = h_cols_tmp[i];
				h_vals[dest] = h_vals_tmp[i];

				h_rowDelimiters[row] --;
			}

			delete [] h_vals_tmp;
		} else {
			for (int i = NNZ - 1; i >= 0; i--) {
				int row  = h_rows_tmp[i];
				int dest = h_rowDelimiters[row];

				h_cols[dest] = h_cols_tmp[i];
				h_vals[dest] = 1.0;

				h_rowDelimiters[row] --;
			}
		}
		delete [] h_cols_tmp;
		delete [] h_rows_tmp;
	}

	REAL *h_vec = new REAL [N];
	REAL *h_out = new REAL [N];
	REAL *ref_out = new REAL [N];

	for (int i = 0; i < N; i++)
		h_vec[i] = 20.0 + .01 * (rand() % 5);

	CPUTimer loc_timer;
	loc_timer.Start();
	for (int i = 0; i < N; i++) {
		REAL tmp = REAL(0);
		int start_idx = h_rowDelimiters[i], end_idx = h_rowDelimiters[i+1];
		for (int j = start_idx; j < end_idx; j++)
			tmp += h_vals[j] * h_vec[h_cols[j]];

		ref_out[i] = tmp;
	}
	loc_timer.Stop();

	string compileFlags = "";

	clReleaseCommandQueue(command_queue);
	command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);

	csrTest<REAL, REAL, false>(device_id, context, compileFlags, command_queue, h_vals, h_cols, h_rowDelimiters,
			h_vec, h_out, N, NNZ,
			ref_out, false, 128);

	delete [] h_vals;
	delete [] h_cols;
	delete [] h_rowDelimiters;
	delete [] h_vec;
	delete [] h_out;
	delete [] ref_out;

	clReleaseMemObject(input);
	clReleaseMemObject(output);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	return 0;
}
