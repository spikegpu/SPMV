/** \file timer.h
 *  \brief CPU and GPU timer classes.
 */

#ifndef TIMER_H
#define TIMER_H

#include "CL/cl.h"

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#endif



/// Base timer class.
class Timer {
public:
	virtual ~Timer() {}
	virtual void Start()=0;
	virtual void Stop()=0;
	virtual double getElapsed()=0;
};

/// GPU timer.
/**
 * CUDA-based GPU timer.
 */
class CUDATimer : public Timer {
protected:
	int gpu_idx;
	cudaEvent_t timeStart;
	cudaEvent_t timeEnd;
public:
	CUDATimer(int g_idx = 0) {
		gpu_idx = g_idx;

		cudaEventCreate(&timeStart);
		cudaEventCreate(&timeEnd);
	}

	virtual ~CUDATimer() {
		cudaEventDestroy(timeStart);
		cudaEventDestroy(timeEnd);
	}

	virtual void Start() {
		cudaEventRecord(timeStart, 0);
	}

	virtual void Stop() {
		cudaEventRecord(timeEnd, 0);
		cudaEventSynchronize(timeEnd);
	}

	virtual double getElapsed() {
		float elapsed;
		cudaEventElapsedTime(&elapsed, timeStart, timeEnd);
		return elapsed;
	}
};


/// GPU timer.
/**
 * OpenCL-based GPU timer.
 */
class OpenCLTimer : public Timer {
protected:
	int gpu_idx;
	cl_ulong timeStart;
	cl_ulong timeEnd;
	cl_event event;
public:
	OpenCLTimer(int g_idx = 0) {
		gpu_idx = g_idx;
	}

	cl_event *getEvent() {return &event;}

	virtual ~OpenCLTimer() {
	}

	virtual void Start() {
		clWaitForEvents(1, &event);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(timeStart), &timeStart, NULL);
	}

	virtual void Stop() {
		clWaitForEvents(1, &event);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(timeEnd), &timeEnd, NULL);
	}

	virtual double getElapsed() {
		return double(timeEnd - timeStart) / 1e6;
	}
};


/// CPU timer.
/**
 * CPU timer using the performance counter for WIN32 and
 * gettimeofday() for Linux.
 */
#ifdef WIN32

class CPUTimer : public Timer
{
public:
	CPUTimer()
	{
		QueryPerformanceFrequency(&m_frequency);
	}
	~CPUTimer()  {}

	virtual void Start() {QueryPerformanceCounter(&m_start);}
	virtual void Stop()  {QueryPerformanceCounter(&m_stop);}

	virtual double getElapsed() {
		return (m_stop.QuadPart - m_start.QuadPart) * 1000.0 / m_frequency.QuadPart;
	}

private:
	LARGE_INTEGER m_frequency;
	LARGE_INTEGER m_start;
	LARGE_INTEGER m_stop;
};

#else // WIN32

class CPUTimer : public Timer {
protected:
	timeval timeStart;
	timeval timeEnd;
public:
	virtual ~CPUTimer() {}

	virtual void Start() {
		gettimeofday(&timeStart, 0);
	}

	virtual void Stop() {
		gettimeofday(&timeEnd, 0);
	}

	virtual double getElapsed() {
		return 1000.0 * (timeEnd.tv_sec - timeStart.tv_sec) + (timeEnd.tv_usec - timeStart.tv_usec) / 1000.0;
	}
};

#endif // WIN32


#endif
