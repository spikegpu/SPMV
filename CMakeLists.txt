CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(spmv)
INCLUDE(cmake/SBELUtils.cmake)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

enable_cuda_support()
enable_openmp_support()

find_path(VIENNA_CL_DIR NAMES viennacl/vector.hpp)
if(VIENNA_CL_DIR)
	message(STATUS "Vienna CL headers found in ${VIENNA_CL_DIR}")
   #else()
   #message(FATAL_ERROR "CUSP headers not found. Please set CUSP_INCLUDE_DIR.")
endif()

find_path(CUSP_INCLUDE_DIR 
          NAMES cusp/array1d.h 
          HINTS ${CUDA_TOOLKIT_INCLUDE} 
          DOC "Top-level CUSP directory")

if(CUSP_INCLUDE_DIR)
   message(STATUS "CUSP headers found in ${CUSP_INCLUDE_DIR}/cusp")
   #else()
   #message(FATAL_ERROR "CUSP headers not found. Please set CUSP_INCLUDE_DIR.")
endif()
include_directories(${VIENNA_CL_DIR})
include_directories(.)
# include_directories(..)
cuda_include_directories(${CUSP_INCLUDE_DIR})

MESSAGE(STATUS "Cuda version: ${CUDA_VERSION}")

SET(SPMV_HEADERS
	src/timer.h
	src/cusparse_wrapper.h
  )

SET(MMIO_FILES
  mm_io/mm_io.h
  mm_io/mm_io.cu
  )

SOURCE_GROUP("Headers" FILES ${SPMV_HEADERS})
SOURCE_GROUP("MM_IO" FILES ${MMIO_FILES})

INCLUDE_DIRECTORIES(
    ${CMAKE_SOURCE_DIR}
    )

#cuda_add_executable(driver_opencl driver_opencl.cpp ${SPMV_HEADERS} ${MMIO_FILES})
cuda_add_executable(driver_cusp_csr driver_cusp_csr.cu ${SPMV_HEADERS})
cuda_add_executable(driver_cusp_coo driver_cusp_coo.cu ${SPMV_HEADERS})
cuda_add_executable(driver_cusp_ell driver_cusp_ell.cu ${SPMV_HEADERS})
cuda_add_executable(driver_cusp_hyb driver_cusp_hyb.cu ${SPMV_HEADERS})
cuda_add_executable(driver_cusparse driver_cusparse.cu ${SPMV_HEADERS})
cuda_add_executable(driver_viennacl_csr driver_viennacl_csr.cpp ${SPMV_HEADERS} ${MMIO_FILES})
cuda_add_executable(driver_viennacl_coo driver_viennacl_coo.cpp ${SPMV_HEADERS} ${MMIO_FILES})
cuda_add_executable(driver_viennacl_ell driver_viennacl_ell.cpp ${SPMV_HEADERS} ${MMIO_FILES})
cuda_add_executable(driver_viennacl_hyb driver_viennacl_hyb.cpp ${SPMV_HEADERS} ${MMIO_FILES})
#target_link_libraries(driver_opencl OpenCL)
target_link_libraries(driver_viennacl_csr OpenCL)
target_link_libraries(driver_viennacl_coo OpenCL)
target_link_libraries(driver_viennacl_ell OpenCL)
target_link_libraries(driver_viennacl_hyb  OpenCL)
target_link_libraries(driver_cusparse cusparse)
