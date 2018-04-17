#ifndef CAFFE_UPSAMPLE_LAYER_HPP_
#define CAFFE_UPSAMPLE_LAYER_HPP_

#include <vector>
 
#include "glog/logging.h"
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>  // cuda driver types


// CUDA: use 512 threads per block
const int CAFFE_CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " \
      << caffe::cublasGetErrorString(status); \
  } while (0)

#define CURAND_CHECK(condition) \
  do { \
    curandStatus_t status = condition; \
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << " " \
      << caffe::curandGetErrorString(status); \
  } while (0)

 

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())



 #define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

 
 __global__ void ReLUForward(const int n, const float* in, float* out,
    float negative_slope);
void Forward_gpu(
     float* input,int b,int c,int w,int h,int stride_, float* output );

 
void  ReluForward_gpu(const float*  bottom,
     float*  top,int b,int c,int w,int h,float  negative_slope);


#endif  // CAFFE_UPSAMPLE_LAYER_HPP_
