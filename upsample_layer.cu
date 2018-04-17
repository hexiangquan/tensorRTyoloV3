#include <vector>
#include "upsample_layer.h"
#include <stdio.h>
 
template <typename Dtype>
__global__ void upsample_kernel(size_t N, const Dtype *x, int w, int h, int c, int batch, int stride, int forward, float scale, Dtype *out)
{
#if 1
        // printf("ReLUForwardsss  01 enqueue  line=%d",__LINE__);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (N); \
       i += blockDim.x * gridDim.x)
       {
          
	    int out_index = i;
	    int out_w = i%(w*stride);
	    i = i/(w*stride);
	    int out_h = i%(h*stride);
	    i = i/(h*stride);
	    int out_c = i%c;
	    i = i/c;
	    int b = i%batch;

	    int in_w = out_w / stride;
	    int in_h = out_h / stride;
	    int in_c = out_c;

	    int in_index = b*w*h*c + in_c*w*h + in_h*w + in_w;
	 
	    // out[out_index] += scale * x[in_index];
	 
	     out[out_index]  =  scale*x[in_index];//坑死了

   }



 
//}



#else
 
#endif
/*
    size_t i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int out_index = i;
    int out_w = i%(w*stride);
    i = i/(w*stride);
    int out_h = i%(h*stride);
    i = i/(h*stride);
    int out_c = i%c;
    i = i/c;
    int b = i%batch;

    int in_w = out_w / stride;
    int in_h = out_h / stride;
    int in_c = out_c;

    int in_index = b*w*h*c + in_c*w*h + in_h*w + in_w;


     out[out_index] += scale * x[in_index];

*/
 //   else atomicAdd(x+in_index, scale * out[out_index]);
}


 
__global__ void ReLUForward(const int n, const float* in, float* out,
    float negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
      //  printf("ReLUForwardsss  01 enqueue  line=%d",__LINE__);
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  }
}



 
void  ReluForward_gpu(const float*  bottom,
     float*  top,int b,int c,int w,int h,float negative_slope) {
  	 const int nthreads =b*c*w*h;  
  // NOLINT_NEXT_LINE(whitespace/operators)
     //   printf("ReLUForwardsss  01 enqueue  line=%d  threads:%d  %d %d %d %d \n",__LINE__,nthreads,b,c,w,h);
  ReLUForward <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      nthreads, bottom, top, negative_slope);
   CUDA_POST_KERNEL_CHECK;
   
}



 
void Forward_gpu(
     float* input,int b,int c,int w,int h,int stride_, float* output ) {
 
	//const Dtype* bottom_data = bottom[0]->gpu_data();
	//Dtype* top_data = top[0]->mutable_gpu_data();
	//const int count = top[0]->count();
	 const int nthreads =b*c*w*h*stride_*stride_;  

	//vector<int> buttom_shape = bottom[0]->shape();

      //  printf("ReLUForwardsss  01 enqueue  line=%d",__LINE__);
	 upsample_kernel<float><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>( nthreads, input,h,  w, c,b,stride_, 1,1, output);
	 CUDA_POST_KERNEL_CHECK;

 
}

 

 
 
