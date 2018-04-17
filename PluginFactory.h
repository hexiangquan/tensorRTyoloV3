#ifndef __PLUGIN_FACTORY_H__  
#define __PLUGIN_FACTORY_H__  
  
#include <algorithm>  
#include <cassert>  
#include <iostream>  
#include <cstring>  
#include <sys/stat.h>  
#include <map>  
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "upsample_layer.h"


using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;
#if 0
//Concat layer . TensorRT Concat only support cross channel
class ConcatPlugin : public IPlugin
{
public:
    ConcatPlugin(int axis){ _axis = axis; };
  //  ConcatPlugin(int axis, const void* buffer, size_t size);
	ConcatPlugin::ConcatPlugin(int axis, const void* buffer, size_t size)
	{
	    assert(size == (18*sizeof(int)));
	    const int* d = reinterpret_cast<const int*>(buffer);

	    dimsConv4_3 = DimsCHW{d[0], d[1], d[2]};
	    dimsFc7 = DimsCHW{d[3], d[4], d[5]};
	    dimsConv6 = DimsCHW{d[6], d[7], d[8]};
	    dimsConv7 = DimsCHW{d[9], d[10], d[11]};
	    dimsConv8 = DimsCHW{d[12], d[13], d[14]};
	    dimsConv9 = DimsCHW{d[15], d[16], d[17]};

	    _axis = axis;

	}


    inline int getNbOutputs() const override {return 1;};
  //  Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override ;

	Dims  getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
	{
	    assert(nbInputDims == 6);

	    if(_axis == 1)
	    {
		top_concat_axis = inputs[0].d[0] + inputs[1].d[0] + inputs[2].d[0] + inputs[3].d[0] + inputs[4].d[0] + inputs[5].d[0];
		return DimsCHW(top_concat_axis, 1, 1);
	    }else if(_axis == 2){
		top_concat_axis = inputs[0].d[1] + inputs[1].d[1] + inputs[2].d[1] + inputs[3].d[1] + inputs[4].d[1] + inputs[5].d[1];
		return DimsCHW(2, top_concat_axis, 1);
	    }else{//_param.concat_axis == 3

		return DimsCHW(0, 0, 0);
	    }
	}


  //  int initialize() 


	int  initialize()  
	{
	    inputs_size = 6;//6个bottom层

	    if(_axis == 1)//c
	    {
		top_concat_axis = dimsConv4_3.c() + dimsFc7.c() + dimsConv6.c() + dimsConv7.c() + dimsConv8.c() + dimsConv9.c();
		bottom_concat_axis[0] = dimsConv4_3.c(); bottom_concat_axis[1] = dimsFc7.c(); bottom_concat_axis[2] = dimsConv6.c();
		bottom_concat_axis[3] = dimsConv7.c(); bottom_concat_axis[4] = dimsConv8.c(); bottom_concat_axis[5] = dimsConv9.c();

		concat_input_size_[0] = dimsConv4_3.h() * dimsConv4_3.w();  concat_input_size_[1] = dimsFc7.h() * dimsFc7.w();
		concat_input_size_[2] = dimsConv6.h() * dimsConv6.w();  concat_input_size_[3] = dimsConv7.h() * dimsConv7.w();
		concat_input_size_[4] = dimsConv8.h() * dimsConv8.w();  concat_input_size_[5] = dimsConv9.h() * dimsConv9.w();

		num_concats_[0] = dimsConv4_3.c(); num_concats_[1] = dimsFc7.c(); num_concats_[2] = dimsConv6.c();
		num_concats_[3] = dimsConv7.c(); num_concats_[4] = dimsConv8.c(); num_concats_[5] = dimsConv9.c();
	    }else if(_axis == 2){//h
		top_concat_axis = dimsConv4_3.h() + dimsFc7.h() + dimsConv6.h() + dimsConv7.h() + dimsConv8.h() + dimsConv9.h();
		bottom_concat_axis[0] = dimsConv4_3.h(); bottom_concat_axis[1] = dimsFc7.h(); bottom_concat_axis[2] = dimsConv6.h();
		bottom_concat_axis[3] = dimsConv7.h(); bottom_concat_axis[4] = dimsConv8.h(); bottom_concat_axis[5] = dimsConv9.h();

		concat_input_size_[0] = dimsConv4_3.w(); concat_input_size_[1] = dimsFc7.w(); concat_input_size_[2] = dimsConv6.w();
		concat_input_size_[3] = dimsConv7.w(); concat_input_size_[4] = dimsConv8.w(); concat_input_size_[5] = dimsConv9.w();

		num_concats_[0] = dimsConv4_3.c() * dimsConv4_3.h();  num_concats_[1] = dimsFc7.c() * dimsFc7.h();
		num_concats_[2] = dimsConv6.c() * dimsConv6.h();  num_concats_[3] = dimsConv7.c() * dimsConv7.h();
		num_concats_[4] = dimsConv8.c() * dimsConv8.h();  num_concats_[5] = dimsConv9.c() * dimsConv9.h();

	    }else{//_param.concat_axis == 3 , w
		top_concat_axis = dimsConv4_3.w() + dimsFc7.w() + dimsConv6.w() + dimsConv7.w() + dimsConv8.w() + dimsConv9.w();
		bottom_concat_axis[0] = dimsConv4_3.w(); bottom_concat_axis[1] = dimsFc7.w(); bottom_concat_axis[2] = dimsConv6.w();
		bottom_concat_axis[3] = dimsConv7.w(); bottom_concat_axis[4] = dimsConv8.w(); bottom_concat_axis[5] = dimsConv9.w();

		concat_input_size_[0] = 1; concat_input_size_[1] = 1; concat_input_size_[2] = 1;
		concat_input_size_[3] = 1; concat_input_size_[4] = 1; concat_input_size_[5] = 1;
		return 0;
	    }

	    return 0;
	}

  //  inline void terminate() override;
	void  terminate()
	{
	    //CUDA_CHECK(cudaFree(scale_data));
	    delete[] bottom_concat_axis;
	    delete[] concat_input_size_;
	    delete[] num_concats_;
	}



    inline size_t getWorkspaceSize(int) const override { return 0; };
    //int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override;

	int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream)
	{
	    float *top_data = reinterpret_cast<float*>(outputs[0]);
	    int offset_concat_axis = 0;
	    const bool kForward = true;
	    for (int i = 0; i < inputs_size; ++i) {
		const float *bottom_data = reinterpret_cast<const float*>(inputs[i]);

		const int nthreads = num_concats_[i] * concat_input_size_[i];
		//const int nthreads = bottom_concat_size * num_concats_[i];
		ConcatLayer(nthreads, bottom_data, kForward, num_concats_[i], concat_input_size_[i], top_concat_axis, bottom_concat_axis[i], offset_concat_axis, top_data, stream);

		offset_concat_axis += bottom_concat_axis[i];
	    }

	    return 0;
	}

   // size_t getSerializationSize() override;
	size_t getSerializationSize()
	{
	    return 18*sizeof(int);
	}


   // void serialize(void* buffer) override;

	void serialize(void* buffer)
	{
	    int* d = reinterpret_cast<int*>(buffer);
	    d[0] = dimsConv4_3.c(); d[1] = dimsConv4_3.h(); d[2] = dimsConv4_3.w();
	    d[3] = dimsFc7.c(); d[4] = dimsFc7.h(); d[5] = dimsFc7.w();
	    d[6] = dimsConv6.c(); d[7] = dimsConv6.h(); d[8] = dimsConv6.w();
	    d[9] = dimsConv7.c(); d[10] = dimsConv7.h(); d[11] = dimsConv7.w();
	    d[12] = dimsConv8.c(); d[13] = dimsConv8.h(); d[14] = dimsConv8.w();
	    d[15] = dimsConv9.c(); d[16] = dimsConv9.h(); d[17] = dimsConv9.w();
	}



   // void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override;

	void  configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int)
	{
	    dimsConv4_3 = DimsCHW{inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]};
	    dimsFc7 = DimsCHW{inputs[1].d[0], inputs[1].d[1], inputs[1].d[2]};
	    dimsConv6 = DimsCHW{inputs[2].d[0], inputs[2].d[1], inputs[2].d[2]};
	    dimsConv7 = DimsCHW{inputs[3].d[0], inputs[3].d[1], inputs[3].d[2]};
	    dimsConv8 = DimsCHW{inputs[4].d[0], inputs[4].d[1], inputs[4].d[2]};
	    dimsConv9 = DimsCHW{inputs[5].d[0], inputs[5].d[1], inputs[5].d[2]};
	}



 

protected:
    DimsCHW dimsConv4_3, dimsFc7, dimsConv6, dimsConv7, dimsConv8, dimsConv9;
    int inputs_size;
    int top_concat_axis;//top 层 concat后的维度
    int* bottom_concat_axis = new int[9];//记录每个bottom层concat维度的shape
    int* concat_input_size_ = new int[9];
    int* num_concats_ = new int[9];
    int _axis;
};

#endif

//SSD Reshape layer : shape{0,-1,21}
template<int OutC>
class Reshape : public IPlugin
{
public:
    Reshape() {}
    Reshape(const void* buffer, size_t size)
    {
        assert(size == sizeof(mCopySize));
        mCopySize = *reinterpret_cast<const size_t*>(buffer);
    }

    int getNbOutputs() const override
    {
        return 1;
    }
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(nbInputDims == 1);
        assert(index == 0);
        assert(inputs[index].nbDims == 3);
        assert((inputs[0].d[0])*(inputs[0].d[1]) % OutC == 0);
        return DimsCHW(OutC, inputs[0].d[0] * inputs[0].d[1] / OutC, inputs[0].d[2]);
    }

    int initialize() override
    {
        return 0;
    }

    void terminate() override
    {
    }

    size_t getWorkspaceSize(int) const override
    {
        return 0;
    }

    // currently it is not possible for a plugin to execute "in place". Therefore we memcpy the data from the input to the output buffer
    int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override
    {
        CHECK(cudaMemcpyAsync(outputs[0], inputs[0], mCopySize * batchSize, cudaMemcpyDeviceToDevice, stream));
        return 0;
    }


    size_t getSerializationSize() override
    {
        return sizeof(mCopySize);
    }

    void serialize(void* buffer) override
    {
        *reinterpret_cast<size_t*>(buffer) = mCopySize;
    }

    void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int)	override
    {
        mCopySize = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2] * sizeof(float);
    }

protected:
    size_t mCopySize;
};



class SofaMaxChannelLayer: public IPlugin
{
public:
    SofaMaxChannelLayer(int axis): _axis(axis),inner_num_(1),outer_num_(3462){}
    
    SofaMaxChannelLayer(int axis,const void* buffer,size_t size)
    {
 	  _axis = axis;
    }

    inline int getNbOutputs() const override { return 1; };
    
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
   
        return DimsCHW(1,3462, 2);
    }

    int initialize() override
    {
        return 0;
    }
    inline void terminate() override
    {
    }

    inline size_t getWorkspaceSize(int) const override { return 0; }
    int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override
    {
 
        return 0;
    }


    size_t getSerializationSize() override
    {
        return 0;
    }

    void serialize(void* buffer) override
    {
     
    }

    void configure(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override
    {
         
#if 0
           int dim = bottom[0]->count() / outer_num_;
          caffe_copy(bottom[0]->count(), bottom_data, top_data);
          // We need to subtract the max to avoid numerical issues, compute the exp,
          // and then normalize.
          for (int i = 0; i < outer_num_; ++i) {
            // initialize scale_data to the first plane
            caffe_copy(inner_num_, bottom_data + i * dim, scale_data);
            for (int j = 0; j < ; j++) {
              for (int k = 0; k < inner_num_; k++) {
                scale_data[k] = std::max(scale_data[k],
                    bottom_data[i * dim + j * inner_num_ + k]);
              }
            }
            // subtraction
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_,
                1, -1., sum_multiplier_.cpu_data(), scale_data, 1., top_data);
            // exponentiation
            caffe_exp<Dtype>(dim, top_data, top_data);
            // sum after exp
            caffe_cpu_gemv<Dtype>(CblasTrans, channels, inner_num_, 1.,
                top_data, sum_multiplier_.cpu_data(), 0., scale_data);
            // division
            for (int j = 0; j < channels; j++) {
              caffe_div(inner_num_, top_data, scale_data, top_data);
              top_data += inner_num_;
            }
          }

  
#endif



    }

protected:
    int _axis;
    int _size;
    float* scale = new float [3462]; //scale
    int inner_num_;
    int outer_num_;
    
    
  
};

# if 0
__global__ void ReLUForward1(const int n, const float* in, float* out,
    float negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
        printf("ReLUForwardsss  01 enqueue  line=%d",__LINE__);
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  }
}
#endif

//SSD Flatten layer
class LReluLayer : public IPlugin
{
public:
    LReluLayer(){}
    LReluLayer(float para):para_(para)
    {


              std::cout<<"LReluLayer0"<<std::endl;
    }

    LReluLayer(const void* buffer,size_t sz, float para):para_(para)
    {


        assert(sz == 4 * sizeof(int));
        const int* d = reinterpret_cast<const int*>(buffer);
        float* p=(float*)(d+3);
	para_=p[0];
	channel_=d[0];
	w_=d[1];
	h_=d[2];

        //std::cout<<"LReluLayer1"<<para_ <<" " <<channel_<<" "<<w_ <<" "<<h_ <<std::endl;
        para_=0.1;

    }

    inline int getNbOutputs() const override { return 1; };
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
         std::cout<<"getOutputDimensions  channel"<<inputs[0].d[0]<<"h:"<<inputs[0].d[1]<<"w:"<<inputs[0].d[2]<<std::endl;

         channel_=inputs[0].d[0];
         w_=inputs[0].d[1];
         h_=inputs[0].d[2];
         
        return DimsCHW(inputs[0].d[0], inputs[0].d[1] , inputs[0].d[2] );
    }

    int initialize() override
    {
        return 0;
    }
    inline void terminate() override
    {
    }

 

    inline size_t getWorkspaceSize(int) const override { return 0; }
    int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override
    {
	

     //   std::cout<<"LReluLayer1 enqueue : "<<batchSize<<"c:"<<channel_<<"w:"<<w_<<"h:"<<h_<<"para_"<<para_<<std::endl;
	ReluForward_gpu((const float*)inputs[0],(float*)outputs[0],batchSize,channel_,w_,h_,para_);

	 //ReLUForward1<<<CAFFE_GET_BLOCKS(batchSize*channel_*w_*h_), CAFFE_CUDA_NUM_THREADS>>>(batchSize*channel_*w_*h_,inputs[0],outputs[0],para_);

	//std::cout<<"flatten enqueue:"<<batchSize<<";"<<_size<<std::endl;
	//CHECK(cudaMemcpyAsync(outputs[0],inputs[0],batchSize*_size*sizeof(float),cudaMemcpyDeviceToDevice,stream));
	//Forward_gpu (
	//  (float*)inputs[0],1,channel_, w_, h_, stride_, (float*)outputs[0] );


        //CHECK(cudaMemcpyAsync(outputs[0], inputs[0], mCopySize * batchSize, cudaMemcpyDeviceToDevice, stream));

        return 0;
    }


    size_t getSerializationSize() override
    {
        return sizeof(int)*3+sizeof(float);
    }

    void serialize(void* buffer) override
    {
        
	 
	//
	//write(q+3, (float)para_);

        float* q = reinterpret_cast<float*>(buffer);
        int* d = reinterpret_cast<int*>(buffer);
        d[0] = channel_; d[1] = w_; d[2] = h_;
        q[4]=para_;

	//serializeFromDevice(d, mKernelWeights);
	//serializeFromDevice(d, mBiasWeights);
	 
 

    }

    void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override
    {
      //  dimBottom = DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
         channel_=inputs[0].d[0];
         w_=inputs[0].d[1];
         h_=inputs[0].d[2];

    }

protected:
    float para_;
    int channel_;
    int w_;
    int h_;
};





//SSD Flatten layer
class UpsampleLayer : public IPlugin
{
public:
    UpsampleLayer(){}
    UpsampleLayer(size_t stride):stride_(stride)
    {
      std::cout<<"UpsampleLayer0"<<std::endl;


    }

    UpsampleLayer(const void* buffer,size_t sz, size_t stride):stride_(stride)
    {

        const int* d = reinterpret_cast<const int*>(buffer);
 
	channel_=d[0];
	w_=d[1];
	h_=d[2];


        std::cout<<"UpsampleLayer1"<<std::endl;

    }

    inline int getNbOutputs() const override { return 1; };
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
         std::cout<<"channel"<<inputs[0].d[0]<<"h:"<<inputs[0].d[1]<<"w:"<<inputs[0].d[2]<<std::endl;

         channel_=inputs[0].d[0];
         w_=inputs[0].d[1];
         h_=inputs[0].d[2];

        return DimsCHW(inputs[0].d[0], inputs[0].d[1]*stride_, inputs[0].d[2]*stride_);
    }

    int initialize() override
    {
        return 0;
    }
    inline void terminate() override
    {
    }



    inline size_t getWorkspaceSize(int) const override { return 0; }
    int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override
    {


    // std::cout<<"UpsampleLayer1 enqueue"<<std::endl;
        //std::cout<<"flatten enqueue:"<<batchSize<<";"<<_size<<std::endl;
        //CHECK(cudaMemcpyAsync(outputs[0],inputs[0],batchSize*_size*sizeof(float),cudaMemcpyDeviceToDevice,stream));
     Forward_gpu((float*)inputs[0],1,channel_, w_, h_, stride_, (float*)outputs[0] );




        return 0;
    }


    size_t getSerializationSize() override
    {
        return 4*sizeof(int);
    }

    void serialize(void* buffer) override
    {
   
        int* d = reinterpret_cast<int*>(buffer);
        d[0] = channel_; d[1] = w_; d[2] = h_;
        d[3]=stride_;
   
    }

    void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override
    {
      //  dimBottom = DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
         channel_=inputs[0].d[0];
         w_=inputs[0].d[1];
         h_=inputs[0].d[2];

    }

protected:
    int stride_;
    int channel_;
    int w_;
    int h_;
};





//SSD Flatten layer
class FlattenLayer : public IPlugin
{
public:
    FlattenLayer(){}
    FlattenLayer(const void* buffer,size_t size)
    {
        assert(size == 3 * sizeof(int));
        const int* d = reinterpret_cast<const int*>(buffer);
        _size = d[0] * d[1] * d[2];
        dimBottom = DimsCHW{d[0], d[1], d[2]};
    }

    inline int getNbOutputs() const override { return 1; };
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(1 == nbInputDims);
        assert(0 == index);
        assert(3 == inputs[index].nbDims);
        _size = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];
        return DimsCHW(_size, 1, 1);
    }

    int initialize() override
    {
        return 0;
    }
    inline void terminate() override
    {
    }

    inline size_t getWorkspaceSize(int) const override { return 0; }
    int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override
    {
        std::cout<<"flatten enqueue:"<<batchSize<<";"<<_size<<std::endl;
        CHECK(cudaMemcpyAsync(outputs[0],inputs[0],batchSize*_size*sizeof(float),cudaMemcpyDeviceToDevice,stream));
        return 0;
    }


    size_t getSerializationSize() override
    {
        return 3 * sizeof(int);
    }

    void serialize(void* buffer) override
    {
        int* d = reinterpret_cast<int*>(buffer);
        d[0] = dimBottom.c(); d[1] = dimBottom.h(); d[2] = dimBottom.w();
    }

    void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override
    {
        dimBottom = DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
    }

protected:
    DimsCHW dimBottom;
    int _size;
};



  
  
class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory {  
public:  
  
        // caffe parser plugin implementation  
        bool isPlugin(const char* name) override  
        {  
         printf("isPlugin %s\n",name);
        return (!strcmp(name, "conv11_mbox_loc_perm")  
            || !strcmp(name, "conv11_mbox_conf_flat")  
            || !strcmp(name, "conv11_mbox_conf_perm")  
            || !strcmp(name, "conv11_mbox_loc_flat") 
            || !strcmp(name, "conv11_mbox_priorbox") 
                
            || !strcmp(name, "conv13_mbox_loc_perm")  
            || !strcmp(name, "conv13_mbox_loc_flat")  
            || !strcmp(name, "conv13_mbox_conf_perm")  
            || !strcmp(name, "conv13_mbox_conf_flat")  
            || !strcmp(name, "conv13_mbox_priorbox")  
                
            || !strcmp(name, "conv14_2_mbox_loc_perm") 
            || !strcmp(name, "conv14_2_mbox_loc_flat") 
            || !strcmp(name, "conv14_2_mbox_conf_perm")      
            || !strcmp(name, "conv14_2_mbox_conf_flat")     
            || !strcmp(name, "conv14_2_mbox_priorbox") 
                
            || !strcmp(name, "conv15_2_mbox_loc_perm")
            || !strcmp(name, "conv15_2_mbox_loc_flat")
            || !strcmp(name, "conv15_2_mbox_conf_perm")                 
            || !strcmp(name, "conv15_2_mbox_conf_flat")      
            || !strcmp(name, "conv15_2_mbox_priorbox")   
                
            || !strcmp(name, "conv16_2_mbox_loc_perm")     
            || !strcmp(name, "conv16_2_mbox_loc_flat")  
            || !strcmp(name, "conv16_2_mbox_conf_perm")  
            || !strcmp(name, "conv16_2_mbox_conf_flat")  
            || !strcmp(name, "conv16_2_mbox_priorbox") 
                
            || !strcmp(name, "conv17_2_mbox_loc_perm")  
            || !strcmp(name, "conv17_2_mbox_loc_flat")  
            || !strcmp(name, "conv17_2_mbox_conf_perm")  
            || !strcmp(name, "conv17_2_mbox_conf_flat")                
            || !strcmp(name, "conv17_2_mbox_priorbox")  
                
            || !strcmp(name, "mbox_conf_reshape")  
            || !strcmp(name, "mbox_conf_flatten")  
            || !strcmp(name, "detection_out")
                
            || !strcmp(name, "mbox_loc")  
            || !strcmp(name, "mbox_conf")  
            || !strcmp(name, "mbox_priorbox")  
            || !strcmp(name, "mbox_conf_softmax")
 
            || !strcmp(name, "layer85-upsample")  
            || !strcmp(name, "layer97-upsample") 

            || !strcmp(name, "layer104-act") 
            || !strcmp(name, "layer28-act") 
            || !strcmp(name, "layer0-act")  
            || !strcmp(name, "layer2-act")  
            || !strcmp(name, "layer1-act")  
            || !strcmp(name, "layer103-act")  
            || !strcmp(name, "layer102-act")  
            || !strcmp(name, "layer101-act")  
            || !strcmp(name, "layer100-act")  
            || !strcmp(name, "layer99-act")  
            || !strcmp(name, "layer96-act")  
            || !strcmp(name, "layer92-act")  
            || !strcmp(name, "layer91-act")  
            || !strcmp(name, "layer90-act")  
            || !strcmp(name, "layer89-act")  
            || !strcmp(name, "layer88-act")  
            || !strcmp(name, "layer87-act")  
            || !strcmp(name, "layer84-act")  
            || !strcmp(name, "layer80-act")  
            || !strcmp(name, "layer79-act")  
            || !strcmp(name, "layer78-act")  



            || !strcmp(name, "layer77-act")  
            || !strcmp(name, "layer76-act")  
            || !strcmp(name, "layer75-act")  
            || !strcmp(name, "layer73-act")  
            || !strcmp(name, "layer72-act")  
            || !strcmp(name, "layer70-act")  
            || !strcmp(name, "layer69-act")  
            || !strcmp(name, "layer67-act")  
            || !strcmp(name, "layer66-act")  
            || !strcmp(name, "layer64-act")  
            || !strcmp(name, "layer63-act")  
            || !strcmp(name, "layer62-act")  
            || !strcmp(name, "layer60-act")  
            || !strcmp(name, "layer59-act")  
            || !strcmp(name, "layer57-act")  
            || !strcmp(name, "layer56-act")  
            || !strcmp(name, "layer54-act")  
            || !strcmp(name, "layer53-act")  
//

            || !strcmp(name, "layer51-act")  
            || !strcmp(name, "layer50-act")  
            || !strcmp(name, "layer48-act")  
            || !strcmp(name, "layer47-act")  
            || !strcmp(name, "layer45-act")  
            || !strcmp(name, "layer44-act")  
            || !strcmp(name, "layer42-act")  
            || !strcmp(name, "layer41-act")  
            || !strcmp(name, "layer39-act")  
            || !strcmp(name, "layer38-act")  
            || !strcmp(name, "layer37-act")  
            || !strcmp(name, "layer35-act")  
            || !strcmp(name, "layer34-act")  
            || !strcmp(name, "layer32-act")  
            || !strcmp(name, "layer31-act")  
            || !strcmp(name, "layer29-act")  
            || !strcmp(name, "layer26-act")  
            || !strcmp(name, "layer25-act")  



            || !strcmp(name, "layer23-act")  
            || !strcmp(name, "layer22-act")  
            || !strcmp(name, "layer20-act")  
            || !strcmp(name, "layer19-act")  
            || !strcmp(name, "layer17-act")  
            || !strcmp(name, "layer16-act")  
            || !strcmp(name, "layer14-act")  
            || !strcmp(name, "layer13-act")  
            || !strcmp(name, "layer12-act")  
            || !strcmp(name, "layer10-act")  
            || !strcmp(name, "layer9-act")  
            || !strcmp(name, "layer7-act")  
            || !strcmp(name, "layer6-act")  
            || !strcmp(name, "layer5-act")  
            || !strcmp(name, "layer3-act")  
     
 
        );  
  
        }  
  
        virtual IPlugin* createPlugin(const char* layerName, const Weights* weights, int nbWeights) override  
        {  
                // there's no way to pass parameters through from the model definition, so we have to define it here explicitly  
            if(!strcmp(layerName, "conv4_3_norm")){  
  
                //INvPlugin *   plugin::createSSDNormalizePlugin (const Weights *scales, bool acrossSpatial, bool channelShared, float eps)  

                _nvPlugins[layerName] = plugin::createSSDNormalizePlugin(weights,false,false,1e-10);  

                return _nvPlugins.at(layerName);  
  
            }else if(!strcmp(layerName, "conv11_mbox_loc_perm") 
            ||  !strcmp(layerName, "conv11_mbox_conf_perm") 
            ||  !strcmp(layerName, "conv13_mbox_loc_perm")  
            ||  !strcmp(layerName,"conv13_mbox_conf_perm") 
            ||  !strcmp(layerName,"conv14_2_mbox_loc_perm")  
            ||  !strcmp(layerName,"conv14_2_mbox_conf_perm")  
            ||  !strcmp(layerName,"conv15_2_mbox_loc_perm")                           
            ||  !strcmp(layerName,"conv15_2_mbox_conf_perm")  
            ||  !strcmp(layerName,"conv16_2_mbox_loc_perm")  
            ||  !strcmp(layerName,"conv16_2_mbox_conf_perm")  
            ||  !strcmp(layerName,"conv17_2_mbox_loc_perm")  
            ||  !strcmp(layerName,"conv17_2_mbox_conf_perm")  
          ){  
  
                _nvPlugins[layerName] = plugin::createSSDPermutePlugin(Quadruple({0,2,3,1}));  
                return _nvPlugins.at(layerName);  
  
           } else if(!strcmp(layerName,"conv11_mbox_priorbox")){  
  
                plugin::PriorBoxParameters params = {0};  
                float minSize[1] = {60.0f};   
                //float maxSize[1] = {60.0f};   
                float aspectRatios[1] = {2.0f};   
                params.minSize = (float*)minSize;  
                //params.maxSize = (float*)maxSize;  
                params.aspectRatios = (float*)aspectRatios;  
                params.numMinSize = 1;  
                params.numMaxSize = 0;  
                params.numAspectRatios = 1;  
                params.flip = true;  
                params.clip = false;  
                params.variance[0] = 0.10000000149;  
                params.variance[1] = 0.10000000149;  
                params.variance[2] = 0.20000000298;  
                params.variance[3] = 0.20000000298;  
                params.imgH = 0;  
                params.imgW = 0;  
                params.stepH = 0.0f;  
                params.stepW = 0.0f;  
                params.offset = 0.5f;  
                _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(params);  

                return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"conv13_mbox_priorbox")){  
  
                plugin::PriorBoxParameters params = {0};  
                float minSize[1] = {105.0f};   
                float maxSize[1] = {150.0f};   
                float aspectRatios[2] = {2.0f, 3.0f};   
                params.minSize = (float*)minSize;  
                params.maxSize = (float*)maxSize;  
                params.aspectRatios = (float*)aspectRatios;  
                params.numMinSize = 1;  
                params.numMaxSize = 1;  
                params.numAspectRatios = 2;  
                params.flip = true;  
                params.clip = false;  
                params.variance[0] = 0.10000000149;  
                params.variance[1] = 0.10000000149;  
                params.variance[2] = 0.20000000298;  
                params.variance[3] = 0.20000000298;  
                params.imgH = 0;  
                params.imgW = 0;  
                params.stepH = 0.0f;  
                params.stepW = 0.0f;  
                params.offset = 0.5f;  
                _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(params);  
  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"conv14_2_mbox_priorbox")){  
                plugin::PriorBoxParameters params = {0};  
                float minSize[1] = {150.0f};   
                float maxSize[1] = {195.0f};   
                float aspectRatios[2] = {2.0f, 3.0f};   
                params.minSize = (float*)minSize;  
                params.maxSize = (float*)maxSize;  
                params.aspectRatios = (float*)aspectRatios;  
                params.numMinSize = 1;  
                params.numMaxSize = 1;  
                params.numAspectRatios = 2;  
                params.flip = true;  
                params.clip = false;  
                params.variance[0] = 0.10000000149;  
                params.variance[1] = 0.10000000149;  
                params.variance[2] = 0.20000000298;  
                params.variance[3] = 0.20000000298;  
                params.imgH = 0;  
                params.imgW = 0;  
                params.stepH = 0.0f;  
                params.stepW = 0.0f;  
                params.offset = 5.0f;  
                _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(params);  

                return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"conv15_2_mbox_priorbox")){  
  
                plugin::PriorBoxParameters params = {0};  
                float minSize[1] = {195.0f};   
                float maxSize[1] = {240.0f};   
                float aspectRatios[2] = {2.0f, 3.0f};   
                params.minSize = (float*)minSize;  
                params.maxSize = (float*)maxSize;  
                params.aspectRatios = (float*)aspectRatios;  
                params.numMinSize = 1;  
                params.numMaxSize = 1;  
                params.numAspectRatios = 2;  
                params.flip = true;  
                params.clip = false;  
                params.variance[0] = 0.10000000149;  
                params.variance[1] = 0.10000000149;  
                params.variance[2] = 0.20000000298;  
                params.variance[3] = 0.20000000298;  
                params.imgH = 0;  
                params.imgW = 0;  
                params.stepH = 0.0f;  
                params.stepW = 0.0f;  
                params.offset = 0.5f;  
                _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(params);  

                return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"conv16_2_mbox_priorbox")){  
  
                plugin::PriorBoxParameters params = {0};  
                float minSize[1] = {240.0f};   
                float maxSize[1] = {285.0f};   
                float aspectRatios[2] = {2.0f,3.0f};   
                params.minSize = (float*)minSize;  
                params.maxSize = (float*)maxSize;  
                params.aspectRatios = (float*)aspectRatios;  
                params.numMinSize = 1;  
                params.numMaxSize = 1;  
                params.numAspectRatios = 2;  
                params.flip = true;  
                params.clip = false;  
                params.variance[0] = 0.10000000149;  
                params.variance[1] = 0.10000000149;  
                params.variance[2] = 0.20000000298;  
                params.variance[3] = 0.20000000298;  
                params.imgH = 0;  
                params.imgW = 0;  
                params.stepH = 0.0f;  
                params.stepW = 0.0f;  
                params.offset = 0.5f;  
                _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(params);  

                return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"conv17_2_mbox_priorbox")){  
                plugin::PriorBoxParameters params = {0};  
                float minSize[1] = {285.0f};   
                float maxSize[1] = {300.0f};   
                float aspectRatios[2] = {2.0f,3.0f};   
                params.minSize = (float*)minSize;  
                params.maxSize = (float*)maxSize;  
                params.aspectRatios = (float*)aspectRatios;  
                params.numMinSize = 1;  
                params.numMaxSize = 1;  
                params.numAspectRatios = 2;  
                params.flip = true;  
                params.clip = false;  
                params.variance[0] = 0.10000000149;  
                params.variance[1] = 0.10000000149;  
                params.variance[2] = 0.20000000298;  
                params.variance[3] = 0.20000000298;  
                params.imgH = 0;  
                params.imgW = 0;  
                params.stepH = 0.0f;  
                params.stepW = 0.0f;  
                params.offset = 0.5f;  
                _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(params);  

                return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"detection_out")){  
                /*  
                bool    shareLocation  
                bool    varianceEncodedInTarget  
                int     backgroundLabelId  
                int     numClasses  
                int     topK  
                int     keepTopK  
                float   confidenceThreshold  
                float   nmsThreshold  
                CodeType_t  codeType  
                */  
                plugin::DetectionOutputParameters params = {0};  
                params.numClasses = 2;  
                params.shareLocation = true;  
                params.varianceEncodedInTarget = false;  
                params.backgroundLabelId = 0;  
                params.keepTopK = 100;  
                params.codeType = CENTER_SIZE;  
                params.nmsThreshold = 0.45;  
                params.topK = 100;  
                params.confidenceThreshold = 0.25;  
                std::cout << "detection_out..." << std::endl;
                _nvPlugins[layerName] = plugin::createSSDDetectionOutputPlugin(params);  
                return _nvPlugins.at(layerName);  
            
        }else if (  
            !strcmp(layerName, "conv11_mbox_conf_flat")  
            ||!strcmp(layerName,"conv11_mbox_loc_flat")  
            ||!strcmp(layerName,"conv13_mbox_loc_flat")  
            ||!strcmp(layerName,"conv13_mbox_conf_flat")  
            ||!strcmp(layerName,"conv14_2_mbox_loc_flat")  
            ||!strcmp(layerName,"conv14_2_mbox_conf_flat")  
            ||!strcmp(layerName,"conv15_2_mbox_loc_flat")  
            ||!strcmp(layerName,"conv15_2_mbox_conf_flat")  
            ||!strcmp(layerName,"conv16_2_mbox_loc_flat")  
            ||!strcmp(layerName,"conv16_2_mbox_conf_flat")  
            ||!strcmp(layerName,"conv17_2_mbox_loc_flat")  
            ||!strcmp(layerName,"conv17_2_mbox_conf_flat")  
            ||!strcmp(layerName,"mbox_conf_flatten")  
        ){  
             _nvPlugins[layerName] = (plugin::INvPlugin*)(new FlattenLayer());  
             return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"mbox_conf_reshape")){  
             _nvPlugins[layerName] = (plugin::INvPlugin*)new Reshape<2>();  
             return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"mbox_loc")){  
             _nvPlugins[layerName] = plugin::createConcatPlugin (1,false);  
             return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"mbox_conf")){  
             _nvPlugins[layerName] = plugin::createConcatPlugin (1,false);  
             return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"mbox_priorbox")){  
             _nvPlugins[layerName] = plugin::createConcatPlugin (2,false);  
             return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName, "mbox_conf_softmax"))
        {
             _nvPlugins[layerName] = (plugin::INvPlugin*)(new SofaMaxChannelLayer(2));
             return _nvPlugins.at(layerName);

 
  
        }else if(!strcmp(layerName, "layer85-upsample"))
        {
             _nvPlugins[layerName] = (plugin::INvPlugin*)(new UpsampleLayer(2));
             return _nvPlugins.at(layerName);

 
  
        }else if(!strcmp(layerName, "layer97-upsample"))
        {
             _nvPlugins[layerName] = (plugin::INvPlugin*)(new UpsampleLayer(2));
             return _nvPlugins.at(layerName);
        }else if (  !strcmp(layerName, "layer2-act")  
            || !strcmp(layerName, "layer104-act")  
            || !strcmp(layerName, "layer1-act")  
            || !strcmp(layerName, "layer0-act")  
            || !strcmp(layerName, "layer28-act")  
            || !strcmp(layerName, "layer103-act")  
            || !strcmp(layerName, "layer102-act")  
            || !strcmp(layerName, "layer101-act")  
            || !strcmp(layerName, "layer100-act")  
            || !strcmp(layerName, "layer99-act")  
            || !strcmp(layerName, "layer96-act")  
            || !strcmp(layerName, "layer92-act")  
            || !strcmp(layerName, "layer91-act")  
            || !strcmp(layerName, "layer90-act")  
            || !strcmp(layerName, "layer89-act")  
            || !strcmp(layerName, "layer88-act")  
            || !strcmp(layerName, "layer87-act")  
            || !strcmp(layerName, "layer84-act")  
            || !strcmp(layerName, "layer80-act")  
            || !strcmp(layerName, "layer79-act")  
            || !strcmp(layerName, "layer78-act")  



            || !strcmp(layerName, "layer77-act")  
            || !strcmp(layerName, "layer76-act")  
            || !strcmp(layerName, "layer75-act")  
            || !strcmp(layerName, "layer73-act")  
            || !strcmp(layerName, "layer72-act")  
            || !strcmp(layerName, "layer70-act")  
            || !strcmp(layerName, "layer69-act")  
            || !strcmp(layerName, "layer67-act")  
            || !strcmp(layerName, "layer66-act")  
            || !strcmp(layerName, "layer64-act")  
            || !strcmp(layerName, "layer63-act")  
            || !strcmp(layerName, "layer62-act")  
            || !strcmp(layerName, "layer60-act")  
            || !strcmp(layerName, "layer59-act")  
            || !strcmp(layerName, "layer57-act")  
            || !strcmp(layerName, "layer56-act")  
            || !strcmp(layerName, "layer54-act")  
            || !strcmp(layerName, "layer53-act")  
//

            || !strcmp(layerName, "layer51-act")  
            || !strcmp(layerName, "layer50-act")  
            || !strcmp(layerName, "layer48-act")  
            || !strcmp(layerName, "layer47-act")  
            || !strcmp(layerName, "layer45-act")  
            || !strcmp(layerName, "layer44-act")  
            || !strcmp(layerName, "layer42-act")  
            || !strcmp(layerName, "layer41-act")  
            || !strcmp(layerName, "layer39-act")  
            || !strcmp(layerName, "layer38-act")  
            || !strcmp(layerName, "layer37-act")  
            || !strcmp(layerName, "layer35-act")  
            || !strcmp(layerName, "layer34-act")  
            || !strcmp(layerName, "layer32-act")  
            || !strcmp(layerName, "layer31-act")  
            || !strcmp(layerName, "layer29-act")  
            || !strcmp(layerName, "layer26-act")  
            || !strcmp(layerName, "layer25-act")  



            || !strcmp(layerName, "layer23-act")  
            || !strcmp(layerName, "layer22-act")  
            || !strcmp(layerName, "layer20-act")  
            || !strcmp(layerName, "layer19-act")  
            || !strcmp(layerName, "layer17-act")  
            || !strcmp(layerName, "layer16-act")  
            || !strcmp(layerName, "layer14-act")  
            || !strcmp(layerName, "layer13-act")  
            || !strcmp(layerName, "layer12-act")  
            || !strcmp(layerName, "layer10-act")  
            || !strcmp(layerName, "layer9-act")  
            || !strcmp(layerName, "layer7-act")  
            || !strcmp(layerName, "layer6-act")  
            || !strcmp(layerName, "layer5-act")  
            || !strcmp(layerName, "layer3-act")  
       ){

         _nvPlugins[layerName] = (plugin::INvPlugin*)(new LReluLayer(0.1));
             return _nvPlugins.at(layerName);


        }else{  
             assert(0);  
             return nullptr;  
        }  
    }  
  
    // deserialization plugin implementation  
    IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override {                
        if(!strcmp(layerName, "conv4_3_norm"))  
        {  
            _nvPlugins[layerName] = plugin::createSSDNormalizePlugin(serialData, serialLength);  
            return _nvPlugins.at(layerName);  
            
        }else if(  !strcmp(layerName, "conv11_mbox_loc_perm") 
            ||  !strcmp(layerName, "conv11_mbox_conf_perm") 
            ||  !strcmp(layerName, "conv13_mbox_loc_perm")  
            ||  !strcmp(layerName,"conv13_mbox_conf_perm") 
            ||  !strcmp(layerName,"conv14_2_mbox_loc_perm")  
            ||  !strcmp(layerName,"conv14_2_mbox_conf_perm")  
            || !strcmp(layerName,"conv15_2_mbox_loc_perm")                           
            || !strcmp(layerName,"conv15_2_mbox_conf_perm")  
            || !strcmp(layerName,"conv16_2_mbox_loc_perm")  
            || !strcmp(layerName,"conv16_2_mbox_conf_perm")  
            || !strcmp(layerName,"conv17_2_mbox_loc_perm")  
            || !strcmp(layerName,"conv17_2_mbox_conf_perm") 
        ){  
            _nvPlugins[layerName] = plugin::createSSDPermutePlugin(serialData, serialLength);  
            return _nvPlugins.at(layerName);  
  
            
        }else if(!strcmp(layerName,"conv11_mbox_priorbox")  
            || !strcmp(layerName,"conv13_mbox_priorbox")     
            || !strcmp(layerName,"conv14_2_mbox_priorbox")  
            || !strcmp(layerName,"conv15_2_mbox_priorbox")  
            || !strcmp(layerName,"conv16_2_mbox_priorbox")  
            || !strcmp(layerName,"conv17_2_mbox_priorbox")  
        ){  
            _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin (serialData, serialLength);  
            return _nvPlugins.at(layerName);  
            
        }else if(!strcmp(layerName,"detection_out")){  
            _nvPlugins[layerName] = plugin::createSSDDetectionOutputPlugin (serialData, serialLength);  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"mbox_conf_reshape")){  
            _nvPlugins[layerName] = (plugin::INvPlugin*)new Reshape<21>(serialData, serialLength);  
            return _nvPlugins.at(layerName);  
        }else if (  
            !strcmp(layerName, "conv11_mbox_conf_flat")  
            ||!strcmp(layerName,"conv11_mbox_loc_flat")  
            ||!strcmp(layerName,"conv13_mbox_loc_flat")  
            ||!strcmp(layerName,"conv13_mbox_conf_flat")  
            ||!strcmp(layerName,"conv14_2_mbox_loc_flat")  
            ||!strcmp(layerName,"conv14_2_mbox_conf_flat")  
            ||!strcmp(layerName,"conv15_2_mbox_loc_flat")  
            ||!strcmp(layerName,"conv15_2_mbox_conf_flat")  
            ||!strcmp(layerName,"conv16_2_mbox_loc_flat")  
            ||!strcmp(layerName,"conv16_2_mbox_conf_flat")  
            ||!strcmp(layerName,"conv17_2_mbox_loc_flat")  
            ||!strcmp(layerName,"conv17_2_mbox_conf_flat")  
            ||!strcmp(layerName,"mbox_conf_flatten")  
        ){  
            _nvPlugins[layerName] = (plugin::INvPlugin*)(new FlattenLayer(serialData, serialLength));  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"mbox_loc")){  
            _nvPlugins[layerName] = plugin::createConcatPlugin (serialData, serialLength);  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"mbox_conf")){  
            _nvPlugins[layerName] = plugin::createConcatPlugin (serialData, serialLength);  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"mbox_priorbox")){  
            _nvPlugins[layerName] = plugin::createConcatPlugin (serialData, serialLength);  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName, "mbox_conf_softmax"))
        {
             _nvPlugins[layerName] = (plugin::INvPlugin*)(new SofaMaxChannelLayer(2,serialData, serialLength));
             return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName, "layer85-upsample"))
        {
             _nvPlugins[layerName] = (plugin::INvPlugin*)(new UpsampleLayer(serialData, serialLength,2));
             return _nvPlugins.at(layerName);

 
  
        }else if(!strcmp(layerName, "layer97-upsample"))
        {
             _nvPlugins[layerName] = (plugin::INvPlugin*)(new UpsampleLayer(serialData, serialLength,2));
             return _nvPlugins.at(layerName);
 
  
        }else if (    !strcmp(layerName, "layer2-act")  
            || !strcmp(layerName, "layer1-act")  
            || !strcmp(layerName, "layer104-act")  
            || !strcmp(layerName, "layer0-act")  
            || !strcmp(layerName, "layer28-act")  
            || !strcmp(layerName, "layer103-act")  
            || !strcmp(layerName, "layer102-act")  
            || !strcmp(layerName, "layer101-act")  
            || !strcmp(layerName, "layer100-act")  
            || !strcmp(layerName, "layer99-act")  
            || !strcmp(layerName, "layer96-act")  
            || !strcmp(layerName, "layer92-act")  
            || !strcmp(layerName, "layer91-act")  
            || !strcmp(layerName, "layer90-act")  
            || !strcmp(layerName, "layer89-act")  
            || !strcmp(layerName, "layer88-act")  
            || !strcmp(layerName, "layer87-act")  
            || !strcmp(layerName, "layer84-act")  
            || !strcmp(layerName, "layer80-act")  
            || !strcmp(layerName, "layer79-act")  
            || !strcmp(layerName, "layer78-act")  



            || !strcmp(layerName, "layer77-act")  
            || !strcmp(layerName, "layer76-act")  
            || !strcmp(layerName, "layer75-act")  
            || !strcmp(layerName, "layer73-act")  
            || !strcmp(layerName, "layer72-act")  
            || !strcmp(layerName, "layer70-act")  
            || !strcmp(layerName, "layer69-act")  
            || !strcmp(layerName, "layer67-act")  
            || !strcmp(layerName, "layer66-act")  
            || !strcmp(layerName, "layer64-act")  
            || !strcmp(layerName, "layer63-act")  
            || !strcmp(layerName, "layer62-act")  
            || !strcmp(layerName, "layer60-act")  
            || !strcmp(layerName, "layer59-act")  
            || !strcmp(layerName, "layer57-act")  
            || !strcmp(layerName, "layer56-act")  
            || !strcmp(layerName, "layer54-act")  
            || !strcmp(layerName, "layer53-act")  
//

            || !strcmp(layerName, "layer51-act")  
            || !strcmp(layerName, "layer50-act")  
            || !strcmp(layerName, "layer48-act")  
            || !strcmp(layerName, "layer47-act")  
            || !strcmp(layerName, "layer45-act")  
            || !strcmp(layerName, "layer44-act")  
            || !strcmp(layerName, "layer42-act")  
            || !strcmp(layerName, "layer41-act")  
            || !strcmp(layerName, "layer39-act")  
            || !strcmp(layerName, "layer38-act")  
            || !strcmp(layerName, "layer37-act")  
            || !strcmp(layerName, "layer35-act")  
            || !strcmp(layerName, "layer34-act")  
            || !strcmp(layerName, "layer32-act")  
            || !strcmp(layerName, "layer31-act")  
            || !strcmp(layerName, "layer29-act")  
            || !strcmp(layerName, "layer26-act")  
            || !strcmp(layerName, "layer25-act")  



            || !strcmp(layerName, "layer23-act")  
            || !strcmp(layerName, "layer22-act")  
            || !strcmp(layerName, "layer20-act")  
            || !strcmp(layerName, "layer19-act")  
            || !strcmp(layerName, "layer17-act")  
            || !strcmp(layerName, "layer16-act")  
            || !strcmp(layerName, "layer14-act")  
            || !strcmp(layerName, "layer13-act")  
            || !strcmp(layerName, "layer12-act")  
            || !strcmp(layerName, "layer10-act")  
            || !strcmp(layerName, "layer9-act")  
            || !strcmp(layerName, "layer7-act")  
            || !strcmp(layerName, "layer6-act")  
            || !strcmp(layerName, "layer5-act")  
            || !strcmp(layerName, "layer3-act")  
       ){

         _nvPlugins[layerName] = (plugin::INvPlugin*)(new LReluLayer(serialData, serialLength,0.1));
             return _nvPlugins.at(layerName);


        }else{  
            assert(0);  
            return nullptr;  
        }  
    }  
  
    void destroyPlugin()  
    {  
        for (auto it=_nvPlugins.begin(); it!=_nvPlugins.end(); ++it){  
            std::cout<<it->first<<std::endl;
            it->second->destroy();  
            _nvPlugins.erase(it);  
        }  
    }  
  
  
private:  
  
        std::map<std::string, plugin::INvPlugin*> _nvPlugins;   
};  
  
  
  
#endif  

