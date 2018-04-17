#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <memory>
#include <string.h>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "common.h"
#include "PluginFactory.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>



using namespace nvinfer1;
using namespace nvcaffeparser1;
// stuff we know about the network and the caffe input/output blobs
static const int INPUT_H = 416;
static const int INPUT_W = 416;
static const int INPUT_C = 3;

static const int TIMING_ITERATIONS = 1000;
struct PPM
{
	std::string magic, fileName;
	int h, w, max;
	uint8_t buffer[INPUT_C*INPUT_H*INPUT_W];
};
#if 0
    box3 = out['layer105-conv']
    box2 = out['layer93-conv']
    box1 = out['layer81-conv']
    #box1=np.transpose(box1, (0,2,3,1))
    ##box2=np.transpose(box2, (0,2,3,1))
    #box3=np.transpose(box3, (0,2,3,1))
 
    print(box1.shape)
    print(box2.shape)
    print(box3.shape)

(1, 255, 13, 13)
(1, 255, 26, 26)
(1, 255, 52, 52)
#endif


static const int OUTPUT_SIZE0 = 255*13*13;
static const int OUTPUT_SIZE1 = 255*26*26;
static const int OUTPUT_SIZE2 = 255*52*52;


static Logger gLogger;

const char* INPUT_BLOB_NAME   = "data";

const char* OUTPUT_BLOB_NAME0 = "layer81-conv";//"mbox_loc";//"detection_out";
const char* OUTPUT_BLOB_NAME1 = "layer93-conv";//"detection_out";
const char* OUTPUT_BLOB_NAME2 = "layer105-conv";//"detection_out";

struct Profiler : public IProfiler
{
	typedef std::pair<std::string, float> Record;
	std::vector<Record> mProfile;

	virtual void reportLayerTime(const char* layerName, float ms)
	{
		auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record& r){ return r.first == layerName; });
		if (record == mProfile.end())
			mProfile.push_back(std::make_pair(layerName, ms));
		else
			record->second += ms;
	}

	void printLayerTimes()
	{
		float totalTime = 0;
		for (size_t i = 0; i < mProfile.size(); i++)
		{
			printf("%-40.40s %4.3fms\n", mProfile[i].first.c_str(), mProfile[i].second / TIMING_ITERATIONS);
			totalTime += mProfile[i].second;
		}
		printf("Time over all layers: %4.3f\n", totalTime / TIMING_ITERATIONS);
	}

} gProfiler;


std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs{"data/samples/mnist/", "data/mnist/"};
    return locateFile(input, dirs);
}

// simple PGM (portable greyscale map) reader
void readPGMFile(const std::string& filename,  uint8_t buffer[INPUT_H*INPUT_W])
{
    readPGMFile(locateFile(filename), buffer, INPUT_H, INPUT_W);
}

void caffeToGIEModel(const std::string& deployFile,					// name for caffe prototxt
					 const std::string& modelFile,					// name for model 
					 const std::vector<std::string>& outputs,		// network outputs
					 unsigned int maxBatchSize,						// batch size - NB must be at least as large as the batch we want to run with)
					 nvcaffeparser1::IPluginFactory* pluginFactory,	// factory for plugin layers
					 IHostMemory *&gieModelStream)					// output stream for the GIE model
{
	// create the builder
        std::cout << "start parsing model..." << std::endl;
	IBuilder* builder = createInferBuilder(gLogger);
        std::cout << "start1 parsing model..." << std::endl;
	// parse the caffe model to populate the network, then set the outputs
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser* parser = createCaffeParser();
	parser->setPluginFactory(pluginFactory);
        std::cout << "start2 parsing model..." << std::endl;
	bool fp16 = builder->platformHasFastFp16();
	const IBlobNameToTensor* blobNameToTensor = parser->parse( deployFile.c_str(),
															   modelFile.c_str(),
															  *network,
															  fp16 ? DataType::kHALF : DataType::kFLOAT);
        std::cout << "start3 parsing model..." << std::endl;
	// specify which tensors are outputs
	for (auto& s : outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));

        std::cout << "start4 parsing model..." << std::endl;
	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(10 << 20);
	builder->setHalf2Mode(fp16);

	ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);
        std::cout << "start5 parsing model..." << std::endl;
	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy();
        std::cout << "start6 parsing model..." << std::endl;
	// serialize the engine, then close everything down
	gieModelStream = engine->serialize();

	engine->destroy();
	builder->destroy();
	shutdownProtobufLibrary();
    
        std::cout << "End parsing model.qq.." << std::endl;
}
 
void doInference(IExecutionContext& context, float* input, float* output0, float* output1, float* output2, int batchSize)
{
	const ICudaEngine& engine = context.getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
    std::cout<<engine.getNbBindings()<<std::endl;
	assert(engine.getNbBindings() == 4);
	void* buffers[4];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME), 
	outputIndex0 = engine.getBindingIndex(OUTPUT_BLOB_NAME0),
	outputIndex1 = engine.getBindingIndex(OUTPUT_BLOB_NAME1),
	outputIndex2 = engine.getBindingIndex(OUTPUT_BLOB_NAME2);
#if 1
    std::cout << "1---------------------" << std::endl;
	// create GPU buffers and a stream
	 cudaMalloc(&buffers[outputIndex0],  1 * OUTPUT_SIZE0 * sizeof(float)) ; // bbox_pred
	 cudaMalloc(&buffers[outputIndex1],  1 * OUTPUT_SIZE1 * sizeof(float)) ;  // cls_prob
	 cudaMalloc(&buffers[outputIndex2],  1 * OUTPUT_SIZE2 * sizeof(float)) ;                // rois

	 cudaMalloc(&buffers[inputIndex],    1 * INPUT_C*INPUT_H * INPUT_W * sizeof(float)) ;

         std::cout << "2---------------------" << std::endl;
	 cudaStream_t stream;
	 cudaStreamCreate(&stream) ;
         std::cout << "3---------------------" << std::endl;

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	cudaMemcpyAsync(buffers[inputIndex], input, batchSize *INPUT_C* INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream) ;
	context.enqueue(batchSize, buffers, stream, nullptr);

	cudaMemcpyAsync(output0, buffers[outputIndex0], batchSize * OUTPUT_SIZE0 *sizeof(float), cudaMemcpyDeviceToHost, stream) ;
	cudaMemcpyAsync(output1, buffers[outputIndex1], batchSize * OUTPUT_SIZE1 *sizeof(float), cudaMemcpyDeviceToHost, stream) ;
	cudaMemcpyAsync(output2, buffers[outputIndex2], batchSize * OUTPUT_SIZE2 *sizeof(float), cudaMemcpyDeviceToHost, stream) ;

	cudaStreamSynchronize(stream);

	// release the stream and the buffers
	cudaStreamDestroy(stream);
	 cudaFree(buffers[inputIndex]);
	 cudaFree(buffers[outputIndex0]);
        cudaFree(buffers[outputIndex1]);
 	 cudaFree(buffers[outputIndex2]);
    #endif
}


// simple PPM (portable pixel map) reader
void readPPMFile(const std::string& filename, PPM& ppm)
{
	ppm.fileName = filename;
	std::ifstream infile( filename, std::ifstream::binary);
	infile >> ppm.magic >> ppm.w >> ppm.h >> ppm.max;
    std::cout <<  ppm.magic<<"---"<< ppm.w<<"---"<<ppm.h<<"----"<<ppm.max<< "\n\n\n" << std::endl;
	infile.seekg(1, infile.cur);
	infile.read(reinterpret_cast<char*>(ppm.buffer), ppm.w * ppm.h * 3);
}


cv::Mat  Preprocess(const cv::Mat& img) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  int num_channels_=3;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != cv::Size(416,416))
    cv::resize(sample, sample_resized, cv::Size(416,416));
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1); 


//  cv::imshow( "Display window1", sample_resized);
 // cv::waitKey(0);

   // cv::cvtColor(sample_resized, sample_resized, cv::COLOR_BGR2RGB);
 //   sample_float=(sample_float-127.5);
  //  sample_float*=1/127.0;

    return sample_resized;
}
using namespace std;

float prob1[OUTPUT_SIZE0]={0};
float prob2[OUTPUT_SIZE1]={0};
float prob3[OUTPUT_SIZE2]={0};

int main(int argc, char** argv)
{
	// create a GIE model from the caffe model and serialize it to a stream
	PluginFactory pluginFactory;
	IHostMemory *gieModelStream{ nullptr };
	caffeToGIEModel("yyy.prototxt", "yyy.caffemodel", std::vector < std::string > { OUTPUT_BLOB_NAME0 ,OUTPUT_BLOB_NAME1,OUTPUT_BLOB_NAME2}, 1, &pluginFactory, gieModelStream);
	pluginFactory.destroyPlugin();
    std::cout << "staraaat deserializeCudaEngine model..." << std::endl;
    
    
    cv::Mat img = cv::imread("/opt/dog.jpg", 1);
    //CHECK(!img.empty()) << "Unable to decode image " << "/opt/deep_learn/traffic_light/2018.1.4/videoShot20171214/35.jpg";
    std::cout << "\n\n\n---------------------------3" << "\n\n\n" << std::endl;
    img=Preprocess(img);
        
 
    
    PPM  imagePPM[1];
    // int num = rand() % 10;
 	readPPMFile( "Free-Converter.com-14515-5164734.ppm", imagePPM[0]);

	// print an ascii representation

	//for (int i = 0; i < INPUT_H*INPUT_W; i++)
        //	std::cout << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % INPUT_W) ? "" : "\n");

        memcpy(imagePPM[0].buffer,img.data,INPUT_C*INPUT_H*INPUT_W);

        float* data = new float[1*INPUT_C*INPUT_H*INPUT_W];
        //     memcpy(imagePPM[0].buffer,img.data,INPUT_C*INPUT_H*INPUT_W);
        // pixel mean used by the Faster R-CNN's author
        //float pixelMean[3]{ 127.5f, 127.5f, 127.5f }; // also in BGR order

        for (int i = 0, volImg = INPUT_C*INPUT_H*INPUT_W; i < 1; ++i)
        {
                for (int c = 0; c < INPUT_C; ++c)
                {
                        // the color image to input should be in RGB order
                        for (unsigned j = 0, volChl = INPUT_H*INPUT_W; j < volChl; ++j)
                                 data[i*volImg + c*volChl + j] = (float(imagePPM[i].buffer[j*INPUT_C + 2 - c]))*1/255.0;
                }
        }


    
    
    
    
	//ICaffeParser* parser = createCaffeParser();
	//IBinaryProtoBlob* meanBlob = parser->parseBinaryProto(locateFile("mnist_mean.binaryproto").c_str());
	//parser->destroy();

	// parse the mean file and 	subtract it from the image
	//const float *meanData = reinterpret_cast<const float*>(meanBlob->getData());

	//float data[INPUT_H*INPUT_W];
	//for (int i = 0; i < INPUT_H*INPUT_W; i++)
	//	data[i] = float(fileData[i])-meanData[i];

	//meanBlob->destroy();

	// deserialize the engine 
	IRuntime* runtime = createInferRuntime(gLogger);
	std::cout << "start deserializeCudaEngine model..." << std::endl;
	ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), &pluginFactory);
	std::cout << "end deserializeCudaEngine model..." << std::endl;
	IExecutionContext *context = engine->createExecutionContext();
	context->setProfiler(&gProfiler);
 

	// run inference

	doInference(*context, data, prob1,prob2,prob3, 1);

        printf("%f %f %f %f %f\n",prob1[0],prob1[1],prob1[2],prob1[3],prob1[4]);
        printf("%f %f %f %f %f\n",prob2[0],prob2[1],prob2[2],prob2[3],prob2[4]);
        printf("%f %f %f %f %f\n",prob3[0],prob3[1],prob3[2],prob3[3],prob3[4]);
	// destroy the engine
	context->destroy();
	engine->destroy();
	gProfiler.printLayerTimes();
	runtime->destroy();
	pluginFactory.destroyPlugin();

	// print a histogram of the output distribution
 

	return 1;
}
