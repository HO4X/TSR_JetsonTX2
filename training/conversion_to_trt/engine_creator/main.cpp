#include <iostream>

#include </usr/local/cuda-10.0/include/cuda_runtime_api.h>

#include "./include/NvInfer.h"
#include "./include/NvInferPlugin.h"
#include "./include/NvOnnxParser.h"
#include "./include/NvUffParser.h"


#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace nvinfer1; //Nvidia ns

#define CHECK(status)                             \
    do                                            \
    {                                             \
        auto ret = (status);                      \
        if (ret != 0)                             \
        {                                         \
            std::cout << "Cuda failure: " << ret; \
            abort();                              \
        }                                         \
    } while (0)

class Logger : public ILogger           
 {
     void log(Severity severity, const char* msg) override
     {
         // suppress info-level messages
         if (severity != Severity::kINFO)
             std::cout << msg << std::endl;
     }
 } gLogger;

int main(void) {    
    auto batchSize = 1; 

    cv::Mat image;
    
    //image = cv::imread("./00030_00007.ppm", CV_LOAD_IMAGE_COLOR); // Class 1
    //image = cv::imread("./00030_00026.ppm", CV_LOAD_IMAGE_COLOR); // Class 1

    image = cv::imread("./00004_00023.ppm", CV_LOAD_IMAGE_COLOR); // Class 0

    cv::resize(image, image, cv::Size(40,40));
    cv::namedWindow("Test Image"); 
    cv::imshow("Test Image", image);
    cv::waitKey(0);
    
    std::cout << "Start Engine creation" << std::endl;

    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();
    
    //Parse Network 
    std::cout << "Parse Network " << std::endl;
    /*
    auto parser = nvonnxparser::createParser(*network, gLogger);
    
    parser->parseFromFile("ts_classifier.onnx", 2);
    */

    builder->setMaxBatchSize(batchSize);
    builder->setMaxWorkspaceSize(1 << 20);
    
    auto parser = nvuffparser::createUffParser();

    //Set output & input tensors before parsing !
    parser->registerInput("mobilenet_1.00_224_input",  Dims3(40, 40, 3), nvuffparser::UffInputOrder::kNHWC);
    parser->registerOutput("activation/Softmax"); 

    auto parser_succsess = parser->parse("../../tf_classifier/frozen_model_v1_40x40.uff", *network, nvinfer1::DataType::kFLOAT); 

    //parse 
    if (!parser_succsess) {
        std::cout << "Failed to parse UFF" << std::endl;
        builder->destroy();
        parser->destroy();
        network->destroy();
        return 1;
    }

    std::cout << "Build Network Engine" << std::endl; 
    ICudaEngine* engine = builder->buildCudaEngine(*network);

    std::cout << " Max batch size: " << engine->getMaxBatchSize() << std::endl;

    parser->destroy();
    
    builder->destroy();

    std::cout << "Finished engine creation " << std::endl;
    
    //Store Engine
    IHostMemory *serializedModel = engine->serialize();

    //Inference 
    IExecutionContext *context = engine->createExecutionContext();

    float* input = new float[3 * 160 * 160]; 

    cv::Mat flat_img = image.reshape(1,1);
    cv::Mat float_img; 

    flat_img.convertTo(float_img,CV_32F, 1/255.0);

    cv::namedWindow("Float image"); 
    cv::imshow("Float image", float_img);
    cv::waitKey(0);

    std::vector< float > array((float*)float_img.data, (float*)float_img.data + float_img.rows * float_img.cols);

    input = (float*)(float_img.ptr<float>(0));

    float* output = new float[1000];

    for(int32_t iter; iter < 1000; iter++) output[iter] = 0; //Init 

    float ms = 0; 

    //Ptrs to GPU-Buffers
    void* buffers[2];

    // create GPU buffers and a stream
    Dims3 inputDims = static_cast<Dims3&&>(context->getEngine().getBindingDimensions(0));
    Dims3 outputDims = static_cast<Dims3&&>(context->getEngine().getBindingDimensions(1));

    std::cout << "Info \t| Expect that first layer is the input layer & last layer is the output layer " << std::endl;
    for(int i = 0; i < 2; i ++) {
        auto tensor = context->getEngine().getBindingDimensions(i);       
        std::cout << "OutputDim \t | " << tensor.nbDims << std::endl;
        std::cout << "        1.\t | " << tensor.d[0] << std::endl;
        std::cout << "        2.\t | " << tensor.d[1] << std::endl;
        std::cout << "        2.\t | " << tensor.d[2] << std::endl;
    }
    
    network->destroy();

    size_t inputSize = batchSize * inputDims.d[0] * inputDims.d[1] * inputDims.d[2] * sizeof(float); 
    size_t outputSize = batchSize * outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * sizeof(float);
    
    std::cout << "\ncudaMalloc \t | inputSize: " << inputSize / sizeof(float) << " \t | outputSize: " << outputSize / sizeof(float) << "\n\n"; 
    
    CHECK(cudaMalloc(&buffers[0], inputSize));
    CHECK(cudaMalloc(&buffers[1], outputSize));

    CHECK(cudaMemcpy(buffers[0], input, inputSize, cudaMemcpyHostToDevice));

    std::cout << "create stream" << std::endl; 
    
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    
    cudaEvent_t start, end;
    
    CHECK(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
    CHECK(cudaEventCreateWithFlags(&end, cudaEventBlockingSync));
    
    std::cout << "run inference" << std::endl;

    cudaEventRecord(start, stream);

    context->enqueue(1, buffers, stream, nullptr);

    cudaEventRecord(end, stream);

    std::cout << "finished inference" << std::endl;

    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    CHECK(cudaMemcpy(output, buffers[1], outputSize, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(buffers[0]));
    CHECK(cudaFree(buffers[1]));

    CHECK(cudaStreamDestroy(stream));

    std::cout << "Finished Enginecreation | Run Time (ms) -> "<< ms << std::endl;
    typedef std::map<float, int > output_map;
    output_map map; 
    for(int i = 0; i < outputSize / sizeof(float); i++) {
        map[output[i]] = i;  
    }

    //Check sum 
    float sum_of_all_predictions = 0; 
    for(int i = 0; i < outputSize / sizeof(float); i++) {
        sum_of_all_predictions += output[i];  
    }

    std::cout << "Summer aller Prädiktionen -> " << sum_of_all_predictions << std::endl; 
    std::cout << std::fixed;

    auto iter = map.end(); 
    for(int i = 0; i < 5; i++) {
        iter--; 
        std::cout << " Class " << iter->second << " " << iter->first << std::endl; 
    }
    std::cout << std::endl;
    return 0; 
}
