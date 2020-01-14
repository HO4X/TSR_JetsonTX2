#pragma once

#include "traffic_sign.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include </usr/local/cuda-10.0/include/cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvInferPlugin.h"
//#include "./include/NvOnnxParser.h"
#include "NvUffParser.h"

#include <map>
#include <iostream>

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

class ShapeClassifier {
    class Logger : public ILogger           
    {
        void log(Severity severity, const char* msg) override
        {
            // suppress info-level messages
            if (severity != Severity::kINFO)
                std::cout << msg << std::endl;
        }
    } gLogger;
public: 
    ShapeClassifier(); 
    void init(); 
    void classifiyShapes(Mat& color_image, std::vector<TrafficSign>& traffic_signs );
private: 
    IRuntime* runtime;
    IExecutionContext* context;
    ICudaEngine* engine;

    //Input pointer to HOST-Memory
    float* input;
    float* output;

    //Ptrs to GPU-Buffers
    void* buffers[2];

    Dims3 inputDims;
    Dims3 outputDims;

    size_t inputSize; 
    size_t outputSize; 

    cudaStream_t stream;

    static constexpr size_t batchSize = 1; 
    static constexpr size_t net_input_size_ = 40; 
    float ms; 

    typedef std::map<float, int > output_map;
};