#include "shape_classifier.h"
#include <fstream>

ShapeClassifier::ShapeClassifier() {
    input = new float[3 * 40 * 40]; 
    output = new float[1000]; //Should be 43
}

void ShapeClassifier::init() {
    runtime = createInferRuntime(gLogger);
    std::cout << "Load GIE File . from ../engine.gie "; 

    std::ifstream gieFile("../engine_40_40_v1.gie");
    
    std::cout << ".. done" <<std::endl; 

    std::stringstream gieModelString;
    
    std::cout << "pass to ss ."; 
    gieModelString.seekg(0, gieModelString.beg); 
    gieModelString << gieFile.rdbuf();
    gieFile.close(); 
    std::cout << ".. done" <<std::endl; 

    std::cout << "get model file size ."; 
    gieModelString.seekg(0, std::ios::end); 
    int modelSize = gieModelString.tellg();
    gieModelString.seekg(0, std::ios::beg); 
    std::cout << ".. done\n\tModelsize is " << modelSize <<std::endl;

    void* serializedModelMemory = malloc(modelSize); 

    std::cout << "read ."; 
    gieModelString.read(reinterpret_cast<char*>(serializedModelMemory), modelSize);
    std::cout << ".. done" <<std::endl;

    //Parse cuda engine
    std::cout << "deserializeCudaEngine ."; 
    engine = runtime->deserializeCudaEngine(serializedModelMemory, modelSize, nullptr); 
    std::cout << ".. done" <<std::endl;

    //create execution context
    std::cout << "create execution context ."; 
    context = engine->createExecutionContext();
    std::cout << ".. done" <<std::endl;

    //Free memory again;
    free(serializedModelMemory);

    // create GPU buffers and a stream
    inputDims = static_cast<Dims3&&>(context->getEngine().getBindingDimensions(0));
    outputDims = static_cast<Dims3&&>(context->getEngine().getBindingDimensions(1));

    inputSize = batchSize * inputDims.d[0] * inputDims.d[1] * inputDims.d[2] * sizeof(float); 
    outputSize = batchSize * outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * sizeof(float);

    std::cout << "\ncudaMalloc \t | inputSize: " << inputSize / sizeof(float) << " \t | outputSize: " << outputSize / sizeof(float) << "\n\n"; 
    
    CHECK(cudaMalloc(&buffers[0], inputSize));
    CHECK(cudaMalloc(&buffers[1], outputSize));

    std::cout << "create stream.. " << std::endl; 
        
    CHECK(cudaStreamCreate(&stream));

    std::cout << " done!" << std::endl; 
}

void ShapeClassifier::classifiyShapes(Mat& color_image, std::vector<TrafficSign>& traffic_signs ) {
    for(int32_t iter = 0; iter < traffic_signs.size(); iter++) {
        rectangle(color_image, traffic_signs[iter].bounding_box_, cv::Scalar(0, 255, 0));
        
        //Get view
        cv::Mat isolated_sign = color_image(traffic_signs[iter].bounding_box_);

        //Resize to NN's input size
        cv::resize(isolated_sign, isolated_sign, cv::Size(net_input_size_,net_input_size_));
        
        //Flatten image
        cv::Mat flat_img;
        try {
            flat_img = isolated_sign.reshape(1,1);
            cv::Mat float_img; 

            //Create norm. image 
            flat_img.convertTo(float_img,CV_32F, 1/255.0);

            std::vector< float > array((float*)float_img.data, (float*)float_img.data + float_img.rows * float_img.cols);

            input = (float*)(float_img.ptr<float>(0));

            //Image uplaod 
            CHECK(cudaMemcpy(buffers[0], input, inputSize, cudaMemcpyHostToDevice));
            
            cudaEvent_t start, end;
            
            CHECK(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
            CHECK(cudaEventCreateWithFlags(&end, cudaEventBlockingSync));
            
            //std::cout << "run inference" << std::endl;

            cudaEventRecord(start, stream);

            context->enqueue(1, buffers, stream, nullptr);

            cudaEventRecord(end, stream);

            //std::cout << "finished inference" << std::endl;

            cudaEventSynchronize(end);
            cudaEventElapsedTime(&ms, start, end);
            cudaEventDestroy(start);
            cudaEventDestroy(end);

            CHECK(cudaMemcpy(output, buffers[1], outputSize, cudaMemcpyDeviceToHost));
            
            output_map map; 

            for(int i = 0; i < outputSize / sizeof(float); i++) {
                map[output[i]] = i;  
            }

            //Check sum 
            float sum_of_all_predictions = 0; 
            for(int i = 0; i < outputSize / sizeof(float); i++) {
                sum_of_all_predictions += output[i];  
            }

            //std::cout << "Summer aller Prädiktionen -> " << sum_of_all_predictions << std::endl; 
            std::cout << std::endl << std::fixed;

            auto map_iter = map.end(); 
            for(int i = 0; i < 1; i++) {
                map_iter--; 
                if(map_iter->first > 0.95f) {
                    std::cout << " Class " << map_iter->second << " " << map_iter->first << std::endl; 
                }
            }
            std::cout << "Run Time (ms) -> "<< ms << std::endl;
        } catch (...) {
            //Super bad coding style 
        }
    } 
}