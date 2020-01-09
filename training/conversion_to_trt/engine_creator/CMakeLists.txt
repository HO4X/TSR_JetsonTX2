cmake_minimum_required (VERSION 2.6)

project (CPPEngineCreator)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

cuda_add_executable(CPPEngineCreator main.cpp)

TARGET_LINK_LIBRARIES(CPPEngineCreator ${CUDA_LIBS})
TARGET_LINK_LIBRARIES(CPPEngineCreator ${OpenCV_LIBS})