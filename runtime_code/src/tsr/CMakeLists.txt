cmake_minimum_required (VERSION 2.6)

project (traffic_sign_recognition)

catkin_package()

find_package(catkin REQUIRED COMPONENTS
  roscpp
  image_transport
  cv_bridge
  sensor_msgs
)

#set (CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

#Headers 
set(HEADER_FILES 

	/usr/include/aarch64-linux-gnu/NvInfer.h
	/usr/include/aarch64-linux-gnu/NvInferPlugin.h
	/usr/include/aarch64-linux-gnu/NvUffParser.h

	src/shape_detector.h
	src/traffic_sign.h
	src/shape_classifier.h
)

set(SRC_FILES 
	src/shape_detector.cpp
	src/shape_classifier.cpp
)

cuda_add_executable(traffic_sign_recognition src/main.cpp ${SRC_FILES})

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

#tsr_runnable
target_link_libraries(traffic_sign_recognition ${catkin_LIBRARIES})

TARGET_LINK_LIBRARIES(traffic_sign_recognition "-L/usr/lib/aarch64-linux-gnu/" libnvinfer.so)
TARGET_LINK_LIBRARIES(traffic_sign_recognition "-L/usr/lib/aarch64-linux-gnu/" libnvparsers.so)
TARGET_LINK_LIBRARIES(traffic_sign_recognition ${OpenCV_LIBS} ${CUDA_LIBS} )

