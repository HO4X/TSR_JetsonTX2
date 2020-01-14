/*
*   Victor Kallenbach
*/
#include <iostream>

#include "traffic_sign.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#pragma once 

using namespace cv; 
using namespace std; 

class ShapeDetector {

public:
    ShapeDetector(); 
    vector<TrafficSign> processMat(Mat& inputCannyImage); 

private: 
    float calculateAngle(Point p1, Point p2, Point p3); 
    vector<Vec4i> hierarchy_;
    vector<vector<Point> > contours_;
    std::vector<cv::Point> approx_;

    static constexpr int32_t min_volume_sign_ = 5; 
};