#pragma once 

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv; 
using namespace std; 

class TrafficSign {
    public:
        enum class ShapeForm : int8_t {
            kTriangle, 
            kRect,
            kCircle 
        };
        // Detector class labels
        enum class TrafficSignClass : int8_t {
            kTwenty = 0,
            kThirty = 1, 
            kFifty = 2
        };

        ShapeForm shape_; 
        TrafficSignClass sign_class_label_;

        Rect bounding_box_;
        
        int32_t top_left_x_;
        int32_t top_left_y_;
        int32_t bottom_right_x_; 
        int32_t bottom_right_y_; 
};
