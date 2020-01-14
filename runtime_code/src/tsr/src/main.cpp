//
//  main.cpp
//
//  Created by Victor Kallenbach on 29.03.18.
//

/*
*   Parameters
*/
//#define benchmark_algo
#define minVolumeSign 20
#define log_timings

#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include "shape_detector.h"
#include "shape_classifier.h"
#include "traffic_sign.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <cmath>
#include <chrono>

using namespace cv;
using namespace std;

#define PLAY 0
#define PAUSE 1

int state = 0;

ShapeDetector shape_detector_;
ShapeClassifier shape_classifier_;

std::vector<TrafficSign> shape_detection_results_; 

/**
 * Helper function to display text 
 */
void setLabel(cv::Mat& im, const std::string label, std::vector<cv::Point>& contour)
{
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.5;
    int thickness = 1;
    int baseline = 0;

    
    cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    cv::Rect r = cv::boundingRect(contour);

    
    cv::Point pt(r.x + ((r.width - text.width) / 2), r.y + ((r.height + text.height) / 2));
    cv::rectangle(im, pt + cv::Point(0, baseline), pt + cv::Point(text.width, -text.height), CV_RGB(255,255,255), CV_FILLED);
    cv::putText(im, label, pt, fontface, scale, CV_RGB(0,0,0), thickness, 8);
}

/*
*   Should sort out proposals inside other proposals
*/
void filterRegionProposals(std::vector<TrafficSign>& proposals) {
    for(int32_t index =  0; index < proposals.size(); index++) {
        auto parent_bounding_box = proposals[index].bounding_box_;
            
        if(index + 1 < proposals.size()) {
            for(int32_t index_child = index + 1; index_child < proposals.size(); index_child++) {

                auto child_bounding_box = proposals[index_child].bounding_box_;
                //Check if child bbox is in the parent bbox
                if( ( child_bounding_box & parent_bounding_box ) == parent_bounding_box )  {
                    //Match!
                    proposals.erase(proposals.begin() + index_child);
                    std::cout << "Ciao!" << std::endl; 
                }

            }
        }

    }
} 

void imageCallback(const sensor_msgs::Image::ConstPtr& msg)
{
    namespace enc = sensor_msgs::image_encodings;
    //ROS_INFO("I heard: [%s]", msg->header.seq);
    std::cout << "callback was called " << std::endl; 
    cv_bridge::CvImagePtr cv_ptr;
    //cv_ptr = cv_bridge::toCvShare(msg, enc::BGR8);
    /*try
    {
        if (enc::isColor(msg->encoding))
            cv_ptr = cv_bridge::toCvShare(msg, enc::BGR8);
        else
            cv_ptr = cv_bridge::toCvShare(msg, enc::MONO8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }*/
    //std::cout << cv_ptr->header << std::endl ; 
    //cv::Mat mat_received = cv_ptr->image;
    //imshow("Processed Image", mat_received);
    //waitKey(1);
}


int main(int argc, char ** argv) {
    shape_classifier_.init();

    // Offline modus -> without ros subscriber
    if(argc > 0) {
        /*VideoCapture* inputVideo;

        if(argv[1][0] == 'v') {
            std::cout << "Use input video" << std::endl;
            inputVideo = new VideoCapture("/home/victor/Downloads/record_bright2_out_videos/record_bright2_square_undistorted.avi"); // USB - Webcam
        } else {
            std::cout << "Use webcam" << std::endl;
            inputVideo = new VideoCapture(1);
        }*/
        VideoCapture* inputVideo = new VideoCapture(1);

        Mat currentFrame, currentFrameColor, currentFrameOrginal, cannyFrame, contourFrame,blurFrame, correctedFrame, drawing = Mat::zeros( cannyFrame.size(), CV_8UC3 );

        Mat IsolatedSign, RectSign, CircleSign, TriangleSign, RoadIsolated = Mat::zeros( cannyFrame.size(), CV_8UC3 );

        vector<vector<Point> > contours;
        
        vector<Vec4i> hierarchy;

        int trshold1 = 150; 
        int trshold2 = 220;

        string TrackbarName1;
        string TrackbarName2;
        
        float fps, fps_avg = 0; 

        /// Create Windows
        //namedWindow("Treshold 1", 1);
        //namedWindow("Treshold 2", 1);
        //createTrackbar( TrackbarName1, "Treshold 1", &trshold1, 1024 );
        //createTrackbar( TrackbarName2, "Treshold 2", &trshold2, 1024 );

        auto current_ticks = std::chrono::high_resolution_clock::now();
        auto  cvtColor_start = current_ticks;
        auto  cvtColor_end = current_ticks;

        auto  blur_start = current_ticks;
        auto  blur_end = current_ticks;

        auto  canny_start = current_ticks;
        auto  canny_end = current_ticks;

        auto start_processing_time = current_ticks;

        if(inputVideo->read(currentFrameOrginal)) std::cout << currentFrameOrginal.size() << std::endl; 

        while(true){
            fps++;
            if(state == PLAY) {
                #ifndef benchmark_algo
                    if(!inputVideo->read(currentFrameOrginal)) break;
                #endif
                start_processing_time = std::chrono::high_resolution_clock::now();
                currentFrame = currentFrameOrginal; 
                
                cv::resize(currentFrameOrginal, currentFrame, cv::Size(), 1, 1);
                currentFrameColor = currentFrame.clone();
                
                // Convert to GREY for Edge detection
                cvtColor_start = std::chrono::high_resolution_clock::now();
                cvtColor( currentFrame, currentFrame, CV_BGR2GRAY );
                cvtColor_end = std::chrono::high_resolution_clock::now();

                //equalizeHist(currentFrame, correctedFrame);

                blur_start = std::chrono::high_resolution_clock::now();
                blur(currentFrame, blurFrame, Size(3,3));
                blur_end = std::chrono::high_resolution_clock::now();

                // Calculate Tresholds https://stackoverflow.com/questions/18194870/canny-edge-image-noise-removal
                //drawHistogram(correctedFrame);
                Mat temp;
                double otsu_thresh_val = cv::threshold( blurFrame, temp, 0, 255, CV_THRESH_OTSU );
                double high_thresh_val  = otsu_thresh_val,
                lower_thresh_val = otsu_thresh_val * 0.5;
                
                canny_start = std::chrono::high_resolution_clock::now();
                Canny(blurFrame, cannyFrame, lower_thresh_val, high_thresh_val, 3);
                canny_end = std::chrono::high_resolution_clock::now();
            }
            
            //Find shapes
            shape_detection_results_ = shape_detector_.processMat(cannyFrame); 

            //filter out nested shapes
            filterRegionProposals(shape_detection_results_);
            
            //Classify shapes 
            shape_classifier_.classifiyShapes(currentFrameColor, shape_detection_results_);

            //Show image 
            //imshow("Canny Image", cannyFrame);
            imshow("Canny Image", cannyFrame);
            imshow("Processed Image", currentFrameColor);

            auto time_diff = chrono::duration_cast<chrono::milliseconds>( std::chrono::high_resolution_clock::now() - current_ticks ).count(); 
            if( time_diff > 1000 /*ms*/ ) {
                current_ticks = std::chrono::high_resolution_clock::now();
                fps_avg = fps; 
                fps = 0; 
            }

            #ifdef log_timings 
            std::cout << "Found shapes" << shape_detection_results_.size() << " \tFPS: " << fps_avg << "\tprocessing time ms " 
            << chrono::duration_cast<chrono::milliseconds>( std::chrono::high_resolution_clock::now() - start_processing_time ).count() 
            << "\t cvtColor ms "<< chrono::duration_cast<chrono::milliseconds>( cvtColor_end - cvtColor_start ).count() 
            << "\t blur ms "<< chrono::duration_cast<chrono::milliseconds>( blur_end - blur_start ).count() 
            << "\t canny ms "<< chrono::duration_cast<chrono::milliseconds>( canny_end - canny_start ).count() << std::endl;
            #endif

            char c = waitKey(1);
        }
    } else {
        std::cout << "Starting in ros mode" << std::endl; 
        // else online mode 
        ros::init(argc, argv, "tsr");
        ros::NodeHandle n;
        ros::Subscriber sub = n.subscribe("/camera/image/postprocessed/undistorted/original", 1, imageCallback);
        ros::spin();
    }
    return 0;
}
