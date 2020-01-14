#include "shape_detector.h"
#include "traffic_sign.h"

ShapeDetector::ShapeDetector() {
    // ToDo: add setter for params
}

vector<TrafficSign> ShapeDetector::processMat(Mat& inputCannyImage) {
    vector<TrafficSign> traffic_sign_canidates_; 
    // Call cv function 
    findContours( inputCannyImage, 
                  contours_, 
                  hierarchy_, 
                  CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    /* recognize shapes */
    for (int i = 0; i < contours_.size(); i++)
    {
        TrafficSign traffic_sign_canidate_; 

        // approx_imate contour with accuracy proportional
        // to the contour perimeter
        approxPolyDP(cv::Mat(contours_[i]), approx_, arcLength(Mat(contours_[i]), true)*0.02, true);

        // Skip small or non-convex objects
        if (std::fabs(cv::contourArea(contours_[i])) < 100 || !cv::isContourConvex(approx_))
            continue;

        if (approx_.size() == 3){
            // triangle
            cv::Rect r  = cv::boundingRect(contours_[i]);

            // padded bounding box
            cv::Rect rtemp( r.x - r.width * 0.15, 
                            r.y - r.height * 0.15, 
                            r.width + r.width * 0.30, 
                            r.height + r.height * 0.30 );

            // Number of vertices of polygonal curve
            unsigned long int vtc = approx_.size();

            // Get the cosines of all corners
            std::vector<double> cos;
            for (int j = 2; j <= vtc; j++) cos.push_back(
                calculateAngle(approx_[j%vtc], approx_[j-2], approx_[j-1]));

            // Sort ascending the cosine values
            std::sort(cos.begin(), cos.end());

            // Get the lowest and the highest cosine
            double mincos = cos.front();
            double maxcos = cos.back();

            if( r.width >= min_volume_sign_ && 
                r.height >= min_volume_sign_ && 
                mincos >= 0.35 && mincos <= 0.65 && 
                maxcos >= 0.35 && maxcos <= 0.65 ) {
                //valid triangle!
                try {
                    traffic_sign_canidate_.shape_ = TrafficSign::ShapeForm::kTriangle;
                    traffic_sign_canidate_.bounding_box_ = rtemp; 
                    traffic_sign_canidates_.push_back(traffic_sign_canidate_);
                } catch (Exception e) {
                    printf("ROI out of bounds!");
                }
            }
        }
        else if (approx_.size() >= 4 && approx_.size() <= 6) {
            // Number of vertices of polygonal curve
            unsigned long int vtc = approx_.size();
           
            // Get the cosines of all corners
            std::vector<double> cos;
            for (int j = 2; j <= vtc; j++) cos.push_back(
                calculateAngle(approx_[j%vtc], approx_[j-2], approx_[j-1]));
    
            // Sort ascending the cosine values
            std::sort(cos.begin(), cos.end());
        
            // Get the lowest and the highest cosine
            double mincos = cos.front();
            double maxcos = cos.back();

            // Use the degrees obtained above and the number of vertices
            // to determine the shape of the contour
            if (vtc == 4 && mincos >= -0.1 && maxcos <= 0.3) {
                //setLabel(dst, "RECT", contours_[i]);
                cv::Rect bounding_box_rect_  = cv::boundingRect(contours_[i]);
                if(bounding_box_rect_.width >= min_volume_sign_ && 
                bounding_box_rect_.height >= min_volume_sign_) {
                    //Valid Rect
                    traffic_sign_canidate_.shape_ = TrafficSign::ShapeForm::kRect;
                    traffic_sign_canidate_.bounding_box_ = bounding_box_rect_; 
                    traffic_sign_canidates_.push_back(traffic_sign_canidate_);
                }
            }/* //
            else if (vtc == 5 && mincos >= -0.34 && maxcos <= -0.27)
                //setLabel(dst, "PENTA", contours_[i]);
            else if (vtc == 6 && mincos >= -0.55 && maxcos <= -0.45)
                //setLabel(dst, "HEXA", contours_[i]);*/
        } else {
            // Detect and label circles
            double area = cv::contourArea(contours_[i]);
            cv::Rect bounding_box_circle_  = cv::boundingRect(contours_[i]);

            // wrong !
            int radius = bounding_box_circle_.width / 2;
            
            if(bounding_box_circle_.width >= min_volume_sign_ && 
            bounding_box_circle_.height >= min_volume_sign_) {
                /*IsolatedSign = currentFrameColor(Rect(r));
                imshow("Circle Traffic Sign",IsolatedSign);
                rectangle(currentFrame, r, cv::Scalar(0, 255, 0));*/
                traffic_sign_canidate_.shape_ = TrafficSign::ShapeForm::kCircle;
                traffic_sign_canidate_.bounding_box_ = bounding_box_circle_; 
                traffic_sign_canidates_.push_back(traffic_sign_canidate_);
            }
            /*
            if (std::abs(1 - ((double)r.width / r.height)) <= 0.2 && std::abs(1 - (area / (CV_PI * std::pow(radius, 2)))) <= 0.2) {
                //setLabel(dst, "CIR", contours_[i]);
            } */
        }
    }
    //ToDo: sort out nested detections 
    return traffic_sign_canidates_;
}

float ShapeDetector::calculateAngle(Point pt0, Point pt1, Point pt2) {
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}