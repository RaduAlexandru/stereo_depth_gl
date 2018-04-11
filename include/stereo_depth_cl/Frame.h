#pragma once

#include <Eigen/Dense>
#include <opencv2/highgui/highgui.hpp>

class Frame {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    cv::Mat rgb;
    cv::Mat gray;
    cv::Mat rgb_small;
    cv::Mat classes;
    cv::Mat probs;
    cv::Mat mask;
    cv::Mat depth;
    unsigned long long int timestamp;
    cv::Mat classes_original_size;
    cv::Mat probs_original_size;
    Eigen::Matrix3d K;
    Eigen::Affine3d tf_cam_world;

    int cam_id; //id of the camera depending on how many cameras we have (it gos from 0 to 1 in the case of stereo)
    int frame_id;
};
