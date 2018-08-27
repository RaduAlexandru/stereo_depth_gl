#pragma once

#include <Eigen/Dense>
#include <opencv2/highgui/highgui.hpp>

class Frame {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    cv::Mat rgb;
    cv::Mat gray;
    cv::Mat grad_x;
    cv::Mat grad_y;
    cv::Mat gray_with_gradients; //gray image and grad_x and grad_y into one RGB32F image, ready to be uploaded to opengl

    cv::Mat mask;
    cv::Mat depth;
    unsigned long long int timestamp;
    Eigen::Matrix3f K;
    Eigen::Matrix<float, 5, 1> distort_coeffs;
    Eigen::Affine3f tf_cam_world;

    int cam_id; //id of the camera depending on how many cameras we have (it gos from 0 to 1 in the case of stereo)
    int frame_idx; //frame idx monotonically increasing

    bool is_last=false; //is true when this image is the last in the dataset
};
