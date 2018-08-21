#pragma once
//C++
#include <iosfwd>
#include <memory>
#include <atomic>

//OpenCV
#include <opencv2/highgui/highgui.hpp>


//My stuff
#include "stereo_depth_gl/Mesh.h"
#include "stereo_depth_gl/Frame.h"

//halide
#include "Halide.h"


//forward declarations
class Profiler;


class DepthEstimatorHalide{
public:
    DepthEstimatorHalide();
    ~DepthEstimatorHalide(); //needed so that forward declarations work
    void init_halide();

    void compute_depth(const Frame& frame_left, const Frame& frame_right);
    // Halide::Func convolution(Halide::Func f, Halide::Func hx, Halide::Expr kernel_width, Halide::Expr kernel_height);
    // void run_speed_test(Frame& frame);
    // void run_speed_test_bright(Frame& frame);
    // void run_speed_test_sobel(Frame& frame);




    //objects
    std::shared_ptr<Profiler> m_profiler;


    //databasse
    bool is_first_frame=true;
    Frame first_frame; //used for debugging
    cv::Mat m_undistort_map_x; //vector containing the undistort map in x direction for cam right
    cv::Mat m_undistort_map_y; //vector containing the undistort map in x direction for cam right
    cv::Mat debug_img_left;
    cv::Mat debug_img_right;

    //params



private:

    cv::Mat undistort_rectify_image(const cv::Mat img, const Frame& frame_left, const Frame& frame_right);
    cv::Mat guided_filter(const cv::Mat& I, const cv::Mat& p, const float radius, const float eps);
    Halide::Func boxfilter(const Halide::Func& I, const int radius);
    Halide::Func mul_elem(const Halide::Func& lhs, const Halide::Func& rhs); //element wise multiplication

};




#define TIME_SCOPE(name)\
    TIME_SCOPE_2(name,m_profiler);

#define TIME_START(name)\
    TIME_START_2(name,m_profiler);

#define TIME_END(name)\
    TIME_END_2(name,m_profiler);
;
