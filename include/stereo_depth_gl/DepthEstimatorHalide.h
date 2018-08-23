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
    Mesh m_mesh;

    //params
    bool m_use_cost_volume_filtering;



private:

    void init_params();

    cv::Mat undistort_rectify_image(const cv::Mat img, const Frame& frame_left, const Frame& frame_right);
    cv::Mat guided_filter(const cv::Mat& I, const cv::Mat& p, const float radius, const float eps);


    Halide::Func integral_img_2D(const Halide::Func& I, const int width, const int height);
    Halide::Func integral_img_3D(const Halide::Func& I, const int width, const int height);
    Halide::Func boxfilter(const Halide::Func& I, const int radius); //box filters a 2d image
    Halide::Func boxfilter_3D(const Halide::Func& I, const int radius); //box filters a 3d image. Returns another 3D image and the filter is ran over the first 2 dimensions(x,y)
    Halide::Func boxfilter_3D_integral_img(const Halide::Func& I, const int radius); //same as above but using an integral image for each slice of the 3D image I
    Halide::Func boxfilter_slice(const Halide::Func& I, const int radius, const int slice); //box filters a certain slice from a 3D vol. filters a channels indicated by slcie and returns a 2D image
    Halide::Func mul_elem(const Halide::Func& lhs, const Halide::Func& rhs); //element wise multiplication of 2D matrices, returns a 2D matrix
    Halide::Func mul_elem_3D(const Halide::Func& lhs, const Halide::Func& rhs); //element wise multiplication of 3D matrices, returns a 3D matrix
    Halide::Func mul_elem_slice_lhs(const Halide::Func& lhs, const Halide::Func& rhs, const int slice); //element wise multiplication of a 3D matrix and a 2D matrix. The 3d matrix gets sliced to a certain channel
    Halide::Func mul_elem_slice_rhs(const Halide::Func& lhs, const Halide::Func& rhs, const int slice);  //element wise multiplication of a 2D matrix and a 3D matrix. The 3d matrix gets sliced to a certain channel
    Halide::Func mul_elem_replicate_lhs(const Halide::Func& lhs, const Halide::Func& rhs); //elem wise multiplication of 2D matrix with 3D matrix. The 2D matrix is replicateted along the channels to coincide with the rhs 3D matrix
    Halide::Func mul_elem_replicate_rhs(const Halide::Func& lhs, const Halide::Func& rhs); //elem wise multiplication of 3D matrix with 2D matrix. The 2D matrix is replicateted along the channels to coincide with the lhs 3D matrix


    cv::Mat cost_volume_cpu(const cv::Mat& gray_left, const cv::Mat& gray_right);

    Mesh disparity_to_mesh(const cv::Mat& disparity);

};




#define TIME_SCOPE(name)\
    TIME_SCOPE_2(name,m_profiler);

#define TIME_START(name)\
    TIME_START_2(name,m_profiler);

#define TIME_END(name)\
    TIME_END_2(name,m_profiler);
;
