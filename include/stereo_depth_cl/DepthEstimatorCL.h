#pragma once
//C++
#include <iosfwd>
#include <memory>
#include <atomic>

//OpenCV
#include <opencv2/highgui/highgui.hpp>

//opencl
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include "CL/cl2.hpp"
#include <CL/cl.h>
// #include "CL/cl.hpp"
#include "Image2DSafe.h"

//My stuff
#include "stereo_depth_cl/Mesh.h"
#include "stereo_depth_cl/Scene.h"
#include "stereo_depth_cl/DataLoader.h"


//forward declarations
class Profiler;
namespace igl {  namespace opengl {namespace glfw{ class Viewer; }}}


class DepthEstimatorCL{
public:
    DepthEstimatorCL();
    ~DepthEstimatorCL(); //needed so that forward declarations work
    void init_opencl();

    void run_speed_test();
    void run_speed_test_img(Frame& frame);
    void run_speed_test_img2(Frame& frame);
    void run_speed_test_img_3_blur(Frame& frame);
    void run_speed_test_img_4_sobel(Frame& frame);
    void run_speed_test_img_4_sobel_gray(Frame& frame);
    void run_speed_test_img_4_blur_gray(Frame& frame);
    void run_speed_test_img_4_blur_gray_safe(Frame& frame);

    void create_blur_mask(std::vector<float>& mask, const int sigma); //create a 1d mask for gaussian blurring (doesn't matter if it's used in x or y)
    void create_half_blur_mask(std::vector<float>& mask, const int sigma); //creates only half of gaussian because it's symetric
    void optimize_blur_for_gpu_sampling(std::vector<float>&gaus_mask, std::vector<float>& gaus_offsets);
    void gaussian_blur(cl::Image2DSafe& dest_img, const cl::Image2DSafe& src_img, const int sigma);

    void compute_depth(Frame& frame);


    // Scene get_scene();
    bool is_modified(){return m_scene_is_modified;};


    //objects
    std::shared_ptr<Profiler> m_profiler;
    std::shared_ptr<igl::opengl::glfw::Viewer> m_view;

    //opencl global stuff
    cl::Context m_context;
    cl::Device m_device;
    cl::CommandQueue m_queue;

    //opencl things for processing the images
    cl::Kernel m_kernel_simple_copy;
    cl::Kernel m_kernel_blur;
    cl::Kernel m_kernel_sobel;
    cl::Kernel m_kernel_blurx;
    cl::Kernel m_kernel_blury;
    cl::Kernel m_kernel_blurx_fast;
    cl::Kernel m_kernel_blury_fast;


    //databasse
    std::atomic<bool> m_scene_is_modified;

    //params
    bool m_cl_profiling_enabled;
    bool m_show_images;

    //stuff for speed test


private:
    void compile_kernels();

};




#define TIME_SCOPE(name)\
    TIME_SCOPE_2(name,m_profiler);

#define TIME_START(name)\
    TIME_START_2(name,m_profiler);

#define TIME_END(name)\
    TIME_END_2(name,m_profiler);

#define TIME_START_CL(name)\
    if (m_cl_profiling_enabled){ m_queue.finish();}\
    TIME_START_2(name,m_profiler);

#define TIME_END_CL(name)\
    if (m_cl_profiling_enabled){ m_queue.finish();}\
    TIME_END_2(name,m_profiler);
