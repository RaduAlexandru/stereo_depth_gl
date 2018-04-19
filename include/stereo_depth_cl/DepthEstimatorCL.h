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
#include "stereo_depth_cl/Pattern.h"

//ceres
#include "ceres/ceres.h"
#include "ceres/cubic_interpolation.h"
#include "ceres/rotation.h"

//boost compute
#include <boost/compute.hpp>
#include <boost/compute/types/struct.hpp>

//TODO settings that should be refactored into a config file
const int cl_MAX_RES_PER_POINT = 10;
const float cl_setting_outlierTH = 12*12;					// higher -> less strict
const float cl_setting_overallEnergyTHWeight = 1;
const float cl_setting_outlierTHSumComponent = 50*50; 		// higher -> less strong gradient-based reweighting .
const float cl_setting_huberTH = 9; // Huber Threshold
const double cl_seed_convergence_sigma2_thresh=200;      //!< threshold on depth uncertainty for convergence.


enum class PointStatus {
    GOOD=0,					// traced well and good
    OOB,					// OOB: end tracking & marginalize!
    OUTLIER,				// energy too high: if happens again: outlier!
    SKIPPED,				// traced well and good (but not actually traced).
    BADCONDITION,			// not traced because of bad condition.
    DELETED,                            // merged with other point or deleted
    UNINITIALIZED};			// not even traced once.

struct Point{
    cl_int idx_host_frame; //idx in the array of frames of the frame which "hosts" this inmature points
    cl_float u,v; //position in host frame of the point

    // cl_float test_array[16];
    // cl_int test_bool_array[16]; //--break it
    // // cl_bool bool_1; //--also breaks it
    // // cl_bool bool_2;
    //  cl_int test_int_array[16];

    cl_float a;                     //!< a of Beta distribution: When high, probability of inlier is large.
    cl_float b;                     //!< b of Beta distribution: When high, probability of outlier is large.
    cl_float mu;                    //!< Mean of normal distribution.
    cl_float z_range;               //!< Max range of the possible depth.
    cl_float sigma2;                //!< Variance of normal distribution.
    cl_float idepth_min;
    cl_float idepth_max;
    cl_float energyTH;
    cl_float quality;
    cl_float3 f; // heading range = Ki * (u,v,1)
    // PointStatus lastTraceStatus;
    // cl_bool converged;
    // cl_bool is_outlier;

    cl_float color[cl_MAX_RES_PER_POINT]; 		// colors in host frame
    cl_float weights[cl_MAX_RES_PER_POINT]; 		// host-weights for respective residuals.
    // Vec2f colorD[MAX_RES_PER_POINT];
    // Vec2f colorGrad[MAX_RES_PER_POINT];
    // Vec2f rotatetPattern[MAX_RES_PER_POINT];
    // cl_bool skipZero [cl_MAX_RES_PER_POINT];

    cl_float ncc_sum_templ;
    cl_float ncc_const_templ;

    //Stuff that may be to be removed
    cl_float2 kp_GT;
    // cl_float kp_GT[2];
    //
    //
    //debug stuff
    cl_float gradient_hessian_det;
    cl_int last_visible_frame;
    cl_float gt_depth;

};

enum class InterpolType {
    NEAREST=0,
    LINEAR,
    CUBIC
};			// not even traced once.

// BOOST_COMPUTE_ADAPT_STRUCT(Point, Point, (idx_host_frame, u, v, a, b, mu, z_range, sigma2, idepth_min, idepth_max,
//                                           energyTH, quality, converged, is_outlier, color, weights, skipZero,
//                                           ncc_sum_templ, ncc_const_templ, gradient_hessian_det, last_visible_frame,
//                                           gt_depth));

// BOOST_COMPUTE_ADAPT_STRUCT(Point, Point, (idx_host_frame, u, v,
//                                           gradient_hessian_det, last_visible_frame, gt_depth));


struct  AffineAutoDiffCostFunctorCL
{
    explicit AffineAutoDiffCostFunctorCL( const double & refColor, const double & newColor )
            :  m_refColor( refColor ), m_newColor( newColor ){ }

    template<typename T>
    bool operator() (const T* scaleA, const T* offsetB, T* residuals) const {
        residuals[0] = T(m_newColor) - (scaleA[0] * T(m_refColor) + offsetB[0]);
        return true;
    }
    static ceres::CostFunction * Create ( const double & refColor, const double & newColor )
    {
        return new ceres::AutoDiffCostFunction<AffineAutoDiffCostFunctorCL,1,1,1>( new AffineAutoDiffCostFunctorCL( refColor, newColor ) );
    }

private:
    const double m_refColor;
    const double m_newColor;
};





//forward declarations
class Profiler;
namespace igl {  namespace opengl {namespace glfw{ class Viewer; }}}


class DepthEstimatorCL{
public:




    DepthEstimatorCL();
    ~DepthEstimatorCL(); //needed so that forward declarations work
    void init_opencl();

    //start with everything
    std::vector<Frame> loadDataFromICLNUIM ( const std::string & dataset_path, const int num_images_to_read );
    Mesh compute_depth(Frame& frame);
    std::vector<Point> create_immature_points (const Frame& frame);
    Eigen::Vector2d estimate_affine(std::vector<Point>& immature_points, const Frame&  cur_frame, const Eigen::Matrix3d& KRKi_cr, const Eigen::Vector3d& Kt_cr);
    float texture_interpolate ( const cv::Mat& img, const float x, const float y , const InterpolType type);
    Mesh create_mesh(const std::vector<Point>& immature_points, const std::vector<Frame>& frames);

	// Eigen::Vector2d estimate_affine( std::vector<ImmaturePoint>& immature_points, const Frame&  cur_frame, const Eigen::Matrix3d& KRKi_cr, const Eigen::Vector3d& Kt_cr);
	// float texture_interpolate ( const cv::Mat& img, const float x, const float y , const InterpolationType type=InterpolationType::NEAREST);
	// std::vector<ImmaturePoint> create_immature_points ( const Frame& frame );
	// void update_immature_points(std::vector<ImmaturePoint>& immature_points, const Frame& frame, const Eigen::Affine3d& tf_cur_host, const Eigen::Matrix3d& KRKi_cr, const Eigen::Vector3d& Kt_cr, const Eigen::Vector2d& affine_cr);
	// void search_epiline_bca(ImmaturePoint& point, const Frame& frame, const Eigen::Matrix3d& hostToFrame_KRKi, const Eigen::Vector3d& Kt_cr, const Eigen::Vector2d& affine_cr);
	// void search_epiline_ncc(ImmaturePoint& point, const Frame& frame, const Eigen::Matrix3d& hostToFrame_KRKi, const Eigen::Vector3d& Kt_cr);
    // void update_idepth(ImmaturePoint& point, const Eigen::Affine3d& tf_host_cur, const float z, const double px_error_angle);
    // double compute_tau(const Eigen::Affine3d & tf_host_cur, const Eigen::Vector3d& f, const double z, const double px_error_angle);
    // void updateSeed(ImmaturePoint& point, const float x, const float tau2);
	// Mesh create_mesh(const std::vector<ImmaturePoint>& immature_points, const std::vector<Frame>& frame);


    // Scene get_scene();
    bool is_modified(){return m_scene_is_modified;};


    //objects
    std::shared_ptr<Profiler> m_profiler;
    std::shared_ptr<igl::opengl::glfw::Viewer> m_view;
	Pattern m_pattern;

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

    //opencl things for depth estimation
    cl::Kernel m_kernel_struct_test;


    //databasse
    std::atomic<bool> m_scene_is_modified;

    //params
    bool m_cl_profiling_enabled;
    bool m_show_images;




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



// // This is for not packed structs but _NOT_ for cl_typeN structs
// #define BOOST_COMPUTE_ADAPT_STRUCT_NOT_PACKED(type, name, members) \
//     BOOST_COMPUTE_TYPE_NAME(type, name) \
//     namespace boost { namespace compute { \
//     template<> \
//     inline std::string type_definition<type>() \
//     { \
//         std::stringstream declaration; \
//         declaration << "typedef struct {\n" \
//                     BOOST_PP_SEQ_FOR_EACH( \
//                         BOOST_COMPUTE_DETAIL_ADAPT_STRUCT_INSERT_MEMBER, \
//                         type, \
//                         BOOST_COMPUTE_PP_TUPLE_TO_SEQ(members) \
//                     ) \
//                     << "} " << type_name<type>() << ";\n"; \
//         return declaration.str(); \
//     } \
//     namespace detail { \
//     template<> \
//     struct inject_type_impl<type> \
//     { \
//         void operator()(meta_kernel &kernel) \
//         { \
//             kernel.add_type_declaration<type>(type_definition<type>()); \
//         } \
//     }; \
//     inline meta_kernel& operator<<(meta_kernel &k, type s) \
//     { \
//         return k << "(" << #name << "){" \
//                BOOST_PP_SEQ_FOR_EACH_I( \
//                    BOOST_COMPUTE_DETAIL_ADAPT_STRUCT_STREAM_MEMBER, \
//                    s, \
//                    BOOST_COMPUTE_PP_TUPLE_TO_SEQ(members) \
//                ) \
//                << "}"; \
//     } \
//     }}}
//
// // Internal (This is for cl_typeN structs)
// #define BOOST_COMPUTE_ADAPT_CL_VECTOR_STRUCT_NAME(type, n, name) \
//     BOOST_COMPUTE_TYPE_NAME(type, name) \
//     namespace boost { namespace compute { \
//     namespace detail { \
//     inline meta_kernel& operator<<(meta_kernel &k, type x) \
//     { \
//         k << "(" << type_name<type>() << ")"; \
//         k << "("; \
//         for(size_t i = 0; i < n; i++){ \
//             k << k.lit(x.s[i]); \
//             \
//             if(i != n - 1){ \
//                 k << ","; \
//             } \
//         } \
//         k << ")"; \
//         return k; \
//     } \
//     }}}
//
// // This is for cl_typeN structs
// #define BOOST_COMPUTE_ADAPT_CL_VECTOR_STRUCT(type, n) \
//     BOOST_COMPUTE_ADAPT_CL_VECTOR_STRUCT_NAME(BOOST_PP_CAT(cl_, BOOST_PP_CAT(type, n)), n, BOOST_PP_CAT(type, n))
