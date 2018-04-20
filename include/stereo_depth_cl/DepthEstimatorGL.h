#pragma once
//C++
#include <iosfwd>
#include <memory>
#include <atomic>

//OpenCV
#include <opencv2/highgui/highgui.hpp>


//My stuff
#include "stereo_depth_cl/Mesh.h"
#include "stereo_depth_cl/Scene.h"
#include "stereo_depth_cl/DataLoader.h"
#include "stereo_depth_cl/Pattern.h"
#include "Texture2D.h"

//ceres
#include "ceres/ceres.h"
#include "ceres/cubic_interpolation.h"
#include "ceres/rotation.h"

//GL
#include <GL/glad.h>
#include <glm/glm.hpp>
#include <glm/glm.hpp>


//TODO settings that should be refactored into a config file
const int cl_MAX_RES_PER_POINT = 16;
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

//needs to be 16 bytes aligned as explained by john conor here https://www.opengl.org/discussion_boards/showthread.php/199303-How-to-get-Uniform-Block-Buffers-to-work-correctly
struct Point{
    int idx_host_frame; //idx in the array of frames of the frame which "hosts" this inmature points
    float u,v; //position in host frame of the point

    // cl_float test_array[16];
    // cl_int test_bool_array[16]; //--break it
    // // cl_bool bool_1; //--also breaks it
    // // cl_bool bool_2;
    //  cl_int test_int_array[16];

    float a;                     //!< a of Beta distribution: When high, probability of inlier is large.
    float b;                     //!< b of Beta distribution: When high, probability of outlier is large.
    float mu;                    //!< Mean of normal distribution.
    float z_range;               //!< Max range of the possible depth.
    float sigma2;                //!< Variance of normal distribution.
    float idepth_min;
    float idepth_max;
    float energyTH;
    float quality;
    glm::vec4 f; // heading range = Ki * (u,v,1) //make it float 4 becuse float 3 gets padded to 4 either way
    // float f[4]; // heading range = Ki * (u,v,1) //make it float 4 becuse float 3 gets padded to 4 either way
    // // PointStatus lastTraceStatus;
    // // cl_bool converged;
    // // cl_bool is_outlier;
    //
    float color[cl_MAX_RES_PER_POINT]; 		// colors in host frame
    float weights[cl_MAX_RES_PER_POINT]; 		// host-weights for respective residuals.
    // Vec2f colorD[MAX_RES_PER_POINT];
    // Vec2f colorGrad[MAX_RES_PER_POINT];
    // Vec2f rotatetPattern[MAX_RES_PER_POINT];
    // cl_bool skipZero [cl_MAX_RES_PER_POINT];
    //
    float ncc_sum_templ;
    float ncc_const_templ;

    //Stuff that may be to be removed
    glm::vec2 kp_GT;
    // // cl_float kp_GT[2];
    //
    //
    //debug stuff
    float gradient_hessian_det;
    float gt_depth;
    int last_visible_frame;

    float debug; //serves as both debug and padding to 16 bytes
    // float padding_1; //to gt the struc to be aligned to 16 bytes

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


struct  AffineAutoDiffCostFunctorGL
{
    explicit AffineAutoDiffCostFunctorGL( const double & refColor, const double & newColor )
            :  m_refColor( refColor ), m_newColor( newColor ){ }

    template<typename T>
    bool operator() (const T* scaleA, const T* offsetB, T* residuals) const {
        residuals[0] = T(m_newColor) - (scaleA[0] * T(m_refColor) + offsetB[0]);
        return true;
    }
    static ceres::CostFunction * Create ( const double & refColor, const double & newColor )
    {
        return new ceres::AutoDiffCostFunction<AffineAutoDiffCostFunctorGL,1,1,1>( new AffineAutoDiffCostFunctorGL( refColor, newColor ) );
    }

private:
    const double m_refColor;
    const double m_newColor;
};





//forward declarations
class Profiler;
namespace igl {  namespace opengl {namespace glfw{ class Viewer; }}}


class DepthEstimatorGL{
public:




    DepthEstimatorGL();
    ~DepthEstimatorGL(); //needed so that forward declarations work
    void init_opengl();

    //start with everything
    std::vector<Frame> loadDataFromICLNUIM ( const std::string & dataset_path, const int num_images_to_read );
    Mesh compute_depth();
    std::vector<Point> create_immature_points (const Frame& frame);
    Eigen::Vector2f estimate_affine(std::vector<Point>& immature_points, const Frame&  cur_frame, const Eigen::Matrix3f& KRKi_cr, const Eigen::Vector3f& Kt_cr);
    float texture_interpolate ( const cv::Mat& img, const float x, const float y , const InterpolType type);
    Mesh create_mesh(const std::vector<Point>& immature_points, const std::vector<Frame>& frames);



    // Scene get_scene();
    bool is_modified(){return m_scene_is_modified;};


    //objects
    std::shared_ptr<Profiler> m_profiler;
    std::shared_ptr<igl::opengl::glfw::Viewer> m_view;
	Pattern m_pattern;

    //gl stuff
    GLuint m_points_gl_buf;
    gl::Texture2D m_cur_frame;

    //gl shaders
    GLuint m_update_depth_prog_id;


    //databasse
    std::atomic<bool> m_scene_is_modified;

    //params
    bool m_gl_profiling_enabled;
    bool m_show_images;




private:
    void compile_shaders();

};






#define TIME_SCOPE(name)\
    TIME_SCOPE_2(name,m_profiler);

#define TIME_START(name)\
    TIME_START_2(name,m_profiler);

#define TIME_END(name)\
    TIME_END_2(name,m_profiler);

#define TIME_START_GL(name)\
    if (m_gl_profiling_enabled) glFinish();\
    TIME_START_2(name,m_profiler);

#define TIME_END_GL(name)\
    if (m_gl_profiling_enabled) glFinish();\
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
