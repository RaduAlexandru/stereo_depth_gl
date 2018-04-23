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
#define GLM_SWIZZLE // https://stackoverflow.com/questions/14657303/convert-glmvec4-to-glmvec3
#include <GL/glad.h>
#include <glm/glm.hpp>

#define MAX_RES_PER_POINT 16

//TODO settings that should be refactored into a config file
const int cl_MAX_RES_PER_POINT = 16;
const float cl_setting_outlierTH = 12*12;					// higher -> less strict
const float cl_setting_overallEnergyTHWeight = 1;
const float cl_setting_outlierTHSumComponent = 50*50; 		// higher -> less strong gradient-based reweighting .
const float cl_setting_huberTH = 9; // Huber Threshold
const double cl_seed_convergence_sigma2_thresh=200;      //!< threshold on depth uncertainty for convergence.
const float cl_settings_Eta=5;

struct Params {
    float outlierTH = 12*12;					// higher -> less strict
    float overallEnergyTHWeight = 1;
    float outlierTHSumComponent = 50*50; 		// higher -> less strong gradient-based reweighting .
    float huberTH = 9; // Huber Threshold
    double convergence_sigma2_thresh=200;      //!< threshold on depth uncertainty for convergence.
    float eta = 5;
};


enum PointStatus {
    STATUS_GOOD=0,					// traced well and good
    STATUS_OOB,					// OOB: end tracking & marginalize!
    STATUS_OUTLIER,				// energy too high: if happens again: outlier!
    STATUS_SKIPPED,				// traced well and good (but not actually traced).
    STATUS_BADCONDITION,			// not traced because of bad condition.
    STATUS_DELETED,                            // merged with other point or deleted
    STATUS_UNINITIALIZED};			// not even traced once.

//needs to be 16 bytes aligned as explained by john conor here https://www.opengl.org/discussion_boards/showthread.php/199303-How-to-get-Uniform-Block-Buffers-to-work-correctly
struct Point{
    int32_t idx_host_frame; //idx in the array of frames of the frame which "hosts" this inmature points
    float u,v; //position in host frame of the point
    float a;                     //!< a of Beta distribution: When high, probability of inlier is large.
    float b;                     //!< b of Beta distribution: When high, probability of outlier is large.
    float mu;                    //!< Mean of normal distribution.
    float z_range;               //!< Max range of the possible depth.
    float sigma2;                //!< Variance of normal distribution.
    float idepth_min;
    float idepth_max;
    float energyTH;
    float quality;
    //-----------------up until here we have 48 bytes so it's padded correctly to 16 bytes

    glm::vec4 f; // heading range = Ki * (u,v,1) //make it float 4 becuse float 3 gets padded to 4 either way
    PointStatus lastTraceStatus;
    int32_t converged;
    int32_t is_outlier;
    int32_t pad_1;
    //
    float color[MAX_RES_PER_POINT]; 		// colors in host frame
    float weights[MAX_RES_PER_POINT]; 		// host-weights for respective residuals.
    Eigen::Vector2f colorD[MAX_RES_PER_POINT];  //gradient in x and y at the pixel of the pattern normalized by the sqrt
    Eigen::Vector2f colorGrad[MAX_RES_PER_POINT]; //just the raw gradient in x and y at the pixel offset of the pattern


    float ncc_sum_templ;
    float ncc_const_templ;
    float pad_2;
    float pad_3;

    //Stuff that may be to be removed
    glm::mat2 gradH;
    glm::vec2 kp_GT;
    float pad_4;
    float pad_5;


    //debug stuff
    float gradient_hessian_det;
    float gt_depth;
    int32_t last_visible_frame;

    float debug; //serves as both debug and padding to 16 bytes
    // float padding_1; //to gt the struc to be aligned to 16 bytes

    float debug2[16];

};

enum class InterpolType {
    NEAREST=0,
    LINEAR,
    CUBIC
};			// not even traced once.




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
    float gaus_pdf(float mean, float sd, float x);
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
    GLuint m_points_gl_buf; //stores all the immature points
    GLuint m_ubo_params; //stores all parameters that may be needed inside the shader
    gl::Texture2D m_cur_frame;

    //gl shaders
    GLuint m_update_depth_prog_id;


    //databasse
    std::atomic<bool> m_scene_is_modified;

    //params
    bool m_gl_profiling_enabled;
    bool m_show_images;
    Params m_params; //parameters for depth estimation that may also be needed inside the gl shader




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
