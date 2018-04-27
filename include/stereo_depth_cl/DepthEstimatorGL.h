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

struct Params {
    float outlierTH = 12*12;					// higher -> less strict
    float overallEnergyTHWeight = 1;
    float outlierTHSumComponent = 50*50; 		// higher -> less strong gradient-based reweighting .
    float huberTH = 9; // Huber Threshold
    float convergence_sigma2_thresh=200;      //!< threshold on depth uncertainty for convergence.
    float eta = 5;

    float gradH_th=800000000; //threshold on the gradient of the pixels. If gradient is above this value we will create immaure point
    int search_epi_method=0; //0=bca, 1=ngf
    //pad to 16 bytes if needed  (blocks of 4 floats)
    // float pad_1;
    // float pad_2;
    //until here it's paded correctly to 16 bytes-----

    int denoise_nr_iterations=0;
    float denoise_depth_range=5.0;
    float denoise_lambda=0.5;
    float denoise_L=sqrt(8.0f);
    float denoise_tau=0.02f;
    float denoise_theta=0.5;
    float pad_1;
    float pad_2;
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

    // glm::vec4 f; // heading range = Ki * (u,v,1) //make it float 4 becuse float 3 gets padded to 4 either way
    Eigen::Vector4f f;
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
    Eigen::Matrix2f gradH;


    //for denoising (indexes iinto the array of points of each of the 8 neighbours)
    int32_t left = -1;
    int32_t right = -1;
    int32_t above = -1;
    int32_t below = -1;
    int32_t left_upper = -1;
    int32_t right_upper = -1;
    int32_t left_lower = -1;
    int32_t right_lower = -1;

    //some other things for denoising
    float g;
    float mu_denoised;
    float mu_head;
    float pad_6;
    Eigen::Vector2f p;
    // glm::vec2 p;
    float pad_7;
    float pad_8;


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

    void init_data(); //Reads the images for the depth estimation and prepares the gl context
    void compute_depth_and_create_mesh(); //from all the immature points created triangulate depth for them, updates the mesh
    void compute_depth_and_create_mesh_cpu();
    void save_depth_image();
    Mesh get_mesh();


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
    GLuint m_denoise_depth_prog_id;


    //databasse
    std::atomic<bool> m_scene_is_modified;
    Mesh m_mesh;
    std::vector<Frame> m_frames;
    std::vector<Point> m_points;

    //params
    bool m_gl_profiling_enabled;
    bool m_show_images;
    bool m_use_rgbd_tum;
    int m_start_frame;
    float m_mean_starting_depth;
    Params m_params; //parameters for depth estimation that may also be needed inside the gl shader




private:
    void init_opengl();
    void compile_shaders();

    //start with everything
    std::vector<Frame> loadDataFromICLNUIM ( const std::string & dataset_path, const int num_images_to_read );
    std::vector<Frame> loadDataFromRGBD_TUM ( const std::string & dataset_path, const int num_images_to_read );
    void undistort_image(cv::Mat gray_img, const Eigen::Matrix3f K, const Eigen::VectorXf distort_coeffs);
    float gaus_pdf(float mean, float sd, float x);
    std::vector<Point> create_immature_points (const Frame& frame);
    Eigen::Vector2f estimate_affine(std::vector<Point>& immature_points, const Frame&  cur_frame, const Eigen::Matrix3f& KRKi_cr, const Eigen::Vector3f& Kt_cr);
    float texture_interpolate ( const cv::Mat& img, const float x, const float y , const InterpolType type);
    void assign_neighbours_for_points( std::vector<Point>& immature_points, const int frame_width, const int frame_height); //assign neighbours based on where the immature points are in the reference frame.
    void denoise_cpu( std::vector<Point>& immature_points, const int frame_width, const int frame_height);
    Mesh create_mesh(const std::vector<Point>& immature_points, const std::vector<Frame>& frames);

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
