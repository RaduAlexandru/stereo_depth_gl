#pragma once
//C++
#include <iosfwd>
#include <memory>
#include <atomic>

//OpenCV
#include <opencv2/highgui/highgui.hpp>


//My stuff
#include "stereo_depth_gl/Mesh.h"
#include "stereo_depth_gl/Pattern.h"
#include "stereo_depth_gl/Frame.h"
#include "Texture2D.h"



#define MAX_RES_PER_POINT 16 //IMPORTANT to change this value also in the shaders

//boost
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

struct Params {
    float maxPerPtError;
    float slackFactor;
    // float residualTH = 0.25;			//ngf		// higher -> less strict
    // float residualTH = 12*12;					// higher -> less strict
    float residualTH = 0.5*0.5;	 //BCA nromalzied values between [0,1]				// higher -> less strict
    float overallEnergyTHWeight = 1;
    float outlierTHSumComponent = 50*50; 		// higher -> less strong gradient-based reweighting .
    // float huberTH = 0.5; //ngf // Huber Threshold
    // float huberTH = 9; // Huber Threshold
    float huberTH = 0.05; //BCA with normalized values between [0,1] // Huber Threshold
    float convergence_sigma2_thresh=200;      //!< threshold on depth uncertainty for convergence.
    float eta = 5;

    float gradH_th=20000000000; //threshold on the gradient of the pixels. If gradient is above this value we will create immaure point
    int search_epi_method=0; //0=bca, 1=ngf
    //pad to 16 bytes if needed  (blocks of 4 floats)
    // float pad_1;
    // float pad_2;
    //until here it's paded correctly to 16 bytes-----

    int denoise_nr_iterations=200;
    float denoise_depth_range=5.0;
    float denoise_lambda=0.5;
    float denoise_L=sqrt(8.0f);
    float denoise_tau=0.02f;
    float denoise_theta=0.5;
    // float pad_1;
    // float pad_2;
};

enum SeedStatus {
    STATUS_GOOD=0,					// traced well and good
    STATUS_OOB,					// OOB: end tracking & marginalize!
    STATUS_OUTLIER,				// energy too high: if happens again: outlier!
    STATUS_SKIPPED,				// traced well and good (but not actually traced).
    STATUS_BADCONDITION,			// not traced because of bad condition.
    STATUS_DELETED,                            // merged with other point or deleted
    STATUS_UNINITIALIZED};			// not even traced once.

//needs to be 16 bytes aligned as explained by john conor here https://www.opengl.org/discussion_boards/showthread.php/199303-How-to-get-Uniform-Block-Buffers-to-work-correctly

struct MinimalDepthFilter{
    int32_t m_converged = 0;
    int32_t m_is_outlier = 0;
    int32_t m_initialized = 0;
    float m_f_scale = 1; //the scale of at the current level on the pyramid
    Eigen::Vector4f m_f; // heading range = Ki * (u,v,1) MAKE IT VECTOR4 so it's memory aligned for GPU usage
    int32_t m_lvl = 0; //pyramid lvl at which the depth filter was created
    float m_alpha;                 //!< a of Beta distribution: When high, probability of inlier is large.
    float m_beta;                  //!< b of Beta distribution: When high, probability of outlier is large.
    float m_mu;                    //!< Mean of normal distribution.
    float m_z_range;               //!< Max range of the possible depth.
    float m_sigma2;                //!< Variance of normal distribution.
    float m_mu_denoised; //for TVL1 denoising
    float m_mu_head; //for TVL1 denoising
    Eigen::Vector2f m_p; //for TVL1 denoising
    float m_g; //for TVL1 denoising
    float pad;
};
struct Seed{
    int32_t idx_keyframe; //idx in the array of frames of the frame which "hosts" this inmature points
    int32_t m_time_alive;
    int32_t m_nr_times_visible;
    float m_energyTH=0;
    float m_intensity[MAX_RES_PER_POINT]; //gray value for each point on the pattern
    Eigen::Vector2f m_normalized_grad[MAX_RES_PER_POINT];
    Eigen::Matrix2f m_gradH; //2x2 matrix for the hessian (gx2, gxgy, gxgy, gy2), used for calculating the alpha value
    Eigen::Vector2f m_uv; //position in x,y of the seed in th host_frame
    Eigen::Vector2f m_scaled_uv; //scaled uv position depending on the pyramid level of the image
    Eigen::Vector2f m_idepth_minmax;
    Eigen::Vector2f m_best_kp; //position at which the matching energy was minimal in another frame
    Eigen::Vector2f m_min_uv; //uv cooresponding to the minimum depth at which to trace
    Eigen::Vector2f m_max_uv; //uv cooresponding to the maximum depth at which to trace
    int32_t m_zero_grad [MAX_RES_PER_POINT]; //indicates fro each point on the pattern if it has zero grad and therefore can be skipped STORE it as int because bools are nasty for memory alignemnt on GPU as they are have different sizes in memory

    int m_active_pattern_points = 0; //nr of points of the pattern that don't have zero_grad
    int m_lvl = 0; //TODO why two time here andfilter?
    float m_igt_depth = 0;
    float m_last_error = 255;
    float m_last_idepth = 0;
    float m_last_tau2 = 0;
    float pad2[2]; //padded until 16 now

    MinimalDepthFilter depth_filter;

    //for denoising (indexes iinto the array of points of each of the 8 neighbours)
    int32_t left = -1;
    int32_t right = -1;
    int32_t above = -1;
    int32_t below = -1;
    int32_t left_upper = -1;
    int32_t right_upper = -1;
    int32_t left_lower = -1;
    int32_t right_lower = -1;

    float debug[16];

};

enum class InterpolType {
    NEAREST=0,
    LINEAR,
    CUBIC
};




//forward declarations
class Profiler;
namespace igl {  namespace opengl {namespace glfw{ class Viewer; }}}


class DepthEstimatorGL{
public:
    DepthEstimatorGL();
    ~DepthEstimatorGL(); //needed so that forward declarations work


    void compute_depth_and_update_mesh_stereo(const Frame& frame_left, const Frame& frame_right);

    //mostly for visualization
    void upload_gray_stereo_pair(const cv::Mat& image_left, const cv::Mat& image_right);
    void upload_rgb_stereo_pair(const cv::Mat& image_left, const cv::Mat& image_right);
    void upload_gray_and_grad_stereo_pair(const cv::Mat& image_left, const cv::Mat& image_right);


    //objects
    std::shared_ptr<Profiler> m_profiler;
	Pattern m_pattern;

    //gl stuff
    GLuint m_ubo_params; //stores all parameters that may be needed inside the shader
    gl::Texture2D m_frame_left; //stored the gray image and the grad_x and grad_y in the other channels, the 4th channel is unused
    gl::Texture2D m_frame_right; //the right camera, same as above
    gl::Texture2D m_frame_rgb_left; //mostly for visualization purposes we upload here the gray image
    gl::Texture2D m_frame_rgb_right;
    //for creating seeds
    gl::Texture2D m_hessian_pointwise_tex; // a 4 channels texture containing the hessian of each point, gx2, gxgy gy2, alpha is set to 255 for visualziation
    gl::Texture2D m_hessian_blurred_tex; //blurred texture of the previous m_hessian_pointwise_tex, using a box blur which is not normalized for speed reasons
    gl::Texture2D m_high_hessian_tex; //thresholded version of the m_hessian_tex which stores 1 for the high ones and 0 for the low ones
    gl::Texture2D m_debug_tex;
    GLuint m_atomic_nr_seeds_created;


    //gl shaders
    GLuint m_update_depth_prog_id;
    GLuint m_compute_hessian_pointwise_prog_id;
    GLuint m_compute_hessian_blurred_prog_id;
    GLuint m_compute_create_seeds_prog_id;
    GLuint m_compute_trace_seeds_prog_id;
    GLuint m_compute_init_seeds_prog_id; //initializey a vector of seeds for which we already know the size


    //databasse
    Mesh m_last_finished_mesh;
    bool m_started_new_keyframe;
    int m_nr_frames_traced;



    //params
    fs::path m_shaders_path;
    bool m_gl_profiling_enabled;
    bool m_debug_enabled;
    std::string m_pattern_file;
    int m_estimated_seeds_per_keyframe; //conservative estimate of nr of seeds created per frame
    int m_nr_buffered_keyframes; //nr of keyframes for which we store the seeds
    Params m_params; //parameters for depth estimation that may also be needed inside the gl shader


    //for debugging we run only icl nuim
    int m_start_frame;
    std::vector<Seed> m_seeds;
    int m_nr_seeds_created;
    GLuint m_seeds_gl_buf; //stores all the immature points
    bool m_seeds_gpu_dirty; //the data changed on te gpu buffer, we need to do a download
    bool m_seeds_cpu_dirty; //the data changed on te cpu vector, we need to do a upload
    gl::Texture2D m_cur_frame;
    gl::Texture2D m_ref_frame_tex;
    Mesh m_mesh;
    Frame m_ref_frame; //frame containing the seed points
    Frame m_last_ref_frame; //last frame which contained the seed points


    // void compute_depth_and_update_mesh(const Frame& frame_left);


private:
    void init_params();
    void init_opengl();
    void init_context(); //for the time when this class should be ripped outside and we don't have anymore the context created by libigl
    void compile_shaders();


    //cleaned up version
    void create_seeds_cpu(const Frame& frame);
    void create_seeds_gpu (const Frame& frame);
    void create_seeds_hybrid (const Frame& frame); //used gpu for hessian calculation and cpu for thresholding on trace
    void trace(const int nr_seeds_created, const Frame& ref_frame, const Frame& cur_frame);
    Mesh create_mesh(const std::vector<Seed>& seeds, Frame& ref_frame);
    void assign_neighbours_for_points( std::vector<Seed>& seeds, const int frame_width, const int frame_height); //assign neighbours based on where the immature points are in the reference frame.
    void denoise_cpu( std::vector<Seed>& seeds, const int iters,  const int frame_width, const int frame_height);
    void remove_grazing_seeds ( std::vector<Seed>& seeds );

    void print_seed(const Seed& s);
    // float texture_interpolate ( const cv::Mat& img, const float x, const float y , const InterpolType type);


    void sync_seeds_buf(); //if the gpu has more recent data, do a download, if the cpu has more recent data, do an upload
    std::vector<Seed> seeds_download(const GLuint& seeds_gl_buf, const int& nr_seeds_created); //downloa from the buffer to the cpu and store in a vec
    void seeds_upload(const std::vector<Seed>& seeds, const GLuint& seeds_gl_buf); //upload the seeds onto m_seeds_gl_buf

};






#define TIME_SCOPE(name)\
    TIME_SCOPE_2(name,m_profiler);

#define TIME_START(name)\
    TIME_START_2(name,m_profiler);

#define TIME_END(name)\
    TIME_END_2(name,m_profiler);

#define TIME_START_GL(name)\
    if (m_debug_enabled) std::cout<<"START: "<<name<<std::endl;\
    if (m_gl_profiling_enabled) glFinish();\
    TIME_START_2(name,m_profiler);

#define TIME_END_GL(name)\
    if (m_debug_enabled) std::cout<<"END: "<<name<<std::endl;\
    if (m_gl_profiling_enabled) glFinish();\
    TIME_END_2(name,m_profiler);
