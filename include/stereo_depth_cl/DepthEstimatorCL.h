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


//TODO settings that should be refactored into a config file
const int MAX_RES_PER_POINT = 10;
const float setting_outlierTH = 12*12;					// higher -> less strict
const float setting_overallEnergyTHWeight = 1;
const float setting_outlierTHSumComponent = 50*50; 		// higher -> less strong gradient-based reweighting .
const float setting_huberTH = 9; // Huber Threshold



enum ImmaturePointStatus {
	IPS_GOOD=0,					// traced well and good
	IPS_OOB,					// OOB: end tracking & marginalize!
	IPS_OUTLIER,				// energy too high: if happens again: outlier!
	IPS_SKIPPED,				// traced well and good (but not actually traced).
	IPS_BADCONDITION,			// not traced because of bad condition.
    IPS_DELETED,                            // merged with other point or deleted
	IPS_UNINITIALIZED};			// not even traced once.

struct ImmaturePoint{
    int idx_host_frame; //idx in the array of frames of the frame which "hosts" this inmature points
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
    Eigen::Vector3d f; // heading range = Ki * (u,v,1)
	ImmaturePointStatus lastTraceStatus;

	float color[MAX_RES_PER_POINT]; 		// colors in host frame
	float weights[MAX_RES_PER_POINT]; 		// host-weights for respective residuals.
	// Vec2f colorD[MAX_RES_PER_POINT];
	// Vec2f colorGrad[MAX_RES_PER_POINT];
	// Vec2f rotatetPattern[MAX_RES_PER_POINT];
	bool skipZero [MAX_RES_PER_POINT];

	float ncc_sum_templ;
	float ncc_const_templ;

	//Stuff that may be to be removed
	Eigen::Matrix2d gradH; //it may need to go because opencl doesn't like eigen
	Eigen::Vector2d kp_GT;


	//debug stuff
	float gradient_hessian_det;
	int last_visible_frame;
    float gt_depth;

};

enum InterpolationType {
	NEAREST=0,
	LINEAR,
	CUBIC};			// not even traced once.



struct  AffineAutoDiffCostFunctor
{
	explicit AffineAutoDiffCostFunctor( const double & refColor, const double & newColor )
			:  m_refColor( refColor ), m_newColor( newColor ){ }

	template<typename T>
	bool operator() (const T* scaleA, const T* offsetB, T* residuals) const {
		residuals[0] = T(m_newColor) - (scaleA[0] * T(m_refColor) + offsetB[0]);
		return true;
	}
	static ceres::CostFunction * Create ( const double & refColor, const double & newColor )
	{
		return new ceres::AutoDiffCostFunction<AffineAutoDiffCostFunctor,1,1,1>( new AffineAutoDiffCostFunctor( refColor, newColor ) );
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

    //start with everything
    std::vector<Frame> loadDataFromICLNUIM ( const std::string & dataset_path, const int num_images_to_read );
    Mesh compute_depth2(Frame& frame);
	Eigen::Vector2d estimate_affine( std::vector<ImmaturePoint>& immature_points, const Frame&  cur_frame, const Eigen::Matrix3d& KRKi_cr, const Eigen::Vector3d& Kt_cr);
	float texture_interpolate ( const cv::Mat& img, const float x, const float y , const InterpolationType type=InterpolationType::NEAREST);
	std::vector<ImmaturePoint> create_immature_points ( const Frame& frame );
	void update_immature_points(std::vector<ImmaturePoint>& immature_points, const Frame& frame, const Eigen::Affine3d& tf_cur_host, const Eigen::Matrix3d& KRKi_cr, const Eigen::Vector3d& Kt_cr, const Eigen::Vector2d& affine_cr);
	void search_epiline_bca(ImmaturePoint& point, const Frame& frame, const Eigen::Matrix3d& hostToFrame_KRKi, const Eigen::Vector3d& Kt_cr, const Eigen::Vector2d& affine_cr);
	void search_epiline_ncc(ImmaturePoint& point, const Frame& frame, const Eigen::Matrix3d& hostToFrame_KRKi, const Eigen::Vector3d& Kt_cr);
    void update_idepth(ImmaturePoint& point, const Eigen::Affine3d& tf_host_cur, const float z, const double px_error_angle);
    double compute_tau(const Eigen::Affine3d & tf_host_cur, const Eigen::Vector3d& f, const double z, const double px_error_angle);
    void updateSeed(ImmaturePoint& point, const float x, const float tau2);
	Mesh create_mesh(const std::vector<ImmaturePoint>& immature_points, const std::vector<Frame>& frame);


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
