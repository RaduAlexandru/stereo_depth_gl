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

//ceres
#include "ceres/ceres.h"
#include "ceres/cubic_interpolation.h"
#include "ceres/rotation.h"

//GL
#define GLM_SWIZZLE // https://stackoverflow.com/questions/14657303/convert-glmvec4-to-glmvec3
#include <GL/glad.h>
#include <glm/glm.hpp>


//TODO settings that should be refactored into a config file
const int MAX_RES_PER_POINT = 16;
const float setting_outlierTH = 12*12;					// higher -> less strict
const float setting_overallEnergyTHWeight = 1;
const float setting_outlierTHSumComponent = 50*50; 		// higher -> less strong gradient-based reweighting .
const float setting_huberTH = 9; // Huber Threshold
const double seed_convergence_sigma2_thresh=200;      //!< threshold on depth uncertainty for convergence.



enum PointStatus {
	IPS_GOOD=0,					// traced well and good
	IPS_OOB,					// OOB: end tracking & marginalize!
	IPS_OUTLIER,				// energy too high: if happens again: outlier!
	IPS_SKIPPED,				// traced well and good (but not actually traced).
	IPS_BADCONDITION,			// not traced because of bad condition.
    IPS_DELETED,                            // merged with other point or deleted
	IPS_UNINITIALIZED};			// not even traced once.

struct Point{
    int idx_host_frame; //idx in the array of frames of the frame which "hosts" this inmature points
    float u,v; //position in host frame of the point
    float a;                     //!< a of Beta distribution: When high, probability of inlier is large.
    float b;                     //!< b of Beta distribution: When high, probability of outlier is large.
    float mu;                    //!< Mean of normal distribution.
    float z_range;               //!< Max range of the possible depth. //TODO only read
    float sigma2;                //!< Variance of normal distribution.
    float idepth_min; //not necesary
    float idepth_max;
	float energyTH; //outside
    float quality;
    // Eigen::Vector3f f; // heading range = Ki * (u,v,1)
	Eigen::Vector4f f_eigen; //make it vec 4 to be more close to the opengl implementation
	glm::vec4 f;
	PointStatus lastTraceStatus;
	bool converged;
	bool is_outlier;

	float color[MAX_RES_PER_POINT]; 		// colors in host frame
	float weights[MAX_RES_PER_POINT]; 		// host-weights for respective residuals.
	// Vec2f colorD[MAX_RES_PER_POINT];
	// Vec2f colorGrad[MAX_RES_PER_POINT];
	// Vec2f rotatetPattern[MAX_RES_PER_POINT];
	bool skipZero [MAX_RES_PER_POINT]; //not really used

	float ncc_sum_templ;
	float ncc_const_templ;

	//Stuff that may be to be removed
	Eigen::Matrix2f gradH; //it may need to go because opencl doesn't like eigen
	// Eigen::Vector2f kp_GT;
	glm::vec2 kp_GT;


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


class DepthEstimatorGL2{
public:
    DepthEstimatorGL2();
    ~DepthEstimatorGL2(); //needed so that forward declarations work
	void init_opengl();
	void compile_shaders();

    //start with everything
    std::vector<Frame> loadDataFromICLNUIM ( const std::string & dataset_path, const int num_images_to_read );
	Mesh compute_depth_simplified();
	float gaus_pdf(float mean, float sd, float x);
	Eigen::Vector2f estimate_affine( std::vector<Point>& immature_points, const Frame&  cur_frame, const Eigen::Matrix3f& KRKi_cr, const Eigen::Vector3f& Kt_cr);
	float texture_interpolate ( const cv::Mat& img, const float x, const float y , const InterpolationType type=InterpolationType::NEAREST);
	std::vector<Point> create_immature_points ( const Frame& frame );
	Mesh create_mesh(const std::vector<Point>& immature_points, const std::vector<Frame>& frame);
	void print_point(const Point& point);


    // Scene get_scene();
    bool is_modified(){return m_scene_is_modified;};


    //objects
    std::shared_ptr<Profiler> m_profiler;
    std::shared_ptr<igl::opengl::glfw::Viewer> m_view;
	Pattern m_pattern;

    //databasse
    std::atomic<bool> m_scene_is_modified;

    //params
    bool m_gl_profiling_enabled;
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

#define TIME_START_GL(name)\
    if (m_gl_profiling_enabled) glFinish();\
    TIME_START_2(name,m_profiler);

#define TIME_END_GL(name)\
    if (m_gl_profiling_enabled) glFinish();\
    TIME_END_2(name,m_profiler);
