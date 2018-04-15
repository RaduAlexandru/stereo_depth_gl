#include "stereo_depth_cl/DepthEstimatorCL.h"

//c++
#include <cmath>
#include <stdlib.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>

//My stuff
#include "stereo_depth_cl/Profiler.h"
#include "stereo_depth_cl/MiscUtils.h"
#include "UtilsCL.h"
#include "cv_interpolation.h"

//Libigl
#include <igl/opengl/glfw/Viewer.h>

//cv
#include <cv_bridge/cv_bridge.h>

//loguru
#include <loguru.hpp>



DepthEstimatorCL::DepthEstimatorCL():
        m_scene_is_modified(false),
        m_cl_profiling_enabled(true),
        m_show_images(false)
        {

    init_opencl();
    std::string pattern_filepath="/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/data/pattern_1.png";
    m_pattern.init_pattern(pattern_filepath);

    //sanity check the pattern
    std::cout << "pattern has nr of points " << m_pattern.get_nr_points() << '\n';
    for (size_t i = 0; i < m_pattern.get_nr_points(); i++) {
        std::cout << "offset for i " << i << " is " << m_pattern.get_offset(i).transpose() << '\n';
    }
}

//needed so that forward declarations work
DepthEstimatorCL::~DepthEstimatorCL(){
}

void DepthEstimatorCL::init_opencl(){
    VLOG(1) << "init opencl";
    std::cout << "init opencl" << '\n';


	// Get list of OpenCL platforms.
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.empty()) {
	    std::cerr << "No platforms found." << std::endl;
	    return;
	}

    //get the platform that has opencl (in order to avoid getting the one with CUDA)
    int idx_platform_chosen=-1;
    for (size_t i = 0; i < platforms.size(); i++) {
        std::string name=platforms[i].getInfo<CL_PLATFORM_NAME>();
        // std::cout << "name of device is " << name << '\n';
        if(name.find("OpenCL")!=std::string::npos){
            idx_platform_chosen=i;
        }
    }
    if (idx_platform_chosen==-1) {
	    std::cerr << "OpenCL platforms not found." << std::endl;
	    return;
	}



	// Get the first Intel GPU device
    std::vector<cl::Device> devices_available;
    platforms[idx_platform_chosen].getDevices(CL_DEVICE_TYPE_GPU, &devices_available);
    std::cout << "found nr of devices " << devices_available.size() << '\n';
    bool found_intel_device=false;
    for (size_t i = 0; i < devices_available.size(); i++) {
        std::string name=devices_available[i].getInfo<CL_DEVICE_NAME>();
        // std::cout << "name of device is " << name << '\n';
        if(name.find("HD Graphics")!=std::string::npos){
            m_device=devices_available[i];
            found_intel_device=true;
        }
    }
    if(!found_intel_device){
        LOG_F(FATAL, "No Intel device found");
    }

    std::cout << m_device.getInfo<CL_DEVICE_NAME>() << std::endl;

    m_context = cl::Context(m_device);

	// // Create command queue.
	m_queue = cl::CommandQueue(m_context,m_device);


    //check svm capabilities
    cl_device_svm_capabilities caps;
    std::cout << "checking SVM capabilities of device id " << m_device() << '\n';
    cl_int err = clGetDeviceInfo( m_device(), CL_DEVICE_SVM_CAPABILITIES, sizeof(cl_device_svm_capabilities), &caps, 0);
    if(err == CL_INVALID_VALUE){
        std::cout << "No SVM support" << '\n';
    }
    if(err == CL_SUCCESS && (caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)){
        std::cout << "Coarse-grained buffer" << '\n';
    }
    if(err == CL_SUCCESS && (caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)){
        std::cout << "Fine-grained buffer" << '\n';
    }
    if(err == CL_SUCCESS && (caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) && (caps & CL_DEVICE_SVM_ATOMICS)){
        std::cout << "Fine-grained buffer with atomics" << '\n';
    }
    if(err == CL_SUCCESS && (caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)){
        std::cout << "Fine-grained system" << '\n';
    }
    if(err == CL_SUCCESS && (caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) && (caps & CL_DEVICE_SVM_ATOMICS)){
        std::cout << "Fine-grained system with atomics" << '\n';
    }


    compile_kernels();
}

void DepthEstimatorCL::compile_kernels(){
    cl::Program program(m_context,
         file_to_string("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/kernels_cl/img_proc_test.cl"));
    program.build({m_device},"-cl-std=CL2.0");
	m_kernel_simple_copy=cl::Kernel (program, "simple_copy");
    m_kernel_blur=cl::Kernel (program, "gaussian_blur");
    m_kernel_sobel=cl::Kernel (program, "sobel");
    m_kernel_blurx=cl::Kernel (program, "blurx");
    m_kernel_blury=cl::Kernel (program, "blury");

    m_kernel_blurx_fast=cl::Kernel (program, "blurx_fast");
    m_kernel_blury_fast=cl::Kernel (program, "blury_fast");
}


Mesh DepthEstimatorCL::compute_depth(Frame& frame){
    //read images from ICL_NUIM
    //calculate pyramid and gradients
    //grab the first frame and calculate inmature points for it

    //for each new frame afterwards we grab pass it to open and update the inmature points depth

    //----------------------------------------------------------------------------------------------------
    std::string dataset_path="/media/alex/Data/Master/Thesis/data/ICL_NUIM/living_room_traj2_frei_png";
    int num_images_to_read=60;
    bool use_modified=false;
    std::vector<Frame> frames=loadDataFromICLNUIM(dataset_path, num_images_to_read);
    std::cout << "frames size is " << frames.size() << "\n";


    //TODO undistort images

    TIME_START_CL("compute_depth");

    //calculate the gradients of it (cpu)
    //calculate immature points (cpu)
    //upload immature points vector to gpu
    //upload each new frame to gpu

    std::vector<Point> immature_points;
    immature_points=create_immature_points(frames[0]);
    std::cout << "immature_points size is " << immature_points.size() << '\n';



    TIME_END_CL("compute_depth");


    Mesh mesh;
    return mesh;

}

std::vector<Frame> DepthEstimatorCL::loadDataFromICLNUIM ( const std::string & dataset_path, const int num_images_to_read){
   std::vector< Frame > frames;
   std::string filename_img = dataset_path + "/associations.txt";
   std::string filename_gt = dataset_path + "/livingRoom2.gt.freiburg";

   //K is from here https://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html
   Eigen::Matrix3d K;
   K.setZero();
   K(0,0)=481.2; //fx
   K(1,1)=-480; //fy
   K(0,2)=319.5; // cx
   K(1,2)=239.5; //cy
   K(2,2)=1.0;

   std::ifstream imageFile ( filename_img, std::ifstream::in );
   std::ifstream grtruFile ( filename_gt, std::ifstream::in );

   int imagesRead = 0;
   for ( imagesRead = 0; imageFile.good() && grtruFile.good() && imagesRead <= num_images_to_read ; ++imagesRead ){
      std::string depthFileName, colorFileName;
      int idc, idd, idg;
      double tsc, tx, ty, tz, qx, qy, qz, qw;
      imageFile >> idd >> depthFileName >> idc >> colorFileName;

      if ( idd == 0 )
          continue;
      grtruFile >> idg >> tx >> ty >> tz >> qx >> qy >> qz >> qw;

      if ( idc != idd || idc != idg ){
          std::cerr << "Error during reading... not correct anymore!" << std::endl;
          break;
      }

      if ( ! depthFileName.empty() ){
         Eigen::Affine3d pose_wc = Eigen::Affine3d::Identity();
         pose_wc.translation() << tx,ty,tz;
         pose_wc.linear() = Eigen::Quaterniond(qw,qx,qy,qz).toRotationMatrix();
         Eigen::Affine3d pose_cw = pose_wc.inverse();

         cv::Mat rgb_cv=cv::imread(dataset_path + "/" + colorFileName);
         cv::Mat depth_cv=cv::imread(dataset_path + "/" + depthFileName, CV_LOAD_IMAGE_UNCHANGED);
         depth_cv.convertTo ( depth_cv, CV_32F, 1./5000. ); //ICLNUIM stores theis weird units so we transform to meters


         Frame cur_frame;
         cur_frame.rgb=rgb_cv;
         cv::cvtColor ( cur_frame.rgb, cur_frame.gray, CV_BGR2GRAY );
         cur_frame.depth=depth_cv;
         cur_frame.tf_cam_world=pose_cw;
         cur_frame.gray.convertTo ( cur_frame.gray, CV_32F );
         cur_frame.K=K;
         cur_frame.frame_id=imagesRead;

         frames.push_back(cur_frame);
         VLOG(1) << "read img " << imagesRead << " " << colorFileName;
      }
   }
   std::cout << "read " << imagesRead << " images. (" << frames.size() <<", " << ")" << std::endl;
   return frames;
}

float DepthEstimatorCL::texture_interpolate ( const cv::Mat& img, const float x, const float y , const InterpolType type){
    //sample only from a cv mat that is of type float with 1 channel
    if(type2string(img.type())!="32FC1"){
        LOG(FATAL) << "trying to use texture inerpolate on an image that is not float valued with 1 channel. Image is of type " <<
        type2string(img.type());
    }

    //Dumb nearest interpolation
   // int clamped_y=clamp((int)y,0,img.rows);
   // int clamped_x=clamp((int)x,0,img.cols);
   // float val=img.at<float>(clamped_y,clamped_x);

    //from oepncv https://github.com/opencv/opencv/blob/master/modules/cudawarping/test/interpolation.hpp
    if(type==InterpolType::NEAREST){
        return NearestInterpolator<float>::getValue(img,y,x,0,cv::BORDER_REPLICATE);
    }else if(type==InterpolType::LINEAR){
        return LinearInterpolator<float>::getValue(img,y,x,0,cv::BORDER_REPLICATE);
    }else if(type==InterpolType::CUBIC){
        return CubicInterpolator<float>::getValue(img,y,x,0,cv::BORDER_REPLICATE);
    }

}

std::vector<Point> DepthEstimatorCL::create_immature_points (const Frame& frame){


    //create all of the pixels as inmature points
    // std::vector<ImmaturePoint> immature_points;
    // for (size_t i = 0; i < frame.gray.rows; i++) {
    //     for (size_t j = 0; j < frame.gray.cols; j++) {
    //         ImmaturePoint point;
    //         point.u=j;
    //         point.v=i;
    //         point.depth=frame.depth.at<float>(i,j);
    //         immature_points.push_back(point);
    //     }
    // }

    //make the sobel in x and y because we need to to calculate the hessian in order to select the immature point
    TIME_START_CL("sobel_host_frame");
    cv::Mat grad_x, grad_y;
    cv::Scharr( frame.gray, grad_x, CV_32F, 1, 0);
    cv::Scharr( frame.gray, grad_y, CV_32F, 0, 1);
    TIME_END_CL("sobel_host_frame");

    TIME_START_CL("hessian_host_frame");
    std::vector<Point> immature_points;
    immature_points.reserve(200000);
    for (size_t i = 10; i < frame.gray.rows-10; i++) {  //--------Do not look around the borders to avoid pattern accesing outside img
        for (size_t j = 10; j < frame.gray.cols-10; j++) {

            //check if this point has enough determinant in the hessian
            Eigen::Matrix2d gradient_hessian;
            gradient_hessian.setZero();
            for (size_t p = 0; p < m_pattern.get_nr_points(); p++) {
                int dx = m_pattern.get_offset_x(p);
                int dy = m_pattern.get_offset_y(p);

                float gradient_x=grad_x.at<float>(i+dy,j+dx); //TODO should be interpolated
                float gradient_y=grad_y.at<float>(i+dy,j+dx);

                Eigen::Vector2d grad;
                grad << gradient_x, gradient_y;
                // std::cout << "gradients are " << gradient_x << " " << gradient_y << '\n';

                gradient_hessian+= grad*grad.transpose();
            }

            //determinant is high enough, add the point
            float hessian_det=gradient_hessian.determinant();
            if(hessian_det > 100000000){
                Point point;
                point.u=j;
                point.v=i;
                point.gradH=gradient_hessian;

                //Seed::Seed
                point.f = (frame.K.inverse() * Eigen::Vector3d(point.u,point.v,1)).normalized();

                //start at an initial value for depth at around 4 meters (depth_filter->Seed::reinit)
                float mean_starting_depth=4.0;
                float min_starting_depth=0.1;
                point.mu = (1.0/mean_starting_depth);
                point.z_range = (1.0/min_starting_depth);
                point.sigma2 = (point.z_range*point.z_range/36);

                float z_inv_min = point.mu + sqrt(point.sigma2);
                float z_inv_max = std::max<float>(point.mu- sqrt(point.sigma2), 0.00000001f);
                point.idepth_min = z_inv_min;
                point.idepth_max = z_inv_max;

                point.a=10.0;
                point.b=10.0;

                //seed constructor deep_filter.h
                point.converged=false;
                point.is_outlier=true;


                //immature point constructor (the idepth min and max are already set so don't worry about those)
                point.lastTraceStatus=PointStatus::UNINITIALIZED;

                //get data for the color of that point (depth_point->ImmaturePoint::ImmaturePoint)---------------------
                for(int p_idx=0;p_idx<m_pattern.get_nr_points();p_idx++){
                    Eigen::Vector2d offset = m_pattern.get_offset(p_idx);

                    point.color[p_idx]=texture_interpolate(frame.gray, point.u+offset(0), point.v+offset(1), InterpolType::NEAREST);

                    float grad_x_val=texture_interpolate(grad_x, point.u+offset(0), point.v+offset(1), InterpolType::NEAREST);
                    float grad_y_val=texture_interpolate(grad_y, point.u+offset(0), point.v+offset(1), InterpolType::NEAREST);
                    float squared_norm=grad_x_val*grad_x_val + grad_y_val*grad_y_val;
                    point.weights[p_idx] = sqrtf(cl_setting_outlierTHSumComponent / (cl_setting_outlierTHSumComponent + squared_norm));
                }
                point.ncc_sum_templ    = 0.0f;
                float ncc_sum_templ_sq = 0.0f;
                for(int p_idx=0;p_idx<m_pattern.get_nr_points();p_idx++){
                    const float templ = point.color[p_idx];
                    point.ncc_sum_templ += templ;
                    ncc_sum_templ_sq += templ*templ;
                }
                point.ncc_const_templ = m_pattern.get_nr_points() * ncc_sum_templ_sq - (double) point.ncc_sum_templ*point.ncc_sum_templ;

                point.energyTH = m_pattern.get_nr_points()*cl_setting_outlierTH;
                point.energyTH *= cl_setting_overallEnergyTHWeight*cl_setting_overallEnergyTHWeight;

                point.quality=10000;
                //-------------------------------------

                //debug stuff
                point.gradient_hessian_det=hessian_det;
                point.last_visible_frame=0;
                point.gt_depth=frame.depth.at<float>(i,j); //add also the gt depth just for visualization purposes

                immature_points.push_back(point);
            }

        }
    }
    TIME_END_CL("hessian_host_frame");




    return immature_points;
}
