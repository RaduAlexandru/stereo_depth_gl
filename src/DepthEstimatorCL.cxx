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



namespace compute = boost::compute;


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


    //check some stats about the devie
    std::cout << "device: max_constant_args: " << m_device.getInfo<CL_DEVICE_MAX_CONSTANT_ARGS>()  << '\n';



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

    //depth estimation
    cl::Program program_depth(m_context,
         file_to_string("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/kernels_cl/depth_estimation.cl"));

     try{ // https://stackoverflow.com/questions/34662333/opencl-how-to-check-for-build-errors-using-the-c-wrapper
         program_depth.build({m_device},"-cl-std=CL2.0");
     }catch (cl::Error& e){
         if (e.err() == CL_BUILD_PROGRAM_FAILURE){
             // Get the build log
             std::string name     = m_device.getInfo<CL_DEVICE_NAME>();
             std::string buildlog = program_depth.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device);
             std::cerr << "Build log for " << name << ":" << std::endl << buildlog << std::endl;
        }else{
            throw e;
        }
    }
    // program_depth.build({m_device},"-cl-std=CL2.0");

    m_kernel_struct_test=cl::Kernel (program_depth, "struct_test");

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




    //upload to gpu
    cl::Buffer immature_points_cl_buf(m_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, immature_points.size() * sizeof(Point), immature_points.data());

    for (size_t i = 1; i < 2; i++) {
        const Eigen::Affine3d tf_cur_host = frames[i].tf_cam_world * frames[0].tf_cam_world.inverse();
        const Eigen::Affine3d tf_host_cur = tf_cur_host.inverse();
        const Eigen::Matrix3d KRKi_cr = frames[i].K * tf_cur_host.linear() * frames[0].K.inverse();
        const Eigen::Vector3d Kt_cr = frames[i].K * tf_cur_host.translation();
        const Eigen::Vector2d affine_cr = estimate_affine( immature_points, frames[i], KRKi_cr, Kt_cr);
        const double focal_length = abs(frames[i].K(0,0));
        double px_noise = 1.0;
        double px_error_angle = atan(px_noise/(2.0*focal_length))*2.0; // law of chord (sehnensatz)

        //upload the image
        std::cout << "img has type " << type2string(frames[i].gray.type()) << '\n';
        std::cout << "img has size " << frames[i].gray.rows << " " << frames[i].gray.cols << '\n';
        int size_bytes=frames[i].gray.step[0] * frames[i].gray.rows;
        cl::ImageFormat cl_img_format(CL_R,CL_FLOAT);
        std::cout << "makign img buffer fo size bytes " << size_bytes << '\n';
        cl::Buffer cl_img_buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_bytes, frames[i].gray.data);
        cl::Image2D cl_img(m_context, cl_img_format, cl_img_buffer, frames[i].gray.cols, frames[i].gray.rows );


        // tf_cur_host, tf_cur_host, KRKi_cr, Kt_cr, affine_cr, px_error_angle
        std::cout << "exeuting kernel" << '\n';
        TIME_START_CL("depth_update_kernel");
        m_kernel_struct_test.setArg(0, immature_points_cl_buf);
        m_kernel_struct_test.setArg(1, cl_img);
        // m_kernel_struct_test.setArg(1, tf_cur_host.data());
        // m_kernel_struct_test.setArg(1, tf_cur_host.data());
        m_queue.enqueueNDRangeKernel(m_kernel_struct_test, cl::NullRange, cl::NDRange(immature_points.size()), cl::NullRange);
        TIME_END_CL("depth_update_kernel");
    }



    //read the points back to cpu
    Point* points_ptr = (Point*)m_queue.enqueueMapBuffer(immature_points_cl_buf, CL_TRUE, CL_MAP_READ, 0, immature_points.size() * sizeof(Point));
    for (size_t i = 0; i < immature_points.size(); i++) {
        immature_points[i]=points_ptr[i];
    }

    // // copy data back to the host
    // std::vector<Point> host_vector(100000);
    // compute::command_queue boost_queue (m_queue());
    // compute::copy(point_cl.begin(), point_cl.end(), host_vector.begin(), boost_queue);
    // for (size_t i = 0; i < immature_points.size(); i++) {
    //     immature_points[i].last_visible_frame=host_vector[i].last_visible_frame;
    // }



    // // //use boost
    // compute::command_queue boost_queue (m_queue());
    // compute::context context = boost_queue.get_context();
    // boost::compute::vector<Point> point_cl(immature_points.size(),context);
    // // copy data to the device
    // compute::copy(immature_points.begin(), immature_points.end(), point_cl.begin(), boost_queue);
    // compute::kernel struct_test_boost = compute::kernel::create_with_source(
    //     file_to_string("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/kernels_cl/depth_estimation.cl"), "struct_test", context);
    // struct_test_boost.set_arg(0,point_cl.get_buffer());
    // boost_queue.enqueue_1d_range_kernel(struct_test_boost, 0, immature_points.size(), 0);
    // // // copy data back to the host
    // std::vector<Point> host_vector(100000);
    // compute::copy(point_cl.begin(), point_cl.end(), host_vector.begin(), boost_queue);
    // for (size_t i = 0; i < immature_points.size(); i++) {
    //     immature_points[i]=host_vector[i];
    // }

    TIME_END_CL("compute_depth");


    Mesh mesh=create_mesh(immature_points, frames);
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
                // point.gradH=gradient_hessian;

                //Seed::Seed
                Eigen::Vector3d f_eigen = (frame.K.inverse() * Eigen::Vector3d(point.u,point.v,1)).normalized();
                point.f = cl_float3{f_eigen(0),f_eigen(1),f_eigen(2)};

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
                // point.converged=false; //TODO don't use bools, it breaks opencl struct padding
                // point.is_outlier=true;


                //immature point constructor (the idepth min and max are already set so don't worry about those)
                // point.lastTraceStatus=PointStatus::UNINITIALIZED;

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

Eigen::Vector2d DepthEstimatorCL::estimate_affine(std::vector<Point>& immature_points, const Frame&  cur_frame, const Eigen::Matrix3d& KRKi_cr, const Eigen::Vector3d& Kt_cr){
    ceres::Problem problem;
    ceres::LossFunction * loss_function = new ceres::HuberLoss(1);
    double scaleA = 1;
    double offsetB = 0;

    TIME_START("creating ceres problem");
    for ( int i = 0; i < immature_points.size(); ++i )
    {
        Point& point = immature_points[i];
        if ( i % 100 != 0 )
            continue;

        //get colors at the current frame
        float color_cur_frame[cl_MAX_RES_PER_POINT];
        float color_host_frame[cl_MAX_RES_PER_POINT];


        if ( 1.0/point.gt_depth > 0 ) {

            const Eigen::Vector3d p = KRKi_cr * Eigen::Vector3d(point.u,point.v,1) + Kt_cr*  (1.0/point.gt_depth);
            Eigen::Vector2d kp_GT = p.hnormalized();


            if ( kp_GT(0) > 4 && kp_GT(0) < cur_frame.gray.cols-4 && kp_GT(1) > 3 && kp_GT(1) < cur_frame.gray.rows-4 ) {

                Pattern pattern_rot=m_pattern.get_rotated_pattern( KRKi_cr.topLeftCorner<2,2>() );

                for(int idx=0;idx<m_pattern.get_nr_points();++idx) {
                    Eigen::Vector2d offset=pattern_rot.get_offset(idx);

                    color_cur_frame[idx]=texture_interpolate(cur_frame.gray, kp_GT(0)+offset(0), kp_GT(1)+offset(1) , InterpolType::LINEAR);
                    color_host_frame[idx]=point.color[idx];

                }
            }
        }


        for ( int i = 0; i < m_pattern.get_nr_points(); ++i) {
            if ( !std::isfinite(color_host_frame[i]) || ! std::isfinite(color_cur_frame[i]) )
                continue;
            if ( color_host_frame[i] <= 0 || color_host_frame[i] >= 255 || color_cur_frame[i] <= 0 || color_cur_frame[i] >= 255  )
                continue;
            ceres::CostFunction * cost_function = AffineAutoDiffCostFunctorCL::Create( color_cur_frame[i], color_host_frame[i] );
            problem.AddResidualBlock( cost_function, loss_function, &scaleA, & offsetB );
        }
    }
    TIME_END("creating ceres problem");
    ceres::Solver::Options solver_options;
    //solver_options.linear_solver_type = ceres::DENSE_QR;//DENSE_SCHUR;//QR;
    solver_options.minimizer_progress_to_stdout = false;
    solver_options.max_num_iterations = 1000;
    solver_options.function_tolerance = 1e-6;
    solver_options.num_threads = 8;
    ceres::Solver::Summary summary;
    ceres::Solve( solver_options, & problem, & summary );
    //std::cout << summary.FullReport() << std::endl;
    std::cout << "scale= " << scaleA << " offset= "<< offsetB << std::endl;
    return Eigen::Vector2d ( scaleA, offsetB );
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

Mesh DepthEstimatorCL::create_mesh(const std::vector<Point>& immature_points, const std::vector<Frame>& frames){
    Mesh mesh;
    mesh.V.resize(immature_points.size(),3);
    mesh.V.setZero();

    for (size_t i = 0; i < immature_points.size(); i++) {
        int u=(int)immature_points[i].u;
        int v=(int)immature_points[i].v;
        // float depth=immature_points[i].gt_depth;
        float depth=1.0;

        // if(std::isfinite(immature_points[i].mu) && immature_points[i].mu>=0.1){
            //backproject the immature point
            Eigen::Vector3d point_screen;
            point_screen << u, v, 1.0;
            Eigen::Vector3d point_dir=frames[0].K.inverse()*point_screen; //TODO get the K and also use the cam to world corresponding to the immature point
            // point_dir.normalize(); //this is just the direction (in cam coordinates) of that pixel
            Eigen::Vector3d point_cam = point_dir*depth;
            point_cam(2)=-point_cam(2); //flip the depth because opengl has a camera which looks at the negative z axis (therefore, more depth means a more negative number)

            Eigen::Vector3d point_world=frames[0].tf_cam_world.inverse()*point_cam;

            mesh.V.row(i)=point_world;
        // }


    }

    //make also some colors based on depth
    mesh.C.resize(immature_points.size(),3);
    // double min_z, max_z;
    // min_z = mesh.V.col(2).minCoeff();
    // max_z = mesh.V.col(2).maxCoeff();
    // std::cout << "min max z is " << min_z << " " << max_z << '\n';
    // for (size_t i = 0; i < mesh.C.rows(); i++) {
    //     float gray_val = lerp(mesh.V(i,2), min_z, max_z, 0.0, 1.0 );
    //     mesh.C(i,0)=mesh.C(i,1)=mesh.C(i,2)=gray_val;
    // }

    //colors based on gradient_hessian det
    // float min=9999999999, max=-9999999999;
    // for (size_t i = 0; i < immature_points.size(); i++) {
    //     if(immature_points[i].gradient_hessian_det<min){
    //         min=immature_points[i].gradient_hessian_det;
    //     }
    //     if(immature_points[i].gradient_hessian_det>max){
    //         max=immature_points[i].gradient_hessian_det;
    //     }
    // }
    // for (size_t i = 0; i < mesh.C.rows(); i++) {
    //      float gray_val = lerp(immature_points[i].gradient_hessian_det, min, max, 0.0, 1.0 );
    //      mesh.C(i,0)=mesh.C(i,1)=mesh.C(i,2)=gray_val;
    //  }

    //colors based on last frame seen
   float min=9999999999, max=-9999999999;
   for (size_t i = 0; i < immature_points.size(); i++) {
       // std::cout << "last_visible_frame is " << immature_points[i].last_visible_frame << '\n';
       if(immature_points[i].last_visible_frame<min){
           min=immature_points[i].last_visible_frame;
       }
       if(immature_points[i].last_visible_frame>max){
           max=immature_points[i].last_visible_frame;
       }
   }
   std::cout << "min max last_visible_frame is " << min << " " << max << '\n';
   for (size_t i = 0; i < mesh.C.rows(); i++) {
        float gray_val = lerp(immature_points[i].last_visible_frame, min, max, 0.0, 1.0 );
        mesh.C(i,0)=mesh.C(i,1)=mesh.C(i,2)=gray_val;
    }






    return mesh;
}
