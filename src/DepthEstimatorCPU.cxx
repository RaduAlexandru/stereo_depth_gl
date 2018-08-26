#include "stereo_depth_gl/DepthEstimatorCPU.h"

//c++
#include <cmath>
#include <stdlib.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>

//My stuff
#include "stereo_depth_gl/Profiler.h"
#include "stereo_depth_gl/MiscUtils.h"
#include "cv_interpolation.h"

//Libigl
#include <igl/opengl/glfw/Viewer.h>

//cv
#include <cv_bridge/cv_bridge.h>

//loguru
#include <loguru.hpp>

//omp
#include <omp.h>

//boost
#include <boost/math/distributions/normal.hpp>





// Compute c = a + b.
static const char source[] =
    "kernel void add(\n"
    "       global const float *a,\n"
    "       global const float *b,\n"
    "       global float *c\n"
    "       )\n"
    "{\n"
    "    size_t i = get_global_id(0);\n"
        "c[i] = pow(a[i], b[i]);\n"
    "}\n";



DepthEstimatorCPU::DepthEstimatorCPU():
        m_scene_is_modified(false),
        m_cl_profiling_enabled(false),
        m_show_images(false)
        {

    // init_opencl();
    std::string pattern_filepath="/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_gl/data/pattern_1.png";
    m_pattern.init_pattern(pattern_filepath);

    //sanity check the pattern
    std::cout << "pattern has nr of points " << m_pattern.get_nr_points() << '\n';
    for (size_t i = 0; i < m_pattern.get_nr_points(); i++) {
        std::cout << "offset for i " << i << " is " << m_pattern.get_offset(i).transpose() << '\n';
    }
}

//needed so that forward declarations work
DepthEstimatorCPU::~DepthEstimatorCPU(){
}



Mesh DepthEstimatorCPU::compute_depth2(Frame& frame){
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

    TIME_START("compute_depth");
    //create inmature points for the first frame
    std::vector<ImmaturePoint> immature_points;
    immature_points=create_immature_points(frames[0]);



    for (size_t i = 1; i < frames.size(); i++) {
        //compute the matrices between the two frames
        Eigen::Affine3f tf_cur_host = frames[i].tf_cam_world * frames[0].tf_cam_world.inverse();
        Eigen::Matrix3f KRKi_cr = frames[i].K * tf_cur_host.linear() * frames[0].K.inverse();
        Eigen::Vector3f Kt_cr = frames[i].K * tf_cur_host.translation();
        Eigen::Vector2f affine_cr = estimate_affine( immature_points, frames[i], KRKi_cr, Kt_cr);

        update_immature_points(immature_points, frames[i], tf_cur_host, KRKi_cr, Kt_cr, affine_cr );
    }

    TIME_END("compute_depth");


    Mesh mesh=create_mesh(immature_points,frames); //creates a mesh from the position of the points and their depth
    return mesh;

}

//https://www.boost.org/doc/libs/1_47_0/libs/math/doc/sf_and_dist/html/math_toolkit/dist/dist_ref/dists/normal_dist.html
float DepthEstimatorCPU::gaus_pdf(float mean, float sd, float x){
    return exp(- (x-mean)*(x-mean)/(2*sd)*(2*sd)  )  / (sd*sqrt(2*M_PI));
}

Mesh DepthEstimatorCPU::compute_depth_simplified(){
    Mesh mesh;
    //read images from ICL_NUIM
    //----------------------------------------------------------------------------------------------------
    std::string dataset_path="/media/alex/Data/Master/Thesis/data/ICL_NUIM/living_room_traj2_frei_png";
    int num_images_to_read=60;
    bool use_modified=false;
    std::vector<Frame> frames=loadDataFromICLNUIM(dataset_path, num_images_to_read);
    std::cout << "frames size is " << frames.size() << "\n";


    //TODO undistort images

    TIME_START("compute_depth");
    //create inmature points for the first frame
    std::vector<ImmaturePoint> immature_points;
    immature_points=create_immature_points(frames[0]);



    for (size_t i = 1; i < frames.size(); i++) {
        //compute the matrices between the two frames
        Eigen::Affine3f tf_cur_host = frames[i].tf_cam_world * frames[0].tf_cam_world.inverse();
        const Eigen::Affine3f tf_host_cur = tf_cur_host.inverse();
        Eigen::Matrix3f KRKi_cr = frames[i].K * tf_cur_host.linear() * frames[0].K.inverse();
        Eigen::Vector3f Kt_cr = frames[i].K * tf_cur_host.translation();
        Eigen::Vector2f affine_cr = estimate_affine( immature_points, frames[i], KRKi_cr, Kt_cr);
        const double focal_length = fabs(frames[i].K(0,0));
        double px_noise = 1.0;
        double px_error_angle = atan(px_noise/(2.0*focal_length))*2.0; // law of chord (sehnensatz)
        Pattern pattern_rot=m_pattern.get_rotated_pattern( KRKi_cr.topLeftCorner<2,2>() );


        //make a ll matrices either 4x4 or 3x3 as required by gl
        Eigen::Matrix4f tf_cur_host_4x4= tf_cur_host.matrix();
        Eigen::Matrix4f tf_host_cur_4x4= tf_host_cur.matrix();



        // update_immature_points(immature_points, frames[i], tf_cur_host, KRKi_cr, Kt_cr, affine_cr );

        //done in paralel for all points in the case of opengl
        for (auto &point : immature_points){

            // // check if point is visible in the current image
            const Eigen::Vector3f p_backproj_xyz= point.f.head<3>() * 1.0/point.mu;
            const Eigen::Vector4f p_backproj_xyzw=Eigen::Vector4f(p_backproj_xyz(0),p_backproj_xyz(1),p_backproj_xyz(2),1.0);
            const Eigen::Vector4f xyz_f_xyzw = tf_cur_host_4x4*  p_backproj_xyzw ;
            const Eigen::Vector3f xyz_f=xyz_f_xyzw.head<3>()/xyz_f_xyzw.w();
            if(xyz_f.z() < 0.0)  {
                continue; // TODO in gl this is a return
            }


            // const Eigen::Vector3f xyz_f( tf_cur_host_4x4*(1.0/point.mu * point.f.head<3>()) );
            // if(xyz_f.z() < 0.0)  {
            //     continue;
            // }
            const Eigen::Vector2f kp_c = (frames[i].K * xyz_f).hnormalized();
            if ( kp_c(0) < 0 || kp_c(0) >= frames[i].gray.cols || kp_c(1) < 0 || kp_c(1) >= frames[i].gray.rows ) {
                continue;
            }


            //point is visible
            point.last_visible_frame=frames[i].frame_idx;

            //update inverse depth coordinates for min and max
            point.idepth_min = point.mu + sqrt(point.sigma2);
            point.idepth_max = std::max<float>(point.mu - sqrt(point.sigma2), 0.00000001f);

            //search epiline-----------------------------------------------------------------------
           // search_epiline_ncc (point, frame, KRKi_cr, Kt_cr );
            // search_epiline_bca (point, frames[i], KRKi_cr, Kt_cr, affine_cr);
            float idepth_mean = (point.idepth_min + point.idepth_max)*0.5;
            Eigen::Vector3f pr = KRKi_cr * Eigen::Vector3f(point.u,point.v, 1);
            Eigen::Vector3f ptpMean = pr + Kt_cr*idepth_mean;
            Eigen::Vector3f ptpMin = pr + Kt_cr*point.idepth_min;
            Eigen::Vector3f ptpMax = pr + Kt_cr*point.idepth_max;
            Eigen::Vector2f uvMean = ptpMean.hnormalized();
            Eigen::Vector2f uvMin = ptpMin.hnormalized();
            Eigen::Vector2f uvMax = ptpMax.hnormalized();

            // Pattern pattern_rot=m_pattern.get_rotated_pattern( KRKi_cr.topLeftCorner<2,2>() );

            Eigen::Vector2f epi_line = uvMax - uvMin;
            float norm_epi = std::max<float>(1e-5f,epi_line.norm());
            Eigen::Vector2f epi_dir = epi_line / norm_epi;
            const float  half_length = 0.5f * norm_epi;

            Eigen::Vector2f bestKp;
            float bestEnergy = 1e10;

            for(float l = -half_length; l <= half_length; l += 0.7f)
            {
                float energy = 0;
                Eigen::Vector2f kp = uvMean + l*epi_dir;

                if( !kp.allFinite() || ( kp(0) >= (frames[i].gray.cols-10) )  || ( kp(1) >= (frames[i].gray.rows-10) ) || ( kp(0) < 10 ) || ( kp(1) < 10) )
                {
                    continue;
                }

                for(int idx=0;idx<pattern_rot.get_nr_points(); ++idx)
                {
                    //float hitColor = getInterpolatedElement31(frame->dI, (float)(kp(0)+rotatetPattern[idx][0]), (float)(kp(1)+rotatetPattern[idx][1]), wG[0]);
                    Eigen::Vector2f offset=pattern_rot.get_offset(idx);
                    float hit_color=texture_interpolate(frames[i].gray, kp(0)+offset(0), kp(1)+offset(1) , InterpolationType::LINEAR);
                    if(!std::isfinite(hit_color)) {energy-=1e5; continue;}

                    const float residual = hit_color - (float)(affine_cr[0] * point.color[idx] + affine_cr[1]);

                    float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
                    energy += hw *residual*residual*(2-hw);
                }
                if ( energy < bestEnergy )
                {
                    bestKp = kp; bestEnergy = energy;
                }
            }

            if ( bestEnergy > point.energyTH * 1.2f ) {
                point.lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
            }
            else
            {
                // float a = (Eigen::Vector2f(epi_dir(0),epi_dir(1)).transpose() * point.gradH * Eigen::Vector2f(epi_dir(0),epi_dir(1)));
                // float b = (Eigen::Vector2f(epi_dir(1),-epi_dir(0)).transpose() * point.gradH * Eigen::Vector2f(epi_dir(1),-epi_dir(0)));
                // float errorInPixel = 0.2f + 0.2f * (a+b) / a; // WO kommt das her? Scheint nicht zu NGF zu passen !
                float errorInPixel=0.0f;

                if( epi_dir(0)*epi_dir(0)>epi_dir(1)*epi_dir(1) )
                {
                    point.idepth_min = (pr[2]*(bestKp(0)-errorInPixel*epi_dir(0)) - pr[0]) / (Kt_cr[0] - Kt_cr[2]*(bestKp(0)-errorInPixel*epi_dir(0)));
                    point.idepth_max = (pr[2]*(bestKp(0)+errorInPixel*epi_dir(0)) - pr[0]) / (Kt_cr[0] - Kt_cr[2]*(bestKp(0)+errorInPixel*epi_dir(0)));
                }
                else
                {
                    point.idepth_min = (pr[2]*(bestKp(1)-errorInPixel*epi_dir(1)) - pr[1]) / (Kt_cr[1] - Kt_cr[2]*(bestKp(1)-errorInPixel*epi_dir(1)));
                    point.idepth_max = (pr[2]*(bestKp(1)+errorInPixel*epi_dir(1)) - pr[1]) / (Kt_cr[1] - Kt_cr[2]*(bestKp(1)+errorInPixel*epi_dir(1)));
                }
                if(point.idepth_min > point.idepth_max) std::swap<float>(point.idepth_min, point.idepth_max);

                point.lastTraceStatus = ImmaturePointStatus::IPS_GOOD;
            }







            double idepth = -1;
            double z = 0;
            if( point.lastTraceStatus == ImmaturePointStatus::IPS_GOOD ) {
                idepth = std::max<double>(1e-5,.5*(point.idepth_min+point.idepth_max));
                z = 1./idepth;
            }
            if ( point.lastTraceStatus == ImmaturePointStatus::IPS_OOB  || point.lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED ){
                continue;
            }
            if ( !std::isfinite(idepth) || point.lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER || point.lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION ) {
                point.b++; // increase outlier probability when no match was found
                continue;
            }


            // update_idepth(point,tf_host_cur, z, px_error_angle);

            // compute tau----------------------------------------------------------------------------
            // double tau = compute_tau(tf_host_cur, point.f, z, px_error_angle);
            Eigen::Vector3f t=  Eigen::Vector3f(tf_host_cur_4x4(0,3), tf_host_cur_4x4(1,3), tf_host_cur_4x4(2,3));
            // Eigen::Vector3f t(tf_host_cur.translation());
            Eigen::Vector3f a = point.f.head<3>()*z-t;
            double t_norm = t.norm();
            double a_norm = a.norm();
            double alpha = acos(point.f.head<3>().dot(t)/t_norm); // dot product
            double beta = acos(a.dot(-t)/(t_norm*a_norm)); // dot product
            double beta_plus = beta + px_error_angle;
            double gamma_plus = M_PI-alpha-beta_plus; // triangle angles sum to PI
            double z_plus = t_norm*sin(beta_plus)/sin(gamma_plus); // law of sines
            double tau= (z_plus - z); // tau
            double tau_inverse = 0.5 * (1.0/std::max<double>(0.0000001, z-tau) - 1.0/(z+tau));

            // update the estimate--------------------------------------------------
            float x=1.0/z;
            float tau2=tau_inverse*tau_inverse;
            // updateSeed(point, 1.0/z, tau_inverse*tau_inverse);
            float norm_scale = sqrt(point.sigma2 + tau2);
            if(std::isnan(norm_scale))
                continue;
            float s2 = 1./(1./point.sigma2 + 1./tau2);
            float m = s2*(point.mu/point.sigma2 + x/tau2);
            float C1 = point.a/(point.a+point.b) * gaus_pdf(point.mu, norm_scale, x);
            float C2 = point.b/(point.a+point.b) * 1./point.z_range;
            float normalization_constant = C1 + C2;
            C1 /= normalization_constant;
            C2 /= normalization_constant;
            float f = C1*(point.a+1.)/(point.a+point.b+1.) + C2*point.a/(point.a+point.b+1.);
            float e = C1*(point.a+1.)*(point.a+2.)/((point.a+point.b+1.)*(point.a+point.b+2.))
                      + C2*point.a*(point.a+1.0f)/((point.a+point.b+1.0f)*(point.a+point.b+2.0f));
            // update parameters
            float mu_new = C1*m+C2*point.mu;
            point.sigma2 = C1*(s2 + m*m) + C2*(point.sigma2 + point.mu*point.mu) - mu_new*mu_new;
            point.mu = mu_new;
            point.a = (e-f)/(f-e/f);
            point.b = point.a*(1.0f-f)/f;


            //not implemented in opengl
            // const float eta_inlier = .6f;
            // const float eta_outlier = .05f;
            // if( ((point.a / (point.a + point.b)) > eta_inlier) && (sqrt(point.sigma2) < point.z_range/seed_convergence_sigma2_thresh)) {
            //     point.is_outlier = false; // The seed converged
            // }else if((point.a-1) / (point.a + point.b - 2) < eta_outlier){ // The seed failed to converge
            //     point.is_outlier = true;
            //     // it->reinit();
            //     //TODO do a better reinit inside a point class
            //     point.a = 10;
            //     point.b = 10;
            //     point.mu = (1.0/4.0);
            //     point.z_range = (1.0/0.1);
            //     point.sigma2 = (point.z_range*point.z_range/36);
            // }
            // // if the seed has converged, we initialize a new candidate point and remove the seed
            // if(sqrt(point.sigma2) < point.z_range/seed_convergence_sigma2_thresh){
            //     point.converged = true;
            // }



        }


    }

    TIME_END("compute_depth");


    mesh=create_mesh(immature_points,frames); //creates a mesh from the position of the points and their depth
    return mesh;

}

std::vector<Frame> DepthEstimatorCPU::loadDataFromICLNUIM ( const std::string & dataset_path, const int num_images_to_read){
   std::vector< Frame > frames;
   std::string filename_img = dataset_path + "/associations.txt";
   std::string filename_gt = dataset_path + "/livingRoom2.gt.freiburg";

   //K is from here https://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html
   Eigen::Matrix3f K;
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
         Eigen::Affine3f pose_wc = Eigen::Affine3f::Identity();
         pose_wc.translation() << tx,ty,tz;
         pose_wc.linear() = Eigen::Quaternionf(qw,qx,qy,qz).toRotationMatrix();
         Eigen::Affine3f pose_cw = pose_wc.inverse();

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
         cur_frame.frame_idx=imagesRead;

         frames.push_back(cur_frame);
         VLOG(1) << "read img " << imagesRead << " " << colorFileName;
      }
   }
   std::cout << "read " << imagesRead << " images. (" << frames.size() <<", " << ")" << std::endl;
   return frames;
}

std::vector<ImmaturePoint> DepthEstimatorCPU::create_immature_points (const Frame& frame){


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
    TIME_START("sobel_host_frame");
    cv::Mat grad_x, grad_y;
    cv::Scharr( frame.gray, grad_x, CV_32F, 1, 0);
    cv::Scharr( frame.gray, grad_y, CV_32F, 0, 1);
    TIME_END("sobel_host_frame");

    TIME_START("hessian_host_frame");
    std::vector<ImmaturePoint> immature_points;
    immature_points.reserve(200000);
    for (size_t i = 10; i < frame.gray.rows-10; i++) {  //--------Do not look around the borders to avoid pattern accesing outside img
        for (size_t j = 10; j < frame.gray.cols-10; j++) {

            //check if this point has enough determinant in the hessian
            Eigen::Matrix2f gradient_hessian;
            gradient_hessian.setZero();
            for (size_t p = 0; p < m_pattern.get_nr_points(); p++) {
                int dx = m_pattern.get_offset_x(p);
                int dy = m_pattern.get_offset_y(p);

                float gradient_x=grad_x.at<float>(i+dy,j+dx); //TODO should be interpolated
                float gradient_y=grad_y.at<float>(i+dy,j+dx);

                Eigen::Vector2f grad;
                grad << gradient_x, gradient_y;
                // std::cout << "gradients are " << gradient_x << " " << gradient_y << '\n';

                gradient_hessian+= grad*grad.transpose();
            }

            //determinant is high enough, add the point
            float hessian_det=gradient_hessian.determinant();
            if(hessian_det > 100000000){
                ImmaturePoint point;
                point.u=j;
                point.v=i;
                point.gradH=gradient_hessian;

                //Seed::Seed
                Eigen::Vector3f f_eigen = (frame.K.inverse() * Eigen::Vector3f(point.u,point.v,1)).normalized();
                point.f = Eigen::Vector4f(f_eigen(0),f_eigen(1),f_eigen(2), 1.0);

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
                point.lastTraceStatus=IPS_UNINITIALIZED;

                //get data for the color of that point (depth_point->ImmaturePoint::ImmaturePoint)---------------------
                for(int p_idx=0;p_idx<m_pattern.get_nr_points();p_idx++){
                    Eigen::Vector2f offset = m_pattern.get_offset(p_idx);

                    point.color[p_idx]=texture_interpolate(frame.gray, point.u+offset(0), point.v+offset(1));

                    float grad_x_val=texture_interpolate(grad_x, point.u+offset(0), point.v+offset(1));
                    float grad_y_val=texture_interpolate(grad_y, point.u+offset(0), point.v+offset(1));
                    float squared_norm=grad_x_val*grad_x_val + grad_y_val*grad_y_val;
                    point.weights[p_idx] = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + squared_norm));
                }
                point.ncc_sum_templ    = 0.0f;
                float ncc_sum_templ_sq = 0.0f;
                for(int p_idx=0;p_idx<m_pattern.get_nr_points();p_idx++){
                    const float templ = point.color[p_idx];
                    point.ncc_sum_templ += templ;
                    ncc_sum_templ_sq += templ*templ;
                }
                point.ncc_const_templ = m_pattern.get_nr_points() * ncc_sum_templ_sq - (double) point.ncc_sum_templ*point.ncc_sum_templ;

                point.energyTH = m_pattern.get_nr_points()*setting_outlierTH;
                point.energyTH *= setting_overallEnergyTHWeight*setting_overallEnergyTHWeight;

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
    TIME_END("hessian_host_frame");




    return immature_points;
}

Eigen::Vector2f DepthEstimatorCPU::estimate_affine(std::vector<ImmaturePoint>& immature_points, const Frame&  cur_frame, const Eigen::Matrix3f& KRKi_cr, const Eigen::Vector3f& Kt_cr){
    // ceres::Problem problem;
    // ceres::LossFunction * loss_function = new ceres::HuberLoss(1);
    // double scaleA = 1;
    // double offsetB = 0;
    //
    // TIME_START("creating ceres problem");
    // for ( int i = 0; i < immature_points.size(); ++i )
    // {
    //     ImmaturePoint& point = immature_points[i];
    //     if ( i % 100 != 0 )
    //         continue;
    //
    //     //get colors at the current frame
    //     float color_cur_frame[MAX_RES_PER_POINT];
    //     float color_host_frame[MAX_RES_PER_POINT];
    //
    //
    //     if ( 1.0/point.gt_depth > 0 ) {
    //
    //         const Eigen::Vector3f p = KRKi_cr * Eigen::Vector3f(point.u,point.v,1) + Kt_cr*  (1.0/point.gt_depth);
    //         point.kp_GT = p.hnormalized();
    //
    //
    //         if ( point.kp_GT(0) > 4 && point.kp_GT(0) < cur_frame.gray.cols-4 && point.kp_GT(1) > 3 && point.kp_GT(1) < cur_frame.gray.rows-4 ) {
    //
    //             Pattern pattern_rot=m_pattern.get_rotated_pattern( KRKi_cr.topLeftCorner<2,2>() );
    //
    //             for(int idx=0;idx<m_pattern.get_nr_points();++idx) {
    //                 Eigen::Vector2f offset=pattern_rot.get_offset(idx);
    //
    //                 color_cur_frame[idx]=texture_interpolate(cur_frame.gray, point.kp_GT(0)+offset(0), point.kp_GT(1)+offset(1) , InterpolationType::LINEAR);
    //                 color_host_frame[idx]=point.color[idx];
    //
    //             }
    //         }
    //     }
    //
    //
    //     for ( int i = 0; i < m_pattern.get_nr_points(); ++i) {
    //         if ( !std::isfinite(color_host_frame[i]) || ! std::isfinite(color_cur_frame[i]) )
    //             continue;
    //         if ( color_host_frame[i] <= 0 || color_host_frame[i] >= 255 || color_cur_frame[i] <= 0 || color_cur_frame[i] >= 255  )
    //             continue;
    //         ceres::CostFunction * cost_function = AffineAutoDiffCostFunctor::Create( color_cur_frame[i], color_host_frame[i] );
    //         problem.AddResidualBlock( cost_function, loss_function, &scaleA, & offsetB );
    //     }
    // }
    // TIME_END("creating ceres problem");
    // ceres::Solver::Options solver_options;
    // //solver_options.linear_solver_type = ceres::DENSE_QR;//DENSE_SCHUR;//QR;
    // solver_options.minimizer_progress_to_stdout = false;
    // solver_options.max_num_iterations = 1000;
    // solver_options.function_tolerance = 1e-6;
    // solver_options.num_threads = 8;
    // ceres::Solver::Summary summary;
    // ceres::Solve( solver_options, & problem, & summary );
    // //std::cout << summary.FullReport() << std::endl;
    // std::cout << "scale= " << scaleA << " offset= "<< offsetB << std::endl;
    // return Eigen::Vector2f ( scaleA, offsetB );

    return Eigen::Vector2f ( 1.0, 0.0 );
}

float DepthEstimatorCPU::texture_interpolate ( const cv::Mat& img, const float x, const float y , const InterpolationType type){
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
    if(type==InterpolationType::NEAREST){
        return NearestInterpolator<float>::getValue(img,y,x,0,cv::BORDER_REPLICATE);
    }else if(type==InterpolationType::LINEAR){
        return LinearInterpolator<float>::getValue(img,y,x,0,cv::BORDER_REPLICATE);
    }else if(type==InterpolationType::CUBIC){
        return CubicInterpolator<float>::getValue(img,y,x,0,cv::BORDER_REPLICATE);
    }

}

void DepthEstimatorCPU::update_immature_points(std::vector<ImmaturePoint>& immature_points, const Frame& frame, const Eigen::Affine3f& tf_cur_host, const Eigen::Matrix3f& KRKi_cr, const Eigen::Vector3f& Kt_cr, const Eigen::Vector2f& affine_cr){

    // TIME_START("update_immature_points");
    //
    // const double focal_length = abs(frame.K(0,0));
    // double px_noise = 1.0;
    // double px_error_angle = atan(px_noise/(2.0*focal_length))*2.0; // law of chord (sehnensatz)
    // const Eigen::Affine3f tf_host_cur = tf_cur_host.inverse();
    //
    //
    // for (auto &point : immature_points){
    //
    //     // check if point is visible in the current image
    //     const Eigen::Vector3f xyz_f( tf_cur_host*(1.0/point.mu * point.f) );
    //     if(xyz_f.z() < 0.0)  {
    //         continue;
    //     }
    //     const Eigen::Vector2f kp_c = (frame.K * xyz_f).hnormalized();
    //     if ( kp_c(0) < 0 || kp_c(0) >= frame.gray.cols || kp_c(1) < 0 || kp_c(1) >= frame.gray.rows ) {
    //         continue;
    //     }
    //
    //
    //     //point is visible
    //     point.last_visible_frame=frame.frame_id;
    //
    //     //update inverse depth coordinates for min and max
    //     point.idepth_min = point.mu + sqrt(point.sigma2);
    //     point.idepth_max = std::max<float>(point.mu - sqrt(point.sigma2), 0.00000001f);
    //
    //     //search epiline----------------------
    //    // search_epiline_ncc (point, frame, KRKi_cr, Kt_cr );
    //     search_epiline_bca (point, frame, KRKi_cr, Kt_cr, affine_cr);
    //
    //     double idepth = -1;
    //     double z = 0;
    //     if( point.lastTraceStatus == ImmaturePointStatus::IPS_GOOD ) {
    //         idepth = std::max<double>(1e-5,.5*(point.idepth_min+point.idepth_max));
    //         z = 1./idepth;
    //     }
    //     if ( point.lastTraceStatus == ImmaturePointStatus::IPS_OOB  || point.lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED ){
    //         continue;
    //     }
    //     if ( !std::isfinite(idepth) || point.lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER || point.lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION ) {
    //         point.b++; // increase outlier probability when no match was found
    //         continue;
    //     }
    //
    //
    //     update_idepth(point,tf_host_cur, z, px_error_angle);
    //
    //
    //
    // }
    //
    // TIME_END("update_immature_points");
}

void DepthEstimatorCPU::search_epiline_bca(ImmaturePoint& point, const Frame& frame, const Eigen::Matrix3f& KRKi_cr, const Eigen::Vector3f& Kt_cr, const Eigen::Vector2f& affine_cr){

    // if(point.lastTraceStatus == ImmaturePointStatus::IPS_OOB || point.lastTraceStatus == ImmaturePointStatus::IPS_DELETED ) return;
    //
    //
    // float idepth_mean = (point.idepth_min + point.idepth_max)*0.5;
    // Eigen::Vector3f pr = KRKi_cr * Eigen::Vector3f(point.u,point.v, 1);
    // Eigen::Vector3f ptpMean = pr + Kt_cr*idepth_mean;
    // Eigen::Vector3f ptpMin = pr + Kt_cr*point.idepth_min;
    // Eigen::Vector3f ptpMax = pr + Kt_cr*point.idepth_max;
    // Eigen::Vector2f uvMean = ptpMean.hnormalized();
    // Eigen::Vector2f uvMin = ptpMin.hnormalized();
    // Eigen::Vector2f uvMax = ptpMax.hnormalized();
    //
    // Pattern pattern_rot=m_pattern.get_rotated_pattern( KRKi_cr.topLeftCorner<2,2>() );
    //
    // Eigen::Vector2f epi_line = uvMax - uvMin;
    // float norm_epi = std::max<float>(1e-5f,epi_line.norm());
    // Eigen::Vector2f epi_dir = epi_line / norm_epi;
    // const float  half_length = 0.5f * norm_epi;
    //
    // Eigen::Vector2f bestKp;
    // float bestEnergy = 1e10;
    //
    // for(float l = -half_length; l <= half_length; l += 0.7f)
    // {
    //     float energy = 0;
    //     Eigen::Vector2f kp = uvMean + l*epi_dir;
    //
    //     if( !kp.allFinite() || ( kp(0) >= (frame.gray.cols-10) )  || ( kp(1) >= (frame.gray.rows-10) ) || ( kp(0) < 10 ) || ( kp(1) < 10) )
    //     {
    //         continue;
    //     }
    //
    //     for(int idx=0;idx<pattern_rot.get_nr_points(); ++idx)
    //     {
    //         //float hitColor = getInterpolatedElement31(frame->dI, (float)(kp(0)+rotatetPattern[idx][0]), (float)(kp(1)+rotatetPattern[idx][1]), wG[0]);
    //         Eigen::Vector2f offset=pattern_rot.get_offset(idx);
    //         float hit_color=texture_interpolate(frame.gray, kp(0)+offset(0), kp(1)+offset(1) , InterpolationType::LINEAR);
    //         if(!std::isfinite(hit_color)) {energy-=1e5; continue;}
    //
    //         const float residual = hit_color - (float)(affine_cr[0] * point.color[idx] + affine_cr[1]);
    //
    //         float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
    //         energy += hw *residual*residual*(2-hw);
    //     }
    //     if ( energy < bestEnergy )
    //     {
    //         bestKp = kp; bestEnergy = energy;
    //     }
    // }
    //
    // if ( bestEnergy > point.energyTH * 1.2f ) {
    //     point.lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
    // }
    // else
    // {
    //     float a = (Eigen::Vector2f(epi_dir(0),epi_dir(1)).transpose() * point.gradH * Eigen::Vector2f(epi_dir(0),epi_dir(1)));
    //     float b = (Eigen::Vector2f(epi_dir(1),-epi_dir(0)).transpose() * point.gradH * Eigen::Vector2f(epi_dir(1),-epi_dir(0)));
    //     float errorInPixel = 0.2f + 0.2f * (a+b) / a; // WO kommt das her? Scheint nicht zu NGF zu passen !
    //
    //     if( epi_dir(0)*epi_dir(0)>epi_dir(1)*epi_dir(1) )
    //     {
    //         point.idepth_min = (pr[2]*(bestKp(0)-errorInPixel*epi_dir(0)) - pr[0]) / (Kt_cr[0] - Kt_cr[2]*(bestKp(0)-errorInPixel*epi_dir(0)));
    //         point.idepth_max = (pr[2]*(bestKp(0)+errorInPixel*epi_dir(0)) - pr[0]) / (Kt_cr[0] - Kt_cr[2]*(bestKp(0)+errorInPixel*epi_dir(0)));
    //     }
    //     else
    //     {
    //         point.idepth_min = (pr[2]*(bestKp(1)-errorInPixel*epi_dir(1)) - pr[1]) / (Kt_cr[1] - Kt_cr[2]*(bestKp(1)-errorInPixel*epi_dir(1)));
    //         point.idepth_max = (pr[2]*(bestKp(1)+errorInPixel*epi_dir(1)) - pr[1]) / (Kt_cr[1] - Kt_cr[2]*(bestKp(1)+errorInPixel*epi_dir(1)));
    //     }
    //     if(point.idepth_min > point.idepth_max) std::swap<float>(point.idepth_min, point.idepth_max);
    //
    //     point.lastTraceStatus = ImmaturePointStatus::IPS_GOOD;
    // }
}

void DepthEstimatorCPU::search_epiline_ncc(ImmaturePoint& point, const Frame& frame, const Eigen::Matrix3f& KRKi_cr, const Eigen::Vector3f& Kt_cr){

   // Eigen::Vector3d pr = KRKi_cr * Eigen::Vector3d(point.u,point.v, 1);
   // Eigen::Vector3d ptpMean = pr + Kt_cr*point.mu;
   // Eigen::Vector3d ptpMin = pr + Kt_cr*point.idepth_min;
   // Eigen::Vector3d ptpMax = pr + Kt_cr*point.idepth_max;
   // Eigen::Vector2d uvMean = ptpMean.hnormalized();
   // Eigen::Vector2d uvMin = ptpMin.hnormalized();
   // Eigen::Vector2d uvMax = ptpMax.hnormalized();
   //
   // Pattern pattern_rot=m_pattern.get_rotated_pattern( KRKi_cr.topLeftCorner<2,2>() );
   //
   // Eigen::Vector2d epi_line = uvMax - uvMin;
   // float norm_epi = std::max<float>(1e-5f,epi_line.norm());
   // Eigen::Vector2d epi_dir = epi_line / norm_epi;
   // const float  half_length = 0.5f * norm_epi;
   //
   // Eigen::Vector2d bestKp;
   // float bestEnergy = -1.0f;
   //
   //  // Retrieve template statistics for NCC matching;
   //  const float sum_templ = point.ncc_sum_templ ;
   //  const float const_templ_denom = point.ncc_const_templ;
   //
   //  for(float l = -half_length; l <= half_length; l += 0.7f)
   //  {
   //      float energy = 0;
   //      float sum_img = 0.f;
   //      float sum_img_sq = 0.f;
   //      float sum_img_templ = 0.f;
   //
   //      Eigen::Vector2d kp = uvMean + l*epi_dir;
   //
   //      if( !kp.allFinite() || ( kp(0) >= (frame.gray.cols-7) )  || ( kp(1) >= (frame.gray.rows-7) ) || ( kp(0) < 7 ) || ( kp(1) < 7) ) {
   //        continue;
   //      }
   //
   //      for(int idx=0;idx<pattern_rot.get_nr_points(); ++idx)
   //      {
   //          //float hitColor = getInterpolatedElement31(frame->dI, (float)(kp(0)+rotatetPattern[idx][0]), (float)(kp(1)+rotatetPattern[idx][1]), wG[0]);
   //
   //
   //          Eigen::Vector2d offset=pattern_rot.get_offset(idx);
   //          float hit_color=texture_interpolate(frame.gray, kp(0)+offset(0), kp(1)+offset(1) );
   //
   //
   //          const float templ = point.color[idx];
   //          const float img = hit_color;
   //          sum_img    += img;
   //          sum_img_sq += img*img;
   //          sum_img_templ += img*templ;
   //      }
   //      const float ncc_numerator = pattern_rot.get_nr_points()*sum_img_templ - sum_img*sum_templ;
   //      const float ncc_denominator = (pattern_rot.get_nr_points()*sum_img_sq - sum_img*sum_img)*const_templ_denom;
   //      energy += ncc_numerator * sqrt(ncc_denominator + 1e-10);
   //
   //      if( energy > bestEnergy )
   //      {
   //          bestKp = kp; bestEnergy = energy;
   //      }
   //  }
   //
   //  if( bestEnergy < .5f ) {
   //      point.lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
   //  } else {
   //      //TODO WTF is this??
   //      // float a = (Eigen::Vector2f(epi_dir(0),epi_dir(1)).transpose() * gradH * Eigen::Vector2f(epi_dir(0),epi_dir(1)));
   //      // float b = (Eigen::Vector2f(epi_dir(1),-epi_dir(0)).transpose() * gradH * Eigen::Vector2f(epi_dir(1),-epi_dir(0)));
   //      // float errorInPixel = 0.2f + 0.2f * (a+b) / a; // WO kommt das her? Scheint nicht zu NGF zu passen !
   //      float errorInPixel=0.0;
   //
   //      if( epi_dir(0)*epi_dir(0)>epi_dir(1)*epi_dir(1) )
   //      {
   //          point.idepth_min = (pr[2]*(bestKp(0)-errorInPixel*epi_dir(0)) - pr[0]) / (Kt_cr[0] - Kt_cr[2]*(bestKp(0)-errorInPixel*epi_dir(0)));
   //          point.idepth_max = (pr[2]*(bestKp(0)+errorInPixel*epi_dir(0)) - pr[0]) / (Kt_cr[0] - Kt_cr[2]*(bestKp(0)+errorInPixel*epi_dir(0)));
   //      }
   //      else
   //      {
   //          point.idepth_min = (pr[2]*(bestKp(1)-errorInPixel*epi_dir(1)) - pr[1]) / (Kt_cr[1] - Kt_cr[2]*(bestKp(1)-errorInPixel*epi_dir(1)));
   //          point.idepth_max = (pr[2]*(bestKp(1)+errorInPixel*epi_dir(1)) - pr[1]) / (Kt_cr[1] - Kt_cr[2]*(bestKp(1)+errorInPixel*epi_dir(1)));
   //      }
   //      if(point.idepth_min > point.idepth_max) std::swap<float>(point.idepth_min, point.idepth_max);
   //
   //      // lastTraceUV = bestKp;
   //      point.lastTraceStatus = ImmaturePointStatus::IPS_GOOD;
   //  }
}

void DepthEstimatorCPU::update_idepth(ImmaturePoint& point, const Eigen::Affine3f& tf_host_cur, const float z, const double px_error_angle){

    // // compute tau
    // double tau = compute_tau(tf_host_cur, point.f, z, px_error_angle);
    // double tau_inverse = 0.5 * (1.0/std::max<double>(0.0000001, z-tau) - 1.0/(z+tau));
    //
    // // update the estimate
    // updateSeed(point, 1.0/z, tau_inverse*tau_inverse);
    //
    //
    // const float eta_inlier = .6f;
    // const float eta_outlier = .05f;
    // if( ((point.a / (point.a + point.b)) > eta_inlier) && (sqrt(point.sigma2) < point.z_range/seed_convergence_sigma2_thresh)) {
    //     point.is_outlier = false; // The seed converged
    // }else if((point.a-1) / (point.a + point.b - 2) < eta_outlier){ // The seed failed to converge
    //     point.is_outlier = true;
    //     // it->reinit();
    //     //TODO do a better reinit inside a point class
    //     point.a = 10;
    //     point.b = 10;
    //     point.mu = (1.0/4.0);
    //     point.z_range = (1.0/0.1);
    //     point.sigma2 = (point.z_range*point.z_range/36);
    // }
    // // if the seed has converged, we initialize a new candidate point and remove the seed
    // if(sqrt(point.sigma2) < point.z_range/seed_convergence_sigma2_thresh){
    //     point.converged = true;
    // }

}

double DepthEstimatorCPU::compute_tau(const Eigen::Affine3f & tf_host_cur, const Eigen::Vector3f& f, const double z, const double px_error_angle){


    return -1;

    // Eigen::Vector3f t(tf_host_cur.translation());
    // Eigen::Vector3f a = f*z-t;
    // double t_norm = t.norm();
    // double a_norm = a.norm();
    // double alpha = acos(f.dot(t)/t_norm); // dot product
    // double beta = acos(a.dot(-t)/(t_norm*a_norm)); // dot product
    // double beta_plus = beta + px_error_angle;
    // double gamma_plus = M_PI-alpha-beta_plus; // triangle angles sum to PI
    // double z_plus = t_norm*sin(beta_plus)/sin(gamma_plus); // law of sines
    // return (z_plus - z); // tau


}

void DepthEstimatorCPU::updateSeed(ImmaturePoint& point, const float x, const float tau2) {
    float norm_scale = sqrt(point.sigma2 + tau2);
    if(std::isnan(norm_scale))
        return;
    boost::math::normal_distribution<float> nd(point.mu, norm_scale);
    float s2 = 1./(1./point.sigma2 + 1./tau2);
    float m = s2*(point.mu/point.sigma2 + x/tau2);
    float C1 = point.a/(point.a+point.b) * boost::math::pdf(nd, x);
    float C2 = point.b/(point.a+point.b) * 1./point.z_range;
    float normalization_constant = C1 + C2;
    C1 /= normalization_constant;
    C2 /= normalization_constant;
    float f = C1*(point.a+1.)/(point.a+point.b+1.) + C2*point.a/(point.a+point.b+1.);
    float e = C1*(point.a+1.)*(point.a+2.)/((point.a+point.b+1.)*(point.a+point.b+2.))
              + C2*point.a*(point.a+1.0f)/((point.a+point.b+1.0f)*(point.a+point.b+2.0f));

    // update parameters
    float mu_new = C1*m+C2*point.mu;
    point.sigma2 = C1*(s2 + m*m) + C2*(point.sigma2 + point.mu*point.mu) - mu_new*mu_new;
    point.mu = mu_new;
    point.a = (e-f)/(f-e/f);
    point.b = point.a*(1.0f-f)/f;
}

Mesh DepthEstimatorCPU::create_mesh(const std::vector<ImmaturePoint>& immature_points, const std::vector<Frame>& frames){
    Mesh mesh;
    mesh.V.resize(immature_points.size(),3);
    mesh.V.setZero();

    for (size_t i = 0; i < immature_points.size(); i++) {
        float u=immature_points[i].u;
        float v=immature_points[i].v;
        float depth=1/immature_points[i].mu;

        if(std::isfinite(immature_points[i].mu) && immature_points[i].mu>=0.1){
            //backproject the immature point
            Eigen::Vector3f point_screen;
            point_screen << u, v, 1.0;
            Eigen::Vector3f point_dir=frames[0].K.inverse()*point_screen; //TODO get the K and also use the cam to world corresponding to the immature point
            // point_dir.normalize(); //this is just the direction (in cam coordinates) of that pixel
            Eigen::Vector3f point_cam = point_dir*depth;
            point_cam(2)=-point_cam(2); //flip the depth because opengl has a camera which looks at the negative z axis (therefore, more depth means a more negative number)


            Eigen::Vector3f point_world=frames[0].tf_cam_world.inverse()*point_cam;

            mesh.V.row(i)=point_world.cast<double>();
        }


    }

    //make also some colors based on depth
    mesh.C.resize(immature_points.size(),3);
    double min_z, max_z;
    min_z = mesh.V.col(2).minCoeff();
    max_z = mesh.V.col(2).maxCoeff();
    std::cout << "min max z is " << min_z << " " << max_z << '\n';
    for (size_t i = 0; i < mesh.C.rows(); i++) {
        float gray_val = lerp(mesh.V(i,2), min_z, max_z, 0.0, 1.0 );
        mesh.C(i,0)=mesh.C(i,1)=mesh.C(i,2)=gray_val;
    }

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
   // float min=9999999999, max=-9999999999;
   // for (size_t i = 0; i < immature_points.size(); i++) {
   //     if(immature_points[i].last_visible_frame<min){
   //         min=immature_points[i].last_visible_frame;
   //     }
   //     if(immature_points[i].last_visible_frame>max){
   //         max=immature_points[i].last_visible_frame;
   //     }
   // }
   // std::cout << "min max z is " << min << " " << max << '\n';
   // for (size_t i = 0; i < mesh.C.rows(); i++) {
   //      float gray_val = lerp(immature_points[i].last_visible_frame, min, max, 0.0, 1.0 );
   //      mesh.C(i,0)=mesh.C(i,1)=mesh.C(i,2)=gray_val;
   //  }






    return mesh;
}


//are we going to use the same windows approach as in dso which add frames and then margianzlies them afterwards? because that is tricky to do in opencl and would require some vector on the gpu which keeps track where the image are in a 3D image and then do like a rolling buffer

//get frame to gray
//get frame to float
//apply blur to img (as done at the finale of Undistort::undistort)

//cv_mat2cl_buf(img_gray)





// full_system:makeNewTraces //computes new imature points and adds them to the current frame
//     pixelSelector->make_maps()
//     for (size_t i = 0; i < count; i++) {
//         for (size_t i = 0; i < count; i++) {
//             if(selectionMap==)continue
//             create_imature_point which contains
//                 weights for each point in the patter (default 8)
//                 gradH value
//                 energuTH
//                 idepthGT
//                 quality
//         }
//     }
