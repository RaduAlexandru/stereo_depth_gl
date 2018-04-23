#include "stereo_depth_cl/DepthEstimatorGL.h"

//c++
#include <cmath>
#include <stdlib.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>

//My stuff
#include "stereo_depth_cl/Profiler.h"
#include "stereo_depth_cl/MiscUtils.h"
#include "cv_interpolation.h"
#include "UtilsGL.h"
#include "Shader.h"

//Libigl
#include <igl/opengl/glfw/Viewer.h>

//cv
#include <cv_bridge/cv_bridge.h>

//loguru
#include <loguru.hpp>

//gl
#include <glm/gtc/type_ptr.hpp>

using namespace glm;

DepthEstimatorGL::DepthEstimatorGL():
        m_scene_is_modified(false),
        m_gl_profiling_enabled(true),
        m_show_images(false)
        {

    init_opengl();
    std::string pattern_filepath="/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/data/pattern_1.png";
    m_pattern.init_pattern(pattern_filepath);

    //sanity check the pattern
    std::cout << "pattern has nr of points " << m_pattern.get_nr_points() << '\n';
    for (size_t i = 0; i < m_pattern.get_nr_points(); i++) {
        std::cout << "offset for i " << i << " is " << m_pattern.get_offset(i).transpose() << '\n';
    }


    //more sanity checks that ensure that however I pad the Point struct it will be correct
    assert(sizeof(float) == 4);
    assert(sizeof(int32_t) == 4);
}

//needed so that forward declarations work
DepthEstimatorGL::~DepthEstimatorGL(){
}

void DepthEstimatorGL::init_opengl(){
    std::cout << "init opengl" << '\n';

    if(GL_ARB_debug_output){
    	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB);
    	glDebugMessageCallbackARB(debug_func, (void*)15);
	}

    glGenBuffers(1, &m_points_gl_buf);

    m_cur_frame.set_wrap_mode(GL_CLAMP_TO_BORDER);
    m_cur_frame.set_filter_mode(GL_LINEAR);

    compile_shaders();

}

void DepthEstimatorGL::compile_shaders(){

    m_update_depth_prog_id=gl::program_init_from_files("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/shaders/compute_update_depth.glsl");

}

//https://www.boost.org/doc/libs/1_47_0/libs/math/doc/sf_and_dist/html/math_toolkit/dist/dist_ref/dists/normal_dist.html
float DepthEstimatorGL::gaus_pdf(float mean, float sd, float x){
    return exp(- (x-mean)*(x-mean)/(2*sd)*(2*sd)  )  / (sd*sqrt(2*M_PI));
}


Mesh DepthEstimatorGL::compute_depth(){
    Mesh mesh;

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

    TIME_START_GL("compute_depth");

    //calculate the gradients of it (cpu)
    //calculate immature points (cpu)
    //upload immature points vector to gpu
    //upload each new frame to gpu

    std::vector<Point> immature_points;
    immature_points=create_immature_points(frames[0]);
    std::cout << "immature_points size is " << immature_points.size() << '\n';




    //upload to gpu the inmature points
    TIME_START_GL("upload_immature_points");
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_points_gl_buf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, immature_points.size() * sizeof(Point), immature_points.data(), GL_DYNAMIC_COPY);
    TIME_END_GL("upload_immature_points");

    glUseProgram(m_update_depth_prog_id);
    for (size_t i = 1; i < frames.size(); i++) {
        TIME_START_GL("update_depth");

        TIME_START_GL("estimate_affine");
        const Eigen::Affine3f tf_cur_host_eigen = frames[i].tf_cam_world * frames[0].tf_cam_world.inverse();
        const Eigen::Affine3f tf_host_cur_eigen = tf_cur_host_eigen.inverse();
        const Eigen::Matrix3f KRKi_cr_eigen = frames[i].K * tf_cur_host_eigen.linear() * frames[0].K.inverse();
        const Eigen::Vector3f Kt_cr_eigen = frames[i].K * tf_cur_host_eigen.translation();
        const Eigen::Vector2f affine_cr_eigen = estimate_affine( immature_points, frames[i], KRKi_cr_eigen, Kt_cr_eigen);
        const double focal_length = fabs(frames[i].K(0,0));
        double px_noise = 1.0;
        double px_error_angle = atan(px_noise/(2.0*focal_length))*2.0; // law of chord (sehnensatz)
        Pattern pattern_rot=m_pattern.get_rotated_pattern( KRKi_cr_eigen.topLeftCorner<2,2>() );

        // std::cout << "pattern_rot has nr of points " << pattern_rot.get_nr_points() << '\n';
        // for (size_t i = 0; i < pattern_rot.get_nr_points(); i++) {
        //     std::cout << "offset for i " << i << " is " << pattern_rot.get_offset(i).transpose() << '\n';
        // }

        TIME_END_GL("estimate_affine");

        // //upload the image
        // TIME_START_GL("upload_gray_img");
        // int size_bytes=frames[i].gray.step[0] * frames[i].gray.rows;
        // m_cur_frame.upload_data(GL_R32F, frames[i].gray.cols, frames[i].gray.rows, GL_RED, GL_FLOAT, frames[i].gray.ptr(), size_bytes);
        // TIME_END_GL("upload_gray_img");


        // //attempt 2 at uploading image, this time with padding to be power of 2
        // TIME_START_GL("upload_gray_img");
        // int padded_img_size=1024;
        // cv::Mat padded_img(padded_img_size,padded_img_size,CV_32FC1);
        // // frames[i].gray.copyTo(padded_img(cv::Rect(0,0,frames[i].gray.cols, frames[i].gray.rows)));
        // frames[i].gray.copyTo(padded_img(cv::Rect(0,padded_img_size-frames[i].gray.rows,frames[i].gray.cols, frames[i].gray.rows)));
        // // cv::imshow("padded_img",padded_img);
        // // cv::waitKey(0);
        // int size_bytes=padded_img.step[0] * padded_img.rows;
        // m_cur_frame.upload_data(GL_R32F, padded_img.cols, padded_img.rows, GL_RED, GL_FLOAT, padded_img.ptr(), size_bytes);
        // TIME_END_GL("upload_gray_img");

        //attempt 3 upload the image as a inmutable storage
        TIME_START_GL("upload_gray_img");
        int size_bytes=frames[i].gray.step[0] * frames[i].gray.rows;
        if(!m_cur_frame.get_tex_storage_initialized()){
            m_cur_frame.allocate_tex_storage_inmutable(GL_R32F,frames[i].gray.cols, frames[i].gray.rows);
        }
        m_cur_frame.upload_without_pbo(0,0,0, frames[i].gray.cols, frames[i].gray.rows, GL_RED, GL_FLOAT, frames[i].gray.ptr());
        TIME_END_GL("upload_gray_img");





        //upload the matrices
        TIME_START_GL("upload_matrices");
        Eigen::Vector2f frame_size;
        frame_size<< frames[i].gray.cols, frames[i].gray.rows;
        const Eigen::Matrix4f tf_cur_host_eigen_trans = tf_cur_host_eigen.matrix().transpose();
        const Eigen::Matrix4f tf_host_cur_eigen_trans = tf_host_cur_eigen.matrix().transpose();
        glUniform2fv(glGetUniformLocation(m_update_depth_prog_id,"frame_size"), 1, frame_size.data());
        // glUniformMatrix4fv(glGetUniformLocation(m_update_depth_prog_id,"tf_cur_host"), 1, GL_FALSE, tf_cur_host_eigen_trans.data());
        // glUniformMatrix4fv(glGetUniformLocation(m_update_depth_prog_id,"tf_host_cur"), 1, GL_FALSE, tf_host_cur_eigen_trans.data());
        glUniformMatrix4fv(glGetUniformLocation(m_update_depth_prog_id,"tf_cur_host"), 1, GL_TRUE, tf_cur_host_eigen.data());
        glUniformMatrix4fv(glGetUniformLocation(m_update_depth_prog_id,"tf_host_cur"), 1, GL_TRUE, tf_host_cur_eigen.data());
        glUniformMatrix3fv(glGetUniformLocation(m_update_depth_prog_id,"K"), 1, GL_FALSE, frames[i].K.data());
        glUniformMatrix3fv(glGetUniformLocation(m_update_depth_prog_id,"KRKi_cr"), 1, GL_FALSE, KRKi_cr_eigen.data());
        glUniform3fv(glGetUniformLocation(m_update_depth_prog_id,"Kt_cr"), 1, Kt_cr_eigen.data());
        glUniform2fv(glGetUniformLocation(m_update_depth_prog_id,"affine_cr"), 1, affine_cr_eigen.data());
        glUniform1f(glGetUniformLocation(m_update_depth_prog_id,"px_error_angle"), px_error_angle);
        glUniform2fv(glGetUniformLocation(m_update_depth_prog_id,"pattern_rot_offsets"), pattern_rot.get_nr_points(), pattern_rot.get_offset_matrix().data()); //upload all the offses as an array of vec2 offsets
        // std::cout << "setting nr of points to " <<  pattern_rot.get_nr_points() << '\n';
        // std::cout << "the uniform location is " << glGetUniformLocation(m_update_depth_prog_id,"pattern_rot_nr_points") << '\n';
        glUniform1i(glGetUniformLocation(m_update_depth_prog_id,"pattern_rot_nr_points"), pattern_rot.get_nr_points());
        TIME_END_GL("upload_matrices");
        glMemoryBarrier(GL_ALL_BARRIER_BITS);


        // tf_cur_host, tf_host_cur, KRKi_cr, Kt_cr, affine_cr, px_error_angle
        TIME_START_GL("depth_update_kernel");
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_points_gl_buf);
        bind_for_sampling(m_cur_frame, 1, glGetUniformLocation(m_update_depth_prog_id,"gray_img_sampler") );
        glDispatchCompute(immature_points.size()/256, 1, 1); //TODO adapt the local size to better suit the gpu
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        TIME_END_GL("depth_update_kernel");


        //do updat depth on cpu--------------------------------------------------------

        // //get all the data as if it was the glsl code
        // glm::vec2 frame_size( frames[i].gray.cols, frames[i].gray.rows);
        // glm::mat4 tf_cur_host=glm::make_mat4x4(tf_cur_host_eigen.matrix().data());
        // glm::mat4 tf_host_cur=glm::make_mat4x4(tf_host_cur_eigen.matrix().data());
        // glm::mat3 K=glm::make_mat3x3(tf_host_cur_eigen.matrix().data());
        // glm::mat3 KRKi_cr=glm::make_mat3x3(KRKi_cr_eigen.data());
        // glm::vec3 Kt_cr=glm::make_vec3(Kt_cr_eigen.data());
        // glm::vec2 affine_cr=glm::make_vec2(affine_cr_eigen.data());
        // glm::vec2 pattern_rot_offsets[cl_MAX_RES_PER_POINT];
        // for (size_t p_idx = 0; p_idx < pattern_rot.get_nr_points(); p_idx++) {
        //     pattern_rot_offsets[p_idx]=glm::make_vec2(pattern_rot.get_offset(p_idx).data());
        // }
        // int pattern_rot_nr_points=pattern_rot.get_nr_points();
        // std::vector<Point>& p=immature_points;
        //
        // //same code as on gpu
        // for (size_t id = 0; id < p.size(); id++) {
        //     // check if point is visible in the current image
        //     const vec3 p_backproj_xyz= p[id].f.xyz() * 1.0f/p[id].mu;
        //     const vec4 p_backproj_xyzw=vec4(p_backproj_xyz,1.0);
        //     const vec4 xyz_f_xyzw = tf_cur_host*  p_backproj_xyzw ;
        //     const vec3 xyz_f=xyz_f_xyzw.xyz()/xyz_f_xyzw.w;
        //     if(xyz_f.z < 0.0)  {
        //         // p[id].debug=1.0;
        //         continue; //TODO in gl this is a return
        //     }
        //     const vec3 kp_c = K * xyz_f;
        //     const vec2 kp_c_h=kp_c.xy()/kp_c.z;
        //     if ( kp_c_h.x < 0 || kp_c_h.x >= frame_size.x || kp_c_h.y < 0 || kp_c_h.y >= frame_size.y ) {
        //         // p[id].debug=1.0;
        //         continue; //TODO in gl this is a return
        //     }
        //
        //
        //     //point is visible
        //
        //     //update inverse depth coordinates for min and max
        //     p[id].idepth_min = p[id].mu + sqrt(p[id].sigma2);
        //     p[id].idepth_max = max(p[id].mu - sqrt(p[id].sigma2), 0.00000001f);
        //     // memoryBarrier();
        //     // barrier();
        //
        //     //search epiline---------------------------------------------------------------
        //     // search_epiline_bca (point, frame, KRKi_cr, Kt_cr, affine_cr);
        //
        //
        //     float idepth_mean = (p[id].idepth_min + p[id].idepth_max)*0.5;
        //     vec3 pr = KRKi_cr * vec3(p[id].u,p[id].v, 1);
        //     vec3 ptpMean = pr + Kt_cr*idepth_mean;
        //     vec3 ptpMin = pr + Kt_cr*p[id].idepth_min;
        //     vec3 ptpMax = pr + Kt_cr*p[id].idepth_max;
        //     vec2 uvMean = ptpMean.xy()/ptpMean.z;
        //     vec2 uvMin = ptpMin.xy()/ptpMin.z;
        //     vec2 uvMax = ptpMax.xy()/ptpMax.z;
        //
        //     // Pattern pattern_rot=m_pattern.get_rotated_pattern( KRKi_cr.topLeftCorner<2,2>() );
        //
        //     vec2 epi_line = uvMax - uvMin;
        //     float norm_epi = max(1e-5f, length(epi_line));
        //     vec2 epi_dir = epi_line / norm_epi;
        //     const float  half_length = 0.5f * norm_epi;
        //
        //     vec2 bestKp;
        //     float bestEnergy = 1e15;
        //
        //     //debug stuff
        //     float residual_debug=0;
        //     int nr_time_switched_best=0;
        //
        //     for(float l = -half_length; l <= half_length; l += 0.7f){
        //         float energy = 0;
        //         residual_debug=0;
        //         vec2 kp = uvMean + l*epi_dir;
        //
        //         if( ( kp.x >= (frame_size.x-10) )  || ( kp.y >= (frame_size.y-10) ) || ( kp.x < 10 ) || ( kp.y < 10) ){
        //             continue;
        //         }
        //
        //         for(int idx=0;idx<pattern_rot_nr_points; ++idx) {
        //
        //             vec2 offset=pattern_rot_offsets[idx];
        //             // float hit_color=texture(gray_img_sampler, vec2( (kp.x + offset.x)/640, (kp.y + offset.y)/480)).x;
        //             float hit_color=texture_interpolate(frames[i].gray, kp.x+offset.x, kp.y+offset.y , InterpolType::LINEAR);
        //             // if(!std::isfinite(hit_color)) {energy-=1e5; continue;}
        //             //
        //             const float residual = hit_color - float(affine_cr.x * p[id].color[idx] + affine_cr.y);
        //             residual_debug+=residual;
        //
        //             float hw = abs(residual) < cl_setting_huberTH ? 1 : cl_setting_huberTH / abs(residual);
        //             energy += hw *residual*residual*(2-hw);
        //         }
        //
        //         // p[id].debug=energy;
        //         if ( energy < bestEnergy ){
        //             bestKp = kp; bestEnergy = energy;
        //             nr_time_switched_best++;
        //         }
        //     }
        //
        //     if(bestEnergy<1e10 ){
        //         p[id].debug=bestEnergy;
        //     }
        //
        //     // p[id].debug=nr_time_switched_best;
        //
        //
        //     if ( bestEnergy > p[id].energyTH * 1.2f ) {
        //         // point.lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
        //     }else{
        //         // float a = (Eigen::Vector2d(epi_dir(0),epi_dir(1)).transpose() * point.gradH * Eigen::Vector2d(epi_dir(0),epi_dir(1)));
        //         // float b = (Eigen::Vector2d(epi_dir(1),-epi_dir(0)).transpose() * point.gradH * Eigen::Vector2d(epi_dir(1),-epi_dir(0)));
        //         // float errorInPixel = 0.2f + 0.2f * (a+b) / a; // WO kommt das her? Scheint nicht zu NGF zu passen !
        //         float errorInPixel=0;
        //
        //         if( epi_dir.x*epi_dir.x>epi_dir.y*epi_dir.y )
        //         {
        //             p[id].idepth_min = (pr.z*(bestKp.x-errorInPixel*epi_dir.x) - pr.x) / (Kt_cr.x - Kt_cr.z*(bestKp.x-errorInPixel*epi_dir.x));
        //             p[id].idepth_max = (pr.z*(bestKp.x+errorInPixel*epi_dir.x) - pr.x) / (Kt_cr.x - Kt_cr.z*(bestKp.x+errorInPixel*epi_dir.x));
        //         }else{
        //             p[id].idepth_min = (pr.z*(bestKp.y-errorInPixel*epi_dir.y) - pr.y) / (Kt_cr.y - Kt_cr.z*(bestKp.y-errorInPixel*epi_dir.y));
        //             p[id].idepth_max = (pr.z*(bestKp.y+errorInPixel*epi_dir.y) - pr.y) / (Kt_cr.y - Kt_cr.z*(bestKp.y+errorInPixel*epi_dir.y));
        //         }
        //         // memoryBarrier();
        //         // barrier();
        //         if(p[id].idepth_min > p[id].idepth_max) {
        //             // std::swap<float>(point.idepth_min, point.idepth_max);
        //             float tmp=p[id].idepth_min;
        //             p[id].idepth_min=p[id].idepth_max;
        //             p[id].idepth_max=tmp;
        //         }
        //         // memoryBarrier();
        //         // barrier();
        //
        //         // point.lastTraceStatus = ImmaturePointStatus::IPS_GOOD;
        //     }
        //     //
        //     //
        //     //
        //     //
        //     // double idepth = -1;
        //     // double z = 0;
        //     // if( point.lastTraceStatus == ImmaturePointStatus::IPS_GOOD ) {
        //     //     idepth = std::max<double>(1e-5,.5*(point.idepth_min+point.idepth_max));
        //     //     z = 1./idepth;
        //     // }
        //     // if ( point.lastTraceStatus == ImmaturePointStatus::IPS_OOB  || point.lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED ){
        //     //     continue;
        //     // }
        //     // if ( !std::isfinite(idepth) || point.lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER || point.lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION ) {
        //     //     point.b++; // increase outlier probability when no match was found
        //     //     continue;
        //     // }
        //     //
        //     //
        //     // update_idepth(point,tf_host_cur, z, px_error_angle);
        //
        //
        //
        //     float idepth = -1;
        //     float z = 0;
        //     idepth = max(1e-5,.5*(p[id].idepth_min+p[id].idepth_max));
        //     z = 1./idepth;
        //
        //
        //
        //
        //     // compute tau-------------------------------------------------------------------------
        //     // double tau = compute_tau(tf_host_cur, point.f, z, px_error_angle);
        //     vec3 t= vec3(tf_host_cur[0][3], tf_host_cur[1][3], tf_host_cur[2][3]);
        //     vec3 a = p[id].f.xyz()*z-t;
        //     float t_norm = length(t);
        //     float a_norm = length(a);
        //     float alpha = acos(dot(p[id].f.xyz(),t)/t_norm); // dot product
        //     float beta = acos(dot(a,-t)/(t_norm*a_norm)); // dot product
        //     float beta_plus = beta + px_error_angle;
        //     float gamma_plus = 3.1415-alpha-beta_plus; // triangle angles sum to PI
        //     float z_plus = t_norm*sin(beta_plus)/sin(gamma_plus); // law of sines
        //     float tau= (z_plus - z); // tau
        //
        //
        //
        //     float tau_inverse = 0.5f * (1.0f/max(0.0000001f, z-tau) - 1.0f/(z+tau));
        //
        //     // update the estimate------------------------------------------------------------------
        //     // updateSeed(point, 1.0/z, tau_inverse*tau_inverse);
        //
        //     float x= 1.0/z;
        //     float tau2=tau_inverse*tau_inverse;
        //     float norm_scale = sqrt(p[id].sigma2 + tau2);
        //     // if(std::isnan(norm_scale))
        //     //     return;
        //     float s2 = 1./(1./p[id].sigma2 + 1./tau2);
        //     float m = s2*(p[id].mu/p[id].sigma2 + x/tau2);
        //     float C1 = p[id].a/(p[id].a+p[id].b) *  gaus_pdf(p[id].mu, norm_scale, x);
        //     float C2 = p[id].b/(p[id].a+p[id].b) * 1.0/p[id].z_range;
        //     float normalization_constant = C1 + C2;
        //     C1 /= normalization_constant;
        //     C2 /= normalization_constant;
        //     float f = C1*(p[id].a+1.)/(p[id].a+p[id].b+1.) + C2*p[id].a/(p[id].a+p[id].b+1.);
        //     float e = C1*(p[id].a+1.)*(p[id].a+2.)/((p[id].a+p[id].b+1.)*(p[id].a+p[id].b+2.))
        //               + C2*p[id].a*(p[id].a+1.0f)/((p[id].a+p[id].b+1.0f)*(p[id].a+p[id].b+2.0f));
        //
        //     // update parameters
        //     float mu_new = C1*m+C2*p[id].mu;
        //     p[id].sigma2 = C1*(s2 + m*m) + C2*(p[id].sigma2 + p[id].mu*p[id].mu) - mu_new*mu_new;
        //     p[id].mu = mu_new;
        //     p[id].a = (e-f)/(f-e/f);
        //     p[id].b = p[id].a*(1.0f-f)/f;
        //     // memoryBarrier();
        //     // barrier();
        // }





        // //cpu but with eigen
        //
        // //make a ll matrices either 4x4 or 3x3 as required by gl
        // Eigen::Matrix4f tf_cur_host_4x4= tf_cur_host_eigen.matrix();
        // Eigen::Matrix4f tf_host_cur_4x4= tf_host_cur_eigen.matrix();
        //
        //
        //
        // // update_immature_points(immature_points, frames[i], tf_cur_host, KRKi_cr, Kt_cr, affine_cr );
        //
        // //done in paralel for all points in the case of opengl
        // for (auto &point : immature_points){
        //
        //     // // check if point is visible in the current image
        //     const Eigen::Vector3f p_backproj_xyz= point.f.head<3>() * 1.0/point.mu;
        //     const Eigen::Vector4f p_backproj_xyzw=Eigen::Vector4f(p_backproj_xyz(0),p_backproj_xyz(1),p_backproj_xyz(2),1.0);
        //     const Eigen::Vector4f xyz_f_xyzw = tf_cur_host_4x4*  p_backproj_xyzw ;
        //     const Eigen::Vector3f xyz_f=xyz_f_xyzw.head<3>()/xyz_f_xyzw.w();
        //     if(xyz_f.z() < 0.0)  {
        //         continue; // TODO in gl this is a return
        //     }
        //
        //
        //     // const Eigen::Vector3f xyz_f( tf_cur_host_4x4*(1.0/point.mu * point.f.head<3>()) );
        //     // if(xyz_f.z() < 0.0)  {
        //     //     continue;
        //     // }
        //     const Eigen::Vector2f kp_c = (frames[i].K * xyz_f).hnormalized();
        //     if ( kp_c(0) < 0 || kp_c(0) >= frames[i].gray.cols || kp_c(1) < 0 || kp_c(1) >= frames[i].gray.rows ) {
        //         continue;
        //     }
        //
        //
        //     //point is visible
        //     point.last_visible_frame=frames[i].frame_id;
        //
        //     //update inverse depth coordinates for min and max
        //     point.idepth_min = point.mu + sqrt(point.sigma2);
        //     point.idepth_max = std::max<float>(point.mu - sqrt(point.sigma2), 0.00000001f);
        //
        //     //search epiline-----------------------------------------------------------------------
        //    // search_epiline_ncc (point, frame, KRKi_cr, Kt_cr );
        //     // search_epiline_bca (point, frames[i], KRKi_cr, Kt_cr, affine_cr);
        //     float idepth_mean = (point.idepth_min + point.idepth_max)*0.5;
        //     Eigen::Vector3f pr = KRKi_cr_eigen * Eigen::Vector3f(point.u,point.v, 1);
        //     Eigen::Vector3f ptpMean = pr + Kt_cr_eigen*idepth_mean;
        //     Eigen::Vector3f ptpMin = pr + Kt_cr_eigen*point.idepth_min;
        //     Eigen::Vector3f ptpMax = pr + Kt_cr_eigen*point.idepth_max;
        //     Eigen::Vector2f uvMean = ptpMean.hnormalized();
        //     Eigen::Vector2f uvMin = ptpMin.hnormalized();
        //     Eigen::Vector2f uvMax = ptpMax.hnormalized();
        //
        //     // Pattern pattern_rot=m_pattern.get_rotated_pattern( KRKi_cr.topLeftCorner<2,2>() );
        //
        //     Eigen::Vector2f epi_line = uvMax - uvMin;
        //     float norm_epi = std::max<float>(1e-5f,epi_line.norm());
        //     Eigen::Vector2f epi_dir = epi_line / norm_epi;
        //     const float  half_length = 0.5f * norm_epi;
        //
        //     Eigen::Vector2f bestKp;
        //     float bestEnergy = 1e10;
        //
        //     for(float l = -half_length; l <= half_length; l += 0.7f)
        //     {
        //         float energy = 0;
        //         Eigen::Vector2f kp = uvMean + l*epi_dir;
        //
        //         if( !kp.allFinite() || ( kp(0) >= (frames[i].gray.cols-10) )  || ( kp(1) >= (frames[i].gray.rows-10) ) || ( kp(0) < 10 ) || ( kp(1) < 10) )
        //         {
        //             continue;
        //         }
        //
        //         for(int idx=0;idx<pattern_rot.get_nr_points(); ++idx)
        //         {
        //             //float hitColor = getInterpolatedElement31(frame->dI, (float)(kp(0)+rotatetPattern[idx][0]), (float)(kp(1)+rotatetPattern[idx][1]), wG[0]);
        //             Eigen::Vector2f offset=pattern_rot.get_offset(idx);
        //             float hit_color=texture_interpolate(frames[i].gray, kp(0)+offset(0), kp(1)+offset(1) , InterpolType::LINEAR);
        //             if(!std::isfinite(hit_color)) {energy-=1e5; continue;}
        //
        //             const float residual = hit_color - (float)(affine_cr_eigen[0] * point.color[idx] + affine_cr_eigen[1]);
        //
        //             float hw = fabs(residual) < cl_setting_huberTH ? 1 : cl_setting_huberTH / fabs(residual);
        //             energy += hw *residual*residual*(2-hw);
        //         }
        //         if ( energy < bestEnergy )
        //         {
        //             bestKp = kp; bestEnergy = energy;
        //         }
        //     }
        //
        //     if ( bestEnergy > point.energyTH * 1.2f ) {
        //         // point.lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
        //     }
        //     else
        //     {
        //         // float a = (Eigen::Vector2f(epi_dir(0),epi_dir(1)).transpose() * point.gradH * Eigen::Vector2f(epi_dir(0),epi_dir(1)));
        //         // float b = (Eigen::Vector2f(epi_dir(1),-epi_dir(0)).transpose() * point.gradH * Eigen::Vector2f(epi_dir(1),-epi_dir(0)));
        //         // float errorInPixel = 0.2f + 0.2f * (a+b) / a; // WO kommt das her? Scheint nicht zu NGF zu passen !
        //         float errorInPixel=0.0;
        //
        //         if( epi_dir(0)*epi_dir(0)>epi_dir(1)*epi_dir(1) )
        //         {
        //             point.idepth_min = (pr[2]*(bestKp(0)-errorInPixel*epi_dir(0)) - pr[0]) / (Kt_cr_eigen[0] - Kt_cr_eigen[2]*(bestKp(0)-errorInPixel*epi_dir(0)));
        //             point.idepth_max = (pr[2]*(bestKp(0)+errorInPixel*epi_dir(0)) - pr[0]) / (Kt_cr_eigen[0] - Kt_cr_eigen[2]*(bestKp(0)+errorInPixel*epi_dir(0)));
        //         }
        //         else
        //         {
        //             point.idepth_min = (pr[2]*(bestKp(1)-errorInPixel*epi_dir(1)) - pr[1]) / (Kt_cr_eigen[1] - Kt_cr_eigen[2]*(bestKp(1)-errorInPixel*epi_dir(1)));
        //             point.idepth_max = (pr[2]*(bestKp(1)+errorInPixel*epi_dir(1)) - pr[1]) / (Kt_cr_eigen[1] - Kt_cr_eigen[2]*(bestKp(1)+errorInPixel*epi_dir(1)));
        //         }
        //         if(point.idepth_min > point.idepth_max) std::swap<float>(point.idepth_min, point.idepth_max);
        //
        //         // point.lastTraceStatus = ImmaturePointStatus::IPS_GOOD;
        //     }
        //
        //
        //
        //
        //
        //
        //
        //     double idepth = -1;
        //     double z = 0;
        //     idepth = std::max<double>(1e-5,.5*(point.idepth_min+point.idepth_max));
        //     z = 1./idepth;
        //     // if( point.lastTraceStatus == ImmaturePointStatus::IPS_GOOD ) {
        //     //
        //     // }
        //     // if ( point.lastTraceStatus == ImmaturePointStatus::IPS_OOB  || point.lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED ){
        //     //     continue;
        //     // }
        //     // if ( !std::isfinite(idepth) || point.lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER || point.lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION ) {
        //     //     point.b++; // increase outlier probability when no match was found
        //     //     continue;
        //     // }
        //
        //
        //     // update_idepth(point,tf_host_cur, z, px_error_angle);
        //
        //     // compute tau----------------------------------------------------------------------------
        //     // double tau = compute_tau(tf_host_cur, point.f, z, px_error_angle);
        //     Eigen::Vector3f t=  Eigen::Vector3f(tf_host_cur_4x4(0,3), tf_host_cur_4x4(1,3), tf_host_cur_4x4(2,3));
        //     // Eigen::Vector3f t(tf_host_cur.translation());
        //     Eigen::Vector3f a = point.f.head<3>()*z-t;
        //     double t_norm = t.norm();
        //     double a_norm = a.norm();
        //     double alpha = acos(point.f.head<3>().dot(t)/t_norm); // dot product
        //     double beta = acos(a.dot(-t)/(t_norm*a_norm)); // dot product
        //     double beta_plus = beta + px_error_angle;
        //     double gamma_plus = M_PI-alpha-beta_plus; // triangle angles sum to PI
        //     double z_plus = t_norm*sin(beta_plus)/sin(gamma_plus); // law of sines
        //     double tau= (z_plus - z); // tau
        //     double tau_inverse = 0.5 * (1.0/std::max<double>(0.0000001, z-tau) - 1.0/(z+tau));
        //
        //     // update the estimate--------------------------------------------------
        //     float x=1.0/z;
        //     float tau2=tau_inverse*tau_inverse;
        //     // updateSeed(point, 1.0/z, tau_inverse*tau_inverse);
        //     float norm_scale = sqrt(point.sigma2 + tau2);
        //     if(std::isnan(norm_scale))
        //         continue;
        //     float s2 = 1./(1./point.sigma2 + 1./tau2);
        //     float m = s2*(point.mu/point.sigma2 + x/tau2);
        //     float C1 = point.a/(point.a+point.b) * gaus_pdf(point.mu, norm_scale, x);
        //     float C2 = point.b/(point.a+point.b) * 1./point.z_range;
        //     float normalization_constant = C1 + C2;
        //     C1 /= normalization_constant;
        //     C2 /= normalization_constant;
        //     float f = C1*(point.a+1.)/(point.a+point.b+1.) + C2*point.a/(point.a+point.b+1.);
        //     float e = C1*(point.a+1.)*(point.a+2.)/((point.a+point.b+1.)*(point.a+point.b+2.))
        //               + C2*point.a*(point.a+1.0f)/((point.a+point.b+1.0f)*(point.a+point.b+2.0f));
        //     // update parameters
        //     float mu_new = C1*m+C2*point.mu;
        //     point.sigma2 = C1*(s2 + m*m) + C2*(point.sigma2 + point.mu*point.mu) - mu_new*mu_new;
        //     point.mu = mu_new;
        //     point.a = (e-f)/(f-e/f);
        //     point.b = point.a*(1.0f-f)/f;
        //
        //
        //     //not implemented in opengl
        //     // const float eta_inlier = .6f;
        //     // const float eta_outlier = .05f;
        //     // if( ((point.a / (point.a + point.b)) > eta_inlier) && (sqrt(point.sigma2) < point.z_range/seed_convergence_sigma2_thresh)) {
        //     //     point.is_outlier = false; // The seed converged
        //     // }else if((point.a-1) / (point.a + point.b - 2) < eta_outlier){ // The seed failed to converge
        //     //     point.is_outlier = true;
        //     //     // it->reinit();
        //     //     //TODO do a better reinit inside a point class
        //     //     point.a = 10;
        //     //     point.b = 10;
        //     //     point.mu = (1.0/4.0);
        //     //     point.z_range = (1.0/0.1);
        //     //     point.sigma2 = (point.z_range*point.z_range/36);
        //     // }
        //     // // if the seed has converged, we initialize a new candidate point and remove the seed
        //     // if(sqrt(point.sigma2) < point.z_range/seed_convergence_sigma2_thresh){
        //     //     point.converged = true;
        //     // }
        //
        //
        //
        // }





        TIME_END_GL("update_depth");
    }



    //read the points back to cpu
    //TODO
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_points_gl_buf);
    Point* ptr = (Point*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    for (size_t i = 0; i < immature_points.size(); i++) {
        immature_points[i]=ptr[i];
    }



    TIME_END_GL("compute_depth");


    mesh=create_mesh(immature_points, frames);
    return mesh;

}

std::vector<Frame> DepthEstimatorGL::loadDataFromICLNUIM ( const std::string & dataset_path, const int num_images_to_read){
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

         cv::Mat rgb_cv=cv::imread(dataset_path + "/" + colorFileName, CV_LOAD_IMAGE_UNCHANGED);
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

std::vector<Point> DepthEstimatorGL::create_immature_points (const Frame& frame){


    //make the sobel in x and y because we need to to calculate the hessian in order to select the immature point
    TIME_START_GL("sobel_host_frame");
    cv::Mat grad_x, grad_y;
    cv::Scharr( frame.gray, grad_x, CV_32F, 1, 0);
    cv::Scharr( frame.gray, grad_y, CV_32F, 0, 1);
    TIME_END_GL("sobel_host_frame");

    TIME_START_GL("hessian_host_frame");
    std::vector<Point> immature_points;
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
            // if(hessian_det > 0){
                Point point;
                point.u=j;
                point.v=i;
                point.gradH=glm::make_mat2x2(gradient_hessian.data());

                //Seed::Seed
                Eigen::Vector3f f_eigen = (frame.K.inverse() * Eigen::Vector3f(point.u,point.v,1)).normalized();
                point.f = glm::vec4(f_eigen(0),f_eigen(1),f_eigen(2), 1.0);

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
                point.converged=0;
                point.is_outlier=1;


                //immature point constructor (the idepth min and max are already set so don't worry about those)
                point.lastTraceStatus=STATUS_UNINITIALIZED;

                //get data for the color of that point (depth_point->ImmaturePoint::ImmaturePoint)---------------------
                for(int p_idx=0;p_idx<m_pattern.get_nr_points();p_idx++){
                    Eigen::Vector2f offset = m_pattern.get_offset(p_idx);

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
                point.debug=0.0;

                immature_points.push_back(point);
            }

        }
    }
    TIME_END_GL("hessian_host_frame");




    return immature_points;
}

Eigen::Vector2f DepthEstimatorGL::estimate_affine(std::vector<Point>& immature_points, const Frame&  cur_frame, const Eigen::Matrix3f& KRKi_cr, const Eigen::Vector3f& Kt_cr){
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

            const Eigen::Vector3f p = KRKi_cr * Eigen::Vector3f(point.u,point.v,1) + Kt_cr*  (1.0/point.gt_depth);
            Eigen::Vector2f kp_GT = p.hnormalized();


            if ( kp_GT(0) > 4 && kp_GT(0) < cur_frame.gray.cols-4 && kp_GT(1) > 3 && kp_GT(1) < cur_frame.gray.rows-4 ) {

                Pattern pattern_rot=m_pattern.get_rotated_pattern( KRKi_cr.topLeftCorner<2,2>() );

                for(int idx=0;idx<m_pattern.get_nr_points();++idx) {
                    Eigen::Vector2f offset=pattern_rot.get_offset(idx);

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
            ceres::CostFunction * cost_function = AffineAutoDiffCostFunctorGL::Create( color_cur_frame[i], color_host_frame[i] );
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
    return Eigen::Vector2f ( scaleA, offsetB );
}

float DepthEstimatorGL::texture_interpolate ( const cv::Mat& img, const float x, const float y , const InterpolType type){
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

Mesh DepthEstimatorGL::create_mesh(const std::vector<Point>& immature_points, const std::vector<Frame>& frames){
    Mesh mesh;
    mesh.V.resize(immature_points.size(),3);
    mesh.V.setZero();

    //debug check the first few vrices
    for (size_t i = 0; i < 10; i++) {
        std::cout << "point " << i << " at uv " << immature_points[i].u << " " << immature_points[i].v  << '\n';
    }

    for (size_t i = 0; i < immature_points.size(); i++) {
        float u=immature_points[i].u;
        float v=immature_points[i].v;
        // float depth=immature_points[i].gt_depth;
        // float depth=1.0;
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
   //     // std::cout << "last_visible_frame is " << immature_points[i].last_visible_frame << '\n';
   //     if(immature_points[i].last_visible_frame<min){
   //         min=immature_points[i].last_visible_frame;
   //     }
   //     if(immature_points[i].last_visible_frame>max){
   //         max=immature_points[i].last_visible_frame;
   //     }
   // }
   // std::cout << "min max last_visible_frame is " << min << " " << max << '\n';
   // for (size_t i = 0; i < mesh.C.rows(); i++) {
   //      float gray_val = lerp(immature_points[i].last_visible_frame, min, max, 0.0, 1.0 );
   //      mesh.C(i,0)=mesh.C(i,1)=mesh.C(i,2)=gray_val;
   //  }

  //  //colors based on debug colors
  // float min=9999999999, max=-9999999999;
  // for (size_t i = 0; i < immature_points.size(); i++) {
  //     // std::cout << "last_visible_frame is " << immature_points[i].last_visible_frame << '\n';
  //     if(immature_points[i].debug<min){
  //         min=immature_points[i].debug;
  //     }
  //     if(immature_points[i].debug>max){
  //         max=immature_points[i].debug;
  //     }
  // }
  // std::cout << "min max debug is " << min << " " << max << '\n';
  // for (size_t i = 0; i < mesh.C.rows(); i++) {
  //      float gray_val = lerp(immature_points[i].debug, min, max, 0.0, 1.0 );
  //      mesh.C(i,0)=mesh.C(i,1)=mesh.C(i,2)=gray_val;
  //  }






    return mesh;
}
