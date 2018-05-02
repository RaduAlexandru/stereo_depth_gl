#include "stereo_depth_cl/DepthEstimatorGL.h"

//c++
#include <cmath>
#include <stdlib.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <sstream>

//My stuff
#include "stereo_depth_cl/Profiler.h"
#include "stereo_depth_cl/MiscUtils.h"
#include "cv_interpolation.h"
#include "UtilsGL.h"
#include "Shader.h"
#include "Texture2DArray.h"
#include "Texture2D.h"

//Libigl
#include <igl/opengl/glfw/Viewer.h>

//cv
#include <cv_bridge/cv_bridge.h>

//loguru
#include <loguru.hpp>

//gl
#include <glm/gtc/type_ptr.hpp>

// using namespace glm;

DepthEstimatorGL::DepthEstimatorGL():
        m_scene_is_modified(false),
        m_gl_profiling_enabled(true),
        m_show_images(false),
        m_use_rgbd_tum(true),
        m_start_frame(0),
        m_mean_starting_depth(4.0)
        {

    init_opengl();
    std::string pattern_filepath="/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/data/pattern_1.png";
    m_pattern.init_pattern(pattern_filepath);

    // //sanity check the pattern
    // std::cout << "pattern has nr of points " << m_pattern.get_nr_points() << '\n';
    // for (size_t i = 0; i < m_pattern.get_nr_points(); i++) {
    //     std::cout << "offset for i " << i << " is " << m_pattern.get_offset(i).transpose() << '\n';
    // }

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
    glGenBuffers(1, &m_ubo_params );


    m_cur_frame.set_wrap_mode(GL_CLAMP_TO_BORDER);
    m_cur_frame.set_filter_mode(GL_LINEAR);

    compile_shaders();

}

void DepthEstimatorGL::compile_shaders(){

    m_update_depth_prog_id=gl::program_init_from_files("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/shaders/compute_update_depth.glsl");

    m_denoise_depth_prog_id=gl::program_init_from_files("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/shaders/update_TVL1_primal_dual.glsl");

    m_copy_to_texture_prog_id=gl::program_init_from_files("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/shaders/compute_copy_to_texture.glsl");

    m_denoise_texture_prog_id=gl::program_init_from_files("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/shaders/update_TVL1_primal_dual_texture.glsl");

    m_copy_from_texture_prog_id=gl::program_init_from_files("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/shaders/copy_from_texture.glsl");

    m_copy_to_texture_fbo_prog_id=gl::program_init_from_files("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/shaders/compute_copy_to_texture_fbo.glsl");

    m_copy_from_texture_fbo_prog_id=gl::program_init_from_files("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/shaders/copy_from_texture_fbo.glsl");

    m_denoise_fbo_prog_id=gl::program_init_from_files("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/shaders/update_TVL1_primal_dual_fbo_vert.glsl", "/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/shaders/update_TVL1_primal_dual_fbo_frag.glsl");

}

//https://www.boost.org/doc/libs/1_47_0/libs/math/doc/sf_and_dist/html/math_toolkit/dist/dist_ref/dists/normal_dist.html
float DepthEstimatorGL::gaus_pdf(float mean, float sd, float x){
    return exp(- (x-mean)*(x-mean)/(2*sd)*(2*sd)  )  / (sd*sqrt(2*M_PI));
}

void DepthEstimatorGL::init_data(){
    //----------------------------------------------------------------------------------------------------
    int num_images_to_read=100;
    bool use_modified=false;
    m_frames.clear();
    if(m_use_rgbd_tum){
        // m_frames=loadDataFromRGBD_TUM("/media/alex/Data/Master/Thesis/data/RGBD_TUM/rgbd_dataset_freiburg1_xyz", num_images_to_read);
        m_frames=loadDataFromRGBD_TUM("/media/alex/Data/Master/Thesis/data/RGBD_TUM/rgbd_dataset_freiburg3_long_office_household", num_images_to_read);
    }else{
        m_frames=loadDataFromICLNUIM("/media/alex/Data/Master/Thesis/data/ICL_NUIM/living_room_traj2_frei_png", num_images_to_read);
    }

    m_frame_size << m_frames[0].gray.cols, m_frames[0].gray.rows;


    std::cout << "frames size is " << m_frames.size() << "\n";
}

void DepthEstimatorGL::compute_depth_and_create_mesh(){
    m_mesh.clear();


    TIME_START_GL("compute_depth");


    std::vector<Point> immature_points;
    immature_points=create_immature_points(m_frames[0]);
    std::cout << "immature_points size is " << immature_points.size() << '\n';


    //upload to gpu the inmature points
    TIME_START_GL("upload_immature_points");
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_points_gl_buf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, immature_points.size() * sizeof(Point), immature_points.data(), GL_DYNAMIC_COPY);
    TIME_END_GL("upload_immature_points");

    glUseProgram(m_update_depth_prog_id);
    for (size_t i = 1; i < m_frames.size(); i++) {
        std::cout << "frame " << i << '\n';
        TIME_START_GL("update_depth");

        TIME_START_GL("estimate_affine");
        const Eigen::Affine3f tf_cur_host_eigen = m_frames[i].tf_cam_world * m_frames[0].tf_cam_world.inverse();
        const Eigen::Affine3f tf_host_cur_eigen = tf_cur_host_eigen.inverse();
        const Eigen::Matrix3f KRKi_cr_eigen = m_frames[i].K * tf_cur_host_eigen.linear() * m_frames[0].K.inverse();
        const Eigen::Vector3f Kt_cr_eigen = m_frames[i].K * tf_cur_host_eigen.translation();
        // const Eigen::Vector2f affine_cr_eigen = estimate_affine( immature_points, frames[i], KRKi_cr_eigen, Kt_cr_eigen);
        const Eigen::Vector2f affine_cr_eigen= Eigen::Vector2f(1,1);
        const double focal_length = fabs(m_frames[i].K(0,0));
        double px_noise = 1.0;
        double px_error_angle = atan(px_noise/(2.0*focal_length))*2.0; // law of chord (sehnensatz)
        Pattern pattern_rot=m_pattern.get_rotated_pattern( KRKi_cr_eigen.topLeftCorner<2,2>() );

        // std::cout << "pattern_rot has nr of points " << pattern_rot.get_nr_points() << '\n';
        // for (size_t i = 0; i < pattern_rot.get_nr_points(); i++) {
        //     std::cout << "offset for i " << i << " is " << pattern_rot.get_offset(i).transpose() << '\n';
        // }

        TIME_END_GL("estimate_affine");

        TIME_START_GL("upload_params");
        //upload params (https://hub.packtpub.com/opengl-40-using-uniform-blocks-and-uniform-buffer-objects/)
        glBindBuffer( GL_UNIFORM_BUFFER, m_ubo_params );
        glBufferData( GL_UNIFORM_BUFFER, sizeof(m_params), &m_params, GL_DYNAMIC_DRAW );
        glBindBufferBase( GL_UNIFORM_BUFFER,  glGetUniformBlockIndex(m_update_depth_prog_id,"params_block"), m_ubo_params );
        TIME_END_GL("upload_params");

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

        // //attempt 3 upload the image as a inmutable storage
        // TIME_START_GL("upload_gray_img");
        // int size_bytes=frames[i].gray.step[0] * frames[i].gray.rows;
        // if(!m_cur_frame.get_tex_storage_initialized()){
        //     m_cur_frame.allocate_tex_storage_inmutable(GL_R32F,frames[i].gray.cols, frames[i].gray.rows);
        // }
        // m_cur_frame.upload_without_pbo(0,0,0, frames[i].gray.cols, frames[i].gray.rows, GL_RED, GL_FLOAT, frames[i].gray.ptr());
        // TIME_END_GL("upload_gray_img");


        //attempt 3 upload the image as a inmutable storage but also pack the gradient into the 2nd and 3rd channel
        TIME_START_GL("upload_gray_img");
        //merge all mats into one with 4 channel
        std::vector<cv::Mat> channels;
        channels.push_back(m_frames[i].gray);
        channels.push_back(m_frames[i].grad_x);
        channels.push_back(m_frames[i].grad_y);
        channels.push_back(m_frames[i].grad_y); //dummy one stored in the alpha channels just to have a 4 channel texture
        cv::Mat img_with_gradients;
        cv::merge(channels, img_with_gradients);

        int size_bytes=img_with_gradients.step[0] * img_with_gradients.rows; //allocate 4 channels because gpu likes multiples of 4
        if(!m_cur_frame.get_tex_storage_initialized()){
            m_cur_frame.allocate_tex_storage_inmutable(GL_RGBA32F,img_with_gradients.cols, img_with_gradients.rows);
        }
        m_cur_frame.upload_without_pbo(0,0,0, img_with_gradients.cols, img_with_gradients.rows, GL_RGBA, GL_FLOAT, img_with_gradients.ptr());
        TIME_END_GL("upload_gray_img");





        //upload the matrices
        TIME_START_GL("upload_matrices");
        Eigen::Vector2f frame_size;
        frame_size<< m_frames[i].gray.cols, m_frames[i].gray.rows;
        const Eigen::Matrix4f tf_cur_host_eigen_trans = tf_cur_host_eigen.matrix().transpose();
        const Eigen::Matrix4f tf_host_cur_eigen_trans = tf_host_cur_eigen.matrix().transpose();
        glUniform2fv(glGetUniformLocation(m_update_depth_prog_id,"frame_size"), 1, frame_size.data());
        // glUniformMatrix4fv(glGetUniformLocation(m_update_depth_prog_id,"tf_cur_host"), 1, GL_FALSE, tf_cur_host_eigen_trans.data());
        // glUniformMatrix4fv(glGetUniformLocation(m_update_depth_prog_id,"tf_host_cur"), 1, GL_FALSE, tf_host_cur_eigen_trans.data());
        glUniformMatrix4fv(glGetUniformLocation(m_update_depth_prog_id,"tf_cur_host"), 1, GL_TRUE, tf_cur_host_eigen.data());
        glUniformMatrix4fv(glGetUniformLocation(m_update_depth_prog_id,"tf_host_cur"), 1, GL_TRUE, tf_host_cur_eigen.data());
        glUniformMatrix3fv(glGetUniformLocation(m_update_depth_prog_id,"K"), 1, GL_FALSE, m_frames[i].K.data());
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

        TIME_END_GL("update_depth");
    }



    //read the points back to cpu
    //TODO
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_points_gl_buf);
    Point* ptr = (Point*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    for (size_t i = 0; i < immature_points.size(); i++) {
        immature_points[i]=ptr[i];
    }
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    assign_neighbours_for_points(immature_points, m_frames[0].gray.cols, m_frames[0].gray.rows);
    // denoise_cpu(immature_points, m_frames[0].gray.cols, m_frames[0].gray.rows);


    //GPU---------------------------------------------------------------------------------------------------------
    // denoise_cpu(immature_points, m_frames[0].gray.cols, m_frames[0].gray.rows);
    // denoise_gpu_vector(immature_points);
    denoise_gpu_texture(immature_points, m_frames[0].gray.cols, m_frames[0].gray.rows);
    // denoise_gpu_framebuffer(immature_points, m_frames[0].gray.cols, m_frames[0].gray.rows);


    TIME_END_GL("compute_depth");


    m_mesh=create_mesh(immature_points, m_frames);
    m_points=immature_points; //save the points in the class in case we need them for later saving to a file

    m_scene_is_modified=true;

}

//grabs a vector of Immature points and does a denoising on the gpu
void DepthEstimatorGL::denoise_gpu_vector(std::vector<Point>& immature_points){
    glUseProgram(m_denoise_depth_prog_id);


    int depth_range=m_params.denoise_depth_range;
    float lambda=m_params.denoise_lambda;
    int iterations=m_params.denoise_nr_iterations;
    const float large_sigma2 = depth_range * depth_range / 72.f;
    // computeWeightsAndMu( )
    for ( auto &point : immature_points){
        const float E_pi = point.a / ( point.a + point.b);
        point.g = std::max<float> ( (E_pi * point.sigma2 + (1.0f-E_pi) * large_sigma2) / large_sigma2, 1.0f );
        point.mu_denoised = point.mu;
        point.mu_head = point.u;
        point.p.setZero();
    }

    //upload to gpu the inmature points
    TIME_START_GL("upload_immature_points");
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_points_gl_buf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, immature_points.size() * sizeof(Point), immature_points.data(), GL_DYNAMIC_COPY);
    TIME_END_GL("upload_immature_points");

    TIME_START_GL("upload_params");
    //upload params (https://hub.packtpub.com/opengl-40-using-uniform-blocks-and-uniform-buffer-objects/)
    glBindBuffer( GL_UNIFORM_BUFFER, m_ubo_params );
    glBufferData( GL_UNIFORM_BUFFER, sizeof(m_params), &m_params, GL_DYNAMIC_DRAW );
    glBindBufferBase( GL_UNIFORM_BUFFER,  glGetUniformBlockIndex(m_denoise_depth_prog_id,"params_block"), m_ubo_params );
    TIME_END_GL("upload_params");


    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_points_gl_buf);
    TIME_START_GL("depth_denoise_kernel");
    std::cout << "running for "<< m_params.denoise_nr_iterations << '\n';
    for (size_t i = 0; i < m_params.denoise_nr_iterations; i++) {
        glDispatchCompute(immature_points.size()/256, 1, 1); //TODO adapt the local size to better suit the gpu
        // glMemoryBarrier(GL_ALL_BARRIER_BITS);
    }
    TIME_END_GL("depth_denoise_kernel");

    //Read to cpu AND APPLY THE DENOISED VALUES
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_points_gl_buf);
    Point* ptr = (Point*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    for (size_t i = 0; i < immature_points.size(); i++) {
        immature_points[i]=ptr[i];
    }
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    for (auto &point : immature_points) {
        // std::cout << "changin mu depth from " << point.mu  << " to " << point.mu_denoised << '\n';
        point.mu=point.mu_denoised;
    }

}

void DepthEstimatorGL::denoise_gpu_texture(std::vector<Point>& immature_points,  const int frame_width, const int frame_height){

    TIME_START_GL("denoise_gpu_texture");

    //set some stuff on the cpu that is needed for denoising
    int depth_range=m_params.denoise_depth_range;
    float lambda=m_params.denoise_lambda;
    int iterations=m_params.denoise_nr_iterations;
    const float large_sigma2 = depth_range * depth_range / 72.f;
    // computeWeightsAndMu( )
    for ( auto &point : immature_points){
        const float E_pi = point.a / ( point.a + point.b);
        point.g = std::max<float> ( (E_pi * point.sigma2 + (1.0f-E_pi) * large_sigma2) / large_sigma2, 1.0f );
        point.mu_denoised = point.mu;
        point.mu_head = point.u;
        point.p.setZero();
    }

    //make a 2d texture array of 6 channels (because we need 6 members for each point)
    gl::Texture2DArray tex;
    tex.set_wrap_mode(GL_CLAMP_TO_EDGE);
    tex.set_filter_mode(GL_NEAREST);
    tex.allocate_tex_storage_inmutable(GL_R32F, frame_width, frame_height, 6); // 6 channels (because we need 6 members for each immature point)

    //clear the texture to -1 so that we know it's uninitilzied
    const float clear=-1;
    glClearTexImage(tex.get_tex_id(), 0, GL_RED, GL_FLOAT, &clear);

    //copy the immature points to the texture using a compute shader
    glUseProgram(m_copy_to_texture_prog_id);

    //upload to gpu the inmature points
    TIME_START_GL("upload_immature_points");
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_points_gl_buf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, immature_points.size() * sizeof(Point), immature_points.data(), GL_DYNAMIC_COPY);
    TIME_END_GL("upload_immature_points");


    TIME_START_GL("upload_params"); //upload params (https://hub.packtpub.com/opengl-40-using-uniform-blocks-and-uniform-buffer-objects/)
    glBindBuffer( GL_UNIFORM_BUFFER, m_ubo_params );
    glBufferData( GL_UNIFORM_BUFFER, sizeof(m_params), &m_params, GL_DYNAMIC_DRAW );
    glBindBufferBase( GL_UNIFORM_BUFFER,  glGetUniformBlockIndex(m_denoise_depth_prog_id,"params_block"), m_ubo_params );
    TIME_END_GL("upload_params");


    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_points_gl_buf);
    TIME_START_GL("copy_to_texture_kernel");
    glBindImageTexture(0, tex.get_tex_id(), 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);
    glDispatchCompute(immature_points.size()/256, 1, 1);
    TIME_END_GL("copy_to_texture_kernel");


    //denoise the texture
    glUseProgram(m_denoise_texture_prog_id);
    glBindImageTexture(0, tex.get_tex_id(), 0, GL_TRUE, 0, GL_READ_WRITE, GL_R32F);

    TIME_START_GL("depth_denoise_kernel");
    std::cout << "running for "<< m_params.denoise_nr_iterations << '\n';
    for (size_t i = 0; i < m_params.denoise_nr_iterations; i++) {
        // std::cout << "denoise iter " << i << '\n';
        glDispatchCompute(frame_width/32, frame_height/32, 1); //TODO adapt the local size to better suit the gpu
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
    }
    TIME_END_GL("depth_denoise_kernel");
    std::cout << "finished denoise kernel" << '\n';


    //copy from texture to vector with another compute shader AND APPLIED THE MU DENOISED
    glUseProgram(m_copy_from_texture_prog_id);
    glDispatchCompute(immature_points.size()/256, 1, 1);


    //Read to cpu AND APPLY THE DENOISED VALUES
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_points_gl_buf);
    Point* ptr = (Point*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    for (size_t i = 0; i < immature_points.size(); i++) {
        immature_points[i]=ptr[i];
    }
    // glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    // for (auto &point : immature_points) {
    //     // std::cout << "changin mu depth from " << point.mu  << " to " << point.mu_denoised << '\n';
    //     point.mu=point.mu_denoised;
    // }

    TIME_END_GL("denoise_gpu_texture");


}

void DepthEstimatorGL::denoise_gpu_framebuffer(std::vector<Point>& immature_points,  const int frame_width, const int frame_height){
    TIME_START_GL("denoise_gpu_texture");

    //set some stuff on the cpu that is needed for denoising
    int depth_range=m_params.denoise_depth_range;
    float lambda=m_params.denoise_lambda;
    int iterations=m_params.denoise_nr_iterations;
    const float large_sigma2 = depth_range * depth_range / 72.f;
    // computeWeightsAndMu( )
    for ( auto &point : immature_points){
        const float E_pi = point.a / ( point.a + point.b);
        point.g = std::max<float> ( (E_pi * point.sigma2 + (1.0f-E_pi) * large_sigma2) / large_sigma2, 1.0f );
        point.mu_denoised = point.mu;
        point.mu_head = point.u;
        point.p.setZero();
    }


    //make a framebuffer of 4 channels to store mu_head, mu denosied and pvec
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    //rgb output
    GLuint fbo_rgba_tex;
    glGenTextures(1, &fbo_rgba_tex);
    glBindTexture(GL_TEXTURE_2D, fbo_rgba_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, frame_width, frame_height, 0, GL_RGBA, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); //the poor filtering is needed
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    //configure
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, fbo_rgba_tex, 0);
    GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, DrawBuffers); // "1" is the size of DrawBuffers
    // Always check that our framebuffer is ok
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE){
       LOG(FATAL) << "something went wrong with the framebuffer creation";
       return;
    }
    // set back as rendering to screen
    glBindFramebuffer(GL_FRAMEBUFFER, 0);


    //make a 2d texture to store mu and gval
    gl::Texture2D tex;
    tex.set_wrap_mode(GL_CLAMP_TO_EDGE);
    tex.set_filter_mode(GL_NEAREST);
    tex.allocate_tex_storage_inmutable(GL_RG32F, frame_width, frame_height); // 6 channels (because we need 6 members for each immature point)

    //clear the texture to -1 so that we know it's uninitilzied
    std::vector<float> clear(2,-1);
    glClearTexImage(tex.get_tex_id(), 0, GL_RG, GL_FLOAT, clear.data());

    //copy stuff into both fbo texture and this tex
    //copy the immature points to the texture using a compute shader

    //upload to gpu the inmature points
    std::cout << "copying to texture" << '\n';
    glUseProgram(m_copy_to_texture_fbo_prog_id);
    TIME_START_GL("upload_immature_points");
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_points_gl_buf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, immature_points.size() * sizeof(Point), immature_points.data(), GL_DYNAMIC_COPY);
    TIME_END_GL("upload_immature_points");
    TIME_START_GL("upload_params"); //upload params (https://hub.packtpub.com/opengl-40-using-uniform-blocks-and-uniform-buffer-objects/)
    glBindBuffer( GL_UNIFORM_BUFFER, m_ubo_params );
    glBufferData( GL_UNIFORM_BUFFER, sizeof(m_params), &m_params, GL_DYNAMIC_DRAW );
    glBindBufferBase( GL_UNIFORM_BUFFER,  glGetUniformBlockIndex(m_copy_to_texture_fbo_prog_id,"params_block"), m_ubo_params );
    TIME_END_GL("upload_params");
    //do the copy

    TIME_START_GL("copy_to_texture_kernel");
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_points_gl_buf);
    glBindImageTexture(0, fbo_rgba_tex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    glBindImageTexture(1, tex.get_tex_id(), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F);
    glDispatchCompute(immature_points.size()/256, 1, 1);
    TIME_END_GL("copy_to_texture_kernel");
    std::cout << "finished copying to texture" << '\n';




    //make a buffer of vertices (2D points which correspond to the immature points)
    GLuint VertexArrayID;
    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray(VertexArrayID);
    GLuint vertexbuffer;
    glGenBuffers(1, &vertexbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    glBufferData(GL_ARRAY_BUFFER, immature_points.size()*sizeof(float)*3, NULL, GL_STATIC_DRAW);
    float* ptr_vert = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
    for (size_t i = 0; i < immature_points.size()*3; i=i+3) {
        int idx_immature_point=i/3;
        ptr_vert[i]  = lerp(immature_points[idx_immature_point].u, 0, frame_width, -1, 1);  //need to be in range [-1,1]
        ptr_vert[i+1]= lerp(immature_points[idx_immature_point].v, 0, frame_height, -1, 1);
        ptr_vert[i+2]= 0.0; //just a negative value so we are sure it's in front of the camera and we can see it
        // std::cout << "written " << ptr_vert[i] << " " << ptr_vert[i+1] << " " << ptr_vert[i+2] << '\n';
    }
    glUnmapBuffer(GL_ARRAY_BUFFER);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    glVertexAttribPointer(
       0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
       3,                  // size
       GL_FLOAT,           // type
       GL_FALSE,           // normalized?
       0,                  // stride
       (void*)0            // array buffer offset
    );

    //start rendering with those points DOES THE UPDATE
    glUseProgram(m_denoise_fbo_prog_id);
    TIME_START_GL("iter update");
    TIME_START_GL("upload_params"); //upload params (https://hub.packtpub.com/opengl-40-using-uniform-blocks-and-uniform-buffer-objects/)
    glBindBuffer( GL_UNIFORM_BUFFER, m_ubo_params );
    glBufferData( GL_UNIFORM_BUFFER, sizeof(m_params), &m_params, GL_DYNAMIC_DRAW );
    glBindBufferBase( GL_UNIFORM_BUFFER,  glGetUniformBlockIndex(m_denoise_fbo_prog_id,"params_block"), m_ubo_params );
    TIME_END_GL("upload_params");
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glClear(GL_DEPTH_BUFFER_BIT);
    glViewport(0,0,frame_width,frame_height);
    glDisable(GL_DEPTH_TEST); //don't perfor depth testing
    glDepthMask(GL_FALSE);    //don't write to depth buffer
    glPointSize(1.0);
    glBindImageTexture(0, fbo_rgba_tex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F); //cuz may need to read stuff from the neighbours
    glBindImageTexture(1, tex.get_tex_id(), 0, GL_FALSE, 0, GL_READ_WRITE, GL_RG32F);
    glBindFragDataLocation(m_denoise_fbo_prog_id, 0, "color");
    for (size_t i = 0; i < m_params.denoise_nr_iterations; i++) {
        glDrawArrays(GL_POINTS, 0, immature_points.size());
    }
    TIME_END_GL("iter update");






    //copy from texture to vector with another compute shader AND APPLIED THE MU DENOISED
    std::cout << "copying from texture" << '\n';
    glUseProgram(m_copy_from_texture_fbo_prog_id);
    glBindImageTexture(0, fbo_rgba_tex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);  //stores val_mu_denoised mu_head amd p_vec
    glDispatchCompute(immature_points.size()/256, 1, 1);

    //Read to cpu
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_points_gl_buf);
    Point* ptr = (Point*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    for (size_t i = 0; i < immature_points.size(); i++) {
        immature_points[i]=ptr[i];
    }



    // set back as rendering to screen
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);










    // //make a 2d texture array of 6 channels (because we need 6 members for each point)
    // gl::Texture2DArray tex;
    // tex.set_wrap_mode(GL_CLAMP_TO_EDGE);
    // tex.set_filter_mode(GL_NEAREST);
    // tex.allocate_tex_storage_inmutable(GL_R32F, frame_width, frame_height, 6); // 6 channels (because we need 6 members for each immature point)
    //
    // //clear the texture to -1 so that we know it's uninitilzied
    // const float clear=-1;
    // glClearTexImage(tex.get_tex_id(), 0, GL_RED, GL_FLOAT, &clear);
    //
    // //copy the immature points to the texture using a compute shader
    // glUseProgram(m_copy_to_texture_prog_id);
    //
    // //upload to gpu the inmature points
    // TIME_START_GL("upload_immature_points");
    // glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_points_gl_buf);
    // glBufferData(GL_SHADER_STORAGE_BUFFER, immature_points.size() * sizeof(Point), immature_points.data(), GL_DYNAMIC_COPY);
    // TIME_END_GL("upload_immature_points");
    //
    //
    // TIME_START_GL("upload_params"); //upload params (https://hub.packtpub.com/opengl-40-using-uniform-blocks-and-uniform-buffer-objects/)
    // glBindBuffer( GL_UNIFORM_BUFFER, m_ubo_params );
    // glBufferData( GL_UNIFORM_BUFFER, sizeof(m_params), &m_params, GL_DYNAMIC_DRAW );
    // glBindBufferBase( GL_UNIFORM_BUFFER,  glGetUniformBlockIndex(m_copy_to_texture_prog_id,"params_block"), m_ubo_params );
    // TIME_END_GL("upload_params");
    //
    //
    // glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_points_gl_buf);
    // TIME_START_GL("copy_to_texture_kernel");
    // glBindImageTexture(0, tex.get_tex_id(), 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);
    // glDispatchCompute(immature_points.size()/256, 1, 1);
    // TIME_END_GL("copy_to_texture_kernel");
    //
    //
    // //denoise the texture
    // glUseProgram(m_denoise_texture_prog_id);
    // glBindImageTexture(0, tex.get_tex_id(), 0, GL_TRUE, 0, GL_READ_WRITE, GL_R32F);
    //
    // TIME_START_GL("depth_denoise_kernel");
    // std::cout << "running for "<< m_params.denoise_nr_iterations << '\n';
    // for (size_t i = 0; i < m_params.denoise_nr_iterations; i++) {
    //     // std::cout << "denoise iter " << i << '\n';
    //     glDispatchCompute(frame_width/32, frame_height/32, 1); //TODO adapt the local size to better suit the gpu
    //     glMemoryBarrier(GL_ALL_BARRIER_BITS);
    // }
    // TIME_END_GL("depth_denoise_kernel");
    // std::cout << "finished denoise kernel" << '\n';
    //
    //
    // //copy from texture to vector with another compute shader AND APPLIED THE MU DENOISED
    // glUseProgram(m_copy_from_texture_prog_id);
    // glDispatchCompute(immature_points.size()/256, 1, 1);
    //
    //
    // //Read to cpu AND APPLY THE DENOISED VALUES
    // glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_points_gl_buf);
    // Point* ptr = (Point*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    // for (size_t i = 0; i < immature_points.size(); i++) {
    //     immature_points[i]=ptr[i];
    // }
    // // glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    // // for (auto &point : immature_points) {
    // //     // std::cout << "changin mu depth from " << point.mu  << " to " << point.mu_denoised << '\n';
    // //     point.mu=point.mu_denoised;
    // // }

    TIME_END_GL("denoise_gpu_texture");
}

void DepthEstimatorGL::compute_depth_and_create_mesh_cpu(){
    // m_mesh.clear();
    // std::vector<Point> immature_points;
    // immature_points=create_immature_points(m_frames[0]);
    // assign_neighbours_for_points(immature_points, m_frames[0].gray.cols, m_frames[0].gray.rows);
    //
    //
    // for (size_t i = 1; i < m_frames.size(); i++) {
    //
    //     // cv::Mat img_display;
    //     // cv::normalize(m_frames[i].gray, img_display, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    //     // cv::imshow("gray_img", img_display);
    //     // cv::waitKey(30);
    //
    //     bool debug=false;
    //     // if(i==120){
    //     //     debug=true;
    //     // }
    //
    //     std::cout << "frame " << i << '\n';
    //     TIME_START_GL("update_depth");
    //
    //     TIME_START_GL("estimate_affine");
    //     const Eigen::Affine3f tf_cur_host_eigen = m_frames[i].tf_cam_world * m_frames[0].tf_cam_world.inverse();
    //     const Eigen::Affine3f tf_host_cur_eigen = tf_cur_host_eigen.inverse();
    //     const Eigen::Matrix3f KRKi_cr_eigen = m_frames[i].K * tf_cur_host_eigen.linear() * m_frames[0].K.inverse();
    //     const Eigen::Vector3f Kt_cr_eigen = m_frames[i].K * tf_cur_host_eigen.translation();
    //     // const Eigen::Vector2f affine_cr_eigen = estimate_affine( immature_points, frames[i], KRKi_cr_eigen, Kt_cr_eigen);
    //     const Eigen::Vector2f affine_cr_eigen= Eigen::Vector2f(1,1);
    //     const double focal_length = fabs(m_frames[i].K(0,0));
    //     double px_noise = 1.0;
    //     double px_error_angle = atan(px_noise/(2.0*focal_length))*2.0; // law of chord (sehnensatz)
    //     Pattern pattern_rot=m_pattern.get_rotated_pattern( KRKi_cr_eigen.topLeftCorner<2,2>() );
    //     TIME_END_GL("estimate_affine");
    //
    //     //just to be more similar to the shader code
    //     std::vector<Point>& p = immature_points;
    //     Eigen::Matrix3f K= m_frames[i].K ;
    //     Eigen::Vector2f frame_size;
    //     frame_size << m_frames[i].gray.cols , m_frames[i].gray.rows;
    //     const Eigen::Affine3f tf_cur_host=tf_cur_host_eigen;
    //     const Eigen::Affine3f tf_host_cur =tf_host_cur_eigen;
    //     const Eigen::Matrix3f KRKi_cr=KRKi_cr_eigen;
    //     const Eigen::Vector3f Kt_cr=Kt_cr_eigen;
    //     const Eigen::Vector2f affine_cr=affine_cr_eigen;
    //
    //
    //     for (size_t id = 0; id < immature_points.size(); id++) {
    //         if(debug){
    //             std::cout << "processing point " << id << '\n';
    //         }
    //         // // check if point is visible in the current image
    //         const Eigen::Vector3f p_backproj_xyz= p[id].f.head<3>() * 1.0f/ p[id].mu;
    //         const Eigen::Vector4f p_backproj_xyzw=Eigen::Vector4f(p_backproj_xyz.x(),p_backproj_xyz.y(),p_backproj_xyz.z(),1.0);
    //         const Eigen::Vector4f xyz_f_xyzw = tf_cur_host_eigen*  p_backproj_xyzw ;
    //         const Eigen::Vector3f xyz_f=xyz_f_xyzw.head<3>()/xyz_f_xyzw.w();
    //         if(xyz_f.z() < 0.0)  {
    //             continue; // TODO in gl this is a return
    //         }
    //
    //
    //         const Eigen::Vector3f kp_c = K * xyz_f;
    //         const Eigen::Vector2f kp_c_h=kp_c.head<2>()/kp_c.z();
    //         if ( kp_c_h.x() < 0 || kp_c_h.x() >= frame_size.x() || kp_c_h.y() < 0 || kp_c_h.y() >= frame_size.y() ) {
    //             continue; // TODO in gl this is a return
    //         }
    //
    //
    //         //point is visible
    //         // point.last_visible_frame=frames[i].frame_id;
    //
    //         //update inverse depth coordinates for min and max
    //         p[id].idepth_min = p[id].mu + sqrt(p[id].sigma2);
    //         p[id].idepth_max = glm::max(p[id].mu - sqrt(p[id].sigma2), 0.00000001);
    //         // memoryBarrier();
    //         // barrier();
    //         // memoryBarrier();
    //
    //
    //         //search epiline-----------------------------------------------------------------------
    //        // search_epiline_ncc (point, frame, KRKi_cr, Kt_cr );
    //         // search_epiline_bca (point, frames[i], KRKi_cr, Kt_cr, affine_cr);
    //         float idepth_mean = (p[id].idepth_min + p[id].idepth_max)*0.5;
    //         Eigen::Vector3f pr = KRKi_cr * Eigen::Vector3f(p[id].u,p[id].v, 1);
    //         Eigen::Vector3f ptpMean = pr + Kt_cr*idepth_mean;
    //         Eigen::Vector3f ptpMin = pr + Kt_cr*p[id].idepth_min;
    //         Eigen::Vector3f ptpMax = pr + Kt_cr*p[id].idepth_max;
    //         Eigen::Vector2f uvMean = ptpMean.head<2>()/ptpMean.z();
    //         Eigen::Vector2f uvMin = ptpMin.head<2>()/ptpMin.z();
    //         Eigen::Vector2f uvMax = ptpMax.head<2>()/ptpMax.z();
    //
    //         //cap the uv min and uv max
    //         // uvMin(0)=clamp(uvMin(0), 0.0f, (float)m_frame_size.x());
    //         // uvMin(1)=clamp(uvMin(1), 0.0f, (float)m_frame_size.y());
    //         // uvMax(0)=clamp(uvMax(0), 0.0f, (float)m_frame_size.x());
    //         // uvMax(1)=clamp(uvMax(1), 0.0f, (float)m_frame_size.y());
    //
    //
    //         // //debug the uv mean
    //         // if(id==100){
    //         //     p[id].debug2[0]=p[id].u;
    //         //     p[id].debug2[1]=p[id].v;
    //         //     p[id].debug2[2]=uvMean.x; 		// higher -> less strong gradient-based reweighting .
    //         //     p[id].debug2[3]=uvMean.y; // Huber Threshold
    //         //     p[id].debug2[4]=99999999;      //!< threshold on depth uncertainty for convergence.
    //         //     p[id].debug2[5]=999999;
    //         // }
    //
    //         Eigen::Vector2f epi_line = uvMax - uvMin;
    //         float norm_epi = glm::max(1e-5f,epi_line.norm());
    //         Eigen::Vector2f epi_dir = epi_line / norm_epi;
    //         const float  half_length = 0.5f * norm_epi;
    //
    //         Eigen::Vector2f bestKp=Eigen::Vector2f(-1.0,-1.0);
    //         float bestEnergy = 1e10;
    //
    //         if(debug){
    //             std::cout << "p uv is " << p[id].u << " " << p[id].v << '\n';
    //             std::cout << "idepth_mean is " << idepth_mean << '\n';
    //             std::cout << "p mu is " << p[id].mu << '\n';
    //             std::cout << "p sigma2 is " << p[id].sigma2 << '\n';
    //             std::cout << "uvMean is " << uvMean.transpose() << '\n';
    //             std::cout << "uvMin is " << uvMin.transpose() << '\n';
    //             std::cout << "uvMax is " << uvMax.transpose() << '\n';
    //             std::cout << "half_length is " << half_length << '\n';
    //         }
    //
    //         for(float l = -half_length; l <= half_length; l += 0.7f)
    //         {
    //             float energy = 0;
    //             Eigen::Vector2f kp = uvMean + l*epi_dir;
    //
    //             if( ( kp.x() >= (frame_size.x()-20) )  || ( kp.y() >= (frame_size.y()-20) ) || ( kp.x() < 20 ) || ( kp.y() < 20) ){
    //                 continue;
    //             }
    //
    //             for(int idx=0;idx<pattern_rot.get_nr_points(); ++idx){
    //                 //float hitColor = getInterpolatedElement31(frame->dI, (float)(kp(0)+rotatetPattern[idx][0]), (float)(kp(1)+rotatetPattern[idx][1]), wG[0]);
    //                 Eigen::Vector2f offset=pattern_rot.get_offset(idx);
    //                 // float hit_color=texture(gray_img_sampler, vec2( (kp.x + offset.x+0.5)/640.0, (kp.y + offset.y+0.5)/480.0)).x;
    //                 // float hit_color=texelFetch(gray_img_sampler, ivec2( (kp.x + offset.x), (kp.y + offset.y)), 0).x;
    //                 float hit_color=texture_interpolate(m_frames[i].gray, kp.x()+offset.x(), kp.y()+offset.y() , InterpolType::LINEAR);
    //                 // if(!std::isfinite(hit_color)) {energy-=1e5; continue;}
    //
    //                 //for the case when the image is padded
    //                 // float hit_color=texture(gray_img_sampler, vec2( (kp.x + offset.x)/1024, ( 1024-480+  kp.y + offset.y)/1024)).x;
    //
    //                 //high qualty filter from openglsuperbible
    //                 // float hit_color=hqfilter(gray_img_sampler, vec2( (kp.x + offset.x+0.5)/640.0, (kp.y + offset.y+0.5)/480.0)).x;
    //
    //                 // float hit_color=0.0;
    //
    //                 const float residual = hit_color - (affine_cr.x() * p[id].color[idx] + affine_cr.y());
    //
    //                 float hw = abs(residual) < m_params.huberTH ? 1 : m_params.huberTH / abs(residual);
    //                 energy += hw *residual*residual*(2-hw);
    //             }
    //             if ( energy < bestEnergy )
    //             {
    //                 bestKp = kp; bestEnergy = energy;
    //             }
    //         }
    //
    //
    //         if ( bestEnergy > p[id].energyTH * 1.2f ) {
    //             p[id].lastTraceStatus = STATUS_OUTLIER;
    //         }
    //         else
    //         {
    //
    //             Eigen::Vector2f epi_dir_inv=Eigen::Vector2f(epi_dir.y(),-epi_dir.x());
    //             float a = epi_dir.transpose() * p[id].gradH * epi_dir;
    //             float b = epi_dir_inv.transpose() * p[id].gradH * epi_dir_inv;
    //             float errorInPixel = 0.2f + 0.2f * (a+b) / a; // WO kommt das her? Scheint nicht zu NGF zu passen !
    //             // float errorInPixel=0.0f;
    //
    //             if( epi_dir.x()*epi_dir.x()>epi_dir.y()*epi_dir.y() )
    //             {
    //                 p[id].idepth_min = (pr.z()*(bestKp.x()-errorInPixel*epi_dir.x()) - pr.x()) / (Kt_cr.x() - Kt_cr.z()*(bestKp.x()-errorInPixel*epi_dir.x()));
    //                 p[id].idepth_max = (pr.z()*(bestKp.x()+errorInPixel*epi_dir.x()) - pr.x()) / (Kt_cr.x() - Kt_cr.z()*(bestKp.x()+errorInPixel*epi_dir.x()));
    //             }
    //             else
    //             {
    //                 p[id].idepth_min = (pr.z()*(bestKp.y()-errorInPixel*epi_dir.y()) - pr.y()) / (Kt_cr.y() - Kt_cr.z()*(bestKp.y()-errorInPixel*epi_dir.y()));
    //                 p[id].idepth_max = (pr.z()*(bestKp.y()+errorInPixel*epi_dir.y()) - pr.y()) / (Kt_cr.y() - Kt_cr.z()*(bestKp.y()+errorInPixel*epi_dir.y()));
    //             }
    //             // memoryBarrier();
    //             // barrier();
    //             // memoryBarrier();
    //             if(p[id].idepth_min > p[id].idepth_max) {
    //                 // std::swap<float>(point.idepth_min, point.idepth_max);
    //                 float tmp=p[id].idepth_min;
    //                 p[id].idepth_min=p[id].idepth_max;
    //                 p[id].idepth_max=tmp;
    //             }
    //             p[id].lastTraceStatus = STATUS_GOOD;
    //             // memoryBarrier();
    //             // barrier();
    //             // memoryBarrier();
    //         }
    //         // memoryBarrier();
    //         // barrier();
    //         // memoryBarrier();
    //
    //
    //         float idepth = -1;
    //         float z = 0;
    //         if( p[id].lastTraceStatus == STATUS_GOOD ) {
    //             idepth = glm::max(1e-5f, 0.5f*(p[id].idepth_min+p[id].idepth_max));
    //             z = 1.0f/idepth;
    //         }
    //         if ( p[id].lastTraceStatus == STATUS_OOB  || p[id].lastTraceStatus == STATUS_SKIPPED ){
    //             continue; //TODO in shader it's a return
    //         }
    //         //0.04 is 25 meters
    //         if ( idepth<0.04 || idepth>1000 || p[id].lastTraceStatus == STATUS_OUTLIER || p[id].lastTraceStatus == STATUS_BADCONDITION ) {
    //             p[id].b++; // increase outlier probability when no match was found
    //             continue; //TODO in shader it's a return
    //         }
    //         // memoryBarrier();
    //         // barrier();
    //         // memoryBarrier();
    //
    //
    //         // update_idepth(point,tf_host_cur, z, px_error_angle);
    //
    //         // compute tau----------------------------------------------------------------------------
    //         // double tau = compute_tau(tf_host_cur, point.f, z, px_error_angle);
    //         // Eigen::Vector3f t=  Eigen::Vector3f(tf_host_cur[0][3], tf_host_cur[1][3], tf_host_cur[2][3]);
    //         Eigen::Vector3f t(tf_host_cur.translation());
    //         Eigen::Vector3f a = p[id].f.head<3>()*z-t;
    //         float t_norm = t.norm();
    //         float a_norm = a.norm();
    //         float alpha = acos(  p[id].f.head<3>().dot(t)  /t_norm); // dot product
    //         float beta = acos( a.dot(-t) / (t_norm*a_norm)); // dot product
    //         float beta_plus = beta + px_error_angle;
    //         float gamma_plus = M_PI-alpha-beta_plus; // triangle angles sum to PI
    //         float z_plus = t_norm*sin(beta_plus)/sin(gamma_plus); // law of sines
    //         float tau= (z_plus - z); // tau
    //         float tau_inverse = 0.5 * (1.0f/glm::max(0.0000001f, z-tau) - 1.0/(z+tau));
    //
    //         // update the estimate--------------------------------------------------
    //         float x=1.0/z;
    //         float tau2=tau_inverse*tau_inverse;
    //         // updateSeed(point, 1.0/z, tau_inverse*tau_inverse);
    //         float norm_scale = sqrt(p[id].sigma2 + tau2);
    //         float s2 = 1./(1./p[id].sigma2 + 1./tau2);
    //         float m = s2*(p[id].mu/p[id].sigma2 + x/tau2);
    //         float C1 = p[id].a/(p[id].a+p[id].b) * gaus_pdf(p[id].mu, norm_scale, x);
    //         float C2 = p[id].b/(p[id].a+p[id].b) * 1./p[id].z_range;
    //         float normalization_constant = C1 + C2;
    //         C1 /= normalization_constant;
    //         C2 /= normalization_constant;
    //         float f = C1*(p[id].a+1.)/(p[id].a+p[id].b+1.) + C2*p[id].a/(p[id].a+p[id].b+1.);
    //         float e = C1*(p[id].a+1.)*(p[id].a+2.)/((p[id].a+p[id].b+1.)*(p[id].a+p[id].b+2.))
    //                   + C2*p[id].a*(p[id].a+1.0f)/((p[id].a+p[id].b+1.0f)*(p[id].a+p[id].b+2.0f));
    //         // update parameters
    //         float mu_new = C1*m+C2*p[id].mu;
    //         p[id].sigma2 = C1*(s2 + m*m) + C2*(p[id].sigma2 + p[id].mu*p[id].mu) - mu_new*mu_new;
    //         p[id].mu = mu_new;
    //         p[id].a = (e-f)/(f-e/f);
    //         // memoryBarrier();
    //         // barrier();
    //         // memoryBarrier();
    //         p[id].b = p[id].a*(1.0f-f)/f;
    //         // memoryBarrier();
    //         // barrier(); //TODO add again the barrier
    //         // memoryBarrier();
    //
    //         // // not implemented in opengl
    //         const float eta_inlier = .6f;
    //         const float eta_outlier = .05f;
    //         if( ((p[id].a / (p[id].a + p[id].b)) > eta_inlier) && (sqrt(p[id].sigma2) < p[id].z_range/m_params.convergence_sigma2_thresh)) {
    //             p[id].is_outlier = 0; // The seed converged
    //         }else if((p[id].a-1) / (p[id].a + p[id].b - 2) < eta_outlier){ // The seed failed to converge
    //             p[id].is_outlier = 1;
    //             // it->reinit();
    //             //TODO do a better reinit inside a point class
    //             p[id].a = 10;
    //             p[id].b = 10;
    //             p[id].mu = (1.0/m_mean_starting_depth);
    //             p[id].z_range = (1.0/0.1);
    //             p[id].sigma2 = (p[id].z_range*p[id].z_range/36);
    //         }
    //         // if the seed has converged, we initialize a new candidate point and remove the seed
    //         if(sqrt(p[id].sigma2) < p[id].z_range/m_params.convergence_sigma2_thresh){
    //             p[id].converged = 1;
    //         }
    //     }
    //
    // }
    //
    // denoise_cpu(immature_points, m_frames[0].gray.cols, m_frames[0].gray.rows);
    //
    // std::cout << "create mesh" << '\n';
    // m_mesh=create_mesh(immature_points, m_frames);
    // m_points=immature_points; //save the points in the class in case we need them for later saving to a file
    //
    // m_scene_is_modified=true;
    //
    // TIME_END_GL("update_depth");
}

Mesh DepthEstimatorGL::get_mesh(){
    m_scene_is_modified=false;
    return m_mesh;
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
   int images_skipped=0;
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

      if(images_skipped<m_start_frame){
          images_skipped++;
          continue;
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
         cv::Scharr( cur_frame.gray, cur_frame.grad_x, CV_32F, 1, 0);
         cv::Scharr( cur_frame.gray, cur_frame.grad_y, CV_32F, 0, 1);
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

std::vector<Frame> DepthEstimatorGL::loadDataFromRGBD_TUM ( const std::string & dataset_path, const int num_images_to_read ){
    std::vector< Frame > frames;

    //use the associated gt poses rgb and depth from Jan
    std::string filename_associate = dataset_path + "/associate.txt";


    //K is from here https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
    Eigen::Matrix3f K;
    K.setZero();
    Eigen::VectorXf distort_coeffs(5);
    //with no distorsion
    // K(0,0)=525.0; //fx
    // K(1,1)=525.0 ; //fy
    // K(0,2)=319.5; // cx
    // K(1,2)=239.5; //cy
    // K(2,2)=1.0;

    if(dataset_path.find("freiburg1")!=std::string::npos){
        std::cout << "using freiburg1 calibration" << '\n';
        //with distorsion
        K(0,0)=517.3; //fx
        K(1,1)=516.5 ; //fy
        K(0,2)=318.6 ; // cx
        K(1,2)=255.3; //cy
        K(2,2)=1.0;
        distort_coeffs(0)=0.2624;
        distort_coeffs(1)=-0.9531;
        distort_coeffs(2)=-0.0054;
        distort_coeffs(3)=0.0026;
        distort_coeffs(4)=1.1633;
    }else if(dataset_path.find("freiburg2")!=std::string::npos){
        std::cout << "using freiburg2 calibration" << '\n';
        //with distorsion
        K(0,0)=520.9; //fx
        K(1,1)=521.0 ; //fy
        K(0,2)=325.1 ; // cx
        K(1,2)=249.7; //cy
        K(2,2)=1.0;
        Eigen::VectorXf distort_coeffs(5);
        distort_coeffs(0)=0.2312;
        distort_coeffs(1)=-0.7849;
        distort_coeffs(2)=-0.0033;
        distort_coeffs(3)=-0.0001;
        distort_coeffs(4)=0.9172;
    }else if(dataset_path.find("freiburg3")!=std::string::npos){
        std::cout << "using freiburg3 calibration" << '\n';
        //with distorsion
        K(0,0)=535.4; //fx
        K(1,1)=539.2 ; //fy
        K(0,2)=320.1 ; // cx
        K(1,2)=247.6; //cy
        K(2,2)=1.0;
        Eigen::VectorXf distort_coeffs(5);
        distort_coeffs.setZero();
    }else{
        LOG(FATAL) << "Please provide a valid filepath";
    }



    std::ifstream associate_file ( filename_associate, std::ifstream::in );


    std::string line;
    int images_read=0;
    int images_skipped=0;
    while(std::getline(associate_file, line)) {


        float timestamp_gt, timestamp_rgb, timestamp_depth;
        std::string rgb_filename;
        std::string depth_filename;
        float tx, ty, tz, qx, qy, qz, qw;

        std::istringstream iss(line);
        iss >> timestamp_gt  >> tx >> ty >> tz >> qx >> qy >> qz >> qw >> timestamp_rgb >> rgb_filename >> timestamp_depth >> depth_filename;

        if(images_skipped<m_start_frame){
            images_skipped++;
            continue;
        }

        std::cout << "got gt pose: " << tx << " " << ty << " " << tz << " " << qx << " " << qy << " " << qz<< " " << qw << '\n';

        std::cout << "reading " << rgb_filename << '\n';
        std::cout << "reading " << depth_filename << '\n';

        Eigen::Affine3f pose_wc = Eigen::Affine3f::Identity();
        pose_wc.translation() << tx,ty,tz;
        pose_wc.linear() = Eigen::Quaternionf(qw,qx,qy,qz).toRotationMatrix();
        Eigen::Affine3f pose_cw = pose_wc.inverse();


        cv::Mat rgb_cv=cv::imread(dataset_path + "/" + rgb_filename, CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat depth_cv=cv::imread(dataset_path + "/" + depth_filename, CV_LOAD_IMAGE_UNCHANGED);
        depth_cv.convertTo ( depth_cv, CV_32F, 1./5000. ); //ICLNUIM stores theis weird units so we transform to meters


        Frame cur_frame;
        cur_frame.rgb=rgb_cv;
        cv::cvtColor ( cur_frame.rgb, cur_frame.gray, CV_BGR2GRAY );
        cur_frame.gray.convertTo ( cur_frame.gray, CV_32F );
        undistort_image( cur_frame.gray, K, distort_coeffs);
        cur_frame.depth=depth_cv;
        cv::Scharr( cur_frame.gray, cur_frame.grad_x, CV_32F, 1, 0);
        cv::Scharr( cur_frame.gray, cur_frame.grad_y, CV_32F, 0, 1);
        cur_frame.tf_cam_world=pose_cw;
        cur_frame.K=K;
        cur_frame.frame_id=images_read;

        frames.push_back(cur_frame);


        images_read++;
        if(images_read>num_images_to_read){
            break;
        }
    }

    return frames;

}

void DepthEstimatorGL::undistort_image(cv::Mat gray_img, const Eigen::Matrix3f K, const Eigen::VectorXf distort_coeffs){

  cv::Mat undistortMapX, undistortMapY;

  cv::Mat_<double> Kc = cv::Mat_<double>::eye( 3, 3 );
  Kc (0,0) = K(0,0);
  Kc (1,1) = K(1,1);
  Kc (0,2) = K(0,2);
  Kc (1,2) = K(1,2);
  cv::Mat_<double> distortion ( 5, 1 );
  distortion ( 0 ) = distort_coeffs(0);
  distortion ( 1 ) = distort_coeffs(1);
  distortion ( 2 ) = distort_coeffs(2);
  distortion ( 3 ) = distort_coeffs(3);
  distortion ( 4 ) = distort_coeffs(4);
  cv::Mat_<double> Id = cv::Mat_<double>::eye ( 3, 3 );
  cv::initUndistortRectifyMap ( Kc, distortion, Id, Kc, gray_img.size(), CV_32FC1, undistortMapX, undistortMapY );

  cv::Mat undistorted_img;
  cv::remap ( gray_img, undistorted_img, undistortMapX, undistortMapY, cv::INTER_LINEAR );
  gray_img=undistorted_img.clone();
}

void DepthEstimatorGL::save_depth_image(){
    std::ofstream f ( "/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/results/depth_image.txt" );
    for ( const Point & point : m_points ){
        const float idepth = point.mu;
        f << point.u << " " << point.v << " " << idepth << " " << std::endl;
    }
    f.close();
}

std::vector<Point> DepthEstimatorGL::create_immature_points (const Frame& frame){


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

                float gradient_x=frame.grad_x.at<float>(i+dy,j+dx); //TODO should be interpolated
                float gradient_y=frame.grad_y.at<float>(i+dy,j+dx);

                Eigen::Vector2f grad;
                grad << gradient_x, gradient_y;
                // std::cout << "gradients are " << gradient_x << " " << gradient_y << '\n';

                gradient_hessian+= grad*grad.transpose();
            }

            //determinant is high enough, add the point
            float hessian_det=gradient_hessian.determinant();
            if(hessian_det > m_params.gradH_th && frame.depth.at<float>(i,j)!=0.0){
            // if(hessian_det > 0){
                Point point;
                point.u=j;
                point.v=i;
                point.gradH=glm::make_mat2x2(gradient_hessian.data());
                // point.gradH=gradient_hessian;

                //Seed::Seed
                Eigen::Vector3f f_eigen = (frame.K.inverse() * Eigen::Vector3f(point.u,point.v,1)).normalized();
                point.f = glm::vec4(f_eigen(0),f_eigen(1),f_eigen(2), 1.0);
                // point.f=Eigen::Vector4f(f_eigen(0),f_eigen(1),f_eigen(2), 1.0);

                //start at an initial value for depth at around 4 meters (depth_filter->Seed::reinit)
                // float mean_starting_depth=4.0;
                // float mean_starting_depth=frame.depth.at<float>(i,j);
                float mean_starting_depth=m_mean_starting_depth;
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

                    float grad_x_val=texture_interpolate(frame.grad_x, point.u+offset(0), point.v+offset(1), InterpolType::NEAREST);
                    float grad_y_val=texture_interpolate(frame.grad_y, point.u+offset(0), point.v+offset(1), InterpolType::NEAREST);
                    float squared_norm=grad_x_val*grad_x_val + grad_y_val*grad_y_val;
                    point.weights[p_idx] = sqrtf(m_params.outlierTHSumComponent / (m_params.outlierTHSumComponent + squared_norm));

                    //for ngf
                    point.colorD[p_idx] = Eigen::Vector2f(grad_x_val,grad_y_val);
                    point.colorD[p_idx] /= sqrt(point.colorD[p_idx].squaredNorm()+m_params.eta);
                    point.colorGrad[p_idx] =  Eigen::Vector2f(grad_x_val,grad_y_val);


                }
                point.ncc_sum_templ    = 0.0f;
                float ncc_sum_templ_sq = 0.0f;
                for(int p_idx=0;p_idx<m_pattern.get_nr_points();p_idx++){
                    const float templ = point.color[p_idx];
                    point.ncc_sum_templ += templ;
                    ncc_sum_templ_sq += templ*templ;
                }
                point.ncc_const_templ = m_pattern.get_nr_points() * ncc_sum_templ_sq - (double) point.ncc_sum_templ*point.ncc_sum_templ;

                point.energyTH = m_pattern.get_nr_points()*m_params.outlierTH;
                point.energyTH *= m_params.overallEnergyTHWeight*m_params.overallEnergyTHWeight;

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

    return immature_points;
    TIME_END_GL("hessian_host_frame");

}

void DepthEstimatorGL::assign_neighbours_for_points(std::vector<Point>& immature_points, const int frame_width, const int frame_height){
    //TODO this works for the reference frame because we know there will not be overlaps but for any other frames we would need to just reproject the points into the frame and then get the one with the smallest depth in case they lie in the same pixel. Also it would need to be done after updating their depth of course.
    //another way to deal with it is to only make the neighbours for their respective host frame, so we would need to pass a parameter to this function that makes it that we only create neighbours for points that have a specific idx_host_frame


    //make an uniqu identifier for each point, assign that identifier to the texture, and then check neighbours
    std::vector<int> point_ids(immature_points.size());
    for (size_t i = 0; i < point_ids.size(); i++) {
        point_ids[i]=i;
    }


    //TODO create it of a size of frame and initialize to -1
    Eigen::MatrixXi texture_indices(frame_height,frame_width);
    texture_indices.setConstant(-1);

    for (size_t i = 0; i < immature_points.size(); i++) {
        int u=immature_points[i].u;
        int v=immature_points[i].v;
        texture_indices(v,u)=i;
    }


    //go through the immature points again and assign the neighbours
    for (size_t i = 0; i < immature_points.size(); i++) {
        int u=immature_points[i].u;
        int v=immature_points[i].v;

        //left
        int point_left=texture_indices(v,u-1);
        // std::cout << "point left is " << point_left << '\n';
        if(point_left!=-1){ immature_points[i].left=point_left; }

        //right
        int point_right=texture_indices(v,u+1);
        if(point_right!=-1){ immature_points[i].right=point_right; }

        //up
        int point_up=texture_indices(v+1,u);
        if(point_up!=-1){ immature_points[i].above=point_up; }

        //down
        int point_down=texture_indices(v-1,u);
        if(point_down!=-1){ immature_points[i].below=point_down; }

        //left_upper
        int point_left_up=texture_indices(v+1,u-1);
        if(point_left_up!=-1){ immature_points[i].left_upper=point_left_up; }

        //righ_upper
        int point_right_up=texture_indices(v+1,u+1);
        if(point_right_up!=-1){ immature_points[i].right_upper=point_right_up; }

        //left_lower
        int point_left_down=texture_indices(v-1,u-1);
        if(point_left_down!=-1){ immature_points[i].left_lower=point_left_down; }

        //right_lower
        int point_right_down=texture_indices(v-1,u+1);
        if(point_right_down!=-1){ immature_points[i].right_lower=point_right_down; }

    }

}

void DepthEstimatorGL::denoise_cpu( std::vector<Point>& immature_points, const int frame_width, const int frame_height){

    int depth_range=m_params.denoise_depth_range;
    float lambda=m_params.denoise_lambda;
    int iterations=m_params.denoise_nr_iterations;

    std::cout << "starting to denoise." << std::endl;
    const float large_sigma2 = depth_range * depth_range / 72.f;

    // computeWeightsAndMu( )
    for ( auto &point : immature_points){
        const float E_pi = point.a / ( point.a + point.b);

        point.g = std::max<float> ( (E_pi * point.sigma2 + (1.0f-E_pi) * large_sigma2) / large_sigma2, 1.0f );
        point.mu_denoised = point.mu;
        point.mu_head = point.u;
        point.p.setZero();
    }


    const float L = sqrt(8.0f);
    const float tau = (0.02f);
    const float sigma = ((1 / (L*L)) / tau);
    const float theta = 0.5f;

    for (size_t i = 0; i < iterations; i++) {
        // std::cout << "iter " << i << '\n';

        int point_idx=0;
        // update dual
        for ( auto &point : immature_points ){
            // std::cout << "update point " << point_idx << '\n';
            point_idx++;
            const float g = point.g;
            const Eigen::Vector2f p = point.p;
            Eigen::Vector2f grad_uhead = Eigen::Vector2f::Zero();
            const float current_u = point.mu_denoised;

            Point & right = (point.right == -1) ? point : immature_points[point.right];
            Point & below = (point.below == -1) ? point : immature_points[point.below];

            // if(point.right != -1){
            //     std::cout << "------------" << '\n';
            //     std::cout << "point is " << point.u << " " << point.v << '\n';
            //     std::cout << "right is " << right.u << " " << right.v << '\n';
            //     std::cout << "point.right is " << point.right << '\n';
            // }


            grad_uhead[0] = right.mu_head - current_u; //->atXY(min<int>(c_img_size.width-1, x+1), y)  - current_u;
            grad_uhead[1] = below.mu_head - current_u; //->atXY(x, min<int>(c_img_size.height-1, y+1)) - current_u;
            const Eigen::Vector2f temp_p = g * grad_uhead * sigma + p;
            const float sqrt_p = temp_p.norm(); //sqrt(temp_p[0] * temp_p[0] + temp_p[1] * temp_p[1]);
            point.p = temp_p / std::max<float>(1.0f, sqrt_p);
        }

        // std::cout << " update primal" << '\n';
        // update primal:
        for ( auto &point : immature_points ){
            //debug
            // std::cout << "point left is " << point.left << '\n';

            const float noisy_depth = point.mu;
            const float old_u = point.mu_denoised;
            const float g = point.g;

            Eigen::Vector2f current_p = point.p;
            Point & left = (point.left == -1) ? point : immature_points[point.left];
            Point & above = (point.above == -1) ? point : immature_points[point.above];
            Eigen::Vector2f w_p = left.p;
            Eigen::Vector2f n_p = above.p;

            const int x = point.u;
            const int y = point.v;
            if ( x == 0)
                w_p[0] = 0.f;
            else if ( x >= frame_width-1 )
                current_p[0] = 0.f;
            if ( y == 0 )
                n_p[1] = 0.f;
            else if ( y >= frame_height-1 )
                current_p[1] = 0.f;

            const float divergence = current_p[0] - w_p[0] + current_p[1] - n_p[1];

            const float tauLambda = tau*lambda;
            const float temp_u = old_u + tau * g * divergence;
            // std::cout << "tmp u - noisy depth is " << temp_u - noisy_depth << '\n';
            // std::cout << "tauLambda is " << tauLambda << '\n';
            if ((temp_u - noisy_depth) > (tauLambda))
            {
                point.mu_denoised = temp_u - tauLambda;
            }
            else if ((temp_u - noisy_depth) < (-tauLambda))
            {
                point.mu_denoised = temp_u + tauLambda;
            }
            else
            {
                point.mu_denoised = noisy_depth;
            }
            point.mu_head = point.mu_denoised + theta * (point.mu_denoised - old_u);
        }
    }


    for (auto &point : immature_points) {
        // std::cout << "changin mu depth from " << point.mu  << " to " << point.mu_denoised << '\n';
        point.mu=point.mu_denoised;
    }
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
        float color_cur_frame[MAX_RES_PER_POINT];
        float color_host_frame[MAX_RES_PER_POINT];


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


    for (size_t i = 0; i < immature_points.size(); i++) {
        float u=immature_points[i].u;
        float v=immature_points[i].v;
        // float depth=immature_points[i].gt_depth;
        // float depth=1.0;
        float depth=1/immature_points[i].mu;

        if(std::isfinite(immature_points[i].mu) && immature_points[i].mu>=0.1 && immature_points[i].converged==1 && immature_points[i].is_outlier==0 ){
        // if(true){

            // float outlier_measure=immature_points[i].a/(immature_points[i].a+immature_points[i].b);
            // if(outlier_measure<0.7){
            //     continue;
            // }
            //
            // if(immature_points[i].sigma2>0.000005){
            //     continue;
            // }
            //
            // std::cout << immature_points[i].sigma2 << '\n';


            //backproject the immature point
            Eigen::Vector3f point_screen;
            point_screen << u, v, 1.0;
            Eigen::Vector3f point_dir=frames[0].K.inverse()*point_screen; //TODO get the K and also use the cam to world corresponding to the immature point
            // point_dir.normalize(); //this is just the direction (in cam coordinates) of that pixel

            // point_dir=Eigen::Vector3f(point_dir.x()/point_dir.z(), point_dir.y()/point_dir.z(), 1.0);
            Eigen::Vector3f point_cam = point_dir*depth;
            point_cam(2)=-point_cam(2); //flip the depth because opengl 7has a camera which looks at the negative z axis (therefore, more depth means a more negative number)
            if(m_use_rgbd_tum){
                point_cam(1)=-point_cam(1);
            }

            // Eigen::Vector3f point_world=frames[0].tf_cam_world.inverse()*point_cam;

            mesh.V.row(i)=point_cam.cast<double>();

        }


    }

    //make also some colors based on depth
    mesh.C.resize(immature_points.size(),3);
    double min_z, max_z;
    min_z = mesh.V.col(2).minCoeff();
    max_z = mesh.V.col(2).maxCoeff();
    // min_z=-6.5;
    // max_z=-4;
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


  //  //colors based on sigma2
  // float min=9999999999, max=-9999999999;
  // for (size_t i = 0; i < immature_points.size(); i++) {
  //     // std::cout << "last_visible_frame is " << immature_points[i].last_visible_frame << '\n';
  //     if(immature_points[i].sigma2<min){
  //         min=immature_points[i].sigma2;
  //     }
  //     if(immature_points[i].sigma2>max){
  //         max=immature_points[i].sigma2;
  //     }
  // }
  // min=1.0e-07;
  // max=2.0e-05;
  // std::cout << "min max debug is " << min << " " << max << '\n';
  // for (size_t i = 0; i < mesh.C.rows(); i++) {
  //      float gray_val = lerp(immature_points[i].sigma2, min, max, 0.0, 1.0 );
  //      mesh.C(i,0)=mesh.C(i,1)=mesh.C(i,2)=gray_val;
  //  }


  // // //debug mesh to check the texture indices
  // Eigen::MatrixXi texture_indices(480,640);
  // texture_indices.setConstant(-1);
  // for (size_t i = 0; i < immature_points.size(); i++) {
  //     int u=immature_points[i].u;
  //     int v=immature_points[i].v;
  //     texture_indices(v,u)=i;
  // }
  // Mesh debug_mesh;
  // debug_mesh.V.resize(immature_points.size(),3);
  // debug_mesh.V.setZero();
  // for (size_t i = 0; i < immature_points.size(); i++) {
  //     int u=immature_points[i].u;
  //     int v=immature_points[i].v;
  //     // if( texture_indices(v,u)!=-1){
  //     //     debug_mesh.V.row(i) << u,v,0.0;
  //     // }
  //
  //     int point_left=texture_indices(v,u-1);
  //     // std::cout << "point left is " << point_left << '\n';
  //     if(point_left!=-1){
  //         debug_mesh.V.row(i) << u,v,0.0;
  //     }
  // }
  // return debug_mesh;



  //debug by checking the value of debug2 for point 100
  int point_idx=100;
  std::cout << "point " << point_idx << " has uv " << immature_points[point_idx].u << " " << immature_points[point_idx].v << " and gt depth " << immature_points[point_idx].gt_depth << '\n';
  for (size_t i = 0; i < 16; i++) {
      std::cout << "debug2 is " << immature_points[point_idx].debug2[i] << '\n';
  }



    return mesh;
}
