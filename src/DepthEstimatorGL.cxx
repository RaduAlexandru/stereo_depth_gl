#include "stereo_depth_gl/DepthEstimatorGL.h"

//c++
#include <cmath>
#include <stdlib.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <sstream>

//My stuff
#include "stereo_depth_gl/Profiler.h"
#include "stereo_depth_gl/MiscUtils.h"
#include "cv_interpolation.h"
#include "UtilsGL.h"
#include "Shader.h"
#include "Texture2DArray.h"
#include "Texture2D.h"

//ros
#include "stereo_depth_gl/RosTools.h"

//Libigl
#include <igl/opengl/glfw/Viewer.h>

//cv
#include <cv_bridge/cv_bridge.h>

//loguru
#include <loguru.hpp>

//gl
#include <glm/gtc/type_ptr.hpp>

// using namespace glm;
using namespace configuru;

DepthEstimatorGL::DepthEstimatorGL():
        m_gl_profiling_enabled(true),
        m_debug_enabled(false),
        m_mean_starting_depth(4.0),
        m_compute_hessian_pointwise_prog_id(-1),
        m_nr_seeds_left(0),
        m_nr_seeds_right(0),
        m_start_frame(0),
        m_started_new_keyframe(false)
        {

    init_params();
    init_opengl();
    m_pattern.init_pattern(m_pattern_file);

    // //sanity check the pattern
    // std::cout << "pattern has nr of points " << m_pattern.get_nr_points() << '\n';
    // for (size_t i = 0; i < m_pattern.get_nr_points(); i++) {
    //     std::cout << "offset for i " << i << " is " << m_pattern.get_offset(i).transpose() << '\n';
    // }

    //more sanity checks that ensure that however I pad the Seed struct it will be correct
    assert(sizeof(float) == 4);
    assert(sizeof(int32_t) == 4);
    std::cout << "size of Seed " << sizeof(Seed) << '\n';
    std::cout << "size of minimaldepthfilet " << sizeof(MinimalDepthFilter) << '\n';
    std::cout << "size of EpiData " << sizeof(EpiData) << '\n';
    std::cout << "size of Params " << sizeof(Params) << '\n';
    std::cout << "size of Affine3f(for curiosity) " << sizeof(Eigen::Affine3f) << '\n';
    assert(sizeof(Seed)%16== 0); //check that it is correctly padded to 16 bytes
    assert(sizeof(MinimalDepthFilter)%16== 0);
    assert(sizeof(EpiData)%16== 0);
    assert(sizeof(Params)%16== 0);
}

//needed so that forward declarations work
DepthEstimatorGL::~DepthEstimatorGL(){
}

void DepthEstimatorGL::init_params(){
    //get the config filename
    ros::NodeHandle private_nh("~");
    std::string config_file= getParamElseThrow<std::string>(private_nh, "config_file");

    //read all the parameters
    Config cfg = configuru::parse_file(std::string(CMAKE_SOURCE_DIR)+"/config/"+config_file, CFG);
    Config depth_config=cfg["depth"];
    m_gl_profiling_enabled = depth_config["gl_profiling_enabled"];
    m_debug_enabled=depth_config["debug_enabled"];
    m_pattern_file= (std::string)depth_config["pattern_file"];
    m_estimated_seeds_per_keyframe=depth_config["estimated_seeds_per_keyframe"];
    m_nr_buffered_keyframes=depth_config["nr_buffered_keyframes"];
    m_min_starting_depth=depth_config["min_starting_depth"];
    m_mean_starting_depth=depth_config["mean_starting_depth"];

    m_nr_times_frame_used_for_seed_creation_per_cam.resize(20,0); //TODO kinda HACK because we will never have that many cameras
    m_keyframes_per_cam.resize(20);

}

void DepthEstimatorGL::init_opengl(){
    std::cout << "init opengl" << '\n';

    // glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE);
    // glClampColor(GL_CLAMP_VERTEX_COLOR, GL_FALSE);
    // glClampColor(GL_CLAMP_FRAGMENT_COLOR, GL_FALSE);


    if(GL_ARB_debug_output){
    	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB);
    	glDebugMessageCallbackARB(debug_func, (void*)15);
	}

    glGenBuffers(1, &m_seeds_left_gl_buf);
    glGenBuffers(1, &m_seeds_right_gl_buf);
    glGenBuffers(1, &m_ubo_params );
    glGenBuffers(1, &m_epidata_vec_gl_buf);

    //textures storing the gray image with the gradients in the other 2 channes
    m_frame_left.set_wrap_mode(GL_CLAMP_TO_BORDER);
    m_frame_left.set_filter_mode(GL_LINEAR);
    m_frame_right.set_wrap_mode(GL_CLAMP_TO_BORDER);
    m_frame_right.set_filter_mode(GL_LINEAR);

    //for visualization purposes we upload here the rgb images
    m_frame_rgb_left.set_wrap_mode(GL_CLAMP_TO_BORDER);
    m_frame_rgb_left.set_filter_mode(GL_LINEAR);
    m_frame_rgb_right.set_wrap_mode(GL_CLAMP_TO_BORDER);
    m_frame_rgb_right.set_filter_mode(GL_LINEAR);

    //for creating seeds
    m_hessian_pointwise_tex.set_wrap_mode(GL_CLAMP_TO_BORDER);
    m_hessian_pointwise_tex.set_filter_mode(GL_LINEAR);
    m_hessian_blurred_tex.set_wrap_mode(GL_CLAMP_TO_BORDER);
    m_hessian_blurred_tex.set_filter_mode(GL_LINEAR);

    //debug texture
    m_debug_tex.set_wrap_mode(GL_CLAMP_TO_BORDER);
    m_debug_tex.set_filter_mode(GL_LINEAR);

    //Seeds are calculated for each keyframe and we store the seeds of only the last few keyframes.make a buffer for seeds big enough to store around X keyframe worth of data.
    //each keyframe will store a maximum of m_estimated_seeds_per_keyframe
    m_nr_total_seeds=m_estimated_seeds_per_keyframe*m_nr_buffered_keyframes;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_seeds_left_gl_buf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, m_nr_total_seeds * sizeof(Seed), NULL, GL_DYNAMIC_COPY);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_seeds_right_gl_buf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, m_nr_total_seeds * sizeof(Seed), NULL, GL_DYNAMIC_COPY);

    //nr of seeds created is counter with an atomic counter
    glGenBuffers(1, &m_atomic_nr_seeds_created);
	glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, m_atomic_nr_seeds_created);
    GLuint zero=0;
	glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint), &(zero), GL_STATIC_COPY);  //sets it to 0

    //for debuggling using icl nuim
    glGenBuffers(1, &m_points_gl_buf);
    glGenBuffers(1, &m_ubo_params );
    m_cur_frame.set_wrap_mode(GL_CLAMP_TO_BORDER);
    m_cur_frame.set_filter_mode(GL_LINEAR);
    m_ref_frame_tex.set_wrap_mode(GL_CLAMP_TO_BORDER);
    m_ref_frame_tex.set_filter_mode(GL_LINEAR);
    //preemptivelly prealocate a big bufffer for the seeds
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_points_gl_buf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, m_estimated_seeds_per_keyframe * sizeof(Seed), NULL, GL_DYNAMIC_COPY);


    compile_shaders();

}

void DepthEstimatorGL::compile_shaders(){


    m_update_depth_prog_id=gl::program_init_from_files("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_gl/shaders/compute_update_depth.glsl");

    m_compute_hessian_pointwise_prog_id=gl::program_init_from_files("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_gl/shaders/compute_hessian_pointwise.glsl");

    m_compute_hessian_blurred_prog_id=gl::program_init_from_files("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_gl/shaders/compute_hessian_blurred.glsl");

    m_compute_create_seeds_prog_id=gl::program_init_from_files("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_gl/shaders/compute_create_seeds.glsl");

    m_compute_trace_seeds_prog_id=gl::program_init_from_files("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_gl/shaders/compute_trace_seeds.glsl");

    m_compute_trace_seeds_icl_prog_id=gl::program_init_from_files("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_gl/shaders/compute_trace_seeds_icl.glsl");

    // m_compute_hessian_pointwise_prog_id=gl::program_init_from_files("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_gl/shaders/compute_update_depth.glsl");

    // m_denoise_depth_prog_id=gl::program_init_from_files("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_gl/shaders/update_TVL1_primal_dual.glsl");
    //
    // m_copy_to_texture_prog_id=gl::program_init_from_files("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_gl/shaders/compute_copy_to_texture.glsl");
    //
    // m_denoise_texture_prog_id=gl::program_init_from_files("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_gl/shaders/update_TVL1_primal_dual_texture.glsl");
    //
    // m_copy_from_texture_prog_id=gl::program_init_from_files("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_gl/shaders/copy_from_texture.glsl");
    //
    // m_copy_to_texture_fbo_prog_id=gl::program_init_from_files("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_gl/shaders/compute_copy_to_texture_fbo.glsl");
    //
    // m_copy_from_texture_fbo_prog_id=gl::program_init_from_files("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_gl/shaders/copy_from_texture_fbo.glsl");
    //
    // m_denoise_fbo_prog_id=gl::program_init_from_files("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_gl/shaders/update_TVL1_primal_dual_fbo_vert.glsl", "/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_gl/shaders/update_TVL1_primal_dual_fbo_frag.glsl");

}

// void DepthEstimatorGL::compute_depth(const Frame& frame_left, const Frame& frame_right){
//
//     TIME_START_GL("upload_params");
//     //upload params (https://hub.packtpub.com/opengl-40-using-uniform-blocks-and-uniform-buffer-objects/)
//     glBindBuffer( GL_UNIFORM_BUFFER, m_ubo_params );
//     glBufferData( GL_UNIFORM_BUFFER, sizeof(m_params), &m_params, GL_DYNAMIC_DRAW );
//     TIME_END_GL("upload_params");
//
//     //trace all the seeds we have with these new frames
//     //create matrix that projects from eachkeyframe into the current frame
//     //create matrix that transform in world coordinates between the keyframes and the current frame
//     //those matrices upload them to the gpu
//     //the trace shader will query the keyframe of each seed and get the matrix to project it into the current frame
//
//     //update epidata(m_keyframes_per_cam, frame_left, frame_right);
//
//     // trace(m_seeds_left_gl_buf, m_nr_seeds_left, frame_right);
//     // trace(m_seeds_right_gl_buf, m_nr_seeds_right, frame_left);
//
//     //tracing oly of the left seeds and only on the left frame because its closeer and easier to debug
//     // trace(m_seeds_right_gl_buf, m_nr_seeds_right, frame_right);
//
//     //create kf every x frames
//     if(frame_left.frame_idx%20==0){
//         m_keyframes_per_cam[0].push_back(create_keyframe(frame_left));
//         m_keyframes_per_cam[1].push_back(create_keyframe(frame_right));
//
//         //create new seeds from those frames
//         TIME_START_GL("create_all_seeds");
//         std::vector<Seed> seeds_left=create_seeds(frame_left);
//         std::vector<Seed> seeds_right=create_seeds(frame_right);
//         TIME_END_GL("create_all_seeds");
//     }
//
//     trace(m_seeds_right_gl_buf, m_nr_seeds_right, frame_left);
//
// }

// void DepthEstimatorGL::compute_depth_icl(const Frame& frame_left, const Frame& frame_right){
//
//     std::cout << "RECEIVED STEREO PAIR " << '\n';
//     std::cout << "frame_right timestamp and frame_idx " << frame_right.timestamp << " " << frame_right.frame_idx << '\n';
//     std::cout << "frame_right has tf_cam_world \n" << frame_right.tf_cam_world.matrix()  << '\n';
//
//     TIME_START_GL("upload_params");
//     //upload params (https://hub.packtpub.com/opengl-40-using-uniform-blocks-and-uniform-buffer-objects/)
//     glBindBuffer( GL_UNIFORM_BUFFER, m_ubo_params );
//     glBufferData( GL_UNIFORM_BUFFER, sizeof(m_params), &m_params, GL_DYNAMIC_DRAW );
//     TIME_END_GL("upload_params");
//
//     if(frame_left.frame_idx%20==0){
//         m_keyframes_per_cam[0].push_back(create_keyframe(frame_left));
//         m_keyframes_per_cam[1].push_back(create_keyframe(frame_right));
//
//         //create new seeds from those frames
//         TIME_START_GL("create_all_seeds");
//         std::vector<Seed> seeds_left=create_seeds(frame_left);
//         std::vector<Seed> seeds_right=create_seeds(frame_right);
//         TIME_END_GL("create_all_seeds");
//     }else{
//         ////trace-----
//
//         // trace(m_seeds_right_gl_buf, m_nr_seeds_right, frame_left);
//
//         const GLuint m_seeds_gl_buf=m_seeds_right_gl_buf;
//         const int m_nr_seeds=m_nr_seeds_right;
//         const Frame& cur_frame=frame_right;
//         const int keyframe_id=1; //right keyframes
//
//         GL_C( glUseProgram(m_compute_trace_seeds_icl_prog_id) );
//
//         //make a vector of epidate which says for each keyframe (either left or right depending on the thing above) how does to transform to the new frame
//
//         // for (size_t i = 0; i < 1; i++) {
//         Frame keyframe=m_keyframes_per_cam[keyframe_id][m_keyframes_per_cam[keyframe_id].size()-1]; //last keyframe
//         EpiData e;
//
//         // const Eigen::Affine3f tf_cur_host_eigen = cur_frame.tf_cam_world * keyframe.tf_cam_world.inverse();
//         // const Eigen::Affine3f tf_host_cur_eigen = tf_cur_host_eigen.inverse();
//         // const Eigen::Matrix3f KRKi_cr_eigen = cur_frame.K * tf_cur_host_eigen.linear() * keyframe.K.inverse();
//         // const Eigen::Vector3f Kt_cr_eigen = cur_frame.K * tf_cur_host_eigen.translation();
//         // const Eigen::Vector2f affine_cr_eigen= Eigen::Vector2f(1,1); //TODO should be 1,0
//         // const double focal_length = fabs(cur_frame.K(0,0));
//         // double px_noise = 1.0;
//         // double px_error_angle = atan(px_noise/(2.0*focal_length))*2.0; // law of chord (sehnensatz)
//         // Pattern pattern_rot=m_pattern.get_rotated_pattern( KRKi_cr_eigen.topLeftCorner<2,2>() );
//
//
//         //seems that the matrices are maybe wrong
//         const Eigen::Affine3f tf_cur_host_eigen = cur_frame.tf_cam_world * m_frames[0].tf_cam_world.inverse();
//         const Eigen::Affine3f tf_host_cur_eigen = tf_cur_host_eigen.inverse();
//         const Eigen::Matrix3f KRKi_cr_eigen = cur_frame.K * tf_cur_host_eigen.linear() * m_frames[0].K.inverse();
//         const Eigen::Vector3f Kt_cr_eigen = cur_frame.K * tf_cur_host_eigen.translation();
//         // const Eigen::Vector2f affine_cr_eigen = estimate_affine( immature_points, frames[i], KRKi_cr_eigen, Kt_cr_eigen);
//         const Eigen::Vector2f affine_cr_eigen= Eigen::Vector2f(1,0);
//         const double focal_length = fabs(cur_frame.K(0,0));
//         double px_noise = 1.0;
//         double px_error_angle = atan(px_noise/(2.0*focal_length))*2.0; // law of chord (sehnensatz)
//         Pattern pattern_rot=m_pattern.get_rotated_pattern( KRKi_cr_eigen.topLeftCorner<2,2>() );
//
//
//         // std::cout << "tf_cur_host_eigen is \n" << tf_cur_host_eigen.matrix() << '\n';
//         // std::cout << "tf_host_cur_eigen is \n " << tf_host_cur_eigen.matrix() << '\n';
//         // std::cout << "KRKi_cr_eigen is \n" << KRKi_cr_eigen << '\n';
//         // std::cout << "Kt_cr_eigen is " << Kt_cr_eigen.transpose() << '\n';
//         // for (size_t i = 0; i < epidata_vec.size(); i++) {
//         //     std::cout << "e.KRKi_cr is \n" << epidata_vec[i].KRKi_cr << '\n';
//         // }
//
//         // //upload that vector of epidata as another ssbo
//         // GL_C( glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_epidata_vec_gl_buf) );
//         // GL_C( glBufferData(GL_SHADER_STORAGE_BUFFER,  m_keyframes_per_cam[keyframe_id].size() * sizeof(EpiData), epidata_vec.data(), GL_DYNAMIC_COPY) );
//
//
//
//         //upload matrices
//         TIME_START_GL("upload_matrices");
//         Eigen::Vector2f frame_size;
//         frame_size<< cur_frame.gray.cols, cur_frame.gray.rows;
//         glUniform2fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"frame_size"), 1, frame_size.data());
//         glUniformMatrix4fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"tf_cur_host"), 1, GL_TRUE, tf_cur_host_eigen.data());
//         glUniformMatrix4fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"tf_host_cur"), 1, GL_TRUE, tf_host_cur_eigen.data());
//         glUniformMatrix3fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"K"), 1, GL_FALSE, cur_frame.K.data());
//         glUniformMatrix3fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"KRKi_cr"), 1, GL_FALSE, KRKi_cr_eigen.data());
//         glUniform3fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"Kt_cr"), 1, Kt_cr_eigen.data());
//         glUniform2fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"affine_cr"), 1, affine_cr_eigen.data());
//         glUniform1f(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"px_error_angle"), px_error_angle);
//         glUniform2fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"pattern_rot_offsets"), pattern_rot.get_nr_points(), pattern_rot.get_offset_matrix().data()); //upload all the offses as an array of vec2 offsets
//         // std::cout << "setting nr of points to " <<  pattern_rot.get_nr_points() << '\n';
//         // std::cout << "the uniform location is " << glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"pattern_rot_nr_points") << '\n';
//         glUniform1i(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"pattern_rot_nr_points"), pattern_rot.get_nr_points());
//         TIME_END_GL("upload_matrices");
//         glMemoryBarrier(GL_ALL_BARRIER_BITS);
//
//
//         //when tracing, the seeds will have knowledge of the keyframe that hosts them, they will index into the epidata_vector and get the epidata so as to trace into the cur_frame
//         VLOG(1) << "tracing";
//         TIME_START_GL("trace");
//
//
//
//         //debug texture
//         if(!m_debug_tex.get_tex_storage_initialized()){
//             m_debug_tex.allocate_tex_storage_inmutable(GL_RGBA32F,cur_frame.gray.cols, cur_frame.gray.rows);
//         }
//         //clear the debug texture
//         std::vector<GLuint> clear_color(4,0);
//         GL_C ( glClearTexSubImage(m_debug_tex.get_tex_id(), 0,0,0,0, cur_frame.gray.cols,cur_frame.gray.rows,1,GL_RGBA, GL_FLOAT, clear_color.data()) );
//         glBindImageTexture(2, m_debug_tex.get_tex_id(), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
//
//
//
//         glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_seeds_gl_buf);
//         glBindBufferBase( GL_UNIFORM_BUFFER,  glGetUniformBlockIndex(m_compute_trace_seeds_icl_prog_id,"params_block"), m_ubo_params );
//         GL_C(bind_for_sampling(m_frame_left, 1, glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"gray_with_gradients_img_sampler") ) );
//         VLOG(1) << "tracing with " << m_nr_seeds;
//         GL_C( glDispatchCompute(m_nr_seeds/256, 1, 1) ); //TODO adapt the local size to better suit the gpu
//         glMemoryBarrier(GL_ALL_BARRIER_BITS);
//         TIME_END_GL("trace");
//
//
//
//         //debug after tracing
//         std::cout << "debug after tracing " << '\n';
//         glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_seeds_right_gl_buf);
//         Seed* ptr = (Seed*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
//         // print_seed(ptr[0]);
//         for (size_t i = 0; i < 16; i++) {
//             std::cout << " debug " << ptr[30000].debug[i] << '\n';
//         }
//         // for (size_t i = 0; i < m_nr_seeds_left; i++) {
//         //     // print_seed(ptr[i]);
//         // }
//         glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
//     }
//
//
//
//
// }


// std::vector<Seed> DepthEstimatorGL::create_seeds (const Frame& frame){
//
//     //attemot 2 maybe the create seeds is bugged
//     std::vector<Seed> immature_points;
//     immature_points=create_immature_points(frame);
//     std::cout << "immature_points size is " << immature_points.size() << '\n';
//     //upload to gpu the inmature points
//     TIME_START_GL("upload_immature_points");
//     glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_seeds_left_gl_buf);
//     glBufferData(GL_SHADER_STORAGE_BUFFER, immature_points.size() * sizeof(Seed), immature_points.data(), GL_DYNAMIC_COPY);
//     glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_seeds_right_gl_buf);
//     glBufferData(GL_SHADER_STORAGE_BUFFER, immature_points.size() * sizeof(Seed), immature_points.data(), GL_DYNAMIC_COPY);
//     TIME_END_GL("upload_immature_points");
//     m_nr_seeds_left=immature_points.size();
//     m_nr_seeds_right=immature_points.size();
//     return immature_points;
//
//
//
//     // std::vector<Seed> seeds;
//     //
//     //
//     // TIME_START_GL("hessian_det_matrix");
//     // //if the m_hessian_pointwise_tex is not initialize allocate memory for it
//     // if(!m_hessian_pointwise_tex.get_tex_storage_initialized()){
//     //     m_hessian_pointwise_tex.allocate_tex_storage_inmutable(GL_RGBA32F,frame.gray.cols, frame.gray.rows);
//     // }
//     // GL_C( glUseProgram(m_compute_hessian_pointwise_prog_id) );
//     // if(frame.cam_id==0){
//     //     GL_C(bind_for_sampling(m_frame_left, 1, glGetUniformLocation(m_compute_hessian_pointwise_prog_id,"gray_with_gradients_img_sampler") ) );
//     //
//     // }else if(frame.cam_id==1){
//     //     GL_C( bind_for_sampling(m_frame_right, 1, glGetUniformLocation(m_compute_hessian_pointwise_prog_id,"gray_with_gradients_img_sampler") ) );
//     // }else{
//     //     LOG(FATAL) << "Invalid cam id";
//     // }
//     // GL_C( glBindImageTexture(0, m_hessian_pointwise_tex.get_tex_id(), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F) );
//     // GL_C( glDispatchCompute(frame.gray.cols/32, frame.gray.rows/16, 1) ); //TODO need to ceil the size otherwise you will have block of the image that are not computed
//     // glMemoryBarrier(GL_ALL_BARRIER_BITS);
//     // TIME_END_GL("hessian_det_matrix");
//     //
//     //
//     //
//     // //get hessian by box blurring
//     // std::cout << "m_compute_hessian_blurred_prog_id " << m_compute_hessian_blurred_prog_id << '\n';
//     // TIME_START_GL("hessian_blurred_matrix");
//     // //if the m_hessian_pointwise_tex is not initialize allocate memory for it
//     // if(!m_hessian_blurred_tex.get_tex_storage_initialized()){
//     //     VLOG(1) << "initializing m_hessian_blurred_tex";
//     //     GL_C(m_hessian_blurred_tex.allocate_tex_storage_inmutable(GL_RGBA32F,frame.gray.cols, frame.gray.rows) );
//     // }
//     // GL_C( glUseProgram(m_compute_hessian_blurred_prog_id) );
//     // bind_for_sampling(m_hessian_pointwise_tex, 1, glGetUniformLocation(m_compute_hessian_blurred_prog_id,"hessian_pointwise_tex_sampler") );
//     // GL_C( glBindImageTexture(0, m_hessian_blurred_tex.get_tex_id(), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F) );
//     // GL_C( glDispatchCompute(frame.gray.cols/32, frame.gray.rows/16, 1) ); //TODO need to ceil the size otherwise you will have block of the image that are not computed
//     // glMemoryBarrier(GL_ALL_BARRIER_BITS);
//     // TIME_END_GL("hessian_blurred_matrix");
//     //
//     //
//     // //create seeds by calculating the determinant of the hessian and if it's big enough add the seed to the seed buffer and atomically increment a counter for the nr of seeds
//     // TIME_START_GL("create_seeds");
//     // //seeds are stored either in m_seeds_left_gl_buf or in m_seeds_right_gl_buf.m_seeds_left_gl_buf is split equally beteween m_nr_buffered_keyframes
//     // int idx_keyframe=m_nr_times_frame_used_for_seed_creation_per_cam[frame.cam_id]%m_nr_buffered_keyframes; //idx between 0 and m_nr_buffered_keyframes
//     // int allocation_start_idx=idx_keyframe*m_estimated_seeds_per_keyframe;
//     // //add seeds with compute shader
//     // glUseProgram(m_compute_create_seeds_prog_id);
//     // //bind seeds buffer and images
//     // if(frame.cam_id==0){
//     //     glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_seeds_left_gl_buf);
//     //     bind_for_sampling(m_frame_left, 1, glGetUniformLocation(m_compute_create_seeds_prog_id,"gray_with_gradients_img_sampler") );
//     // }else if(frame.cam_id==1){
//     //     glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_seeds_right_gl_buf);
//     //     bind_for_sampling(m_frame_right, 1, glGetUniformLocation(m_compute_create_seeds_prog_id,"gray_with_gradients_img_sampler") );
//     // }else{
//     //     LOG(FATAL) << "Invalid cam_id " << frame.cam_id;
//     // }
//     // bind_for_sampling(m_hessian_blurred_tex, 1, glGetUniformLocation(m_compute_create_seeds_prog_id,"hessian_blurred_sampler") );
//     // //zero o out the counter for seeds creatted
//     // glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 1, m_atomic_nr_seeds_created);
//     // GLuint zero=0;
// 	// glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint), &(zero), GL_STATIC_COPY);
//     //
//     // //debug texture
//     // if(!m_debug_tex.get_tex_storage_initialized()){
//     //     m_debug_tex.allocate_tex_storage_inmutable(GL_RGBA32F,frame.gray.cols, frame.gray.rows);
//     // }
//     // //clear the debug texture
//     // std::vector<GLuint> clear_color(4,0);
//     // GL_C ( glClearTexSubImage(m_debug_tex.get_tex_id(), 0,0,0,0, frame.gray.cols,frame.gray.rows,1,GL_RGBA, GL_FLOAT, clear_color.data()) );
//     // glBindImageTexture(2, m_debug_tex.get_tex_id(), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
//     //
//     // //uniforms needed for creating the seeds
//     // //upload all the offses as an array of vec2 offsets
//     // glUniform2fv(glGetUniformLocation(m_compute_create_seeds_prog_id,"pattern_rot_offsets"), m_pattern.get_nr_points(), m_pattern.get_offset_matrix().data());
//     // glUniform1i(glGetUniformLocation(m_compute_create_seeds_prog_id,"pattern_rot_nr_points"), m_pattern.get_nr_points());
//     // glUniformMatrix3fv(glGetUniformLocation(m_compute_create_seeds_prog_id,"K"), 1, GL_FALSE, frame.K.data());
//     // Eigen::Matrix3f K_inv=frame.K.inverse();
//     // glUniformMatrix3fv(glGetUniformLocation(m_compute_create_seeds_prog_id,"K_inv"), 1, GL_FALSE, K_inv.data());
//     // glUniform1f(glGetUniformLocation(m_compute_create_seeds_prog_id,"min_starting_depth"), m_min_starting_depth);
//     // glUniform1f(glGetUniformLocation(m_compute_create_seeds_prog_id,"mean_starting_depth"), m_mean_starting_depth);
//     // glUniform1i(glGetUniformLocation(m_compute_create_seeds_prog_id,"seeds_start_idx"), 0); //TODO
//     // std::cout << "setting idx_keyframe " << m_keyframes_per_cam[0].size()-1 << '\n';
//     // glUniform1i(glGetUniformLocation(m_compute_create_seeds_prog_id,"idx_keyframe"), m_keyframes_per_cam[0].size()-1);
//     // //TODO maybe change the 0 in the previous m_keyframes_per_cam to something else because we now assume that if we make a keyframe for left cam we also make for right
//     //
//     //
//     // glDispatchCompute(frame.gray.cols/32, frame.gray.rows/16, 1);
//     // m_nr_times_frame_used_for_seed_creation_per_cam[frame.cam_id]++;
//     // glMemoryBarrier(GL_ALL_BARRIER_BITS);
//     //
//     // // get how many seeds were created
//     // GLuint* atomic_nr_seeds_created_cpu= (GLuint*)glMapBufferRange(GL_ATOMIC_COUNTER_BUFFER,0,sizeof(GLuint),GL_MAP_READ_BIT);
//     // glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);
//     // if(frame.cam_id==0){
//     //     m_nr_seeds_left=atomic_nr_seeds_created_cpu[0];
//     // }else{
//     //     m_nr_seeds_right=atomic_nr_seeds_created_cpu[0];
//     // }
//     // std::cout << "atomic_nr_seeds_created_cpu " << atomic_nr_seeds_created_cpu[0] << '\n';
//     //
//     // TIME_END_GL("create_seeds");
//     //
//     //
//     //
//     // // //debug read the seeds back to cpu
//     // // std::cout << "cam " << frame.cam_id << '\n';
//     // // if(frame.cam_id==0){
//     // //     glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_seeds_left_gl_buf);
//     // // }else if(frame.cam_id==1){
//     // //     glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_seeds_right_gl_buf);
//     // // }else{
//     // //     LOG(FATAL) << "Invalid cam_id " << frame.cam_id;
//     // // }
//     // // // glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_seeds_right_gl_buf);
//     // // Seed* ptr = (Seed*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
//     // // for (size_t i = 0; i < std::min((int)atomic_nr_seeds_created_cpu[0],1); i++) {
//     // //     // for (size_t d = 0; d < 16; d++) {
//     // //     //     std::cout << "debug val is " << ptr[i].debug[d] << '\n';
//     // //     // }
//     // //     // std::cout << "uv of seed is " << ptr[i].m_uv.transpose() << '\n';
//     // //     print_seed(ptr[i]);
//     // // }
//     // // glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
//     //
//     //
//     //
//     //
//     // //cpu-----------------
//     // // TIME_START_GL("hessian_det_matrix");
//     // // std::vector<Seed> seeds;
//     // // seeds.reserve(200000);
//     // //
//     // //
//     // // // //multiply the gradx with grady to get a hessian for each point
//     // // // cv::Mat hessian_pointwise=frame.grad_x.mul(frame.grad_y);
//     // // // //the hessian at each point can be blurred a bit to get something from the neighbouring points too
//     // // // cv::Mat hessian;
//     // // // cv::boxFilter(hessian_pointwise,hessian,-1,cv::Size(3,3));
//     // //
//     // //
//     // // //hessian point wise is a 2x2 matrix obtained by multiplying [gx][gx,gy]
//     // // //                                                           [gy]
//     // // //the 2x2  hessian matrix will contain [ gx2 gxgy ]
//     // // //                                     [ gxgy gy2 ]
//     // // //can be stored as a cv mat with 4 channels, each channel will be the multiplication of gradx and gray
//     // // // summing the hessian over a small patch means summing this matrix, which means summing the elements
//     // // // it's the same as blurring each one of the 4 channels
//     // // //the determinant of the hessian is then
//     // // cv::Mat hessian_pointwise = cv::Mat(frame.gray.rows, frame.gray.cols, CV_32FC4);
//     // // for (size_t i = 0; i < hessian_pointwise.rows; i++) {
//     // //     for (size_t j = 0; j < hessian_pointwise.cols; j++) {
//     // //         cv::Vec4f& elem = hessian_pointwise.at<cv::Vec4f>(i,j);
//     // //         float gx=frame.grad_x.at<float>(i,j);
//     // //         float gy=frame.grad_y.at<float>(i,j);
//     // //         //the 4 channels will store the 4 elements of the hessian in row major fashion
//     // //         elem[0]=gx*gx;
//     // //         elem[1]=gx*gy;
//     // //         elem[2]=elem[1];
//     // //         elem[3]=gy*gy;
//     // //     }
//     // // }
//     // //
//     // // //summing over an area the matrix means blurring it
//     // // cv::Mat hessian;
//     // // cv::boxFilter(hessian_pointwise,hessian,-1,cv::Size(3,3));
//     // // // hessian=hessian_pointwise.clone();
//     // //
//     // // //determinant matrix
//     // // cv::Mat determinant(hessian.rows, hessian.cols, CV_32FC1);
//     // // //-Do not look around the borders to avoid pattern accesing outside img
//     // // for (size_t i = m_pattern.get_size().y(); i < frame.gray.rows- m_pattern.get_size().y(); i++) {
//     // //     for (size_t j =  m_pattern.get_size().x(); j < frame.gray.cols- m_pattern.get_size().x(); j++) {
//     // //         //determinant is high enough, add the point
//     // //         cv::Vec4f& elem = hessian.at<cv::Vec4f>(i,j);
//     // //         float hessian_det=elem[0]*elem[3]-elem[1]*elem[2];
//     // //         determinant.at<float>(i,j)=hessian_det;
//     // //     }
//     // // }
//     // //
//     // // // //see determinant
//     // // // cv::Mat determinant_vis;
//     // // // cv::normalize(determinant, determinant_vis, 0, 1.0, cv::NORM_MINMAX);
//     // // // cv::imshow("determinant", determinant_vis);
//     // //
//     // // //whatever is bigger than the gradH_th, we set to 1.0 to indicate that we will create a seed, otherwise set to 0
//     // // cv::Mat high_determinant;
//     // // cv::threshold(determinant, high_determinant, m_params.gradH_th, 1.0, cv::THRESH_BINARY);
//     // // TIME_END_GL("hessian_det_matrix");
//     // //
//     // //
//     // // TIME_START_GL("create_seeds");
//     // // //--------Do not look around the borders to avoid pattern accesing outside img
//     // // for (size_t i = m_pattern.get_size().y(); i < frame.gray.rows- m_pattern.get_size().y(); i++) {
//     // //     for (size_t j =  m_pattern.get_size().x(); j < frame.gray.cols- m_pattern.get_size().x(); j++) {
//     // //
//     // //         //determinant is high enough, add the point
//     // //         float high_det=high_determinant.at<float>(i,j);
//     // //
//     // //         if(high_det==1.0){
//     // //             Seed point;
//     // //             seeds.push_back(point);
//     // //         }
//     // //
//     // //     }
//     // // }
//     //
//     //
//     //
//     //
//     //
//     //
//     //
//     //
//     //
//     //
//     //
//     //
//     // // //--------Do not look around the borders to avoid pattern accesing outside img
//     // // for (size_t i = m_pattern.get_size().y(); i < frame.gray.rows- m_pattern.get_size().y(); i++) {
//     // //     for (size_t j =  m_pattern.get_size().x(); j < frame.gray.cols- m_pattern.get_size().x(); j++) {
//     // //
//     // //         //check if this point has enough determinant in the hessian
//     // //         Eigen::Matrix2f gradient_hessian;
//     // //         gradient_hessian.setZero();
//     // //         for (size_t p = 0; p < m_pattern.get_nr_points(); p++) {
//     // //             int dx = m_pattern.get_offset_x(p);
//     // //             int dy = m_pattern.get_offset_y(p);
//     // //
//     // //             float gradient_x=frame.grad_x.at<float>(i+dy,j+dx);
//     // //             float gradient_y=frame.grad_y.at<float>(i+dy,j+dx);
//     // //
//     // //             Eigen::Vector2f grad;
//     // //             grad << gradient_x, gradient_y;
//     // //             // std::cout << "gradients are " << gradient_x << " " << gradient_y << '\n';
//     // //
//     // //             gradient_hessian+= grad*grad.transpose();
//     // //         }
//     // //
//     // //
//     // //         //determinant is high enough, add the point
//     // //         float hessian_det=gradient_hessian.determinant();
//     // //         if(hessian_det > m_params.gradH_th ){
//     // //         // if(hessian_det > 0){
//     // //             Seed point;
//     // //             point.u=j;
//     // //             point.v=i;
//     // //             point.gradH=glm::make_mat2x2(gradient_hessian.data());
//     // //             // point.gradH=gradient_hessian;
//     // //
//     // //             //Seed::Seed
//     // //             Eigen::Vector3f f_eigen = (frame.K.inverse() * Eigen::Vector3f(point.u,point.v,1)).normalized();
//     // //             point.f = glm::vec4(f_eigen(0),f_eigen(1),f_eigen(2), 1.0);
//     // //             // point.f=Eigen::Vector4f(f_eigen(0),f_eigen(1),f_eigen(2), 1.0);
//     // //
//     // //             //start at an initial value for depth at around 4 meters (depth_filter->Seed::reinit)
//     // //             // float mean_starting_depth=4.0;
//     // //             // float mean_starting_depth=frame.depth.at<float>(i,j);
//     // //             float mean_starting_depth=m_mean_starting_depth;
//     // //             float min_starting_depth=0.1;
//     // //             point.mu = (1.0/mean_starting_depth);
//     // //             point.z_range = (1.0/min_starting_depth);
//     // //             point.sigma2 = (point.z_range*point.z_range/36);
//     // //
//     // //             float z_inv_min = point.mu + sqrt(point.sigma2);
//     // //             float z_inv_max = std::max<float>(point.mu- sqrt(point.sigma2), 0.00000001f);
//     // //             point.idepth_min = z_inv_min;
//     // //             point.idepth_max = z_inv_max;
//     // //
//     // //             point.a=10.0;
//     // //             point.b=10.0;
//     // //
//     // //             //seed constructor deep_filter.h
//     // //             point.converged=0;
//     // //             point.is_outlier=1;
//     // //
//     // //
//     // //             //immature point constructor (the idepth min and max are already set so don't worry about those)
//     // //             point.lastTraceStatus=STATUS_UNINITIALIZED;
//     // //
//     // //             //get data for the color of that point (depth_point->ImmatureSeed::ImmatureSeed)---------------------
//     // //             for(int p_idx=0;p_idx<m_pattern.get_nr_points();p_idx++){
//     // //                 Eigen::Vector2f offset = m_pattern.get_offset(p_idx);
//     // //
//     // //                 point.color[p_idx]=texture_interpolate(frame.gray, point.u+offset(0), point.v+offset(1), InterpolType::NEAREST);
//     // //
//     // //                 float grad_x_val=texture_interpolate(frame.grad_x, point.u+offset(0), point.v+offset(1), InterpolType::NEAREST);
//     // //                 float grad_y_val=texture_interpolate(frame.grad_y, point.u+offset(0), point.v+offset(1), InterpolType::NEAREST);
//     // //                 float squared_norm=grad_x_val*grad_x_val + grad_y_val*grad_y_val;
//     // //                 point.weights[p_idx] = sqrtf(m_params.outlierTHSumComponent / (m_params.outlierTHSumComponent + squared_norm));
//     // //
//     // //                 //for ngf
//     // //                 point.colorD[p_idx] = Eigen::Vector2f(grad_x_val,grad_y_val);
//     // //                 point.colorD[p_idx] /= sqrt(point.colorD[p_idx].squaredNorm()+m_params.eta);
//     // //                 point.colorGrad[p_idx] =  Eigen::Vector2f(grad_x_val,grad_y_val);
//     // //
//     // //
//     // //             }
//     // //             // point.ncc_sum_templ    = 0.0f;
//     // //             // float ncc_sum_templ_sq = 0.0f;
//     // //             // for(int p_idx=0;p_idx<m_pattern.get_nr_points();p_idx++){
//     // //             //     const float templ = point.color[p_idx];
//     // //             //     point.ncc_sum_templ += templ;
//     // //             //     ncc_sum_templ_sq += templ*templ;
//     // //             // }
//     // //             // point.ncc_const_templ = m_pattern.get_nr_points() * ncc_sum_templ_sq - (double) point.ncc_sum_templ*point.ncc_sum_templ;
//     // //
//     // //             point.energyTH = m_pattern.get_nr_points()*m_params.residualTH;
//     // //             point.energyTH *= m_params.overallEnergyTHWeight*m_params.overallEnergyTHWeight;
//     // //
//     // //             point.quality=10000;
//     // //             //-------------------------------------
//     // //
//     // //             //debug stuff
//     // //             point.gradient_hessian_det=hessian_det;
//     // //             point.last_visible_frame=0;
//     // //             // point.gt_depth=frame.depth.at<float>(i,j); //add also the gt depth just for visualization purposes
//     // //             point.debug=0.0;
//     // //
//     // //             seeds.push_back(point);
//     // //         }
//     // //
//     // //     }
//     // // }
//     // // TIME_END_GL("create_seeds");
//     //
//     // // std::cout << "seeds has size " << seeds.size() << '\n';
//     //
//     //
//     //
//     //
//     // return seeds;
//
// }

// void DepthEstimatorGL::trace(const GLuint m_seeds_gl_buf, const int m_nr_seeds, const Frame& cur_frame){
//
//     std::cout << '\n';
//     std::cout << "cur_frame idx is " << cur_frame.frame_idx  << '\n';
//     if(m_nr_seeds==0){
//         return;
//     }
//
//     //are we tracing from a left or a right keyframe?
//     int keyframe_id=-1;
//     if(m_seeds_gl_buf==m_seeds_left_gl_buf){
//         keyframe_id=0;
//     }else{
//         keyframe_id=1;
//     }
//
//     //make a vector of epidate which says for each keyframe (either left or right depending on the thing above) how does to transform to the new frame
//     std::vector<EpiData> epidata_vec;
//     for (size_t i = 0; i < m_keyframes_per_cam[keyframe_id].size(); i++) {
//     // for (size_t i = 0; i < 1; i++) {
//         Frame keyframe=m_keyframes_per_cam[keyframe_id][i];
//         EpiData e;
//
//         Eigen::Affine3f tf_cur_host;
//         tf_cur_host=cur_frame.tf_cam_world * keyframe.tf_cam_world.inverse();
//
//         // e.tf_cur_host=tf_cur_host.matrix();
//         // e.tf_host_cur=e.tf_cur_host.inverse();
//         // e.KRKi_cr=cur_frame.K * tf_cur_host.linear() * keyframe.K.inverse();
//         // e.Kt_cr=cur_frame.K * tf_cur_host.translation();
//         // Pattern pattern_rot=m_pattern.get_rotated_pattern( e.KRKi_cr.topLeftCorner<2,2>() );
//         // e.pattern_rot_offsets.resize(MAX_RES_PER_POINT,2);
//         // e.pattern_rot_offsets.setZero();
//         // e.pattern_rot_offsets.block(0,0,pattern_rot.get_nr_points(),2)=pattern_rot.get_offset_matrix();
//
//         // //attempt 1 but setting the rest to 0
//         // e.tf_cur_host.setZero();
//         // e.tf_host_cur.setZero();
//         // e.KRKi_cr=cur_frame.K * tf_cur_host.linear() * keyframe.K.inverse();
//         // e.Kt_cr.setZero();
//         // e.pattern_rot_offsets.resize(MAX_RES_PER_POINT,2);
//         // e.pattern_rot_offsets.setZero();
//
//
//         //attempt 3 with only 1 matrix
//         // Eigen::Matrix3f KRKi_cr=cur_frame.K * tf_cur_host.linear() * keyframe.K.inverse();
//         // e.KRKi_cr.setZero();
//         // e.KRKi_cr.block(0,0,3,3)=KRKi_cr;
//         // e.KRKi_cr=cur_frame.K * tf_cur_host.linear() * keyframe.K.inverse();
//
//         // //attempt 4 all matrices but aligned
//         // e.tf_cur_host=tf_cur_host.matrix();
//         // e.tf_host_cur=e.tf_cur_host.inverse();
//         // Eigen::Matrix3f KRKi_cr=cur_frame.K * tf_cur_host.linear() * keyframe.K.inverse();
//         // e.KRKi_cr.setZero();
//         // e.KRKi_cr.block(0,0,3,3)=KRKi_cr;
//         // Eigen::Vector3f Kt_cr;
//         // Kt_cr=cur_frame.K * tf_cur_host.translation();
//         // e.Kt_cr.setZero();
//         // e.Kt_cr.block(0,0,1,3)=Kt_cr;
//         // Pattern pattern_rot=m_pattern.get_rotated_pattern( KRKi_cr.topLeftCorner<2,2>() );
//         // e.pattern_rot_offsets.resize(MAX_RES_PER_POINT,2);
//         // e.pattern_rot_offsets.setZero();
//         // e.pattern_rot_offsets.block(0,0,pattern_rot.get_nr_points(),2)=pattern_rot.get_offset_matrix();
//
//
//
//
//         //attempt 5 as the previous one only read correctly the epidata for the first keyframe
//         e.tf_cur_host=tf_cur_host.matrix();
//         e.tf_host_cur=e.tf_cur_host.inverse();
//         Eigen::Matrix3f KRKi_cr=cur_frame.K * tf_cur_host.linear() * keyframe.K.inverse();
//         e.KRKi_cr.setZero();
//         e.KRKi_cr.block(0,0,3,3)=KRKi_cr;
//         Eigen::Vector3f Kt_cr;
//         Kt_cr=cur_frame.K * tf_cur_host.translation();
//         e.Kt_cr.setZero();
//         e.Kt_cr.block(0,0,1,3)=Kt_cr;
//         Pattern pattern_rot=m_pattern.get_rotated_pattern( KRKi_cr.topLeftCorner<2,2>() );
//         e.pattern_rot_offsets.setZero();
//         e.pattern_rot_offsets.block(0,0,pattern_rot.get_nr_points(),2)=pattern_rot.get_offset_matrix();
//
//
//         // if(epidata_vec.size()==0){
//             // std::cout << "------------epidata_vec " << epidata_vec.size() << '\n';
//             // std::cout << "e.tf_cur_host is \n" << e.tf_cur_host << '\n';
//             // std::cout << "e.tf_host_cur is \n" << e.tf_host_cur << '\n';
//             // std::cout << "e.KRKi_cr is \n" << e.KRKi_cr << '\n';
//             // std::cout << "Kt_cr is \n" << Kt_cr << '\n';
//             // std::cout << "e.Kt_cr is \n" << e.Kt_cr << '\n';
//             // std::cout << "pattern offsets is \n " << e.pattern_rot_offsets << '\n';
//         // }
//
//
//         //some of them have to be transposed because of opengl
//         e.tf_cur_host.transposeInPlace();
//         e.tf_host_cur.transposeInPlace();
//         // e.KRKi_cr.transposeInPlace();
//
//         epidata_vec.push_back(e);
//     }
//     for (size_t i = 0; i < epidata_vec.size(); i++) {
//         std::cout << "e.KRKi_cr is \n" << epidata_vec[i].KRKi_cr << '\n';
//     }
//     assert(epidata_vec[0]%16==0);
//     // std::cout << "epidata_vec size  " << epidata_vec.size() << '\n';
//     // std::cout << "sizeof epidata_vec[0]" << sizeof(epidata_vec[0]) << '\n';
//     // //epidata size is not calculated correctly because it contains eigenmatix which is dynamic in size
//     // int size_mat4x4=64; //64bytes because 4x4x4;
//     // int size_vec4=16;
//     // int size_pattern_offsets=MAX_RES_PER_POINT*2*4; //16 points wach with 2 coordinates and each with 4 bytes
//     // // int size_epidata=2*size_mat4x4 + 1*size_mat3x3 + 1*size_vec3 + size_pattern_offsets;
//     // // int size_epidata=3*size_mat4x4 + 1*size_vec4 + size_pattern_offsets;
//     // int size_epidata=3*size_mat4x4 + 1*size_vec4 + size_pattern_offsets;
//     // std::cout << "size manual epidata " << size_epidata << '\n';
//
//
//
//
//     //upload that vector of epidata as another ssbo
//     GL_C( glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_epidata_vec_gl_buf) );
//     GL_C( glBufferData(GL_SHADER_STORAGE_BUFFER,  m_keyframes_per_cam[keyframe_id].size() * sizeof(EpiData), epidata_vec.data(), GL_DYNAMIC_COPY) );
//
//     //TODO get the other necessary data
//     const double focal_length = fabs(cur_frame.K(0,0));
//     double px_noise = 1.0;
//     double px_error_angle = atan(px_noise/(2.0*focal_length))*2.0; // law of chord (sehnensatz)
//
//
//     //when tracing, the seeds will have knowledge of the keyframe that hosts them, they will index into the epidata_vector and get the epidata so as to trace into the cur_frame
//     VLOG(1) << "tracing";
//     TIME_START_GL("trace");
//     GL_C( glUseProgram(m_compute_trace_seeds_prog_id) );
//
//
//     //debug texture
//     if(!m_debug_tex.get_tex_storage_initialized()){
//         m_debug_tex.allocate_tex_storage_inmutable(GL_RGBA32F,cur_frame.gray.cols, cur_frame.gray.rows);
//     }
//     //clear the debug texture
//     std::vector<GLuint> clear_color(4,0);
//     GL_C ( glClearTexSubImage(m_debug_tex.get_tex_id(), 0,0,0,0, cur_frame.gray.cols,cur_frame.gray.rows,1,GL_RGBA, GL_FLOAT, clear_color.data()) );
//     glBindImageTexture(2, m_debug_tex.get_tex_id(), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
//
//     glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_seeds_gl_buf);
//     glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_epidata_vec_gl_buf);
//     glBindBufferBase( GL_UNIFORM_BUFFER,  glGetUniformBlockIndex(m_compute_trace_seeds_prog_id,"params_block"), m_ubo_params );
//     glUniformMatrix3fv(glGetUniformLocation(m_compute_trace_seeds_prog_id,"K"), 1, GL_FALSE, cur_frame.K.data());
//     glUniform1f(glGetUniformLocation(m_compute_trace_seeds_prog_id,"px_error_angle"), px_error_angle);
//     GL_C( glUniform1i(glGetUniformLocation(m_compute_trace_seeds_prog_id,"pattern_rot_nr_points"), m_pattern.get_nr_points()) );
//     if(cur_frame.cam_id==0){
//         GL_C(bind_for_sampling(m_frame_left, 1, glGetUniformLocation(m_compute_trace_seeds_prog_id,"gray_with_gradients_img_sampler") ) );
//
//     }else if(cur_frame.cam_id==1){
//         GL_C( bind_for_sampling(m_frame_right, 1, glGetUniformLocation(m_compute_trace_seeds_prog_id,"gray_with_gradients_img_sampler") ) );
//     }else{
//         LOG(FATAL) << "Invalid cam id";
//     }
//     VLOG(1) << "tracing with " << m_nr_seeds;
//     GL_C( glDispatchCompute(m_nr_seeds/256, 1, 1) ); //TODO adapt the local size to better suit the gpu
//     glMemoryBarrier(GL_ALL_BARRIER_BITS);
//     TIME_END_GL("trace");
//
//
//
//     //debug after tracing
//     std::cout << "debug after tracing " << '\n';
//     glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_seeds_right_gl_buf);
//     Seed* ptr = (Seed*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
//     // print_seed(ptr[0]);
//     for (size_t i = 0; i < 16; i++) {
//         std::cout << " debug " << ptr[30000].debug[i] << '\n';
//     }
//     // for (size_t i = 0; i < m_nr_seeds_left; i++) {
//     //     // print_seed(ptr[i]);
//     // }
//     glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
//
//
//     // glUniformMatrix4fv(glGetUniformLocation(m_update_depth_prog_id,"tf_cur_host"), 1, GL_FALSE, tf_cur_host_eigen_trans.data());
//     // // glUniformMatrix4fv(glGetUniformLocation(m_update_depth_prog_id,"tf_host_cur"), 1, GL_FALSE, tf_host_cur_eigen_trans.data());
//     // glUniformMatrix4fv(glGetUniformLocation(m_update_depth_prog_id,"tf_cur_host"), 1, GL_TRUE, tf_cur_host_eigen.data());
//     // glUniformMatrix4fv(glGetUniformLocation(m_update_depth_prog_id,"tf_host_cur"), 1, GL_TRUE, tf_host_cur_eigen.data());
//     // glUniformMatrix3fv(glGetUniformLocation(m_update_depth_prog_id,"K"), 1, GL_FALSE, m_frames[i].K.data());
//     // glUniformMatrix3fv(glGetUniformLocation(m_update_depth_prog_id,"KRKi_cr"), 1, GL_FALSE, KRKi_cr_eigen.data());
//     // glUniform3fv(glGetUniformLocation(m_update_depth_prog_id,"Kt_cr"), 1, Kt_cr_eigen.data());
//
// }

Mesh DepthEstimatorGL::create_point_cloud(){
    // std::cout << "cam " << frame.cam_id << '\n';
    // if(frame.cam_id==0){
    //     glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_seeds_left_gl_buf);
    // }else if(frame.cam_id==1){
    //     glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_seeds_right_gl_buf);
    // }else{
    //     LOG(FATAL) << "Invalid cam_id " << frame.cam_id;
    // }

    Mesh mesh;
    if(m_nr_seeds_right==0){
        return mesh;
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_seeds_right_gl_buf);
    Seed* ptr = (Seed*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    // for (size_t i = 0; i < m_nr_seeds_left; i++) {
    //     // print_seed(ptr[i]);
    // }




    mesh.V.resize(m_nr_seeds_right,3);
    mesh.V.setZero();

    int nr_infinite_mus=0;
    for (size_t i = 0; i < m_nr_seeds_right; i++) {
        float u=ptr[i].m_uv.x();
        float v=ptr[i].m_uv.y();
        // float depth=immature_points[i].gt_depth;
        // float depth=1.0;
        float depth=1/ptr[i].depth_filter.m_mu;

        if(std::isfinite(ptr[i].depth_filter.m_mu)){
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
            Eigen::Vector3f point_dir=m_keyframes_per_cam[0][0].K.inverse()*point_screen; //TODO get the K and also use the cam to world corresponding to the immature point
            // point_dir.normalize(); //this is just the direction (in cam coordinates) of that pixel

            // point_dir=Eigen::Vector3f(point_dir.x()/point_dir.z(), point_dir.y()/point_dir.z(), 1.0);
            Eigen::Vector3f point_cam = point_dir*depth;
            // point_cam(2)=-point_cam(2); //flip the depth because opengl 7has a camera which looks at the negative z axis (therefore, more depth means a more negative number)
            Eigen::Vector3f point_world=m_keyframes_per_cam[0][0].tf_cam_world.inverse()*point_cam;
            mesh.V.row(i)=point_world.cast<double>();

        }else{
            nr_infinite_mus++;
        }


    }
    LOG(WARNING) << "nr_infinite_mus " << nr_infinite_mus;

    //make also some colors based on depth
    mesh.C.resize(m_nr_seeds_right,3);
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


    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    return mesh;


}

// void DepthEstimatorGL::compute_depth_and_create_mesh(){
//     m_mesh.clear();
//
//
//     TIME_START_GL("compute_depth");
//
//
//     std::vector<Seed> seeds;
//     seeds=create_seeds(m_frames[0]);
//     std::cout << "seeds size is " << seeds.size() << '\n';
//
//
//     //upload to gpu the inmature points
//     TIME_START_GL("upload_seeds");
//     glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_seeds_gl_buf);
//     glBufferData(GL_SHADER_STORAGE_BUFFER, seeds.size() * sizeof(Seed), seeds.data(), GL_DYNAMIC_COPY);
//     TIME_END_GL("upload_seeds");
//
//     glUseProgram(m_update_depth_prog_id);
//     for (size_t i = 1; i < m_frames.size(); i++) {
//         std::cout << "frame " << i << '\n';
//         TIME_START_GL("update_depth");
//
//         TIME_START_GL("estimate_affine");
//         const Eigen::Affine3f tf_cur_host_eigen = m_frames[i].tf_cam_world * m_frames[0].tf_cam_world.inverse();
//         const Eigen::Affine3f tf_host_cur_eigen = tf_cur_host_eigen.inverse();
//         const Eigen::Matrix3f KRKi_cr_eigen = m_frames[i].K * tf_cur_host_eigen.linear() * m_frames[0].K.inverse();
//         const Eigen::Vector3f Kt_cr_eigen = m_frames[i].K * tf_cur_host_eigen.translation();
//         // const Eigen::Vector2f affine_cr_eigen = estimate_affine( seeds, frames[i], KRKi_cr_eigen, Kt_cr_eigen);
//         const Eigen::Vector2f affine_cr_eigen= Eigen::Vector2f(1,1);
//         const double focal_length = fabs(m_frames[i].K(0,0));
//         double px_noise = 1.0;
//         double px_error_angle = atan(px_noise/(2.0*focal_length))*2.0; // law of chord (sehnensatz)
//         Pattern pattern_rot=m_pattern.get_rotated_pattern( KRKi_cr_eigen.topLeftCorner<2,2>() );
//
//         // std::cout << "pattern_rot has nr of points " << pattern_rot.get_nr_points() << '\n';
//         // for (size_t i = 0; i < pattern_rot.get_nr_points(); i++) {
//         //     std::cout << "offset for i " << i << " is " << pattern_rot.get_offset(i).transpose() << '\n';
//         // }
//
//         TIME_END_GL("estimate_affine");
//
//         TIME_START_GL("upload_params");
//         //upload params (https://hub.packtpub.com/opengl-40-using-uniform-blocks-and-uniform-buffer-objects/)
//         glBindBuffer( GL_UNIFORM_BUFFER, m_ubo_params );
//         glBufferData( GL_UNIFORM_BUFFER, sizeof(m_params), &m_params, GL_DYNAMIC_DRAW );
//         GL_C( glBindBufferBase( GL_UNIFORM_BUFFER,  glGetUniformBlockIndex(m_update_depth_prog_id,"params_block"), m_ubo_params ) );
//         TIME_END_GL("upload_params");
//
//         // //upload the image
//         // TIME_START_GL("upload_gray_img");
//         // int size_bytes=frames[i].gray.step[0] * frames[i].gray.rows;
//         // m_cur_frame.upload_data(GL_R32F, frames[i].gray.cols, frames[i].gray.rows, GL_RED, GL_FLOAT, frames[i].gray.ptr(), size_bytes);
//         // TIME_END_GL("upload_gray_img");
//
//
//         // //attempt 2 at uploading image, this time with padding to be power of 2
//         // TIME_START_GL("upload_gray_img");
//         // int padded_img_size=1024;
//         // cv::Mat padded_img(padded_img_size,padded_img_size,CV_32FC1);
//         // // frames[i].gray.copyTo(padded_img(cv::Rect(0,0,frames[i].gray.cols, frames[i].gray.rows)));
//         // frames[i].gray.copyTo(padded_img(cv::Rect(0,padded_img_size-frames[i].gray.rows,frames[i].gray.cols, frames[i].gray.rows)));
//         // // cv::imshow("padded_img",padded_img);
//         // // cv::waitKey(0);
//         // int size_bytes=padded_img.step[0] * padded_img.rows;
//         // m_cur_frame.upload_data(GL_R32F, padded_img.cols, padded_img.rows, GL_RED, GL_FLOAT, padded_img.ptr(), size_bytes);
//         // TIME_END_GL("upload_gray_img");
//
//         // //attempt 3 upload the image as a inmutable storage
//         // TIME_START_GL("upload_gray_img");
//         // int size_bytes=frames[i].gray.step[0] * frames[i].gray.rows;
//         // if(!m_cur_frame.get_tex_storage_initialized()){
//         //     m_cur_frame.allocate_tex_storage_inmutable(GL_R32F,frames[i].gray.cols, frames[i].gray.rows);
//         // }
//         // m_cur_frame.upload_without_pbo(0,0,0, frames[i].gray.cols, frames[i].gray.rows, GL_RED, GL_FLOAT, frames[i].gray.ptr());
//         // TIME_END_GL("upload_gray_img");
//
//
//         //attempt 3 upload the image as a inmutable storage but also pack the gradient into the 2nd and 3rd channel
//         TIME_START_GL("upload_gray_img");
//         //merge all mats into one with 4 channel
//         std::vector<cv::Mat> channels;
//         channels.push_back(m_frames[i].gray);
//         channels.push_back(m_frames[i].grad_x);
//         channels.push_back(m_frames[i].grad_y);
//         channels.push_back(m_frames[i].grad_y); //dummy one stored in the alpha channels just to have a 4 channel texture
//         cv::Mat img_with_gradients;
//         cv::merge(channels, img_with_gradients);
//
//
//         int size_bytes=img_with_gradients.step[0] * img_with_gradients.rows; //allocate 4 channels because gpu likes multiples of 4
//         if(!m_cur_frame.get_tex_storage_initialized()){
//             std::cout << "allocating " << img_with_gradients.cols << "x" << img_with_gradients.rows << '\n';
//             GL_C( m_cur_frame.allocate_tex_storage_inmutable(GL_RGBA32F,img_with_gradients.cols, img_with_gradients.rows) );
//         }
//         std::cout << "uploading" << '\n';
//         m_cur_frame.upload_without_pbo(0,0,0, img_with_gradients.cols, img_with_gradients.rows, GL_RGBA, GL_FLOAT, img_with_gradients.ptr());
//         TIME_END_GL("upload_gray_img");
//
//
//
//
//
//         //upload the matrices
//         TIME_START_GL("upload_matrices");
//         Eigen::Vector2f frame_size;
//         frame_size<< m_frames[i].gray.cols, m_frames[i].gray.rows;
//         const Eigen::Matrix4f tf_cur_host_eigen_trans = tf_cur_host_eigen.matrix().transpose();
//         const Eigen::Matrix4f tf_host_cur_eigen_trans = tf_host_cur_eigen.matrix().transpose();
//         glUniform2fv(glGetUniformLocation(m_update_depth_prog_id,"frame_size"), 1, frame_size.data());
//         // glUniformMatrix4fv(glGetUniformLocation(m_update_depth_prog_id,"tf_cur_host"), 1, GL_FALSE, tf_cur_host_eigen_trans.data());
//         // glUniformMatrix4fv(glGetUniformLocation(m_update_depth_prog_id,"tf_host_cur"), 1, GL_FALSE, tf_host_cur_eigen_trans.data());
//         glUniformMatrix4fv(glGetUniformLocation(m_update_depth_prog_id,"tf_cur_host"), 1, GL_TRUE, tf_cur_host_eigen.data());
//         glUniformMatrix4fv(glGetUniformLocation(m_update_depth_prog_id,"tf_host_cur"), 1, GL_TRUE, tf_host_cur_eigen.data());
//         glUniformMatrix3fv(glGetUniformLocation(m_update_depth_prog_id,"K"), 1, GL_FALSE, m_frames[i].K.data());
//         glUniformMatrix3fv(glGetUniformLocation(m_update_depth_prog_id,"KRKi_cr"), 1, GL_FALSE, KRKi_cr_eigen.data());
//         glUniform3fv(glGetUniformLocation(m_update_depth_prog_id,"Kt_cr"), 1, Kt_cr_eigen.data());
//         glUniform2fv(glGetUniformLocation(m_update_depth_prog_id,"affine_cr"), 1, affine_cr_eigen.data());
//         glUniform1f(glGetUniformLocation(m_update_depth_prog_id,"px_error_angle"), px_error_angle);
//         glUniform2fv(glGetUniformLocation(m_update_depth_prog_id,"pattern_rot_offsets"), pattern_rot.get_nr_points(), pattern_rot.get_offset_matrix().data()); //upload all the offses as an array of vec2 offsets
//         // std::cout << "setting nr of points to " <<  pattern_rot.get_nr_points() << '\n';
//         // std::cout << "the uniform location is " << glGetUniformLocation(m_update_depth_prog_id,"pattern_rot_nr_points") << '\n';
//         glUniform1i(glGetUniformLocation(m_update_depth_prog_id,"pattern_rot_nr_points"), pattern_rot.get_nr_points());
//         TIME_END_GL("upload_matrices");
//         glMemoryBarrier(GL_ALL_BARRIER_BITS);
//
//
//         // tf_cur_host, tf_host_cur, KRKi_cr, Kt_cr, affine_cr, px_error_angle
//         TIME_START_GL("depth_update_kernel");
//         glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_seeds_gl_buf);
//         bind_for_sampling(m_cur_frame, 1, glGetUniformLocation(m_update_depth_prog_id,"gray_img_sampler") );
//         glDispatchCompute(seeds.size()/256, 1, 1); //TODO adapt the local size to better suit the gpu
//         glMemoryBarrier(GL_ALL_BARRIER_BITS);
//         TIME_END_GL("depth_update_kernel");
//
//         TIME_END_GL("update_depth");
//     }
//
//
//
//     //read the points back to cpu
//     //TODO
//     glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_seeds_gl_buf);
//     Seed* ptr = (Seed*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
//     for (size_t i = 0; i < seeds.size(); i++) {
//         seeds[i]=ptr[i];
//     }
//     glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
//
//     assign_neighbours_for_points(seeds, m_frames[0].gray.cols, m_frames[0].gray.rows);
//     // denoise_cpu(seeds, m_frames[0].gray.cols, m_frames[0].gray.rows);
//
//
//     //GPU---------------------------------------------------------------------------------------------------------
//     // denoise_cpu(seeds, m_frames[0].gray.cols, m_frames[0].gray.rows);
//     // denoise_gpu_vector(seeds);
//     denoise_gpu_texture(seeds, m_frames[0].gray.cols, m_frames[0].gray.rows);
//     // denoise_gpu_framebuffer(seeds, m_frames[0].gray.cols, m_frames[0].gray.rows);
//
//
//     TIME_END_GL("compute_depth");
//
//
//     m_mesh=create_mesh(seeds, m_frames);
//     m_points=seeds; //save the points in the class in case we need them for later saving to a file
//
//     m_scene_is_modified=true;
//
// }



Eigen::Vector2f DepthEstimatorGL::estimate_affine(std::vector<Seed>& seeds, const Frame&  cur_frame, const Eigen::Matrix3f& KRKi_cr, const Eigen::Vector3f& Kt_cr){
    // ceres::Problem problem;
    // ceres::LossFunction * loss_function = new ceres::HuberLoss(1);
    // double scaleA = 1;
    // double offsetB = 0;
    //
    // TIME_START("creating ceres problem");
    // for ( int i = 0; i < seeds.size(); ++i )
    // {
    //     Seed& point = seeds[i];
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
    //         Eigen::Vector2f kp_GT = p.hnormalized();
    //
    //
    //         if ( kp_GT(0) > 4 && kp_GT(0) < cur_frame.gray.cols-4 && kp_GT(1) > 3 && kp_GT(1) < cur_frame.gray.rows-4 ) {
    //
    //             Pattern pattern_rot=m_pattern.get_rotated_pattern( KRKi_cr.topLeftCorner<2,2>() );
    //
    //             for(int idx=0;idx<m_pattern.get_nr_points();++idx) {
    //                 Eigen::Vector2f offset=pattern_rot.get_offset(idx);
    //
    //                 color_cur_frame[idx]=texture_interpolate(cur_frame.gray, kp_GT(0)+offset(0), kp_GT(1)+offset(1) , InterpolType::LINEAR);
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
    //         ceres::CostFunction * cost_function = AffineAutoDiffCostFunctorGL::Create( color_cur_frame[i], color_host_frame[i] );
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

void DepthEstimatorGL::upload_gray_stereo_pair(const cv::Mat& image_left, const cv::Mat& image_right){
    TIME_START_GL("upload_rgb_stereo_pair");
    int size_bytes=image_left.step[0] * image_left.rows;
    m_frame_rgb_left.upload_data(GL_R32F, image_left.cols, image_left.rows, GL_RED, GL_FLOAT, image_left.ptr(), size_bytes);


    size_bytes=image_right.step[0] * image_right.rows;
    m_frame_rgb_right.upload_data(GL_R32F, image_right.cols, image_right.rows, GL_RED, GL_FLOAT, image_right.ptr(), size_bytes);
    TIME_END_GL("upload_rgb_stereo_pair");
}

void DepthEstimatorGL::upload_rgb_stereo_pair(const cv::Mat& image_left, const cv::Mat& image_right){
    TIME_START_GL("upload_rgb_stereo_pair");
    int size_bytes=image_left.step[0] * image_left.rows;
    // m_frame_gray_tex.upload_data(GL_R32F, image_left.cols, image_left.rows, GL_RED, GL_FLOAT, image_left.ptr(), size_bytes);
    m_frame_rgb_left.upload_data(GL_RGB, image_left.cols, image_left.rows, GL_BGR, GL_UNSIGNED_BYTE, image_left.ptr(), size_bytes);


    size_bytes=image_right.step[0] * image_right.rows;
    // m_frame_gray_stereo_tex.upload_data(GL_R32F, image_right.cols, image_right.rows, GL_RED, GL_FLOAT, image_right.ptr(), size_bytes);
    m_frame_rgb_right.upload_data(GL_RGB, image_right.cols, image_right.rows, GL_BGR, GL_UNSIGNED_BYTE, image_right.ptr(), size_bytes);
    TIME_END_GL("upload_rgb_stereo_pair");
}

void DepthEstimatorGL::upload_gray_and_grad_stereo_pair(const cv::Mat& image_left, const cv::Mat& image_right){
    TIME_START_GL("upload_gray_and_grad");
    int size_bytes=image_left.step[0] * image_left.rows;
    // m_frame_gray_tex.upload_data(GL_R32F, image_left.cols, image_left.rows, GL_RED, GL_FLOAT, image_left.ptr(), size_bytes);
    m_frame_left.upload_data(GL_RGB32F, image_left.cols, image_left.rows, GL_RGB, GL_FLOAT, image_left.ptr(), size_bytes);


    size_bytes=image_right.step[0] * image_right.rows;
    // m_frame_gray_stereo_tex.upload_data(GL_R32F, image_right.cols, image_right.rows, GL_RED, GL_FLOAT, image_right.ptr(), size_bytes);
    m_frame_right.upload_data(GL_RGB32F, image_right.cols, image_right.rows, GL_RGB, GL_FLOAT, image_right.ptr(), size_bytes);
    TIME_END_GL("upload_gray_and_grad");
}

void DepthEstimatorGL::print_seed(const Seed& s){
    std::cout << "idx_keyframe " << s.idx_keyframe << '\n';
    std::cout << "m_energyTH " << s.m_energyTH << '\n';
    // std::cout << "m_gradH \n \t" << s.m_gradH << '\n';
    std::cout << "m_uv " << s.m_uv.transpose() << '\n';
    std::cout << "m_idepth_minmax " << s.m_idepth_minmax.transpose() << '\n';
    std::cout << "m_best_kp " <<  s.m_best_kp.transpose() << '\n';
    std::cout << "m_min_uv " << s.m_min_uv.transpose() << '\n';
    std::cout << "m_max_uv " << s.m_max_uv.transpose() << '\n';
    std::cout << "m_active_pattern_points " << s.m_active_pattern_points << '\n';

    std::cout << "m_converged " << s.depth_filter.m_converged << '\n';
    std::cout << "m_is_outlier " << s.depth_filter.m_is_outlier << '\n';
    std::cout << "m_initialized " << s.depth_filter.m_initialized << '\n';
    std::cout << "m_f " << s.depth_filter.m_f.transpose() << '\n';
    std::cout << "m_alpha " << s.depth_filter.m_alpha << '\n';
    std::cout << "m_beta " << s.depth_filter.m_beta << '\n';
    std::cout << "m_mu " << s.depth_filter.m_mu << '\n';
    std::cout << "m_z_range " << s.depth_filter.m_z_range << '\n';
    std::cout << "m_sigma2 " << s.depth_filter.m_sigma2 << '\n';
    std::cout << '\n';
}

Frame DepthEstimatorGL::create_keyframe(const Frame& frame){
    Frame keyframe;

    keyframe=frame;
    //get rid of the images since we don't need them at the moment
    keyframe.rgb.release();
    keyframe.gray.release();
    keyframe.grad_x.release();
    keyframe.grad_y.release();
    keyframe.gray_with_gradients.release();
    keyframe.mask.release();
    keyframe.depth.release();

    return keyframe;
}



















// ///ONLY ICL_NUIM-------------------------------------------------------------------------------
//
// std::vector<Frame> DepthEstimatorGL::loadDataFromICLNUIM ( const std::string & dataset_path, const int num_images_to_read){
//    std::vector< Frame > frames;
//    std::string filename_img = dataset_path + "/associations.txt";
//    std::string filename_gt = dataset_path + "/livingRoom2.gt.freiburg";
//
//    //K is from here https://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html
//    Eigen::Matrix3f K;
//    K.setZero();
//    K(0,0)=481.2; //fx
//    K(1,1)=-480; //fy
//    K(0,2)=319.5; // cx
//    K(1,2)=239.5; //cy
//    K(2,2)=1.0;
//
//    std::ifstream imageFile ( filename_img, std::ifstream::in );
//    std::ifstream grtruFile ( filename_gt, std::ifstream::in );
//
//    int imagesRead = 0;
//    int images_skipped=0;
//    for ( imagesRead = 0; imageFile.good() && grtruFile.good() && imagesRead <= num_images_to_read ; imagesRead++ ){
//       std::string depthFileName, colorFileName;
//       int idc, idd, idg;
//       double tsc, tx, ty, tz, qx, qy, qz, qw;
//       imageFile >> idd >> depthFileName >> idc >> colorFileName;
//
//       if ( idd == 0 ){
//           continue;
//       }
//       grtruFile >> idg >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
//
//       if ( idc != idd || idc != idg ){
//           break;
//       }
//
//       if(images_skipped<m_start_frame){
//           std::cout << "skip" << '\n';
//           images_skipped++;
//           continue;
//       }
//
//       if ( ! depthFileName.empty() ){
//          Eigen::Affine3f pose_wc = Eigen::Affine3f::Identity();
//          pose_wc.translation() << tx,ty,tz;
//          pose_wc.linear() = Eigen::Quaternionf(qw,qx,qy,qz).toRotationMatrix();
//          Eigen::Affine3f pose_cw = pose_wc.inverse();
//
//          cv::Mat rgb_cv=cv::imread(dataset_path + "/" + colorFileName, CV_LOAD_IMAGE_UNCHANGED);
//
//          Frame cur_frame;
//          cur_frame.rgb=rgb_cv;
//          cv::cvtColor ( cur_frame.rgb, cur_frame.gray, CV_BGR2GRAY );
//          cur_frame.gray.convertTo ( cur_frame.gray, CV_32F );
//          cv::Scharr( cur_frame.gray, cur_frame.grad_x, CV_32F, 1, 0);
//          cv::Scharr( cur_frame.gray, cur_frame.grad_y, CV_32F, 0, 1);
//          cur_frame.tf_cam_world=pose_cw;
//          cur_frame.K=K;
//          cur_frame.frame_idx=imagesRead;
//
//          // std::cout << "pusng frame with frame idx " << cur_frame.frame_idx << '\n';
//          // std::cout << "frames size" <<  frames.size() <<  "cur_frame.tf_cam_world \n" << cur_frame.tf_cam_world.matrix() << '\n';
//
//          frames.push_back(cur_frame);
//          VLOG(1) << "read img " << imagesRead << " " << colorFileName;
//       }
//    }
//    std::cout << "read " << imagesRead << " images. (" << frames.size() <<", " << ")" << std::endl;
//    return frames;
// }
//
// std::vector<Seed> DepthEstimatorGL::create_immature_points (const Frame& frame){
//
//
//     TIME_START_GL("create_immature_points");
//     std::vector<Seed> immature_points;
//     for (size_t i = 10; i < frame.gray.rows-10; i++) {  //--------Do not look around the borders to avoid pattern accesing outside img
//         for (size_t j = 10; j < frame.gray.cols-10; j++) {
//
//             //check if this point has enough determinant in the hessian
//             Eigen::Matrix2f gradient_hessian;
//             gradient_hessian.setZero();
//             for (size_t p = 0; p < m_pattern.get_nr_points(); p++) {
//                 int dx = m_pattern.get_offset_x(p);
//                 int dy = m_pattern.get_offset_y(p);
//
//                 float gradient_x=frame.grad_x.at<float>(i+dy,j+dx); //TODO should be interpolated
//                 float gradient_y=frame.grad_y.at<float>(i+dy,j+dx);
//
//                 Eigen::Vector2f grad;
//                 grad << gradient_x, gradient_y;
//                 // std::cout << "gradients are " << gradient_x << " " << gradient_y << '\n';
//
//                 gradient_hessian+= grad*grad.transpose();
//             }
//
//             //determinant is high enough, add the point
//             float hessian_det=gradient_hessian.determinant();
//             if(hessian_det > m_params.gradH_th){
//             // if(hessian_det > 0){
//                 Seed point;
//                 point.m_uv << j,i;
//                 point.m_gradH=gradient_hessian;
//
//                 //Seed::Seed
//                 Eigen::Vector3f f_eigen = (frame.K.inverse() * Eigen::Vector3f(point.m_uv.x(),point.m_uv.y(),1)).normalized();
//                 // f_eigen.normalize();
//                 // point.f = glm::vec4(f_eigen(0),f_eigen(1),f_eigen(2), 1.0);
//                 point.depth_filter.m_f=Eigen::Vector4f(f_eigen(0),f_eigen(1),f_eigen(2), 1.0);
//
//                 //start at an initial value for depth at around 4 meters (depth_filter->Seed::reinit)
//                 // float mean_starting_depth=4.0;
//                 // float mean_starting_depth=frame.depth.at<float>(i,j);
//                 float mean_starting_depth=m_mean_starting_depth;
//                 float min_starting_depth=0.1;
//                 point.depth_filter.m_mu = (1.0/mean_starting_depth);
//                 point.depth_filter.m_z_range = (1.0/min_starting_depth);
//                 point.depth_filter.m_sigma2 = (point.depth_filter.m_z_range*point.depth_filter.m_z_range/36);
//
//                 float z_inv_min = point.depth_filter.m_mu + sqrt(point.depth_filter.m_sigma2);
//                 float z_inv_max = std::max<float>(point.depth_filter.m_mu- sqrt(point.depth_filter.m_sigma2), 0.00000001f);
//                 point.m_idepth_minmax <<  z_inv_min, z_inv_max;
//
//                 point.depth_filter.m_alpha=10.0;
//                 point.depth_filter.m_beta=10.0;
//
//
//
//                 //get data for the color of that point (depth_point->ImmaturePoint::ImmaturePoint)---------------------
//                 for(int p_idx=0;p_idx<m_pattern.get_nr_points();p_idx++){
//                     Eigen::Vector2f offset = m_pattern.get_offset(p_idx);
//
//                     point.m_intensity[p_idx]=texture_interpolate(frame.gray, point.m_uv.x()+offset(0), point.m_uv.y()+offset(1), InterpolType::NEAREST);
//
//                     // float grad_x_val=texture_interpolate(frame.grad_x,  point.m_uv.x()+offset(0), point.m_uv.y()+offset(1), InterpolType::NEAREST);
//                     // float grad_y_val=texture_interpolate(frame.grad_y,  point.m_uv.x()+offset(0), point.m_uv.y()+offset(1), InterpolType::NEAREST);
//                     // float squared_norm=grad_x_val*grad_x_val + grad_y_val*grad_y_val;
//                     // point.weights[p_idx] = sqrtf(m_params.outlierTHSumComponent / (m_params.outlierTHSumComponent + squared_norm));
//                     // //for ngf
//                     // point.colorD[p_idx] = Eigen::Vector2f(grad_x_val,grad_y_val);
//                     // point.colorD[p_idx] /= sqrt(point.colorD[p_idx].squaredNorm()+m_params.eta);
//                     // point.colorGrad[p_idx] =  Eigen::Vector2f(grad_x_val,grad_y_val);
//
//                     //for ngf attempt 2
//                     float grad_x_val=texture_interpolate(frame.grad_x,  point.m_uv.x()+offset(0), point.m_uv.y()+offset(1), InterpolType::NEAREST);
//                     float grad_y_val=texture_interpolate(frame.grad_y,  point.m_uv.x()+offset(0), point.m_uv.y()+offset(1), InterpolType::NEAREST);
//                     point.m_normalized_grad[p_idx] << grad_x_val, grad_y_val;
//                     point.m_normalized_grad[p_idx] /= sqrt(point.m_normalized_grad[p_idx].squaredNorm() + m_params.eta);
//                     if(point.m_normalized_grad[p_idx].norm()<1e-3){
//                         point.m_zero_grad[p_idx]=1;
//                     }else{
//                         point.m_zero_grad[p_idx]=0;
//                         point.m_active_pattern_points++;
//                     }
//
//
//                 }
//                 point.m_energyTH = m_pattern.get_nr_points()*m_params.residualTH;
//                 // point.m_energyTH *= m_params.overallEnergyTHWeight*m_params.overallEnergyTHWeight;
//
//                 point.m_last_error = -1;
//                 point.depth_filter.m_is_outlier = 0;
//
//                 immature_points.push_back(point);
//             }
//
//         }
//     }
//
//     TIME_END_GL("create_immature_points");
//     return immature_points;
//
//
// }
//
//
// void DepthEstimatorGL::compute_depth_and_create_mesh_ICL(){
//
//     m_frames=loadDataFromICLNUIM("/media/alex/Data/Master/Thesis/data/ICL_NUIM/living_room_traj2_frei_png", 60);
//
//     TIME_START_GL("compute_depth");
//
//
//     std::vector<Seed> immature_points;
//     immature_points=create_immature_points(m_frames[0]);
//     std::cout << "immature_points size is " << immature_points.size() << '\n';
//
//     glUseProgram(m_compute_trace_seeds_icl_prog_id);
//
//
//     //upload to gpu the inmature points
//     TIME_START_GL("upload_immature_points");
//     glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_points_gl_buf);
//     glBufferData(GL_SHADER_STORAGE_BUFFER, immature_points.size() * sizeof(Seed), immature_points.data(), GL_DYNAMIC_COPY);
//     TIME_END_GL("upload_immature_points");
//
//
//     for (size_t i = 1; i < m_frames.size(); i++) {
//         std::cout << "frame " << i << '\n';
//         TIME_START_GL("update_depth");
//
//         TIME_START_GL("estimate_affine");
//         const Eigen::Affine3f tf_cur_host_eigen = m_frames[i].tf_cam_world * m_frames[0].tf_cam_world.inverse();
//         const Eigen::Affine3f tf_host_cur_eigen = tf_cur_host_eigen.inverse();
//         const Eigen::Matrix3f KRKi_cr_eigen = m_frames[i].K * tf_cur_host_eigen.linear() * m_frames[0].K.inverse();
//         const Eigen::Vector3f Kt_cr_eigen = m_frames[i].K * tf_cur_host_eigen.translation();
//         // const Eigen::Vector2f affine_cr_eigen = estimate_affine( immature_points, frames[i], KRKi_cr_eigen, Kt_cr_eigen);
//         const Eigen::Vector2f affine_cr_eigen= Eigen::Vector2f(1,0);
//         const double focal_length = fabs(m_frames[i].K(0,0));
//         double px_noise = 1.0;
//         double px_error_angle = atan(px_noise/(2.0*focal_length))*2.0; // law of chord (sehnensatz)
//         Pattern pattern_rot=m_pattern.get_rotated_pattern( KRKi_cr_eigen.topLeftCorner<2,2>() );
//
//         // std::cout << "tf_cur_host_eigen is \n" << tf_cur_host_eigen.matrix() << '\n';
//         // std::cout << "tf_host_cur_eigen is \n " << tf_host_cur_eigen.matrix() << '\n';
//         // std::cout << "KRKi_cr_eigen is \n" << KRKi_cr_eigen << '\n';
//         // std::cout << "Kt_cr_eigen is " << Kt_cr_eigen.transpose() << '\n';
//
//         // std::cout << "pattern_rot has nr of points " << pattern_rot.get_nr_points() << '\n';
//         // for (size_t i = 0; i < pattern_rot.get_nr_points(); i++) {
//         //     std::cout << "offset for i " << i << " is " << pattern_rot.get_offset(i).transpose() << '\n';
//         // }
//
//         TIME_END_GL("estimate_affine");
//
//         TIME_START_GL("upload_params");
//         //upload params (https://hub.packtpub.com/opengl-40-using-uniform-blocks-and-uniform-buffer-objects/)
//         glBindBuffer( GL_UNIFORM_BUFFER, m_ubo_params );
//         glBufferData( GL_UNIFORM_BUFFER, sizeof(m_params), &m_params, GL_DYNAMIC_DRAW );
//         glBindBufferBase( GL_UNIFORM_BUFFER,  glGetUniformBlockIndex(m_compute_trace_seeds_icl_prog_id,"params_block"), m_ubo_params );
//         TIME_END_GL("upload_params");
//
//
//         //attempt 3 upload the image as a inmutable storage but also pack the gradient into the 2nd and 3rd channel
//         TIME_START_GL("upload_gray_img");
//         //merge all mats into one with 4 channel
//         std::vector<cv::Mat> channels;
//         channels.push_back(m_frames[i].gray);
//         channels.push_back(m_frames[i].grad_x);
//         channels.push_back(m_frames[i].grad_y);
//         channels.push_back(m_frames[i].grad_y); //dummy one stored in the alpha channels just to have a 4 channel texture
//         cv::Mat img_with_gradients;
//         cv::merge(channels, img_with_gradients);
//
//         int size_bytes=img_with_gradients.step[0] * img_with_gradients.rows; //allocate 4 channels because gpu likes multiples of 4
//         if(!m_cur_frame.get_tex_storage_initialized()){
//             m_cur_frame.allocate_tex_storage_inmutable(GL_RGBA32F,img_with_gradients.cols, img_with_gradients.rows);
//         }
//         m_cur_frame.upload_without_pbo(0,0,0, img_with_gradients.cols, img_with_gradients.rows, GL_RGBA, GL_FLOAT, img_with_gradients.ptr());
//         TIME_END_GL("upload_gray_img");
//
//
//
//
//
//         //upload the matrices
//         TIME_START_GL("upload_matrices");
//         Eigen::Vector2f frame_size;
//         frame_size<< m_frames[i].gray.cols, m_frames[i].gray.rows;
//         glUniform2fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"frame_size"), 1, frame_size.data());
//         // glUniformMatrix4fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"tf_cur_host"), 1, GL_FALSE, tf_cur_host_eigen_trans.data());
//         // glUniformMatrix4fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"tf_host_cur"), 1, GL_FALSE, tf_host_cur_eigen_trans.data());
//         glUniformMatrix4fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"tf_cur_host"), 1, GL_TRUE, tf_cur_host_eigen.data());
//         glUniformMatrix4fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"tf_host_cur"), 1, GL_TRUE, tf_host_cur_eigen.data());
//         glUniformMatrix3fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"K"), 1, GL_FALSE, m_frames[i].K.data());
//         glUniformMatrix3fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"KRKi_cr"), 1, GL_FALSE, KRKi_cr_eigen.data());
//         glUniform3fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"Kt_cr"), 1, Kt_cr_eigen.data());
//         glUniform2fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"affine_cr"), 1, affine_cr_eigen.data());
//         glUniform1f(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"px_error_angle"), px_error_angle);
//         glUniform2fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"pattern_rot_offsets"), pattern_rot.get_nr_points(), pattern_rot.get_offset_matrix().data()); //upload all the offses as an array of vec2 offsets
//         // std::cout << "setting nr of points to " <<  pattern_rot.get_nr_points() << '\n';
//         // std::cout << "the uniform location is " << glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"pattern_rot_nr_points") << '\n';
//         glUniform1i(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"pattern_rot_nr_points"), pattern_rot.get_nr_points());
//         TIME_END_GL("upload_matrices");
//         glMemoryBarrier(GL_ALL_BARRIER_BITS);
//
//
//         // tf_cur_host, tf_host_cur, KRKi_cr, Kt_cr, affine_cr, px_error_angle
//         TIME_START_GL("depth_update_kernel");
//         glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_points_gl_buf);
//         bind_for_sampling(m_cur_frame, 1, glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"gray_img_sampler") );
//         glDispatchCompute(immature_points.size()/256, 1, 1); //TODO adapt the local size to better suit the gpu
//         glMemoryBarrier(GL_ALL_BARRIER_BITS);
//         TIME_END_GL("depth_update_kernel");
//
//         TIME_END_GL("update_depth");
//     }
//
//
//
//     //read the points back to cpu
//     //TODO
//     glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_points_gl_buf);
//     Seed* ptr = (Seed*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
//     for (size_t i = 0; i < immature_points.size(); i++) {
//         immature_points[i]=ptr[i];
//     }
//     glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
//
//     TIME_END_GL("compute_depth");
//
//
//     m_mesh=create_mesh_ICL(immature_points, m_frames);
//
// }
//
// Mesh DepthEstimatorGL::create_mesh_ICL(const std::vector<Seed>& immature_points, const std::vector<Frame>& frames){
//     Mesh mesh;
//     mesh.V.resize(immature_points.size(),3);
//     mesh.V.setZero();
//
//     Eigen::VectorXd betas(immature_points.size());
//     betas.setZero();
//     Eigen::VectorXd sigmas(immature_points.size());
//     sigmas.setZero();
//     Eigen::VectorXd last_errors(immature_points.size());
//     last_errors.setZero();
//
//
//     for (size_t i = 0; i < immature_points.size(); i++) {
//         float u=immature_points[i].m_uv.x();
//         float v=immature_points[i].m_uv.y();
//         // float depth=immature_points[i].gt_depth;
//         // float depth=1.0;
//         float depth=1/immature_points[i].depth_filter.m_mu;
//
//         if(std::isfinite(immature_points[i].depth_filter.m_mu) && immature_points[i].depth_filter.m_mu>=0.1
//             && immature_points[i].depth_filter.m_is_outlier==0 ){
//         // if(true){
//
//             // float outlier_measure=immature_points[i].a/(immature_points[i].a+immature_points[i].b);
//             // if(outlier_measure<0.7){
//             //     continue;
//             // }
//             //
//             // if(immature_points[i].sigma2>0.000005){
//             //     continue;
//             // }
//             //
//             // std::cout << immature_points[i].sigma2 << '\n';
//
//
//             //backproject the immature point
//             Eigen::Vector3f point_screen;
//             point_screen << u, v, 1.0;
//             Eigen::Vector3f point_dir=frames[0].K.inverse()*point_screen; //TODO get the K and also use the cam to world corresponding to the immature point
//             // point_dir.normalize(); //this is just the direction (in cam coordinates) of that pixel
//
//             // point_dir=Eigen::Vector3f(point_dir.x()/point_dir.z(), point_dir.y()/point_dir.z(), 1.0);
//             Eigen::Vector3f point_cam = point_dir*depth;
//             point_cam(2)=-point_cam(2); //flip the depth because opengl 7has a camera which looks at the negative z axis (therefore, more depth means a more negative number)
//             // Eigen::Vector3f point_world=frames[0].tf_cam_world.inverse()*point_cam;
//
//             mesh.V.row(i)=point_cam.cast<double>();
//
//             //store also some atributes fro the seeds
//             betas(i)=immature_points[i].depth_filter.m_beta;
//             sigmas(i)=immature_points[i].depth_filter.m_sigma2;
//             last_errors(i)=immature_points[i].m_last_error;
//
//         }
//
//
//     }
//
//     //make also some colors based on depth
//     mesh.C.resize(immature_points.size(),3);
//     double min_z, max_z;
//     min_z = mesh.V.col(2).minCoeff();
//     max_z = mesh.V.col(2).maxCoeff();
//     min_z=-5.5;
//     max_z=-2;
//     std::cout << "min max z is " << min_z << " " << max_z << '\n';
//     for (size_t i = 0; i < mesh.C.rows(); i++) {
//         float gray_val = lerp(mesh.V(i,2), min_z, max_z, 0.0, 1.0 );
//         mesh.C(i,0)=mesh.C(i,1)=mesh.C(i,2)=gray_val;
//     }
//
//
//     // // //make also some colors based on the beta value
//     // mesh.C.resize(immature_points.size(),3);
//     // double min_beta, max_beta;
//     // min_beta = betas.minCoeff();
//     // max_beta = betas.maxCoeff();
//     // // min_z=-6.5;
//     // // max_z=-4;
//     // std::cout << "min max beta is " << min_beta << " " << max_beta << '\n';
//     // for (size_t i = 0; i < mesh.C.rows(); i++) {
//     //     float gray_val = lerp(betas(i), min_beta, max_beta, 0.0, 1.0 );
//     //     mesh.C(i,0)=mesh.C(i,1)=mesh.C(i,2)=gray_val;
//     // }
//
//     // // //make also some colors based on the sigma value
//     // mesh.C.resize(immature_points.size(),3);
//     // double min_sigma, max_sigma;
//     // min_sigma = sigmas.minCoeff();
//     // max_sigma = sigmas.maxCoeff();
//     // max_sigma=0.05;
//     // // min_z=-6.5;
//     // // max_z=-4;
//     // std::cout << "min max sigma is " << min_sigma << " " << max_sigma << '\n';
//     // for (size_t i = 0; i < mesh.C.rows(); i++) {
//     //     float gray_val = lerp(sigmas(i), min_sigma, max_sigma, 0.0, 1.0 );
//     //     mesh.C(i,0)=mesh.C(i,1)=mesh.C(i,2)=gray_val;
//     // }
//
//     // // //make also some colors based on the last_error value
//     // mesh.C.resize(immature_points.size(),3);
//     // double min_error, max_error;
//     // min_error = last_errors.minCoeff();
//     // max_error = last_errors.maxCoeff();
//     // max_error=800;
//     // // min_z=-6.5;
//     // // max_z=-4;
//     // std::cout << "min max error is " << min_error << " " << max_error << '\n';
//     // for (size_t i = 0; i < mesh.C.rows(); i++) {
//     //     float gray_val = lerp(last_errors(i), min_error, max_error, 0.0, 1.0 );
//     //     mesh.C(i,0)=mesh.C(i,1)=mesh.C(i,2)=gray_val;
//     // }
//
//
//     return mesh;
// }
//
//
//
// void DepthEstimatorGL::compute_depth_and_create_mesh_ICL_incremental(const Frame& frame_left, const Frame& frame_right){
//
//     std::cout << "RECEIVED FRAME " << '\n';
//     std::cout << "frame_right.frame_idx " << frame_right.frame_idx << '\n';
//
//     if(frame_right.frame_idx%50==0){
//         ref_frame=frame_right;
//         m_immature_points=create_immature_points(ref_frame);
//         std::cout << "immature_points size is " << m_immature_points.size() << '\n';
//     }else{
//         glUseProgram(m_compute_trace_seeds_icl_prog_id);
//
//         Frame cur_frame=frame_right;
//         // Frame cur_frame=m_frames[frame_right.frame_idx];
//
//
//         std::cout << "cur_frame.frame_idx " << cur_frame.frame_idx << '\n';
//         std::cout << "cur_frame.tf_cam_world \n" << cur_frame.tf_cam_world.matrix() << '\n';
//         std::cout << "cur_frame.K \n" << cur_frame.K << '\n';
//         std::cout << "frame_right.frame_idx " << frame_right.frame_idx << '\n';
//         std::cout << "frame_right.tf_cam_world \n" << frame_right.tf_cam_world.matrix() << '\n';
//         std::cout << "frame_right.K \n" << frame_right.K << '\n';
//
//         //upload to gpu the inmature points
//         TIME_START_GL("upload_immature_points");
//         glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_points_gl_buf);
//         glBufferData(GL_SHADER_STORAGE_BUFFER, m_immature_points.size() * sizeof(Seed), m_immature_points.data(), GL_DYNAMIC_COPY);
//         TIME_END_GL("upload_immature_points");
//
//         TIME_START_GL("update_depth");
//
//         TIME_START_GL("estimate_affine");
//         const Eigen::Affine3f tf_cur_host_eigen = cur_frame.tf_cam_world * ref_frame.tf_cam_world.inverse();
//         const Eigen::Affine3f tf_host_cur_eigen = tf_cur_host_eigen.inverse();
//         const Eigen::Matrix3f KRKi_cr_eigen = cur_frame.K * tf_cur_host_eigen.linear() * ref_frame.K.inverse();
//         const Eigen::Vector3f Kt_cr_eigen = cur_frame.K * tf_cur_host_eigen.translation();
//         // const Eigen::Vector2f affine_cr_eigen = estimate_affine( immature_points, frames[i], KRKi_cr_eigen, Kt_cr_eigen);
//         const Eigen::Vector2f affine_cr_eigen= Eigen::Vector2f(1,0);
//         const double focal_length = fabs(cur_frame.K(0,0));
//         double px_noise = 1.0;
//         double px_error_angle = atan(px_noise/(2.0*focal_length))*2.0; // law of chord (sehnensatz)
//         Pattern pattern_rot=m_pattern.get_rotated_pattern( KRKi_cr_eigen.topLeftCorner<2,2>() );
//
//         // std::cout << "tf_cur_host_eigen is \n" << tf_cur_host_eigen.matrix() << '\n';
//         // std::cout << "tf_host_cur_eigen is \n " << tf_host_cur_eigen.matrix() << '\n';
//         // std::cout << "KRKi_cr_eigen is \n" << KRKi_cr_eigen << '\n';
//         // std::cout << "Kt_cr_eigen is " << Kt_cr_eigen.transpose() << '\n';
//
//         // std::cout << "pattern_rot has nr of points " << pattern_rot.get_nr_points() << '\n';
//         // for (size_t i = 0; i < pattern_rot.get_nr_points(); i++) {
//         //     std::cout << "offset for i " << i << " is " << pattern_rot.get_offset(i).transpose() << '\n';
//         // }
//
//         TIME_END_GL("estimate_affine");
//
//         TIME_START_GL("upload_params");
//         //upload params (https://hub.packtpub.com/opengl-40-using-uniform-blocks-and-uniform-buffer-objects/)
//         glBindBuffer( GL_UNIFORM_BUFFER, m_ubo_params );
//         glBufferData( GL_UNIFORM_BUFFER, sizeof(m_params), &m_params, GL_DYNAMIC_DRAW );
//         glBindBufferBase( GL_UNIFORM_BUFFER,  glGetUniformBlockIndex(m_compute_trace_seeds_icl_prog_id,"params_block"), m_ubo_params );
//         TIME_END_GL("upload_params");
//
//
//         //attempt 3 upload the image as a inmutable storage but also pack the gradient into the 2nd and 3rd channel
//         TIME_START_GL("upload_gray_img");
//         //merge all mats into one with 4 channel
//         std::vector<cv::Mat> channels;
//         channels.push_back(cur_frame.gray);
//         channels.push_back(cur_frame.grad_x);
//         channels.push_back(cur_frame.grad_y);
//         channels.push_back(cur_frame.grad_y); //dummy one stored in the alpha channels just to have a 4 channel texture
//         cv::Mat img_with_gradients;
//         cv::merge(channels, img_with_gradients);
//
//         int size_bytes=img_with_gradients.step[0] * img_with_gradients.rows; //allocate 4 channels because gpu likes multiples of 4
//         if(!m_cur_frame.get_tex_storage_initialized()){
//             m_cur_frame.allocate_tex_storage_inmutable(GL_RGBA32F,img_with_gradients.cols, img_with_gradients.rows);
//         }
//         m_cur_frame.upload_without_pbo(0,0,0, img_with_gradients.cols, img_with_gradients.rows, GL_RGBA, GL_FLOAT, img_with_gradients.ptr());
//         TIME_END_GL("upload_gray_img");
//
//
//
//
//
//         //upload the matrices
//         TIME_START_GL("upload_matrices");
//         Eigen::Vector2f frame_size;
//         frame_size<< cur_frame.gray.cols, cur_frame.gray.rows;
//         glUniform2fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"frame_size"), 1, frame_size.data());
//         // glUniformMatrix4fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"tf_cur_host"), 1, GL_FALSE, tf_cur_host_eigen_trans.data());
//         // glUniformMatrix4fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"tf_host_cur"), 1, GL_FALSE, tf_host_cur_eigen_trans.data());
//         glUniformMatrix4fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"tf_cur_host"), 1, GL_TRUE, tf_cur_host_eigen.data());
//         glUniformMatrix4fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"tf_host_cur"), 1, GL_TRUE, tf_host_cur_eigen.data());
//         glUniformMatrix3fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"K"), 1, GL_FALSE, cur_frame.K.data());
//         glUniformMatrix3fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"KRKi_cr"), 1, GL_FALSE, KRKi_cr_eigen.data());
//         glUniform3fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"Kt_cr"), 1, Kt_cr_eigen.data());
//         glUniform2fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"affine_cr"), 1, affine_cr_eigen.data());
//         glUniform1f(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"px_error_angle"), px_error_angle);
//         glUniform2fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"pattern_rot_offsets"), pattern_rot.get_nr_points(), pattern_rot.get_offset_matrix().data()); //upload all the offses as an array of vec2 offsets
//         // std::cout << "setting nr of points to " <<  pattern_rot.get_nr_points() << '\n';
//         // std::cout << "the uniform location is " << glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"pattern_rot_nr_points") << '\n';
//         glUniform1i(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"pattern_rot_nr_points"), pattern_rot.get_nr_points());
//         TIME_END_GL("upload_matrices");
//         glMemoryBarrier(GL_ALL_BARRIER_BITS);
//
//
//         // tf_cur_host, tf_host_cur, KRKi_cr, Kt_cr, affine_cr, px_error_angle
//         TIME_START_GL("depth_update_kernel");
//         glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_points_gl_buf);
//         bind_for_sampling(m_cur_frame, 1, glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"gray_img_sampler") );
//         glDispatchCompute(m_immature_points.size()/256, 1, 1); //TODO adapt the local size to better suit the gpu
//         glMemoryBarrier(GL_ALL_BARRIER_BITS);
//         TIME_END_GL("depth_update_kernel");
//
//         TIME_END_GL("update_depth");
//
//         //read the points back to cpu
//         glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_points_gl_buf);
//         Seed* ptr = (Seed*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
//         for (size_t i = 0; i < m_immature_points.size(); i++) {
//             m_immature_points[i]=ptr[i];
//         }
//         glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
//
//         m_mesh=create_mesh_ICL(m_immature_points, m_frames);
//     }
//
//
//
// }
//


//CLEANED UP versions:

void DepthEstimatorGL::compute_depth_and_update_mesh(const Frame& frame){

    std::cout << "RECEIVED FRAME " << '\n';
    if(frame.frame_idx%50==0){
       m_ref_frame=frame;
       m_seeds=create_seeds(frame);

    }else{
        //trace the created seeds
        trace(m_seeds, m_ref_frame, frame);
        //create a mesh
        m_mesh=create_mesh(m_seeds, m_ref_frame);
    }
}

void DepthEstimatorGL::compute_depth_and_update_mesh_stereo(const Frame& frame_left, const Frame& frame_right){

    std::cout << "RECEIVED FRAME " << '\n';
    if(frame_left.frame_idx%50==0){
        m_last_ref_frame=m_ref_frame;
        m_ref_frame=frame_left;
        m_seeds=create_seeds(frame_left);
        // m_seeds=create_seeds_gpu(frame_left);
        // if(frame_left.frame_idx==0){ //because for some reason the first frame fails to create seeds on gpu...
        //     m_seeds=create_seeds_gpu(frame_left);
        // }
        // assign_neighbours_for_points(m_seeds, m_ref_frame.gray.cols, m_ref_frame.gray.rows);
        m_started_new_keyframe=true;
        m_last_finished_mesh=m_mesh;
    }else{
        m_started_new_keyframe=false;
        //trace the created seeds
        trace(m_seeds, m_ref_frame, frame_right);
        //create a mesh

        // //next one we will create a keyframe
        // if( (frame_left.frame_idx+1) %50==0){
        //     assign_neighbours_for_points(m_seeds, m_ref_frame.gray.cols, m_ref_frame.gray.rows);
        //     denoise_cpu(m_seeds, m_ref_frame.gray.cols, m_ref_frame.gray.rows);
        // }

        //only after 10 frames from the keyframe because by then we should have a good estimate of the depth
        if(frame_left.frame_idx%50>20){
            // denoise_cpu(m_seeds, 5,  m_ref_frame.gray.cols, m_ref_frame.gray.rows);
        }

        // //next one we will create a keyframe
        if( (frame_left.frame_idx+1) %50==0){
            remove_grazing_seeds(m_seeds);
            // we will create a new keyframe but before that, do a trace on the previous keyframe
            if(!m_last_ref_frame.gray.empty()){ //if it's the first keyframe then the last one will be empty
                trace(m_seeds, m_ref_frame, m_last_ref_frame);
            }

            // denoise_cpu(m_seeds, 200,  m_ref_frame.gray.cols, m_ref_frame.gray.rows);
        }


        m_mesh=create_mesh(m_seeds, m_ref_frame);
    }
}

std::vector<Seed> DepthEstimatorGL::create_seeds(const Frame& frame){
    TIME_START_GL("create_seeds");
    std::vector<Seed> seeds;
    for (size_t i = 10; i < frame.gray.rows-10; i++) {  //--------Do not look around the borders to avoid pattern accesing outside img
        for (size_t j = 10; j < frame.gray.cols-10; j++) {

            //check if this point has enough determinant in the hessian
            Eigen::Matrix2f gradient_hessian;
            gradient_hessian.setZero();
            for (size_t p = 0; p < m_pattern.get_nr_points(); p++) {
                int dx = m_pattern.get_offset_x(p);
                int dy = m_pattern.get_offset_y(p);

                float gradient_x=frame.grad_x.at<float>(i+dy,j+dx);
                float gradient_y=frame.grad_y.at<float>(i+dy,j+dx);

                Eigen::Vector2f grad;
                grad << gradient_x, gradient_y;

                gradient_hessian+= grad*grad.transpose();
            }

            //determinant is high enough, add the point
            float hessian_det=gradient_hessian.determinant();
            float trace=gradient_hessian.trace();
            if(trace > 17){
            // if(hessian_det > 0){
                Seed point;
                point.m_uv << j,i;
                point.m_gradH=gradient_hessian;
                point.m_time_alive=0;
                point.m_nr_times_visible=0;

                //Seed::Seed
                Eigen::Vector3f f_eigen = (frame.K.inverse() * Eigen::Vector3f(point.m_uv.x(),point.m_uv.y(),1)).normalized();
                point.depth_filter.m_f=Eigen::Vector4f(f_eigen(0),f_eigen(1),f_eigen(2), 1.0);

                float mean_starting_depth=m_mean_starting_depth;
                float min_starting_depth=0.1;
                point.depth_filter.m_mu = (1.0/mean_starting_depth);
                point.depth_filter.m_z_range = (1.0/min_starting_depth);
                point.depth_filter.m_sigma2 = (point.depth_filter.m_z_range*point.depth_filter.m_z_range/36);

                float z_inv_min = point.depth_filter.m_mu + sqrt(point.depth_filter.m_sigma2);
                float z_inv_max = std::max<float>(point.depth_filter.m_mu- sqrt(point.depth_filter.m_sigma2), 0.00000001f);
                point.m_idepth_minmax <<  z_inv_min, z_inv_max;

                point.depth_filter.m_alpha=10.0;
                point.depth_filter.m_beta=10.0;



                //get data for the color of that point (depth_point->ImmaturePoint::ImmaturePoint)---------------------
                for(int p_idx=0;p_idx<m_pattern.get_nr_points();p_idx++){
                    Eigen::Vector2f offset = m_pattern.get_offset(p_idx);

                    // point.m_intensity[p_idx]=texture_interpolate(frame.gray, point.m_uv.x()+offset(0), point.m_uv.y()+offset(1), InterpolType::NEAREST);
                    //
                    // //for ngf
                    // float grad_x_val=texture_interpolate(frame.grad_x,  point.m_uv.x()+offset(0), point.m_uv.y()+offset(1), InterpolType::NEAREST);
                    // float grad_y_val=texture_interpolate(frame.grad_y,  point.m_uv.x()+offset(0), point.m_uv.y()+offset(1), InterpolType::NEAREST);

                    point.m_intensity[p_idx]=frame.gray.at<float>( point.m_uv.y()+offset(1), point.m_uv.x()+offset(0) );

                    //for ngf
                    float grad_x_val=frame.grad_x.at<float>( point.m_uv.y()+offset(1), point.m_uv.x()+offset(0) );
                    float grad_y_val=frame.grad_y.at<float>( point.m_uv.y()+offset(1), point.m_uv.x()+offset(0) );

                    point.m_normalized_grad[p_idx] << grad_x_val, grad_y_val;
                    point.m_normalized_grad[p_idx] /= sqrt(point.m_normalized_grad[p_idx].squaredNorm() + m_params.eta);
                    if(point.m_normalized_grad[p_idx].norm()<1e-3){
                        point.m_zero_grad[p_idx]=1;
                    }else{
                        point.m_zero_grad[p_idx]=0;
                        point.m_active_pattern_points++;
                    }

                }
                point.m_energyTH = m_pattern.get_nr_points()*m_params.residualTH;
                // point.m_energyTH *= m_params.overallEnergyTHWeight*m_params.overallEnergyTHWeight;

                point.m_last_error = -1;
                point.depth_filter.m_is_outlier = 0;


                //as the neighbours indices set the current points so if we access it we get this point
                int seed_idx=seeds.size();
                point.left = seed_idx;
                point.right = seed_idx;
                point.above = seed_idx;
                point.below = seed_idx;
                point.left_upper = seed_idx;
                point.right_upper = seed_idx;
                point.left_lower = seed_idx;
                point.right_lower = seed_idx;

                seeds.push_back(point);
            }

        }
    }

    TIME_END_GL("create_seeds");
    std::cout << "seeds size is " << seeds.size() << '\n';
    return seeds;

}

std::vector<Seed> DepthEstimatorGL::create_seeds_gpu (const Frame& frame){


    // std::vector<Seed> seeds;
    //
    //
    // TIME_START_GL("hessian_det_matrix");
    // //if the m_hessian_pointwise_tex is not initialize allocate memory for it
    // if(!m_hessian_pointwise_tex.get_tex_storage_initialized()){
    //     m_hessian_pointwise_tex.allocate_tex_storage_inmutable(GL_RGBA32F,frame.gray.cols, frame.gray.rows);
    // }
    // GL_C( glUseProgram(m_compute_hessian_pointwise_prog_id) );
    // if(frame.cam_id==0){
    //     GL_C(bind_for_sampling(m_frame_left, 1, glGetUniformLocation(m_compute_hessian_pointwise_prog_id,"gray_with_gradients_img_sampler") ) );
    //
    // }else if(frame.cam_id==1){
    //     GL_C( bind_for_sampling(m_frame_right, 1, glGetUniformLocation(m_compute_hessian_pointwise_prog_id,"gray_with_gradients_img_sampler") ) );
    // }else{
    //     LOG(FATAL) << "Invalid cam id";
    // }
    // GL_C( glBindImageTexture(0, m_hessian_pointwise_tex.get_tex_id(), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F) );
    // GL_C( glDispatchCompute(frame.gray.cols/32, frame.gray.rows/16, 1) ); //TODO need to ceil the size otherwise you will have block of the image that are not computed
    // glMemoryBarrier(GL_ALL_BARRIER_BITS);
    // TIME_END_GL("hessian_det_matrix");
    //
    //
    //
    // //get hessian by box blurring
    // std::cout << "m_compute_hessian_blurred_prog_id " << m_compute_hessian_blurred_prog_id << '\n';
    // TIME_START_GL("hessian_blurred_matrix");
    // //if the m_hessian_pointwise_tex is not initialize allocate memory for it
    // if(!m_hessian_blurred_tex.get_tex_storage_initialized()){
    //     VLOG(1) << "initializing m_hessian_blurred_tex";
    //     GL_C(m_hessian_blurred_tex.allocate_tex_storage_inmutable(GL_RGBA32F,frame.gray.cols, frame.gray.rows) );
    // }
    // GL_C( glUseProgram(m_compute_hessian_blurred_prog_id) );
    // bind_for_sampling(m_hessian_pointwise_tex, 1, glGetUniformLocation(m_compute_hessian_blurred_prog_id,"hessian_pointwise_tex_sampler") );
    // GL_C( glBindImageTexture(0, m_hessian_blurred_tex.get_tex_id(), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F) );
    // GL_C( glDispatchCompute(frame.gray.cols/32, frame.gray.rows/16, 1) ); //TODO need to ceil the size otherwise you will have block of the image that are not computed
    // glMemoryBarrier(GL_ALL_BARRIER_BITS);
    // TIME_END_GL("hessian_blurred_matrix");


    //create seeds by calculating the determinant of the hessian and if it's big enough add the seed to the seed buffer and atomically increment a counter for the nr of seeds

    //upload img
    TIME_START_GL("upload_gray_and_grad");
    int size_bytes=frame.gray_with_gradients.step[0] * frame.gray_with_gradients.rows;
    std::cout << "size bytes is " << size_bytes << '\n';
    GL_C(m_ref_frame_tex.upload_data(GL_RGB32F, frame.gray_with_gradients.cols, frame.gray_with_gradients.rows, GL_RGB, GL_FLOAT, frame.gray_with_gradients.ptr(), size_bytes));
    TIME_END_GL("upload_gray_and_grad");


    TIME_START_GL("create_seeds");
    GL_C( glUseProgram(m_compute_create_seeds_prog_id) );
    //bind seeds buffer and images
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_points_gl_buf);
    bind_for_sampling(m_ref_frame_tex, 1, glGetUniformLocation(m_compute_create_seeds_prog_id,"gray_with_gradients_img_sampler") );


    //zero o out the counter for seeds creatted
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 1, m_atomic_nr_seeds_created);
    GLuint zero=0;
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint), &(zero), GL_STATIC_COPY);

    //debug texture
    if(!m_debug_tex.get_tex_storage_initialized()){
      m_debug_tex.allocate_tex_storage_inmutable(GL_RGBA32F,frame.gray.cols, frame.gray.rows);
    }
    //clear the debug texture
    std::vector<GLuint> clear_color(4,0);
    GL_C ( glClearTexSubImage(m_debug_tex.get_tex_id(), 0,0,0,0, frame.gray.cols,frame.gray.rows,1,GL_RGBA, GL_FLOAT, clear_color.data()) );
    glBindImageTexture(2, m_debug_tex.get_tex_id(), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

    //uniforms needed for creating the seeds
    //upload all the offses as an array of vec2 offsets
    glUniform2fv(glGetUniformLocation(m_compute_create_seeds_prog_id,"pattern_rot_offsets"), m_pattern.get_nr_points(), m_pattern.get_offset_matrix().data());
    glUniform1i(glGetUniformLocation(m_compute_create_seeds_prog_id,"pattern_rot_nr_points"), m_pattern.get_nr_points());
    glUniformMatrix3fv(glGetUniformLocation(m_compute_create_seeds_prog_id,"K"), 1, GL_FALSE, frame.K.data());
    Eigen::Matrix3f K_inv=frame.K.inverse();
    glUniformMatrix3fv(glGetUniformLocation(m_compute_create_seeds_prog_id,"K_inv"), 1, GL_FALSE, K_inv.data());
    glUniform1f(glGetUniformLocation(m_compute_create_seeds_prog_id,"min_starting_depth"), m_min_starting_depth);
    glUniform1f(glGetUniformLocation(m_compute_create_seeds_prog_id,"mean_starting_depth"), m_mean_starting_depth);
    glUniform1i(glGetUniformLocation(m_compute_create_seeds_prog_id,"seeds_start_idx"), 0); //TODO
    glUniform1i(glGetUniformLocation(m_compute_create_seeds_prog_id,"idx_keyframe"), frame.frame_idx);
    //TODO maybe change the 0 in the previous m_keyframes_per_cam to something else because we now assume that if we make a keyframe for left cam we also make for right


    glDispatchCompute(frame.gray.cols/32, frame.gray.rows/16, 1);
    // m_nr_times_frame_used_for_seed_creation_per_cam[frame.cam_id]++;
    glMemoryBarrier(GL_ALL_BARRIER_BITS);

    // get how many seeds were created
    GLuint* atomic_nr_seeds_created_cpu= (GLuint*)glMapBufferRange(GL_ATOMIC_COUNTER_BUFFER,0,sizeof(GLuint),GL_MAP_READ_BIT);
    int nr_seeds_created=atomic_nr_seeds_created_cpu[0]; //need to store it in another buffer because we will unmap this pointer
    glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);
    std::cout << "nr_seeds_created " << nr_seeds_created << '\n';


    //debug read the seeds back to cpu
    std::vector<Seed> seeds(nr_seeds_created);
    std::cout << "cam " << frame.cam_id << '\n';
    // glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_seeds_right_gl_buf);
    Seed* ptr = (Seed*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    for (size_t i = 0; i < nr_seeds_created; i++) {
        // // for (size_t d = 0; d < 16; d++) {
        // //     std::cout << "debug val is " << ptr[i].debug[d] << '\n';
        // // }
        // // std::cout << "uv of seed is " << ptr[i].m_uv.transpose() << '\n';
        // print_seed(ptr[i]);
        seeds[i]=ptr[i];
    }
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    TIME_END_GL("create_seeds");

    return seeds;

}


void DepthEstimatorGL::trace(std::vector<Seed>& seeds,const Frame& ref_frame, const Frame& cur_frame){

    if(seeds.size()==0){
        std::cout << "No seeds, so there will be no tracing" << '\n';
        return;
    }

    std::cout << "Tracing with " << seeds.size() << " seeds" << '\n';

    //upload to gpu the inmature points
    TIME_START_GL("upload_immature_points");
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_points_gl_buf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, seeds.size() * sizeof(Seed), seeds.data(), GL_DYNAMIC_COPY);
    TIME_END_GL("upload_immature_points");


    glUseProgram(m_compute_trace_seeds_icl_prog_id);

    //get matrices between the host and the cur frame
    const Eigen::Affine3f tf_cur_host_eigen = cur_frame.tf_cam_world * ref_frame.tf_cam_world.inverse();
    const Eigen::Affine3f tf_host_cur_eigen = tf_cur_host_eigen.inverse();
    const Eigen::Matrix3f KRKi_cr_eigen = cur_frame.K * tf_cur_host_eigen.linear() * ref_frame.K.inverse();
    const Eigen::Vector3f Kt_cr_eigen = cur_frame.K * tf_cur_host_eigen.translation();
    const double focal_length = fabs(cur_frame.K(0,0));
    double px_noise = 1.0;
    double px_error_angle = atan(px_noise/(2.0*focal_length))*2.0; // law of chord (sehnensatz)
    Pattern pattern_rot=m_pattern.get_rotated_pattern( KRKi_cr_eigen.topLeftCorner<2,2>() );


    TIME_START_GL("upload_params");
    //upload params (https://hub.packtpub.com/opengl-40-using-uniform-blocks-and-uniform-buffer-objects/)
    glBindBuffer( GL_UNIFORM_BUFFER, m_ubo_params );
    glBufferData( GL_UNIFORM_BUFFER, sizeof(m_params), &m_params, GL_DYNAMIC_DRAW );
    glBindBufferBase( GL_UNIFORM_BUFFER,  glGetUniformBlockIndex(m_compute_trace_seeds_icl_prog_id,"params_block"), m_ubo_params );
    TIME_END_GL("upload_params");


    TIME_START_GL("upload_gray_img");
    //merge all mats into one with 4 channel
    std::vector<cv::Mat> channels;
    channels.push_back(cur_frame.gray);
    channels.push_back(cur_frame.grad_x);
    channels.push_back(cur_frame.grad_y);
    channels.push_back(cur_frame.grad_y); //dummy one stored in the alpha channels just to have a 4 channel texture
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
    frame_size<< cur_frame.gray.cols, cur_frame.gray.rows;
    glUniform2fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"frame_size"), 1, frame_size.data());
    glUniformMatrix4fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"tf_cur_host"), 1, GL_TRUE, tf_cur_host_eigen.data());
    glUniformMatrix4fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"tf_host_cur"), 1, GL_TRUE, tf_host_cur_eigen.data());
    glUniformMatrix3fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"K"), 1, GL_FALSE, cur_frame.K.data());
    glUniformMatrix3fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"KRKi_cr"), 1, GL_FALSE, KRKi_cr_eigen.data());
    glUniform3fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"Kt_cr"), 1, Kt_cr_eigen.data());
    glUniform1f(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"px_error_angle"), px_error_angle);
    glUniform2fv(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"pattern_rot_offsets"), pattern_rot.get_nr_points(), pattern_rot.get_offset_matrix().data()); //upload all the offses as an array of vec2 offsets
    glUniform1i(glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"pattern_rot_nr_points"), pattern_rot.get_nr_points());
    TIME_END_GL("upload_matrices");
    glMemoryBarrier(GL_ALL_BARRIER_BITS);


    //debug texture
    if(!m_debug_tex.get_tex_storage_initialized()){
        GL_C(m_debug_tex.allocate_tex_storage_inmutable(GL_RGBA32F,cur_frame.gray.cols, cur_frame.gray.rows) );
    }
    //clear the debug texture
    std::vector<GLuint> clear_color(4,0);
    GL_C ( glClearTexSubImage(m_debug_tex.get_tex_id(), 0,0,0,0, cur_frame.gray.cols,cur_frame.gray.rows,1,GL_RGBA, GL_FLOAT, clear_color.data()) );
    glBindImageTexture(2, m_debug_tex.get_tex_id(), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);



    //Start tracing
    TIME_START_GL("depth_update_kernel");
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_points_gl_buf);
    bind_for_sampling(m_cur_frame, 1, glGetUniformLocation(m_compute_trace_seeds_icl_prog_id,"gray_img_sampler") );
    glDispatchCompute(seeds.size()/256, 1, 1); //TODO adapt the local size to better suit the gpu
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
    TIME_END_GL("depth_update_kernel");


    //read the points back to cpu
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_points_gl_buf);
    Seed* ptr = (Seed*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    for (size_t i = 0; i < seeds.size(); i++) {
        seeds[i]=ptr[i];
    }
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

}

Mesh DepthEstimatorGL::create_mesh(const std::vector<Seed>& seeds, Frame& ref_frame){
    Mesh mesh;
    mesh.V.resize(seeds.size(),3);
    mesh.V.setZero();

    for (size_t i = 0; i < seeds.size(); i++) {
        float u=seeds[i].m_uv.x();
        float v=seeds[i].m_uv.y();
        // float depth=immature_points[i].gt_depth;
        // float depth=1.0;
        float depth=1/seeds[i].depth_filter.m_mu;

        if(std::isfinite(seeds[i].depth_filter.m_mu) && seeds[i].depth_filter.m_mu>=0.1
            && seeds[i].depth_filter.m_is_outlier==0 ){

            //backproject the immature point
            Eigen::Vector3f point_screen;
            point_screen << u, v, 1.0;
            Eigen::Vector3f point_dir=ref_frame.K.inverse()*point_screen;
            Eigen::Vector3f point_cam = point_dir*depth;
            // point_cam(2)=-point_cam(2);
            // point_cam(1)=-point_cam(1);
            // point_cam(2)=-point_cam(2); //flip the depth because opengl 7has a camera which looks at the negative z axis (therefore, more depth means a more negative number)
            Eigen::Vector3f point_world=ref_frame.tf_cam_world.inverse()*point_cam;
            // mesh.V.row(i)=point_cam.cast<double>();
            mesh.V.row(i)=point_world.cast<double>();
        }


    }

    // //make also some colors based on depth
    // mesh.C.resize(seeds.size(),3);
    // double min_z, max_z;
    // min_z = mesh.V.col(2).minCoeff();
    // max_z = mesh.V.col(2).maxCoeff();
    // min_z=-5.5;
    // max_z=-2;
    // std::cout << "min max z is " << min_z << " " << max_z << '\n';
    // for (size_t i = 0; i < mesh.C.rows(); i++) {
    //     float gray_val = lerp(mesh.V(i,2), min_z, max_z, 0.0, 1.0 );
    //     mesh.C(i,0)=mesh.C(i,1)=mesh.C(i,2)=gray_val;
    // }

    //make colors from the intensity value stored in the seeds
    mesh.C.resize(seeds.size(),3);
    for (size_t i = 0; i < mesh.C.rows(); i++) {
        float gray_val = seeds[i].m_intensity[5]; //center points of the pattern;
        mesh.C(i,0)=mesh.C(i,1)=mesh.C(i,2)=gray_val;
    }


    return mesh;
}



void DepthEstimatorGL::assign_neighbours_for_points(std::vector<Seed>& seeds, const int frame_width, const int frame_height){
    //TODO this works for the reference frame because we know there will not be overlaps but for any other frames we would need to just reproject the points into the frame and then get the one with the smallest depth in case they lie in the same pixel. Also it would need to be done after updating their depth of course.
    //another way to deal with it is to only make the neighbours for their respective host frame, so we would need to pass a parameter to this function that makes it that we only create neighbours for points that have a specific idx_host_frame


    //TODO create it of a size of frame and initialize to -1
    Eigen::MatrixXi texture_indices(frame_height,frame_width);
    texture_indices.setConstant(-1);

    for (size_t i = 0; i < seeds.size(); i++) {
        int u=seeds[i].m_uv.x();
        int v=seeds[i].m_uv.y();
        texture_indices(v,u)=i;
    }


    //go through the immature points again and assign the neighbours
    for (size_t i = 0; i < seeds.size(); i++) {
        int u=seeds[i].m_uv.x();
        int v=seeds[i].m_uv.y();

        //left
        int point_left=texture_indices(v,u-1);
        // std::cout << "point left is " << point_left << '\n';
        if(point_left!=-1){ seeds[i].left=point_left; }

        //right
        int point_right=texture_indices(v,u+1);
        if(point_right!=-1){ seeds[i].right=point_right; }

        //up
        int point_up=texture_indices(v+1,u);
        if(point_up!=-1){ seeds[i].above=point_up; }

        //down
        int point_down=texture_indices(v-1,u);
        if(point_down!=-1){ seeds[i].below=point_down; }

        //left_upper
        int point_left_up=texture_indices(v+1,u-1);
        if(point_left_up!=-1){ seeds[i].left_upper=point_left_up; }

        //righ_upper
        int point_right_up=texture_indices(v+1,u+1);
        if(point_right_up!=-1){ seeds[i].right_upper=point_right_up; }

        //left_lower
        int point_left_down=texture_indices(v-1,u-1);
        if(point_left_down!=-1){ seeds[i].left_lower=point_left_down; }

        //right_lower
        int point_right_down=texture_indices(v-1,u+1);
        if(point_right_down!=-1){ seeds[i].right_lower=point_right_down; }

    }

}

void DepthEstimatorGL::denoise_cpu( std::vector<Seed>& seeds, const int iters, const int frame_width, const int frame_height){

    int depth_range=5;
    float lambda=0.5;
    int iterations=iters;

    std::cout << "starting to denoise." << std::endl;
    const float large_sigma2 = depth_range * depth_range / 72.f;

    // computeWeightsAndMu( )
    for ( auto &point : seeds){
        const float E_pi = point.depth_filter.m_alpha / ( point.depth_filter.m_alpha + point.depth_filter.m_beta);

        point.depth_filter.m_g = std::max<float> ( (E_pi * point.depth_filter.m_sigma2 + (1.0f-E_pi) * large_sigma2) / large_sigma2, 1.0f );
        point.depth_filter.m_mu_denoised = point.depth_filter.m_mu;
        point.depth_filter.m_mu_head = point.m_uv.x();
        point.depth_filter.m_p.setZero();
    }


    const float L = sqrt(8.0f);
    const float tau = (0.02f);
    const float sigma = ((1 / (L*L)) / tau);
    const float theta = 0.5f;

    for (size_t i = 0; i < iterations; i++) {
        // std::cout << "iter " << i << '\n';

        int point_idx=0;
        // update dual
        for ( auto &point : seeds ){
            // std::cout << "update point " << point_idx << '\n';
            point_idx++;
            const float g = point.depth_filter.m_g;
            const Eigen::Vector2f p = point.depth_filter.m_p;
            Eigen::Vector2f grad_uhead = Eigen::Vector2f::Zero();
            const float current_u = point.depth_filter.m_mu_denoised;


            Seed & right = (seeds[point.right].depth_filter.m_is_outlier==1) ? point : seeds[point.right];
            Seed & below = (seeds[point.below].depth_filter.m_is_outlier==1) ? point : seeds[point.below];

            // if(point.right != -1){
            //     std::cout << "------------" << '\n';
            //     std::cout << "point is " << point.u << " " << point.v << '\n';
            //     std::cout << "right is " << right.u << " " << right.v << '\n';
            //     std::cout << "point.right is " << point.right << '\n';
            // }


            grad_uhead[0] = right.depth_filter.m_mu_head - current_u; //->atXY(min<int>(c_img_size.width-1, x+1), y)  - current_u;
            grad_uhead[1] = below.depth_filter.m_mu_head - current_u; //->atXY(x, min<int>(c_img_size.height-1, y+1)) - current_u;
            const Eigen::Vector2f temp_p = g * grad_uhead * sigma + p;
            const float sqrt_p = temp_p.norm(); //sqrt(temp_p[0] * temp_p[0] + temp_p[1] * temp_p[1]);
            point.depth_filter.m_p = temp_p / std::max<float>(1.0f, sqrt_p);
        }

        // std::cout << " update primal" << '\n';
        // update primal:
        for ( auto &point : seeds ){
            //debug
            // std::cout << "point left is " << point.left << '\n';

            const float noisy_depth = point.depth_filter.m_mu;
            const float old_u = point.depth_filter.m_mu_denoised;
            const float g = point.depth_filter.m_g;

            Eigen::Vector2f current_p = point.depth_filter.m_p;
            Seed & left = (seeds[point.left].depth_filter.m_is_outlier==1) ? point : seeds[point.left];
            Seed & above = (seeds[point.above].depth_filter.m_is_outlier==1) ? point : seeds[point.above];
            Eigen::Vector2f w_p = left.depth_filter.m_p;
            Eigen::Vector2f n_p = above.depth_filter.m_p;

            const int x = point.m_uv.x();
            const int y = point.m_uv.y();
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
                point.depth_filter.m_mu_denoised = temp_u - tauLambda;
            }
            else if ((temp_u - noisy_depth) < (-tauLambda))
            {
                point.depth_filter.m_mu_denoised = temp_u + tauLambda;
            }
            else
            {
                point.depth_filter.m_mu_denoised = noisy_depth;
            }
            point.depth_filter.m_mu_head = point.depth_filter.m_mu_denoised + theta * (point.depth_filter.m_mu_denoised - old_u);
        }
    }


    for (auto &point : seeds) {
        // std::cout << "changin mu depth from " << point.mu  << " to " << point.mu_denoised << '\n';
        point.depth_filter.m_mu=point.depth_filter.m_mu_denoised;
    }
}


void DepthEstimatorGL::remove_grazing_seeds ( std::vector<Seed>& seeds ){

    //for each seed check the neighbouring ones, if the depth of them is higher than a certain theshold set them as outliers because it most likely a lonely point

    //TODO store in point.left the index to the point itself and not a -1. Therefore we can just index the points without using an if

    int nr_points_removed=0;
    for ( auto &point : seeds ){
        // std::cout << "update point " << point_idx << '\n';

        //if the min and max between the neighbours is too high set them as outliers

        float min_mu=std::numeric_limits<float>::max();
        float max_mu=std::numeric_limits<float>::min();

        int nr_neighbours=0;


        if(point.right==-1){
            std::cout << "what--------------------------------------------------" << '\n';
        }

        Seed & right=seeds[point.right];
        Seed & left=seeds[point.left];
        Seed & below=seeds[point.below];
        Seed & above=seeds[point.above];
        Seed & left_upper=seeds[point.left_upper];
        Seed & right_upper=seeds[point.right_upper];
        Seed & left_lower=seeds[point.left_lower];
        Seed & right_lower=seeds[point.right_lower];
        // if(point.right != -1){
        //     right=seeds[point.right];
        //     nr_neighbours++;
        // }
        // if(point.left != -1){
        //     left=seeds[point.left];
        //     nr_neighbours++;
        // }
        // if(point.below != -1){
        //     below=seeds[point.below];
        //     nr_neighbours++;
        // }
        // if(point.above != -1){
        //     above=seeds[point.above];
        //     nr_neighbours++;
        // }
        // if(point.left_upper != -1){
        //     left_upper=seeds[point.left_upper];
        //     nr_neighbours++;
        // }
        // if(point.right_upper != -1){
        //     right_upper=seeds[point.right_upper];
        //     nr_neighbours++;
        // }
        // if(point.left_lower != -1){
        //     left_lower=seeds[point.left_lower];
        //     nr_neighbours++;
        // }
        // if(point.right_lower != -1){
        //     right_lower=seeds[point.right_lower];
        //     nr_neighbours++;
        // }



        // min max
        if(right.depth_filter.m_mu<min_mu){
            min_mu=right.depth_filter.m_mu;
        }
        if(right.depth_filter.m_mu>max_mu){
            max_mu=right.depth_filter.m_mu;
        }

        if(left.depth_filter.m_mu<min_mu){
            min_mu=left.depth_filter.m_mu;
        }
        if(left.depth_filter.m_mu>max_mu){
            max_mu=left.depth_filter.m_mu;
        }


        if(below.depth_filter.m_mu<min_mu){
            min_mu=below.depth_filter.m_mu;
        }
        if(below.depth_filter.m_mu>max_mu){
            max_mu=below.depth_filter.m_mu;
        }


        if(above.depth_filter.m_mu<min_mu){
            min_mu=above.depth_filter.m_mu;
        }
        if(above.depth_filter.m_mu>max_mu){
            max_mu=above.depth_filter.m_mu;
        }


        //diagonals
        if(left_upper.depth_filter.m_mu<min_mu){
            min_mu=left_upper.depth_filter.m_mu;
        }
        if(left_upper.depth_filter.m_mu>max_mu){
            max_mu=left_upper.depth_filter.m_mu;
        }

        if(right_upper.depth_filter.m_mu<min_mu){
            min_mu=right_upper.depth_filter.m_mu;
        }
        if(right_upper.depth_filter.m_mu>max_mu){
            max_mu=right_upper.depth_filter.m_mu;
        }


        if(left_lower.depth_filter.m_mu<min_mu){
            min_mu=left_lower.depth_filter.m_mu;
        }
        if(left_lower.depth_filter.m_mu>max_mu){
            max_mu=left_lower.depth_filter.m_mu;
        }


        if(right_lower.depth_filter.m_mu<min_mu){
            min_mu=right_lower.depth_filter.m_mu;
        }
        if(right_lower.depth_filter.m_mu>max_mu){
            max_mu=right_lower.depth_filter.m_mu;
        }


        if(max_mu-min_mu>0.05){
            point.depth_filter.m_is_outlier=1;

            // right.depth_filter.m_is_outlier=1;
            // left.depth_filter.m_is_outlier=1;
            // above.depth_filter.m_is_outlier=1;
            // below.depth_filter.m_is_outlier=1;
            //
            // left_upper.depth_filter.m_is_outlier=1;
            // right_upper.depth_filter.m_is_outlier=1;
            // left_lower.depth_filter.m_is_outlier=1;
            // right_lower.depth_filter.m_is_outlier=1;

            nr_points_removed++;
        }

        // //points if left lonely
        // if(nr_points_removed>=nr_neighbours){
        //     point.depth_filter.m_is_outlier=1;
        //     nr_points_removed++;
        // }

        // std::cout << "max vmin " << max_mu-min_mu << '\n';

        // std::cout << "nr nrihbs" << nr_neighbours << '\n';




    }






    // if all neighbours are outliers this points is also an outlier
    int removed_lonely=0;
    for ( auto &point : seeds ){
        Seed & right=seeds[point.right];
        Seed & left=seeds[point.left];
        Seed & below=seeds[point.below];
        Seed & above=seeds[point.above];
        Seed & left_upper=seeds[point.left_upper];
        Seed & right_upper=seeds[point.right_upper];
        Seed & left_lower=seeds[point.left_lower];
        Seed & right_lower=seeds[point.right_lower];

        int nr_neighbours_outliers=0;
        if(left.depth_filter.m_is_outlier==1) nr_neighbours_outliers++;
        if(right.depth_filter.m_is_outlier==1) nr_neighbours_outliers++;
        if(above.depth_filter.m_is_outlier==1) nr_neighbours_outliers++;
        if(below.depth_filter.m_is_outlier==1) nr_neighbours_outliers++;
        if(left_upper.depth_filter.m_is_outlier==1) nr_neighbours_outliers++;
        if(right_upper.depth_filter.m_is_outlier==1) nr_neighbours_outliers++;
        if(left_lower.depth_filter.m_is_outlier==1) nr_neighbours_outliers++;
        if(right_lower.depth_filter.m_is_outlier==1) nr_neighbours_outliers++;

        if(nr_neighbours_outliers>=7){
            point.depth_filter.m_is_outlier=1;
            removed_lonely++;
        }

    }


    std::cout << "--------------------------------nr_points_removed " << nr_points_removed << '\n';
    std::cout << "--------------------------------nr_lonely " << removed_lonely << '\n';


}
