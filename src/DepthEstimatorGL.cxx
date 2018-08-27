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
        m_start_frame(0),
        m_started_new_keyframe(false),
        m_seeds_gpu_dirty(false),
        m_seeds_cpu_dirty(false)
        {

    init_params();
    init_opengl();
    m_pattern.init_pattern( (fs::path(CMAKE_SOURCE_DIR)/"data"/m_pattern_file).string() );

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
    // std::cout << "size of EpiData " << sizeof(EpiData) << '\n';
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



}

void DepthEstimatorGL::init_opengl(){
    std::cout << "init opengl" << '\n';

    // glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE);
    // glClampColor(GL_CLAMP_VERTEX_COLOR, GL_FALSE);
    // glClampColor(GL_CLAMP_FRAGMENT_COLOR, GL_FALSE);

    print_supported_extensions();


    if(GL_ARB_debug_output){
    	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB);
    	glDebugMessageCallbackARB(debug_func, (void*)15);
	 }

    // if(GL_ARB_texture_storage){
    //   std::cout << "we have GL_ARB_texture_storage" << '\n';
    // }else{
    //   LOG(FATAL) << "we dont GL_ARB_texture_storage";
    //

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
    // m_nr_total_seeds=m_estimated_seeds_per_keyframe*m_nr_buffered_keyframes;

    //nr of seeds created is counter with an atomic counter
    glGenBuffers(1, &m_atomic_nr_seeds_created);
	glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, m_atomic_nr_seeds_created);
    GLuint zero=0;
	glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint), &(zero), GL_STATIC_COPY);  //sets it to 0

    //for debuggling using icl nuim
    glGenBuffers(1, &m_seeds_gl_buf);
    glGenBuffers(1, &m_ubo_params );
    m_cur_frame.set_wrap_mode(GL_CLAMP_TO_BORDER);
    m_cur_frame.set_filter_mode(GL_LINEAR);
    m_ref_frame_tex.set_wrap_mode(GL_CLAMP_TO_BORDER);
    m_ref_frame_tex.set_filter_mode(GL_LINEAR);
    //preemptivelly prealocate a big bufffer for the seeds
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_seeds_gl_buf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, m_estimated_seeds_per_keyframe * sizeof(Seed), NULL, GL_DYNAMIC_COPY);


    compile_shaders();

}

void DepthEstimatorGL::init_context(){

    // /* Initialize the library */
    // if (!glfwInit())
    //     return -1;
    //
    // /* Create a ofscreen context and its OpenGL context */
    // GLFWwindow* offscreen_context = glfwCreateWindow(640, 480, "", NULL, NULL);
    // if (!offscreen_context) {
    //     glfwTerminate();
    //     return -1;
    // }
    //
    // /* Make the window's context current */
    // glfwMakeContextCurrent(offscreen_context);
}

void DepthEstimatorGL::compile_shaders(){


    m_update_depth_prog_id=gl::program_init_from_files( std::string(CMAKE_SOURCE_DIR)+"/shaders/compute_update_depth.glsl" );

    m_compute_hessian_pointwise_prog_id=gl::program_init_from_files( std::string(CMAKE_SOURCE_DIR)+"/shaders/compute_hessian_pointwise.glsl" );

    m_compute_hessian_blurred_prog_id=gl::program_init_from_files( std::string(CMAKE_SOURCE_DIR)+"/shaders/compute_hessian_blurred.glsl" );

    m_compute_create_seeds_prog_id=gl::program_init_from_files( std::string(CMAKE_SOURCE_DIR)+"/shaders/compute_create_seeds.glsl" );

    m_compute_trace_seeds_prog_id=gl::program_init_from_files( std::string(CMAKE_SOURCE_DIR)+"/shaders/compute_trace_seeds_icl.glsl" );

    m_compute_init_seeds_prog_id=gl::program_init_from_files( std::string(CMAKE_SOURCE_DIR)+"/shaders/compute_init_seeds.glsl" );

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



// void DepthEstimatorGL::compute_depth_and_update_mesh(const Frame& frame){
//
//     std::cout << "RECEIVED FRAME " << '\n';
//     if(frame.frame_idx%50==0){
//        m_ref_frame=frame;
//        m_seeds=create_seeds(frame);
//
//     }else{
//         //trace the created seeds
//         trace(m_seeds, m_ref_frame, frame);
//         //create a mesh
//         m_mesh=create_mesh(m_seeds, m_ref_frame);
//     }
// }

void DepthEstimatorGL::compute_depth_and_update_mesh_stereo(const Frame& frame_left, const Frame& frame_right){

    bool do_post_process=true;

    LOG(INFO) << "Received frame";
    TIME_START_GL("ALL");
    if(frame_left.frame_idx%50==0){
        m_last_ref_frame=m_ref_frame;
        m_ref_frame=frame_left;

        // create_seeds_cpu(frame_left);

        // create_seeds_gpu(frame_left);
        // if(frame_left.frame_idx==0){ //because for some reason the first frame fails to create seeds on gpu...
        //     create_seeds_gpu(frame_left);
        // }

        create_seeds_hybrid(frame_left);



        // assign_neighbours_for_points(m_seeds, m_ref_frame.gray.cols, m_ref_frame.gray.rows);
        m_started_new_keyframe=true;
        m_last_finished_mesh=m_mesh;
    }else{

        //trace the created seeds
        trace(m_nr_seeds_created, m_ref_frame, frame_right);
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

        // // //next one we will create a keyframe

        if( (  do_post_process && (frame_left.frame_idx+1) %50==0 || frame_left.is_last ) ){
            sync_seeds_buf(); //after this, the cpu and cpu will have the same data, in m_seeds and m_seeds_gl_buf
            assign_neighbours_for_points(m_seeds, m_ref_frame.gray.cols, m_ref_frame.gray.rows);
            remove_grazing_seeds(m_seeds);
            // // we will create a new keyframe but before that, do a trace on the previous keyframe
            // if(!m_last_ref_frame.gray.empty()){ //if it's the first keyframe then the last one will be empty
            //     trace(m_seeds, m_ref_frame, m_last_ref_frame);
            // }

            // denoise_cpu(m_seeds, 200,  m_ref_frame.gray.cols, m_ref_frame.gray.rows);
        }



        m_started_new_keyframe=false;
    }
    // std::vector<Seed> seeds=seeds_download(m_seeds_gl_buf, m_nr_seeds_created);
    // m_seeds=seeds_download(m_seeds_gl_buf, m_nr_seeds_created);


    //we need to do a sync because we need the data on the cpu side
    sync_seeds_buf(); //after this, the cpu and cpu will have the same data, in m_seeds and m_seeds_gl_buf
    m_mesh=create_mesh(m_seeds, m_ref_frame);
    TIME_END_GL("ALL");
}

void DepthEstimatorGL::create_seeds_cpu(const Frame& frame){
    TIME_START_GL("create_seeds");
    m_seeds.clear();
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
                int seed_idx=m_seeds.size();
                point.left = seed_idx;
                point.right = seed_idx;
                point.above = seed_idx;
                point.below = seed_idx;
                point.left_upper = seed_idx;
                point.right_upper = seed_idx;
                point.left_lower = seed_idx;
                point.right_lower = seed_idx;

                m_seeds.push_back(point);
            }

        }
    }



    //attempt 4 by box blurring on the gpu-------------------------------------------------------------------------



    TIME_END_GL("create_seeds");
    std::cout << "-------------------------seeds size is " << m_seeds.size() << '\n';

    m_nr_seeds_created=m_seeds.size();

    m_seeds_cpu_dirty=true;
    sync_seeds_buf();


    // return seeds;


}

void DepthEstimatorGL::create_seeds_hybrid (const Frame& frame){
    TIME_START_GL("create_seeds");
    m_seeds.clear();

    //upload img
    TIME_START_GL("upload_gray_and_grad");
    int size_bytes=frame.gray_with_gradients.step[0] * frame.gray_with_gradients.rows;
    std::cout << "size bytes is " << size_bytes << '\n';
    GL_C(m_ref_frame_tex.upload_data(GL_RGB32F, frame.gray_with_gradients.cols, frame.gray_with_gradients.rows, GL_RGB, GL_FLOAT, frame.gray_with_gradients.ptr(), size_bytes));
    TIME_END_GL("upload_gray_and_grad");


    TIME_START_GL("hessian_det_matrix");
    //if the m_hessian_pointwise_tex is not initialize allocate memory for it
    if(!m_hessian_pointwise_tex.get_tex_storage_initialized()){
        GL_C(m_hessian_pointwise_tex.allocate_tex_storage_inmutable(GL_RGBA32F,frame.gray.cols, frame.gray.rows));
    }
    std::cout << "starting with use program" << '\n';
    GL_C( glUseProgram(m_compute_hessian_pointwise_prog_id) );
    GL_C(bind_for_sampling(m_ref_frame_tex, 1, glGetUniformLocation(m_compute_hessian_pointwise_prog_id,"gray_with_gradients_img_sampler") ) );
    GL_C( glBindImageTexture(0, m_hessian_pointwise_tex.get_tex_id(), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F) );
    GL_C( glDispatchCompute(frame.gray.cols/32, frame.gray.rows/16, 1) ); //TODO need to ceil the size otherwise you will have block of the image that are not computed
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
    TIME_END_GL("hessian_det_matrix");



    //get hessian by box blurring
    std::cout << "m_compute_hessian_blurred_prog_id " << m_compute_hessian_blurred_prog_id << '\n';
    TIME_START_GL("hessian_blurred_matrix");
    //if the m_hessian_pointwise_tex is not initialize allocate memory for it
    if(!m_hessian_blurred_tex.get_tex_storage_initialized()){
        VLOG(1) << "initializing m_hessian_blurred_tex";
        GL_C(m_hessian_blurred_tex.allocate_tex_storage_inmutable(GL_RGBA32F,frame.gray.cols, frame.gray.rows) );
    }
    GL_C( glUseProgram(m_compute_hessian_blurred_prog_id) );
    bind_for_sampling(m_hessian_pointwise_tex, 1, glGetUniformLocation(m_compute_hessian_blurred_prog_id,"hessian_pointwise_tex_sampler") );
    GL_C( glBindImageTexture(0, m_hessian_blurred_tex.get_tex_id(), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F) );
    GL_C( glDispatchCompute(frame.gray.cols/32, frame.gray.rows/16, 1) ); //TODO need to ceil the size otherwise you will have block of the image that are not computed
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
    TIME_END_GL("hessian_blurred_matrix");

    //Read the the image back to cpu and the the ones with the high trace
    cv::Mat hessian_blurred_cv(frame.gray.rows, frame.gray.cols, CV_32FC3);
    GL_C(m_hessian_blurred_tex.bind());
    GL_C(glGetTexImage(GL_TEXTURE_2D,0, GL_RGB, GL_FLOAT, hessian_blurred_cv.data));

    //--------Do not look around the borders to avoid pattern accesing outside img
    m_seeds.clear();
    m_seeds.reserve(40000);
    for (size_t i = 10; i < frame.gray.rows-10; i++) {
        for (size_t j = 10; j < frame.gray.cols-10; j++) {

            //check if this point has enough determinant in the hessian
            Eigen::Matrix2f gradient_hessian;
            gradient_hessian.setZero();

            cv::Vec3f hessian=hessian_blurred_cv.at<cv::Vec3f>(i,j);
            // Eigen::Vector2f grad;
            // grad << color_and_grads[1], color_and_grads[2];
            //
            // gradient_hessian+= grad*grad.transpose();

            float trace = hessian[0]+hessian[2];
            if(trace>3){
                Seed point;
                point.m_uv << j,i;
                m_seeds.push_back(point);
            }

        }
    }
    m_nr_seeds_created=m_seeds.size();


    //upload the seeds we have because we need the shader to see the uv coordinates
    TIME_START_GL("upload_immature_points");
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_seeds_gl_buf);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, m_seeds.size() * sizeof(Seed), m_seeds.data());   //have to do it subdata because we want to keep the big size of the buffer so that create_seeds_gpu can write to it
    TIME_END_GL("upload_immature_points");



    TIME_START_GL("initialize_seeds");
    GL_C( glUseProgram(m_compute_init_seeds_prog_id) );

    //debug texture
    if(!m_debug_tex.get_tex_storage_initialized()){
      m_debug_tex.allocate_tex_storage_inmutable(GL_RGBA32F,frame.gray.cols, frame.gray.rows);
    }
    //clear the debug texture
    std::vector<GLuint> clear_color(4,0);
    GL_C ( glClearTexSubImage(m_debug_tex.get_tex_id(), 0,0,0,0, frame.gray.cols,frame.gray.rows,1,GL_RGBA, GL_FLOAT, clear_color.data()) );
    glBindImageTexture(2, m_debug_tex.get_tex_id(), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

    TIME_START_GL("upload_params");
    //upload params (https://hub.packtpub.com/opengl-40-using-uniform-blocks-and-uniform-buffer-objects/)
    glBindBuffer( GL_UNIFORM_BUFFER, m_ubo_params );
    glBufferData( GL_UNIFORM_BUFFER, sizeof(m_params), &m_params, GL_DYNAMIC_DRAW );
    glBindBufferBase( GL_UNIFORM_BUFFER,  glGetUniformBlockIndex(m_compute_init_seeds_prog_id,"params_block"), m_ubo_params );
    TIME_END_GL("upload_params");

    //uniforms needed for creating the seeds
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_seeds_gl_buf); //it is already prealocated to a big amount
    glUniform2fv(glGetUniformLocation(m_compute_init_seeds_prog_id,"pattern_rot_offsets"), m_pattern.get_nr_points(), m_pattern.get_offset_matrix().data()); //upload all the offses as an array of vec2 offsets
    glUniform1i(glGetUniformLocation(m_compute_init_seeds_prog_id,"pattern_rot_nr_points"), m_pattern.get_nr_points());
    glUniformMatrix3fv(glGetUniformLocation(m_compute_init_seeds_prog_id,"K"), 1, GL_FALSE, frame.K.data());
    Eigen::Matrix3f K_inv=frame.K.inverse();
    glUniformMatrix3fv(glGetUniformLocation(m_compute_init_seeds_prog_id,"K_inv"), 1, GL_FALSE, K_inv.data());
    glUniform1f(glGetUniformLocation(m_compute_init_seeds_prog_id,"min_starting_depth"), m_min_starting_depth);
    glUniform1f(glGetUniformLocation(m_compute_init_seeds_prog_id,"mean_starting_depth"), m_mean_starting_depth);
    glUniform1i(glGetUniformLocation(m_compute_init_seeds_prog_id,"idx_keyframe"), frame.frame_idx);
    //TODO maybe change the 0 in the previous m_keyframes_per_cam to something else because we now assume that if we make a keyframe for left cam we also make for right

    bind_for_sampling(m_ref_frame_tex, 1, glGetUniformLocation(m_compute_init_seeds_prog_id,"gray_with_gradients_img_sampler") );
    bind_for_sampling(m_hessian_blurred_tex, 2, glGetUniformLocation(m_compute_init_seeds_prog_id,"hessian_blurred_sampler") );

    std::cout << "launching with size " << m_seeds.size() << '\n';
    glDispatchCompute(m_seeds.size()/256, 1, 1);
    // glMemoryBarrier(GL_ALL_BARRIER_BITS);

    TIME_END_GL("initialize_seeds");



    m_seeds_gpu_dirty=true; //we intialized the seeds on the gpu side so we would need to download




    TIME_END_GL("create_seeds");
    std::cout << "-------------------------seeds size is " << m_seeds.size() << '\n';
    // return seeds;


}

void DepthEstimatorGL::create_seeds_gpu (const Frame& frame){


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
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_seeds_gl_buf);
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
    m_nr_seeds_created=atomic_nr_seeds_created_cpu[0]; //need to store it in another buffer because we will unmap this pointer
    glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);
    std::cout << "nr_seeds_created " << m_nr_seeds_created << '\n';

    m_seeds_gpu_dirty=true; //we intialized the seeds on the gpu side so we would need to download


    // //debug read the seeds back to cpu
    // std::vector<Seed> seeds;
    // std::cout << "cam " << frame.cam_id << '\n';
    // glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_seeds_gl_buf);
    // std::cout << "mapping" << '\n';
    // Seed* ptr = (Seed*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    // std::cout << "did the mapping " << '\n';
    // for (size_t i = 0; i < m_nr_seeds_created; i++) {
    //     // // for (size_t d = 0; d < 16; d++) {
    //     // //     std::cout << "debug val is " << ptr[i].debug[d] << '\n';
    //     // // }
    //     // // std::cout << "uv of seed is " << ptr[i].m_uv.transpose() << '\n';
    //     // print_seed(ptr[i]);
    //     seeds.push_back(ptr[i]);
    // }
    // glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    TIME_END_GL("create_seeds");

    // return seeds;

}


void DepthEstimatorGL::trace(const int nr_seeds_created, const Frame& ref_frame, const Frame& cur_frame){

    if(nr_seeds_created==0){
        std::cout << "No seeds, so there will be no tracing" << '\n';
        return;
    }

    VLOG(2) << "Tracing with " << nr_seeds_created << " seeds";

    // //upload to gpu the inmature points
    // if(m_started_new_keyframe){
    //     TIME_START_GL("upload_immature_points");
    //     glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_seeds_gl_buf);
    //     //have to do it subdata because we want to keep the big size of the buffer so that create_seeds_gpu can write to it
    //     glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, seeds.size() * sizeof(Seed), seeds.data());
    //     // glBufferData(GL_SHADER_STORAGE_BUFFER, seeds.size() * sizeof(Seed), seeds.data(), GL_DYNAMIC_COPY);
    //     TIME_END_GL("upload_immature_points");
    // }

    //we assume the seeds are already on the GPU in the m_seeds_gl_buf



    glUseProgram(m_compute_trace_seeds_prog_id);

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
    glBindBufferBase( GL_UNIFORM_BUFFER,  glGetUniformBlockIndex(m_compute_trace_seeds_prog_id,"params_block"), m_ubo_params );
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
    glUniform2fv(glGetUniformLocation(m_compute_trace_seeds_prog_id,"frame_size"), 1, frame_size.data());
    glUniformMatrix4fv(glGetUniformLocation(m_compute_trace_seeds_prog_id,"tf_cur_host"), 1, GL_TRUE, tf_cur_host_eigen.data());
    glUniformMatrix4fv(glGetUniformLocation(m_compute_trace_seeds_prog_id,"tf_host_cur"), 1, GL_TRUE, tf_host_cur_eigen.data());
    glUniformMatrix3fv(glGetUniformLocation(m_compute_trace_seeds_prog_id,"K"), 1, GL_FALSE, cur_frame.K.data());
    glUniformMatrix3fv(glGetUniformLocation(m_compute_trace_seeds_prog_id,"KRKi_cr"), 1, GL_FALSE, KRKi_cr_eigen.data());
    glUniform3fv(glGetUniformLocation(m_compute_trace_seeds_prog_id,"Kt_cr"), 1, Kt_cr_eigen.data());
    glUniform1f(glGetUniformLocation(m_compute_trace_seeds_prog_id,"px_error_angle"), px_error_angle);
    glUniform2fv(glGetUniformLocation(m_compute_trace_seeds_prog_id,"pattern_rot_offsets"), pattern_rot.get_nr_points(), pattern_rot.get_offset_matrix().data()); //upload all the offses as an array of vec2 offsets
    glUniform1i(glGetUniformLocation(m_compute_trace_seeds_prog_id,"pattern_rot_nr_points"), pattern_rot.get_nr_points());
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
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_seeds_gl_buf);
    bind_for_sampling(m_cur_frame, 1, glGetUniformLocation(m_compute_trace_seeds_prog_id,"gray_img_sampler") );
    glDispatchCompute(nr_seeds_created/256, 1, 1); //TODO adapt the local size to better suit the gpu
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
    TIME_END_GL("depth_update_kernel");

    m_seeds_gpu_dirty=true; //the data changed on the gpu side, we should download it

}

Mesh DepthEstimatorGL::create_mesh(const std::vector<Seed>& seeds, Frame& ref_frame){
    VLOG(1) << "creating mesh from nr of seeds " << seeds.size();
    Mesh mesh;
    mesh.V.resize(seeds.size(),3);
    mesh.V.setZero();

    for (size_t i = 0; i < seeds.size(); i++) {
        float u=seeds[i].m_uv.x();
        float v=seeds[i].m_uv.y();
        // float depth=immature_points[i].gt_depth;
        // float depth=1.0;
        float depth=1/seeds[i].depth_filter.m_mu;

        // std::cout << "seeds has x y and depth " << u << " " << v << " " << depth << '\n';

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

    m_seeds_cpu_dirty=true;
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

    m_seeds_cpu_dirty=true;
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


        if(max_mu-min_mu>0.01){
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

        if(nr_neighbours_outliers>=6){
            point.depth_filter.m_is_outlier=1;
            removed_lonely++;
        }

    }


    std::cout << "--------------------------------nr_points_removed " << nr_points_removed << '\n';
    std::cout << "--------------------------------nr_lonely " << removed_lonely << '\n';

    m_seeds_cpu_dirty=true;

}


void DepthEstimatorGL::sync_seeds_buf(){

    if(m_seeds_cpu_dirty && m_seeds_gpu_dirty){
        LOG(FATAL) << "Both buffer are dirty. We don't know whether we should do an upload or a download";
    }

    //the data on the cpu has changed we should upload it
    if(m_seeds_cpu_dirty){
        seeds_upload(m_seeds, m_seeds_gl_buf);
        m_seeds_cpu_dirty=false;
    }

    if(m_seeds_gpu_dirty){
        m_seeds=seeds_download(m_seeds_gl_buf, m_nr_seeds_created);
        m_seeds_gpu_dirty=false;
    }

}
std::vector<Seed> DepthEstimatorGL::seeds_download(const GLuint& seeds_gl_buf, const int& nr_seeds_created){
    //download from the buffer to the cpu and store in a vec
    std::vector<Seed> seeds(nr_seeds_created);

    TIME_START_GL("download_seeds");
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, seeds_gl_buf);
    Seed* ptr = (Seed*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    for (size_t i = 0; i < nr_seeds_created; i++) {
        seeds[i]=ptr[i];
    }
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    TIME_END_GL("download_seeds");

    return seeds;

}
void DepthEstimatorGL::seeds_upload(const std::vector<Seed>& seeds, const GLuint& seeds_gl_buf){
    //upload the seeds onto m_seeds_gl_buf

    TIME_START_GL("upload_seeds");
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, seeds_gl_buf);
    //have to do it subdata because we want to keep the big size of the buffer so that create_seeds_gpu can write to it
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, seeds.size() * sizeof(Seed), seeds.data());
    // glBufferData(GL_SHADER_STORAGE_BUFFER, seeds.size() * sizeof(Seed), seeds.data(), GL_DYNAMIC_COPY);
    TIME_END_GL("upload_seeds");


}
