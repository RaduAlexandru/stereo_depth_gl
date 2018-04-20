#include "stereo_depth_cl/Texturer.h"


//My stuff
#include "stereo_depth_cl/Profiler.h"
#include "stereo_depth_cl/MiscUtils.h"
#include "stereo_depth_cl/LabelMngr.h"
#include "UtilsGL.h"
#include "Shader.h"


//Libigl
#include <igl/opengl/glfw/Viewer.h>

//cv
#include <cv_bridge/cv_bridge.h>

//c++
#include <chrono>



Texturer::Texturer():
        m_scenes(NUM_SCENES_BUFFER),
        m_scene_is_modified(false),
        m_finished_scene_idx(-1),
        m_working_scene_idx(0),
        m_iter_nr(0),
        m_gl_profiling_enabled(true),
        m_show_images(false),
        m_fbo_shadow(-1),
        m_fbo_shadow_rgb_tex(-1),
        m_fbo_shadow_depth_tex(-1),
        m_pbo_idx_write(0),
        m_pbo_idx_read(0),
        m_num_downloads(0),
        m_fbo_uv_baking(-1),
        m_fbo_uv_baking_rgb_tex(-1),
        m_fbo_uv_baking_depth_tex(-1),
        m_show_one_class_prob(false),
        m_one_class_id(0),
        m_rgb_global_tex_size(2048),
        // m_rgb_global_tex_size(4096),
        // m_rgb_global_tex_size(8192),
        // m_tex_size(1024)
        m_semantics_global_tex_size(2048)
        // m_semantics_global_tex_size(4096)
        // m_tex_size(4096)
        // m_tex_size(8192)
        // m_tex_size(16384)

        {

        m_label_mngr.init("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/data/classes.txt");

        m_bake_view_size=std::max(m_rgb_global_tex_size,m_semantics_global_tex_size);
}

//needed so that forward declarations work
Texturer::~Texturer(){
}

Scene& Texturer::start_working(){
    return m_scenes[m_working_scene_idx]; //TODO may need to make it into a reference to speed things up
}

void Texturer::finish_working(const Scene& scene){
    m_finished_scene_idx = m_working_scene_idx;
    m_working_scene_idx = (m_working_scene_idx + 1) % NUM_SCENES_BUFFER;
    m_scene_is_modified = true;

    //update_buffers_with_new_scene , m_working_scene_idx was already incremented so we just copy in that position
    m_scenes[m_working_scene_idx]=scene;
}

Scene Texturer::get_scene() {
    m_scene_is_modified = false;
    return m_scenes[m_finished_scene_idx];
}




void Texturer::texture_scene(const int object_id, const Frame& frame){


    TIME_START_GL("texture_scene");

    m_object_id=object_id;

    int width=frame.rgb.cols;
    int height=frame.rgb.rows;

    //fix the sizes of the rgb and the depth
    if(m_fbo_shadow==-1){
        init_fbo_shadow(width, height);
    }

    //bind stuff needed to draw the geom mesh
    TIME_START_GL("binding");
    glBindVertexArray(m_view->data_list[object_id].meshgl.vao_mesh);
    //bing the positions and texcoords of vertices of the mesh
    glBindBuffer(GL_ARRAY_BUFFER, m_view->data_list[object_id].meshgl.vbo_V);
    glVertexAttribPointer(0, m_view->data_list[object_id].V.cols(), GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0); //coincides with the locations of the input variables in the shaders
    glBindBuffer(GL_ARRAY_BUFFER, m_view->data_list[object_id].meshgl.vbo_V_uv);
    glVertexAttribPointer(1, m_view->data_list[object_id].V_uv.cols(), GL_FLOAT, GL_FALSE, 0, 0);
    GL_C(glEnableVertexAttribArray(1));
    glBindBuffer(GL_ARRAY_BUFFER, m_view->data_list[object_id].meshgl.vbo_V_normals);
    glVertexAttribPointer(2, m_view->data_list[object_id].V_normals.cols(), GL_FLOAT, GL_FALSE, 0, 0);
    GL_C(glEnableVertexAttribArray(2));
    TIME_END_GL("binding");


    //only depth
    TIME_START_GL("depth_computation");
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo_shadow);
    glClear(GL_DEPTH_BUFFER_BIT);
    glViewport(0,0,width,height);
    Eigen::Matrix4f proj=intrinsics_to_opengl_proj(frame.K.cast<double>(), width, height);
    Eigen::Matrix4f view= frame.tf_cam_world.matrix().cast<float>();
    Eigen::Matrix4f model= Eigen::Matrix4f::Identity();
    Eigen::Matrix4f mvp= proj*view*model;
    //use shadow shader
    glUseProgram(m_light_prog_id);
    glUniformMatrix4fv(glGetUniformLocation(m_light_prog_id,"MVP"), 1, GL_FALSE, mvp.data());
    glDrawElements(GL_TRIANGLES, 3*m_view->data_list[object_id].meshgl.F_vbo.rows(), GL_UNSIGNED_INT, 0);
    TIME_END_GL("depth_computation"); //about 3 ms


    // //depth but rendering with te igl shader o we can see the rgb from the camera point of view
    // m_view->data_list[0].meshgl.bind_mesh();
    // glBindFramebuffer(GL_FRAMEBUFFER, m_fbo_shadow);
    // glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // glViewport(0,0,width,height);
    // Eigen::Matrix4f proj=intrinsics_to_opengl_proj(frame.K, width, height);
    // Eigen::Matrix4f view= frame.tf_cam_world.matrix().cast<float>();
    // Eigen::Matrix4f model= Eigen::Matrix4f::Identity();
    // Eigen::Matrix4f mvp= proj*view*model;
    //
    // // Send transformations to the GPU
    // GLint modeli = glGetUniformLocation(m_view->data_list[0].meshgl.shader_mesh,"model");
    // GLint viewi  = glGetUniformLocation(m_view->data_list[0].meshgl.shader_mesh,"view");
    // GLint proji  = glGetUniformLocation(m_view->data_list[0].meshgl.shader_mesh,"proj");
    // glUniformMatrix4fv(modeli, 1, GL_FALSE, model.data());
    // glUniformMatrix4fv(viewi, 1, GL_FALSE, view.data());
    // glUniformMatrix4fv(proji, 1, GL_FALSE, proj.data());
    //
    // m_view->data_list[0].meshgl.draw_mesh(true);
    // TIME_END_GL("depth_computation"); //about 3 ms



    //read the pixels from the txture 3d and commit pages
    m_pages_to_be_commited_vec.clear();
    //read pixels
    TIME_START_GL("read_pixels");
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, m_pbos[m_pbo_idx_write]);
    m_pages_to_be_commited_volume.bind();
    glGetTexImage(GL_TEXTURE_2D_ARRAY,0,GL_RED_INTEGER,GL_UNSIGNED_BYTE,0);
    m_pbo_idx_write= (m_pbo_idx_write+1)%NUM_PBOS;

    if(m_num_downloads>=NUM_PBOS-1){
        // TIME_START("map_buffer");
        glBindBuffer(GL_PIXEL_PACK_BUFFER, m_pbos[m_pbo_idx_read]);
        unsigned char* ptr = (unsigned char*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
        // TIME_END("map_buffer");
        m_pbo_idx_read= (m_pbo_idx_read+1)%NUM_PBOS;
        // TIME_START("read_and_commit");
        for (size_t i = 0; i < m_nr_of_pages_per_side_x*m_nr_of_pages_per_side_y*m_label_mngr.get_nr_classes(); i++) {
            if(ptr[i]==1){
                int x_page_idx = i%m_nr_of_pages_per_side_x;
                int y_page_idx = (i/m_nr_of_pages_per_side_x)%m_nr_of_pages_per_side_y;
                int label = i/(m_nr_of_pages_per_side_x*m_nr_of_pages_per_side_y);


                if(m_is_page_allocated_linear[i]==0){ //page is not allocated
                    m_pages_to_be_commited_vec.push_back(PageToBeCommited(x_page_idx*m_page_size.x,
                                                                        y_page_idx*m_page_size.y,
                                                                        label));
                    m_is_page_allocated_linear[i]=1;
                }
            }
        }
        // TIME_END("read_and_commit");
        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    }
    m_num_downloads++;
    TIME_END_GL("read_pixels");

    glMemoryBarrier(GL_ALL_BARRIER_BITS);

    // commit_pages();


     //UV visibilty AND baking ----------------------------------------------------
     TIME_START_GL("uv_vis_and_baking");
     glDisable(GL_DEPTH_TEST); //don't perfor depth testing
     glDepthMask(GL_FALSE);    //don't write to depth buffer
     glColorMask(GL_FALSE,GL_FALSE,GL_FALSE,GL_FALSE);    //don't write to color buffer
     //data prep
     Eigen::Matrix4f bias;
     bias << 0.5, 0.0, 0.0, 0.0,
             0.0, 0.5, 0.0, 0.0,
             0.0, 0.0, 0.5, 0.0,
             0.5, 0.5, 0.5, 1.0;
     Eigen::Matrix4f light_bias_mvp=bias.transpose()*mvp;
     glBindFramebuffer(GL_FRAMEBUFFER, m_fbo_uv_baking);
     glViewport(0,0,m_bake_view_size,m_bake_view_size);
     //switch to shader for uv visibility
     glUseProgram(m_uv_fuse_prog_id);
     //uniforms
     GL_C(glUniformMatrix4fv(0, 1, GL_FALSE, light_bias_mvp.data()));
     Eigen::Vector3f eye_pos=frame.tf_cam_world.inverse().cast<float>().translation();
     GL_C(glUniform3fv(1, 1, eye_pos.data() ));
     //samplers
     bind_for_sampling(GL_TEXTURE_2D, m_fbo_shadow_depth_tex, 1, glGetUniformLocation(m_uv_fuse_prog_id,"shadow_map") );
     bind_for_sampling(m_rgb_local_tex, 2, glGetUniformLocation(m_uv_fuse_prog_id,"rgb_sampler") );
     bind_for_sampling(m_mask_local_tex, 3, glGetUniformLocation(m_uv_fuse_prog_id,"mask_sampler") );
     bind_for_sampling(m_classes_idxs_local_tex, 4, glGetUniformLocation(m_uv_fuse_prog_id,"classes_idxs_sampler") );
     bind_for_sampling(m_classes_probs_local_tex, 5, glGetUniformLocation(m_uv_fuse_prog_id,"probs_sampler") );
     bind_for_sampling(m_pages_commited_volume, 6, glGetUniformLocation(m_uv_fuse_prog_id,"pages_commited_volume_sampler") );
     // bind our texture to binding point 3. This means we can access it in our shaders using "layout(binding=2)"
     glBindImageTexture(0, m_rgb_global_tex.get_tex_id(), 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);
     glBindImageTexture(1, m_rgb_modified_volume.get_tex_id(), 0, GL_TRUE, 0, GL_READ_WRITE, GL_R8UI);
     glBindImageTexture(2, m_semantics_modified_volume.get_tex_id(), 0, GL_TRUE, 0, GL_READ_WRITE, GL_R8UI);
     glBindImageTexture(3, m_classes_probs_global_volume.get_tex_id(), 0, GL_TRUE, 0, GL_READ_WRITE, GL_R32F);
     glBindImageTexture(4, m_pages_to_be_commited_volume.get_tex_id(),  0, GL_TRUE, 0, GL_READ_WRITE, GL_R8UI);
     glBindImageTexture(5, m_rgb_quality_tex.get_tex_id(),  0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F);
     glDrawElements(GL_TRIANGLES, 3*m_view->data_list[object_id].meshgl.F_vbo.rows(), GL_UNSIGNED_INT, 0);
     TIME_END_GL("uv_vis_and_baking");
     // make sure writing to image has finished before read
     // glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
     glMemoryBarrier(GL_ALL_BARRIER_BITS);


     //fill rgb
     TIME_START_GL("fill_rgb");
     glUseProgram(m_fill_rgb_prog_id);
     bind_for_sampling(m_rgb_global_tex, 1, glGetUniformLocation(m_fill_rgb_prog_id,"rgb_global_tex_sampler") );
     bind_for_sampling(m_rgb_modified_volume, 2, glGetUniformLocation(m_fill_rgb_prog_id,"rgb_modified_volume_sampler") );
     glDispatchCompute(m_rgb_global_tex_size/32, m_rgb_global_tex_size/32, 1);
     TIME_END_GL("fill_rgb");



     //compute shader normalize and argmax and write to the semantics class idx and semantics prob
     TIME_START_GL("normalize_and_argmax");
     glUseProgram(m_normalize_argmax_prog_id);
     bind_for_sampling(m_classes_probs_global_volume, 1, glGetUniformLocation(m_normalize_argmax_prog_id,"classes_probs_global_volume_sampler") );
     bind_for_sampling(m_semantics_modified_volume, 2, glGetUniformLocation(m_normalize_argmax_prog_id,"semantics_modified_volume_sampler") );
     bind_for_sampling(m_pages_commited_volume, 3, glGetUniformLocation(m_normalize_argmax_prog_id,"pages_commited_volume_sampler") );
     glBindImageTexture(0, m_semantics_idxs_global_tex.get_tex_id(), 0, GL_FALSE, 0, GL_READ_WRITE, GL_R8UI);
     glBindImageTexture(1, m_semantics_probs_global_tex.get_tex_id(), 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F);
     glBindImageTexture(2, m_semantics_modified_volume.get_tex_id(), 0, GL_TRUE, 0, GL_READ_WRITE, GL_R8UI);
     glBindImageTexture(3, m_classes_probs_global_volume.get_tex_id(), 0, GL_TRUE, 0, GL_READ_WRITE, GL_R32F);
     glBindImageTexture(4, m_semantics_nr_times_modified_global_tex.get_tex_id(), 0, GL_FALSE, 0, GL_READ_WRITE, GL_R16UI);
     glDispatchCompute(m_semantics_global_tex_size/32, m_semantics_global_tex_size/32, 1);
     TIME_END_GL("normalize_and_argmax");

     //compute shader to apply color map to the semantics_classes_idx
     TIME_START_GL("color_map");
     glUseProgram(m_color_map_prog_id);
     bind_for_sampling(m_semantics_idxs_global_tex, 1, glGetUniformLocation(m_color_map_prog_id,"semantics_idxs_global_tex_sampler") );
     bind_for_sampling(m_semantics_modified_volume, 2, glGetUniformLocation(m_color_map_prog_id,"semantics_modified_volume_sampler") );
     glBindImageTexture(0, m_semantics_global_tex.get_tex_id(), 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);
     glBindImageTexture(1, m_semantics_modified_volume.get_tex_id(), 0, GL_TRUE, 0, GL_READ_WRITE, GL_R8UI);
     glDispatchCompute(m_semantics_global_tex_size/32, m_semantics_global_tex_size/32, 1);
     TIME_END_GL("color_map");

     //overlapy the probabilities of one class if necesarry
     if(m_show_one_class_prob){
         overlap_probabilities_of_class(m_one_class_id);
     }




     //finalize the frame, reset the modified for the frame back to 0
     TIME_START_GL("finalize_frame");
     //https://stackoverflow.com/questions/7195130/how-to-efficiently-initialize-texture-with-zeroes
     // std::vector<GLuint> clear_color(4,0);
     // glClearTexSubImage(m_rgb_modified_volume.get_tex_id(), 0,0,0,1, m_rgb_global_tex_size,m_rgb_global_tex_size,1,GL_RED_INTEGER, GL_UNSIGNED_BYTE, clear_color.data());
     // glClearTexSubImage(m_semantics_modified_volume.get_tex_id(), 0,0,0,1, m_semantics_global_tex_size,m_semantics_global_tex_size,1,GL_RED_INTEGER, GL_UNSIGNED_BYTE, clear_color.data());


     // m_rgb_modified_volume.upload_pbo_to_tex(0,0,1, m_rgb_global_tex_size,m_rgb_global_tex_size,1,
     //                                         GL_RED_INTEGER, GL_UNSIGNED_BYTE); //set the visbility back to 0 for the next frame
     // m_semantics_modified_volume.upload_pbo_to_tex(0,0,1, m_semantics_global_tex_size,m_semantics_global_tex_size,1,
     //                                         GL_RED_INTEGER, GL_UNSIGNED_BYTE); //set the visbility back to 0 for the next frame

    // set back as rendering to screen
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);

    TIME_END_GL("finalize_frame");



    m_iter_nr++;

    TIME_END_GL("texture_scene");
}

void Texturer::commit_pages(){
    TIME_START_GL("commit_pages");
    m_classes_probs_global_volume.bind();
    m_classes_probs_global_volume.bind_pbo();
    for (size_t i = 0; i < m_pages_to_be_commited_vec.size(); i++) {
        glTexPageCommitmentARB(GL_TEXTURE_2D_ARRAY, 0,
        m_pages_to_be_commited_vec[i].xoffset, m_pages_to_be_commited_vec[i].yoffset, m_pages_to_be_commited_vec[i].zoffset,
        m_page_size.x , m_page_size.y, 1, GL_TRUE);
        // upload_pbo of the clear page_to_tex
        m_classes_probs_global_volume.upload_pbo_to_tex_no_binds(m_pages_to_be_commited_vec[i].xoffset,
                                                        m_pages_to_be_commited_vec[i].yoffset,
                                                        m_pages_to_be_commited_vec[i].zoffset,
                                                        m_page_size.x,
                                                        m_page_size.y,
                                                        1,
                                                        GL_RED,
                                                        GL_FLOAT);


        //clear
        // std::vector<GLfloat> clear_color(4,0.0);
        // glClearTexSubImage(m_classes_probs_global_volume.get_tex_id(), 0,
        //                    m_pages_to_be_commited_vec[i].xoffset,
        //                    m_pages_to_be_commited_vec[i].yoffset,
        //                    m_pages_to_be_commited_vec[i].zoffset,
        //                    m_page_size.x,
        //                    m_page_size.y,
        //                    1,GL_RED, GL_FLOAT, clear_color.data());
    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    VLOG(1) << "commited nr of pages " << m_pages_to_be_commited_vec.size();
    m_pages_to_be_commited_vec.clear();

    //upload to gpu the pages that are actually commited
    int size_bytes=m_is_page_allocated_linear.size()*sizeof(uchar);
    m_pages_commited_volume.upload_data(GL_R8UI, m_nr_of_pages_per_side_x, m_nr_of_pages_per_side_y, m_label_mngr.get_nr_classes(),
                                        GL_RED_INTEGER, GL_UNSIGNED_BYTE, m_is_page_allocated_linear.data(), size_bytes);

    glMemoryBarrier(GL_ALL_BARRIER_BITS);
    TIME_END_GL("commit_pages");
}



void Texturer::init_opengl(){

    // print_supported_extensions();


    if(GL_ARB_debug_output){
    	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB);
    	glDebugMessageCallbackARB(debug_func, (void*)15);
	}

    int maxTextureSize;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTextureSize);
    std::cout << "max tex size is " << maxTextureSize << '\n';


    //rgb_local texture
    m_rgb_local_tex.set_wrap_mode(GL_CLAMP_TO_BORDER);
    m_rgb_local_tex.set_filter_mode(GL_LINEAR);


    //mask (filtering is set for nearest in order to not interpolate between white and black pixels)
    m_mask_local_tex.set_wrap_mode(GL_CLAMP_TO_BORDER);
    m_mask_local_tex.set_filter_mode(GL_NEAREST);


    //frame.classes and frame.probs
    m_classes_idxs_local_tex.set_wrap_mode(GL_CLAMP_TO_BORDER);
    m_classes_idxs_local_tex.set_filter_mode(GL_NEAREST);
    m_classes_probs_local_tex.set_wrap_mode(GL_CLAMP_TO_BORDER);
    m_classes_probs_local_tex.set_filter_mode(GL_NEAREST);

    //global textures
    cv::Mat dummy_img=cv::imread("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/data/solid_aqua.jpg");
    // cv::Mat dummy_img=cv::imread("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/data/UVMap.png");
    cv::Size size(m_rgb_global_tex_size, m_rgb_global_tex_size);
    cv::resize(dummy_img, dummy_img, size);
    //needs to be an image with alpha channel becaset image_load and image_store only work with that kind of data
    cv::Mat_<cv::Vec4b> dummy_img_alpha;
    create_alpha_mat(dummy_img,dummy_img_alpha);
    m_rgb_global_tex.set_wrap_mode(GL_CLAMP_TO_BORDER);
    m_rgb_global_tex.set_filter_mode(GL_LINEAR);
    m_rgb_global_tex.allocate_tex_storage_inmutable(GL_RGBA8,m_rgb_global_tex_size,m_rgb_global_tex_size);
    m_rgb_global_tex.upload_without_pbo(0,0,0,dummy_img_alpha.cols,dummy_img_alpha.rows,GL_BGRA,GL_UNSIGNED_BYTE,dummy_img_alpha.ptr());
    cv::Size size_sem(m_semantics_global_tex_size, m_semantics_global_tex_size);
    cv::resize(dummy_img_alpha, dummy_img_alpha, size_sem);
    m_semantics_global_tex.set_wrap_mode(GL_CLAMP_TO_BORDER);
    m_semantics_global_tex.set_filter_mode(GL_LINEAR);
    m_semantics_global_tex.allocate_tex_storage_inmutable(GL_RGBA8,m_semantics_global_tex_size,m_semantics_global_tex_size);
    m_semantics_global_tex.upload_without_pbo(0,0,0,dummy_img_alpha.cols,dummy_img_alpha.rows,GL_BGRA,GL_UNSIGNED_BYTE,dummy_img_alpha.ptr());

    //global texture semantics idxs and class probabilities
    std::vector<GLuint> clear_color_uint(4,0);
    m_semantics_idxs_global_tex.set_wrap_mode(GL_CLAMP_TO_BORDER);
    m_semantics_idxs_global_tex.set_filter_mode(GL_NEAREST);
    m_semantics_idxs_global_tex.allocate_tex_storage_inmutable(GL_R8UI,m_semantics_global_tex_size,m_semantics_global_tex_size);
    glClearTexSubImage(m_semantics_idxs_global_tex.get_tex_id(), 0,0,0,0, m_semantics_global_tex_size,m_semantics_global_tex_size,1,GL_RED_INTEGER, GL_UNSIGNED_BYTE, clear_color_uint.data());
    std::vector<float> clear_color_float(4,0.0);
    m_semantics_probs_global_tex.set_wrap_mode(GL_CLAMP_TO_BORDER);
    m_semantics_probs_global_tex.set_filter_mode(GL_NEAREST);
    m_semantics_probs_global_tex.allocate_tex_storage_inmutable(GL_R32F,m_semantics_global_tex_size,m_semantics_global_tex_size);
    glClearTexSubImage(m_semantics_probs_global_tex.get_tex_id(), 0,0,0,0, m_semantics_global_tex_size,m_semantics_global_tex_size,1,GL_RED, GL_FLOAT, clear_color_float.data());

    //global texture for the probabilities for only one class
    m_semantics_probs_one_class_global_tex.set_wrap_mode(GL_CLAMP_TO_BORDER);
    m_semantics_probs_one_class_global_tex.set_filter_mode(GL_NEAREST);
    m_semantics_probs_one_class_global_tex.allocate_tex_storage_inmutable(GL_RGBA8,m_semantics_global_tex_size,m_semantics_global_tex_size);
    m_semantics_probs_one_class_global_tex.upload_without_pbo(0,0,0,dummy_img_alpha.cols,dummy_img_alpha.rows,GL_BGRA,GL_UNSIGNED_BYTE,dummy_img_alpha.ptr());


    //rgb_global_quality
    m_rgb_quality_tex.set_wrap_mode(GL_CLAMP_TO_BORDER);
    m_rgb_quality_tex.set_filter_mode(GL_NEAREST);
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> zero_tex(m_rgb_global_tex_size,m_rgb_global_tex_size);
    zero_tex.setConstant(0);
    int bytes_zero_tex=m_rgb_global_tex_size*m_rgb_global_tex_size*sizeof(float);
    m_rgb_quality_tex.upload_data(GL_R32F, m_rgb_global_tex_size, m_rgb_global_tex_size, GL_RED, GL_FLOAT, zero_tex.data(), bytes_zero_tex);

    //texels modified rgb
    m_rgb_modified_volume.set_wrap_mode(GL_CLAMP_TO_BORDER);
    m_rgb_modified_volume.set_filter_mode(GL_NEAREST);
    m_rgb_modified_volume.allocate_tex_storage_inmutable(GL_R8UI, m_rgb_global_tex_size, m_rgb_global_tex_size, 2);
    std::vector<uchar> zero_buffer_0(m_rgb_global_tex_size*m_rgb_global_tex_size,0);
    m_rgb_modified_volume.upload_without_pbo(0,0,0,0,
                                            m_rgb_global_tex_size,m_rgb_global_tex_size,1,
                                            GL_RED_INTEGER,GL_UNSIGNED_BYTE,zero_buffer_0.data());
    m_rgb_modified_volume.upload_without_pbo(0,0,0,1,
                                            m_rgb_global_tex_size,m_rgb_global_tex_size,1,
                                            GL_RED_INTEGER,GL_UNSIGNED_BYTE,zero_buffer_0.data());
    m_rgb_modified_volume.upload_to_pbo(zero_buffer_0.data(),zero_buffer_0.size()*sizeof(uchar));


    //texels modified semantic
    m_semantics_modified_volume.set_wrap_mode(GL_CLAMP_TO_BORDER);
    m_semantics_modified_volume.set_filter_mode(GL_NEAREST);
    m_semantics_modified_volume.allocate_tex_storage_inmutable(GL_R8UI, m_semantics_global_tex_size, m_semantics_global_tex_size, 2);
    std::vector<uchar> zero_buffer_1(m_semantics_global_tex_size*m_semantics_global_tex_size,0);
    m_semantics_modified_volume.upload_without_pbo(0,0,0,0,
                                            m_semantics_global_tex_size,m_semantics_global_tex_size,1,
                                            GL_RED_INTEGER,GL_UNSIGNED_BYTE,zero_buffer_1.data());
    m_semantics_modified_volume.upload_without_pbo(0,0,0,1,
                                            m_semantics_global_tex_size,m_semantics_global_tex_size,1,
                                            GL_RED_INTEGER,GL_UNSIGNED_BYTE,zero_buffer_1.data());
    m_semantics_modified_volume.upload_to_pbo(zero_buffer_1.data(),zero_buffer_1.size()*sizeof(uchar));

    //semantics nr times modified
    m_semantics_nr_times_modified_global_tex.set_wrap_mode(GL_CLAMP_TO_BORDER);
    m_semantics_nr_times_modified_global_tex.set_filter_mode(GL_NEAREST);
    m_semantics_nr_times_modified_global_tex.allocate_tex_storage_inmutable(GL_R16UI, m_semantics_global_tex_size, m_semantics_global_tex_size);
    std::vector<unsigned short> zero_buffer_ushort(m_semantics_global_tex_size*m_semantics_global_tex_size,0);
    m_semantics_nr_times_modified_global_tex.upload_without_pbo(0,0,0,
                                            m_semantics_global_tex_size,m_semantics_global_tex_size,
                                            GL_RED_INTEGER,GL_UNSIGNED_SHORT,zero_buffer_ushort.data());



    glGetInternalformativ(GL_TEXTURE_2D_ARRAY, GL_R32F, GL_VIRTUAL_PAGE_SIZE_X_ARB, 1, &m_page_size.x);
	glGetInternalformativ(GL_TEXTURE_2D_ARRAY, GL_R32F, GL_VIRTUAL_PAGE_SIZE_Y_ARB, 1, &m_page_size.y);
    glGetInternalformativ(GL_TEXTURE_2D_ARRAY, GL_R32F, GL_VIRTUAL_PAGE_SIZE_Z_ARB, 1, &m_page_size.z);
    std::cout << "page size is " << m_page_size.x << " " << m_page_size.y << " " << m_page_size.z << '\n';
    m_nr_of_pages_per_side_x=std::ceil(m_semantics_global_tex_size/m_page_size.x);
    m_nr_of_pages_per_side_y=std::ceil(m_semantics_global_tex_size/m_page_size.y);
    std::cout << "nr_of nr_of_pages_per_side_x " << m_nr_of_pages_per_side_x << '\n';
    std::cout << "nr_of nr_of_pages_per_side_y " << m_nr_of_pages_per_side_y << '\n';
    m_is_page_allocated_linear.resize(m_nr_of_pages_per_side_x*m_nr_of_pages_per_side_y*m_label_mngr.get_nr_classes(),0); //to acces with a linear idx
    // m_clear_page_glm.resize(m_page_size.x*m_page_size.y, glm::packSnorm1x16(1.0/m_label_mngr.get_nr_classes()) );
    m_clear_page_glm.resize(m_page_size.x*m_page_size.y, 0.0);




    //default classes_probs -sparse
    m_classes_probs_global_volume.set_wrap_mode(GL_CLAMP_TO_BORDER);
    m_classes_probs_global_volume.set_filter_mode(GL_NEAREST);
    m_classes_probs_global_volume.set_sparse(GL_TRUE);
    // m_classes_probs_global_volume.bind();
    // glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_VIRTUAL_PAGE_SIZE_INDEX_ARB, 1);
    m_classes_probs_global_volume.allocate_tex_storage_inmutable(GL_R32F, m_semantics_global_tex_size, m_semantics_global_tex_size, m_label_mngr.get_nr_classes());
    m_classes_probs_global_volume.upload_to_pbo(m_clear_page_glm.data(), m_clear_page_glm.size()*sizeof(float));


    //pages_to_be commited 2d array
    m_pages_to_be_commited_volume.set_wrap_mode(GL_CLAMP_TO_BORDER);
    m_pages_to_be_commited_volume.set_filter_mode(GL_NEAREST);
    m_pages_to_be_commited_volume.allocate_tex_storage_inmutable(GL_R8UI, m_nr_of_pages_per_side_x, m_nr_of_pages_per_side_y, m_label_mngr.get_nr_classes());

    //pages_already commited texture 2d array
    m_pages_commited_volume.set_wrap_mode(GL_CLAMP_TO_BORDER);
    m_pages_commited_volume.set_filter_mode(GL_NEAREST);







    //shaders
    compile_shaders();
    // m_pages_to_be_commited_cpu = (GLubyte *) malloc(sizeof(GLubyte) * m_nr_of_pages_per_side_x * m_nr_of_pages_per_side_y* m_label_mngr.get_nr_classes());




    //default_classes_colored
    // cv::Mat dummy_img=cv::imread("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/data/muse_uv_01-01-1024x1024.jpg");
    // cv::Mat dummy_img=cv::imread("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/data/solid_aqua.jpg");
    // // cv::Mat dummy_img=cv::imread("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/data/UVMap.png");
    // cv::Size size(m_tex_size, m_tex_size);
    // cv::resize(dummy_img, dummy_img, size);
    // //needs to be an image with alpha channel becaset image_load and image_store only work with that kind of data
    // cv::Mat_<cv::Vec4b> dummy_img_alpha;
    // create_alpha_mat(dummy_img,dummy_img_alpha);

    //upload directly to the opengl_state->vbo_tex becase that's already an rgb texture that gets binded on every call
    // glActiveTexture(GL_TEXTURE0);
    // std::cout << "vbo_tex is " << m_view->opengl.vbo_tex << "\n";
    // GL_C(glBindTexture(GL_TEXTURE_2D, m_view->data_list[0].meshgl.vbo_tex));
    // GL_C(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER));
    // GL_C(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER));
    // GL_C(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
    // GL_C(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    // glTexImage2D(GL_TEXTURE_2D,         // Type of texture
    //              0,                   // Pyramid level (for mip-mapping) - 0 is the top level
    //              GL_RGBA8,              // Internal colour format to convert to
    //              dummy_img_alpha.cols,          // Image width  i.e. 640 for Kinect in standard mode
    //              dummy_img_alpha.rows,          // Image height i.e. 480 for Kinect in standard mode
    //              0,                   // Border width in pixels (can either be 1 or 0)
    //              GL_BGRA,              // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
    //              GL_UNSIGNED_BYTE,    // Image data type
    //              dummy_img_alpha.ptr());        // The actual image data itself




    init_fbo_uv_baking(m_bake_view_size, m_bake_view_size);


    //pbos
    m_pbos = new GLuint[NUM_PBOS];
    std::cout << "creating pbos " << NUM_PBOS << '\n';
    glGenBuffers(NUM_PBOS, m_pbos);
    // m_pbo_ptrs.resize(NUM_PBOS);
    for (size_t i = 0; i < NUM_PBOS; i++) {
        glBindBuffer(GL_PIXEL_PACK_BUFFER, m_pbos[i]);
        // int nbytes = m_tex_size * m_tex_size * 1; //the size of the m_classes_visibility_tex which is m_tex_size*m_tex_size*sizeof(uchar)
        int nbytes = m_nr_of_pages_per_side_x*m_nr_of_pages_per_side_y*m_label_mngr.get_nr_classes() * 1; //the size of the pages_to_be_commited which is m_nr_of_pages_per_side_x*m_nr_of_pages_per_side_y*nr_classes

        glBufferData(GL_PIXEL_PACK_BUFFER, nbytes, NULL, GL_STREAM_READ);

        //for persistent mapping, however not really necesary since mapping the buffer is quite fast
        // const GLbitfield flags = GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
        // glBufferStorage(GL_PIXEL_PACK_BUFFER, nbytes, NULL, flags);
        // m_pbo_ptrs[i]= glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0 , nbytes, flags);

        std::cout << "created pbo " << m_pbos[i]<< '\n';
    }
     glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);


    set_global_texture();

    VLOG(1) << "finished init_opengl";

}

void Texturer::init_fbo_shadow(const int width, const int height){


//    //attempt 3
//    //framebuffer for depth rendering (also has an rgb component for possible usage with the igl shader to also see what we are drawing)
//    glGenFramebuffers(1, &m_fbo_shadow);
//    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo_shadow);
//    //rgb output
//    glGenTextures(1, &m_fbo_shadow_rgb_tex);
//    glBindTexture(GL_TEXTURE_2D, m_fbo_shadow_rgb_tex);
//    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, 0); //dummy values for the size
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); //the poor filtering is needed
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
//    //depth output
//    glGenTextures(1, &m_fbo_shadow_depth_tex);
//    glBindTexture(GL_TEXTURE_2D, m_fbo_shadow_depth_tex);
//    glTexImage2D(GL_TEXTURE_2D, 0,GL_DEPTH_COMPONENT24, width, height, 0,GL_DEPTH_COMPONENT, GL_FLOAT, 0);
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); //the poor filtering is needed
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
//    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, m_fbo_shadow_depth_tex, 0);
//    //configure
//    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, m_fbo_shadow_rgb_tex, 0);
//    GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
//    glDrawBuffers(1, DrawBuffers); // "1" is the size of DrawBuffers
//    // Always check that our framebuffer is ok
//    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE){
//        LOG(FATAL) << "something went wrong with the framebuffer creation";
//        return;
//    }
//    // set back as rendering to screen
//    glBindFramebuffer(GL_FRAMEBUFFER, 0);



    //previous work but has also rgb, now we do ony depth
    glGenFramebuffers(1, &m_fbo_shadow);
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo_shadow);

    // Depth texture. Slower than a depth buffer, but you can sample it later in your shader
    glGenTextures(1, &m_fbo_shadow_depth_tex);
    glBindTexture(GL_TEXTURE_2D, m_fbo_shadow_depth_tex);
    glTexImage2D(GL_TEXTURE_2D, 0,GL_DEPTH_COMPONENT32, width, height, 0,GL_DEPTH_COMPONENT, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, m_fbo_shadow_depth_tex, 0);

    glDrawBuffer(GL_NONE); // No color buffer is drawn to.

    // Always check that our framebuffer is ok
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE){
        LOG(FATAL) << "something went wrong with the framebuffer creation";
        return;
    }
    // set back as rendering to screen
    glBindFramebuffer(GL_FRAMEBUFFER, 0);


}

void Texturer::init_fbo_uv_baking(const int width, const int height ){
    //framebuffer for uv_baking, just too see what is happening inside it
    VLOG(1) << "init_fbo_uv_baking";
    glGenFramebuffers(1, &m_fbo_uv_baking);
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo_uv_baking);
    // //rgb output
    glGenTextures(1, &m_fbo_uv_baking_rgb_tex);
    glBindTexture(GL_TEXTURE_2D, m_fbo_uv_baking_rgb_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, 0); //dummy values for the size
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); //the poor filtering is needed
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    //use the texture for classes colored as rgb
    // glBindTexture(GL_TEXTURE_2D, m_view->opengl.vbo_tex);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    // glTexImage2D(GL_TEXTURE_2D,         // Type of texture
    //              0,                   // Pyramid level (for mip-mapping) - 0 is the top level
    //              GL_RGB,              // Internal colour format to convert to
    //              width,          // Image width  i.e. 640 for Kinect in standard mode
    //              height,          // Image height i.e. 480 for Kinect in standard mode
    //              0,                   // Border width in pixels (can either be 1 or 0)
    //              GL_BGR,              // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
    //              GL_UNSIGNED_BYTE,    // Image data type
    //              0);        // The actual image data itself


    //depth output
    glGenTextures(1, &m_fbo_uv_baking_depth_tex);
    glBindTexture(GL_TEXTURE_2D, m_fbo_uv_baking_depth_tex);
    glTexImage2D(GL_TEXTURE_2D, 0,GL_DEPTH_COMPONENT32, width, height, 0,GL_DEPTH_COMPONENT, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); //the poor filtering is needed
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, m_fbo_uv_baking_depth_tex, 0);
    //configure
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, m_fbo_uv_baking_rgb_tex, 0);
    // glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, m_view->opengl.vbo_tex, 0);
    GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, DrawBuffers); // "1" is the size of DrawBuffers
    // Always check that our framebuffer is ok
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE){
        LOG(FATAL) << "something went wrong with the framebuffer creation";
        return;
    }
    // set back as rendering to screen
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // glFinish();
}

void Texturer::compile_shaders(){
    m_light_prog_id=gl::program_init_from_files("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/shaders/light_vert_shader.glsl", "/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/shaders/light_frag_shader.glsl");
    // m_light_mvp_id=glGetUniformLocation(m_light_prog_id, "MVP");

    m_uv_fuse_prog_id=gl::program_init_from_files("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/shaders/uv_fuse_vert_shader.glsl", "/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/shaders/uv_fuse_frag_shader.glsl");
    glProgramUniform1i(m_uv_fuse_prog_id, glGetUniformLocation(m_uv_fuse_prog_id, "rgb_global_tex_size"), m_rgb_global_tex_size);
    glProgramUniform1i(m_uv_fuse_prog_id, glGetUniformLocation(m_uv_fuse_prog_id, "semantics_global_tex_size"), m_semantics_global_tex_size);
    glProgramUniform1i(m_uv_fuse_prog_id, glGetUniformLocation(m_uv_fuse_prog_id, "page_size_x"), m_page_size.x);
    glProgramUniform1i(m_uv_fuse_prog_id, glGetUniformLocation(m_uv_fuse_prog_id, "page_size_y"), m_page_size.y);

    m_normalize_argmax_prog_id=gl::program_init_from_files("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/shaders/compute_normalize_argmax.glsl");
    glProgramUniform1i(m_normalize_argmax_prog_id, glGetUniformLocation(m_normalize_argmax_prog_id, "page_size_x"), m_page_size.x);
    glProgramUniform1i(m_normalize_argmax_prog_id, glGetUniformLocation(m_normalize_argmax_prog_id, "page_size_y"), m_page_size.y);

    m_fill_rgb_prog_id=gl::program_init_from_files("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/shaders/compute_fill_rgb.glsl");

    m_color_map_prog_id=gl::program_init_from_files("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/shaders/compute_color_map.glsl");

    m_prob_one_class_prog_id=gl::program_init_from_files("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/shaders/compute_one_class_prob.glsl");
    glProgramUniform1i(m_prob_one_class_prog_id, glGetUniformLocation(m_prob_one_class_prog_id, "page_size_x"), m_page_size.x);
    glProgramUniform1i(m_prob_one_class_prog_id, glGetUniformLocation(m_prob_one_class_prog_id, "page_size_y"), m_page_size.y);

}

void Texturer::overlap_probabilities_of_class(const int class_id){
    //compute shader that read the probabilities volume that writes the values into m_semantics_probs_one_class_global_tex
    // VLOG(1) << "view_probabilities_of_class";

    // //get the slice of texture and get the min max of it
    // int size_bytes=sizeof(float) * m_semantics_global_tex_size * m_semantics_global_tex_size;
    // float* m_classes_probs_cpu =(float *) malloc(size_bytes);
    // // glPixelStorei(GL_PACK_ALIGNMENT, 1);
    // // m_classes_probs_global_volume.bind();
    // glGetTextureSubImage(m_classes_probs_global_volume.get_tex_id(),0,
    //                     0,0,class_id, m_semantics_global_tex_size, m_semantics_global_tex_size, 1,
    //                     GL_RED,GL_FLOAT, size_bytes,m_classes_probs_cpu);
    // float min=9999999;
    // float max=-99999999;
    // for (size_t i = 0; i < m_semantics_global_tex_size*m_semantics_global_tex_size; i++) {
    //     float val =(float) m_classes_probs_cpu[i];
    //     if(val<min) min=val;
    //     if(val>max) max=val;
    // }
    // std::cout << "min max is " << min << " " << max << '\n';



    //compute shader to apply color map to the semantics_classes_idx
    TIME_START_GL("prob_one_class");
    glUseProgram(m_prob_one_class_prog_id);

    glProgramUniform1i(m_prob_one_class_prog_id, glGetUniformLocation(m_prob_one_class_prog_id, "class_id"), class_id);
    glProgramUniform1f(m_prob_one_class_prog_id, glGetUniformLocation(m_prob_one_class_prog_id, "min_prob"), m_min_one_class_prob);
    glProgramUniform1f(m_prob_one_class_prog_id, glGetUniformLocation(m_prob_one_class_prog_id, "max_prob"), m_max_one_class_prob);
    glProgramUniform1i(m_prob_one_class_prog_id, glGetUniformLocation(m_prob_one_class_prog_id, "global_texture_type"), m_global_texture_type);
    glProgramUniform1i(m_prob_one_class_prog_id, glGetUniformLocation(m_prob_one_class_prog_id, "rgb_scale_multiplier"), m_rgb_global_tex_size/m_semantics_global_tex_size);

    bind_for_sampling(m_classes_probs_global_volume, 1, glGetUniformLocation(m_prob_one_class_prog_id,"classes_probs_global_volume_sampler") );
    bind_for_sampling(m_pages_commited_volume, 2, glGetUniformLocation(m_prob_one_class_prog_id,"pages_commited_volume_sampler") );
    bind_for_sampling(m_rgb_global_tex, 3, glGetUniformLocation(m_prob_one_class_prog_id,"rgb_global_sampler") );
    bind_for_sampling(m_semantics_global_tex, 4, glGetUniformLocation(m_prob_one_class_prog_id,"semantics_global_sampler") );
    glBindImageTexture(0, m_semantics_probs_one_class_global_tex.get_tex_id(), 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);
    glBindImageTexture(1, m_classes_probs_global_volume.get_tex_id(), 0, GL_TRUE, 0, GL_READ_WRITE, GL_R32F);
    glBindImageTexture(2, m_semantics_nr_times_modified_global_tex.get_tex_id(), 0, GL_FALSE, 0, GL_READ_WRITE, GL_R16UI);
    glBindImageTexture(3, m_rgb_global_tex.get_tex_id(), 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);
    glBindImageTexture(4, m_semantics_global_tex.get_tex_id(), 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);
    glDispatchCompute(m_semantics_global_tex_size/32, m_semantics_global_tex_size/32, 1);

    m_view->data_list[0].meshgl.vbo_tex=m_semantics_probs_one_class_global_tex.get_tex_id(); //switch to that texture
    TIME_END_GL("prob_one_class");

}

//TODO hardcored object id for the datalist
void Texturer::set_global_texture(){
    if(m_show_one_class_prob){
        m_view->data_list[0].meshgl.vbo_tex=m_semantics_probs_one_class_global_tex.get_tex_id(); //switch to that texture
    }else if(m_global_texture_type==0){
        m_view->data_list[0].meshgl.vbo_tex=m_rgb_global_tex.get_tex_id();
    }else if(m_global_texture_type==1){
        m_view->data_list[0].meshgl.vbo_tex=m_semantics_global_tex.get_tex_id();
    }else if(m_global_texture_type==2){
        //WHY should it be dark and no one class prob??
    }

}

void Texturer::upload_rgb_texture(const cv::Mat& image){
    int size_bytes=image.step[0] * image.rows;
    m_rgb_local_tex.upload_data(GL_RGB, image.cols, image.rows, GL_BGR, GL_UNSIGNED_BYTE, image.ptr(), size_bytes);
}

void Texturer::upload_mask_texture(const cv::Mat& image){
    int size_bytes=image.step[0] * image.rows;
    m_mask_local_tex.upload_data(GL_R8UI, image.cols, image.rows, GL_RED_INTEGER, GL_UNSIGNED_BYTE, image.ptr(), size_bytes);

}

void Texturer::upload_frame_classes_and_probs_texture(const cv::Mat& classes, const cv::Mat& probs){

    int size_bytes_classes=classes.step[0] * classes.rows;
    int size_bytes_probs=probs.step[0] * probs.rows;

    m_classes_idxs_local_tex.upload_data(GL_R8UI, classes.cols, classes.rows, GL_RED_INTEGER, GL_UNSIGNED_BYTE, classes.ptr(), size_bytes_classes);
    m_classes_probs_local_tex.upload_data(GL_R32F, probs.cols, probs.rows, GL_RED, GL_FLOAT, probs.ptr(), size_bytes_probs);
}
