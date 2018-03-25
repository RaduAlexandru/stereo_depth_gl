#pragma once
//C++
#include <iosfwd>
#include <memory>
#include <atomic>
#include <tuple>

//OpenCV
#include <opencv2/highgui/highgui.hpp>

//My stuff
#include "stereo_depth_cl/Mesh.h"
#include "stereo_depth_cl/Scene.h"
#include "stereo_depth_cl/DataLoader.h"
#include "stereo_depth_cl/LabelMngr.h"
#include "Texture2D.h"
#include "Texture2DArray.h"


//GL
#include <GL/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/packing.hpp>



#define NUM_SCENES_BUFFER 5
#define NUM_PBOS 2

//forward declarations
class Profiler;
namespace igl {  namespace opengl {namespace glfw{ class Viewer; }}}

struct PageToBeCommited {
    PageToBeCommited(int xoffset, int yoffset, int zoffset):
        xoffset(xoffset),
        yoffset(yoffset),
        zoffset(zoffset){
        }
    int xoffset, yoffset, zoffset;
};

class Texturer{
public:
    Texturer();
    ~Texturer(); //needed so that forward declarations work
    void init_opengl();
    void texture_scene(const int object_id, const Frame& frame);
    void compile_shaders();
    void commit_pages();
    void overlap_probabilities_of_class(const int class_id);


    Scene get_scene();
    bool is_modified(){return m_scene_is_modified;};

    void upload_rgb_texture(const cv::Mat& image);
    void upload_mask_texture(const cv::Mat& image);
    void upload_frame_classes_and_probs_texture(const cv::Mat& classes, const cv::Mat& probs);

    void set_global_texture();

    int m_object_id; //idx inside the viewer.datalist of the object we want to texture
    int m_bake_view_size; //the size of the viewport in which the baking takes place, its the max between the rgb and the semantics globa textures

    //objects
    std::shared_ptr<Profiler> m_profiler;
    std::shared_ptr<igl::opengl::glfw::Viewer> m_view;
    LabelMngr m_label_mngr;

    //Local textures
    gl::Texture2D m_rgb_local_tex;
    gl::Texture2D m_mask_local_tex;
    gl::Texture2D m_classes_idxs_local_tex;
    gl::Texture2D m_classes_probs_local_tex;

    //global textures
    gl::Texture2D m_rgb_global_tex;
    gl::Texture2D m_semantics_global_tex; //the rgb color map for semantics
    gl::Texture2D m_semantics_idxs_global_tex;  //globally stores the idx of the most probable class
    gl::Texture2D m_semantics_probs_global_tex; //globally stores the probability for the best class
    gl::Texture2D m_semantics_probs_one_class_global_tex; //just for showing the probabilities for one class (a slice of the probabilites volume)

    //checking if the texels were modified
    //first texture we store the texels that were modified in all frames
    //in the second, only the ones that were modified in this frame
    gl::Texture2DArray m_rgb_modified_volume;
    gl::Texture2DArray m_semantics_modified_volume;

    //nr of times the semantics texels were modified (increases by one each time we process the texel) (used for normalization)
    gl::Texture2D m_semantics_nr_times_modified_global_tex;



    //stuff needed for depth computation
    GLuint m_light_prog_id;
    GLuint m_fbo_shadow;
    GLuint m_fbo_shadow_rgb_tex;
    GLuint m_fbo_shadow_depth_tex;

    //stuff needed for uv baking
    GLuint m_uv_fuse_prog_id;
    gl::Texture2DArray m_classes_probs_global_volume;
    gl::Texture2DArray m_pages_to_be_commited_volume;
    gl::Texture2DArray m_pages_commited_volume;
    std::vector<PageToBeCommited> m_pages_to_be_commited_vec;
    glm::ivec3 m_page_size;
    int m_nr_of_pages_per_side_x;
    int m_nr_of_pages_per_side_y;
    GLuint m_fbo_uv_baking; //just too see the output of the uv_baking
    GLuint m_fbo_uv_baking_rgb_tex;
    GLuint m_fbo_uv_baking_depth_tex;
    GLuint* m_pbos;
    int m_pbo_idx_write;
    int m_pbo_idx_read;
    int m_num_downloads;
    gl::Texture2D m_rgb_quality_tex;
    std::vector<unsigned char> m_is_page_allocated_linear;
    // std::vector<glm::uint16> m_clear_page_glm;
    std::vector<float> m_clear_page_glm;

    //compute shaders
    GLuint m_normalize_argmax_prog_id;
    GLuint m_fill_rgb_prog_id;
    GLuint m_color_map_prog_id;
    GLuint m_prob_one_class_prog_id;

    //for viewing one class prob
    float m_min_one_class_prob;
    float m_max_one_class_prob;





    //databasse
    int m_iter_nr;
    std::vector<Scene> m_scenes;
    int m_finished_scene_idx; //idx pointing to the most recent finished scene
    int m_working_scene_idx; //idx poiting to the scene we are currently working on
    std::atomic<bool> m_scene_is_modified;

    //opengl params
    int m_rgb_global_tex_size;
    int m_semantics_global_tex_size;

    //params
    bool m_gl_profiling_enabled;
    bool m_show_images;
    // bool m_show_rgb_global;
    // bool m_show_semantics_global;
    bool m_show_one_class_prob;
    int m_one_class_id;

    int m_global_texture_type;
    const char* m_global_texture_type_desc[3] =
      {
              "RGB",
              "Semantics",
              "Empty"
      };



private:
    // void init_opengl();
    void init_fbo_shadow(const int width, const int height );
    void init_fbo_uv_baking(const int width, const int height );
    Scene& start_working();
    void finish_working(const Scene& scene);

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


//inline void CheckOpenGLError(const char* stmt, const char* fname, int line)
//{
//    GLenum err = glGetError();
//    //  const GLubyte* sError = gluErrorString(err);
//
//    if (err != GL_NO_ERROR){
//        printf("OpenGL error %08x, at %s:%i - for %s.\n", err, fname, line, stmt);
//        exit(1);
//    }
//}
//
//// GL Check Macro. Will terminate the program if a GL error is detected.
//#define GL_C(stmt) do {					\
//	stmt;						\
//	CheckOpenGLError(#stmt, __FILE__, __LINE__);	\
//} while (0)
