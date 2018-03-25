#pragma once
//C++
#include <iosfwd>
#include <memory>

//My stuff
#include "stereo_depth_cl/Mesh.h"
#include "Texture2D.h"

//GL
#include <GL/glad.h>

//forward declarations
class Profiler;
namespace igl {  namespace opengl {namespace glfw{ class Viewer; }}}

class SurfelSplatter{
public:
    SurfelSplatter();
    void render(const Mesh& mesh);

    //objects
    std::shared_ptr<Profiler> m_profiler;
    std::shared_ptr<igl::opengl::glfw::Viewer> m_view;


    //misc plotting and things
    void distance_point_cloud_to_mesh(); //meshing introduces an error to the point cloud, we should see how big it is
    std::vector<float> m_nr_points_in_bucket;
    std::vector<float> m_sum_distances_in_bucket_orientated;

private:

    void create_gl_mesh(const Mesh& mesh);
    void compile_shaders();
    void update_mvp();


    bool m_mesh_initialized;
    gl::Texture2D m_surfel_tex;
    GLuint m_vao_mesh;
    GLuint m_vbo_V;
    GLuint m_vbo_V_normals;
    GLuint m_vbo_V_radius;
    GLuint m_splat_shader_prog_id;

};
