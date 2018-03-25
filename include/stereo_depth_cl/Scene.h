#pragma once

#include "stereo_depth_cl/Mesh.h"

#include <vector>
#include <memory>

namespace igl {  namespace opengl {namespace glfw{ class Viewer; }}}

class Scene{

public:
    Scene();
    void add_mesh(const Mesh& mesh, const std::string name);
    int get_nr_meshes();
    int get_total_nr_vertices();
    int get_total_nr_faces();
    Mesh& get_mesh_with_name(const std::string name);
    Mesh& get_mesh_with_idx(const int idx);
    int get_idx_for_name(const std::string name);


    std::shared_ptr<igl::opengl::glfw::Viewer> m_view;

private:
    std::vector<Mesh> m_meshes;
    bool m_is_first_appended_mesh;

};
