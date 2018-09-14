#include "stereo_depth_gl/Scene.h"

//c++

//my stuff
#include "stereo_depth_gl/MiscUtils.h"

//loguru
#include <loguru.hpp>

//libigl
#include <igl/opengl/glfw/Viewer.h>

Scene::Scene():
    m_is_first_appended_mesh(true){

}


void Scene::add_mesh(const Mesh& mesh, const std::string name){
    if( !m_is_first_appended_mesh){ //The viewer start already with one empty dummy mesh data, so on the first one we don't need to allocate anything
        m_view->append_mesh();
    }

    m_meshes.push_back(mesh);
    m_meshes.back().name=name;
    m_is_first_appended_mesh=false;
}

int Scene::get_nr_meshes(){
    return m_meshes.size();
}

int Scene::get_total_nr_vertices(){
    int V_nr=0;
    for (size_t i = 0; i < m_meshes.size(); i++) {
        if(m_meshes[i].m_is_visible){
            V_nr+=m_meshes[i].V.rows();
        }
    }
    return V_nr;
}

int Scene::get_total_nr_faces(){
    int F_nr=0;
    for (size_t i = 0; i < m_meshes.size(); i++) {
        if(m_meshes[i].m_is_visible){
            F_nr+=m_meshes[i].F.rows();
        }
    }
    return F_nr;
}

Mesh& Scene::get_mesh_with_name(const std::string name){
    for (size_t i = 0; i < m_meshes.size(); i++) {
        if(m_meshes[i].name==name){
            return m_meshes[i];
        }
    }
    LOG_S(ERROR) << "No mesh with name " << name;
}

Mesh& Scene::get_mesh_with_idx(const int idx){
    if(idx<m_meshes.size()){
        return m_meshes[idx];
    }else{
        LOG_S(ERROR) << "No mesh with idx " << idx;
    }

}

int Scene::get_idx_for_name(const std::string name){
    for (size_t i = 0; i < m_meshes.size(); i++) {
        if(m_meshes[i].name==name){
            return i;
        }
    }
    LOG_S(ERROR) << "No mesh with name " << name;

}

bool Scene::does_mesh_with_name_exist(const std::string name){
    for (size_t i = 0; i < m_meshes.size(); i++) {
        if(m_meshes[i].name==name){
            return true;
        }
    }
    return false;
}

void Scene::merge_mesh(const Mesh& mesh, const std::string name){
    if(!does_mesh_with_name_exist(name)){
        add_mesh(mesh,name);
    }else{
        Mesh& stored_mesh=get_mesh_with_name(name);
        stored_mesh.add(mesh);
        stored_mesh.m_visualization_should_change=true;
    }
}
