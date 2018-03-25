#include "stereo_depth_cl/SurfelSplatter.h"

//c++
#include <iostream>

//My stuff
#include "UtilsGL.h"
#include "Shader.h"

//lbigl
#include <igl/opengl/glfw/Viewer.h>
#include <igl/look_at.h>
#include <igl/frustum.h>
#include <igl/readPLY.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/sort.h>
#include <igl/per_face_normals.h>

//cv
#include <cv_bridge/cv_bridge.h>


SurfelSplatter::SurfelSplatter():
        m_mesh_initialized(false){

    m_nr_points_in_bucket.clear();
}

void SurfelSplatter::render(const Mesh& mesh){

    if(!m_mesh_initialized){
        GL_C(create_gl_mesh(mesh));
        GL_C(compile_shaders());
        m_mesh_initialized=true;
    }
    m_view->core.clear_framebuffers();

    update_mvp();

    //bind the stuff
    GL_C(glBindVertexArray(m_vao_mesh));
    GL_C(glBindBuffer(GL_ARRAY_BUFFER, m_vbo_V));
    GL_C(glVertexAttribPointer(0, mesh.V.cols(), GL_FLOAT, GL_FALSE, 0, 0));
    GL_C(glEnableVertexAttribArray(0)); //coincides with the locations of the input variables in the shaders
    GL_C(glBindBuffer(GL_ARRAY_BUFFER, m_vbo_V_normals));
    GL_C(glVertexAttribPointer(1, mesh.NV.cols(), GL_FLOAT, GL_FALSE, 0, 0));
    GL_C(glEnableVertexAttribArray(1));
    GL_C(glBindBuffer(GL_ARRAY_BUFFER, m_vbo_V_radius));
    GL_C(glVertexAttribPointer(2, mesh.V_radius.cols(), GL_FLOAT, GL_FALSE, 0, 0));
    GL_C(glEnableVertexAttribArray(2));

    //mvp matrices
    GL_C(glUseProgram(m_splat_shader_prog_id));
    GL_C(glViewport(m_view->core.viewport(0), m_view->core.viewport(1), m_view->core.viewport(2), m_view->core.viewport(3)));
    Eigen::Matrix4f mvp= m_view->core.proj*m_view->core.view*m_view->core.model;
    GL_C(glUniformMatrix4fv(glGetUniformLocation(m_splat_shader_prog_id,"MVP"), 1, GL_FALSE, mvp.data()));
    GL_C(glUniformMatrix4fv(glGetUniformLocation(m_splat_shader_prog_id,"model"), 1, GL_FALSE, m_view->core.model.data()));
    GL_C(glUniformMatrix4fv(glGetUniformLocation(m_splat_shader_prog_id,"view"), 1, GL_FALSE, m_view->core.view.data()));
    GL_C(glUniformMatrix4fv(glGetUniformLocation(m_splat_shader_prog_id,"proj"), 1, GL_FALSE, m_view->core.proj.data()));


    //set light position
    Eigen::Vector3f rev_light = -1.*m_view->core.light_position;
    GL_C(glUniform3fv(glGetUniformLocation(m_splat_shader_prog_id,"light_position_world"), 1, rev_light.data()));
//    glUniform3fv(light_position_worldi, 1, rev_light.data());
//    glUniform1f(lighting_factori, lighting_factor); // enables lighting
//    glUniform4f(fixed_colori, 0.0, 0.0, 0.0, 0.0);

//    glUniformMatrix4fv(modeli, 1, GL_FALSE, m_view->core.model.data());
//    glUniformMatrix4fv(viewi, 1, GL_FALSE, m_view->core.view.data());
//    glUniformMatrix4fv(proji, 1, GL_FALSE, m_view->core.proj.data());
    bind_for_sampling(m_surfel_tex, 1, glGetUniformLocation(m_splat_shader_prog_id,"surfel_tex_sampler") );

    //render
//    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    GL_C(glEnable(GL_PROGRAM_POINT_SIZE));
    GL_C(glDrawArrays(GL_POINTS, 0, mesh.V.rows()));



}

void SurfelSplatter::create_gl_mesh(const Mesh& mesh){
    std::cout << "create gl mesh" << "\n";
    
    //create the buffers
    glGenVertexArrays(1, &m_vao_mesh);
    glBindVertexArray(m_vao_mesh);
    glGenBuffers(1, &m_vbo_V);
    glGenBuffers(1, &m_vbo_V_normals);
    glGenBuffers(1, &m_vbo_V_radius);

    //fill V positions
    typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> RowMatrixXf;
    RowMatrixXf V_float=mesh.V.cast<float>();
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo_V);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*V_float.size(), V_float.data(), GL_DYNAMIC_DRAW);

    //fill V_normals positions
    RowMatrixXf NV_float=mesh.NV.cast<float>();
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo_V_normals);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*NV_float.size(), NV_float.data(), GL_DYNAMIC_DRAW);

    //fill V_radius
    RowMatrixXf V_radius_float=mesh.V_radius.cast<float>();
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo_V_radius);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*V_radius_float.size(), V_radius_float.data(), GL_DYNAMIC_DRAW);


    //create surfel texture
    cv::Mat dummy_img=cv::imread("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/data/surfel_tex_2.png", cv::IMREAD_UNCHANGED);
    // cv::Mat dummy_img=cv::imread("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/data/UVMap.png");
    cv::Size size(100, 100);
    cv::resize(dummy_img, dummy_img, size);
    //needs to be an image with alpha channel becaset image_load and image_store only work with that kind of data
    cv::Mat_<cv::Vec4b> dummy_img_alpha;
//    create_alpha_mat(dummy_img,dummy_img_alpha);
    dummy_img_alpha=dummy_img; //whent the png already has alpha we can just use it directly

    m_surfel_tex.set_wrap_mode(GL_CLAMP_TO_BORDER);
    m_surfel_tex.set_filter_mode(GL_LINEAR);
    m_surfel_tex.allocate_tex_storage_inmutable(GL_RGBA8,100,100);
    m_surfel_tex.upload_without_pbo(0,0,0,dummy_img_alpha.cols,dummy_img_alpha.rows,GL_RGBA,GL_UNSIGNED_BYTE,dummy_img_alpha.ptr());


}

void SurfelSplatter::compile_shaders(){
    std::cout << "compile shaders" << "\n";

    m_splat_shader_prog_id=gl::program_init_from_files("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/shaders/surfel/splat_vert_shader.glsl",
                                                       "/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/shaders/surfel/splat_frag_shader.glsl",
                                                       "/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/shaders/surfel/splat_geom_shader.glsl");

}

void SurfelSplatter::update_mvp(){


    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f view  = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f proj  = Eigen::Matrix4f::Identity();

    // Set view
    igl::look_at( m_view->core.camera_eye, m_view->core.camera_center, m_view->core.camera_up, view);

    float width  = m_view->core.viewport(2);
    float height = m_view->core.viewport(3);


    float fH = tan(m_view->core.camera_view_angle / 360.0 * igl::PI) * m_view->core.camera_dnear;
    float fW = fH * (double)width/(double)height;
    igl::frustum(-fW, fW, -fH, fH, m_view->core.camera_dnear, m_view->core.camera_dfar,proj);
    // end projection

    // Set model transformation
    float mat[16];
    igl::quat_to_mat(m_view->core.trackball_angle.coeffs().data(), mat);

    for (unsigned i=0;i<4;++i)
        for (unsigned j=0;j<4;++j)
            model(i,j) = mat[i+4*j];

    // Why not just use Eigen::Transform<double,3,Projective> for model...?
    model.topLeftCorner(3,3)*=m_view->core.camera_zoom;
    model.topLeftCorner(3,3)*=m_view->core.model_zoom;
    model.col(3).head(3) += model.topLeftCorner(3,3)*m_view->core.model_translation;

    //set in core altough it's not necesarry and we can read the matrices directly from this class
    m_view->core.model=model;
    m_view->core.view=view;
    m_view->core.proj=proj;

}


void SurfelSplatter::distance_point_cloud_to_mesh(){
    std::string mesh_filename="/media/alex/Data/Master/SHK/c_ws/src/laser_mesher/aggregated_clouds/mesh_all_depth_11_screen_density.ply";
    // std::string mesh_filename="/media/alex/Data/Master/SHK/c_ws/src/laser_mesher/aggregated_clouds/igl_decimated_from_11_at50k_closed_holes.ply";
    Mesh mesh;
    igl::readPLY(mesh_filename, mesh.V, mesh.F);
    std::cout << "mesh is " << mesh << '\n';

    std::string point_cloud_filename="/media/alex/Data/Master/SHK/c_ws/src/laser_mesher/aggregated_clouds/points_aggregated_for_poisson_1_3.ply";
    Mesh point_cloud;
    igl::readPLY(point_cloud_filename, point_cloud.V, point_cloud.F);
    std::cout << "point cloud is " << point_cloud << '\n';

    Eigen::VectorXd sqrD;
    Eigen::VectorXi closest_faces;
    Eigen::MatrixXd closest_points;
    std::cout << "calculating distances" << '\n';
    igl::point_mesh_squared_distance(point_cloud.V, mesh.V, mesh.F, sqrD, closest_faces, closest_points);
    std::cout << "finished distances" << '\n';

    //sort by distance
    Eigen::VectorXd sqrD_sorted=sqrD;
    Eigen::VectorXd IX; //at each position says where the point ended up in the sorted vector
    igl::sort(sqrD,1,true,sqrD_sorted,IX);

    //bucket the distances
    float initial_bucket_size=0.05;
    float bucket_size=initial_bucket_size;
    m_nr_points_in_bucket.clear();
    int nr_point_current_bucket=0;
    for (size_t i = 0; i < sqrD_sorted.rows(); i++) {
        // std::cout << "iter " << i << " dist is " << sqrD_sorted(i) << '\n';
        if(std::sqrt(sqrD_sorted(i))/bucket_size < 1.0){
            nr_point_current_bucket++;
        }else{
            // std::cout << "about to push back "  << nr_point_current_bucket << '\n';
            m_nr_points_in_bucket.push_back(nr_point_current_bucket);
            nr_point_current_bucket=0;
            bucket_size*=2;
        }
    }

    // for (size_t i = 0; i < m_nr_points_in_bucket.size(); i++) {
    //     std::cout << "bucket " << initial_bucket_size*i << "-" << initial_bucket_size*(i+1) << " has " << m_nr_points_in_bucket[i] << '\n';
    // }



    //check also for orientation
    igl::per_face_normals(mesh.V, mesh.F, mesh.N_faces);
    std::vector<float> distances_orientated(sqrD.rows(),0.0);
    for (size_t i = 0; i < point_cloud.V.rows(); i++) {
        //vector from face to point in point cloud
        Eigen::Vector3d face_to_cloud =  Eigen::Vector3d(point_cloud.V.row(i)) - Eigen::Vector3d(closest_points.row(i));
        face_to_cloud.normalize();

        //dot
        double dot=face_to_cloud.dot(Eigen::Vector3d(mesh.N_faces.row(closest_faces(i))));
        if(dot<0.0){
            distances_orientated[i]= - std::sqrt(sqrD(i));
        }else{
            distances_orientated[i]=std::sqrt(sqrD(i));
        }
    }


    //sort them
    std::vector<float> distances_orientated_sorted(sqrD.rows(),0.0);
    for (size_t i = 0; i < point_cloud.V.rows(); i++) {
        distances_orientated_sorted[i] = distances_orientated[IX(i)];
    }


    //bucket them
    // bucket_size=initial_bucket_size;
    // m_nr_points_in_bucket_orientated.clear();
    // nr_point_current_bucket=0;
    // for (size_t i = 0; i < point_cloud.V.rows(); i++) {
    //     // std::cout << "iter " << i << " dist is " << sqrD_sorted(i) << '\n';
    //     if(std::fabs(distances_orientated_sorted[i])/bucket_size < 1.0){
    //         nr_point_current_bucket++;
    //     }else{
    //         // std::cout << "about to push back "  << nr_point_current_bucket << '\n';
    //         m_nr_points_in_bucket_orientated.push_back(nr_point_current_bucket);
    //         nr_point_current_bucket=0;
    //         bucket_size*=2;
    //     }
    // }


    bucket_size=initial_bucket_size;
    m_sum_distances_in_bucket_orientated.clear();
    float sum_distances_current_bucket=0;
    nr_point_current_bucket=0;
    for (size_t i = 0; i < point_cloud.V.rows(); i++) {
        // std::cout << "iter " << i << " dist is " << sqrD_sorted(i) << '\n';
        if(std::fabs(distances_orientated_sorted[i])/bucket_size < 1.0){
            // std::cout << "summing " << distances_orientated_sorted[i] << " in bucket " << m_sum_distances_in_bucket_orientated.size() << '\n';
            sum_distances_current_bucket+=distances_orientated_sorted[i];
            nr_point_current_bucket++;
        }else{
            // std::cout << "about to push back "  << nr_point_current_bucket << '\n';
            m_sum_distances_in_bucket_orientated.push_back(sum_distances_current_bucket/nr_point_current_bucket);
            sum_distances_current_bucket=0;
            nr_point_current_bucket=0;
            bucket_size*=2;
        }
    }

    std::cout << '\n';
    for (size_t i = 0; i < m_sum_distances_in_bucket_orientated.size(); i++) {
        std::cout << "dist range: " << initial_bucket_size*i << "-" << initial_bucket_size*(i+1) <<
                    " nr points " << (m_nr_points_in_bucket[i]/point_cloud.V.rows())*100 <<"\%" << " (" << m_nr_points_in_bucket[i]  << ")"<<
                    " avg dist: " << m_sum_distances_in_bucket_orientated[i] << '\n';
    }



}
