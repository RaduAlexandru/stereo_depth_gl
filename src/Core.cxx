//C++
#include <iostream>

#include <Eigen/Dense>

//My stuff
#include "stereo_depth_gl/Core.h"
#include "stereo_depth_gl/MiscUtils.h"
#include "stereo_depth_gl/Profiler.h"
#include "stereo_depth_gl/RosBagPlayer.h"
#include "stereo_depth_gl/DepthEstimatorCPU.h"
#include "stereo_depth_gl/DepthEstimatorRenegade.h"
#include "stereo_depth_gl/DepthEstimatorGL.h"
// #include "stereo_depth_gl/DepthEstimatorGL2.h"
#include "stereo_depth_gl/DataLoader.h"
// #include "stereo_depth_gl/SurfelSplatter.h"

//libigl
#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOBJ.h>
#include <igl/readPLY.h>
#include <igl/writePLY.h>
#include <igl/writeOBJ.h>
#include <igl/per_vertex_normals.h>
// #include <igl/embree/ambient_occlusion.h>
// #include <igl/embree/EmbreeIntersector.h>

//ROS
#include "stereo_depth_gl/RosTools.h"

//gl
#include "UtilsGL.h"

//boost
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


Core::Core(std::shared_ptr<igl::opengl::glfw::Viewer> view, std::shared_ptr<Profiler> profiler) :
        m_viewer_initialized(false),
        m_player(new RosBagPlayer),
        m_depth_estimator(new DepthEstimatorCPU),
        m_depth_estimator_renegade(new DepthEstimatorRenegade),
        m_depth_estimator_cl(new DepthEstimatorGL),
        // m_depth_estimator_gl2(new DepthEstimatorGL2),
        m_loader(new DataLoader),
        // m_splatter(new SurfelSplatter),
        m_nr_callbacks(0),
        dir_watcher("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_gl/shaders/",5),
        m_surfel_rendering(false){

    m_view = view;
    m_profiler=profiler;
    m_depth_estimator->m_profiler=profiler;
    m_depth_estimator->m_view=m_view;
    m_depth_estimator_renegade->m_profiler=profiler;
    m_depth_estimator_renegade->m_view=m_view;
    m_depth_estimator_cl->m_profiler=profiler;
    m_depth_estimator_cl->m_view=m_view;
    // m_depth_estimator_gl2->m_profiler=profiler;
    // m_depth_estimator_gl2->m_view=m_view;
    // m_splatter->m_view=m_view;
    m_loader->m_profiler=profiler;
    m_loader->m_player=m_player;
    for (size_t i = 0; i < m_loader->get_nr_cams(); i++) {
        m_loader->set_mask_for_cam("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_gl/data/mask_cam_"+std::to_string(i)+".png", i);

    }
    // m_loader->set_mask_for_cam("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_gl/data/mask_cam_1.png", 1);
    m_scene.m_view=m_view;

    init_params();


    m_player->play(m_bag_args); //the bag seems to need to be started from the main thread therwise it segment faults when starting playing..


     boost::thread t(&DataLoader::load_data, m_loader);

     // m_depth_estimator->run_speed_test();

     Frame dummy_frame;
     // Mesh depth_mesh=m_depth_estimator->compute_depth2(dummy_frame);
     // Mesh depth_mesh=m_depth_estimator->compute_depth_simplified();  // works on cpu
     // Mesh depth_mesh=m_depth_estimator_renegade->compute_depth(dummy_frame);  //just reads the things that were written from RENEGADE
     // // Mesh depth_mesh=m_depth_estimator_cl->compute_depth();
     // depth_mesh.m_show_points=true;
     // m_scene.add_mesh(depth_mesh, "depth_mesh");


     //good one
     m_depth_estimator_cl->init_data();
     m_depth_estimator_cl->compute_depth_and_create_mesh();
     // m_depth_estimator_cl->compute_depth_and_create_mesh_cpu();


     std::cout << "finished computing depth-------------------" << '\n';
     // Mesh depth_mesh=m_depth_estimator_gl2->compute_depth_simplified();  // works
     // depth_mesh.m_show_points=true;
     // m_scene.add_mesh(depth_mesh, "depth_mesh");



}


void Core::update() {

    // //hotload the shaders
    // std::vector<std::string> changed_files=dir_watcher.poll_files();
    // if(changed_files.size()>0){
    //     m_texturer->compile_shaders();
    //     Frame frame=m_loader->get_frame_for_cam(0); //TODO hotloading only works for one camera
    //     m_texturer->texture_scene(0, frame);
    // }


    if(m_depth_estimator_cl->is_modified()){
        std::string mesh_name="depth_mesh";
        Mesh depth_mesh=m_depth_estimator_cl->get_mesh();
        depth_mesh.m_show_points=true;
        depth_mesh.name=mesh_name;
        if(m_scene.does_mesh_with_name_exist(mesh_name)){
            m_scene.get_mesh_with_name(mesh_name)=depth_mesh; //it exists, just assign to it
        }else{
            m_scene.add_mesh(depth_mesh, mesh_name); //doesn't exist, add it to the scene
        }
    }


    if (m_loader->is_modified()) {

        int nr_cams_processed=0;
        for (size_t i = 0; i < m_loader->get_nr_cams(); i++) {
            if(m_loader->is_cam_modified(i)){
                // m_texturer->commit_pages();
                // Frame frame=m_loader->get_frame_for_cam(i); //get frame for cam i
                // // display_frame(frame);
                // TIME_START("upload frame");
                // m_texturer->upload_rgb_texture(frame.rgb);
                // m_texturer->upload_mask_texture(frame.mask);
                // m_texturer->upload_frame_classes_and_probs_texture(frame.classes, frame.probs);
                // TIME_END("upload frame");
                // m_texturer->texture_scene(m_scene.get_idx_for_name("geom"), frame);
                //
                // //update camera frustum mesh
                // std::string cam_name= "cam_" + std::to_string(frame.cam_id);
                // compute_camera_frustum_mesh(m_scene.get_mesh_with_name(cam_name), frame);
                // m_scene.get_mesh_with_name(cam_name).m_visualization_should_change=true;

                // //debug update the position of a certain debug point
                // Eigen::Vector3d eye_pos=frame.tf_cam_world.inverse().translation();
                // m_scene.get_mesh_with_name("debug_mesh").V=eye_pos.transpose();
                // m_scene.get_mesh_with_name("debug_mesh").m_visualization_should_change=true;


                // // cl depth
                // Frame frame=m_loader->get_frame_for_cam(i); //get frame for cam i
                // m_depth_estimator->compute_depth2(frame);
                // // display_frame(frame);



                // //renegade depth
                // Frame frame=m_loader->get_frame_for_cam(i); //get frame for cam i
                // Mesh depth_mesh=m_depth_estimator_renegade->compute_depth2(frame);
                // m_scene.add_mesh(depth_mesh, "depth_mesh");
                // display_frame(frame);

                //----------the good one
                // Frame frame=m_loader->get_frame_for_cam(i); //get frame for cam i
                // Mesh depth_mesh=m_depth_estimator->compute_depth2(frame);
                // depth_mesh.m_show_points=true;
                // m_scene.add_mesh(depth_mesh, "depth_mesh");
                // display_frame(frame);



                nr_cams_processed++;
            }
        }
        VLOG(1) << "nr_cams_processed " << nr_cams_processed;
    }


    TIME_START("set_mesh");
    for (size_t i = 0; i < m_scene.get_nr_meshes(); i++) {
        if(m_scene.get_mesh_with_idx(i).m_visualization_should_change){
            LOG_F(INFO, "Core:: saw that the scene was modified");
            m_view->selected_data_index=i;
            m_view->data().clear();

            if(m_scene.get_mesh_with_idx(i).m_is_visible){
                if (m_scene.get_mesh_with_idx(i).m_show_mesh) {
                    set_mesh(m_scene.get_mesh_with_idx(i));  // the scene is internally broken down into various independent meshes
                }
                if (m_scene.get_mesh_with_idx(i).m_show_points) {
                    set_points(m_scene.get_mesh_with_idx(i));
                }
                if (m_scene.get_mesh_with_idx(i).m_show_edges) {
                    set_edges(m_scene.get_mesh_with_idx(i));
                }
            }


            m_scene.get_mesh_with_idx(i).m_visualization_should_change=false;
        }
    }
    TIME_END("set_mesh");


    // if(m_player->is_paused() &&  m_player->m_player_should_continue_after_step){
    //     m_player->m_player_should_do_one_step=true; //so that when it starts the callback it puts the bag back into pause
    //     m_player->pause(); //starts the bag
    // }


    //render
    // if(m_surfel_rendering){
    //     m_splatter->render(m_scene.get_mesh_with_name("few_scans"));
    // }else{
    // if (m_viewer_initialized) {
    //     m_view->draw();
    // } else {
    //     m_view->core.clear_framebuffers(); //if there is no mesh to draw then just put the color of the background
    // }
    // }


}


void Core::init_params() {
    //read the parameters from the launch file
    ros::NodeHandle private_nh("~");

    //verbosity
    loguru::g_stderr_verbosity = getParamElseDefault<int>(private_nh, "loguru_verbosity", -1);

    m_bag_args = getParamElseThrow<std::string>(private_nh, "bag_args");

    //TODO read all the other parameters from the launch file
}


void Core::display_frame(const Frame& frame){
    int max_size=400;

    //resize to a reasonable size
    cv::Mat rgb_vis, classes_vis, probs_vis;
    double scale = static_cast<double>(max_size) / std::max(frame.rgb.rows, frame.rgb.cols);
    cv::Size size(frame.rgb.cols*scale, frame.rgb.rows*scale);
    cv::resize(frame.rgb, rgb_vis, size );
    cv::resize(frame.classes_original_size, classes_vis, size, 0, 0, cv::INTER_NEAREST );
    cv::resize(frame.probs_original_size, probs_vis, size, 0, 0, cv::INTER_NEAREST);

    //color them
    cv::Mat classes_vis_colored;


    //display
    cv::imshow("rgb", rgb_vis);
    // cv::imshow("probs", probs_vis);
    cv::waitKey(10);


}



Mesh Core::split_mesh_from_uv(const Mesh& mesh, const Mesh& uv_mesh){
    Mesh mesh_with_seams;

    Eigen::MatrixXd V_new(uv_mesh.V.rows(),3);  //need to duplicate points to account for the seams int the uv
    for (size_t i = 0; i < uv_mesh.V.rows(); i++) {
        int original_point_idx = (int)std::round(uv_mesh.V(i,2));
        V_new.row(i) = mesh.V.row(original_point_idx);
    }
    //also the faces
    Eigen::MatrixXi F_new=uv_mesh.F;

    //get the uv map
    Eigen::MatrixXd UV= uv_mesh.V.leftCols(2);

    // VLOG(1) << "min max x " << UV.col(0).minCoeff() << " " << UV.col(0).maxCoeff();
    // VLOG(1) << "min max y " << UV.col(1).minCoeff() << " " << UV.col(1).maxCoeff();
    UV.array()/=UV.maxCoeff();
    // UV.array()/=10.0f;


    mesh_with_seams.V=V_new;
    mesh_with_seams.F=F_new;
    mesh_with_seams.UV=UV;

    return mesh_with_seams;

}

Mesh Core::read_mesh_from_file(std::string file_path) {

   Mesh mesh;

   std::string fileExt = file_path.substr(file_path.find_last_of(".") + 1);
   if (fileExt == "off") {
       igl::readOFF(file_path, mesh.V, mesh.F);
   } else if (fileExt == "ply") {
       igl::readPLY(file_path, mesh.V, mesh.F, mesh.NV, mesh.UV);
   } else if (fileExt == "obj") {
       igl::readOBJ(file_path, mesh.V, mesh.F);
   }

   return mesh;
}

void Core::set_mesh(const Mesh &mesh) {
    if(mesh.is_empty() || mesh.F.rows()==0){
        return;
    }


   m_view->data().set_mesh(mesh.V, mesh.F);
   if (mesh.C.rows() == mesh.V.rows() || mesh.C.rows() == mesh.F.rows()) {
       m_view->data().set_colors(mesh.C);
   }else{
       m_view->data().set_colors(color_points(mesh));
   }

   if(mesh.UV.size()){
       m_view->data().set_uv(mesh.UV);
   }


   if (!m_viewer_initialized) {
       std::cout << "aligning camera" << '\n';
       m_viewer_initialized = true;
       m_view->core.align_camera_center(mesh.V, mesh.F);
   }
}

void Core::set_points(const Mesh &mesh) {
    if(mesh.is_empty()){
        return;
    }

    // if there are none, then make some colors based on height
   if (mesh.C.rows() != mesh.V.rows()) {
       m_view->data().set_points(mesh.V, color_points(mesh));
   } else {
       m_view->data().set_points(mesh.V, mesh.C);
   }

   m_view->data().point_size = 2;


   if (!m_viewer_initialized) {
       m_viewer_initialized = true;
       m_view->core.align_camera_center(mesh.V);
   }
}

void Core::set_edges(const Mesh &mesh) {
    if(mesh.is_empty()){
        return;
    }

    //make some colors
    Eigen::MatrixXd C(mesh.E.rows(), 3);
    for (size_t i = 0; i < C.rows(); i++) {
        C(i, 0) = 1.0;
        C(i, 1) = 0.0;
        C(i, 2) = 0.0;
    }

   m_view->data().set_edges(mesh.V, mesh.E, C);


   if (!m_viewer_initialized) {
       m_viewer_initialized = true;
       m_view->core.align_camera_center(mesh.V);
   }
}


Eigen::MatrixXd Core::color_points(const Mesh& mesh)const{
    Eigen::MatrixXd C = mesh.V;
    double min_y, max_y;
    min_y = mesh.V.col(1).minCoeff();
    max_y = mesh.V.col(1).maxCoeff();

    if(mesh.m_color_type==0){
        for (size_t i = 0; i < C.rows(); i++) {
            std::vector<float> color_vec = jet_color(mesh.V(i, 1), min_y, max_y);
            C(i, 0) = color_vec[0];
            C(i, 1) = color_vec[1];
            C(i, 2) = color_vec[2];
        }

    }else if(mesh.m_color_type==1){
        for (size_t i = 0; i < C.rows(); i++) {
             float gray_val = lerp(mesh.V(i,1), min_y, max_y, 0.0, 1.0 );
             C(i,0)=C(i,1)=C(i,2)=gray_val;
        }

    }else if(mesh.m_color_type==2){

        double min_d, max_d;
        min_d=mesh.D.minCoeff();
        max_d=mesh.D.maxCoeff();
        for (size_t i = 0; i < C.rows(); i++) {
            C(i,0)=C(i,1)=C(i,2)=  lerp(mesh.D(i), min_d, max_d, 0.0, 1.0 );;
        }

    }else if(mesh.m_color_type==3){
        Eigen::VectorXi C_idx(mesh.V.rows());
        C_idx.setLinSpaced(mesh.V.rows(), 0 ,mesh.V.rows()-1);
        C = C_idx.template cast<double>().replicate(1,3);
        //normalize the colors
        double max = C.maxCoeff();
        double min = C.minCoeff();
        double range = std::fabs(max) + std::fabs(min);
        double inv_range = 1.0/range;
        C.array() = (C.array() - min) *inv_range;

    //AO
    }else if(mesh.m_color_type==4){
        LOG(WARNING) << "not implemented yet";
        // int num_samples=500;
        // Eigen::VectorXd ao;
        // igl::embree::EmbreeIntersector ei;
        // ei.init(mesh.V.cast<float>(),mesh.F.cast<int>());
        // Eigen::MatrixXd N_vertices;
        // igl::per_vertex_normals(mesh.V, mesh.F, N_vertices);
        // igl::embree::ambient_occlusion(ei, mesh.V, N_vertices, num_samples, ao);
        // ao=1.0-ao.array(); //a0 is 1.0 in occluded places and 0.0 in non ocluded so we flip it to have 0.0 (dark) in occluded
        // C = ao.replicate(1,3);

    //AO(Gold)
    }else if(mesh.m_color_type==5){
        LOG(WARNING) << "not implemented yet";
        // int num_samples=500;
        // Eigen::VectorXd ao;
        // igl::embree::EmbreeIntersector ei;
        // ei.init(mesh.V.cast<float>(),mesh.F.cast<int>());
        // Eigen::MatrixXd N_vertices;
        // igl::per_vertex_normals(mesh.V, mesh.F, N_vertices);
        // igl::embree::ambient_occlusion(ei, mesh.V, N_vertices, num_samples, ao);
        // ao=1.0-ao.array(); //a0 is 1.0 in occluded places and 0.0 in non ocluded so we flip it to have 0.0 (dark) in occluded
        //
        // VLOG(1) << "C is size " << C.rows() << " " << C.cols();
        // VLOG(1) << "ao is size " << ao.rows() << " " << ao.cols();
        // C.col(0).setConstant(m_mesh_color(0));
        // C.col(1).setConstant(m_mesh_color(1));
        // C.col(2).setConstant(m_mesh_color(2));
        // VLOG(1) << "doing multiplication";
        // // C=C.transpose().array().colwise()*ao.array(); // I dunno why it doesnt fucking work
        // for (size_t i = 0; i < C.rows(); i++) {
        //     double ao_val=ao(i);
        //     for (size_t j = 0; j < C.cols(); j++) {
        //         C(i,j)=C(i,j)*ao_val;
        //     }
        // }


    //default
    }else if(mesh.m_color_type==6) {
        C.col(0).setConstant(0.41);
        C.col(1).setConstant(0.58);
        C.col(2).setConstant(0.59);

    //GOLD
    }else if(mesh.m_color_type==7) {
        C.col(0).setConstant(mesh.m_mesh_color(0));
        C.col(1).setConstant(mesh.m_mesh_color(1));
        C.col(2).setConstant(mesh.m_mesh_color(2));
    }

    return C;
}

void Core::write_ply(){
    int idx= m_view->selected_data_index;
    Mesh& mesh = m_scene.get_mesh_with_idx(idx);
    strcat (m_exported_filename,".ply");
    igl::writePLY(m_exported_filename, mesh.V, mesh.F);
}

void Core::write_obj(){
    int idx= m_view->selected_data_index;
    Mesh& mesh = m_scene.get_mesh_with_idx(idx);
    strcat (m_exported_filename,".obj");
    igl::writeOBJ(m_exported_filename, mesh.V, mesh.F);
}


void Core::compute_camera_frustum_mesh(Mesh& frustum_mesh, const Frame& frame){
    // https://gamedev.stackexchange.com/questions/29999/how-do-i-create-a-bounding-frustum-from-a-view-projection-matrix

    Eigen::Matrix4f proj=intrinsics_to_opengl_proj(frame.K.cast<double>(), frame.rgb.cols, frame.rgb.rows, 0.5, 2.5);
    Eigen::Matrix4f view= frame.tf_cam_world.matrix().cast<float>();
    Eigen::Matrix4f view_projection= proj*view;
    Eigen::Matrix4f view_projection_inv=view_projection.inverse();
    Eigen::MatrixXf frustum_V_in_NDC(8,3); //cube in range [-1,1]
    frustum_V_in_NDC <<
        // near face
        1, 1, -1,
        -1, 1, -1,
        -1, -1, -1,
        1, -1, -1,
        //far face
        1, 1, 1,
        -1, 1, 1,
        -1, -1, 1,
        1, -1, 1;

    //edges
    Eigen::MatrixXi E(12,2);
    E <<
        //near face
        0,1,
        1,2,
        2,3,
        3,0,
        //far face
        4,5,
        5,6,
        6,7,
        7,4,
        //in between
        0,4,
        5,1,
        6,2,
        7,3;


    // Eigen::MatrixXf frustum_in_world=frustum_V_in_NDC*view_projection_inv;
    Eigen::MatrixXf frustum_in_world=(view_projection_inv*frustum_V_in_NDC.transpose().colwise().homogeneous()).transpose();
    Eigen::MatrixXf frustrum_in_world_postw;
    // frustrum_in_world_postw=frustum_in_world.leftCols(3).array().rowwise()/frustum_in_world.col(3).transpose().array();
    frustrum_in_world_postw.resize(8,3);
    for (size_t i = 0; i < frustum_in_world.rows(); i++) {
        float w=frustum_in_world(i,3);
        for (size_t j = 0; j < 3; j++) {
            frustrum_in_world_postw(i,j)=frustum_in_world(i,j)/w;
        }
    }
    // std::cout << "frustrum_in_world_postw is " << frustrum_in_world_postw.rows() << " " << frustrum_in_world_postw.cols() << '\n';

    frustum_mesh.V=frustrum_in_world_postw.cast<double>();
    frustum_mesh.E=E;

    // std::cout << "frustum mesh is " <<  frustum_mesh << '\n';


}
