//C++
#include <iostream>
#include <random>

#include <Eigen/Dense>

//My stuff
#include "stereo_depth_gl/Core.h"
#include "stereo_depth_gl/MiscUtils.h"
#include "stereo_depth_gl/Profiler.h"
// #include "stereo_depth_gl/RosBagPlayer.h"
// #include "stereo_depth_gl/DepthEstimatorCPU.h"
// #include "stereo_depth_gl/DepthEstimatorRenegade.h"
#include "stereo_depth_gl/DepthEstimatorGL.h"
#ifdef WITH_HALIDE
    #include "stereo_depth_gl/DepthEstimatorHalide.h"
#endif
// #include "stereo_depth_gl/DepthEstimatorGL2.h"
// #include "stereo_depth_gl/DataLoader.h"
#ifdef WITH_LOADER_PNG
    #include "stereo_depth_gl/DataLoaderPNG.h"
#endif
#include "stereo_depth_gl/DataLoaderRos.h"
// #include "stereo_depth_gl/SurfelSplatter.h"


//libigl
#ifdef WITH_VIEWER
#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOBJ.h>
#include <igl/readPLY.h>
#include <igl/writePLY.h>
#include <igl/writeOBJ.h>
#include <igl/per_vertex_normals.h>

//pcd for writing pcd files
#include <pcl/io/pcd_io.h>

#endif
// #include <igl/embree/ambient_occlusion.h>
// #include <igl/embree/EmbreeIntersector.h>

//ROS
#include "stereo_depth_gl/RosTools.h"

//gl
#include "UtilsGL.h"

//boost
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

// //configuru
// #define CONFIGURU_WITH_EIGEN 1
// #define CONFIGURU_IMPLICIT_CONVERSIONS 1
// #include <configuru.hpp>
using namespace configuru;


Core::Core() :
        m_viewer_initialized(false),
        // m_player(new RosBagPlayer),
        // m_depth_estimator(new DepthEstimatorCPU),
        // m_depth_estimator_renegade(new DepthEstimatorRenegade),
        m_depth_estimator_gl(new DepthEstimatorGL),
        // m_depth_estimator_halide(new DepthEstimatorHalide),
        // m_depth_estimator_gl2(new DepthEstimatorGL2),
        // m_loader(new DataLoader),
        m_loader_ros(new DataLoaderRos),
        // m_splatter(new SurfelSplatter),
        m_nr_callbacks(0),
        // dir_watcher("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_gl/shaders/",5),
        m_player_paused(true),
        m_player_should_do_one_step(false),
        m_preload_mesh_subsample_factor(1),
        m_magnification(1.0){

    init_params();

    #ifdef WITH_HALIDE
    m_depth_estimator_halide=std::shared_ptr<DepthEstimatorHalide>(new DepthEstimatorHalide);
    #endif

    #ifdef WITH_LOADER_PNG
    m_loader_png=std::shared_ptr<DataLoaderPNG>(new DataLoaderPNG);
    #endif

}

void Core::init_links(){

    m_depth_estimator_gl->m_profiler=m_profiler;

    #ifdef WITH_HALIDE
        m_depth_estimator_halide->m_profiler=m_profiler;
    #endif

    #ifdef WITH_LOADER_PNG
        m_loader_png->m_profiler=m_profiler;
    #endif

    m_loader_ros->m_profiler=m_profiler;

    #ifdef WITH_VIEWER
        m_scene.m_view=m_view;
    #endif

}

void Core::start(){

    init_links(); //link all the object to what they need

    #ifdef WITH_LOADER_PNG
        m_loader_png->start_reading(); //can only do it from here because only now we have linked with the profiler
    #endif

    m_loader_ros->start_reading();

     // //add a mesh for the camera frustum (WORKS FOR THE BAG LOADER AND MULTIPLE CAMERAS)
     #ifdef WITH_VIEWER
        #ifdef WITH_LOADER_PNG
         for (size_t i = 0; i < m_loader_png->get_nr_cams(); i++) {
             Mesh mesh_cam_frustum;
             mesh_cam_frustum.m_show_edges=true;
             std::string cam_name="cam_" + std::to_string(i);
             m_scene.add_mesh(mesh_cam_frustum, cam_name);
         }
         #endif

         //add also a preloaded mesh if needed
         if(m_preload_mesh){
             Mesh mesh=read_mesh_from_file(m_preload_mesh_path);
             std::cout << "Mesh is " << mesh << '\n';
             mesh=subsample_point_cloud(mesh);
             mesh.m_show_points=true;
             mesh.m_color_type=0; //jetcolor
             // mesh.m_color_type=1; //graysclae

            //  //apply some correction given by a matrix from Jan. The poses from eth are fucked up for some reason
            //  Eigen::Affine3d correction;
            //  correction.linear()<< 0.235, -0.356, 0.904,
            //                        -0.971, -0.116, 0.207,
            //                         0.031, -0.927, -0.373;
            // correction.translation() << 0.85, 2.192, 0.938;
            // VLOG(1) << "geom mesh correction " << correction.matrix();
            // mesh.apply_transform(correction.inverse());

             m_scene.add_mesh(mesh,"geom");
         }

         // add a mesh for the path of the agent
         Mesh path_mesh=show_poses_as_mesh(m_loader_png->m_worldROS_baselink_vec);
         m_scene.add_mesh(path_mesh, "path");
     #endif



}


void Core::update() {

    // //hotload the shaders
    // std::vector<std::string> changed_files=dir_watcher.poll_files();
    // if(changed_files.size()>0){
    //     m_texturer->compile_shaders();
    //     Frame frame=m_loader->get_frame_for_cam(0); //TODO hotloading only works for one camera
    //     m_texturer->texture_scene(0, frame);
    // }


    // // if ( m_loader_png->has_data_for_all_cams() ) {
    // if( m_loader_png->has_data_for_all_cams()  &&  (!m_player_paused || m_player_should_do_one_step ) ){
    //     m_player_should_do_one_step=false;
    //     Frame frame_left=m_loader_png->get_next_frame_for_cam(0);
    //     Frame frame_right=m_loader_png->get_next_frame_for_cam(1);
    //
    //     //halide one
    //     m_depth_estimator_halide->compute_depth(frame_left,frame_right);
    //     // m_depth_estimator_halide->test_opencl_example(frame_left,frame_right);
    //     // m_depth_estimator_halide->test_again(frame_left,frame_right);
    //     // m_depth_estimator_halide->test_again_real_imgs(frame_left,frame_right);
    //     //to visualize what halide is doing
    //     m_depth_estimator_gl->upload_gray_stereo_pair(m_depth_estimator_halide->debug_img_left, m_depth_estimator_halide->debug_img_right);
    //
    // }






    // if ( m_loader_png->has_data_for_all_cams() ) {
    #ifdef WITH_LOADER_PNG
    if( m_loader_png->has_data_for_all_cams()  &&  (!m_player_paused || m_player_should_do_one_step ) ){
        m_player_should_do_one_step=false;
        Frame frame_left=m_loader_png->get_next_frame_for_cam(0);
        Frame frame_right=m_loader_png->get_next_frame_for_cam(1);

        #ifdef WITH_VIEWER
            m_depth_estimator_gl->upload_rgb_stereo_pair(frame_left.rgb, frame_right.rgb);
        #endif


        //don't do anything with this but rather just republish it
        if(frame_left.frame_idx%50==0){
            frame_left.is_keyframe=true;
            frame_right.is_keyframe=true;
        }
        // m_loader_ros->publish_stereo_frame(frame_left, frame_right);
        m_loader_ros->publish_single_frame(frame_left);
        m_loader_ros->publish_single_frame(frame_right);

    }
    #endif

    if(  m_loader_ros->has_data_for_all_cams()  ){

        Frame frame_left=m_loader_ros->get_next_frame_for_cam(0);
        Frame frame_right=m_loader_ros->get_next_frame_for_cam(1);

        m_depth_estimator_gl->compute_depth_and_update_mesh_stereo(frame_left,frame_right);


        //update mesh from the debug icl_incremental
        Mesh point_cloud=m_depth_estimator_gl->m_mesh;
        m_loader_ros->publish_map(point_cloud);
        #ifdef WITH_VIEWER
            std::string cloud_name="point_cloud";
            point_cloud.name=cloud_name;
            point_cloud.m_show_points=true;
            if(m_scene.does_mesh_with_name_exist(cloud_name)){
                m_scene.get_mesh_with_name(cloud_name)=point_cloud; //it exists, just assign to it
            }else{
                m_scene.add_mesh(point_cloud, cloud_name); //doesn't exist, add it to the scene
            }
        #endif
        if(m_accumulate_meshes && m_depth_estimator_gl->m_started_new_keyframe){
            Mesh last_cloud=m_depth_estimator_gl->m_last_finished_mesh;
            m_loader_ros->publish_map_finished(last_cloud);
            #ifdef WITH_VIEWER
                std::string cloud_name="finished_cloud";
                last_cloud.name=cloud_name;
                last_cloud.m_show_points=true;

                if(m_merge_meshes){
                    m_scene.merge_mesh(last_cloud,cloud_name);
                }else{
                    m_scene.add_mesh(last_cloud, cloud_name);
                }

            #endif
        }



        #ifdef WITH_VIEWER
        #ifdef WITH_LOADER_PNG
        //update camera frustum mesh
        for (size_t cam_id = 0; cam_id < m_loader_png->get_nr_cams(); cam_id++) {
            std::string cam_name= "cam_"+std::to_string(cam_id);
            Mesh new_frustum_mesh;
            if(cam_id==0){
                new_frustum_mesh=compute_camera_frustum_mesh(frame_left, m_frustum_scale_multiplier);
            }else if( cam_id==1){
                new_frustum_mesh=compute_camera_frustum_mesh(frame_right, m_frustum_scale_multiplier);
            }
            // new_frustum_mesh=compute_camera_frustum_mesh(frame_left, m_frustum_scale_multiplier);
            new_frustum_mesh.name=cam_name;
            m_scene.get_mesh_with_name(cam_name)=new_frustum_mesh;
            m_scene.get_mesh_with_name(cam_name).m_visualization_should_change=true;
        }
        #endif
        #endif
    }







    #ifdef WITH_VIEWER
    glActiveTexture(GL_TEXTURE0);
    for (size_t i = 0; i < m_scene.get_nr_meshes(); i++) {
        if(m_scene.get_mesh_with_idx(i).m_visualization_should_change){
            LOG_F(INFO, "Core:: saw that the scene was modified");
            m_view->selected_data_index=i;
            m_view->data().clear();

            Mesh mesh=m_scene.get_mesh_with_idx(i);
            //TIME_START("set_mesh");
            if(mesh.m_is_visible){
                if(m_do_transform_mesh_to_worlGL){
                   mesh.apply_transform(m_loader_ros->m_tf_worldGL_worldROS.cast<double>());
                }
                if (mesh.m_show_mesh) {
                    set_mesh(mesh);  // the scene is internally broken down into various independent meshes
                }
                if (mesh.m_show_points) {
                    set_points(mesh);
                }
                if (mesh.m_show_edges) {
                    set_edges(mesh);
                }
            }
            //TIME_END("set_mesh");

            m_scene.get_mesh_with_idx(i).m_visualization_should_change=false; //need to set it for the mesh in the scene because this mesh is no longer a reference but a copy. We made it a copy so as to be able to apply one time m_do_transform_mesh_to_worlGL otherwise they would accumular on top of the mesh
            mesh.m_visualization_should_change=false;
        }
    }
    #endif






}


void Core::init_params() {
    //get the config filename
    ros::NodeHandle private_nh("~");
    std::string config_file= getParamElseThrow<std::string>(private_nh, "config_file");

    //read core the parameters
    Config cfg = configuru::parse_file(std::string(CMAKE_SOURCE_DIR)+"/config/"+config_file, CFG);
    Config core_cfg = cfg["core"];
    loguru::g_stderr_verbosity = core_cfg["loguru_verbosity"];

    //read visualziation params
    Config vis_cfg = cfg["visualization"];
    m_frustum_scale_multiplier=vis_cfg["frustum_scale_multiplier"];
    m_preload_mesh=vis_cfg["preload_mesh"];
    m_preload_mesh_path=(std::string)vis_cfg["preload_mesh_path"];
    m_preload_mesh_subsample_factor=vis_cfg["preload_mesh_subsample_factor"];
    m_do_transform_mesh_to_worlGL=vis_cfg["do_transform_mesh_to_worlGL"];
    m_accumulate_meshes=vis_cfg["accumulate_meshes"];
    m_merge_meshes=vis_cfg["merge_meshes"];

    Config loader_cfg = cfg["loader"];
    m_player_paused=loader_cfg["player_paused"];




    //TODO read all the other parameters from the launch file
}


#ifdef WITH_VIEWER

    Mesh Core::read_mesh_from_file(std::string file_path) {

       Mesh mesh;

       std::string fileExt = file_path.substr(file_path.find_last_of(".") + 1);
       if (fileExt == "off") {
           igl::readOFF(file_path, mesh.V, mesh.F);
       } else if (fileExt == "ply") {
           igl::readPLY(file_path, mesh.V, mesh.F, mesh.NV, mesh.UV, mesh.C);
           mesh.C/=255.0;
       } else if (fileExt == "obj") {
           igl::readOBJ(file_path, mesh.V, mesh.F);
       } else if(fileExt == "pcd"){
           pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
           int ret=pcl::io::loadPCDFile<pcl::PointXYZ> (file_path, *cloud);
           if (ret==-1){
               LOG(ERROR) << "Couldn't read file " << file_path;
           }
           mesh.V.resize(cloud->points.size(),3);
           for (size_t i = 0; i < cloud->points.size (); ++i){
               mesh.V.row(i) << cloud->points[i].x,cloud->points[i].y,cloud->points[i].z;
           }
       }else{
           LOG(ERROR) << "Cannot read mesh with extension " << fileExt;
       }

       return mesh;
    }



    void Core::set_mesh(const Mesh &mesh) {
        if(mesh.is_empty() || mesh.F.rows()==0){
            VLOG(1) << "set_mesh: returning because mesh " << mesh.name << " is empty";
            return;
        }
        VLOG(1) << "Setting mesh :  mesh " << mesh.name;


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
          VLOG(1) << "aligning camera";
          m_viewer_initialized = true;
          m_view->core.align_camera_center(mesh.V, mesh.F);
       }
    }

    void Core::set_points(const Mesh &mesh) {
        if(mesh.is_empty()){
            VLOG(1) << "set_points: returning because mesh " << mesh.name << " is empty";
            return;
        }
        VLOG(1) << "Setting points :  mesh " << mesh.name;


        // if there are none, then make some colors based on height
       if (mesh.C.rows() != mesh.V.rows()) {
           m_view->data().set_points(mesh.V, color_points(mesh));
       } else {
           m_view->data().set_points(mesh.V, mesh.C);
       }

       m_view->data().point_size = 2;

       if (!m_viewer_initialized) {
           VLOG(1) << "aligning camera";
           m_viewer_initialized = true;
           m_view->core.align_camera_center(mesh.V);
       }
    }

    void Core::set_edges(const Mesh &mesh) {
        if(mesh.is_empty()){
            VLOG(1) << "set_edges: returning because mesh " << mesh.name << " is empty";
            return;
        }
        VLOG(1) << "Setting edges :  mesh " << mesh.name;

        //make some colors
        Eigen::MatrixXd C(mesh.E.rows(), 3);
        for (size_t i = 0; i < C.rows(); i++) {
            C(i, 0) = 1.0;
            C(i, 1) = 0.0;
            C(i, 2) = 0.0;
        }

       m_view->data().set_edges(mesh.V, mesh.E, C);


       if (!m_viewer_initialized) {
           VLOG(1) << "aligning camera";
           m_viewer_initialized = true;
           m_view->core.align_camera_center(mesh.V);
       }
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

    void Core::write_pcd(){
        int idx= m_view->selected_data_index;
        Mesh& mesh = m_scene.get_mesh_with_idx(idx);
        strcat (m_exported_filename,".pcd");

        //make cloud
        pcl::PointCloud<pcl::PointXYZ> cloud;
        cloud.width    = mesh.V.rows();
        cloud.height   = 1;
        cloud.is_dense = false;
        cloud.points.resize (mesh.V.rows());
        for (size_t i = 0; i < mesh.V.rows(); i++) {
            cloud.points[i].x=mesh.V(i,0);
            cloud.points[i].y=mesh.V(i,1);
            cloud.points[i].z=mesh.V(i,2);
        }

        pcl::io::savePCDFileASCII (m_exported_filename, cloud);
    }

#endif

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




Mesh Core::compute_camera_frustum_mesh(const Frame& frame, const float scale_multiplier){
    // https://gamedev.stackexchange.com/questions/29999/how-do-i-create-a-bounding-frustum-from-a-view-projection-matrix
    Mesh frustum_mesh;

    Eigen::Matrix4f proj=intrinsics_to_opengl_proj(frame.K, frame.gray.cols, frame.gray.rows, 0.5*scale_multiplier, 2.5*scale_multiplier);
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
    frustum_mesh.m_show_mesh=false;
    frustum_mesh.m_show_edges=true;

    // std::cout << "frustum mesh is " <<  frustum_mesh << '\n';

    return frustum_mesh;

}

Mesh Core::subsample_point_cloud(const Mesh& mesh){

    Mesh mesh_subsampled;
    if(m_preload_mesh_subsample_factor<=1){
        mesh_subsampled=mesh;
    }else{
        int nr_points_to_keep=mesh.V.rows()/m_preload_mesh_subsample_factor;
        mesh_subsampled.V.resize(nr_points_to_keep,3);
        mesh_subsampled.V.setZero();
        mesh_subsampled.C.resize(nr_points_to_keep,3);
        mesh_subsampled.C.setZero();

        std::random_device rd;     // only used once to initialise (seed) engine
        std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
        std::uniform_int_distribution<int> uni(0,mesh.V.rows()-1); // guaranteed unbiased



        for (size_t i = 0; i < nr_points_to_keep; i++) {
            //get random point
            int random_idx = uni(rng);

            //copy it into the mesh_subsampled
            mesh_subsampled.V.row(i)=mesh.V.row(random_idx);
            mesh_subsampled.C.row(i)=mesh.C.row(random_idx);
        }
    }



    return mesh_subsampled;

}

Mesh Core::show_poses_as_mesh(const std::vector<std::pair<uint64_t, Eigen::Affine3f> >&  worldROS_baselink_vec){

    Mesh path_mesh;

    //Make vertices as many as poses
    Eigen::MatrixXd V(worldROS_baselink_vec.size(),3);

    for (size_t i = 0; i < worldROS_baselink_vec.size(); i++) {
        Eigen::Affine3f tf_worldGL_baselink=  worldROS_baselink_vec[i].second;
        Eigen::Vector3f translation=tf_worldGL_baselink.translation();
        V.row(i)=translation.cast<double>();
    }

    //edges in between pairs
    Eigen::MatrixXi E(V.rows()-1,2);
    for (size_t i = 0; i < V.rows()-1; i++) {
        E.row(i) << i, i+1;
    }


    path_mesh.V=V;
    path_mesh.E=E;

    path_mesh.m_show_points=false;
    path_mesh.m_show_edges=true;

    return path_mesh;
}


///-------MOVIES--------------------------------------------------------------------------------


void Core::write_single_png(){
    fs::path dir (m_results_path);
    fs::path png_name (m_single_png_filename);
    fs::path full_path = dir / png_name;
    fs::create_directory(dir);
    std::cout << " write_single_png: " << full_path << std::endl;

    write_viewer_to_png(m_view, full_path.string(), m_magnification);

}


void Core::write_viewer_to_png(const std::shared_ptr<igl::opengl::glfw::Viewer> view, const std::string full_path, const float magnification ){
    std::cout << "writing viewer to png " << '\n';


    int width  = std::round(view->core.viewport(2));
    int height = std::round(view->core.viewport(3));
    width*=magnification;
    height*=magnification;


    // unsigned x = R.rows();
    // unsigned y = R.cols();

    // Create frame buffer
    GLuint frameBuffer;
    glGenFramebuffers(1, &frameBuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);

    // Create texture to hold color buffer
    GLuint texColorBuffer;
    glGenTextures(1, &texColorBuffer);
    glBindTexture(GL_TEXTURE_2D, texColorBuffer);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texColorBuffer, 0);

    // Create Renderbuffer Object to hold depth and stencil buffers
    GLuint rboDepthStencil;
    glGenRenderbuffers(1, &rboDepthStencil);
    glBindRenderbuffer(GL_RENDERBUFFER, rboDepthStencil);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rboDepthStencil);

    assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);



    std::cout << "neww" << '\n';
    glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
    std::cout << "rendering" << '\n';

    // Clear the buffer
    glClearColor(view->core.background_color(0), view->core.background_color(1), view->core.background_color(2), 0.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Save old viewport
    Eigen::Vector4f viewport_ori = view->core.viewport;
    view->core.viewport << 0,0,width,height;


    // Draw
    // view->core.draw(view->core.data,false);
    // view->draw();
    for(int i = 0;i<view->data_list.size();i++){
        view->core.draw(view->data_list[i],false);
    }

    // Restore viewport
    view->core.viewport = viewport_ori;

    // Copy back in the given Eigen matrices
    GLubyte* pixels = (GLubyte*)calloc(width*height*4,sizeof(GLubyte));
    glReadPixels
    (
     0, 0,
     width, height,
     GL_RGBA, GL_UNSIGNED_BYTE, pixels
     );

    cv::Mat img;
    img=cv::Mat(height,width,CV_8UC4);
    int count = 0;
    for (unsigned i=0; i<height; ++i){
      for (unsigned j=0; j<width; ++j){
        img.at<cv::Vec4b>(i,j)[2] = pixels[count*4+0];
        img.at<cv::Vec4b>(i,j)[1] = pixels[count*4+1];
        img.at<cv::Vec4b>(i,j)[0] = pixels[count*4+2];
        img.at<cv::Vec4b>(i,j)[3] = pixels[count*4+3];
        // alpha.at<uchar>(i,j) = pixels[count*4+3];
        // std::cout << "rgba is " << int(pixels[count*4+0]) << " " << int(pixels[count*4+1]) << " " << int(pixels[count*4+2])  << " " << int(pixels[count*4+3]) << '\n';
        ++count;
      }
    }



    // std::cout << "type of rgb image is" << type2string(rgba_img.type()) << '\n';


    cv::Mat final_img_flipped;
    cv::flip(img, final_img_flipped, 0);
    cv::imwrite(full_path,final_img_flipped);


    // Clean up
    free(pixels);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteRenderbuffers(1, &rboDepthStencil);
    glDeleteTextures(1, &texColorBuffer);
    glDeleteFramebuffers(1, &frameBuffer);


}
