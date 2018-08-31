#pragma once
//C++
#include <iosfwd>
#include <atomic>
#include <mutex>

#include <memory>

//My stuff
#include "stereo_depth_gl/Mesh.h"
#include "stereo_depth_gl/Frame.h"
#ifdef WITH_VIEWER
    #include "stereo_depth_gl/Scene.h"
#endif


//ROS
#include <ros/ros.h>
// #include <pcl_ros/point_cloud.h>
// #include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/CompressedImage.h>
// #include "stereo_depth_gl/point_types.h"

//dir watcher
#include "dir_watcher.hpp"


#define NUM_CLASSES 66

//forward declarations
// class DepthEstimatorCPU;
// class DepthEstimatorRenegade;
class DepthEstimatorGL;
class DepthEstimatorHalide;
 // class DepthEstimatorGL2;
class Profiler;
// class RosBagPlayer;
// class DataLoader;
class DataLoaderPNG;
class DataLoaderRos;
// class SurfelSplatter;

#ifdef WITH_VIEWER
    namespace igl {  namespace opengl {namespace glfw{ class Viewer; }}}
#endif


#define TIME_SCOPE(name)\
    TIME_SCOPE_2(name,m_profiler);

#define TIME_START(name)\
    TIME_START_2(name,m_profiler);


class Core{
public:
    Core();

    void update();
    void init_params();
    void init_links(); //links the object to other object that they need
    void init_ros();
    void start(); // to be executed by main after all the core objcts have been linked together correctly

    void read_ros_msgs();

    void read_scene(std::string file_path);
    void read_uv_map_thekla(std::string file_path);
    Mesh split_mesh_from_uv(const Mesh& mesh, const Mesh& uv_mesh);
    Mesh read_mesh_from_file(std::string file_path);
    void append_mesh(const Mesh& mesh, const std::string name);
    #ifdef WITH_VIEWER
        void set_mesh(const Mesh& mesh);
        void set_points(const Mesh& mesh);
        void set_edges(const Mesh& mesh);
        void write_ply();
        void write_obj();
    #endif
    Eigen::MatrixXd color_points(const Mesh& mesh)const;  //creates color for all the points in the mesh depending on the m_color_type value


    Mesh compute_camera_frustum_mesh(const Frame& frame, const float scale_multiplier);
    Mesh subsample_point_cloud(const Mesh& mesh);


    //objects dependencies
    #ifdef WITH_VIEWER
        Scene m_scene;
        std::shared_ptr<igl::opengl::glfw::Viewer> m_view;
    #endif
    std::shared_ptr<Profiler> m_profiler;
    // std::shared_ptr<DepthEstimatorCPU> m_depth_estimator; //does a cpu implementation
    // std::shared_ptr<DepthEstimatorRenegade> m_depth_estimator_renegade; //just reads the file written by renegade
    std::shared_ptr<DepthEstimatorGL> m_depth_estimator_gl;
    std::shared_ptr<DepthEstimatorHalide> m_depth_estimator_halide;
    // std::shared_ptr<DepthEstimatorGL2> m_depth_estimator_gl2;
    // std::shared_ptr<DataLoader> m_loader;
    std::shared_ptr<DataLoaderPNG> m_loader_png;
    std::shared_ptr<DataLoaderRos> m_loader_ros;
    // std::shared_ptr<RosBagPlayer> m_player;
    // std::shared_ptr<SurfelSplatter> m_splatter;
    emilib::DelayedDirWatcher dir_watcher;


    //Misc
    // std::unordered_map<std::string> m_object_name2idx;  //for each object name in the scene we store the selected_data_index
    char m_exported_filename[32] = "./scene";
    uint64_t m_last_timestamp;
    int m_nr_callbacks;
    bool m_viewer_initialized;


    //visualizer params
    float m_frustum_scale_multiplier;
    bool m_show_sensor_poses;
    const char* m_color_types_desc[8] =
      {
              "Jet color",
              "Gray scale",
              "Distance to sensor",
              "By idx in the V vector",
              "Ambient occlusion",
              "AO(Gold)",
              "Default",
              "Gold"
      };


      //params
      bool m_do_transform_mesh_to_worlGL;
      bool m_preload_mesh;
      std::string m_preload_mesh_path;
      int m_preload_mesh_subsample_factor; //keep only a 1/subsample factor of the points in the preload mesh
      bool m_player_paused;
      bool m_player_should_do_one_step;
      bool m_accumulate_meshes;


private:

    // void callback(const sensor_msgs::ImageConstPtr& img_msg, const sensor_msgs::CameraInfoConstPtr& cam_info_msg, const sensor_msgs::PointCloud2ConstPtr& cloud_msg);
    //without cam info
    // void callback(const sensor_msgs::CompressedImageConstPtr& img_msg, const sensor_msgs::PointCloud2ConstPtr& cloud_msg);
    // only RGB
    void display_frame(const Frame& frame);

};


#define TIME_SCOPE(name)\
    TIME_SCOPE_2(name,m_profiler);

#define TIME_START(name)\
    TIME_START_2(name,m_profiler);

#define TIME_END(name)\
    TIME_END_2(name,m_profiler);
