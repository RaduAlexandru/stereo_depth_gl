#pragma once
//C++
#include <iosfwd>
#include <atomic>
#include <mutex>

#include <memory>

//My stuff
#include "stereo_depth_cl/Mesh.h"
#include "stereo_depth_cl/Scene.h"
#include "stereo_depth_cl/Frame.h"


//ROS
#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/CompressedImage.h>
#include "stereo_depth_cl/point_types.h"

//dir watcher
#include "dir_watcher.hpp"

#define NUM_CLASSES 66

//forward declarations
class DepthEstimatorCL;
class Profiler;
class RosBagPlayer;
class DataLoader;
class SurfelSplatter;
namespace igl {  namespace opengl {namespace glfw{ class Viewer; }}}


#define TIME_SCOPE(name)\
    TIME_SCOPE_2(name,m_profiler);

#define TIME_START(name)\
    TIME_START_2(name,m_profiler);


class Core{
public:
    Core(std::shared_ptr<igl::opengl::glfw::Viewer> view, std::shared_ptr<Profiler> profiler);
    void update();
    void init_params();
    void init_ros();
    void read_ros_msgs();

    void read_scene(std::string file_path);
    void read_uv_map_thekla(std::string file_path);
    Mesh split_mesh_from_uv(const Mesh& mesh, const Mesh& uv_mesh);
    Mesh read_mesh_from_file(std::string file_path);
    void append_mesh(const Mesh& mesh, const std::string name);
    void set_mesh(const Mesh& mesh);
    void set_points(const Mesh& mesh);
    void set_edges(const Mesh& mesh);
    Eigen::MatrixXd color_points(const Mesh& mesh)const;  //creates color for all the points in the mesh depending on the m_color_type value
    void write_ply();
    void write_obj();

    void compute_camera_frustum_mesh(Mesh& frustum_mesh, const Frame& frame);


    //objects dependencies
    std::shared_ptr<igl::opengl::glfw::Viewer> m_view;
    std::shared_ptr<Profiler> m_profiler;
    std::shared_ptr<DepthEstimatorCL> m_depth_estimator;
    std::shared_ptr<DataLoader> m_loader;
    std::shared_ptr<RosBagPlayer> m_player;
    std::shared_ptr<SurfelSplatter> m_splatter;
    emilib::DelayedDirWatcher dir_watcher;


    //Misc
    Scene m_scene;
    // std::unordered_map<std::string> m_object_name2idx;  //for each object name in the scene we store the selected_data_index
    char m_exported_filename[32] = "./scene";
    uint64_t m_last_timestamp;
    int m_nr_callbacks;


    //ros
    std::string m_bag_args;



    bool m_viewer_initialized;
    std::mutex m_mutex_recon;
    bool m_surfel_rendering;


    //visualizer params

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
