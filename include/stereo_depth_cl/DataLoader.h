#pragma once
//C++
#include <iosfwd>
#include <memory>
#include <atomic>
#include <unordered_map>
#include <mutex>

//OpenCV
#include <opencv2/highgui/highgui.hpp>

//ROS
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <geometry_msgs/TransformStamped.h>
#include <eigen_conversions/eigen_msg.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_listener.h>

//My stuff
#include "stereo_depth_cl/Frame.h"
#include "stereo_depth_cl/ringbuffer.h"

//GL

#define NUM_DATA_BUFFER 5 // num of frames that we buffer for each cam

//forward declarations
class Profiler;
class RosBagPlayer;


class DataLoader{
public:
    DataLoader();
    void load_data();
    Frame get_frame_for_cam(int cam_id);
    int get_nr_cams();
    bool is_modified(){return m_loader_is_modified;};
    bool is_cam_modified(int cam_id){return m_cam_buffers_modified[cam_id];};
    void set_mask_for_cam(const std::string mask_filename, const int cam_id); //set a mask which will cause parts of the rgb, classes and probs images to be ignored


    //objects
    std::shared_ptr<Profiler> m_profiler;
    std::shared_ptr<RosBagPlayer> m_player;

    //transforms
    Eigen::Affine3f m_tf_alg_vel; //transformation from velodyne frame to the algorithm frame
    Eigen::Affine3f m_tf_baselink_vel;
    Eigen::Affine3f m_tf_worldGL_worldROS;
    std::unordered_map<uint64_t, Eigen::Affine3f> m_worldROS_baselink_map;

private:

    //ros
    int m_max_nr_threads;
    std::vector<ros::Subscriber> m_img_subs;
    int m_nr_cams;
    std::string m_frame_world;
    std::vector<std::string> m_cam_frames;
    std::vector<std::string> m_cam_img_topics;
    // std::vector<std::string> m_cam_info_topics;

    tf::TransformListener m_tf_listener;

    //databasse
    std::vector< ringbuffer<Frame,NUM_DATA_BUFFER> > m_frames_buffer_per_cam;
    std::vector<std::mutex> m_mutex_access_cam;
    std::vector<bool> m_cam_buffers_modified;
    // std::vector<Frame> m_frames;
    int m_finished_frame_idx; //idx pointing to the most recent finished scene
    int m_working_frame_idx; //idx poiting to the scene we are currently working on
    std::atomic<bool> m_loader_is_modified;
    std::vector<cv::Mat> m_masks;




    void init_params();
    void callback(const sensor_msgs::CompressedImageConstPtr &img_msg, const int cam_id);
    bool get_tf(Eigen::Affine3f& tf, const std::string& origin_frame, const std::string& dest_frame, const ros::Time query_time );
    void read_pose_file();
    bool get_pose_at_timestamp(Eigen::Affine3f& pose, uint64_t timestamp);
    void create_transformation_matrices();


};



#define TIME_SCOPE(name)\
    TIME_SCOPE_2(name,m_profiler);

#define TIME_START(name)\
    TIME_START_2(name,m_profiler);

#define TIME_END(name)\
    TIME_END_2(name,m_profiler);
