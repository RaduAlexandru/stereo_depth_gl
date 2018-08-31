#pragma once
//C++
#include <iosfwd>
#include <memory>
#include <atomic>
#include <unordered_map>
#include <thread>
#include <math.h>       /* log2 */


//OpenCV
#include <opencv2/highgui/highgui.hpp>

//My stuff
#include "stereo_depth_gl/Frame.h"
#include "stereo_depth_gl/Mesh.h"

//ros
#include <ros/ros.h>
#include <stereo_ros_msg/StereoPair.h>

//readerwriterqueue
#include "readerwriterqueue/readerwriterqueue.h"

#define BUFFER_SIZE 16


//forward declarations
class Profiler;

class DataLoaderRos{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DataLoaderRos();
    ~DataLoaderRos();
    void start_reading();

    bool has_data_for_cam(const int cam_id);
    bool has_data_for_all_cams();
    Frame get_next_frame_for_cam(const int cam_id);
    int get_nr_cams(){return m_nr_cams; };


    // void read_data();
    // Frame get_next_frame();
    void reset(); //starts reading back from the start of the data log
    void clear_buffers(); //empties the ringbuffers, usefull for when scrolling through time

    void publish_stereo_frame(const Frame& frame_left, const Frame& frame_right);
    void publish_map(const Mesh& mesh);
    void publish_map_finished(const Mesh& mesh);

    //objects
    std::shared_ptr<Profiler> m_profiler;

    //transforms
    Eigen::Affine3f m_tf_worldGL_worldROS;




private:

    std::vector< moodycamel::ReaderWriterQueue<Frame> > m_frames_buffer_per_cam;

    std::thread m_loader_thread;
    int m_nr_cams;
    std::string m_topic;
    int m_nr_callbacks;
    // int m_idx_img_to_read;


    //params
    float m_tf_worldGL_worldROS_angle;
    std::string m_tf_worldGL_worldROS_axis;
    // std::string m_dataset_type;


    void init_params();
    void create_transformation_matrices();
    void read_data();
    void callback(const stereo_ros_msg::StereoPair& stereo_pair);

    //for testing that we can publish and receive ros messages correctly
    ros::Publisher m_stereo_publisher;
    ros::Publisher m_cloud_pub;
    ros::Publisher m_cloud_finished_pub;


};



#define TIME_SCOPE(name)\
    TIME_SCOPE_2(name,m_profiler);

#define TIME_START(name)\
    TIME_START_2(name,m_profiler);

#define TIME_END(name)\
    TIME_END_2(name,m_profiler);
