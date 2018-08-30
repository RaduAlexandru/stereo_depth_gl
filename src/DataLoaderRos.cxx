#include "stereo_depth_gl/DataLoaderRos.h"

//c++
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>


//loguru
#define LOGURU_NO_DATE_TIME 1
#define LOGURU_NO_UPTIME 1
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp>

//My stuff
#include "stereo_depth_gl/Profiler.h"
#include "stereo_depth_gl/MiscUtils.h"

//cv
#include <cv_bridge/cv_bridge.h>

//ros
#include "stereo_depth_gl/RosTools.h"
// #include <message_filters/subscriber.h>

// //configuru
// #define CONFIGURU_WITH_EIGEN 1
// #define CONFIGURU_IMPLICIT_CONVERSIONS 1
// #include <configuru.hpp>
using namespace configuru;





DataLoaderRos::DataLoaderRos(){

    init_params();
    create_transformation_matrices();


    // read_pose_file();


    // init_params_configuru();

}

DataLoaderRos::~DataLoaderRos(){
    m_loader_thread.join();
}

void DataLoaderRos::init_params(){
    //get the config filename
    ros::NodeHandle private_nh("~");
    std::string config_file= getParamElseThrow<std::string>(private_nh, "config_file");

    //read all the parameters
    Config cfg = configuru::parse_file(std::string(CMAKE_SOURCE_DIR)+"/config/"+config_file, CFG);
    Config loader_config=cfg["loader_ros"];
    m_nr_cams = loader_config["nr_cams"];
    m_topic = (std::string)loader_config["topic"];


    Config vis_config=cfg["visualization"];
    m_tf_worldGL_worldROS_angle=vis_config["tf_worldGL_worldROS_angle"];
    m_tf_worldGL_worldROS_axis=(std::string)vis_config["tf_worldGL_worldROS_axis"];


}



void DataLoaderRos::start_reading(){

    VLOG(1) << "start_reading";

    //starts a thread that spins continously and reads stuff
    m_loader_thread=std::thread(&DataLoaderRos::read_data, this);


}

void DataLoaderRos::read_data(){
    loguru::set_thread_name("ros_thread");

    VLOG(1) << "subscribing";

    ros::NodeHandle private_nh("~");
    ros::Subscriber sub = private_nh.subscribe(m_topic, 100, &DataLoaderRos::callback, this);

    ros::spin();
}

void DataLoaderRos::callback(const stereo_ros_msg::StereoPair& stereo_pair){


    VLOG(1) << "callback";

    int nr_frames_read_for_cam=0;



    // // std::cout << "size approx is " << m_queue.size_approx() << '\n';
    // // std::cout << "m_idx_img_to_read is " << m_idx_img_to_read << '\n';
    // if(m_frames_buffer_per_cam[cam_id].size_approx()<BUFFER_SIZE-1){ //there is enough space
    //     //read the frame and everything else and push it to the queue
    //
    //
    //
    //     m_frames_buffer_per_cam[cam_id].enqueue(frame);
    //     nr_frames_read_for_cam++;
    //
    // }

}


bool DataLoaderRos::has_data_for_cam(const int cam_id){
    // return !m_queue.empty();
    if(m_frames_buffer_per_cam[cam_id].peek()==nullptr){
        return false;
    }else{
        return true;
    }
}

bool DataLoaderRos::has_data_for_all_cams(){
    for (size_t i = 0; i < m_nr_cams; i++) {
        if(!has_data_for_cam(i)){
            return false;
        }
    }
    return true;
}

Frame DataLoaderRos::get_next_frame_for_cam(const int cam_id){
    TIME_SCOPE("get_next_frame");


    Frame frame ;
    m_frames_buffer_per_cam[cam_id].try_dequeue(frame);


    return frame;

}

void DataLoaderRos::reset(){
    for (size_t i = 0; i < m_nr_cams; i++) {
        //deque everything (we can do it safely from here because while this is running, the core is not reading since the Core and GUI share thread)
        m_frames_buffer_per_cam[i]=moodycamel::ReaderWriterQueue<Frame>(BUFFER_SIZE);
    }

}

void DataLoaderRos::clear_buffers(){
    for (size_t i = 0; i < m_nr_cams; i++) {
        //deque everything (we can do it safely from here because while this is running, the core is not reading since the Core and GUI share thread)
        m_frames_buffer_per_cam[i]=moodycamel::ReaderWriterQueue<Frame>(BUFFER_SIZE);
    }
}

void DataLoaderRos::create_transformation_matrices(){



    /*
     *
     *
     *           Z
     *           |
     *           |       X
     *           |     /
     *           |   /
     *           | /
     *   Y-------
     *

     * ROS world frame
     * Explained here: http://www.ros.org/reps/rep-0103.html
     *
     * */


    m_tf_worldGL_worldROS.setIdentity();
    Eigen::Matrix3f worldGL_worldROS_rot;
    Eigen::Vector3f axis;
    if(m_tf_worldGL_worldROS_axis=="x"){
        axis=Eigen::Vector3f::UnitX();
    }else if(m_tf_worldGL_worldROS_axis=="y"){
        axis=Eigen::Vector3f::UnitY();
    }else if(m_tf_worldGL_worldROS_axis=="z"){
        axis=Eigen::Vector3f::UnitZ();
    }else{
        LOG(FATAL) << "No valid m_tf_worldGL_worldROS_axis. Need to be either x,y or z";
    }
    worldGL_worldROS_rot = Eigen::AngleAxisf(m_tf_worldGL_worldROS_angle, axis);
    m_tf_worldGL_worldROS.matrix().block<3,3>(0,0)=worldGL_worldROS_rot;




}
