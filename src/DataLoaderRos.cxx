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





DataLoaderRos::DataLoaderRos():
    m_nr_callbacks(0)
    {

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

    for (size_t i = 0; i < m_nr_cams; i++) {
        m_frames_buffer_per_cam.push_back( moodycamel::ReaderWriterQueue<Frame>(BUFFER_SIZE));
    }

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

    // VLOG(1) << "stereo_pair.img_gray_left.height and width is " << stereo_pair.img_gray_left.height << " " << stereo_pair.img_gray_left.width;
    // VLOG(1) << "stereo_pair.img_gray_right.height and width is " << stereo_pair.img_gray_right.height << " " << stereo_pair.img_gray_right.width;

    //Get images
    Frame frame_left, frame_right;
    cv_bridge::CvImageConstPtr cv_ptr;
    try{
        sensor_msgs::ImageConstPtr ptr_left( new sensor_msgs::Image( stereo_pair.img_gray_left ) );
        cv_ptr = cv_bridge::toCvShare( ptr_left );
        cv_ptr->image.copyTo(frame_left.gray);

        sensor_msgs::ImageConstPtr ptr_right( new sensor_msgs::Image( stereo_pair.img_gray_right ) );
        cv_ptr = cv_bridge::toCvShare( ptr_right );
        cv_ptr->image.copyTo(frame_right.gray);

        //cv::flip(img_cv,img_cv, -1); //TODO this line needs to be commented
    }catch (cv_bridge::Exception& e){
            ROS_ERROR( "cv_bridge exception: %s", e.what() );
            return;
    }

    // VLOG(1) << "Managed to read the images";

    //read poses
    frame_left.tf_cam_world.matrix() = Eigen::Map<Eigen::Matrix4f, Eigen::Unaligned>((float*)stereo_pair.tf_cam_world_left.data(), 4,4);
    frame_right.tf_cam_world.matrix() = Eigen::Map<Eigen::Matrix4f, Eigen::Unaligned>((float*)stereo_pair.tf_cam_world_right.data(), 4,4);
    // VLOG(1) << "loaded pose \n" << frame_left.tf_cam_world.matrix() ;

    //read K
    frame_left.K = Eigen::Map<Eigen::Matrix3f, Eigen::Unaligned>((float*)stereo_pair.K_left.data(), 3,3);
    frame_right.K = Eigen::Map<Eigen::Matrix3f, Eigen::Unaligned>((float*)stereo_pair.K_right.data(), 3,3);
    // VLOG(1) << "loaded K \n" << frame_left.K;

    frame_left.is_keyframe=stereo_pair.is_keyframe;
    frame_right.is_keyframe=stereo_pair.is_keyframe;

    frame_left.frame_idx=m_nr_callbacks;
    frame_right.frame_idx=m_nr_callbacks;

    frame_left.cam_id=0;
    frame_right.cam_id=1;


    //process it
    cv::Scharr( frame_left.gray, frame_left.grad_x, CV_32F, 1, 0);
    cv::Scharr( frame_left.gray, frame_left.grad_y, CV_32F, 0, 1);
    cv::Scharr( frame_right.gray, frame_right.grad_x, CV_32F, 1, 0);
    cv::Scharr( frame_right.gray, frame_right.grad_y, CV_32F, 0, 1);

    std::vector<cv::Mat> channels_left;
    channels_left.push_back(frame_left.gray);
    channels_left.push_back(frame_left.grad_x);
    channels_left.push_back(frame_left.grad_y);
    cv::merge(channels_left, frame_left.gray_with_gradients);

    std::vector<cv::Mat> channels_right;
    channels_right.push_back(frame_right.gray);
    channels_right.push_back(frame_right.grad_x);
    channels_right.push_back(frame_right.grad_y);
    cv::merge(channels_right, frame_right.gray_with_gradients);

    if(m_frames_buffer_per_cam[0].size_approx()<BUFFER_SIZE-1){ //there is enough space
        m_frames_buffer_per_cam[0].enqueue(frame_left);
        m_frames_buffer_per_cam[1].enqueue(frame_right);
    }

    m_nr_callbacks++;


    // int nr_frames_read_for_cam=0;


    // if(m_frames_buffer_per_cam[cam_id].size_approx()<BUFFER_SIZE-1){ //there is enough space
    //     //read the frame and everything else and push it to the queue
    //
    //
    //
    //     m_frames_buffer_per_cam[cam_id].enqueue(frame);
    //     nr_frames_read_for_cam++;
    //
    // }else{
    //
    // }


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
