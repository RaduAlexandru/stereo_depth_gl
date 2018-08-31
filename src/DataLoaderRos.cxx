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
#include <stereo_ros_msg/StereoPair.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>

// #include <message_filters/subscriber.h>

// //configuru
// #define CONFIGURU_WITH_EIGEN 1
// #define CONFIGURU_IMPLICIT_CONVERSIONS 1
// #include <configuru.hpp>
using namespace configuru;





DataLoaderRos::DataLoaderRos()
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
    m_topic_per_cam.resize(m_nr_cams);
    for (size_t cam_id = 0; cam_id < m_nr_cams; cam_id++) {
        m_topic_per_cam[cam_id] = (std::string)loader_config["topic_cam_"+std::to_string(cam_id)];
    }


    Config vis_config=cfg["visualization"];
    m_tf_worldGL_worldROS_angle=vis_config["tf_worldGL_worldROS_angle"];
    m_tf_worldGL_worldROS_axis=(std::string)vis_config["tf_worldGL_worldROS_axis"];


}



void DataLoaderRos::start_reading(){

    VLOG(1) << "publshing thingies ";
    ros::NodeHandle n("~");
    m_single_img_publisher_per_cam.resize(m_nr_cams);
    for (size_t i = 0; i < m_nr_cams; i++) {
        m_single_img_publisher_per_cam[i] = n.advertise< stereo_ros_msg::ImgWithPose >( m_topic_per_cam[i] , 5 );
    }
    m_sub_per_cam.resize(m_nr_cams);


    VLOG(1) << "start_reading";
    m_nr_callbacks_per_cam.resize(m_nr_cams,0);
    for (size_t i = 0; i < m_nr_cams; i++) {
        m_frames_buffer_per_cam.push_back( moodycamel::ReaderWriterQueue<Frame>(BUFFER_SIZE));
    }



    //starts a thread that spins continously and reads stuff
    m_loader_thread=std::thread(&DataLoaderRos::read_data, this);

}


void DataLoaderRos::read_data(){

    //multithread spinner
    for (size_t cam_id = 0; cam_id < m_nr_cams; cam_id++) {
        VLOG(1) << "subscribing to " << m_topic_per_cam[cam_id];
        ros::NodeHandle private_nh("~");
        m_sub_per_cam[cam_id]=private_nh.subscribe(m_topic_per_cam[cam_id], 100, &DataLoaderRos::callback_single_cam, this);
    }
    //multithreaded spinning, each callback (from different cameras) will run on different threads in paralel
    ros::MultiThreadedSpinner spinner(std::min(m_nr_cams, 4)); // Use 4 threads
    spinner.spin();
}


void DataLoaderRos::callback_single_cam(const stereo_ros_msg::ImgWithPose& img_msg){


    VLOG(1) << "callback from single cam " << img_msg.cam_id;


    //Get images
    Frame frame;
    cv_bridge::CvImageConstPtr cv_ptr;
    try{
        sensor_msgs::ImageConstPtr ptr( new sensor_msgs::Image( img_msg.img_gray ) );
        cv_ptr = cv_bridge::toCvShare( ptr );
        cv_ptr->image.copyTo(frame.gray);

        //cv::flip(img_cv,img_cv, -1); //TODO this line needs to be commented
    }catch (cv_bridge::Exception& e){
        ROS_ERROR( "cv_bridge exception: %s", e.what() );
        return;
    }



    //read poses
    frame.tf_cam_world.matrix() = Eigen::Map<Eigen::Matrix4f, Eigen::Unaligned>((float*)img_msg.tf_cam_world.data(), 4,4);
    VLOG(4) << "loaded pose \n" << frame.tf_cam_world.matrix() ;

    //read K
    frame.K = Eigen::Map<Eigen::Matrix3f, Eigen::Unaligned>((float*)img_msg.K.data(), 3,3);
    VLOG(4) << "loaded K \n" << frame.K;

    frame.is_keyframe=img_msg.is_keyframe;

    frame.frame_idx=m_nr_callbacks_per_cam[img_msg.cam_id];

    frame.cam_id=img_msg.cam_id;

    frame.min_depth=img_msg.min_depth;
    frame.mean_depth=img_msg.mean_depth;
    frame.ngf_eta=img_msg.ngf_eta;



    //process it
    cv::Scharr( frame.gray, frame.grad_x, CV_32F, 1, 0);
    cv::Scharr( frame.gray, frame.grad_y, CV_32F, 0, 1);


    std::vector<cv::Mat> channels;
    channels.push_back(frame.gray);
    channels.push_back(frame.grad_x);
    channels.push_back(frame.grad_y);
    cv::merge(channels, frame.gray_with_gradients);


    if(m_frames_buffer_per_cam[img_msg.cam_id].size_approx()<BUFFER_SIZE-1){ //there is enough space
        m_frames_buffer_per_cam[img_msg.cam_id].enqueue(frame);
    }

    m_nr_callbacks_per_cam[img_msg.cam_id]++;




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


void DataLoaderRos::publish_single_frame(const Frame& frame){
    VLOG(1) << "publishing the single frame "<< frame.cam_id;
    ros::NodeHandle n("~");


    stereo_ros_msg::ImgWithPose msg;

    cv_bridge::CvImage cv_msg;
    cv_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    cv_msg.image    = frame.gray; // Your cv::Mat
    msg.img_gray=*cv_msg.toImageMsg();
    VLOG(4) << "msg.img_gray.height and width is " << msg.img_gray.height << " " << msg.img_gray.width;


    //store the pose
    VLOG(4) << "storing pose \n" << frame.tf_cam_world.matrix();
    Eigen::Matrix4f::Map(msg.tf_cam_world.data(), 4,4) = frame.tf_cam_world.matrix();


    //store the K
    VLOG(4) << "storing K \n" << frame.K;
    Eigen::Matrix3f::Map(msg.K.data(), 3,3) = frame.K;

    msg.cam_id=frame.cam_id;

    msg.is_keyframe=frame.is_keyframe;

    msg.min_depth=frame.min_depth;
    msg.mean_depth=frame.mean_depth;
    msg.ngf_eta=frame.ngf_eta;



    m_single_img_publisher_per_cam[frame.cam_id].publish (msg);
}



void DataLoaderRos::publish_map(const Mesh& mesh){

    ros::NodeHandle n("~");
    m_cloud_pub = n.advertise<pcl::PointCloud<pcl::PointXYZRGB> > ( "semi_dense_map", 10 );


    if ( m_cloud_pub.getNumSubscribers() == 0 ) return;
    const char * MAP_FRAME_ID = "/world";
    const char * POINTS_NAMESPACE = "GlMapPoints";
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr msg (new pcl::PointCloud<pcl::PointXYZRGB>);
    msg->header.frame_id = MAP_FRAME_ID;
    for (size_t i = 0; i < mesh.V.rows(); i++) {
        pcl::PointXYZRGB point;
        point.x=mesh.V(i,0);
        point.y=mesh.V(i,1);
        point.z=mesh.V(i,2);
        point.r=mesh.C(i,0)*255;
        point.g=mesh.C(i,1)*255;
        point.b=mesh.C(i,2)*255;
        msg->points.push_back(point);
    }
    msg->height=1;
    msg->width = mesh.V.rows();



    pcl_conversions::toPCL(ros::Time::now(), msg->header.stamp);
    m_cloud_pub.publish (msg);
}


void DataLoaderRos::publish_map_finished(const Mesh& mesh){

    ros::NodeHandle n("~");
    m_cloud_finished_pub = n.advertise<pcl::PointCloud<pcl::PointXYZRGB> > ( "semi_dense_map_finished", 10 );


    if ( m_cloud_pub.getNumSubscribers() == 0 ) return;
    const char * MAP_FRAME_ID = "/world";
    const char * POINTS_NAMESPACE = "GlMapPoints";
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr msg (new pcl::PointCloud<pcl::PointXYZRGB>);
    msg->header.frame_id = MAP_FRAME_ID;
    for (size_t i = 0; i < mesh.V.rows(); i++) {
        pcl::PointXYZRGB point;
        point.x=mesh.V(i,0);
        point.y=mesh.V(i,1);
        point.z=mesh.V(i,2);
        point.r=mesh.C(i,0)*255;
        point.g=mesh.C(i,1)*255;
        point.b=mesh.C(i,2)*255;
        msg->points.push_back(point);
    }
    msg->height=1;
    msg->width = mesh.V.rows();



    pcl_conversions::toPCL(ros::Time::now(), msg->header.stamp);
    m_cloud_finished_pub.publish (msg);

}
