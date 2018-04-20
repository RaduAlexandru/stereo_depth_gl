#include "stereo_depth_cl/DataLoader.h"

//c++
#include <iostream>
#include <fstream>

//loguru
#define LOGURU_NO_DATE_TIME 1
#define LOGURU_NO_UPTIME 1
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp>

//My stuff
#include "stereo_depth_cl/Profiler.h"
#include "stereo_depth_cl/RosBagPlayer.h"
#include "stereo_depth_cl/MiscUtils.h"

//cv
#include <cv_bridge/cv_bridge.h>

//ros
#include "stereo_depth_cl/RosTools.h"

//boost
#include <boost/bind.hpp>




DataLoader::DataLoader():
        m_max_nr_threads(4),
        m_loader_is_modified(false),
        m_finished_frame_idx(-1),
        m_working_frame_idx(0){

    read_pose_file();
    create_transformation_matrices();
    init_params();

}

void DataLoader::init_params(){
    ros::NodeHandle private_nh("~");
    //ros
    m_frame_world = getParamElseDefault<std::string>(private_nh, "world_frame", "world");
    m_nr_cams = getParamElseThrow<int>(private_nh, "nr_cams");

    for (size_t i = 0; i < m_nr_cams; i++) {
        //read the frame and the image topics for the cameras
        std::string frame_name= "cam_frame_" + std::to_string(i);
        std::string topic_name= "/cam_img_" + std::to_string(i);
        m_cam_frames.push_back(getParamElseDefault<std::string>(private_nh, frame_name, "camera"));
        // m_cam_img_topics.push_back(getParamElseDefault<std::string>(private_nh, topic_name, "camera"));
    }

    //init the buffers for loading frames
    m_cam_buffers_modified.resize(m_nr_cams,false);
    m_frames_buffer_per_cam.resize(m_nr_cams);
    for (size_t i = 0; i < m_nr_cams; i++) {
        std::cout << "frame buffer " << i << " hass size " << m_frames_buffer_per_cam[i].max_size() << '\n';
    }

    //init ros subscribers, one for each cam
    m_img_subs.resize(m_nr_cams);

    //init the possible masks for the cams
    m_masks.resize(m_nr_cams);

    //init the access mutexes for each cams buffers so that we don't read with get frame and pushback at the same time
    std::vector<std::mutex> new_mutex_vector(m_nr_cams); //the resizing needs to be done in two steps as mutexes are not movable
    m_mutex_access_cam.swap(new_mutex_vector);
    // m_mutex_access_cam.resize(m_nr_cams);

}


void DataLoader::load_data(){
    loguru::set_thread_name("ROS thread");
    ros::NodeHandle private_nh("~");

    for (size_t i = 0; i < m_nr_cams; i++) {
        // img_sub = private_nh.subscribe("/cam_img_0", 1000, &DataLoader::callback, this);
        std::string topic_name= "/cam_img_" + std::to_string(i);
        m_img_subs[i] = private_nh.subscribe<sensor_msgs::CompressedImage> (topic_name, 3, boost::bind(&DataLoader::callback, this, _1, i));
    }

    // img_sub = private_nh.subscribe("/cam_img_0", 1000, &DataLoader::callback, this);
    // ros::spin();

    //multithreaded spinning, each callback (from different cameras) will run on different threads in paralel
    ros::MultiThreadedSpinner spinner(std::max(m_nr_cams, m_max_nr_threads)); // Use 4 threads
    spinner.spin();

}

Frame DataLoader::get_frame_for_cam(const int cam_id) {
    std::lock_guard<std::mutex> lock(m_mutex_access_cam[cam_id]);
    m_loader_is_modified = false;
    m_cam_buffers_modified[cam_id] = false;
    return m_frames_buffer_per_cam[cam_id].back(); //retrun last frame for that cam
}

int DataLoader::get_nr_cams(){
    return m_nr_cams;
}





void DataLoader::callback(const sensor_msgs::CompressedImageConstPtr &img_msg, const int cam_id){
    loguru::set_thread_name(("ros_thread_"+std::to_string(cam_id)).c_str());
    TIME_SCOPE("callback");

    if(m_player->m_player_should_do_one_step){
        m_player->m_player_should_do_one_step=false;
        m_player->pause();
    }

    Frame frame;
    frame.cam_id=cam_id;

    LOG_S(INFO) << "calback------------------------------ from cam " << cam_id;


    frame.timestamp=img_msg->header.stamp.toNSec();

    //Get image to opencv mat
    // TIME_START("get_img_to_cv_mat"); //60ms
    cv::Mat img_cv;
    try {
        img_cv = cv::imdecode(cv::Mat(img_msg->data), 1);//convert compressed image data to cv::Mat
    } catch (cv_bridge::Exception &e) {
                LOG(ERROR) << "Core::callback: Could not convert to image!";
    }
    frame.rgb=img_cv;
    // TIME_END("get_img_to_cv_mat");


    //store also the mask in case we have one otherwie just store a white image (we accept every pixels)
    if(!m_masks[cam_id].data){
        m_masks[cam_id]=cv::Mat(frame.rgb.rows, frame.rgb.cols, CV_8U);
        m_masks[cam_id] = cv::Scalar(255);
    }
    frame.mask=m_masks[cam_id];


    //get also the segmentation
    std::string cam_left_segmentation_path="/home/alex/tmp/cam_left_segmentation/";
    // VLOG(2) << "looking for segmentation with name: " << frame.timestamp;
    std::string classes_path=cam_left_segmentation_path + std::to_string(frame.timestamp) + "_classes.png";
    std::string probs_path=cam_left_segmentation_path + std::to_string(frame.timestamp) + "_probs.png";
    frame.classes_original_size=cv::imread(classes_path, cv::IMREAD_GRAYSCALE );
    frame.probs_original_size=cv::imread(probs_path, cv::IMREAD_GRAYSCALE );
    if(!frame.classes_original_size.data || !frame.probs_original_size.data){
                LOG(FATAL) << "Could not read segmentation image ";
    }


    frame.classes_original_size=frame.classes_original_size - cv::Scalar(1); //because lua starts counting the classes from 1 an we want to start from 0
    frame.probs_original_size.convertTo(frame.probs_original_size, CV_32F);  //need to be converted to float first in order for the scaling to 0-1 to work
    frame.probs_original_size/=255.0;


     //resize them to the size of the rgb image to make it easier to sample from them
     cv::resize(frame.classes_original_size, frame.classes, frame.rgb.size(), 0, 0, cv::INTER_NEAREST);
     cv::resize(frame.probs_original_size, frame.probs, frame.rgb.size(), 0, 0, cv::INTER_NEAREST);

     //put an alpha channel to the rgb so it has 4 channels (4 bytes aligntes and more GPU friendly)
     //Seems to be actually slower
     // frame.rgb=create_alpha_mat(frame.rgb);

     // //save also a small rgb image for visualization
     // int max_size=300;
     // double scale = static_cast<double>(max_size) / std::max(frame.rgb.rows, frame.rgb.cols);
     // cv::Size size(frame.rgb.cols*scale, frame.rgb.rows*scale);
     // cv::resize(frame.rgb, frame.rgb_small, size );



    //POSE---
    Eigen::Affine3f sensor_pose;  //maps from baselink to world ros
    uint64_t rounded_timestamp= (uint64_t)std::round(img_msg->header.stamp.toNSec()/100000000.0);
    if (!get_pose_at_timestamp(sensor_pose, rounded_timestamp)){
        LOG(WARNING) << "Not found any pose at timestamp " << rounded_timestamp << " Discarding";
        return;
    }
    sensor_pose=m_tf_worldGL_worldROS*sensor_pose; // from baselink to world ros and from world ros to worlf_gl (now sensor pose maps from baselink to world_gl)
    //TODO read K from the cam info
    Eigen::Matrix3f K;
    K << 2404.76891592, 0, 1245.04813256,
            0, 2400.9170077, 1003.61646891,
            0, 0, 1;
    Eigen::Affine3f tf_cam_baselink;
    // VLOG(2) << "get tf";
    get_tf(tf_cam_baselink, "base_link", m_cam_frames[cam_id], ros::Time(0) ); // ros::Time(0) means we get the latest known transform
    Eigen::Affine3f tf_cam_world= tf_cam_baselink *sensor_pose.inverse(); //from world_gl to baselink and from baselink to cam

    frame.K=K;
    frame.tf_cam_world=tf_cam_world;

    m_mutex_access_cam[cam_id].lock(); // lock the mutex so that we don't push and read the frames at the same time
    m_frames_buffer_per_cam[cam_id].push(frame);
    m_cam_buffers_modified[cam_id]=true;
    m_loader_is_modified = true;
    m_mutex_access_cam[cam_id].unlock();


    LOG_S(INFO) << "finished from cam " << cam_id;

}

void DataLoader::set_mask_for_cam(const std::string mask_filename, const int cam_id){
    cv::Mat mask=cv::imread(mask_filename, cv::IMREAD_GRAYSCALE);
    if(!mask.data ){
        LOG(ERROR) << "Could not open or find the image";
    }
    m_masks[cam_id]=mask;
}

bool DataLoader::get_tf(Eigen::Affine3f& tf, const std::string& origin_frame, const std::string& dest_frame, const ros::Time query_time ) {
    tf::StampedTransform tf_transform;
    try {
        m_tf_listener.lookupTransform( dest_frame, origin_frame, query_time, tf_transform );
    } catch ( tf::TransformException exc ) {
                LOG(WARNING) << "exc.what()";
        return false;
    }
    Eigen::Affine3d tf_double;
    transformTFToEigen( tf_transform, tf_double );
    tf=tf_double.cast<float>();
    return true;
}

void DataLoader::read_pose_file(){
    std::ifstream infile( "/media/alex/Data/Master/SHK/c_ws/src/laser_mesher/data/graph_viewer_scan_poses_00.txt" );
    uint64_t scan_nr;
    uint64_t timestamp;
    Eigen::Vector3f position;
    Eigen::Quaternionf quat;


    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        iss >> scan_nr >> timestamp
            >> position(0) >> position(1) >> position(2)
            >> quat.w() >> quat.x() >> quat.y() >> quat.z();
//        std::cout << "input is \n" << scan_nr << " " << timestamp << " " << position << " " << quat.matrix()  << "\n";
        Eigen::Affine3f pose;
        pose.matrix().block<3,3>(0,0)=quat.toRotationMatrix();
        pose.matrix().block<3,1>(0,3)=position;
        timestamp=(uint64_t)std::round(timestamp/100000.0); ////TODO this is a dirty hack to reduce the discretization of time because the timestamps don't exactly match
//        VLOG(2) << "recorded tmestamp is " << timestamp;
//        VLOG(2) << "recorded scan_nr is " << scan_nr;
        m_worldROS_baselink_map[timestamp]=pose;
    }

}

bool DataLoader::get_pose_at_timestamp(Eigen::Affine3f& pose, uint64_t timestamp){

    auto got = m_worldROS_baselink_map.find (timestamp);

    if ( got == m_worldROS_baselink_map.end() ){
                LOG(WARNING) << "get_pose: pose query for the scan does not exist at timestamp" << timestamp;
        return false;
    }else{
        pose = got->second;
//        VLOG(2) << "returning pose at scan_nr  \n" << pose.matrix();
        return true;
    }

}

void DataLoader::create_transformation_matrices(){


    /*
     *  All frames are right handed systems
     *  The gap of the laser is always on the far side of the camera, somewhere on the negative Z axis in the alg frame
     *
     *                                                                                  Y
     *                                                                                  |
     *                                                                                  |
     *                                                                                  |
     *                                                                                  |
     *                                                                                  |
     *  Y-------                                        --->                            ---------- X
     *         /|                                                                      /
     *       /  |                                                                    /
     *     /    |                                                                  /
     *    X     |                                                                 Z
     *          |
     *          Z
     * Velodyne frame represented in the odom frame                             Algorithm frame (and also how the GL frame looks like)
     * This is what we see in rviz                                              Frame which the algorithms in the Mesher use to create the mesh
     * X looking towards camera
     * Z downwards
     * Y to the left
     *
     * The transformation between these two is what the function called fix_cloud_orientation was doing:
     *  Reminder:
     *  cloud->points[idx].x = -y;
     *  cloud->points[idx].y = -z;
     *  cloud->points[idx].z = x;
     *
     * The matrix correspinding to the transformation will be only a rotation matrix
     *  since we only rotate around the origin and don't do any translation
     *
     * Positives angles of rotiation here:
     *   https://stackoverflow.com/questions/31191752/right-handed-euler-angles-xyz-to-left-handed-euler-angles-xyz
     *
     * */

    m_tf_alg_vel.setIdentity();
    Eigen::Matrix3f alg_vel_rot;
    // rot = Eigen::AngleAxisd(-0.5*M_PI, Eigen::Vector3d::UnitZ())  //this rotation is done second and rotates around the Z axis of the velodyne frame but after it was rotated by the first rotation.
    //   * Eigen::AngleAxisd(0.5*M_PI, Eigen::Vector3d::UnitY());   //this rotation is done first. Performed on the Y axis of the velodyne frame (after this the y is pointing left, x is up and z is towards the camera)
    // // m_tf_alg_vel.matrix().block<3,3>(0,0)=rot.transpose();

    alg_vel_rot = Eigen::AngleAxisf(-0.5*M_PI+M_PI, Eigen::Vector3f::UnitY())  //this rotation is done second and rotates around the Y axis of alg frame
                  * Eigen::AngleAxisf(0.5*M_PI, Eigen::Vector3f::UnitX());   //this rotation is done first. Performed on the X axis of alg frame (after this the y is pointing towards camera, x is right and z is down)
    m_tf_alg_vel.matrix().block<3,3>(0,0)=alg_vel_rot;



    m_tf_baselink_vel.setIdentity();
    Eigen::Vector3f baselink_vel_t(-0.000, -0.000, -0.177);
    // Eigen::Quaterniond baselink_vel_quat(-0.692, 0.722, -0.000, -0.000);

    //TODO the quaternion didn't quite work here
    Eigen::AngleAxisf rollAngle(-3.142, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf yawAngle(0.0, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf pitchAngle(-1.614, Eigen::Vector3f::UnitZ());
    Eigen::Quaternion<float> baselink_vel_quat = pitchAngle * yawAngle * rollAngle;

    m_tf_baselink_vel.matrix().block<3,3>(0,0)=baselink_vel_quat.toRotationMatrix();
    m_tf_baselink_vel.matrix().block<3,1>(0,3)=baselink_vel_t;



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
    worldGL_worldROS_rot = Eigen::AngleAxisf(-0.5*M_PI, Eigen::Vector3f::UnitX());
    m_tf_worldGL_worldROS.matrix().block<3,3>(0,0)=worldGL_worldROS_rot;




}
