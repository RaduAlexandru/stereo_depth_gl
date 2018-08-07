#include "stereo_depth_gl/DataLoaderPNG.h"

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






DataLoaderPNG::DataLoaderPNG(){

    // init_params();
    // init_data_reading();
    // read_pose_file();
    // create_transformation_matrices();

    init_params_configuru();

}

DataLoaderPNG::~DataLoaderPNG(){
    for (size_t i = 0; i < m_nr_cams; i++) {
        m_loader_threads[i].join();
    }
}

void DataLoaderPNG::init_params(){
    ros::NodeHandle private_nh("~");

    //params
    // m_do_live_segmentation=getParamElseThrow<bool>(private_nh, "do_live_segmentation");


    //m_tf_worldGL_worldROS trasofmr params
    m_tf_worldGL_worldROS_angle=getParamElseThrow<float>(private_nh, "tf_worldGL_worldROS_angle");
    m_tf_worldGL_worldROS_axis=getParamElseThrow<std::string>(private_nh, "tf_worldGL_worldROS_axis");

    //input for the images
    m_rgb_subsample_factor=getParamElseThrow<float>(private_nh, "rgb_subsample_factor");
    m_pose_file=getParamElseThrow<std::string>(private_nh, "pose_file");
    m_nr_cams = getParamElseThrow<int>(private_nh, "nr_cams");
    m_idx_img_to_read_per_cam.resize(m_nr_cams,0);
    m_rgb_filenames_per_cam.resize(m_nr_cams);
    // m_frames_buffer_per_cam.resize(m_nr_cams, moodycamel::ReaderWriterQueue<Frame>(BUFFER_SIZE));
    for (size_t i = 0; i < m_nr_cams; i++) {
        m_rgb_imgs_path_per_cam.push_back( getParamElseThrow<std::string>(private_nh, "rgb_path_cam_"+std::to_string(i)  ) );
        m_labels_imgs_path_per_cam.push_back( getParamElseThrow<std::string>(private_nh, "labels_path_cam_"+std::to_string(i)  ) );

        // //get K
        // std::string intrinsics_string=getParamElseThrow<std::string>(private_nh, "intrinsics_cam_"+std::to_string(i)  );
        // std::vector<std::string> intrinsics_split=split(intrinsics_string," ");
        // Eigen::Matrix3f K;
        // K.setIdentity();
        // K(0,0)=std::stof(intrinsics_split[0]); //fx
        // K(1,1)=std::stof(intrinsics_split[1]); //fy
        // K(0,2)=std::stof(intrinsics_split[2]); // cx
        // K(1,2)=std::stof(intrinsics_split[3]); //cy
        // K*=1.0/m_rgb_subsample_factor;
        // m_intrinsics_per_cam.push_back(K);

        m_frames_buffer_per_cam.push_back(moodycamel::ReaderWriterQueue<Frame>(BUFFER_SIZE));
    }

    //params
    m_imgs_to_skip=getParamElseDefault<int>(private_nh, "imgs_to_skip", 0);
    m_nr_images_to_read=getParamElseThrow<int>(private_nh, "nr_images_to_read");
    std::cout << "m_img_to_skip " << m_imgs_to_skip << '\n';



}

void DataLoaderPNG::init_params_configuru(){
    std::cout << "READINGCONFIGURU:............................................." << '\n';
}

void DataLoaderPNG::init_data_reading(){
    std::cout << "init data reading" << '\n';
    //look into m_imgs_path insithe there should be 2 folders: rgb, labels, possibly also the labels_colored and depth
    for (size_t i = 0; i < m_nr_cams; i++) {
        if(!fs::is_directory(m_rgb_imgs_path_per_cam[i])) {
            LOG(FATAL) << "No directory " << m_rgb_imgs_path_per_cam[i];
        }
        if(!fs::is_directory(m_labels_imgs_path_per_cam[i])) {
            LOG(FATAL) << "No directory " << m_labels_imgs_path_per_cam[i];
        }

        //see how many images we have and read the files paths into a vector
        std::vector<fs::path> rgb_filenames_all;
        for (fs::directory_iterator itr(m_rgb_imgs_path_per_cam[i]); itr!=fs::directory_iterator(); ++itr){
            rgb_filenames_all.push_back(itr->path());
        }


        //TODO sort by name so that we process the frames in the correct order


        std::sort(rgb_filenames_all.begin(), rgb_filenames_all.end(), file_timestamp_comparator());

        //read a maximum nr of images HAVE TO DO IT HERE BECAUSE WE HAVE TO SORT THEM FIRST
        for (size_t img_idx = 0; img_idx < rgb_filenames_all.size(); img_idx++) {
            if(img_idx>=m_imgs_to_skip && (m_rgb_filenames_per_cam[i].size()<m_nr_images_to_read || m_nr_images_to_read<0 ) ){
                m_rgb_filenames_per_cam[i].push_back(rgb_filenames_all[img_idx]);
            }
        }
        std::cout << "Nr rgb images on cam " << i << ": " << rgb_filenames_all.size() << std::endl;
        // std::cout << "Nr rgb images on cam resized  " << i << ": " << m_rgb_filenames_per_cam[i].size() << std::endl;


        // std::cout << "stems are "  << '\n';
        // for (size_t d = 0; d < m_rgb_filenames_per_cam[i].size(); d++) {
        //     std::cout << "path is is " << m_rgb_filenames_per_cam[i][d] << '\n';
        //     std::cout << "stem is " << m_rgb_filenames_per_cam[i][d].stem() << '\n';
        // }

        // std::cout << "stem is " << m_rgb_filenames_per_cam[i][0].stem().string() << '\n';
    }

    m_last_frame_per_cam.resize(m_nr_cams);
    m_get_last_published_frame_for_cam.resize(m_nr_cams,false);


    // //get the same nr of images from both cams (drop the ones that have more) so we get images in pairs more easily
    // // int cam_with_less_images=-1;
    // unsigned long long min_nr_images=99999999999999;
    // for (size_t i = 0; i < m_nr_cams; i++) {
    //     if(m_rgb_filenames_per_cam[i].size()<min_nr_images){
    //         min_nr_images=m_rgb_filenames_per_cam[i].size();
    //         // cam_with_less_images=m_nr_cams;
    //     }
    // }
    // for (size_t i = 0; i < m_nr_cams; i++) {
    //     m_rgb_filenames_per_cam[i].resize(min_nr_images);
    // }


    // m_mask_cam_0=cv::imread("/media/alex/Data/Master/Thesis/data/courtyard_data/janis_data/mask.png",cv::IMREAD_GRAYSCALE);

}

void DataLoaderPNG::start_reading(){

    m_loader_threads.resize(m_nr_cams);
    for (size_t i = 0; i < m_nr_cams; i++) {
        m_loader_threads[i]=std::thread(&DataLoaderPNG::read_data_for_cam, this, i);
    }

}

void DataLoaderPNG::read_data_for_cam(const int cam_id){
    std::cout << "----------READING DATA for cam " << cam_id << '\n';
    loguru::set_thread_name(("loader_thread_"+std::to_string(cam_id)).c_str());

    int nr_frames_read_for_cam=0;
    while (ros::ok()) {

        //we finished reading so we wait here for a reset
        if(m_idx_img_to_read_per_cam[cam_id]>=m_rgb_filenames_per_cam[cam_id].size()){
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            // std::cout << "waiting" << '\n';
            continue;
        }

        // if( (int)m_rgb_filenames_per_cam[cam_id].size()- (int)m_idx_img_to_read_per_cam[cam_id] < 60){
        //     m_idx_img_to_read_per_cam[cam_id]++;
        //     continue;
        // }

        // std::cout << "size approx is " << m_queue.size_approx() << '\n';
        // std::cout << "m_idx_img_to_read is " << m_idx_img_to_read << '\n';
        if(m_frames_buffer_per_cam[cam_id].size_approx()<BUFFER_SIZE-1){ //there is enough space
            //read the frame and everything else and push it to the queue


            Frame frame;
            frame.cam_id=cam_id;
            frame.frame_idx=nr_frames_read_for_cam;

            fs::path rgb_filename=m_rgb_filenames_per_cam[cam_id][ m_idx_img_to_read_per_cam[cam_id] ];
            // std::cout << "rgb_filename is " << rgb_filename << '\n';
            m_idx_img_to_read_per_cam[cam_id]++;
            uint64_t timestamp=std::stoull(rgb_filename.stem().string());
            frame.timestamp=timestamp; //store the unrounded one because when we check for the labels we check for the same filename



            //TODO---------------parametrize this depending on the dataset because the pose may be to a different frame and so on
            //POSE---
            Eigen::Affine3f sensor_pose;  //maps from baselink to world ros
            if (!get_pose_at_timestamp(sensor_pose, timestamp)){
                // LOG(WARNING) << "Not found any pose at timestamp " << timestamp << " Discarding";
                continue;
            }

            // std::cout << "we got pose for image " << m_idx_img_to_read_per_cam[cam_id] << '\n';

            // if(m_poses_from_courtyard_data){
            //     sensor_pose=m_tf_worldGL_worldROS*sensor_pose; // from baselink to world ros and from world ros to worlf_gl (now sensor pose maps from baselink to world_gl)
            //     //TODO
            //     Eigen::Affine3f tf_cam_baselink;
            //     tf_cam_baselink.setIdentity();
            //     if(cam_id==0){
            //         //rosrun tf tf_echo /cam_left /base_link
            //         Eigen::Vector3f cam_baselink_t(-0.050, -0.100, -0.100);
            //         // Eigen::Quaterniond cam_baselink_quat(0.707, 0.000, 0.000, 0.707);
            //         //TODO the quaternion didn't quite work here
            //         Eigen::AngleAxisf rollAngle(1.571, Eigen::Vector3f::UnitX());
            //         Eigen::AngleAxisf yawAngle(0.0, Eigen::Vector3f::UnitY());
            //         Eigen::AngleAxisf pitchAngle(0.0, Eigen::Vector3f::UnitZ());
            //         Eigen::Quaternion<float> cam_baselink_quat = pitchAngle * yawAngle * rollAngle;
            //         tf_cam_baselink.matrix().block<3,3>(0,0)=cam_baselink_quat.toRotationMatrix();
            //         tf_cam_baselink.matrix().block<3,1>(0,3)=cam_baselink_t;
            //     }else if(cam_id==1){
            //         //rosrun tf tf_echo /cam_right /base_link
            //         Eigen::Vector3f cam_baselink_t(0.050, -0.100, -0.100);
            //         // Eigen::Quaterniond cam_baselink_quat(0.707, 0.000, 0.000, 0.707);
            //         //TODO the quaternion didn't quite work here
            //         Eigen::AngleAxisf rollAngle(-1.571, Eigen::Vector3f::UnitX());
            //         Eigen::AngleAxisf yawAngle(0.0, Eigen::Vector3f::UnitY());
            //         Eigen::AngleAxisf pitchAngle(-3.142, Eigen::Vector3f::UnitZ());
            //         Eigen::Quaternion<float> cam_baselink_quat = pitchAngle * yawAngle * rollAngle;
            //         tf_cam_baselink.matrix().block<3,3>(0,0)=cam_baselink_quat.toRotationMatrix();
            //         tf_cam_baselink.matrix().block<3,1>(0,3)=cam_baselink_t;
            //     }
            //
            //     sensor_pose= tf_cam_baselink *sensor_pose.inverse(); //from world_gl to baselink and from baselink to cam
            // }else if(m_poses_from_semantic_fusion) {
            //     sensor_pose=m_tf_worldGL_worldROS*sensor_pose; // from baselink to world ros and from world ros to worlf_gl (now sensor pose maps from baselink to world_gl)
            //     sensor_pose=sensor_pose.inverse();
            // }else if(m_poses_from_synthia){
            //     sensor_pose=sensor_pose.inverse(); //nothing else to do
            //     //TODO for some reason the camera is pointing backwards so we rotate it
            //     Eigen::Affine3f rot;
            //     Eigen::AngleAxisf rollAngle(3.1415, Eigen::Vector3f::UnitX());
            //     Eigen::AngleAxisf yawAngle(0.0, Eigen::Vector3f::UnitY());
            //     Eigen::AngleAxisf pitchAngle(0.0, Eigen::Vector3f::UnitZ());
            //     Eigen::Quaternion<float> rot_quat = pitchAngle * yawAngle * rollAngle;
            //     rot.matrix().block<3,3>(0,0)=rot_quat.toRotationMatrix();
            //     rot.matrix().block<3,1>(0,3)  << 0.0, 0.0, 0.0;
            //     sensor_pose= rot *sensor_pose; //from world_gl to baselink and from baselink to cam
            // }


            //Get rgb
            // TIME_START("read_rgb_img");
            frame.rgb=cv::imread(rgb_filename.string());
            cv::resize(frame.rgb, frame.rgb, cv::Size(), 1.0/m_rgb_subsample_factor, 1.0/m_rgb_subsample_factor);








            // TIME_END("read_rgb_img");
            // if(m_poses_from_synthia){
            //     cv::Mat flipped;
            //     cv::flip(frame.rgb, flipped, 0);
            //     frame.rgb=flipped;
            // }







            //first few frames will not be good because the spline is not initialized completely
            // if(frame.frame_idx<=50){
            //     nr_frames_read_for_cam++;
            //     continue;
            // }


            frame.K=m_intrinsics_per_cam[cam_id].cast<float>();
            frame.tf_cam_world=sensor_pose;


            //correct the pose with the spline
            // VLOG(1) << "before pose is : \n" << frame.tf_cam_world.matrix() << std::endl;
            // if(m_do_spline_correction){
            //     VLOG(2) << "Correcting with spline";
            //     double offset_global_and_local=m_spline_global_offset_ms+(-100)*frame.deviation_ms;
            //     frame.tf_cam_world_not_corrected=frame.tf_cam_world;
            //     update_pose_with_spline(frame,offset_global_and_local, m_extrinsic_vector_cam0_baselink.cast<double>(),m_extrinsic_vector_cam1_baselink.cast<double>() );
            // }
            // VLOG(1) << "after pose is : \n" << frame.tf_cam_world.matrix() << std::endl;
            //if the pose was not valid then we will get a nan. TODO make it a bit nicer, maybe return a bool for success or check why id fails in the first place
            if(!std::isfinite(frame.tf_cam_world.matrix()(0,0))){
                LOG(WARNING) << "Spline pose not valid" ;
                continue;
            }

            m_frames_buffer_per_cam[cam_id].enqueue(frame);
            nr_frames_read_for_cam++;





        }
    }
    VLOG(1) << "Finished reading all the images";
}

bool DataLoaderPNG::is_finished(){
    //check if this loader has loaded everything for every camera
    for (size_t cam_id = 0; cam_id < m_nr_cams; cam_id++) {
        if(m_idx_img_to_read_per_cam[cam_id]<m_rgb_filenames_per_cam[cam_id].size()){
            // VLOG(1) << "there is still more files to read for cam " << cam_id << " " << m_idx_img_to_read_per_cam[cam_id] << " out of " <<  m_rgb_filenames_per_cam[cam_id].size() ;
            return false; //there is still more files to read
        }
    }

    //check that there is nothing in the ring buffers
    for (size_t cam_id = 0; cam_id < m_nr_cams; cam_id++) {
        if(m_frames_buffer_per_cam[cam_id].peek()!=nullptr){
            // VLOG(1) << "There is still smething in the buffer";
            return false; //there is still something in the buffer
        }
    }

    return true; //there is nothing more to read and nothing more in the buffer so we are finished

}


bool DataLoaderPNG::is_finished_reading(){
    //check if this loader has loaded everything for every camera
    for (size_t cam_id = 0; cam_id < m_nr_cams; cam_id++) {
        if(m_idx_img_to_read_per_cam[cam_id]<m_rgb_filenames_per_cam[cam_id].size()){
            return false; //there is still more files to read
        }
    }

    return true; //there is nothing more to read and nothing more in the buffer so we are finished

}

bool DataLoaderPNG::has_data_for_cam(const int cam_id){
    // return !m_queue.empty();
    if(m_frames_buffer_per_cam[cam_id].peek()==nullptr){
        return false;
    }else{
        return true;
    }
}

bool DataLoaderPNG::has_data_for_all_cams(){
    for (size_t i = 0; i < m_nr_cams; i++) {
        if(!has_data_for_cam(i)){
            return false;
        }
    }
    return true;
}

Frame DataLoaderPNG::get_next_frame_for_cam(const int cam_id){
    TIME_SCOPE("get_next_frame");

    if(m_get_last_published_frame_for_cam[cam_id] && m_last_frame_per_cam[cam_id].rgb.data){
        m_get_last_published_frame_for_cam[cam_id]=false;

        //this frame has to be recorrected with the spline
        Frame frame=m_last_frame_per_cam[cam_id];

        // VLOG(1) << "after pose is : \n" << frame.tf_cam_world.matrix() << std::endl;
        //if the pose was not valid then we will get a nan. TODO make it a bit nicer, maybe return a bool for success or check why id fails in the first place
        if(!std::isfinite(frame.tf_cam_world.matrix()(0,0))){
            LOG(WARNING) << "Spline pose not valid" ;
        }
        return frame;
    }

    Frame frame ;
    m_frames_buffer_per_cam[cam_id].try_dequeue(frame);

    //store also the last frame in case we need to republish it
    m_last_frame_per_cam[cam_id]=frame;

    return frame;

}

void DataLoaderPNG::reset(){
    for (size_t i = 0; i < m_nr_cams; i++) {
        m_idx_img_to_read_per_cam[i]=0;
        //deque everything (we can do it safely from here because while this is running, the core is not reading since the Core and GUI share thread)
        m_frames_buffer_per_cam[i]=moodycamel::ReaderWriterQueue<Frame>(BUFFER_SIZE);
    }

}

void DataLoaderPNG::clear_buffers(){
    for (size_t i = 0; i < m_nr_cams; i++) {
        //deque everything (we can do it safely from here because while this is running, the core is not reading since the Core and GUI share thread)
        m_frames_buffer_per_cam[i]=moodycamel::ReaderWriterQueue<Frame>(BUFFER_SIZE);
    }
}




void DataLoaderPNG::read_pose_file(){
    std::ifstream infile( m_pose_file );
    if(!infile.is_open()){
        LOG(FATAL) << "Could not open pose file " << m_pose_file;
    }
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
        // if(m_do_timestamp_rounding_when_reading_file){
        //     timestamp=(uint64_t)std::round(timestamp/100000.0); ////TODO this is a dirty hack to reduce the discretization of time because the timestamps don't exactly match
        // }
       // VLOG(2) << "recorded tmestamp is " << timestamp;
//        VLOG(2) << "recorded scan_nr is " << scan_nr;
        m_worldROS_baselink_map[timestamp]=pose;
        m_worldROS_baselink_vec.push_back ( std::pair<uint64_t, Eigen::Affine3f>(timestamp,pose) );
    }

}

// void DataLoaderPNG::read_pose_file_semantic_fusion(){
//     std::ifstream infile( m_pose_file );
// if(!infile.is_open()){
//     LOG(FATAL) << "Could not open pose file " << m_pose_file;
// }
//     uint64_t scan_nr;
//     uint64_t timestamp;
//     Eigen::Vector3f position;
//     Eigen::Quaterniond quat;
//
//
//     std::string line;
//     while (std::getline(infile, line)) {
//         std::istringstream iss(line);
//         iss >> scan_nr >> timestamp
//             >> position(0) >> position(1) >> position(2)
//             >> quat.w() >> quat.x() >> quat.y() >> quat.z();
// //        std::cout << "input is \n" << scan_nr << " " << timestamp << " " << position << " " << quat.matrix()  << "\n";
//         Eigen::Affine3f pose;
//         pose.matrix().block<3,3>(0,0)=quat.toRotationMatrix();
//         pose.matrix().block<3,1>(0,3)=position;
// //        VLOG(2) << "recorded tmestamp is " << timestamp;
// //        VLOG(2) << "recorded scan_nr is " << scan_nr;
//         m_worldROS_baselink_map[timestamp]=pose;
//     }
//
// }

bool DataLoaderPNG::get_pose_at_timestamp(Eigen::Affine3f& pose, const uint64_t timestamp){

    auto got = m_worldROS_baselink_map.find (timestamp);

//     if ( got == m_worldROS_baselink_map.end() ){
//         LOG(WARNING) << "get_pose: pose query for the scan does not exist at timestamp" << timestamp;
//         return false;
//     }else{
//         pose = got->second;
// //        VLOG(2) << "returning pose at scan_nr  \n" << pose.matrix();
//         return true;
//     }


    //return the closest one
    uint64_t closest_idx=-1;
    double smallest_timestamp_diff=std::numeric_limits<double>::max();
    double smallest_timestamp_diff_no_abs=std::numeric_limits<double>::max();
    for (size_t i = 0; i < m_worldROS_baselink_vec.size(); i++) {
        uint64_t recorded_timestamp=m_worldROS_baselink_vec[i].first;
        Eigen::Affine3f pose=m_worldROS_baselink_vec[i].second;
        // std::cout << "comparing recorded_timestamp to timestmp \n" << recorded_timestamp << "\n" << timestamp << '\n';
        if (  abs((double)timestamp- (double)recorded_timestamp) < smallest_timestamp_diff){
            closest_idx=i;
            smallest_timestamp_diff=abs(timestamp-recorded_timestamp);
            smallest_timestamp_diff_no_abs=(double)timestamp - (double)recorded_timestamp;
        }
    }
    if ( smallest_timestamp_diff > 1e7 )
    {
        LOG(WARNING) << "time difference for pose is way too large! " << (smallest_timestamp_diff/1e6) << "s." << '\n';
        return false;
    }
    // std::cout << "smallest_timestamp_diff is " << smallest_timestamp_diff << '\n';
    // std::cout << "smallest_timestamp_diff_no_abs is " << smallest_timestamp_diff_no_abs << '\n';
    // std::cout << "deviation_ms is " << deviation_ms << '\n';
    pose=m_worldROS_baselink_vec[closest_idx].second;
    return true;


}

void DataLoaderPNG::create_transformation_matrices(){


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
    // rot = Eigen::AngleAxisf(-0.5*M_PI, Eigen::Vector3f::UnitZ())  //this rotation is done second and rotates around the Z axis of the velodyne frame but after it was rotated by the first rotation.
    //   * Eigen::AngleAxisf(0.5*M_PI, Eigen::Vector3f::UnitY());   //this rotation is done first. Performed on the Y axis of the velodyne frame (after this the y is pointing left, x is up and z is towards the camera)
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

void DataLoaderPNG::republish_last_frame_from_cam(const int cam_id){
    m_get_last_published_frame_for_cam[cam_id]=true;
    // std::cout << "republish check" << '\n';
    // if(m_last_frame_per_cam[cam_id].rgb.data){
    //     std::cout << "republish " << '\n';
    //     m_frames_buffer_per_cam[cam_id].enqueue(m_last_frame_per_cam[cam_id]);
    // }

    //set it so that the next

}

void DataLoaderPNG::republish_last_frame_all_cams(){
    for (size_t i = 0; i < m_nr_cams; i++) {
        m_get_last_published_frame_for_cam[i]=true;
    }
}
