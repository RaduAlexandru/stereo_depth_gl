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

// //configuru
// #define CONFIGURU_WITH_EIGEN 1
// #define CONFIGURU_IMPLICIT_CONVERSIONS 1
// #include <configuru.hpp>
using namespace configuru;





DataLoaderPNG::DataLoaderPNG(){

    init_params();
    init_data_reading();
    create_transformation_matrices();
    if(m_dataset_type==DatasetType::ETH){
        read_pose_file_eth();
    }else if(m_dataset_type==DatasetType::ICL){
        read_pose_file_icl();
    }
    // read_pose_file();


    // init_params_configuru();

}

DataLoaderPNG::~DataLoaderPNG(){
    for (size_t i = 0; i < m_nr_cams; i++) {
        m_loader_threads[i].join();
    }
}

void DataLoaderPNG::init_params(){
    //get the config filename
    ros::NodeHandle private_nh("~");
    std::string config_file= getParamElseThrow<std::string>(private_nh, "config_file");

    //read all the parameters
    Config cfg = configuru::parse_file(std::string(CMAKE_SOURCE_DIR)+"/config/"+config_file, CFG);
    Config loader_config=cfg["loader"];
    m_nr_cams = loader_config["nr_cams"];
    m_imgs_to_skip=loader_config["imgs_to_skip"];
    m_nr_images_to_read=loader_config["nr_images_to_read"];
    for (size_t i = 0; i < m_nr_cams; i++) {
        m_rgb_imgs_path_per_cam.push_back( (std::string)loader_config["rgb_path_cam_"+std::to_string(i)] );
        m_frames_buffer_per_cam.push_back( moodycamel::ReaderWriterQueue<Frame>(BUFFER_SIZE));
    }
    std::string dataset_type_string=(std::string)loader_config["dataset_type"];
    if(dataset_type_string=="eth") m_dataset_type=DatasetType::ETH;
    if(dataset_type_string=="icl") m_dataset_type=DatasetType::ICL;
    else LOG(FATAL) << " Dataset type is not known " << dataset_type_string;
    m_pose_file=(std::string)loader_config["pose_file"];


    Config vis_config=cfg["visualization"];
    m_tf_worldGL_worldROS_angle=vis_config["tf_worldGL_worldROS_angle"];
    m_tf_worldGL_worldROS_axis=(std::string)vis_config["tf_worldGL_worldROS_axis"];

    // //input for the images
    // m_rgb_subsample_factor=getParamElseThrow<float>(private_nh, "rgb_subsample_factor");
    // m_pose_file=getParamElseThrow<std::string>(private_nh, "pose_file");
    // m_nr_cams = getParamElseThrow<int>(private_nh, "nr_cams");
    // m_idx_img_to_read_per_cam.resize(m_nr_cams,0);
    // m_rgb_filenames_per_cam.resize(m_nr_cams);
    // // m_frames_buffer_per_cam.resize(m_nr_cams, moodycamel::ReaderWriterQueue<Frame>(BUFFER_SIZE));
    // for (size_t i = 0; i < m_nr_cams; i++) {
    //     m_rgb_imgs_path_per_cam.push_back( getParamElseThrow<std::string>(private_nh, "rgb_path_cam_"+std::to_string(i)  ) );
    //     m_labels_imgs_path_per_cam.push_back( getParamElseThrow<std::string>(private_nh, "labels_path_cam_"+std::to_string(i)  ) );
    //
    //     // //get K
    //     // std::string intrinsics_string=getParamElseThrow<std::string>(private_nh, "intrinsics_cam_"+std::to_string(i)  );
    //     // std::vector<std::string> intrinsics_split=split(intrinsics_string," ");
    //     // Eigen::Matrix3f K;
    //     // K.setIdentity();
    //     // K(0,0)=std::stof(intrinsics_split[0]); //fx
    //     // K(1,1)=std::stof(intrinsics_split[1]); //fy
    //     // K(0,2)=std::stof(intrinsics_split[2]); // cx
    //     // K(1,2)=std::stof(intrinsics_split[3]); //cy
    //     // K*=1.0/m_rgb_subsample_factor;
    //     // m_intrinsics_per_cam.push_back(K);
    //
    //     m_frames_buffer_per_cam.push_back(moodycamel::ReaderWriterQueue<Frame>(BUFFER_SIZE));
    // }
    //
    // //params
    // m_imgs_to_skip=getParamElseDefault<int>(private_nh, "imgs_to_skip", 0);
    // m_nr_images_to_read=getParamElseThrow<int>(private_nh, "nr_images_to_read");
    // std::cout << "m_img_to_skip " << m_imgs_to_skip << '\n';



}

void DataLoaderPNG::init_params_configuru(){
    std::cout << "READINGCONFIGURU:............................................." << '\n';

    Config cfg = configuru::parse_file("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_gl/config/config.cfg", FORGIVING);
    float alpha = cfg["alpha"];
    std::cout << "alpha is " << alpha  << '\n';

    if (cfg["matrix"].is_array()) {
    	std::cout << "First element: " << cfg["matrix"][0];
    	for (const configuru::Config& element : cfg["matrix"].as_array()) {
    		std::cout << element << std::endl;
    	}
    }

    // std::cout << "reading vector2f" << '\n';
    // Eigen::Vector2f vec= as<Eigen::Vector2f>(cfg["vec"]);
    // std::cout << "vec is " << vec << '\n';
    //
    std::cout << "reading vector2f as stdvec" << '\n';
    std::vector<float> vec_std= cfg["vec"];
    std::cout << "vec_std has size is " << vec_std.size() << '\n';
    //
    std::cout << "reading vector2f again" << '\n';
    Eigen::Matrix< float , 2 , 1> vec_again= cfg["vec"];
    std::cout << "vec_again is " << vec_again << '\n';
    //
    std::cout << "reading mat " << '\n';
    Eigen::Affine3f mat= cfg["matrix"] ;
    std::cout << "mat as matrix is \n" << mat.matrix() << '\n';

    bool do_thing_true=cfg["do_thing_true"];
    bool do_thing_false=cfg["do_thing_false"];

    std::cout << "true and false " << do_thing_true << " " << do_thing_false << '\n';



}

void DataLoaderPNG::init_data_reading(){
    std::cout << "init data reading" << '\n';

    m_idx_img_to_read_per_cam.resize(m_nr_cams,0);
    m_rgb_filenames_per_cam.resize(m_nr_cams);
    m_last_frame_per_cam.resize(m_nr_cams);
    m_get_last_published_frame_for_cam.resize(m_nr_cams,false);
    m_undistort_map_x_per_cam.resize(m_nr_cams);
    m_undistort_map_y_per_cam.resize(m_nr_cams);


    for (size_t i = 0; i < m_nr_cams; i++) {
        if(!fs::is_directory(m_rgb_imgs_path_per_cam[i])) {
            LOG(FATAL) << "No directory " << m_rgb_imgs_path_per_cam[i];
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
            continue;
        }

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

            //POSE---
            if (!get_pose_at_timestamp(frame.tf_cam_world, timestamp, cam_id )){
                LOG(WARNING) << "Not found any pose at timestamp " << timestamp << " Discarding";
                continue;
            }

            //intrinsics
            get_intrinsics(frame.K, frame.distort_coeffs, cam_id);




            //Get images, rgb, gradients etc
            TIME_START("read_imgs");
            frame.rgb=cv::imread(rgb_filename.string());

            //gray
            cv::cvtColor ( frame.rgb, frame.gray, CV_BGR2GRAY );
            frame.gray.convertTo(frame.gray, CV_32F);
            frame.gray/=255.0; //gray is normalized
            frame.gray=undistort_image(frame.gray, frame.K, frame.distort_coeffs, cam_id); //undistort only the gray image because rgb is only used for visualization
            // frame.gray/=255.0;

            //TODO remove this as we only use the rgb for visualization and debug
            frame.rgb=undistort_image(frame.rgb, frame.K, frame.distort_coeffs, cam_id);

            //gradients
            cv::Scharr( frame.gray, frame.grad_x, CV_32F, 1, 0);
            cv::Scharr( frame.gray, frame.grad_y, CV_32F, 0, 1);
            frame.grad_x = cv::abs(frame.grad_x);
            frame.grad_y = cv::abs(frame.grad_y);

            //merge the gray image and the gradients into one 3 channel image
            std::vector<cv::Mat> channels;
            channels.push_back(frame.gray);
            channels.push_back(frame.grad_x);
            channels.push_back(frame.grad_y);
            cv::merge(channels, frame.gray_with_gradients);
            TIME_END("read_imgs");

            // std::cout << "pusing frame with tf corld of " << frame.tf_cam_world.matrix() << '\n';
            // std::cout << "pusing frame with K of " << frame.K << '\n';


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




// void DataLoaderPNG::read_pose_file(){
//     std::ifstream infile( m_pose_file );
//     if(!infile.is_open()){
//         LOG(FATAL) << "Could not open pose file " << m_pose_file;
//     }
//     uint64_t scan_nr;
//     uint64_t timestamp;
//     Eigen::Vector3f position;
//     Eigen::Quaternionf quat;
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
//         // if(m_do_timestamp_rounding_when_reading_file){
//         //     timestamp=(uint64_t)std::round(timestamp/100000.0); ////TODO this is a dirty hack to reduce the discretization of time because the timestamps don't exactly match
//         // }
//        // VLOG(2) << "recorded tmestamp is " << timestamp;
// //        VLOG(2) << "recorded scan_nr is " << scan_nr;
//         m_worldROS_baselink_map[timestamp]=pose;
//         m_worldROS_baselink_vec.push_back ( std::pair<uint64_t, Eigen::Affine3f>(timestamp,pose) );
//     }
//
// }

void DataLoaderPNG::read_pose_file_eth(){
    std::ifstream infile( m_pose_file );
    if(!infile.is_open()){
        LOG(FATAL) << "Could not open pose file " << m_pose_file;
    }
    VLOG(1) << "Reading pose file for ETH mav dataset";

    uint64_t scan_nr;
    uint64_t timestamp;
    Eigen::Vector3f position;
    Eigen::Quaternionf quat;

    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);

        //skip comments
        if(line.at(0)=='#'){
            continue;
        }

        std::vector<std::string> tokens=split(line,",");
        timestamp=stod(tokens[0]);
        position(0)=stod(tokens[1]);
        position(1)=stod(tokens[2]);
        position(2)=stod(tokens[3]);
        quat.w()=stod(tokens[4]);
        quat.x()=stod(tokens[5]);
        quat.y()=stod(tokens[6]);
        quat.z()=stod(tokens[7]);

        // std::cout << "input is \n" << " " << timestamp << " " << position << " " << quat.matrix()  << "\n";
        Eigen::Affine3f pose;
        pose.matrix().block<3,3>(0,0)=quat.toRotationMatrix();
        pose.matrix().block<3,1>(0,3)=position;

        m_worldROS_baselink_map[timestamp]=pose;
        m_worldROS_baselink_vec.push_back ( std::pair<uint64_t, Eigen::Affine3f>(timestamp,pose) );
    }

}

void DataLoaderPNG::read_pose_file_icl(){
    std::ifstream infile( m_pose_file );
    if(!infile.is_open()){
        LOG(FATAL) << "Could not open pose file " << m_pose_file;
    }
    VLOG(1) << "Reading pose file for ICL-NUIM dataset";

    uint64_t scan_nr;
    uint64_t timestamp;
    Eigen::Vector3f position;
    Eigen::Quaternionf quat;

    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);

        //skip comments
        if(line.at(0)=='#'){
            continue;
        }

        iss >> timestamp
            >> position(0) >> position(1) >> position(2)
            >> quat.x() >> quat.y() >> quat.z() >> quat.w();

        // std::cout << "input is \n" << " " << timestamp << " " << position << " " << quat.matrix()  << "\n";
        Eigen::Affine3f pose;
        pose.matrix().block<3,3>(0,0)=quat.toRotationMatrix();
        pose.matrix().block<3,1>(0,3)=position;



        m_worldROS_baselink_map[timestamp]=pose;
        m_worldROS_baselink_vec.push_back ( std::pair<uint64_t, Eigen::Affine3f>(timestamp,pose) );
    }

}

bool DataLoaderPNG::get_pose_at_timestamp(Eigen::Affine3f& pose, const uint64_t timestamp, const uint64_t cam_id){


    //return the closest one
    uint64_t closest_idx=-1;
    double smallest_timestamp_diff=std::numeric_limits<double>::max();
    for (size_t i = 0; i < m_worldROS_baselink_vec.size(); i++) {
        uint64_t recorded_timestamp=m_worldROS_baselink_vec[i].first;
        Eigen::Affine3f pose=m_worldROS_baselink_vec[i].second;
        // std::cout << "comparing recorded_timestamp to timestmp \n" << recorded_timestamp << "\n" << timestamp << '\n';
        double diff=fabs((double)timestamp- (double)recorded_timestamp);
        if (  diff < smallest_timestamp_diff){
            closest_idx=i;
            smallest_timestamp_diff=diff;
            // std::cout << "smallest_timestamp_diff " << smallest_timestamp_diff << '\n';
        }
    }
    // if ( smallest_timestamp_diff > 1e7 )
    // {
    //     LOG(WARNING) << "time difference for pose is way too large! " << (smallest_timestamp_diff/1e6) << "s." << '\n';
    //     return false;
    // }
    // std::cout << "smallest_timestamp_diff is " << smallest_timestamp_diff << '\n';
    // std::cout << "smallest_timestamp_diff_no_abs is " << smallest_timestamp_diff_no_abs << '\n';
    // std::cout << "deviation_ms is " << deviation_ms << '\n';
    Eigen::Affine3f pose_from_file=m_worldROS_baselink_vec[closest_idx].second;


    //this pose may be already the correct one or it may be transfromed ot another frame depending on the dataset type
    if(m_dataset_type==DatasetType::ETH){
        // pose_from_file is only the transformation from base to world
        if(cam_id==0){
            // camera to base link
            Eigen::Matrix4f tf_baselink_cam;
            tf_baselink_cam.row(0) << 0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975;
            tf_baselink_cam.row(1) << 0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768;
            tf_baselink_cam.row(2) << -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949;
            tf_baselink_cam.row(3) <<  0.0, 0.0, 0.0, 1.0;

            //pose is only from base to world but we need to return a pose that is tf_cam_world (so from world to cam)
            pose= Eigen::Affine3f(tf_baselink_cam).inverse() *  pose_from_file.inverse(); //world to base and base to cam
        }else if(cam_id==1){
            // camera to base link
            Eigen::Matrix4f tf_baselink_cam;
            tf_baselink_cam.row(0) << 0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556;
            tf_baselink_cam.row(1) << 0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024;
            tf_baselink_cam.row(2) << -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038;
            tf_baselink_cam.row(3) << 0.0, 0.0, 0.0, 1.0;

            //pose is only from base to world but we need to return a pose that is tf_cam_world (so from world to cam)
            pose= Eigen::Affine3f(tf_baselink_cam).inverse() *  pose_from_file.inverse(); //world to base and base to cam
        }else{
            LOG(FATAL) << "Now a known cam_id at " << cam_id;
        }

    }else if(m_dataset_type==DatasetType::ICL){
        pose=pose_from_file.inverse();
    }else{
        LOG(FATAL) << "Unknown dataset";
    }

    // std::cout << "closest idx is " << closest_idx << '\n';
    // std::cout << " timestamp is " << timestamp << " closest timestamp is " << m_worldROS_baselink_vec[closest_idx].first << '\n';
    // std::cout << "returning cam pose \n" << pose.matrix()  << '\n';


    return true;


}

void DataLoaderPNG::get_intrinsics(Eigen::Matrix3f& K, Eigen::Matrix<float, 5, 1>& distort_coeffs, const uint64_t cam_id){
    K.setIdentity();

    if(m_dataset_type==DatasetType::ETH){
        K.setIdentity();
        if(cam_id==0){
            K(0,0) = 458.654;
            K(1,1) = 457.296;
            K(0,2) = 367.215;
            K(1,2) = 248.375;
            distort_coeffs(0) = -0.28340811;
            distort_coeffs(1) = 0.07395907;
            distort_coeffs(2) = 0.00019359;
            distort_coeffs(3) = 1.76187114e-05;
            distort_coeffs(4) = 0.;
        }else if(cam_id==1){
            K(0,0) = 457.587;
            K(1,1) = 456.134;
            K(0,2) = 379.999;
            K(1,2) = 255.238;
            distort_coeffs(0) = -0.28368365;
            distort_coeffs(1) = 0.07451284;
            distort_coeffs(2) = -0.00010473;
            distort_coeffs(3) = -3.55590700e-05;
            distort_coeffs(4) = 0.;
        }
    }else if(m_dataset_type==DatasetType::ICL){
        K.setIdentity();
        if(cam_id==0){
            K(0,0)=481.2; //fx
            K(1,1)=-480; //fy
            K(0,2)=319.5; // cx
            K(1,2)=239.5; //cy
            distort_coeffs.setZero();
        }else if(cam_id==1){
            //even though we have one cam we set this one too because it's easier to deal with it like this for now.
            K(0,0)=481.2; //fx
            K(1,1)=-480; //fy
            K(0,2)=319.5; // cx
            K(1,2)=239.5; //cy
            distort_coeffs.setZero();
        }
    }else{
        LOG(FATAL) << "Unknown dataset";
    }
}



void DataLoaderPNG::create_transformation_matrices(){



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

cv::Mat DataLoaderPNG::undistort_image(const cv::Mat& gray_img, const Eigen::Matrix3f& K, const Eigen::VectorXf& distort_coeffs, const int cam_id){

    TIME_START("undistort");
    //if we don't have the undistorsion maps yet, create them
    if ( m_undistort_map_x_per_cam[cam_id].empty() ||  m_undistort_map_y_per_cam[cam_id].empty() ){
        cv::Mat_<double> Kc = cv::Mat_<double>::eye( 3, 3 );
        Kc (0,0) = K(0,0);
        Kc (1,1) = K(1,1);
        Kc (0,2) = K(0,2);
        Kc (1,2) = K(1,2);
        cv::Mat_<double> distortion ( 5, 1 );
        distortion ( 0 ) = distort_coeffs(0);
        distortion ( 1 ) = distort_coeffs(1);
        distortion ( 2 ) = distort_coeffs(2);
        distortion ( 3 ) = distort_coeffs(3);
        distortion ( 4 ) = distort_coeffs(4);
        cv::Mat_<double> Id = cv::Mat_<double>::eye ( 3, 3 );
        cv::initUndistortRectifyMap ( Kc, distortion, Id, Kc, gray_img.size(), CV_32FC1, m_undistort_map_x_per_cam[cam_id], m_undistort_map_y_per_cam[cam_id] );
    }

    cv::Mat undistorted_img;
    cv::remap ( gray_img, undistorted_img, m_undistort_map_x_per_cam[cam_id], m_undistort_map_y_per_cam[cam_id], cv::INTER_LINEAR );
    // gray_img=undistorted_img.clone(); //remap cannot function in-place so we copy the gray image back
    TIME_END("undistort");
    return undistorted_img;

}
