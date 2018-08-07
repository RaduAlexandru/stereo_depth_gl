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


//readerwriterqueue
#include "readerwriterqueue/readerwriterqueue.h"

//boost
#include <boost/bind.hpp>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#define BUFFER_SIZE 16

struct file_timestamp_comparator
{
    inline bool operator() (const fs::path& lhs, const fs::path& rhs)
    {
        double lhs_val=stod(lhs.stem().string());
        double rhs_val=stod(rhs.stem().string());
        return (lhs_val < rhs_val);
    }
};


//forward declarations
class Profiler;

class DataLoaderPNG{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DataLoaderPNG();
    ~DataLoaderPNG();
    void start_reading();

    bool is_finished(); //returns true when we have finished reading AND processing everything
    bool is_finished_reading(); //returns true when we have finished reading everything but maybe not processing
    bool has_data_for_cam(const int cam_id);
    bool has_data_for_all_cams();
    Frame get_next_frame_for_cam(const int cam_id);
    int get_nr_cams(){return m_nr_cams; };
    void set_mask_for_cam(const std::string mask_filename, const int cam_id); //set a mask which will cause parts of the rgb, classes and probs images to be ignored
    void republish_last_frame_from_cam(const int cam_id); //put the last frame back into the ringbuffer so we cna read it from the core
    void republish_last_frame_all_cams();


    // void read_data();
    // Frame get_next_frame();
    void reset(); //starts reading back from the start of the data log
    void clear_buffers(); //empties the ringbuffers, usefull for when scrolling through time

    //objects
    std::shared_ptr<Profiler> m_profiler;

    //transforms
    Eigen::Affine3f m_tf_alg_vel; //transformation from velodyne frame to the algorithm frame
    Eigen::Affine3f m_tf_baselink_vel;
    Eigen::Affine3f m_tf_worldGL_worldROS;
    std::unordered_map<uint64_t, Eigen::Affine3f> m_worldROS_baselink_map;
    std::vector<std::pair<uint64_t, Eigen::Affine3f> >m_worldROS_baselink_vec;

    std::vector< std::vector<fs::path> > m_rgb_filenames_per_cam; //list of images paths for each cam to read
    std::vector<int> m_idx_img_to_read_per_cam;




private:

    std::vector< moodycamel::ReaderWriterQueue<Frame> > m_frames_buffer_per_cam;



    std::vector<std::thread> m_loader_threads;
    std::vector< fs::path > m_rgb_imgs_path_per_cam;
    std::vector< fs::path > m_labels_imgs_path_per_cam;
    std::vector< Eigen::Matrix3f > m_intrinsics_per_cam;
    std::vector<cv::Mat > m_mask_per_cam;
    int m_nr_cams;
    // int m_idx_img_to_read;


    //params
    float m_tf_worldGL_worldROS_angle;
    std::string m_tf_worldGL_worldROS_axis;
    std::string m_pose_file;
    std::vector<Frame> m_last_frame_per_cam; //stores the last frame for each of the cameras
    std::vector<bool> m_get_last_published_frame_for_cam; //if we shoudl return the last published frame or not

    float m_rgb_subsample_factor;
    int m_imgs_to_skip;
    int m_nr_images_to_read; //nr images to read starting from m_imgs_to_skip


    void init_params();
    void init_params_configuru();
    void init_data_reading();
    void read_pose_file();
    void read_data_for_cam(const int cam_id);
    // void read_pose_file_semantic_fusion();
    bool get_pose_at_timestamp(Eigen::Affine3f& pose, const uint64_t timestamp);
    void create_transformation_matrices();

};



#define TIME_SCOPE(name)\
    TIME_SCOPE_2(name,m_profiler);

#define TIME_START(name)\
    TIME_START_2(name,m_profiler);

#define TIME_END(name)\
    TIME_END_2(name,m_profiler);
