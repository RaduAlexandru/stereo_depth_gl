#include "stereo_depth_cl/Pattern.h"

//c++
#include <iostream>

//My stuff
#include "stereo_depth_cl/MiscUtils.h"

//loguru
#include <loguru.hpp>

Pattern::Pattern(){

    std::string pattern_filepath="/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/data/pattern_1.png";
    init_pattern(pattern_filepath);
}

//needed so that forward declarations work
Pattern::~Pattern(){
}

void Pattern::init_pattern(const std::string& pattern_filepath){
    cv::Mat pattern_mat=cv::imread(pattern_filepath,CV_LOAD_IMAGE_GRAYSCALE);

    if(!pattern_mat.data ){
       LOG(FATAL) << "Could not find pattern file on " << pattern_filepath;
   }

    for (size_t i = 0; i < pattern_mat.rows; i++) {
        for (size_t j = 0; j < pattern_mat.cols; j++) {
            if(pattern_mat.at<uchar>(i,j)==0){
                m_offsets_x.push_back(j-pattern_mat.cols/2);
                m_offsets_y.push_back(i-pattern_mat.rows/2);
            }
        }
    }

}

int Pattern::get_nr_points(){
    return m_offsets_x.size();
}

Eigen::Vector2d Pattern::get_offset(const int point_idx){
    Eigen::Vector2d offsets;
    offsets << m_offsets_x[point_idx], m_offsets_y[point_idx];
    return offsets;
}

int Pattern::get_offset_x(const int point_idx){
    return m_offsets_x[point_idx];
}

int Pattern::get_offset_y(const int point_idx){
    return m_offsets_y[point_idx];
}

Pattern Pattern::rotate_pattern(const Eigen::Matrix2d& rotation){
    Pattern rotated_pattern;
    rotated_pattern.m_offsets_x=m_offsets_x;
    rotated_pattern.m_offsets_y=m_offsets_y;
    for (size_t i = 0; i < get_nr_points(); i++) {
        Eigen::Vector2d rotated_offset=rotation * Eigen::Vector2d(get_offset_x(i), get_offset_y(i));
        rotated_pattern.m_offsets_x[i]=rotated_offset(0);
        rotated_pattern.m_offsets_y[i]=rotated_offset(1);
    }
    return rotated_pattern;
}
