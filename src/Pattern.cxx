#include "stereo_depth_cl/Pattern.h"

//c++
#include <iostream>

//My stuff
#include "stereo_depth_cl/MiscUtils.h"

//loguru
#include <loguru.hpp>

Pattern::Pattern(){

}

//needed so that forward declarations work
Pattern::~Pattern(){
}

void Pattern::init_pattern(const std::string& pattern_filepath){
    cv::Mat pattern_mat=cv::imread(pattern_filepath,CV_LOAD_IMAGE_GRAYSCALE);

    if(!pattern_mat.data ){
       LOG(FATAL) << "Could not find pattern file on " << pattern_filepath;
   }

    std::vector<int> offsets_x; //ofsets of the pattern points with repect to the central pixel
    std::vector<int> offsets_y;

    for (size_t i = 0; i < pattern_mat.rows; i++) {
        for (size_t j = 0; j < pattern_mat.cols; j++) {
            if(pattern_mat.at<uchar>(i,j)==0){
                offsets_x.push_back(j-pattern_mat.cols/2);
                offsets_y.push_back(i-pattern_mat.rows/2);
            }
        }
    }

    m_offsets.resize(offsets_x.size(), 2);
    for (int i = 0; i < offsets_x.size(); ++i) {
        m_offsets.row(i) << offsets_x[i] , offsets_y[i];
    }

}

int Pattern::get_nr_points(){
//    return m_offsets_x.size();
    return m_offsets.rows();
}

Eigen::Vector2d Pattern::get_offset(const int point_idx){
//    Eigen::Vector2d offsets;
//    offsets << m_offsets_x[point_idx], m_offsets_y[point_idx];
//    return offsets;
    return m_offsets.row(point_idx);
}


int Pattern::get_offset_x(const int point_idx){
//    return m_offsets_x[point_idx];
    return m_offsets(point_idx,0);
}

int Pattern::get_offset_y(const int point_idx){
//    return m_offsets_y[point_idx];
    return m_offsets(point_idx,1);
}

Pattern Pattern::get_rotated_pattern(const Eigen::Matrix2d& rotation){
//    Pattern rotated_pattern;
//    rotated_pattern.m_offsets_x=m_offsets_x;
//    rotated_pattern.m_offsets_y=m_offsets_y;
//    for (size_t i = 0; i < get_nr_points(); i++) {
//        Eigen::Vector2d rotated_offset=rotation * Eigen::Vector2d(get_offset_x(i), get_offset_y(i));
//        rotated_pattern.m_offsets_x[i]=rotated_offset(0);
//        rotated_pattern.m_offsets_y[i]=rotated_offset(1);
//    }
//    return rotated_pattern;

//    Pattern rotated_pattern=*this;
//    rotated_

    Pattern rotated_pattern;
    rotated_pattern.m_offsets=m_offsets;
    for (int i = 0; i < m_offsets.rows(); ++i) {
        rotated_pattern.m_offsets.row(i)=rotation*m_offsets.row(i).transpose();
    }
//    rotated_pattern.m_offsets=rotation*m_offsets.transpose();
    return rotated_pattern;

}

