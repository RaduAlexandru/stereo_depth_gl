#pragma once
//C++
#include <iosfwd>
#include <memory>
#include <atomic>

//OpenCV
#include <opencv2/highgui/highgui.hpp>

//Eigen
#include <Eigen/Dense>

class Pattern{
public:
    Pattern();
    ~Pattern();
    void init_pattern(const std::string& pattern_filepath);
    int get_nr_points(); //nr of active points in the pattern (black pixels of the image)
    Eigen::Vector2f get_offset(const int point_idx); //offset in x,y
    int get_offset_x(const int point_idx);
    int get_offset_y(const int point_idx);
    Eigen::MatrixXf get_offset_matrix();
    Pattern get_rotated_pattern(const Eigen::Matrix2f& rotation);
    Eigen::Vector2i get_size(); //get the size of the pattern image in x and y


private:


//    std::vector<int> m_offsets_x; //ofsets of the pattern points with repect to the central pixel
//    std::vector<int> m_offsets_y;

    Eigen::MatrixXf m_offsets; //2xN ofsets of the pattern points with repect to the central pixel (x,y)
    Eigen::Vector2i m_pattern_size; //size of the pattern in x and y, usually it's 5x5


};
