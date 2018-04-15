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
    Eigen::Vector2d get_offset(const int point_idx); //offset in x,y
    int get_offset_x(const int point_idx);
    int get_offset_y(const int point_idx);
    Pattern get_rotated_pattern(const Eigen::Matrix2d& rotation);


private:


//    std::vector<int> m_offsets_x; //ofsets of the pattern points with repect to the central pixel
//    std::vector<int> m_offsets_y;

    Eigen::MatrixXd m_offsets; //Nx2 ofsets of the pattern points with repect to the central pixel (x,y)


};
