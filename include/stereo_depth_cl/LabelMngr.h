#pragma once
//C++
#include <iosfwd>
#include <vector>
#include <unordered_map>

//OpenCV
#include <opencv2/highgui/highgui.hpp>

//Eigen
#include <Eigen/Dense>


class LabelMngr{
public:
    LabelMngr();
    void init(std::string file_with_classes);
    int get_nr_classes();
    int get_idx_unlabeled();  //idx of the class that is assigned to unabeled
    int get_idx_invalid();    //idx of a class that will for sure never appear in the dataset, like -1 or nr_classes +1 (we use the latter for the reason that store then in unsigned data storage)
    std::string idx2label(int idx);
    int label2idx(std::string label);
    cv::Mat apply_color_map(cv::Mat classes);

private:
    int m_nr_classes;
    std::vector<std::string> m_idx2label;
    std::unordered_map<std::string,int> m_label2idx;
    Eigen::MatrixXd m_C_per_class;

    void initialize_colormap_with_visutils_colors(Eigen::MatrixXd& C_per_class);
    void initialize_colormap_with_mapillary_colors(Eigen::MatrixXd& C_per_class);

};
