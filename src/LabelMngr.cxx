#include "stereo_depth_gl/LabelMngr.h"

//c++
#include <iostream>
#include <fstream>


LabelMngr::LabelMngr():
    m_nr_classes(0){
}

void LabelMngr::init(std::string file_with_classes){
    std::ifstream input(file_with_classes );
    for( std::string line; getline( input, line ); ){
        m_idx2label.push_back(line);
        m_label2idx[line]=m_nr_classes;
        m_nr_classes++;
        // std::cout << "read: " << line << '\n';
    }
    initialize_colormap_with_mapillary_colors(m_C_per_class);
    std::cout << "LabelMngr: nr of classes read " << m_nr_classes << '\n';

    // std::cout << "colors is \n" << m_C_per_class << '\n';
}

int LabelMngr::get_nr_classes(){
    return m_nr_classes;
}

int LabelMngr::get_idx_unlabeled(){
    return m_label2idx["Unlabeled"];
}

int LabelMngr::get_idx_invalid(){
    return m_nr_classes+1;
}

std::string LabelMngr::idx2label(int idx){
    return m_idx2label[idx];
}
int LabelMngr::label2idx(std::string label){
    return m_label2idx[label];
}








cv::Mat LabelMngr::apply_color_map(cv::Mat classes){
    cv::Mat classes_colored(classes.rows, classes.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    for (size_t i = 0; i < classes.rows; i++) {
        for (size_t j = 0; j < classes.cols; j++) {
            int label=(int)classes.at<unsigned char>(i,j);
            classes_colored.at<cv::Vec3b>(i,j)[0]=m_C_per_class(label,2)*255; //remember opencv is BGR so we put the B first
            classes_colored.at<cv::Vec3b>(i,j)[1]=m_C_per_class(label,1)*255; //remember opencv is BGR so we put the B first
            classes_colored.at<cv::Vec3b>(i,j)[2]=m_C_per_class(label,0)*255; //remember opencv is BGR so we put the B first
        }
    }
    return classes_colored;
}

void LabelMngr::initialize_colormap_with_visutils_colors(Eigen::MatrixXd& C_per_class){
    C_per_class.resize(100,3);
    C_per_class.setZero();

    C_per_class <<  216,222,105,
                	71,55,202,
                	128,231,64,
                	153,66,227,
                	207,241,61,
                	229,82,234,
                	77,225,103,
                	194,41,193,
                	74,172,40,
                	124,48,177,
                	231,228,59,
                	59,31,137,
                	166,214,67,
                	181,100,232,
                	135,221,113,
                	227,56,168,
                	88,229,154,
                	160,55,158,
                	194,240,132,
                	101,105,231,
                	225,194,41,
                	55,78,173,
                	157,178,43,
                	96,33,118,
                	119,169,68,
                	218,114,216,
                	60,147,63,
                	145,108,208,
                	234,161,48,
                	95,130,223,
                	237,64,32,
                	89,232,203,
                	215,43,58,
                	84,181,121,
                	219,47,103,
                	160,229,164,
                	49,18,76,
                	213,236,158,
                	165,44,117,
                	157,189,116,
                	45,55,113,
                	231,104,37,
                	93,203,219,
                	179,52,21,
                	109,158,222,
                	214,117,48,
                	79,112,170,
                	197,175,69,
                	113,80,151,
                	233,218,132,
                	43,21,36,
                	202,231,195,
                	90,23,54,
                	153,222,203,
                	138,34,43,
                	70,166,139,
                	228,90,79,
                	77,148,169,
                	160,72,35,
                	153,192,225,
                	177,130,36,
                	190,145,223,
                	55,97,26,
                	215,103,170,
                	122,134,40,
                	120,49,101,
                	220,214,165,
                	61,51,81,
                	227,167,97,
                	31,44,45,
                	229,97,135,
                	51,110,69,
                	219,147,198,
                	34,62,36,
                	208,186,229,
                	135,85,24,
                	190,222,228,
                	78,36,32,
                	234,210,200,
                	113,59,35,
                	138,168,146,
                	168,69,102,
                	111,141,89,
                	201,98,96,
                	72,113,102,
                	228,130,95,
                	59,91,116,
                	177,165,101,
                	134,100,140,
                	118,107,43,
                	131,135,163,
                	169,117,69,
                	179,147,163,
                	78,74,41,
                	229,169,178,
                	131,84,91,
                	223,170,135,
                	221,134,145,
                	182,163,140,
                	148,115,96;
    C_per_class.array()/=255.0;
}

void LabelMngr::initialize_colormap_with_mapillary_colors(Eigen::MatrixXd& C_per_class){
    C_per_class.resize(66,3);
    C_per_class.setZero();

    C_per_class <<  165, 42, 42,
                    0, 192, 0,
                    196, 196, 196,
                    190, 153, 153,
                    180, 165, 180,
                    102, 102, 156,
                    102, 102, 156,
                    128, 64, 255,
                    140, 140, 200,
                    170, 170, 170,
                    250, 170, 160,
                    96, 96, 96,
                    230, 150, 140,
                    128, 64, 128,
                    110, 110, 110,
                    244, 35, 232,
                    150, 100, 100,
                    70, 70, 70,
                    150, 120, 90,
                    220, 20, 60,
                    255, 0, 0,
                    255, 0, 0,
                    255, 0, 0,
                    200, 128, 128,
                    255, 255, 255,
                    64, 170, 64,
                    128, 64, 64,
                    70, 130, 180,
                    255, 255, 255,
                    152, 251, 152,
                    107, 142, 35,
                    0, 170, 30,
                    255, 255, 128,
                    250, 0, 30,
                    0, 0, 0,
                    220, 220, 220,
                    170, 170, 170,
                    222, 40, 40,
                    100, 170, 30,
                    40, 40, 40,
                    33, 33, 33,
                    170, 170, 170,
                    0, 0, 142,
                    170, 170, 170,
                    210, 170, 100,
                    153, 153, 153,
                    128, 128, 128,
                    0, 0, 142,
                    250, 170, 30,
                    192, 192, 192,
                    220, 220, 0,
                    180, 165, 180,
                    119, 11, 32,
                    0, 0, 142,
                    0, 60, 100,
                    0, 0, 142,
                    0, 0, 90,
                    0, 0, 230,
                    0, 80, 100,
                    128, 64, 64,
                    0, 0, 110,
                    0, 0, 70,
                    0, 0, 192,
                    32, 32, 32,
                    0, 0, 0,
                    0, 0, 0;
    C_per_class.array()/=255.0;
}
