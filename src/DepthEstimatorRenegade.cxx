#include "stereo_depth_cl/DepthEstimatorRenegade.h"

//c++
#include <cmath>
#include <stdlib.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <fstream>

//My stuff
#include "stereo_depth_cl/Profiler.h"
#include "stereo_depth_cl/MiscUtils.h"
#include "UtilsCL.h"

//Libigl
#include <igl/opengl/glfw/Viewer.h>

//cv
#include <cv_bridge/cv_bridge.h>

//loguru
#include <loguru.hpp>

//omp
#include <omp.h>

//renegade module
#include "types.h"
#include <cmath>
#include <cstdio>
#include <iostream>
#include "ceres/ceres.h"
#include "ceres/cubic_interpolation.h"
#include "ceres/rotation.h"
#include <algorithm>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <Eigen/Dense>
#include <fstream>
#include <omp.h>
#include "depth_point.h"
#include "depth_filter.h"
#include "depth_opt.h"


DepthEstimatorRenegade::DepthEstimatorRenegade():
        m_scene_is_modified(false)
        {

}

//needed so that forward declarations work
DepthEstimatorRenegade::~DepthEstimatorRenegade(){
}



Mesh DepthEstimatorRenegade::compute_depth(Frame& frame){

    TIME_START("compute_depth");
    Mesh mesh;

    // bool useRPG = false;
    // bool useModified = false;
    // int numImages = 60;
    // ParamsCfg cfg = setParams( useRPG );
    // std::vector<double> times;
    // const std::string filePath = useRPG ? "/home/jquenzel/bags/urban/rpg/" : "/media/alex/Data/Master/Thesis/data/ICL_NUIM/living_room_traj2_frei_png/";
    // const std::vector< ImageDataPtr > images = useRPG ?
    //             loadDataFromRPG ( filePath, numImages, times, useModified ) :
    //             loadDataFromICLNUIM( filePath, numImages, times, useModified );
    //
    // if ( images.empty() ){
    //     std::cerr << "no images found!" << std::endl;
    // }
    // optimizeForInverseDepth( images, cfg );



    //read back the depth from the opencv img
    cv::Mat depth_mat;
    std::cout << "read img" << '\n';
    depth_mat=cv::imread("/media/alex/Data/Master/SHK/c_ws/src/renegade/results/inverse_depth_filter_mat.png", CV_LOAD_IMAGE_ANYDEPTH);
    std::cout << "depth_mat is of type " << type2string(depth_mat.type()) << '\n';

    int nr_points=depth_mat.cols*depth_mat.rows;
    std::cout << "nr points is " << nr_points << '\n';
    mesh.V.resize(nr_points,3);
    mesh.V.setZero();

    // std::cout << "looping" << '\n';
    // for (size_t i = 0; i < depth_mat.rows; i++) {
    //     for (size_t j = 0; j < depth_mat.cols; j++) {
    //         int idx_linear = i*depth_mat.cols + j;
    //         // std::cout << "putting into " << idx_linear  <<  "values" << i << " " << j << " " << depth_mat.at<float>(i,j) << '\n';
    //         mesh.V.row(idx_linear) << j ,i ,depth_mat.at<uchar>(i,j);
    //     }
    // }



    std::ifstream infile( "/media/alex/Data/Master/SHK/c_ws/src/renegade/results/inverse_depth_filter.txt" );
    std::string line;
    int x,y;
    float depth_val;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        iss >> x >> y >> depth_val;
        int idx_linear = y*depth_mat.cols + x;
        // std::cout << "putting into " << idx_linear  <<  " values " << x << " " << y << " " << depth_val << '\n';
        mesh.V.row(idx_linear) << x, -y, depth_val;
    }







    TIME_END("compute_depth");
    return mesh;
}



//are we going to use the same windows approach as in dso which add frames and then margianzlies them afterwards? because that is tricky to do in opencl and would require some vector on the gpu which keeps track where the image are in a 3D image and then do like a rolling buffer

//get frame to gray
//get frame to float
//apply blur to img (as done at the finale of Undistort::undistort)

//cv_mat2cl_buf(img_gray)





// full_system:makeNewTraces //computes new imature points and adds them to the current frame
//     pixelSelector->make_maps()
//     for (size_t i = 0; i < count; i++) {
//         for (size_t i = 0; i < count; i++) {
//             if(selectionMap==)continue
//             create_imature_point which contains
//                 weights for each point in the patter (default 8)
//                 gradH value
//                 energuTH
//                 idepthGT
//                 quality
//         }
//     }
