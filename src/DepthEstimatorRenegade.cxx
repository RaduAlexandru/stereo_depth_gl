#include "stereo_depth_gl/DepthEstimatorRenegade.h"

//c++
#include <cmath>
#include <stdlib.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <fstream>

//My stuff
#include "stereo_depth_gl/Profiler.h"
#include "stereo_depth_gl/MiscUtils.h"

//Libigl
#include <igl/opengl/glfw/Viewer.h>

//cv
#include <cv_bridge/cv_bridge.h>

//loguru
#include <loguru.hpp>

//omp
#include <omp.h>

//renegade module

#include <cmath>
#include <cstdio>
#include <iostream>
// #include "ceres/ceres.h"
// #include "ceres/cubic_interpolation.h"
// #include "ceres/rotation.h"
#include <algorithm>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <Eigen/Dense>
#include <fstream>
#include <omp.h>

//renegade
// #include "types.h"
// #include "depth_point.h"
// #include "depth_filter.h"
// #include "depth_opt.h"


DepthEstimatorRenegade::DepthEstimatorRenegade():
        m_scene_is_modified(false)
        {

}

//needed so that forward declarations work
DepthEstimatorRenegade::~DepthEstimatorRenegade(){
}


Eigen::Vector3d inline unproject ( const double & row, const double & col, const double & depthValue , const Eigen::Matrix3d & K ){
   return Eigen::Vector3d ( ( col - K ( 0, 2 ) ) * depthValue / K ( 0, 0 ), ( row - K ( 1, 2 ) ) * depthValue / K ( 1, 1 ), depthValue );
}

Mesh DepthEstimatorRenegade::compute_depth(Frame& frame){

    TIME_START("compute_depth");
    Mesh mesh;


    mesh.V.resize(640*480,3);
    mesh.V.setZero();

    Eigen::Matrix3d K;
    K.setZero();

    //iclnuim
    // K(0,0)=481.2; //fx
    // K(1,1)=-480; //fy
    // K(0,2)=319.5; // cx
    // K(1,2)=239.5; //cy
    // K(2,2)=1.0;

    // rgbd tum freiburg 3
    K(0,0)=535.4; //fx
    K(1,1)=539.2 ; //fy
    K(0,2)=320.1 ; // cx
    K(1,2)=247.6; //cy
    K(2,2)=1.0;

    std::ifstream infile( "/media/alex/Data/Master/SHK/c_ws/src/RENEGADE/results/inverse_depth_filter.txt" );
    std::string line;
    int x,y, seed_converged, seed_is_outlier;
    float idepth_val, denoised_idepth_val;
    float a,b, sigma2;

    int nr_valid_points=0;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        iss >> x >> y >> idepth_val >> seed_converged >> seed_is_outlier >> denoised_idepth_val >> a >> b >> sigma2;
        int idx_linear = y*480 + x;


        // std::cout << "readl values " << x << " " << y << " " << idepth_val << " " << seed_converged << " " << seed_is_outlier << " " << denoised_idepth_val << '\n';
        // if(std::isfinite(idepth_val) && idepth_val>=0.1 && seed_converged==1 && seed_is_outlier==0){
        if(std::isfinite(idepth_val) && idepth_val>=0.1){
        // if(std::isfinite(denoised_idepth_val) && denoised_idepth_val>=0.001){


            //check the a and b for inliers
            // float diff=a-b;
            // if(diff<23.5){
            //     continue;
            // }

            // float outlier_measure=a/(a+b);
            // if(outlier_measure<0.7){
            //     continue;
            // }
            //
            // if(sigma2>0.01){
            //     continue;
            // }

            // std::cout << "putting into " << idx_linear  <<  " values " << x << " " << y << " " << 1/denoised_idepth_val << '\n';
            Eigen::Vector3d point_cam_coords=unproject(y,x, 1/idepth_val,K);
            // mesh.V.row(idx_linear) << x, -y, depth_val;

            point_cam_coords(2)=-point_cam_coords(2);
            mesh.V.row(idx_linear) = point_cam_coords;
            nr_valid_points++;
        }
    }
    std::cout << "nr of valid points " << nr_valid_points << '\n';







    TIME_END("compute_depth");
    return mesh;
}
