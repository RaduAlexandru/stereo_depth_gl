#ifndef TYPES_H_
#define TYPES_H_

#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <memory>

const int patchSizeHalf = 1;
const int patchSize = patchSizeHalf * 2 + 1;
const int patchSize2 = patchSize*patchSize;

struct ParamsCfg
{
    double fx = 640;
    double fy = 640;
    double cx = 480;
    double cy = 320;
    double k1 = 0;
    double k2 = 0;
    double p1 = 0;
    double p2 = 0;
    double k3 = 0;
    int lvl = 5;
    bool undistort = true;
    bool useRPG = true;
};

struct ImageData
{
    explicit ImageData( const std::string & color, const std::string & depth, const Eigen::Affine3d & pose_cw )
        : colorFile(color), depthFile ( depth ), m_pose_cw( pose_cw ){}

    Eigen::Affine3d m_pose_cw;
    std::string colorFile;
    std::string depthFile;
    std::vector<cv::Mat> grayImages;
    std::vector<cv::Mat> gradImagesX;
    std::vector<cv::Mat> gradImagesY;
    std::vector<cv::Mat> depthImages;
    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d> > K_c;
    std::vector<Eigen::Matrix2Xd, Eigen::aligned_allocator<Eigen::Matrix2Xd> > kp_c;
    //std::vector<std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d> > > patch_c;
    std::vector<Eigen::Matrix3Xd, Eigen::aligned_allocator<Eigen::Matrix3Xd> > p_c;
    std::vector<Eigen::RowVectorXd, Eigen::aligned_allocator<Eigen::RowVectorXd> > I_c;
    std::vector<Eigen::Matrix2Xd, Eigen::aligned_allocator<Eigen::Matrix2Xd> > Ig_c;

    Eigen::Vector3f* dI = nullptr;
    Eigen::Affine3d m_pose_cr;

    ~ImageData(){if ( dI ) delete[] dI; dI = nullptr; }



    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};
typedef std::shared_ptr< ImageData > ImageDataPtr; 


#endif // TYPES_H_
