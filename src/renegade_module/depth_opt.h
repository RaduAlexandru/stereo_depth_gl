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

using namespace ceres;


cv::Mat readImageFile ( const std::string & fileName )
{
   cv::Mat readFile = cv::imread ( fileName, CV_LOAD_IMAGE_UNCHANGED );
   std::cout << "reading image: " << fileName << " r: " << readFile.rows << " c: " << readFile.cols << std::endl;
   return readFile.clone(); // clone not necessary?
}

cv::Mat readDepthFile ( const std::string & fileName, const ParamsCfg & cfg )
{
   // RPG only, depth is distance from origin to point along viewing ray
   cv::Mat readFile = cv::Mat::zeros(480,640,CV_32F);
   std::ifstream depthFile ( fileName, std::ifstream::in );
   int numNumbersToRead = 640*480, numbersRead = 0;
   float val = 0;
   for ( numbersRead = 0; depthFile.good() && numbersRead < numNumbersToRead ; ++numbersRead )
   {
      depthFile >> val;
      readFile.at<float>( numbersRead ) = val;
   }
   if ( numbersRead != numNumbersToRead)
   {
       std::cerr << "error during reading image." << std::endl;
       return cv::Mat();
   }

   for ( int row = 0; row < readFile.rows; ++row )
       for ( int col = 0; col < readFile.cols; ++col )
       {
           val = readFile.at<float>( row, col );
           if ( val > 1e10 )
               val = 0;
           else
           {
               Eigen::Vector3d f ( (col-cfg.cx)/cfg.fx, (row-cfg.cy)/cfg.fy, 1 );
               f.normalize();
               val *= f(2); // now depth
           }
           readFile.at<float>( row, col ) = val;
       }

//   cv::Mat pngDepth;
//   readFile.convertTo(pngDepth,CV_8UC1);
//   cv::imwrite("/home/jquenzel/depth.png",pngDepth);
   //std::cout << "reading image: " << fileName << " r: " << readFile.rows << " c: " << readFile.cols << " #"<< numbersRead << std::endl;
   return readFile.clone(); // clone not necessary?
}


Eigen::Vector3d inline unproject ( const double & row, const double & col, const double & depthValue , const Eigen::Matrix3d & K )
{
   return Eigen::Vector3d ( ( col - K ( 0, 2 ) ) * depthValue / K ( 0, 0 ), ( row - K ( 1, 2 ) ) * depthValue / K ( 1, 1 ), depthValue );
}

void computeAllPoints ( cv::Mat & gray, cv::Mat & depth, cv::Mat & gx, cv::Mat & gy, Eigen::Matrix2Xd & _kp_c, Eigen::Matrix3Xd & _p_c, Eigen::RowVectorXd & _intensity, Eigen::Matrix3d & K, Eigen::Matrix2Xd & Ig_c ) //std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d> > & _patch )
{
    Eigen::Matrix2Xd kp_c = Eigen::Matrix2Xd::Zero ( 2, gray.cols * gray.rows );
    Eigen::Matrix3Xd p_c = Eigen::Matrix3Xd::Zero ( 3, gray.cols * gray.rows );
    Eigen::RowVectorXd intensity = Eigen::RowVectorXd::Zero ( 1, gray.cols * gray.rows );
    //_patch.reserve ( gray.cols * gray.rows );
    Eigen::Matrix2Xd ig_c = Eigen::Matrix2Xd::Zero ( 2, gray.cols * gray.rows );

    int idx = 0;
    cv::Rect roi ( 1, 1, gray.cols-1, gray.rows-1 );
//    Eigen::Matrix3d patch;
    for ( int row = roi.y; row < roi.y + roi.height; ++row )
    {
        for ( int col = roi.x; col < roi.x + roi.width; ++col )
        {
            const float depthValue = depth.at<float> ( row, col );

            if (  depthValue > 0.01 && std::isfinite ( depthValue ) )
            {
                kp_c.col ( idx ).noalias() = Eigen::Vector2d ( col, row );
                p_c.col ( idx ).noalias() = unproject ( row, col, depthValue, K );
                intensity ( idx ) = float ( gray.at<uchar> ( row, col ) );
                ig_c.col(idx) <<  gx.at<float>(row,col), gy.at<float>(row,col) ;
//                for ( int dy = -1; dy <= 1; ++dy)
//                    for ( int dx = -1; dx <= 1; ++dx )
//                        patch(dx,dy) = gray.at<uchar>(row+dy,col+dx);

//                _patch.push_back( patch );
                ++idx;
            }
        }
    }
    _p_c = p_c.leftCols ( idx );
    _kp_c = kp_c.leftCols ( idx ) ;
    _intensity = intensity.leftCols ( idx );
    Ig_c = ig_c.leftCols(idx);
    //std::cout << " we have #" << idx << " candidates" << std::endl;
}

ImageDataPtr createImageData ( const std::string & colorFile, const std::string & depthFile, const Eigen::Affine3d & pose_cw )
{
    return std::make_shared<ImageData>( colorFile, depthFile, pose_cw );
}

void loadImageData ( ImageDataPtr & cur, const std::string & colorFile, const std::string & depthFile, ParamsCfg & cfg )
{
    cur->grayImages.resize(cfg.lvl);
    cur->gradImagesX.resize(cfg.lvl);
    cur->gradImagesY.resize(cfg.lvl);
    cur->depthImages.resize(cfg.lvl);
    cur->K_c.resize(cfg.lvl);
    cur->p_c.resize(cfg.lvl);
    cur->I_c.resize(cfg.lvl);
    cur->kp_c.resize(cfg.lvl);
    //cur->patch_c.resize(cfg.lvl);
    cur->Ig_c.resize(cfg.lvl);

    cv::Mat grayImg;
    cv::Mat colorImg = readImageFile( colorFile );
    cvtColor ( colorImg, grayImg, CV_BGR2GRAY );

    cv::Mat depthImg;
    if ( cfg.useRPG )
    {
        depthImg = readDepthFile( depthFile, cfg );
        depthImg.copyTo( cur->depthImages[0] );
    }
    else
    {
        depthImg = readImageFile ( depthFile );
        depthImg.convertTo ( cur->depthImages[0], CV_32F, 1./5000. );
    }

    if ( cfg.undistort )
    {
       static cv::Mat undistortMapX, undistortMapY;
       if ( undistortMapX.empty() || undistortMapY.empty() )
       {
          cv::Mat_<double> Kc = cv::Mat_<double>::eye( 3, 3 );
          Kc ( 0, 0 ) = cfg.fx;
          Kc ( 1, 1 ) = cfg.fy;
          Kc ( 0, 2 ) = cfg.cx;
          Kc ( 1, 2 ) = cfg.cy;
          cv::Mat_<double> distortion ( 5, 1 );
          distortion ( 0 ) = cfg.k1;
          distortion ( 1 ) = cfg.k2;
          distortion ( 2 ) = cfg.p1;
          distortion ( 3 ) = cfg.p2;
          distortion ( 4 ) = cfg.k3;
          cv::Mat_<double> Id = cv::Mat_<double>::eye ( 3, 3 );
          cv::initUndistortRectifyMap ( Kc, distortion, Id, Kc, grayImg.size(), CV_32FC1, undistortMapX, undistortMapY );
       }
       cv::remap ( grayImg, cur->grayImages[0], undistortMapX, undistortMapY, cv::INTER_LINEAR );
    }
    else
        grayImg.copyTo(cur->grayImages[0]);


    for ( int lvl = 0; lvl < cfg.lvl; ++lvl )
    {
        double scaleFactor = std::pow ( 0.5, lvl );

        if ( lvl > 0 )
        {
           Eigen::Matrix3d K_c = cur->K_c[lvl - 1];
           cur->K_c[lvl] << K_c ( 0, 0 ) *.5, 0, K_c ( 0, 2 ) *.5 - 0.25, 0, K_c ( 1, 1 ) *.5, K_c ( 1, 2 ) *.5 - 0.25, 0, 0, 1;
        }
        else
           cur->K_c[lvl] << cfg.fx, 0, cfg.cx, 0, cfg.fy, cfg.cy, 0, 0, 1;

        //std::cout << "scaleFactor: " << scaleFactor  << " for lvl: " << lvl << " K: " << std::endl << cur->K_c[lvl] << std::endl;

        // pyr down grayImg and depthImg
        if ( lvl+1 < cfg.lvl)
        {
            //cv::pyrDown( curRgbdData->grayImg[lvl], curRgbdData->grayImg[lvl+1]);
            cv::resize ( cur->grayImages[lvl], cur->grayImages[lvl + 1], cv::Size(), 1. / 2, 1. / 2 );
            cv::resize ( cur->depthImages[lvl], cur->depthImages[lvl + 1], cv::Size(), 0.5, 0.5, cv::INTER_NEAREST );
        }
        int scale = 1;
        int delta = 0;
        int ddepth = CV_32F;
        cv::Sobel( cur->grayImages[lvl], cur->gradImagesX[lvl], ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
        cv::Sobel( cur->grayImages[lvl], cur->gradImagesY[lvl], ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );

        //std::cout << "computing points" << std::endl;
        computeAllPoints( cur->grayImages[lvl], cur->depthImages[lvl],cur->gradImagesX[lvl],cur->gradImagesY[lvl], cur->kp_c[lvl], cur->p_c[lvl], cur->I_c[lvl], cur->K_c[lvl], cur->Ig_c[lvl]);//, cur->patch_c[lvl]);
    }
    //cur->depthImages.clear();  // not used further on.
    //return cur;
}

std::vector< ImageDataPtr > loadDataFromICLNUIM ( const std::string & filePath, const int numImagesToRead, std::vector<double> & times, const bool useModified = false )
{
   std::vector< ImageDataPtr > data;
   std::string fileNameIm = filePath + "/associations.txt";
   std::string fileNameGt = filePath + "/livingRoom2.gt.freiburg";

   std::ifstream imageFile ( fileNameIm, std::ifstream::in );
   std::ifstream grtruFile ( fileNameGt, std::ifstream::in );

   std::string folderString = useModified ? "/mod/" : "/";

   times.clear();
   int imagesRead = 0;
   for ( imagesRead = 0; imageFile.good() && grtruFile.good() && imagesRead < numImagesToRead ; ++imagesRead )
   {
      std::string depthFileName, colorFileName;
      int idc, idd, idg;
      double tsc, tx, ty, tz, qx, qy, qz, qw;
      imageFile >> idd >> depthFileName >> idc >> colorFileName;

      if ( idd == 0 )
          continue;
      grtruFile >> idg >> tx >> ty >> tz >> qx >> qy >> qz >> qw;

      if ( idc != idd || idc != idg )
      {
          std::cerr << "Error during reading... not correct anymore!" << std::endl;
          break;
      }

      if ( ! depthFileName.empty() )
      {
         Eigen::Affine3d pose_wc = Eigen::Affine3d::Identity();
         pose_wc.translation() << tx,ty,tz;
         pose_wc.linear() = Eigen::Quaterniond(qw,qx,qy,qz).toRotationMatrix();
         Eigen::Affine3d pose_cw = pose_wc.inverse();

         ImageDataPtr cur = createImageData ( filePath + folderString + colorFileName, filePath + "/" + depthFileName, pose_cw );

         data.push_back ( cur );
         times.push_back( tsc );
      }
   }
   std::cout << "read " << imagesRead << " images. (" << data.size() <<", " << times.size() << ")" << std::endl;
   return data;
}


std::vector< ImageDataPtr > loadDataFromRPG ( const std::string & filePath, const int numImagesToRead, std::vector<double> & times, const bool useModified = true )
{
   std::vector< ImageDataPtr > data;
   std::string fileNameDe = filePath + "/info/depthmaps.txt";
   std::string fileNameIm = filePath + "/info/images.txt";
   std::string fileNameGt = filePath + "/info/groundtruth.txt";

   std::ifstream depthFile ( fileNameDe, std::ifstream::in );
   std::ifstream imageFile ( fileNameIm, std::ifstream::in );
   std::ifstream grtruFile ( fileNameGt, std::ifstream::in );

   std::string folderString = useModified ? "/data/mod/" : "/data/";

   times.clear();
   int imagesRead = 0;
   for ( imagesRead = 0; depthFile.good() && imageFile.good() && grtruFile.good() && imagesRead < numImagesToRead ; ++imagesRead )
   {
      std::string depthFileName, colorFileName;
      int idc, idd, idg;
      double tsc, tx, ty, tz, qx, qy, qz, qw;
      imageFile >> idc >> tsc >> colorFileName;
      depthFile >> idd >> depthFileName;
      grtruFile >> idg >> tx >> ty >> tz >> qx >> qy >> qz >> qw;

      if ( idc != idd || idc != idg )
      {
          std::cerr << "Error during reading... not correct anymore!" << std::endl;
          break;
      }

      if ( ! depthFileName.empty() )
      {
         Eigen::Affine3d pose_wc = Eigen::Affine3d::Identity();
         pose_wc.translation() << tx,ty,tz;
         pose_wc.linear() = Eigen::Quaterniond(qw,qx,qy,qz).toRotationMatrix();
         Eigen::Affine3d pose_cw = pose_wc.inverse();

         ImageDataPtr cur = createImageData ( filePath + folderString + colorFileName, filePath + "/data/" + depthFileName, pose_cw );

         data.push_back ( cur );
         times.push_back( tsc );
      }
   }
   std::cout << "read " << imagesRead << " images. (" << data.size() <<", " << times.size() << ")" << std::endl;
   return data;
}

// ORB part...
void extractFeatures( ImageDataPtr imgPtr, std::vector<cv::KeyPoint> & kps, cv::Mat & descs )
{
    static auto orb = cv::ORB::create(1000); //cv::xfeatures2d::SIFT::create();//
    orb->setScaleFactor(2.);
    //orb->setNLevels(5);
    orb->detectAndCompute( imgPtr->grayImages[0], cv::noArray(), kps, descs);
}

void matchFeatures ( cv::Mat & refDescs, cv::Mat & curDescs, std::vector<cv::DMatch> & matches )
{
    static auto bf = cv::BFMatcher::create( cv::NORM_HAMMING, true ); // cv::NORM_L1, true );
    bf->match( curDescs, refDescs, matches ); // query, then train
}

void backProjectFeatures ( ImageDataPtr refImgPtr, const std::vector<cv::KeyPoint> & refKps, std::vector<cv::Point3f> & refMps )
{
    refMps.clear();
    refMps.resize( refKps.size(), cv::Point3f(0,0,std::numeric_limits<float>::signaling_NaN()) );
    const Eigen::Matrix3f Ki_ref = refImgPtr->K_c[0].inverse().cast<float>();
    int numBackProjected = 0;
    for(std::size_t i = 0; i < refKps.size(); ++i)
    {
        const auto & refKp = refKps[i];
        const int refKpx = std::round(refKp.pt.x);
        const int refKpy = std::round(refKp.pt.y);

        //if ( refKp.octave > 1 )
        //    continue;

        if( refKpx < 0 || refKpy < 0 || refKpx >= (int) refImgPtr->depthImages[0].cols || refKpy >= (int) refImgPtr->depthImages[0].rows )
            continue;
        const float depthValue = refImgPtr->depthImages[0].at<float>( refKpy, refKpx );
        if ( ! std::isfinite(depthValue) || depthValue < 0.01 )
            continue;

        Eigen::Vector3f pt = depthValue * Ki_ref * Eigen::Vector3f(refKpx,refKpy,1);
        refMps[i] = cv::Point3f( pt(0), pt(1), pt(2) );
        //std::cout << "pt: " << refMps[i] << std::endl;
        ++numBackProjected;
    }
    //std::cout << "backProjected:" << numBackProjected << std::endl;

//    Eigen::Matrix3Xd M ( 3, numBackProjected );
//    for ( int i = 0; i < refMps.size(); ++i )
//    {
//        M.col(i) << refMps[i].x, refMps[i].y, refMps[i].z;
//    }
//    std::cout << "M=[" << M.row(0) << std::endl << M.row(1) <<std::endl << M.row(2) <<"]"<< std::endl;
}

void create3dTo2dPairs ( const std::vector<cv::DMatch> & matches, const std::vector<cv::Point3f> & refMps, const std::vector<cv::KeyPoint> & curKps, std::vector<cv::Point3f> & selectedRefMps, std::vector<cv::Point2f> & selectedCurKps, std::vector<std::size_t> & matchIndexes )
{
    selectedCurKps.clear();
    selectedRefMps.clear();
    matchIndexes.clear();
    //std::vector<double> octave;
    for(std::size_t i = 0; i < matches.size(); ++i)
    {
        const auto & match = matches[i];
        const auto & refMp = refMps[match.trainIdx];

        if( !std::isfinite(refMp.z) || refMp.z < 0.01 )
            continue;

        cv::Point2f curKp = curKps[match.queryIdx].pt;

        selectedRefMps.push_back( refMp );
        selectedCurKps.push_back( curKp );
      //  octave.push_back( curKps[match.queryIdx].octave );
        matchIndexes.push_back( i );
    }

//    Eigen::Matrix3Xd M ( 3, selectedRefMps.size() );
//    for ( int i = 0; i < selectedRefMps.size(); ++i )
//    {
//        M.col(i) << selectedRefMps[i].x, selectedRefMps[i].y, selectedRefMps[i].z;
//    }
//    std::cout << "M=[" << M.row(0) << std::endl << M.row(1) <<std::endl << M.row(2) <<"]"<< std::endl;
//    Eigen::Matrix2Xd K ( 2, selectedCurKps.size() );
//    for ( int i = 0; i < selectedCurKps.size(); ++i )
//    {
//        K.col(i) << selectedCurKps[i].x, selectedCurKps[i].y;
//    }
//    std::cout << "K=[" << K.row(0) << std::endl << K.row(1) <<"]"<< std::endl;
//    Eigen::RowVectorXd S ( 1, selectedCurKps.size() );
//    for ( int i = 0; i < selectedCurKps.size(); ++i )
//    {
//        S(i) = octave[i];
//    }
//    std::cout << "S=[" << S <<"]"<< std::endl;
}


struct  AffineAutoDiffCostFunctor
{
  explicit AffineAutoDiffCostFunctor( const double & refColor, const double & newColor )
  :  m_refColor( refColor ), m_newColor( newColor ){ }

template<typename T>
bool operator() (const T* scaleA, const T* offsetB, T* residuals) const {
    residuals[0] = T(m_newColor) - (scaleA[0] * T(m_refColor) + offsetB[0]);
    return true;
}
static CostFunction * Create ( const double & refColor, const double & newColor )
{
  return new AutoDiffCostFunction<AffineAutoDiffCostFunctor,1,1,1>( new AffineAutoDiffCostFunctor( refColor, newColor ) );
}

private:
    const double m_refColor;
    const double m_newColor;
};

Eigen::Vector2f estimateAffine( std::vector<ImmaturePointPtr> & invDepthPts, ImageDataPtr curImgPtr, const  Mat33f & KRKi_cr, const Vec3f & Kt_cr )
{
    ceres::Problem problem;
    ceres::LossFunction * loss_function = new ceres::HuberLoss(1);
    double scaleA = 1;
    double offsetB = 0;
    for ( int i = 0; i < invDepthPts.size(); ++i )
    {
        ImmaturePointPtr & invDepthPt = invDepthPts[i];
        if ( ! invDepthPt )
            continue;
        if ( i % 100 != 0 )
            continue;

        Eigen::VectorXd refColors;
        Eigen::VectorXd gtColors = invDepthPt->getGTColor( curImgPtr, KRKi_cr, Kt_cr, refColors );

        for ( int i = 0; i < gtColors.rows(); ++i)
        {
            if ( !std::isfinite(refColors[i]) || ! std::isfinite(gtColors[i]) )
                continue;
            if ( refColors[i] <= 0 || refColors[i] >= 255 || gtColors[i] <= 0 || gtColors[i] >= 255  )
                continue;
            ceres::CostFunction * cost_function = AffineAutoDiffCostFunctor::Create( gtColors[i], refColors[i] );
            problem.AddResidualBlock( cost_function, loss_function, &scaleA, & offsetB );
        }
    }
    ceres::Solver::Options solver_options;
    //solver_options.linear_solver_type = ceres::DENSE_QR;//DENSE_SCHUR;//QR;
    solver_options.minimizer_progress_to_stdout = false;
    solver_options.max_num_iterations = 1000;
    solver_options.function_tolerance = 1e-6;
    solver_options.num_threads = 8;
    ceres::Solver::Summary summary;
    ceres::Solve( solver_options, & problem, & summary );
    // std::cout << summary.FullReport() << std::endl;
    std::cout << "scale= " << scaleA << " offset= "<< offsetB << std::endl;
    return Eigen::Vector2f ( scaleA, offsetB );
}

std::vector<ImmaturePointPtr> createInverseDepthPoints ( ImageDataPtr curImgPtr )
{
    const cv::Mat & img = curImgPtr->grayImages[0];
    const cv::Mat & depthImg = curImgPtr->depthImages[0];
    std::vector<ImmaturePointPtr> invDepthPts ( img.rows*img.cols, ImmaturePointPtr());
    int numValidPts = 0;
    for ( int row = 0; row < img.rows; ++row)
        for ( int col = 0; col < img.cols; ++col )
        {
            const float gtDepth = depthImg.empty() ? 0 : (1.0f / depthImg.at<float>(row, col));
            const int idx = row * img.cols + col;
            invDepthPts[idx].reset( new ImmaturePoint( col, row, curImgPtr, gtDepth ) );
            if ( invDepthPts[idx]->gradH.determinant() < 1000 )
            {
                invDepthPts[idx].reset();
            }
            else
            {
                ++numValidPts;
            }

        }
    std::cout << "created " << numValidPts << " of " << invDepthPts.size() << std::endl;



    return invDepthPts;
}

DepthFilterPtr createDepthFilter ( ImageDataPtr curImgPtr )
{
    DepthFilterPtr depthFilter ( new DepthFilter() );
    std::vector<ImmaturePointPtr> invDepthPts = createInverseDepthPoints ( curImgPtr );
    depthFilter->initializeSeeds ( invDepthPts );
    return depthFilter;
}

void saveDepthFilter( DepthFilterPtr depthFilter )
{
    if ( ! depthFilter )
    {
        std::cerr << "depthfilter does not exist." <<std::endl;
        return;
    }
    std::ofstream f ( "/media/alex/Data/Master/SHK/c_ws/src/renegade/results/inverse_depth_filter.txt" );
    auto seeds = depthFilter->getSeeds();
    for ( const Seed & seed : seeds )
    {
        const float idepth = seed.mu;
        const float denoised_idepth = seed.u;
        f << seed.ftr->u << " " << seed.ftr->v << " " << idepth << " " << seed.converged << " " << seed.is_outlier << " " << denoised_idepth << " " << seed.a << " " << seed.b << " " << seed.sigma2 << " " << seed.z_range << std::endl;
    }
    f.close();


    //save also a cv mat
    cv::Mat depth_mat(480,640,CV_32F);
    for ( const Seed & seed : seeds ){
        const float idepth = seed.mu;
        const float denoised_idepth = seed.u;

        //do we need to unproject it?

        depth_mat.at<float>( seed.ftr->v, seed.ftr->u)=idepth;
        // f << seed.ftr->u << " " << seed.ftr->v << " " << idepth << " " << seed.converged << " " << seed.is_outlier << " " << denoised_idepth << " " << seed.a << " " << seed.b << " " << seed.sigma2 << " " << seed.z_range << std::endl;
    }
    cv::Mat depth_mat_normalized;
    cv::normalize(depth_mat, depth_mat_normalized, 0, 255, cv::NORM_MINMAX, CV_32F);
    cv::imwrite("/media/alex/Data/Master/SHK/c_ws/src/renegade/results/inverse_depth_filter_mat.png",depth_mat_normalized);

}


#include <omp.h>
#include <unordered_map>
Vec2f updateInverseDepthTrace( std::vector<ImmaturePointPtr> & invDepthPts, ImageDataPtr refImgPtr, ImageDataPtr curImgPtr, const Eigen::Affine3d & pose_c2_c1 )
{
    const float setting_minTraceQuality = 3;
    int good = 0, oob = 0, outlier = 0, skipped = 0, badcondi = 0, uninit = 0, sameBestIdx = 0, closeBestIdx = 0, smallerBestIdx = 0, largerBestIdx = 0, goodTraceQuality = 0, goodNgf = 0, goodBCA = 0, viableGt = 0, viableProjectionGt = 0, inValidTrace = 0;
    int activatable = 0, activatableWithViableProjectionGT = 0, activatableWithViableGT = 0;
    const Mat33f KRKi_cr = (curImgPtr->K_c[0] * pose_c2_c1.linear() * refImgPtr->K_c[0].inverse()).cast<float>();
    const Vec3f Kt_cr = (curImgPtr->K_c[0] * pose_c2_c1.translation()).cast<float>();
    //const Vec2f affine_cr  ( 1, 0 );
    const Vec2f affine_cr = estimateAffine( invDepthPts, curImgPtr, KRKi_cr, Kt_cr );
    return affine_cr;

    int existingPointNum = 0;
    std::vector<ImmaturePointPtr> newPoints;
    #pragma omp parallel for
    for ( int i = 0; i < invDepthPts.size(); ++i )
    {
        ImmaturePointPtr & invDepthPt = invDepthPts[i];
        if ( !invDepthPt ) {
            // std::cout << "why is there no inv depth here?" << '\n';
            continue;
        }


        ImmaturePointPtr newPt = invDepthPt->traceOn( curImgPtr, KRKi_cr, Kt_cr, affine_cr );

//        #pragma omp atomic
        #pragma omp critical
        {
            ++existingPointNum;
            if ( newPt ){
                newPoints.push_back( newPt );
            }else{
                // std::cout << "why is new poinr null" << '\n';
            }
        }
    }

    //THE FLOW NEVER REACHES HERE! WHY?
    std::cout << "existingPointNum is " << existingPointNum << '\n';
    std::cout << "nre new points is " << newPoints.size()  << '\n';
    // return affine_cr;


    std::vector<ImmaturePointPtr> ptsToDelete;
    std::unordered_map<int,std::vector<ImmaturePointPtr> > samePtsMap;
    for ( ImmaturePointPtr & invDepthPt : invDepthPts )
    {
        if ( !invDepthPt )
            continue;
        const int idx = invDepthPt->v * refImgPtr->grayImages[0].cols + invDepthPt->u;
        samePtsMap[idx].push_back( invDepthPt );
    }
    int numDuplicated = 0, numWithMultipleValids = 0, numAtLeastContained = 0, numShown = 0;
    for ( auto & elem : samePtsMap )
    {
        if ( elem.second.size() > 1 )
        {
            ++numDuplicated;
            std::vector<ImmaturePointPtr> curValidPts;
            for ( ImmaturePointPtr & invDepthPt : elem.second )
            {
                // TODO check if one is better and the other can then be deleted.
                if ( !std::isfinite(invDepthPt->idepthNgf_est) ||!std::isfinite(invDepthPt->idepth_min) || !std::isfinite(invDepthPt->idepth_max) || invDepthPt->idepth_max < 0 )
                {
                    ptsToDelete.push_back( invDepthPt );
                    continue;
                }
                curValidPts.push_back(invDepthPt);
            }
            if ( curValidPts.size() > 1 )
            {
                ++numWithMultipleValids;
                bool atLeastOneContained = false;
                int numMergeIters = 0;
                do
                {
                    ++numMergeIters;
                    atLeastOneContained = false;
                    ImmaturePointPtr refMergePt ( nullptr );
                    std::vector<ImmaturePointPtr> toMerge;
                    for ( ImmaturePointPtr & refValidPt : curValidPts )
                    {
                        for ( ImmaturePointPtr & curValidPt : curValidPts )
                        {
                            if ( curValidPt == refValidPt )
                                continue;
                            // check if depth of points overlap -> no reason to create more than one, rather use the maximum
                            bool isContained = refValidPt->isContained(curValidPt);

                            if ( isContained )
                            {
                                atLeastOneContained = true;
                                toMerge.push_back(curValidPt);
                                refMergePt = refValidPt;
                            }
                        }
                        if ( atLeastOneContained ) break;
                    }
                    if ( !toMerge.empty() )
                    {
                        for ( ImmaturePointPtr & curValidPt : toMerge )
                        {
                            refMergePt->idepth_min = std::min<float>(refMergePt->idepth_min, curValidPt->idepth_min );
                            refMergePt->idepth_max = std::max<float>(refMergePt->idepth_max, curValidPt->idepth_max );
                            std::cout << "storing min max "<< refMergePt->idepth_min << " " << refMergePt->idepth_max << '\n';
                            const auto & it = std::find(curValidPts.begin(),curValidPts.end(), curValidPt);
                            if ( it != curValidPts.end())
                            {
                                ptsToDelete.push_back( curValidPt );
                                curValidPts.erase ( it );
                                curValidPt->lastTraceStatus = ImmaturePointStatus::IPS_DELETED;
                            }
                        }
                    }
                }while ( atLeastOneContained );
                if ( numMergeIters > 1 )
                    ++numAtLeastContained;
                else
                {
                    // more interesting case: how to discern between those that are ok and those that are not?
                    ++numShown;
                    if ( numShown < 0 )
                    {
                        for ( ImmaturePointPtr & invDepthPt : curValidPts )
                        {
                            if ( invDepthPt->idepth_GT > 0 && invDepthPt->idepthNgf_est > 0 && std::abs<float>(1./invDepthPt->idepthNgf_est - 1./invDepthPt->idepth_GT ) < 0.2 )
                                std::cout << "u="<<invDepthPt->u << " v=" << invDepthPt->v << " within gt (" << invDepthPt->idepth_GT << ") range. min: " << invDepthPt->idepth_min << " max: " << invDepthPt->idepth_max << " lastState: " << invDepthPt->lastTraceStatus <<std::endl;
                            else
                                std::cout << "u="<<invDepthPt->u << " v=" << invDepthPt->v << " not in gt (" << invDepthPt->idepth_GT << ") range. min: " << invDepthPt->idepth_min << " max: " << invDepthPt->idepth_max << " lastState: " << invDepthPt->lastTraceStatus <<std::endl;
                        }
                    }

                    // TODO: add those with outlier to delete.
                    for ( ImmaturePointPtr & invDepthPt : curValidPts )
                    {
                        if ( invDepthPt && (invDepthPt->lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER || invDepthPt->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION ) )
                        {
                            invDepthPt->lastTraceStatus = ImmaturePointStatus::IPS_DELETED;
                            ptsToDelete.push_back(invDepthPt);
                        }
                    }
                }
            }
        }
        bool ngfOneGood = false;
        bool bcaOneGood = false;
        bool oneGtWithinRange = false;
        bool oneGoodProjection = false;
        bool lastTraceInvalid = false;
        for ( ImmaturePointPtr & invDepthPt : elem.second )
        {
            if ( invDepthPt->idepth_GT > 0 )
            {
                if ( invDepthPt->idepthNgf_est > 0 && std::abs<float>(1./invDepthPt->idepthNgf_est - 1./invDepthPt->idepth_GT ) < 0.2 )
                    ngfOneGood = true;
                if ( invDepthPt->idepthBCA_est > 0 && std::abs<float>(1./invDepthPt->idepthBCA_est - 1./invDepthPt->idepth_GT ) < 0.2 )
                    bcaOneGood = true;
                if ( invDepthPt->idepth_min <= invDepthPt->idepth_GT && invDepthPt->idepth_GT <= invDepthPt->idepth_max )
                    oneGtWithinRange = true;
                if ( (invDepthPt->kp_GT-invDepthPt->lastValidUV).norm() < 2 ) //&& (invDepthPt->lastTraceUV-Vec2f(-1,-1)).norm() > 1 )
                    oneGoodProjection = true;
                if ( (invDepthPt->lastTraceUV-Vec2f(-1,-1)).norm() < 1 )
                    lastTraceInvalid = true;
            }
        }
        if ( ngfOneGood ) ++goodNgf;
        if ( bcaOneGood ) ++goodBCA;
        if ( oneGtWithinRange ) ++viableGt;
        if ( oneGoodProjection ) ++viableProjectionGt;
        if ( lastTraceInvalid ) ++inValidTrace;

        for ( ImmaturePointPtr & invDepthPt : elem.second )
        {
            bool canActivate = (invDepthPt->lastTraceStatus == IPS_GOOD
                                || invDepthPt->lastTraceStatus == IPS_SKIPPED
                                || invDepthPt->lastTraceStatus == IPS_BADCONDITION
                                || invDepthPt->lastTraceStatus == IPS_OOB )
                    && invDepthPt->lastTracePixelInterval < 8*2
                    && invDepthPt->quality > 1.05//setting_minTraceQuality
                    && (invDepthPt->idepth_max+invDepthPt->idepth_min) > 0;
            if ( canActivate )
            {
                ++activatable;
                if ( (invDepthPt->kp_GT-invDepthPt->lastValidUV).norm() < 2 ) //&& (invDepthPt->lastTraceUV-Vec2f(-1,-1)).norm() > 1 )
                    ++activatableWithViableProjectionGT;
                if ( invDepthPt->idepth_min <= invDepthPt->idepth_GT && invDepthPt->idepth_GT <= invDepthPt->idepth_max )
                    ++activatableWithViableGT;
            }
        }
    }
    samePtsMap.clear();

    std::cout << "oldSize: " << existingPointNum << " newPts: " << newPoints.size()<< " numDupl: "<< numDuplicated << " numMult: "<< numWithMultipleValids << " minCont: " << numAtLeastContained<< " ptsToDelete: " << ptsToDelete.size() << std::endl;
    for ( ImmaturePointPtr & pt : ptsToDelete )
    {
        pt->lastTraceStatus = ImmaturePointStatus::IPS_DELETED;
    }
    for ( ImmaturePointPtr & invDepthPt : invDepthPts )
    {
        if ( !invDepthPt )
            continue;
        if ( invDepthPt->lastTraceStatus == ImmaturePointStatus::IPS_DELETED )
        {
            invDepthPt.reset();
            continue;
        }
    }
    invDepthPts.insert(invDepthPts.end(),newPoints.begin(),newPoints.end());

    for ( ImmaturePointPtr & invDepthPt : invDepthPts )
    {
        if ( !invDepthPt )
            continue;
        const ImmaturePointStatus st = invDepthPt->lastTraceStatus;
        if ( ImmaturePointStatus::IPS_GOOD == st )
            ++good;
        if ( ImmaturePointStatus::IPS_OOB == st)
            ++oob;
        if ( ImmaturePointStatus::IPS_OUTLIER == st )
            ++outlier;
        if ( ImmaturePointStatus::IPS_SKIPPED == st )
            ++skipped;
        if ( ImmaturePointStatus::IPS_BADCONDITION == st )
            ++badcondi;
        if ( ImmaturePointStatus::IPS_UNINITIALIZED == st )
            ++uninit;

        if ( invDepthPt->quality > setting_minTraceQuality )
            ++goodTraceQuality;

        if ( invDepthPt->bestIdxDSO >= 0 && invDepthPt->bestIdxNgf >= 0)
        {
            sameBestIdx+= (invDepthPt->bestIdxDSO==invDepthPt->bestIdxNgf);
            closeBestIdx += abs(invDepthPt->bestIdxDSO-invDepthPt->bestIdxNgf) < 3;
            smallerBestIdx += abs(invDepthPt->bestIdxDSO-invDepthPt->bestIdxNgf) < 3 && invDepthPt->bestIdxDSO > invDepthPt->bestIdxNgf;
            largerBestIdx += abs(invDepthPt->bestIdxDSO-invDepthPt->bestIdxNgf) < 3 && invDepthPt->bestIdxDSO < invDepthPt->bestIdxNgf;
        }
    }
    std::cout << "good: " << good << " oob: " << oob <<  " outlier: " << outlier << " skipped: " << skipped << " bad: " << badcondi << " uninit: " << uninit << " translation=" << pose_c2_c1.translation().norm() << " same: " << sameBestIdx << " close: "<< closeBestIdx << " smaller: " << smallerBestIdx << " larger: " << largerBestIdx <<" goodQuality: " << goodTraceQuality << " bca: "<< goodBCA << " ngf: " << goodNgf << " gtInRange: " << viableGt << " closeGtProj: " << viableProjectionGt << " invalidTrace: "<<inValidTrace<< std::endl;
    std::cout <<"activatable: " << activatable << " withinGTRange: "<< activatableWithViableGT << " Projection: " << activatableWithViableProjectionGT << std::endl;
    return affine_cr;
}

void calculateInverseDepth ( std::vector<PointPtr> & pt, std::vector<ImmaturePointPtr> & invDepthPts, ImageDataPtr refImgPtr, std::vector<ImageDataPtr> & curImages )
{
    std::chrono::high_resolution_clock::time_point t_start = std::chrono::high_resolution_clock::now();
    const float setting_minTraceQuality = 3;
    int goodTraceInterval = 0, goodTraceQuality = 0;

    pt.clear();
    pt.reserve(invDepthPts.size());
    for ( ImmaturePointPtr & invDepthPt : invDepthPts )
    {
        if ( !invDepthPt)
            continue;
        // delete points that have never been traced successfully, or that are outlier on the last trace.
        if(!std::isfinite(invDepthPt->idepth_max) || invDepthPt->lastTraceStatus == IPS_OUTLIER)
            continue;

        // can activate only if this is true.
        bool canActivate = (invDepthPt->lastTraceStatus == IPS_GOOD
                            || invDepthPt->lastTraceStatus == IPS_SKIPPED
                            || invDepthPt->lastTraceStatus == IPS_BADCONDITION
                            || invDepthPt->lastTraceStatus == IPS_OOB );
//                && invDepthPt->lastTracePixelInterval < 8
//                && invDepthPt->quality > setting_minTraceQuality
//                && (invDepthPt->idepth_max+invDepthPt->idepth_min) > 0;

        if ( invDepthPt->lastTracePixelInterval < 8 )
            ++goodTraceInterval;
        if ( invDepthPt->quality > setting_minTraceQuality )
            ++goodTraceQuality;

        // if I cannot activate the point, skip it. Maybe also delete it.
//        if(!canActivate)
//            continue;

        PointPtr newPoint = invDepthPt->optimize( curImages );
        if ( newPoint )
            pt.push_back(newPoint);
    }
    std::cout << "inverse Depth computation took: " << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_start).count() << " goodTrace: "<< goodTraceInterval << " goodQuali:"<< goodTraceQuality << std::endl;
}

void saveDepthMapImmature( std::vector<ImmaturePointPtr> & invDepthPts )
{
    const float setting_minTraceQuality = 3;
    std::ofstream f ( "/media/alex/Data/Master/SHK/c_ws/src/renegade/results/inverse_depth_minmax_3.txt" );
    int nr_points_considered=0;
    int nr_points_stored=0;
    for ( const ImmaturePointPtr & invDepthPt : invDepthPts )
    {
        if ( ! invDepthPt ){
            // std::cout << "saveDepthMapImmature no inv depth point" << '\n';
            continue;

        }
        // can activate only if this is true.
        bool canActivate = (invDepthPt->lastTraceStatus == IPS_GOOD
                            || invDepthPt->lastTraceStatus == IPS_SKIPPED
                            || invDepthPt->lastTraceStatus == IPS_BADCONDITION
                            || invDepthPt->lastTraceStatus == IPS_OOB )
                && invDepthPt->lastTracePixelInterval < 8
                && invDepthPt->quality > setting_minTraceQuality
                && (invDepthPt->idepth_max+invDepthPt->idepth_min) > 0;

        // if I cannot activate the point, skip it. Maybe also delete it.
        //if(!canActivate)
        //    continue;

        nr_points_considered++;
        const float & id_min = invDepthPt->idepth_min;
        const float & id_max = invDepthPt->idepth_max;
        const float invDepth = (id_min + id_max)/2;
        // std::cout << "inv_depth_avg is " << invDepth << '\n';
        if ( !std::isfinite(id_min) ) {
            // std::cout << "min is " << id_min << '\n';
            continue;
        }
        if ( !std::isfinite(id_max) || id_max < 0 ){
            // std::cout << "max is " << id_max << '\n';
            continue;

        }
        nr_points_stored++;
        f << invDepthPt->u << " " << invDepthPt->v << " " << id_min << " " << id_max << " " << invDepthPt->idepth_GT << " " << invDepthPt->quality << std::endl;
    }
    f.close();

    std::cout << "save depth map inmature considered nr of points " << nr_points_considered << '\n';
    std::cout << "save depth map inmature wrote nr of points " << nr_points_stored << '\n';
}

void saveDepthMap( std::vector<PointPtr> & invDepthPts )
{
    int numValidPts = 0;
    std::ofstream f ( "/media/alex/Data/Master/SHK/c_ws/src/renegade/results/inverse_depth_3.txt" );
    for ( const PointPtr & invDepthPt : invDepthPts )
    {
        if ( ! invDepthPt )
            continue;
        if ( invDepthPt->status != Point::PtStatus::ACTIVE )
            continue;
        ++numValidPts;
        if ( !std::isfinite( invDepthPt->idepth ) )
             continue;
        if ( invDepthPt->idepth < 0 )
            continue;
//        const float & id_min = invDepthPt->idepth_min;
//        const float & id_max = invDepthPt->idepth_max;
//        if ( ! std::isfinite(id_min) || id_min < 0 )
//            continue;
//        if ( ! std::isfinite(id_max) || id_max < 0 )
//            continue;
//        const float invDepth = (id_min + id_max)/2;
        f << invDepthPt->u << " " << invDepthPt->v << " " << invDepthPt->idepth << std::endl;
    }
    f.close();
    //exit(EXIT_SUCCESS);
}

void filterMatchesByRelativePose ( ImageDataPtr curImgPtr, const Eigen::Affine3d & pose_c2_c1, const std::vector<cv::Point3f> & refMps, const std::vector<cv::Point2f> & curKps, std::vector<int> & inliers )
{
    const Eigen::Matrix3d & K_c = curImgPtr->K_c[0];
    cv::Mat_<double> intrinsics(3,3);
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            intrinsics(i,j) = K_c(i,j);

    cv::Mat_<double> rvec(3,1); // input rotation vector
    cv::Mat_<double> tvec(3,1); // input translation vector

    cv::Mat_<double> R(3,3);
    for( int i = 0; i < 3; ++i)
    {
        tvec(i) = pose_c2_c1.translation()(i);
        for(int j = 0; j < 3; ++j)
            R(i,j) = pose_c2_c1.linear()(i,j);
    }
    cv::Rodrigues(R, rvec);

    //std::cout << "rvec:" << rvec << " " << tvec << " K_c="<<intrinsics<< " R "<< R << std::endl;
    // inliers are all with a good reprojection error.
    cv::solvePnPRansac( refMps, curKps, intrinsics, cv::noArray(),
            rvec, tvec, true,
            100,
            1.0,
            0.99,
            inliers
    );
    //std::cout << "inliers=" << int(inliers.size()) << std::endl;
    //std::cout << "rvec:" << rvec << " " << tvec << std::endl;
    //exit(-1);
}

void extractFilteredMatches ( const std::vector<cv::DMatch> & matches, const std::vector<std::size_t> & matchIndexes, const std::vector<int> & inliers, std::vector<cv::DMatch> & inlier_matches )
{
    for(auto& idx : inliers)
        inlier_matches.push_back(matches[matchIndexes[idx]]);
}

void showMatches ( ImageDataPtr refImgPtr, ImageDataPtr curImgPtr, const std::vector<cv::KeyPoint> & refKps, const std::vector<cv::KeyPoint> & curKps, const std::vector<cv::DMatch> & inlier_matches  )
{
    cv::Mat vis;
    cv::drawMatches( curImgPtr->grayImages[0], curKps, refImgPtr->grayImages[0], refKps, inlier_matches, vis);

    cv::imshow("matches", vis);
    cv::waitKey(0);

}

struct KeyPointData
{
    std::vector<cv::KeyPoint> kps;
    cv::Mat descs;
    std::vector<std::pair<int,int> > assoc; // associated reference index and the kpIdx there.
    //std::vector<double> intensities;
    std::vector<std::vector<double> > intensities;
    std::vector<std::vector<double> > weights;
    std::vector<std::vector<double> > radius;
    std::vector<std::vector<double> > x;
    std::vector<std::vector<double> > y;
    std::vector<long> ids;
};

void extractIntensities ( ImageDataPtr imgPtr, KeyPointData & kpd )
{
    int maxLvl = imgPtr->grayImages.size()-1;
    std::vector<cv::Mat> imgs = imgPtr->grayImages;
    std::vector<std::shared_ptr<Grid2D<uchar,1> > > grids(maxLvl+1);
    std::vector<std::shared_ptr<BiCubicInterpolator<Grid2D<uchar,1> > > > curImages(maxLvl+1);
    for ( int lvl = 0; lvl < maxLvl; ++lvl)
    {
        double invScaleFactor = 1. / std::pow(2.,lvl);
        //cv::Mat img = imgPtr->grayImages[lvl];
        std::cout << invScaleFactor <<" " << imgs[lvl].empty()<< " rxc=" << imgs[lvl].rows << "x"<<imgs[lvl].cols << " kps:" << kpd.kps.size() << std::endl;

        //Grid2D<uchar,1> imgGrid( img.ptr<uchar>(), 0, img.rows, 0, img.cols );
        grids[lvl].reset( new Grid2D<uchar,1>( imgs[lvl].ptr<uchar>(), 0, imgs[lvl].rows, 0, imgs[lvl].cols ) );
        //BiCubicInterpolator< Grid2D<uchar,1> > curImage( imgGrid );
        curImages[lvl].reset ( new BiCubicInterpolator< Grid2D<uchar,1> >( *grids[lvl] ) );
    }

    static long newId = 1;
    kpd.intensities.resize(kpd.kps.size());
    kpd.weights.resize(kpd.kps.size());
    kpd.radius.resize(kpd.kps.size());
    kpd.x.resize(kpd.kps.size());
    kpd.y.resize(kpd.kps.size());
    kpd.ids.resize(kpd.kps.size(),0);
    //std::cout << "extracting intensity for " << int(kpd.kps.size())<< " kps." << std::endl;

    Eigen::Matrix2Xd offset(2,patchSize2);
    for ( int j = -patchSizeHalf, idx = 0; j <= patchSizeHalf; ++j)
        for ( int i = -patchSizeHalf; i <= patchSizeHalf; ++i, ++idx)
            offset.col(idx) << i, j;
    //offset.row(0) << -1,0,1,-1,0,1,-1,0,1;
    //offset.row(1) << -1,-1,-1,0,0,0,1,1,1;

    for ( int idx = 0; idx < kpd.kps.size(); ++idx)
    {
        const cv::KeyPoint & kp = kpd.kps[idx];
        const int curLvl = std::min(kp.octave,maxLvl);
        double invScaleFactor = 1. / std::pow(2.,curLvl);
        std::vector<double> intensities ( patchSize2, 0. );
        std::vector<double> extraWeights ( patchSize2, 0. );
        std::vector<double> radi ( patchSize2, 0. );
        std::vector<double> xs ( patchSize2, 0. );
        std::vector<double> ys ( patchSize2, 0. );

        const Eigen::Vector2d center ( imgs[curLvl].cols/2,imgs[curLvl].rows/2 );
        const Eigen::Vector2d maxPt ( imgs[curLvl].cols-1, imgs[curLvl].rows-1 );
        const double maxRadi = (maxPt-center).norm();

        Eigen::Matrix2d R;

        const double angle = kp.angle * M_PI / 180.;
        R.row(0) << std::cos(angle),-std::sin(angle);
        R.row(1) << std::sin(angle), std::cos(angle);

        Eigen::Vector2d kpPt ( kp.pt.x * invScaleFactor, kp.pt.y * invScaleFactor);
        for ( int i = 0; i < patchSize2; ++i)
        {
            Eigen::Vector2d curkp = kpPt + R*offset.col(i);
            //double kpy = kp.pt.y * invScaleFactor;
            //double kpx = kp.pt.x * invScaleFactor;
            double curIntensity, curIntensityDr, curIntensityDc;
            curImages[curLvl]->Evaluate( curkp(1), curkp(0), &curIntensity, &curIntensityDr, &curIntensityDc );

            if ( curIntensity < 1 || curIntensity > 254 )
                continue;

            const double eta = 1;
            const double extraWeight = eta / ( eta + Eigen::Vector2d(curIntensityDc,curIntensityDr).squaredNorm() );
            //kpd.intensities[idx] = curIntensity / 255.;
            //kpd.weights[idx] = extraWeight;
            intensities[i] = curIntensity / 255.;
            extraWeights[i] = extraWeight;
            radi[i] = (curkp-center).norm() / maxRadi;
            xs[i] = std::min(1.,std::max(curkp(0) / maxPt(0),0.));
            ys[i] = std::min(1.,std::max(curkp(1) / maxPt(1),0.));
        }
        kpd.intensities[idx] = intensities;
        kpd.weights[idx] = extraWeights;
        kpd.radius[idx] = radi;
        kpd.x[idx] = xs;
        kpd.y[idx] = ys;
        kpd.ids[idx] = newId;
        ++newId;
    }
    //std::cout << "all extracted, new maxId = " << newId << std::endl;
}

void extractAndTrackFeatures ( const std::vector<ImageDataPtr> & images, ParamsCfg & cfg, std::vector<KeyPointData> & reducedKeyPointData )
{
    int maxDx = 55;
    std::deque<KeyPointData> kpdDeque;
    std::deque<ImageDataPtr> imgDeque;
    for ( int idx = 0; idx <= maxDx && idx < images.size(); ++idx )
    {
        ImageDataPtr imgPtr( new ImageData (*images[idx]) );
        loadImageData ( imgPtr, imgPtr->colorFile, imgPtr->depthFile, cfg );
        imgDeque.push_back( imgPtr );

        kpdDeque.push_back(KeyPointData());
        KeyPointData & kpd = kpdDeque.back();
        extractFeatures( imgPtr, kpd.kps, kpd.descs);
        kpd.assoc.resize(kpd.kps.size(),std::pair<int,int>(-1,-1)); //set unassociated
    }

    //std::vector<KeyPointData> reducedKeyPointData; // only matched data.
    reducedKeyPointData.clear();
    reducedKeyPointData.resize(images.size());
    for ( int idx = 0; idx < images.size(); ++idx)
    {
        //std::cout << "starting round." << std::endl;
        // backproject current stuff
        //ImagePtr refImgPtr = images[idx];
        ImageDataPtr refImgPtr = imgDeque[0];
        KeyPointData & refKpd = kpdDeque[0];
        std::vector<cv::Point3f> refMps;
        backProjectFeatures ( refImgPtr, refKpd.kps, refMps );

        std::vector<ImmaturePointPtr> invDepthPts = createInverseDepthPoints ( refImgPtr );
        DepthFilterPtr depthFilter = createDepthFilter( refImgPtr );

        std::chrono::duration<double> inverseDepthTime ( 0 ), inverseSeedsDepthTime ( 0 );
        Eigen::Affine3d old_pose_c2_c1 = Eigen::Affine3d::Identity();
        //std::cout << "got 3d points for refFrame." << std::endl;
        std::vector<ImageDataPtr> curImages;
        bool notYet = false; int notYetCnter = 0;
        for ( int dx = 1; dx <= maxDx && dx < kpdDeque.size(); ++dx )
        {
            ImageDataPtr curImgPtr = imgDeque[dx];
            KeyPointData & curKpd = kpdDeque[dx];

            // match features
            std::vector<cv::DMatch> matches;
            matchFeatures( refKpd.descs, curKpd.descs, matches );
            //std::cout << "got matches: " << int(matches.size()) << std::endl;

            // create pairs
            //std::vector<std::size_t> matchIndexes;
            //std::vector<cv::Point3f> selectedRefMps;
            //std::vector<cv::Point2f> selectedCurKps;
            //create3dTo2dPairs ( matches, refMps, curKpd.kps, selectedRefMps, selectedCurKps, matchIndexes );

            //std::cout << "got pairs: " << int(matchIndexes.size()) << std::endl;

            // filter by relative pose
            const Eigen::Affine3d pose_c2_c1 = curImgPtr->m_pose_cw * refImgPtr->m_pose_cw.inverse();
            //std::cout << "pose_c2c1=[" << pose_c2_c1.matrix() <<"]"<< std::endl;

            //std::vector<int> inliers;
            //filterMatchesByRelativePose ( curImgPtr, pose_c2_c1, selectedRefMps, selectedCurKps, inliers );

//            curImgPtr->m_pose_cr = pose_c2_c1;
//            if ( std::abs( pose_c2_c1.translation().norm() - old_pose_c2_c1.translation().norm() ) > 0.02 )
//            {
//                old_pose_c2_c1 = pose_c2_c1;
//            }
//            else
//                continue;


            // TODO: update inverse depth:
            std::cout << "updating inverse depth." << std::endl;
            std::chrono::high_resolution_clock::time_point t_start = std::chrono::high_resolution_clock::now();
            const Vec2f affine_cr = updateInverseDepthTrace( invDepthPts, refImgPtr, curImgPtr, pose_c2_c1 );
            std::chrono::duration<double> lastIterTrace ( std::chrono::high_resolution_clock::now() - t_start );
            inverseDepthTime += lastIterTrace;

            std::chrono::high_resolution_clock::time_point s_start = std::chrono::high_resolution_clock::now();
            depthFilter->updateSeeds( refImgPtr, curImgPtr, pose_c2_c1, affine_cr );
            std::chrono::duration<double> lastIterSeeds ( std::chrono::high_resolution_clock::now() - s_start );
            inverseSeedsDepthTime += lastIterSeeds;

            std::cout << "depth updated, took (ms): "<<
                std::chrono::duration_cast<std::chrono::milliseconds>(lastIterTrace).count() << " and " <<
                std::chrono::duration_cast<std::chrono::milliseconds>(lastIterSeeds).count() << std::endl;
            curImages.push_back( curImgPtr );

//            if ( notYetCnter % 5 == 4 )
//            {
//                depthFilter->denoise( refImgPtr, 5.f, 0.1f, 1, true );
//            }

            //if ( notYetCnter == 10 )

            //if ( std::abs( pose_c2_c1.translation().norm() - old_pose_c2_c1.translation().norm() ) > 0.1 )
            if ( dx > 50 ) //|| depthFilter->getConvergencePercentage() > 0.3 )
            //if ( notYetCnter > 9 )
            {
                saveDepthMapImmature ( invDepthPts );
                std::vector<PointPtr> inverseDepthPoints;
                calculateInverseDepth ( inverseDepthPoints, invDepthPts, refImgPtr, curImages );
                saveDepthMap( inverseDepthPoints );

                depthFilter->denoise( refImgPtr, 5.f, 0.5f, 200 );
                saveDepthFilter( depthFilter );
                exit ( - 1 );
            }
            //notYet = true;
            ++notYetCnter;
        }
        saveDepthMapImmature ( invDepthPts );
        std::cout << "images: "<< curImages.size() << " tracing took: " << inverseDepthTime.count() << std::endl;
        std::vector<PointPtr> inverseDepthPoints;
        calculateInverseDepth ( inverseDepthPoints, invDepthPts, refImgPtr, curImages );
        saveDepthMap( inverseDepthPoints );

        // load data for next round
        if ( idx + maxDx +1 < images.size() )
        {
            // load next data
            ImageDataPtr imgPtr( new ImageData (*images[idx+maxDx+1]) );
            loadImageData ( imgPtr, imgPtr->colorFile, imgPtr->depthFile, cfg );
            imgDeque.push_back( imgPtr );

            kpdDeque.push_back(KeyPointData());
            KeyPointData & kpd = kpdDeque.back();
            extractFeatures( imgPtr, kpd.kps, kpd.descs);
            kpd.assoc.resize(kpd.kps.size(),std::pair<int,int>(-1,-1)); //set unassociated
        }

        std::cout << "removing old data." << std::endl;
        // remove first, which is the now old data
        kpdDeque.pop_front();
        imgDeque.pop_front();
        std::cout << "continue iter."<< std::endl;
    }
    std::cout << "done extraction." << std::endl;
}

void optimizeForInverseDepth( const std::vector< ImageDataPtr > & images, ParamsCfg & cfg )
{
    if ( images.empty() )
    {
        std::cerr << "no images." << std::endl;
        return;
    }

    std::cout << "starting feature extraction." << std::endl;
    std::vector<KeyPointData> kpds;
    kpds.reserve ( images.size() );
    extractAndTrackFeatures( images, cfg, kpds );
    std::cout << "done." << std::endl;
}

ParamsCfg setParams( const bool useRPG = true )
{
    ParamsCfg c;
    c.useRPG = useRPG;
    if ( useRPG )
    {
        // RPG urban city dataset
        c.fx = 329.115520046;
        c.fy = 329.115520046;
        c.cx = 320.;
        c.cy = 240.;
    }
    else
    {
        // ICL NUIM dataset
        c.fx = 481.2;
        c.fy = -480.; //-
        c.cx = 319.5;
        c.cy = 239.5;
    }
    c.k1 = 0.;
    c.k2 = 0.;
    c.p1 = 0.;
    c.p2 = 0.;
    c.k3 = 0.;
    c.undistort = false;
    return c;
}

// int main(int argc, char** argv) {
//
//     google::InitGoogleLogging(argv[0]);
//
//     bool useRPG = false;
//     bool useModified = false;
//     int numImages = 60;
//
//     ParamsCfg cfg = setParams( useRPG );
//
//     std::vector<double> times;
//     const std::string filePath = useRPG ? "/home/jquenzel/bags/urban/rpg/" : "/media/alex/Data/Master/Thesis/data/ICL_NUIM/living_room_traj2_frei_png/";
//     const std::vector< ImageDataPtr > images = useRPG ?
//                 loadDataFromRPG ( filePath, numImages, times, useModified ) :
//                 loadDataFromICLNUIM( filePath, numImages, times, useModified );
//
//     if ( images.empty() )
//     {
//         std::cerr << "no images found!" << std::endl;
//         return -1;
//     }
//
//     optimizeForInverseDepth( images, cfg );
//
//     return 0;
// }
