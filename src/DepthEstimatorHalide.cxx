#include "stereo_depth_gl/DepthEstimatorHalide.h"

//c++
#include <cmath>
#include <stdlib.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>

//My stuff
#include "stereo_depth_gl/Profiler.h"
#include "stereo_depth_gl/MiscUtils.h"

//Libigl

//cv
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/eigen.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"

//loguru
#include <loguru.hpp>

//omp
#include <omp.h>

using namespace Halide;


DepthEstimatorHalide::DepthEstimatorHalide(){

    init_halide();
}

//needed so that forward declarations work
DepthEstimatorHalide::~DepthEstimatorHalide(){
}

void DepthEstimatorHalide::init_halide(){

}


void DepthEstimatorHalide::compute_depth(const Frame& frame_left, const Frame& frame_right){

    std::cout << "compute depth" << '\n';

    int max_disparity=100;

    // if(is_first_frame){
    //     is_first_frame=false;
    //     first_frame=frame_left;
    // }

    //for some reason it doesnt work for the eth one the rectification
    // cv::Mat gray_left=frame_left.gray;
    // cv::Mat gray_right=undistort_rectify_image(frame_right.gray, frame_left, frame_right);
    // gray_left/=255;
    // gray_right/=255;
    // debug_img_left=gray_left;
    // debug_img_right=gray_right;


    //tsukuba images
    cv::Mat gray_left=frame_left.gray/255;
    cv::Mat gray_right=frame_right.gray/255;

    std::cout << "mat gray type is " << type2string(gray_left.type()) << '\n';


    //wrap the image in a halide buffer
    Buffer<float> buf_left((float*)gray_left.data, gray_left.cols, gray_left.rows);
    Buffer<float> buf_right((float*)gray_right.data, gray_right.cols, gray_right.rows);

    //ALGORITHM

    //clamp the x and y so as to not go out of bounds
    Var x("x"),y("y"), d("d");
    Func clamped("clamped");
    Expr clamped_x_d = clamp(x+d, 0, buf_right.width()-1);


    //cost volume
    Func cost_vol("cost_vol");
    cost_vol(x,y,d)=999999.0;
    cost_vol(x,y,d) = Halide::abs(buf_left(x,y) - buf_right(clamped_x_d,y));

    //filter the cost volume
    //TODO


    //argmax the cost volume
    Func argmax_disparity_tex("argmax_disparity_tex");
    RDom d_range(0, max_disparity);
    argmax_disparity_tex(x,y)=cast<float>(argmin(cost_vol(x,y,d_range))[0] ); //argmax returns a tuple containing the index in the reduction domain that gave the smallest value and the value itseld


    //schedule
    //TODO
    // cost_vol.compute_root();
    argmax_disparity_tex.compute_root();


    //realize
    cv::Mat img_output=gray_left.clone(); //so we allocate a new memory for the output and don't trample over the input memory
    Halide::Buffer<float> buf_out((float*)img_output.data, img_output.cols, img_output.rows);
    argmax_disparity_tex.realize(buf_out);
    TIME_START("compute_depth_halide");
    argmax_disparity_tex.realize(buf_out);
    TIME_END("compute_depth_halide");
    argmax_disparity_tex.compile_to_lowered_stmt("argmax_disparity_tex.html", {}, HTML);

    //view_result
    debug_img_left=img_output/max_disparity;
    // cv:normalize(img_output, debug_img_left, 0, 1.0, cv::NORM_MINMAX);
    // debug_img_left=;





    // debug_img_left=frame_left.gray;
    // debug_img_right=frame_right.gray;

}



cv::Mat DepthEstimatorHalide::undistort_rectify_image(const cv::Mat img, const Frame& frame_left, const Frame& frame_right){

    //WARNING only works for the right cam at the moment since we use R2

    TIME_START("undistort");
    //if we don't have the undistorsion maps yet, create them
    // if ( m_undistort_map_x.empty() ){
        cv::Mat_<double> K_left = cv::Mat_<double>::eye( 3, 3 );
        K_left (0,0) = frame_left.K(0,0);
        K_left (1,1) = frame_left.K(1,1);
        K_left (0,2) = frame_left.K(0,2);
        K_left (1,2) = frame_left.K(1,2);
        cv::Mat_<double> distortion_left ( 5, 1 );
        //we assume that frames are already undistorted by the dataloader
        distortion_left ( 0 ) = 0;
        distortion_left ( 1 ) = 0;
        distortion_left ( 2 ) = 0;
        distortion_left ( 3 ) = 0;
        distortion_left ( 4 ) = 0;

        cv::Mat_<double> K_right = cv::Mat_<double>::eye( 3, 3 );
        K_right (0,0) = frame_right.K(0,0);
        K_right (1,1) = frame_right.K(1,1);
        K_right (0,2) = frame_right.K(0,2);
        K_right (1,2) = frame_right.K(1,2);
        cv::Mat_<double> distortion_right ( 5, 1 );
        distortion_right ( 0 ) = 0;
        distortion_right ( 1 ) = 0;
        distortion_right ( 2 ) = 0;
        distortion_right ( 3 ) = 0;
        distortion_right ( 4 ) = 0;

        //r and t from the left cam to the right one
        cv::Mat_<double> Rot  = cv::Mat_<double>::eye ( 3, 3 );
        cv::Mat_<double> Trans  = cv::Mat_<double>::eye ( 3, 1 );


        std::cout << "cam left \n " << frame_left.tf_cam_world.matrix()  << '\n';
        std::cout << "cam right \n " << frame_right.tf_cam_world.matrix()  << '\n';

        Eigen::Affine3f tf_right_left = frame_right.tf_cam_world * frame_left.tf_cam_world.inverse();
        // const Eigen::Affine3f tf_right_left = frame_left.tf_cam_world * frame_right.tf_cam_world.inverse();

        std::cout << "tf_right_left \n " << tf_right_left.matrix() << '\n';
        Eigen::Matrix3f Rot_eigen=tf_right_left.linear();
        Eigen::Vector3f Trans_eigen=tf_right_left.translation();
        cv::eigen2cv(Rot_eigen, Rot);
        cv::eigen2cv(Trans_eigen, Trans);
        cv::Mat Rot_64;
        cv::Mat Trans_64;
        Rot.convertTo(Rot_64,CV_64F);
        Trans.convertTo(Trans_64,CV_64F);

        std::cout << "Rot is \n " << Rot_64 << '\n';
        std::cout << "Trans is \n " << Trans_64 << '\n';




        cv::Mat_<double> R1  = cv::Mat_<double>::eye ( 3, 3 );
        cv::Mat_<double> R2  = cv::Mat_<double>::eye ( 3, 3 );
        cv::Mat_<double> proj_matrix_left  = cv::Mat_<double>::eye ( 3, 4 );
        cv::Mat_<double> proj_matrix_right  = cv::Mat_<double>::eye ( 3, 4 );
        cv::Mat_<double> Q  = cv::Mat_<double>::eye ( 3, 4 );


        cv::stereoRectify(K_left, distortion_left, K_right, distortion_right, img.size(),
    Rot_64, Trans_64, R1, R2, proj_matrix_left, proj_matrix_right, Q);

        //only rectify the right cam
        cv::initUndistortRectifyMap ( K_right, distortion_right, R2, K_right, img.size(), CV_32FC1, m_undistort_map_x, m_undistort_map_y);

        // cv::initUndistortRectifyMap ( K_left, distortion_left, R1, K_left, img.size(), CV_32FC1, m_undistort_map_x, m_undistort_map_y);
    // }

    //only rectify the right cam
    cv::Mat undistorted_img;
    cv::remap ( img, undistorted_img, m_undistort_map_x, m_undistort_map_y, cv::INTER_LINEAR );
    // gray_img=undistorted_img.clone(); //remap cannot function in-place so we copy the gray image back
    TIME_END("undistort");
    return undistorted_img;

}




























// void DepthEstimatorHalide::run_speed_test(Frame& frame){
//
//     VLOG(1) << "running speed test";
//     Halide::Func gradient;
//     Halide::Var x, y;
//     gradient(x, y) = x + y;
//
//     gradient.parallel(y);
//     gradient.vectorize(x, 32);
//
//
//     //run once just to get the jit compiler to do it's thing
//     Halide::Buffer<int32_t> output = gradient.realize(8192, 8192);
//     TIME_START("run_speed_test");
//     output = gradient.realize(8192, 8192);
//     TIME_END("run_speed_test");
//
//     TIME_START("naive");
//     for (int j = 0; j < output.height(); j++) {
//         for (int i = 0; i < output.width(); i++) {
//             output(i, j)=i+j;
//         }
//     }
//     TIME_END("naive");
//
//     TIME_START("sanity_check");
//     for (int j = 0; j < output.height(); j++) {
//         for (int i = 0; i < output.width(); i++) {
//             // We can access a pixel of an Buffer object using similar
//             // syntax to defining and using functions.
//             if (output(i, j) != i + j) {
//                 printf("Something went wrong!\n"
//                        "Pixel %d, %d was supposed to be %d, but instead it's %d\n",
//                        i, j, i+j, output(i, j));
//                 return;
//             }
//         }
//     }
//     TIME_END("sanity_check");
//
// }
//
// void DepthEstimatorHalide::run_speed_test_bright(Frame& frame){
//     //get grayscale img
//     cv::Mat img_gray;
//     cv::cvtColor(frame.rgb, img_gray, CV_BGR2GRAY);
//
//     //wrap the image in a halide buffer
//     Halide::Buffer<uint8_t> buf(img_gray.data, img_gray.cols, img_gray.rows);
//
//     //algorithm
//     Halide::Var x("x"),y("y");
//     Halide::Func brighter;
//     brighter(x,y) = Halide::cast<uint8_t>(Halide::clamp( buf(x,y)+50, 0 ,255 ));
//
//     //schedule
//
//
//     //realize
//     brighter.realize(buf);
//     // gradient.realize(buf);
//
//     //view_result
//     frame.rgb=img_gray;
//
// }
//
//
// Halide::Func DepthEstimatorHalide::convolution(Halide::Func f, Halide::Func hx, Halide::Expr kernel_width, Halide::Expr kernel_height){
//     Halide::Var x("x2"),y("y2");
// 	Halide::Func convolved("convolved");
//     convolved(x,y)=0;
// 	Halide::RDom k (0,kernel_width,0,kernel_height);
// 	convolved(x,y) += ( hx(k.x,k.y)*f(x+k.x-(kernel_width/2),y+k.y-(kernel_height/2)));
//     // convolved(x,y) += ( hx(k.x,k.y)*f(x,y+k.y-1));
//     // return f;
//
//     return convolved;
// }
//
// void DepthEstimatorHalide::run_speed_test_sobel(Frame& frame){
//
//     //get grayscale img
//     cv::Mat img_gray;
//     cv::cvtColor(frame.rgb, img_gray, CV_BGR2GRAY);
//
//     //wrap the image in a halide buffer
//     Halide::Buffer<uint8_t> buf(img_gray.data, img_gray.cols, img_gray.rows);
//
//
//     //attempt 3 (problem was that I was realizing the buffer inplace which means in was overlapping)
//     //algorithm
//     Halide::Var x("x"),y("y");
//     int W = img_gray.cols;
// 	int H = img_gray.rows;
// 	Halide::Func in("in");
//     in(x,y) = buf(Halide::clamp(x,0,W-1),Halide::clamp(y,0,H-1));
//
//     // Calclate Gy
// 	Halide::Func vx1("vx1");
// 	vx1(x,y) = 0;
// 	vx1(0,0) = 1; vx1(1,0) = 2; vx1(2,0) = 1;
// 	Halide::Func vx2;
// 	vx2(x,y) = 0;
// 	vx2(0,0) = 1;
// 	vx2(0,2) = -1;
// 	Halide::Func Gy("Gy");
// 	Gy = convolution(convolution(in,vx1,3,1),vx2,1,3);
//
//     //Calculate Gx
// 	Halide::Func hx1;
// 	hx1(x,y) = 0;
// 	hx1(0,0) = 1; hx1(1,0) = 0; hx1(2,0) = -1;
// 	Halide::Func hx2;
// 	hx2(x,y) = 0;
// 	hx2(0,0) = 1;
// 	hx2(1,0) = 2;
// 	hx2(2,0) = 1;
// 	Halide::Func Gx("Gx");
// 	Gx = convolution(convolution(in,hx1,3,1),hx2,1,3);
//
// 	Halide::Func mag("mag");
// 	Halide::Expr m = Halide::sqrt(Gx(x,y) * Gx(x,y) + Gy(x,y)*Gy(x,y));
// 	// mag(x,y) = Halide::cast<uint8_t>(Halide::min(m ,255.0f));
//     mag(x,y) = Halide::cast<uint8_t>(Halide::clamp( m, 0 ,255 ));
//
//
//
//     //autoschedule
//     // std::cout << "frame rgb size " << frame.rgb.cols << " " << frame.rgb.rows << '\n';
//     // // buf.dim(0).set_bounds_estimate(0, frame.rgb.cols);
//     // // buf.dim(1).set_bounds_estimate(0, frame.rgb.rows);
//     // Halide::Pipeline pipeline(mag);
//     // mag.estimate(x, 0, frame.rgb.cols)
//     //     .estimate(y, 0, frame.rgb.rows);
//     // // halide_set_num_threads(1);
//     // std::string schedule=pipeline.auto_schedule(Halide::get_jit_target_from_environment());
//     // std::cout << "schedule is \n" << schedule  << '\n';
//
//     // //schedule
//     // Gy.compute_at(mag,x);
//     // Gy.store_root();
//     // Gx.compute_at(mag,x);
//     // Gx.store_root();
//     // mag.parallel(x,32);
//     // mag.vectorize(x,64);
//
//
//     //autoschedules one and copied back without parallel
//     Halide::Pipeline pipeline(mag);
//     {
//     using namespace Halide;
//     Var x2_vi("x2_vi");
// Var x2_vo("x2_vo");
// Var x_i("x_i");
// Var x_i_vi("x_i_vi");
// Var x_i_vo("x_i_vo");
// Var x_o("x_o");
// Var y_i("y_i");
// Var y_o("y_o");
//
// Func convolved = pipeline.get_func(2);
// Func convolved_1 = pipeline.get_func(4);
// Func convolved_2 = pipeline.get_func(6);
// Func convolved_3 = pipeline.get_func(8);
// Func f0 = pipeline.get_func(3);
// Func f1 = pipeline.get_func(5);
// Func f2 = pipeline.get_func(7);
// Func mag = pipeline.get_func(9);
// Func vx1 = pipeline.get_func(1);
//
// {
//     Var x2 = convolved.args()[0];
//     convolved
//         .compute_at(mag, x_o)
//         .split(x2, x2_vo, x2_vi, 8)
//         .vectorize(x2_vi);
//     convolved.update(0)
//         // .reorder(r4$x, x2, r4$y, y2)
//         .split(x2, x2_vo, x2_vi, 8, TailStrategy::GuardWithIf)
//         .vectorize(x2_vi);
// }
// {
//     Var x2 = convolved_1.args()[0];
//     convolved_1
//         .compute_at(mag, x_o)
//         .split(x2, x2_vo, x2_vi, 8)
//         .vectorize(x2_vi);
//     convolved_1.update(0)
//         // .reorder(r9$x, x2, r9$y, y2)
//         .split(x2, x2_vo, x2_vi, 8)
//         .vectorize(x2_vi);
// }
// {
//     Var x2 = convolved_2.args()[0];
//     convolved_2
//         .compute_at(mag, x_o)
//         .split(x2, x2_vo, x2_vi, 8)
//         .vectorize(x2_vi);
//     convolved_2.update(0)
//         // .reorder(r14$x, x2, r14$y, y2)
//         .split(x2, x2_vo, x2_vi, 8, TailStrategy::GuardWithIf)
//         .vectorize(x2_vi);
// }
// {
//     Var x2 = convolved_3.args()[0];
//     convolved_3
//         .compute_at(mag, x_o)
//         .split(x2, x2_vo, x2_vi, 8)
//         .vectorize(x2_vi);
//     convolved_3.update(0)
//         // .reorder(r19$x, x2, r19$y, y2)
//         .split(x2, x2_vo, x2_vi, 8)
//         .vectorize(x2_vi);
// }
// {
//     Var x = f0.args()[0];
//     Var y = f0.args()[1];
//     f0
//         .compute_root()
//         // .parallel(y)
//         // .parallel(x);
//         ;
// }
// {
//     Var x = f1.args()[0];
//     Var y = f1.args()[1];
//     f1
//         .compute_root()
//         // .parallel(y)
//         // .parallel(x);
//         ;
// }
// {
//     Var x = f2.args()[0];
//     Var y = f2.args()[1];
//     f2
//         .compute_root()
//         // .parallel(y)
//         // .parallel(x);
//         ;
// }
// {
//     Var x = mag.args()[0];
//     Var y = mag.args()[1];
//     mag
//         .compute_root()
//         .split(x, x_o, x_i, 64)
//         .split(y, y_o, y_i, 64)
//         .reorder(x_i, y_i, x_o, y_o)
//         .split(x_i, x_i_vo, x_i_vi, 32)
//         .vectorize(x_i_vi)
//         // .parallel(y_o);
//         ;
// }
// {
//     Var x = vx1.args()[0];
//     Var y = vx1.args()[1];
//     vx1
//         .compute_root()
//         // .parallel(y)
//         // .parallel(x);
//         ;
// }
// }
//
//
//     //realize
//     // mag.realize(buf);
//     // gradient.realize(buf);
//     cv::Mat img_output=img_gray.clone(); //so we allocate a new memory for the output and don't trample over the input memory
//     Halide::Buffer<uint8_t> buf_out(img_output.data, img_output.cols, img_output.rows);
//     mag.realize(buf_out);
//     TIME_START("sobel_halide");
//     mag.realize(buf_out);
//     TIME_END("sobel_halide");
//
//     //view_result
//     frame.rgb=img_output;
//
//
//     //opencv sobel
//     cv::setNumThreads (4);
//     int ddepth = CV_16S;
//     cv::Mat grad_x, grad_y;
//     cv::Mat abs_grad_x, abs_grad_y;
//
//     TIME_START("sobel_cv");
//     cv::Sobel( img_gray, grad_x, ddepth, 1, 0, 3 );
//     cv::convertScaleAbs( grad_x, abs_grad_x );
//
//     /// Gradient Y
//     //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
//     cv::Sobel( img_gray, grad_y, ddepth, 0, 1, 3 );
//     cv::convertScaleAbs( grad_y, abs_grad_y );
//
//     /// Total Gradient (approximate)
//     cv::Mat grad;
//     cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
//     TIME_END("sobel_cv");
//     // frame.rgb=grad;
//
//
//
//
// }
