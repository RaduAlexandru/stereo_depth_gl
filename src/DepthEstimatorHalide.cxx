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

//ros
#include "stereo_depth_gl/RosTools.h"

//loguru
#include <loguru.hpp>

//omp
#include <omp.h>

using namespace Halide;
using namespace configuru;


DepthEstimatorHalide::DepthEstimatorHalide(){

    init_params();
    init_halide();
}

//needed so that forward declarations work
DepthEstimatorHalide::~DepthEstimatorHalide(){
}

void DepthEstimatorHalide::init_params(){
    //get the config filename
    ros::NodeHandle private_nh("~");
    std::string config_file= getParamElseThrow<std::string>(private_nh, "config_file");

    //read all the parameters
    Config cfg = configuru::parse_file(std::string(CMAKE_SOURCE_DIR)+"/config/"+config_file, CFG);
    Config depth_config=cfg["depth_halide"];
    m_use_cost_volume_filtering = depth_config["use_cost_volume_filtering"];

}

void DepthEstimatorHalide::init_halide(){

}

Func DepthEstimatorHalide::generate_disparity_func(const cv::Mat& gray_left, const cv::Mat& gray_right, const int max_disparity, const int radius){

    //wrap the image in a halide buffer
    Buffer<float> buf_left((float*)gray_left.data, gray_left.cols, gray_left.rows);
    Buffer<float> buf_right((float*)gray_right.data, gray_right.cols, gray_right.rows);

    //ALGORITHM

    //clamp the x and y so as to not go out of bounds
    Var x("x"),y("y"), c("c");
    Func buf_right_cl = BoundaryConditions::repeat_edge(buf_right);


    //cost volume
    Func cost_vol("cost_vol");
    cost_vol(x,y,c)=999999.0;
    cost_vol(x,y,c) = Halide::abs(buf_left(x,y) - buf_right_cl(x-c,y));


    Func cost_vol_filtered("cost_vol_filtered");
    if(m_use_cost_volume_filtering){
        // int radius=9;
        // float eps=0.001;
        //things that can be precomputed
        // Func I = BoundaryConditions::repeat_edge(buf_left);
        // Func mean_I = boxfilter(I, radius);
        // Func mean_II = boxfilter(mul_elem(I,I), radius);
        // Func var_I;     var_I(x,y) = mean_II(x,y) - mul_elem(mean_I,mean_I)(x,y);
        // //things that depend on the image itself and not the guidance
        Func mean_p = boxfilter_3D(cost_vol, radius);
        // Func mean_Ip = boxfilter_3D(mul_elem_replicate_lhs(I,cost_vol),radius);
        // Func cov_Ip;    cov_Ip(x,y,d)=mean_Ip(x,y,d) - mul_elem_replicate_lhs(mean_I,mean_p)(x,y,d);
        // Func a;         a(x,y,d)=cov_Ip(x,y,d) / (var_I(x,y) +eps);
        // Func b;         b(x,y,d)=mean_p(x,y,d) - mul_elem_replicate_rhs(a,mean_I)(x,y,d);
        // Func mean_a = boxfilter_3D(a, radius);
        // Func mean_b = boxfilter_3D(b,radius);
        // Func out;      out(x,y,d)=mul_elem_replicate_rhs(mean_a,I)(x,y,d) + mean_b(x,y,d);

        // cost_vol_filtered(x,y,d)=out(x,y,d);
        cost_vol_filtered(x,y,c)=mean_p(x,y,c);
    }else{
        cost_vol_filtered(x,y,c)=cost_vol(x,y,c);
    }



    //argmax the cost volume
    Func disparity("disparity");
    RDom d_range(0, max_disparity);
    disparity(x,y)=cast<float>(argmin(cost_vol_filtered(x,y,d_range))[0] ); //argmax returns a tuple containing the index in the reduction domain that gave the smallest value and the value itseld



    // // schedule
    // // mean_p.compute_root();
    // cost_vol.compute_root();
    // disparity.compute_root();

    // auto schedule
    const int kParallelism = 32;
    const int kLastLevelCacheSize = 314573;
    const int kBalance = 1; //how much more expensive is the memory vs arithmetic costs. Higher values means less compute_root
    MachineParams machine_params(kParallelism, kLastLevelCacheSize, kBalance);
    Halide::Pipeline pipeline(disparity);
    disparity.estimate(x, 0, gray_left.cols)
            .estimate(y, 0, gray_left.rows);
    // halide_set_num_threads(1);
    std::string schedule=pipeline.auto_schedule(Halide::get_jit_target_from_environment(), machine_params);
    pipeline.compile_jit();
    std::cout << "schedule is \n" << schedule  << '\n';


//     Var x_i("x_i");
// Var x_i_vi("x_i_vi");
// Var x_i_vo("x_i_vo");
// Var x_o("x_o");
// Var x_vi("x_vi");
// Var x_vo("x_vo");
// Var y_i("y_i");
// Var y_o("y_o");
//
// Func argmin = pipeline.get_func(8);
// Func disparity = pipeline.get_func(9);
// Func sum = pipeline.get_func(3);
// Func sum_1 = pipeline.get_func(5);
//
// {
//     Var x = argmin.args()[0];
//     RVar r25$x(argmin.update(0).get_schedule().rvars()[0].var);
//     argmin
//         .compute_at(disparity, x_o)
//         .split(x, x_vo, x_vi, 8)
//         .vectorize(x_vi);
//     argmin.update(0)
//         .reorder(x, r25$x, y)
//         .split(x, x_vo, x_vi, 8)
//         .vectorize(x_vi);
// }
// {
//     Var x = disparity.args()[0];
//     Var y = disparity.args()[1];
//     disparity
//         .compute_root()
//         .split(x, x_o, x_i, 64)
//         .split(y, y_o, y_i, 4)
//         .reorder(x_i, y_i, x_o, y_o)
//         .split(x_i, x_i_vo, x_i_vi, 8)
//         .vectorize(x_i_vi)
//         .parallel(y_o);
// }
// {
//     Var x = sum.args()[0];
//     RVar r4$x(sum.update(0).get_schedule().rvars()[0].var);
//     sum
//         .compute_at(disparity, x_o)
//         .split(x, x_vo, x_vi, 8)
//         .vectorize(x_vi);
//     sum.update(0)
//         .reorder(r4$x, x, c, y)
//         .split(x, x_vo, x_vi, 8)
//         .vectorize(x_vi);
// }
// {
//     Var x = sum_1.args()[0];
//     RVar r4$x(sum_1.update(0).get_schedule().rvars()[0].var);
//     sum_1
//         .compute_at(disparity, x_o)
//         .split(x, x_vo, x_vi, 8)
//         .vectorize(x_vi);
//     sum_1.update(0)
//         .reorder(x, r4$x, y, c)
//         .split(x, x_vo, x_vi, 8)
//         .vectorize(x_vi);
// }
// pipeline.compile_jit();





    disparity.compile_to_lowered_stmt("disparity.html", {}, HTML);

    return disparity;

}


void DepthEstimatorHalide::compute_depth(const Frame& frame_left, const Frame& frame_right){

    std::cout << "compute depth" << '\n';

    int img_subsample_factor=1;

    //get images
    cv::Mat gray_left=frame_left.gray/255;
    // cv::Mat gray_right=undistort_rectify_image(frame_right.gray, frame_left, frame_right);
    // gray_right=gray_right/255;
    cv::Mat gray_right=frame_right.gray/255;
    cv::resize(gray_left, gray_left, cv::Size(), 1.0/img_subsample_factor, 1.0/img_subsample_factor);
    cv::resize(gray_right, gray_right, cv::Size(), 1.0/img_subsample_factor, 1.0/img_subsample_factor);
    std::cout << "received frame of size " << gray_left.rows << " " << gray_left.cols << '\n';


    int max_disparity=50;
    int radius=5;

    if(m_is_first_frame){
        m_is_first_frame=false;
        m_disparity=generate_disparity_func(gray_left, gray_right, max_disparity, radius);
    }

    //output
    cv::Mat img_output=cv::Mat(gray_left.rows, gray_left.cols, gray_left.type());
    Halide::Buffer<float> buf_out((float*)img_output.data, img_output.cols, img_output.rows);

    m_disparity.realize(buf_out);
    TIME_START("disparity");
    // for (size_t i = 0; i < 10; i++) {
        m_disparity.realize(buf_out);
    // }
    TIME_END("disparity");

    m_mesh=disparity_to_mesh(img_output);

    //view_result
    // debug_img_left=img_output/img_output.cols;
    // debug_img_left=img_output/15;
    cv:normalize(img_output, debug_img_left, 0, 1.0, cv::NORM_MINMAX);



    // // guided filter
    // cv::Mat gray_left_smooth=guided_filter(gray_left, gray_left, 5, 0.001);
    //
    // debug_img_left=gray_left;
    // debug_img_right=gray_left_smooth;


    // //try a 2D integral image
    // Var x("x"),y("y"), c("c");
    // //wrap the image in a halide buffer
    // Buffer<float> buf_left((float*)gray_left.data, gray_left.cols, gray_left.rows);
    // Buffer<float> buf_right((float*)gray_right.data, gray_right.cols, gray_right.rows);
    // Func buf_left_cl = BoundaryConditions::repeat_edge(buf_left);
    // Func buf_right_cl = BoundaryConditions::repeat_edge(buf_right);
    // Func out=integral_img_2D(buf_left_cl, gray_left.cols, gray_left.rows);
    // // Func out=boxfilter(buf_left_cl, 9);
    // //output
    // cv::Mat img_output=gray_left.clone(); //so we allocate a new memory for the output and don't trample over the input memory
    // Halide::Buffer<float> buf_out((float*)img_output.data, img_output.cols, img_output.rows);
    // // auto schedule
    // const int kParallelism = 32;
    // const int kLastLevelCacheSize = 314573;
    // const int kBalance = 1; //how much more expensive is the memory vs arithmetic costs. Higher values means less compute_root
    // MachineParams machine_params(kParallelism, kLastLevelCacheSize, kBalance);
    // Halide::Pipeline pipeline(out);
    // out.estimate(x, 0, img_output.cols)
    //         .estimate(y, 0, img_output.rows);
    // std::string schedule=pipeline.auto_schedule(Halide::get_jit_target_from_environment(), machine_params);
    // pipeline.compile_jit();
    // std::cout << "schedule is \n" << schedule  << '\n';
    // //realize
    // out.realize(buf_out);
    // TIME_START("compute_integral");
    // // for (size_t i = 0; i < 10; i++) {
    //     out.realize(buf_out);
    // // }
    // TIME_END("compute_integral");
    // //view result
    // debug_img_left=img_output;

}

cv::Mat DepthEstimatorHalide::cost_volume_cpu(const cv::Mat& gray_left, const cv::Mat& gray_right){
    int max_disparity=50;
    std::vector<cv::Mat> cost_vol(max_disparity);
    for (size_t d = 0; d < max_disparity; d++) {
        cost_vol[d]=cv::Mat(gray_left.rows,gray_left.cols, CV_32F);
        cost_vol[d]=999999.0;
    }
    // cv::Mat cost_vol(gray_left.rows, gray_left.cols, CV_8UC(max_disparity), CV_32F);

    TIME_START("cost_vol_cpu");

    for (size_t y = 0; y < gray_left.rows; y++) {
        for (size_t x = 0; x < gray_left.cols; x++) {

            //grab a pixel on the left image
            float pix_left=gray_left.at<float>(y,x);

            //check dispairties
            for (size_t d = 0; d < max_disparity; d++) {
                int idx_disp=std::max((int)(x-d), 0);
                float pix_right=gray_right.at<float>(y,idx_disp);

                float error=std::fabs(pix_left-pix_right);

                cost_vol[d].at<float>(y,x)=error;
            }


        }
    }


    //grab the index with the minimum a error
    cv::Mat disparity=cv::Mat(gray_left.rows,gray_left.cols, CV_32F);
    disparity=0.0;
    for (size_t y = 0; y < gray_left.rows; y++) {
        for (size_t x = 0; x < gray_left.cols; x++) {

            float min_error=999999;
            int best_disp=0;

            //check for the minimal error
            for (size_t d = 0; d < max_disparity; d++) {
                float error=cost_vol[d].at<float>(y,x);
                if(error<min_error){
                    min_error=error;
                    best_disp=d;
                }
            }

            //store the best disp
            disparity.at<float>(y,x)=best_disp;


        }
    }



    TIME_END("cost_vol_cpu");

    return disparity;
}

Mesh DepthEstimatorHalide::disparity_to_mesh(const cv::Mat& disp_img){
    Mesh mesh;

    int nr_pixels=disp_img.rows*disp_img.cols;

    mesh.V.resize(nr_pixels,3);
    mesh.V.setZero();

    for (size_t i = 0; i < disp_img.rows; i++) {
        for (size_t j = 0; j < disp_img.cols; j++) {
            float d=disp_img.at<float>(i,j);

            int idx_p=i*disp_img.cols+j;
            mesh.V.row(idx_p) << j,i,d*20;
        }
    }

    // //make also some colors based on disparity
    mesh.C.resize(nr_pixels,3);
    double min_d, max_d;
    min_d =  mesh.V.col(2).minCoeff();
    max_d =  mesh.V.col(2).maxCoeff();
    std::cout << "min max d is " << min_d << " " << max_d << '\n';
    for (size_t i = 0; i < mesh.C.rows(); i++) {
        float gray_val = lerp(mesh.V(i,2), min_d, max_d, 0.0, 1.0 );
        mesh.C(i,0)=mesh.C(i,1)=mesh.C(i,2)=gray_val;
    }


    return mesh;

}


cv::Mat DepthEstimatorHalide::guided_filter(const cv::Mat& I_cv, const cv::Mat& p_cv, const float radius, const float eps){

    Buffer<float> I_buf((float*)I_cv.data, I_cv.cols, I_cv.rows);
    Buffer<float> p_buf((float*)p_cv.data, p_cv.cols, p_cv.rows);
    Func I = BoundaryConditions::repeat_edge(I_buf);
    Func p = BoundaryConditions::repeat_edge(p_buf);

    //clamp the x and y so as to not go out of bounds
    Var x("x"),y("y"), d("d");



    // //p is the same as I because the guidance image and the smoothign image are the same
    // RDom r(-radius,radius,-radius,radius);
    // Expr rx=x+r.x;
    // Expr ry=y+r.y;
    //
    // Func mean_I;
    // mean_I(x,y)=sum(I(rx, ry))/(radius*radius);
    // Func II;
    // II(x,y)=I(x,y)*I(x,y);
    // Func mean_II;
    // mean_II(x,y) = sum(II(rx, ry))/(radius*radius);
    // Func cov_Ip;
    // cov_Ip(x,y)=mean_II(x,y)-mean_I(x,y)*mean_I(x,y);
    //
    // Func a;
    // a(x,y)=cov_Ip(x,y)/ (cov_Ip(x,y) +eps);
    // Func b;
    // b(x,y)=mean_I(x,y) - a(x,y)*mean_I(x,y);
    //
    // Func mean_a;
    // mean_a(x,y)=sum(a(rx, ry))/(radius*radius);
    // Func mean_b;
    // mean_b(x,y)=sum(b(rx, ry))/(radius*radius);
    //
    // Func out;
    // out(x,y)=mean_a(x,y)*I(x,y) + mean_b(x,y);
    // //trying out to ignore some parts
    // // out(x,y)=select(x<400, 0, mean_a(x,y)*I(x,y) + mean_b(x,y));
    //
    //
    // // cv::Mat mean_p = boxfilter(p, r);
    // // cv::Mat mean_Ip = boxfilter(I.mul(p), r);
    // // cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p); // this is the covariance of (I, p) in each local patch.
    // //
    // // cv::Mat a = cov_Ip / (var_I + eps); // Eqn. (5) in the paper;
    // // cv::Mat b = mean_p - a.mul(mean_I); // Eqn. (6) in the paper;
    // //
    // // cv::Mat mean_a = boxfilter(a, r);
    // // cv::Mat mean_b = boxfilter(b, r);







    //attempt 2 guidance and image are different
    //things that can be precomputed
    Func mean_I = boxfilter(I, radius);
    Func mean_II = boxfilter(mul_elem(I,I), radius);
    Func var_I;     var_I(x,y) = mean_II(x,y) - mul_elem(mean_I,mean_I)(x,y);

    //things that depend on the image itself and not the guidance
    Func mean_p = boxfilter(p, radius);
    Func mean_Ip = boxfilter(mul_elem(I,p),radius);
    Func cov_Ip;    cov_Ip(x,y)=mean_Ip(x,y) - mul_elem(mean_I,mean_p)(x,y);
    Func a;         a(x,y)=cov_Ip(x,y) / (var_I(x,y) +eps);
    Func b;         b(x,y)=mean_p(x,y) - mul_elem(a,mean_I)(x,y);
    Func mean_a = boxfilter(a, radius);
    Func mean_b = boxfilter(b,radius);
    Func out;      out(x,y)=mul_elem(mean_a,I)(x,y) + mean_b(x,y);



    //output
    cv::Mat img_output=I_cv.clone(); //so we allocate a new memory for the output and don't trample over the input memory
    Halide::Buffer<float> buf_out((float*)img_output.data, I_cv.cols, I_cv.rows);


    //schedule
    // a.compute_root();
    // b.compute_root();
    // mean_a.compute_root();
    // mean_b.compute_root();
    // out.compute_root();

    //get autoschedule
    // Halide::Pipeline pipeline(out);
    // out.estimate(x, 0, I_cv.cols)
    //         .estimate(y, 0, I_cv.rows);
    // // halide_set_num_threads(1);
    // std::string schedule=pipeline.auto_schedule(Halide::get_jit_target_from_environment());
    // std::cout << "schedule is \n" << schedule  << '\n';

    //get autoschedule attempt 2
    const int kParallelism = 32;
    // const int kLastLevelCacheSize = 16 * 1024 * 1024;
    const int kLastLevelCacheSize = 314573;
    const int kBalance = 1; //how much more expensive is the memory vs arithmetic costs. Higher values means less compute_root
    MachineParams machine_params(kParallelism, kLastLevelCacheSize, kBalance);
    Halide::Pipeline pipeline(out);
    out.estimate(x, 0, I_cv.cols)
            .estimate(y, 0, I_cv.rows);
    // halide_set_num_threads(1);
    std::string schedule=pipeline.auto_schedule(get_host_target(),machine_params);
    std::cout << "schedule is \n" << schedule  << '\n';
    pipeline.compile_jit();


    //realize
    out.realize(buf_out);
    TIME_START("guided_filter");
    for (size_t i = 0; i < 100; i++) {
        out.realize(buf_out);
    }
    TIME_END("guided_filter");





    return img_output;

}

Halide::Func DepthEstimatorHalide::boxfilter(const Halide::Func& I, const int radius){
    Var x("x"),y("y");
    RDom r(-radius,radius);
    Expr rx=clamp(x+r.x,0,999999);
    Expr ry=clamp(y+r.x,0,999999);

    Func blur_x;
    Func output;
    blur_x(x,y)=sum(I(rx, y))/radius;
    output(x,y)=sum(blur_x(x, ry))/radius;

    return output;
}

Halide::Func DepthEstimatorHalide::boxfilter_slice(const Halide::Func& I, const int radius, const int slice){
    Var x("x"),y("y");
    RDom r(-radius,radius);
    Expr rx=clamp(x+r.x,0,999999);
    Expr ry=clamp(y+r.x,0,999999);

    Func blur_x;
    Func output;
    blur_x(x,y)=sum(I(rx, y, slice))/radius;
    output(x,y)=sum(blur_x(x, ry))/radius;

    return output;
}

Halide::Func DepthEstimatorHalide::boxfilter_3D(const Halide::Func& I, const int radius){
    Var x("x"),y("y"),c("c");
    RDom r(-radius,radius);
    Expr rx=clamp(x+r.x,0,999999);
    Expr ry=clamp(y+r.x,0,999999);

    Func blur_x;
    Func output;
    blur_x(x,y,c)=sum(I(rx, y, c))/radius;
    output(x,y,c)=sum(blur_x(x, ry, c))/radius;

    return output;
}

Halide::Func DepthEstimatorHalide::boxfilter_3D_integral_img(const Halide::Func& I, const int radius){
    //idea from https://github.com/halide/Halide/blob/master/test/correctness/multi_pass_reduction.cpp

    Var x("x"),y("y"),c("c");
    //
    Expr width=I.output_buffer().width();
    Expr height=I.output_buffer().height();
    Expr channels=I.output_buffer().channels();
    //
    // RDom rx(1, width);
    // RDom ry(1, height);
    // Func integral;
    // integral(x,y,c)=I(x,y,c);
    // integral(x, ry, c) += integral(x, ry - 1, c);
    // integral(rx, y, c) += integral(rx - 1, y, c);


    // Func integral = I;
    // RDom r(0,width,0,height,"r_box_3D_integral"); //domain over the whole image
    // integral(x, r.y, c) += integral(x, r.y - 1, c);
    // integral(r.x, y, c) += integral(r.x - 1, y, c);


//     Func integralImage;
// integralImage(x,y,c) = 0.0f; // Pure definition
// integralImage(intImDom.x,intImDom.y,c) = ip(intImDom.x,intImDom.y,c)
//                   + integralImage(intImDom.x-1,intImDom.y,c)
//                   + integralImage(intImDom.x,intImDom.y-1,c)
//                   - integralImage(intImDom.x-1,intImDom.y-1,c);

    // // Walk down the image in vectors
    // integral.update(1).vectorize(x, 4);
    //
    // // Walk across the image in parallel. We need to do an unsafe
    // // reorder operation here to move y to the outer loop, because
    // // we don't have the ability to reorder vars with rvars yet.
    // integral.update(2).reorder(Var(rx.x.name()), y).parallel(y);



    //attempt 2  http://halide-lang.org/docs/_r_dom_8h_source.html
    RDom r(0,width,0,height,"r_box_3D_integral"); //domain over the whole image
    Func sum_x, sum_y;
    sum_x(x, y, c)   = I(x, y, c);
    sum_x(r.x, y, c) = sum_x(r.x, y, c) + sum_x(r.x-1, y, c);
    // sum_y(x, y, c)   = sum_x(x, y, c);
    // sum_y(x, r.y, c) = sum_y(x, r.y, c) + sum_y(x, r.y-1, c);



    return sum_x;
}

Halide::Func DepthEstimatorHalide::mul_elem(const Halide::Func& lhs, const Halide::Func& rhs){
    Var x("x"),y("y");
    Func mul;
    mul(x,y)=lhs(x,y)*rhs(x,y);
    return mul;
}

Halide::Func mul_elem_3D(const Halide::Func& lhs, const Halide::Func& rhs){
    Var x("x"),y("y"),c("c");
    Func mul;
    mul(x,y,c)=lhs(x,y,c)*rhs(x,y,c);
    return mul;
}

Halide::Func DepthEstimatorHalide::mul_elem_slice_lhs(const Halide::Func& lhs, const Halide::Func& rhs, const int slice){
    Var x("x"),y("y");
    Func mul;
    mul(x,y)=lhs(x,y,slice)*rhs(x,y);
    return mul;
}

Halide::Func DepthEstimatorHalide::mul_elem_slice_rhs(const Halide::Func& lhs, const Halide::Func& rhs, const int slice){
    Var x("x"),y("y");
    Func mul;
    mul(x,y)=lhs(x,y)*rhs(x,y,slice);
    return mul;
}

Halide::Func DepthEstimatorHalide::mul_elem_replicate_lhs(const Halide::Func& lhs, const Halide::Func& rhs){
    Var x("x"),y("y"),c("c");
    Func mul;
    mul(x,y,c)=lhs(x,y)*rhs(x,y,c);
    return mul;
}
Halide::Func DepthEstimatorHalide::mul_elem_replicate_rhs(const Halide::Func& lhs, const Halide::Func& rhs){
    Var x("x"),y("y"),c("c");
    Func mul;
    mul(x,y,c)=lhs(x,y,c)*rhs(x,y);
    return mul;
}

Halide::Func DepthEstimatorHalide::integral_img_2D(const Halide::Func& I, const int width, const int height){
    Func sum_x("sum_x"), sum_y("sum_y");
    Var x("x"), y("y");
    RDom r(1,width-1,1,height-1,"r_integ_2D");
    sum_x(x, y)   = I(x, y);
    sum_x(r.x, y) = sum_x(r.x, y) + sum_x(r.x-1, y);
    sum_y(x, y)   = sum_x(x, y);
    sum_y(x, r.y) = sum_y(x, r.y) + sum_y(x, r.y-1);
    return sum_y;

    // Func integral("integral");
    // Var x("x"), y("y");
    // RDom r(1,width,1,height);
    // integral(x, y)   = I(x, y);
    // integral(x, r.y) += integral(x, r.y - 1);
    // integral(r.x, y) += integral(r.x - 1, y);

//     // Walk down the image in vectors
//         integral.update(0).vectorize(x, 4);
//
// //         // Walk across the image in parallel. We need to do an unsafe
// //         // reorder operation here to move y to the outer loop, because
// //         // we don't have the ability to reorder vars with rvars yet.
// integral.update(1).reorder(Var(r.x.name()), y).parallel(y);

    // return integral;


}
Halide::Func DepthEstimatorHalide::integral_img_3D(const Halide::Func& I, const int width, const int height){
    Func sum_x("sum_x"), sum_y("sum_y");
    Var x("x"), y("y"), c("c");
    RDom r(1,width-1,1,height-1);
    sum_x(x, y, c)   = I(x, y, c);
    sum_x(r.x, y, c) = sum_x(r.x, y, c) + sum_x(r.x-1, y, c);
    sum_y(x, y, c)   = sum_x(x, y, c);
    sum_y(x, r.y, c) = sum_y(x, r.y, c) + sum_y(x, r.y-1, c);

    return sum_y;
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
