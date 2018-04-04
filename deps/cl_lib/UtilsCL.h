#pragma once

//my stuff
#include "Image2DSafe.h";

//c++
#include <iostream>

//loguru
#include <loguru.hpp>

//opencl
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include "CL/cl2.hpp"
#include <CL/cl.h>

//OpenCV
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

namespace cl{

//from a cv_mat creates a cl_img which buffer is allocated in host and is memory aligned
inline cl::Image2DSafe cv_mat2cl_img(const cv::Mat& cv_mat, const cl::ImageFormat& format, const cl_mem_flags& flags, const Context& context){
    int size_bytes=cv_mat.step[0] * cv_mat.rows;
    round_up_to_nearest_multiple(size_bytes,64); //allocate memory that is multiple of 64 bytes

    //create the cl image
    cl::Image2DSafe cl_img(context, format, flags, cv_mat.cols, cv_mat.rows, size_bytes);

    //fill it with data
    if(cv_mat.isContinuous()){
        uchar* mat_buf = cl_img.get_host_buf_ptr();

        int idx_insert=0;
        const uchar* pixel = cv_mat.ptr<uchar>(0);
        for (size_t i = 0; i < cv_mat.rows*cv_mat.cols; i++) {
            mat_buf[idx_insert]= *pixel;
            pixel++;
            idx_insert+=1;
        }
    }else{
        LOG(FATAL) << "RGB image not continuous, which means the pixel iteration must be done in differnt way: look at https://docs.opencv.org/2.4/doc/tutorials/core/how_to_scan_images/how_to_scan_images.html#howtoscanimagesopencv";
    }
    return cl_img;
}

//creates a cl_img with the same format, size and flags as the source one but with empty buffer
inline cl::Image2DSafe cl_img_like(const cl::Image2DSafe& src_cl_img){
    cl::Image2DSafe cl_img( src_cl_img.get_context(), src_cl_img.get_format(), src_cl_img.get_flags(),
                            src_cl_img.get_width(), src_cl_img.get_height(), src_cl_img.get_size_bytes());
    return cl_img;
}

//same but user specified format
inline cl::Image2DSafe cl_img_like(const cl::Image2DSafe& src_cl_img, const cl::ImageFormat& format){
    cl::Image2DSafe cl_img( src_cl_img.get_context(), format, src_cl_img.get_flags(),
                            src_cl_img.get_width(), src_cl_img.get_height(), src_cl_img.get_size_bytes());
    return cl_img;
}


//if the ptr is aligned to 4096 and multiple of 64 than opencl can do a zero copy of it if we use CL_MEM_USE_HOST_PTR
//https://vjordan.info/log/fpga/day-6-opencl-malloc.html
inline unsigned int verifyZeroCopyPtr(void *ptr, unsigned int sizeOfContentsOfPtr){
        // int status; // so we only have one exit point from function
        // if((unsigned int)ptr % 4096 == 0){ // page alignment and cache alignment
        //     if(sizeOfContentsOfPtr % 64 == 0){ // multiple of cache size
        //         status = 1;
        //     }else {
        //         status = 0;
        //     }
        // }else {
        //     status = 0;
        // }
        // return status;

        return 1;
}



}
