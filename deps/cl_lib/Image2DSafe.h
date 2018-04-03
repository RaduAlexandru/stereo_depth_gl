#pragma once

//c++
#include <cstdlib>
#include <memory>
#include <iostream>

//opencl
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include "CL/cl2.hpp"
#include <CL/cl.h>

//TAKES CARE OF MEMORY ALLOCATION ON THE HOST AND DEALLOCATING WHEN IT'S DONE WITH IT

namespace cl{

    class Image2DSafe{
    public:
        Image2DSafe(const Context& context, const ImageFormat& format, const cl_mem_flags& flags, const int width, const int height, const int size_bytes){
            m_host_buf = (uchar *)aligned_alloc(4096, sizeof(uchar) * size_bytes);
            cl::Buffer cl_buffer(context, flags | CL_MEM_USE_HOST_PTR, size_bytes, m_host_buf);
            m_img=std::make_shared<cl::Image2D>(cl::Image2D(context, format, cl_buffer, width, height));
        }

        ~Image2DSafe(){
            free(m_host_buf);
        }

        cl::Image2D get_img(){
            return *m_img;
        }

        uchar* get_host_buf_ptr(){
            return m_host_buf;
        }


    private:
        uchar* m_host_buf;
        std::shared_ptr<cl::Image2D> m_img;

    };





}
