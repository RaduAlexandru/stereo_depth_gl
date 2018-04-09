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
        Image2DSafe(const Context& context, const ImageFormat& format, const cl_mem_flags& flags, const int width, const int height, const int size_bytes):
            m_context(context){
            m_size_bytes=size_bytes;
            m_format=format;
            m_flags=flags;
            m_width=width;
            m_height=height;

            m_host_buf = (uchar *)aligned_alloc(4096, sizeof(uchar) * size_bytes);
            cl::Buffer cl_buffer(context, flags | CL_MEM_USE_HOST_PTR, size_bytes, m_host_buf);
            m_img=std::make_shared<cl::Image2D>(cl::Image2D(context, format, cl_buffer, width, height));
        }

        ~Image2DSafe(){
            std::cout << "freeing" << '\n';
            free(m_host_buf);
        }

        cl::Image2D get_img()const{
            return *m_img;
        }

        uchar* get_host_buf_ptr(){
            return m_host_buf;
        }

        //getters
        const Context& get_context()const{ return m_context; }
        ImageFormat get_format()const{ return m_format; }
        cl_mem_flags get_flags()const{ return m_flags; }
        int get_width()const{ return m_width; }
        int get_height()const{ return m_height; }
        int get_size_bytes()const{ return m_size_bytes; }



    private:
        const Context& m_context; //Watchout this is a reference to the context, doing it like this so we don't copy the whole thing
        ImageFormat m_format;
        cl_mem_flags m_flags;
        int m_width;
        int m_height;
        int m_size_bytes;



        uchar* m_host_buf;
        std::shared_ptr<cl::Image2D> m_img;

    };





}
