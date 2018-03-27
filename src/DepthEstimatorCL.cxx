#include "stereo_depth_cl/DepthEstimatorCL.h"

//c++
#include <cmath>
#include <stdlib.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>

//My stuff
#include "stereo_depth_cl/Profiler.h"
#include "stereo_depth_cl/MiscUtils.h"

//Libigl
#include <igl/opengl/glfw/Viewer.h>

//cv
#include <cv_bridge/cv_bridge.h>

//loguru
#include <loguru.hpp>

//omp
#include <omp.h>


// Compute c = a + b.
static const char source[] =
    "kernel void add(\n"
    "       global const float *a,\n"
    "       global const float *b,\n"
    "       global float *c\n"
    "       )\n"
    "{\n"
    "    size_t i = get_global_id(0);\n"
        "c[i] = pow(a[i], b[i]);\n"
    "}\n";


//from https://raw.githubusercontent.com/KhronosGroup/OpenCL-Headers/master/opencl22/CL/cl.h
/* cl_device_svm_capabilities */
// #define CL_DEVICE_SVM_COARSE_GRAIN_BUFFER           (1 << 0)
// #define CL_DEVICE_SVM_FINE_GRAIN_BUFFER             (1 << 1)
// #define CL_DEVICE_SVM_FINE_GRAIN_SYSTEM             (1 << 2)
// #define CL_DEVICE_SVM_ATOMICS                       (1 << 3)

DepthEstimatorCL::DepthEstimatorCL():
        m_scene_is_modified(false),
        m_cl_profiling_enabled(true),
        m_show_images(false)
        {

    init_opencl();
}

//needed so that forward declarations work
DepthEstimatorCL::~DepthEstimatorCL(){
}

void DepthEstimatorCL::init_opencl(){
    VLOG(1) << "init opencl";
    std::cout << "init opencl" << '\n';


	// Get list of OpenCL platforms.
	std::vector<cl::Platform> platform;
	cl::Platform::get(&platform);
	if (platform.empty()) {
	    std::cerr << "OpenCL platforms not found." << std::endl;
	    return;
	}

	// Get first available GPU device
    std::vector<cl::Device> devices_available;
    platform[0].getDevices(CL_DEVICE_TYPE_GPU, &devices_available);
    m_device=devices_available[0];

    std::cout << m_device.getInfo<CL_DEVICE_NAME>() << std::endl;

    m_context = cl::Context(m_device);

	// // Create command queue.
	m_queue = cl::CommandQueue(m_context,m_device);


    //check svm capabilities
    cl_device_svm_capabilities caps;
    std::cout << "checking SVM capabilities of device id " << m_device() << '\n';
    cl_int err = clGetDeviceInfo( m_device(), CL_DEVICE_SVM_CAPABILITIES, sizeof(cl_device_svm_capabilities), &caps, 0);
    if(err == CL_INVALID_VALUE){
        std::cout << "No SVM support" << '\n';
    }
    if(err == CL_SUCCESS && (caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)){
        std::cout << "Coarse-grained buffer" << '\n';
    }
    if(err == CL_SUCCESS && (caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)){
        std::cout << "Fine-grained buffer" << '\n';
    }
    if(err == CL_SUCCESS && (caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) && (caps & CL_DEVICE_SVM_ATOMICS)){
        std::cout << "Fine-grained buffer with atomics" << '\n';
    }
    if(err == CL_SUCCESS && (caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)){
        std::cout << "Fine-grained system" << '\n';
    }
    if(err == CL_SUCCESS && (caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) && (caps & CL_DEVICE_SVM_ATOMICS)){
        std::cout << "Fine-grained system with atomics" << '\n';
    }


    compile_kernels();
}

void DepthEstimatorCL::compile_kernels(){
    cl::Program program(m_context,
         file_to_string("/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/kernels_cl/img_proc_test.cl"));
    program.build({m_device},"-cl-std=CL2.0");
	m_kernel_simple_copy=cl::Kernel (program, "simple_copy");
}

void DepthEstimatorCL::run_speed_test(){



    cl::Program program(m_context, source);
    program.build({m_device},"-cl-std=CL2.0");
	cl::Kernel add(program, "add");

	// Prepare input data.
    TIME_START("create_vecs");
    const int vector_size=64*100000;
	// std::vector<float> a(vector_size, 3);
	// std::vector<float> b(vector_size, 4);
    //for using CL_MEM_USE_HOST_PTR we need aligned memory as mentiones here https://software.intel.com/en-us/articles/getting-the-most-from-opencl-12-how-to-increase-performance-by-minimizing-buffer-copies-on-intel-processor-graphics
    //We must create a buffer that is aligned to a 4096 byte boundary and have a size that is a multiple of 64 bytes
    float *a_buf = (float *)aligned_alloc(4096, sizeof(float) * vector_size);
    float *b_buf = (float *)aligned_alloc(4096, sizeof(float) * vector_size);
    float *c_buf = (float *)aligned_alloc(4096, sizeof(float) * vector_size);
    for (size_t i = 0; i < vector_size; i++) {
        a_buf[i]=3;
        b_buf[i]=4;
    }
	// std::vector<float> c(vector_size);
    TIME_END("create_vecs");

    for (size_t i = 0; i < 1; i++) {

        TIME_START_CL("TOTAL_cl");
        TIME_START_CL("transfer_cl");
    	// Allocate device buffers and transfer input data to device.
    	cl::Buffer A(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, vector_size * sizeof(float), a_buf);
    	cl::Buffer B(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, vector_size * sizeof(float), b_buf);
    	// cl::Buffer C(m_context, CL_MEM_READ_WRITE, c.size() * sizeof(float));
        cl::Buffer C(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, vector_size * sizeof(float), c_buf);
        TIME_END_CL("transfer_cl");

        // TIME_START_CL("transfer_cl");
        // cl::Buffer A(m_context, CL_MEM_ALLOC_HOST_PTR, a.size() * sizeof(float), NULL );
        // float* A_ptr = (float*)m_queue.enqueueMapBuffer(A, CL_TRUE, CL_MAP_WRITE, 0, vector_size*sizeof(float));
        // memcpy(A_ptr, a.data(), vector_size*sizeof(float));
        // cl::Buffer B(m_context, CL_MEM_ALLOC_HOST_PTR, b.size() * sizeof(float), NULL );
        // float* B_ptr = (float*)m_queue.enqueueMapBuffer(B, CL_TRUE, CL_MAP_WRITE, 0, vector_size*sizeof(float));
        // memcpy(B_ptr, b.data(), vector_size*sizeof(float));
        // cl::Buffer C(m_context, CL_MEM_READ_WRITE, c.size() * sizeof(float));
        // TIME_END_CL("transfer_cl");




        TIME_START_CL("setting_args");
    	// Set kernel parameters.
    	add.setArg(0, A);
    	add.setArg(1, B);
    	add.setArg(2, C);
        TIME_END_CL("setting_args");


    	// Launch kernel on the compute device.
        TIME_START_CL("sum_cl");
    	m_queue.enqueueNDRangeKernel(add, cl::NullRange, vector_size, cl::NullRange);
        // kernel(buffer_A, buffer_B, buffer_C);
        TIME_END_CL("sum_cl");

    	// Get result back to host.
        TIME_START_CL("readbuffer_cl");
    	// m_queue.enqueueReadBuffer(C, CL_TRUE, 0, c.size() * sizeof(float), c.data());
        float * map_ptr= (float*)m_queue.enqueueMapBuffer(C, CL_TRUE, CL_MAP_READ, 0, vector_size*sizeof(float) );
        std::cout << "map_ptr is " << map_ptr[0] << '\n';
        TIME_END_CL("readbuffer_cl");
        TIME_END_CL("TOTAL_cl");

    	// Should get '3' here.
    	// std::cout << c[42] << std::endl;





        TIME_START("sum_cpu");
        for (size_t i = 0; i < vector_size; i++) {
            // c[i]=a[i]+b[i];
            // c[i]=a[i]*b[i];
            c_buf[i]=std::pow(a_buf[i],b_buf[i]);
        }
        std::cout << "c on cpu is " << c_buf[0] << '\n';
        TIME_END("sum_cpu");

    }
}

void DepthEstimatorCL::run_speed_test_img(Frame& frame){

    TIME_START_CL("run_speed_test_img");

    TIME_START_CL("add_alpha");
    cv::Mat mat;
	cvtColor(frame.rgb, mat, CV_BGR2BGRA);
	int width = mat.size().width;
    int height = mat.size().height;
    TIME_END_CL("add_alpha");

	// Create OpenCL image
    TIME_START_CL("transfer");
    cl::ImageFormat cl_img_format(CL_RGBA,CL_UNORM_INT8);
    cl::Image2D cl_img(m_context,CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, cl_img_format, width, height, 0, mat.data );
    cl::Image2D cl_img_out(m_context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, cl_img_format, width, height);
    TIME_END_CL("transfer");


    // Set the kernel arguments
    TIME_START_CL("set_args");
    m_kernel_simple_copy.setArg(0, cl_img);
    m_kernel_simple_copy.setArg(1, cl_img_out);
    TIME_END_CL("set_args");


    //run
    TIME_START_CL("run_kenel");
    m_queue.enqueueNDRangeKernel(m_kernel_simple_copy, cl::NullRange, cl::NDRange(width, height), cl::NullRange);
    TIME_END_CL("run_kenel");

    // //read result back to host
    // int size=4*width*height;
    // std::uint8_t* destination=(std::uint8_t*)malloc(size*sizeof(std::uint8_t));
    // auto origin = cl::array<cl::size_type, 3U>{0, 0, 0};
    // auto region = cl::array<cl::size_type, 3U>{width, height, 1};
    // m_queue.enqueueReadImage(cl_img_out, CL_TRUE, origin, region, 0, 0, destination);
    //
    // // for (size_t i = 0; i < size; i++) {
    // //     std::cout << "val is " << destination[i] << '\n';
    // // }
    //
    // cv::Mat wrapped(height, width, CV_8UC4, destination);
    // frame.rgb=wrapped.clone();


    //read the results back attempt 2
    TIME_START_CL("read_results");
    auto origin = cl::array<cl::size_type, 3U>{0, 0, 0};
    auto region = cl::array<cl::size_type, 3U>{width, height, 1};
    cl::size_type row_pitch, slice_pitch;
    std::uint8_t* destination = (std::uint8_t*)m_queue.enqueueMapImage(cl_img_out, CL_TRUE,CL_MAP_READ, origin, region, &row_pitch, &slice_pitch);
    TIME_END_CL("read_results");

    //put into a mat so as to view it
    TIME_START_CL("put_int_mat");
    cv::Mat wrapped(height, width, CV_8UC4, destination, row_pitch); //row pitch is the step of a opencv mat
    frame.rgb=wrapped.clone();
    TIME_END_CL("put_int_mat");

    TIME_END_CL("run_speed_test_img");

}

void DepthEstimatorCL::run_speed_test_img2(Frame& frame){

    TIME_START_CL("run_speed_test_img");

    TIME_START_CL("add_alpha");
    uchar *mat_buf;
    int size_bytes=frame.rgb.size().width*frame.rgb.size().height*4;
    //round up to nearest multiple of 64 bytes
    std::cout << "size in bytes was " << size_bytes << '\n';
    round_up_to_nearest_multiple(size_bytes,64); //allocate memory that is multiple of 64 bytes
    std::cout << "size in bytes is now " << size_bytes << '\n';
    if(frame.rgb.isContinuous()){
        TIME_START_CL("alloc");
        mat_buf = (uchar *)aligned_alloc(4096, sizeof(uchar) * size_bytes);
        TIME_END_CL("alloc");
        // TIME_START_CL("copy");
        // int idx_insert=0;
        // // #pragma omp parallel for schedule(dynamic,1) collapse(2)
        // for (size_t i = 0; i < frame.rgb.rows; i++) {
        //     cv::Vec3b* pixel = frame.rgb.ptr<cv::Vec3b>(i); // point to first pixel in row
        //     for (int j = 0; j < frame.rgb.cols; ++j){
        //         mat_buf[idx_insert]  = pixel[j][0]; //b
        //         mat_buf[idx_insert+1]= pixel[j][1]; //g
        //         mat_buf[idx_insert+2]= pixel[j][2]; //r
        //         mat_buf[idx_insert+3]= 50;
        //         idx_insert+=4;
        //     }
        // }
        // TIME_END_CL("copy");

        // copy attempt 2
        TIME_START_CL("copy");
        int idx_insert=0;
        cv::Vec3b* pixel = frame.rgb.ptr<cv::Vec3b>(0);
        for (size_t i = 0; i < frame.rgb.rows*frame.rgb.cols; i++) {
            std::memcpy(&mat_buf[idx_insert], &((*pixel)[0]),4*sizeof(uchar)); //WARNING it does copy undefined values for alpha but we don't care
            // mat_buf[idx_insert]  = (*pixel)[0]; //b
            // mat_buf[idx_insert+1]= (*pixel)[1]; //g
            // mat_buf[idx_insert+2]= (*pixel)[2]; //r
            // mat_buf[idx_insert+3]= 255;
            pixel++;
            idx_insert+=4;
        }
        TIME_END_CL("copy");


        // //copy attempt 3 (using threads)
        // TIME_START_CL("copy");
        // // int idx_insert=0;
        // int num_threads=8;
        // #pragma omp parallel for schedule(static) num_threads(num_threads)
        // for (size_t i = 0; i < frame.rgb.rows; i++) {
        //     // int threadID = omp_get_thread_num();
        //     cv::Vec3b* pixel = frame.rgb.ptr<cv::Vec3b>(i); // point to first pixel in row
        //     for (int j = 0; j < frame.rgb.cols; ++j){
        //         int idx_insert=(i*frame.rgb.cols+j)*4;
        //         std::memcpy(&mat_buf[idx_insert], &(pixel[j][0]),4*sizeof(uchar)); //WARNING it does copy undefined values for alpha but we don't care
        //         // idx_insert+=4;
        //     }
        // }
        // TIME_END_CL("copy");



    }else{
        LOG(FATAL) << "RGB image not continuous, which means the pixel iteration must be done in differnt way: look at https://docs.opencv.org/2.4/doc/tutorials/core/how_to_scan_images/how_to_scan_images.html#howtoscanimagesopencv";
    }
    int width = frame.rgb.size().width;
    int height = frame.rgb.size().height;
    TIME_END_CL("add_alpha");

	// Create OpenCL image
    TIME_START_CL("transfer");
    // cl::ImageFormat cl_img_format(CL_RGBA,CL_UNORM_INT8);
    // cl::Image2D cl_img(m_context,CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, cl_img_format, width, height, 0, mat_buf );
    // cl::Image2D cl_img_out(m_context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, cl_img_format, width, height);

    //make a buffer and then view it as image because it's faster as it's actually using the fully allocated buffer which is aligned and everything
    cl::ImageFormat cl_img_format(CL_RGBA,CL_UNORM_INT8);

    cl::Buffer cl_img_buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_bytes, mat_buf);
    cl::Image2D cl_img(m_context, cl_img_format, cl_img_buffer, width, height );

    //out using an aligned buffer makes the mapping of it way faster
    uchar* mat_out_buf = (uchar *)aligned_alloc(4096, sizeof(uchar) * size_bytes);
    cl::Buffer cl_img_out_buffer(m_context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, size_bytes, mat_out_buf);
    cl::Image2D cl_img_out(m_context, cl_img_format, cl_img_out_buffer, width, height);

    //creating an image without an aligned buffer makes the mapping of it way slower
    // cl::Image2D cl_img_out(m_context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, cl_img_format, width, height);
    TIME_END_CL("transfer");


    // Set the kernel arguments
    TIME_START_CL("set_args");
    m_kernel_simple_copy.setArg(0, cl_img);
    m_kernel_simple_copy.setArg(1, cl_img_out);
    TIME_END_CL("set_args");


    //run
    TIME_START_CL("run_kenel");
    m_queue.enqueueNDRangeKernel(m_kernel_simple_copy, cl::NullRange, cl::NDRange(width, height), cl::NullRange);
    TIME_END_CL("run_kenel");

    // //read result back to host
    // int size=4*width*height;
    // std::uint8_t* destination=(std::uint8_t*)malloc(size*sizeof(std::uint8_t));
    // auto origin = cl::array<cl::size_type, 3U>{0, 0, 0};
    // auto region = cl::array<cl::size_type, 3U>{width, height, 1};
    // m_queue.enqueueReadImage(cl_img_out, CL_TRUE, origin, region, 0, 0, destination);
    //
    // // for (size_t i = 0; i < size; i++) {
    // //     std::cout << "val is " << destination[i] << '\n';
    // // }
    //
    // cv::Mat wrapped(height, width, CV_8UC4, destination);
    // frame.rgb=wrapped.clone();


    //read the results back attempt 2
    TIME_START_CL("read_results");
    auto origin = cl::array<cl::size_type, 3U>{0, 0, 0};
    auto region = cl::array<cl::size_type, 3U>{width, height, 1};
    cl::size_type row_pitch, slice_pitch;
    std::uint8_t* destination = (std::uint8_t*)m_queue.enqueueMapImage(cl_img_out, CL_TRUE,CL_MAP_READ, origin, region, &row_pitch, &slice_pitch);
    std::cout << "row_pitch is " << row_pitch << '\n';
    TIME_END_CL("read_results");

    //put into a mat so as to view it
    TIME_START_CL("put_int_mat");
    cv::Mat wrapped(height, width, CV_8UC4, destination, row_pitch); //row pitch is the step of a opencv mat
    frame.rgb=wrapped.clone();
    TIME_END_CL("put_int_mat");

    TIME_END_CL("run_speed_test_img");


    //cleanup
    free(mat_buf);
    free(mat_out_buf);
    m_queue.enqueueUnmapMemObject(cl_img_out,destination);


}
