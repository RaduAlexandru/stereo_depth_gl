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
#include "UtilsCL.h"

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
    m_kernel_blur=cl::Kernel (program, "gaussian_blur");
    m_kernel_sobel=cl::Kernel (program, "sobel");
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

float * createBlurMask(float sigma, int * maskSizePointer) {
    int maskSize = (int)ceil(3.0f*sigma);
    float * mask = new float[(maskSize*2+1)*(maskSize*2+1)];
    float sum = 0.0f;
    for(int a = -maskSize; a < maskSize+1; a++) {
        for(int b = -maskSize; b < maskSize+1; b++) {
            float temp = exp(-((float)(a*a+b*b) / (2*sigma*sigma)));
            sum += temp;
            mask[a+maskSize+(b+maskSize)*(maskSize*2+1)] = temp;
        }
    }
    // Normalize the mask
    for(int i = 0; i < (maskSize*2+1)*(maskSize*2+1); i++)
        mask[i] = mask[i] / sum;

    *maskSizePointer = maskSize;

    return mask;
}

void DepthEstimatorCL::create_blur_mask(std::vector<float>& mask, const int sigma){
    int mask_size = (int)ceil(3.0f*sigma);
    mask.resize(mask_size);
    float sum=0.0;
    int half_way = mask_size/2;
    for (int i = -half_way; i < half_way+1; i++) {
        float temp = exp(-((float)(i*i) / (2*sigma*sigma)));
        sum += temp;
        mask[i+half_way]=temp;
    }
    // Normalize the mask
    for(int i = 0; i < mask_size; i++){
        mask[i] = mask[i] / sum;
    }

}

void DepthEstimatorCL::run_speed_test_img_3_blur(Frame& frame){

    //based on https://github.com/smistad/OpenCL-Gaussian-Blur

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

        // copy attempt 2
        TIME_START_CL("copy");
        int idx_insert=0;
        cv::Vec3b* pixel = frame.rgb.ptr<cv::Vec3b>(0);
        for (size_t i = 0; i < frame.rgb.rows*frame.rgb.cols; i++) {
            std::memcpy(&mat_buf[idx_insert], &((*pixel)[0]),4*sizeof(uchar)); //WARNING it does copy undefined values for alpha but we don't care
            pixel++;
            idx_insert+=4;
        }
        TIME_END_CL("copy");


    }else{
        LOG(FATAL) << "RGB image not continuous, which means the pixel iteration must be done in differnt way: look at https://docs.opencv.org/2.4/doc/tutorials/core/how_to_scan_images/how_to_scan_images.html#howtoscanimagesopencv";
    }
    int width = frame.rgb.size().width;
    int height = frame.rgb.size().height;
    TIME_END_CL("add_alpha");

	// Create OpenCL image
    TIME_START_CL("transfer");
    //make a buffer and then view it as image because it's faster as it's actually using the fully allocated buffer which is aligned and everything
    cl::ImageFormat cl_img_format(CL_RGBA,CL_UNORM_INT8);

    cl::Buffer cl_img_buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_bytes, mat_buf);
    cl::Image2D cl_img(m_context, cl_img_format, cl_img_buffer, width, height );

    //out using an aligned buffer makes the mapping of it way faster
    uchar* mat_out_buf = (uchar *)aligned_alloc(4096, sizeof(uchar) * size_bytes);
    cl::Buffer cl_img_out_buffer(m_context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, size_bytes, mat_out_buf);
    cl::Image2D cl_img_out(m_context, cl_img_format, cl_img_out_buffer, width, height);
    TIME_END_CL("transfer");


    //gaussian kernel
    int maskSize;
    float * mask = createBlurMask(10.0f, &maskSize);
    // Create buffer for mask and transfer it to the device
    cl::Buffer clMask = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*(maskSize*2+1)*(maskSize*2+1), mask);


    // Set the kernel arguments
    TIME_START_CL("set_args");
    m_kernel_blur.setArg(0, cl_img);
    m_kernel_blur.setArg(1, clMask);
    m_kernel_blur.setArg(2, maskSize);
    m_kernel_blur.setArg(3, cl_img_out);
    TIME_END_CL("set_args");


    //run
    TIME_START_CL("run_kenel");
    m_queue.enqueueNDRangeKernel(m_kernel_blur, cl::NullRange, cl::NDRange(width, height), cl::NullRange);
    TIME_END_CL("run_kenel");

    // //read result back to host


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

void DepthEstimatorCL::run_speed_test_img_4_sobel(Frame& frame){

    //based on https://github.com/smistad/OpenCL-Gaussian-Blur

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

        // copy attempt 2
        TIME_START_CL("copy");
        int idx_insert=0;
        cv::Vec3b* pixel = frame.rgb.ptr<cv::Vec3b>(0);
        for (size_t i = 0; i < frame.rgb.rows*frame.rgb.cols; i++) {
            std::memcpy(&mat_buf[idx_insert], &((*pixel)[0]),4*sizeof(uchar)); //WARNING it does copy undefined values for alpha but we don't care
            pixel++;
            idx_insert+=4;
        }
        TIME_END_CL("copy");


    }else{
        LOG(FATAL) << "RGB image not continuous, which means the pixel iteration must be done in differnt way: look at https://docs.opencv.org/2.4/doc/tutorials/core/how_to_scan_images/how_to_scan_images.html#howtoscanimagesopencv";
    }
    int width = frame.rgb.size().width;
    int height = frame.rgb.size().height;
    TIME_END_CL("add_alpha");

	// Create OpenCL image
    TIME_START_CL("transfer");
    //make a buffer and then view it as image because it's faster as it's actually using the fully allocated buffer which is aligned and everything
    cl::ImageFormat cl_img_format(CL_RGBA,CL_UNORM_INT8);

    cl::Buffer cl_img_buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_bytes, mat_buf);
    cl::Image2D cl_img(m_context, cl_img_format, cl_img_buffer, width, height );

    //out using an aligned buffer makes the mapping of it way faster
    uchar* mat_out_buf = (uchar *)aligned_alloc(4096, sizeof(uchar) * size_bytes);
    cl::Buffer cl_img_out_buffer(m_context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, size_bytes, mat_out_buf);
    cl::Image2D cl_img_out(m_context, cl_img_format, cl_img_out_buffer, width, height);
    TIME_END_CL("transfer");




    // Set the kernel arguments
    TIME_START_CL("set_args");
    m_kernel_sobel.setArg(0, cl_img);
    m_kernel_sobel.setArg(1, cl_img_out);
    TIME_END_CL("set_args");


    //run
    TIME_START_CL("run_kenel");
    m_queue.enqueueNDRangeKernel(m_kernel_sobel, cl::NullRange, cl::NDRange(width, height), cl::NullRange);
    TIME_END_CL("run_kenel");

    // //read result back to host


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

void DepthEstimatorCL::run_speed_test_img_4_sobel_gray(Frame& frame){

    //based on https://github.com/smistad/OpenCL-Gaussian-Blur

    TIME_START_CL("run_speed_test_img");

    TIME_START_CL("add_alpha");
    uchar *mat_buf;
    cv::Mat img_gray;
    cv::cvtColor(frame.rgb, img_gray, CV_BGR2GRAY);
    int size_bytes=img_gray.cols*img_gray.rows*1;
    //round up to nearest multiple of 64 bytes
    std::cout << "size in bytes was " << size_bytes << '\n';
    round_up_to_nearest_multiple(size_bytes,64); //allocate memory that is multiple of 64 bytes
    std::cout << "size in bytes is now " << size_bytes << '\n';
    if(img_gray.isContinuous()){
        TIME_START_CL("alloc");
        mat_buf = (uchar *)aligned_alloc(4096, sizeof(uchar) * size_bytes);
        TIME_END_CL("alloc");

        // copy attempt 2
        TIME_START_CL("copy");
        int idx_insert=0;
        uchar* pixel = img_gray.ptr<uchar>(0);
        for (size_t i = 0; i < img_gray.rows*img_gray.cols; i++) {
            mat_buf[idx_insert]= *pixel;
            pixel++;
            idx_insert+=1;
        }

        TIME_END_CL("copy");


    }else{
        LOG(FATAL) << "RGB image not continuous, which means the pixel iteration must be done in differnt way: look at https://docs.opencv.org/2.4/doc/tutorials/core/how_to_scan_images/how_to_scan_images.html#howtoscanimagesopencv";
    }
    int width = img_gray.cols;
    int height = img_gray.rows;
    TIME_END_CL("add_alpha");

	// Create OpenCL image
    TIME_START_CL("transfer");
    //make a buffer and then view it as image because it's faster as it's actually using the fully allocated buffer which is aligned and everything
    cl::ImageFormat cl_img_format(CL_R,CL_UNORM_INT8);

    cl::Buffer cl_img_buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_bytes, mat_buf);
    cl::Image2D cl_img(m_context, cl_img_format, cl_img_buffer, width, height );

    //out using an aligned buffer makes the mapping of it way faster
    uchar* mat_out_buf = (uchar *)aligned_alloc(4096, sizeof(uchar) * size_bytes);
    cl::Buffer cl_img_out_buffer(m_context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, size_bytes, mat_out_buf);
    cl::Image2D cl_img_out(m_context, cl_img_format, cl_img_out_buffer, width, height);
    TIME_END_CL("transfer");




    // Set the kernel arguments
    TIME_START_CL("set_args");
    m_kernel_sobel.setArg(0, cl_img);
    m_kernel_sobel.setArg(1, cl_img_out);
    TIME_END_CL("set_args");


    //run
    TIME_START_CL("run_kenel");
    m_queue.enqueueNDRangeKernel(m_kernel_sobel, cl::NullRange, cl::NDRange(width, height), cl::NullRange);
    TIME_END_CL("run_kenel");

    // //read result back to host


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
    cv::Mat wrapped(height, width, CV_8UC1, destination, row_pitch); //row pitch is the step of a opencv mat
    frame.rgb=wrapped.clone();
    TIME_END_CL("put_int_mat");

    TIME_END_CL("run_speed_test_img");


    //cleanup
    free(mat_buf);
    free(mat_out_buf);
    m_queue.enqueueUnmapMemObject(cl_img_out,destination);


}

void DepthEstimatorCL::run_speed_test_img_4_blur_gray(Frame& frame){

    //based on https://github.com/smistad/OpenCL-Gaussian-Blur
    std::vector<float> gaus_mask;
    create_blur_mask(gaus_mask,3);

    TIME_START_CL("run_speed_test_img");

    TIME_START_CL("add_alpha");
    uchar *mat_buf;
    cv::Mat img_gray;
    cv::cvtColor(frame.rgb, img_gray, CV_BGR2GRAY);
    int size_bytes=img_gray.cols*img_gray.rows*1;
    //round up to nearest multiple of 64 bytes
    std::cout << "size in bytes was " << size_bytes << '\n';
    round_up_to_nearest_multiple(size_bytes,64); //allocate memory that is multiple of 64 bytes
    std::cout << "size in bytes is now " << size_bytes << '\n';
    if(img_gray.isContinuous()){
        TIME_START_CL("alloc");
        mat_buf = (uchar *)aligned_alloc(4096, sizeof(uchar) * size_bytes);
        TIME_END_CL("alloc");

        // copy attempt 2
        TIME_START_CL("copy");
        int idx_insert=0;
        uchar* pixel = img_gray.ptr<uchar>(0);
        for (size_t i = 0; i < img_gray.rows*img_gray.cols; i++) {
            mat_buf[idx_insert]= *pixel;
            pixel++;
            idx_insert+=1;
        }

        TIME_END_CL("copy");


    }else{
        LOG(FATAL) << "RGB image not continuous, which means the pixel iteration must be done in differnt way: look at https://docs.opencv.org/2.4/doc/tutorials/core/how_to_scan_images/how_to_scan_images.html#howtoscanimagesopencv";
    }
    int width = img_gray.cols;
    int height = img_gray.rows;
    TIME_END_CL("add_alpha");

	// Create OpenCL image
    TIME_START_CL("transfer");
    //make a buffer and then view it as image because it's faster as it's actually using the fully allocated buffer which is aligned and everything
    cl::ImageFormat cl_img_format(CL_R,CL_UNORM_INT8);

    cl::Buffer cl_img_buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_bytes, mat_buf);
    cl::Image2D cl_img(m_context, cl_img_format, cl_img_buffer, width, height );

    //out using an aligned buffer makes the mapping of it way faster
    uchar* mat_out_buf = (uchar *)aligned_alloc(4096, sizeof(uchar) * size_bytes);
    cl::Buffer cl_img_out_buffer(m_context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, size_bytes, mat_out_buf);
    cl::Image2D cl_img_out(m_context, cl_img_format, cl_img_out_buffer, width, height);
    TIME_END_CL("transfer");




    // Set the kernel arguments
    TIME_START_CL("set_args");
    m_kernel_sobel.setArg(0, cl_img);
    m_kernel_sobel.setArg(1, cl_img_out);
    TIME_END_CL("set_args");


    //run
    TIME_START_CL("run_kenel");
    m_queue.enqueueNDRangeKernel(m_kernel_sobel, cl::NullRange, cl::NDRange(width, height), cl::NullRange);
    TIME_END_CL("run_kenel");

    // //read result back to host


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
    cv::Mat wrapped(height, width, CV_8UC1, destination, row_pitch); //row pitch is the step of a opencv mat
    frame.rgb=wrapped.clone();
    TIME_END_CL("put_int_mat");

    TIME_END_CL("run_speed_test_img");


    //cleanup
    free(mat_buf);
    free(mat_out_buf);
    m_queue.enqueueUnmapMemObject(cl_img_out,destination);

}

//uses the class from ym cl_lib which does host alocation and free by itself
void DepthEstimatorCL::run_speed_test_img_4_blur_gray_safe(Frame& frame){

    //based on https://github.com/smistad/OpenCL-Gaussian-Blur
    std::vector<float> gaus_mask;
    create_blur_mask(gaus_mask,3);

    TIME_START_CL("run_speed_test_img");

    cv::Mat img_gray;
    cv::cvtColor(frame.rgb, img_gray, CV_BGR2GRAY);
    int width=img_gray.cols;
    int height=img_gray.rows;

    TIME_START_CL("create_cl_img");
    cl::ImageFormat cl_img_format(CL_R,CL_UNORM_INT8);
    cl::Image2DSafe cl_img=cv_mat2cl_img(img_gray, cl_img_format, CL_MEM_READ_ONLY, m_context);
    TIME_END_CL("create_cl_img");

    //make an image for out
    int size_bytes=img_gray.step[0] * img_gray.rows;
    cl::Image2DSafe cl_img_out(m_context, cl_img_format, CL_MEM_WRITE_ONLY, img_gray.cols, img_gray.rows, size_bytes);


    // Set the kernel arguments
    TIME_START_CL("set_args");
    m_kernel_sobel.setArg(0, cl_img.get_img());
    m_kernel_sobel.setArg(1, cl_img_out.get_img());
    TIME_END_CL("set_args");


    //run
    TIME_START_CL("run_kenel");
    m_queue.enqueueNDRangeKernel(m_kernel_sobel, cl::NullRange, cl::NDRange(width, height), cl::NullRange);
    TIME_END_CL("run_kenel");

    // //read result back to host


    //read the results back attempt 2
    auto origin = cl::array<cl::size_type, 3U>{0, 0, 0};
    auto region = cl::array<cl::size_type, 3U>{width, height, 1};
    cl::size_type row_pitch, slice_pitch;
    std::uint8_t* destination = (std::uint8_t*)m_queue.enqueueMapImage(cl_img_out.get_img(), CL_TRUE,CL_MAP_READ, origin, region, &row_pitch, &slice_pitch);
    std::cout << "row_pitch is " << row_pitch << '\n';
    TIME_END_CL("read_results");

    //put into a mat so as to view it
    TIME_START_CL("put_int_mat");
    cv::Mat wrapped(height, width, CV_8UC1, destination, row_pitch); //row pitch is the step of a opencv mat
    frame.rgb=wrapped.clone();
    TIME_END_CL("put_int_mat");

    TIME_END_CL("run_speed_test_img");


    //cleanup
    m_queue.enqueueUnmapMemObject(cl_img_out.get_img(),destination);

}

void DepthEstimatorCL::gaussian_blur(cl::Image2D& dest_img, const cl::Image2D& src_img, const int sigma){

    //get the size in bytes
    int width = src_img.getImageInfo<CL_IMAGE_WIDTH>();
    int height = src_img.getImageInfo<CL_IMAGE_HEIGHT>();
    int pitch =src_img.getImageInfo<CL_IMAGE_ROW_PITCH>(); //Return size in bytes of a row in the img
    int size_bytes=height*pitch;

    cl_image_format format =src_img.getImageInfo<CL_IMAGE_FORMAT>();
    cl::ImageFormat cl_img_format{format.image_channel_order, format.image_channel_data_type};

    //create an auxiliary buffer to store the values after blurx
    uchar* aux_buf = (uchar *)aligned_alloc(4096, sizeof(uchar) * size_bytes);
    cl::Buffer cl_buffer_aux(m_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, size_bytes, aux_buf);
    cl::Image2D cl_img_aux(m_context, cl_img_format, cl_buffer_aux, width, height);

    //allocate dest img only id the reference is not the same as the src


    free(aux_buf);
}

void DepthEstimatorCL::compute_depth(const Frame& frame){
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



}
