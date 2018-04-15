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
#include "cv_interpolation.h"

//Libigl
#include <igl/opengl/glfw/Viewer.h>

//cv
#include <cv_bridge/cv_bridge.h>

//loguru
#include <loguru.hpp>

//omp
#include <omp.h>

//boost
#include <boost/math/distributions/normal.hpp>





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
    std::string pattern_filepath="/media/alex/Data/Master/SHK/c_ws/src/stereo_depth_cl/data/pattern_1.png";
    m_pattern.init_pattern(pattern_filepath);

    //sanity check the pattern
    std::cout << "pattern has nr of points " << m_pattern.get_nr_points() << '\n';
    for (size_t i = 0; i < m_pattern.get_nr_points(); i++) {
        std::cout << "offset for i " << i << " is " << m_pattern.get_offset(i).transpose() << '\n';
    }
}

//needed so that forward declarations work
DepthEstimatorCL::~DepthEstimatorCL(){
}

void DepthEstimatorCL::init_opencl(){
    VLOG(1) << "init opencl";
    std::cout << "init opencl" << '\n';


	// Get list of OpenCL platforms.
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.empty()) {
	    std::cerr << "No platforms found." << std::endl;
	    return;
	}

    //get the platform that has opencl (in order to avoid getting the one with CUDA)
    int idx_platform_chosen=-1;
    for (size_t i = 0; i < platforms.size(); i++) {
        std::string name=platforms[i].getInfo<CL_PLATFORM_NAME>();
        // std::cout << "name of device is " << name << '\n';
        if(name.find("OpenCL")!=std::string::npos){
            idx_platform_chosen=i;
        }
    }
    if (idx_platform_chosen==-1) {
	    std::cerr << "OpenCL platforms not found." << std::endl;
	    return;
	}



	// Get the first Intel GPU device
    std::vector<cl::Device> devices_available;
    platforms[idx_platform_chosen].getDevices(CL_DEVICE_TYPE_GPU, &devices_available);
    std::cout << "found nr of devices " << devices_available.size() << '\n';
    bool found_intel_device=false;
    for (size_t i = 0; i < devices_available.size(); i++) {
        std::string name=devices_available[i].getInfo<CL_DEVICE_NAME>();
        // std::cout << "name of device is " << name << '\n';
        if(name.find("HD Graphics")!=std::string::npos){
            m_device=devices_available[i];
            found_intel_device=true;
        }
    }
    if(!found_intel_device){
        LOG_F(FATAL, "No Intel device found");
    }

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
    m_kernel_blurx=cl::Kernel (program, "blurx");
    m_kernel_blury=cl::Kernel (program, "blury");

    m_kernel_blurx_fast=cl::Kernel (program, "blurx_fast");
    m_kernel_blury_fast=cl::Kernel (program, "blury_fast");
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
    int maskSize = (int)std::ceil(3.0f*sigma);
    float * mask = new float[(maskSize*2+1)*(maskSize*2+1)];
    float sum = 0.0f;
    for(int a = -maskSize; a < maskSize+1; a++) {
        for(int b = -maskSize; b < maskSize+1; b++) {
            float temp = std::exp(-((float)(a*a+b*b) / (2*sigma*sigma)));
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
    int mask_size = (int)std::ceil(3.0f*sigma);
    mask.resize(mask_size);
    float sum=0.0;
    int half_way = mask_size/2;
    for (int i = -half_way; i < half_way+1; i++) {
        float temp = std::exp(-((float)(i*i) / (2*sigma*sigma)));
        sum += temp;
        mask[i+half_way]=temp;
    }
    // Normalize the mask
    for(int i = 0; i < mask_size; i++){
        mask[i] = mask[i] / sum;
    }

    // for (size_t i = 0; i < mask_size; i++) {
    //     std::cout << "mask is " << mask[i] << '\n';
    // }
}

void DepthEstimatorCL::create_half_blur_mask(std::vector<float>& mask, const int sigma){
    int mask_size = (int)std::ceil(3.0f*sigma)/2 +1;
    mask.resize(mask_size);
    float sum=0.0;
    for (int i = 0; i < mask_size; i++) {
        float temp = std::exp(-((float)(i*i) / (2*sigma*sigma)));
        sum += temp;
        if(i!=0) sum+=temp; //(the sum is not complete yet because the mask is only half) so we add another time to make up for the other side of the gaussian
        mask[i]=temp;
    }
    // Normalize the mask
    for(int i = 0; i < mask_size; i++){
        mask[i] = mask[i] / sum;
    }

    // for (size_t i = 0; i < mask_size; i++) {
    //     std::cout << "mask is " << mask[i] << '\n';
    // }


}

void DepthEstimatorCL::optimize_blur_for_gpu_sampling(std::vector<float>&gaus_mask, std::vector<float>& gaus_offsets){
    gaus_offsets.resize(gaus_mask.size());
    for (size_t i = 0; i < gaus_mask.size(); i++) {
        gaus_offsets[i]=i;
    }

    std::vector<float> gaus_mask_optimized(1+gaus_mask.size()/2); //1 because the middle points stays the same, for the rest we need N/2 less texture fetches
    std::vector<float> gaus_offsets_optimized(1+gaus_mask.size()/2);

    //optimize them // http://roxlu.com/2014/045/fast-opengl-blur-shader and // http://rastergrid.com/blog/2010/09/efficient-gaussian-blur-with-linear-sampling/
    gaus_mask_optimized[0]=gaus_mask[0];
    gaus_offsets_optimized[0]=0.0;
    int idx_original_gaus=1;
    for (size_t i = 1; i < gaus_mask_optimized.size(); i++) {  //go from 2 by 2 because a texture fetch actually gets us 2 pixels
        gaus_mask_optimized[i] = gaus_mask[idx_original_gaus]+gaus_mask[idx_original_gaus+1];
        gaus_offsets_optimized[i] = (gaus_mask[idx_original_gaus]*gaus_offsets[idx_original_gaus] +
                                     gaus_mask[idx_original_gaus+1]*gaus_offsets[idx_original_gaus+1] ) / gaus_mask_optimized[i];
        idx_original_gaus+=2;
    }

    gaus_mask=gaus_mask_optimized;
    gaus_offsets=gaus_offsets_optimized;

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

//dest_img and src_img can be the same reference but both need to have allocated buffers
void DepthEstimatorCL::gaussian_blur(cl::Image2DSafe& dest_img, const cl::Image2DSafe& src_img, const int sigma){


    // //works
    // std::vector<float> gaus_mask;
    // create_blur_mask(gaus_mask,sigma);
    // // Create buffer for mask and transfer it to the device
    // cl::Buffer gaus_mask_cl = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*gaus_mask.size(), gaus_mask.data());
    //
    // //attempt 3
    // TIME_START_CL("make_alloc_host");
    // int size_bytes=src_img.get_size_bytes();
    // cl::Buffer cl_buffer(m_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size_bytes);
    // cl::Image2D cl_img_aux(m_context, src_img.get_format(), cl_buffer, src_img.get_width(), src_img.get_height());
    // TIME_END_CL("make_alloc_host");
    //
    // //blurx
    // m_kernel_blurx.setArg(0, src_img.get_img());
    // m_kernel_blurx.setArg(1, gaus_mask_cl);
    // m_kernel_blurx.setArg(2, (int)gaus_mask.size());
    // m_kernel_blurx.setArg(3, cl_img_aux);
    // m_queue.enqueueNDRangeKernel(m_kernel_blurx, cl::NullRange, cl::NDRange(src_img.get_width(), src_img.get_height()), cl::NullRange);
    //
    // //blury
    // m_kernel_blury.setArg(0, cl_img_aux);
    // m_kernel_blury.setArg(1, gaus_mask_cl);
    // m_kernel_blury.setArg(2, (int)gaus_mask.size());
    // m_kernel_blury.setArg(3, dest_img.get_img());
    // m_queue.enqueueNDRangeKernel(m_kernel_blury, cl::NullRange, cl::NDRange(src_img.get_width(), src_img.get_height()), cl::NullRange);




    // // attempt 2
    // std::vector<float> gaus_mask;
    // std::vector<float> gaus_offsets;
    // create_half_blur_mask(gaus_mask,sigma);
    // optimize_blur_for_gpu_sampling(gaus_mask,gaus_offsets); //offset in order to have the sampler perfor bilinear interpolation in shader
    // cl::Buffer gaus_mask_cl = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*gaus_mask.size(), gaus_mask.data());
    // cl::Buffer gaus_offsets_cl = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*gaus_offsets.size(), gaus_offsets.data());
    //
    // //attempt 3
    // TIME_START_CL("make_alloc_host");
    // int size_bytes=src_img.get_size_bytes();
    // cl::Buffer cl_buffer(m_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size_bytes);
    // cl::Image2D cl_img_aux(m_context, src_img.get_format(), cl_buffer, src_img.get_width(), src_img.get_height());
    // TIME_END_CL("make_alloc_host");
    //
    // //blurx
    // m_kernel_blurx_fast.setArg(0, src_img.get_img());
    // m_kernel_blurx_fast.setArg(1, gaus_mask_cl);
    // m_kernel_blurx_fast.setArg(2, (int)gaus_mask.size());
    // m_kernel_blurx_fast.setArg(3, cl_img_aux);
    // m_queue.enqueueNDRangeKernel(m_kernel_blurx_fast, cl::NullRange, cl::NDRange(src_img.get_width(), src_img.get_height()), cl::NullRange);
    //
    // //blury
    // m_kernel_blury_fast.setArg(0, cl_img_aux);
    // m_kernel_blury_fast.setArg(1, gaus_mask_cl);
    // m_kernel_blury_fast.setArg(2, (int)gaus_mask.size());
    // m_kernel_blury_fast.setArg(3, dest_img.get_img());
    // m_queue.enqueueNDRangeKernel(m_kernel_blury_fast, cl::NullRange, cl::NDRange(src_img.get_width(), src_img.get_height()), cl::NullRange);




    //attempt 3 wtf is happening with attemp 2
    std::vector<float> gaus_mask;
    std::vector<float> gaus_offsets;
    create_half_blur_mask(gaus_mask,sigma);
    optimize_blur_for_gpu_sampling(gaus_mask,gaus_offsets); //offset in order to have the sampler perfor bilinear interpolation in shader
    cl::Buffer gaus_mask_cl = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*gaus_mask.size(), gaus_mask.data());
    cl::Buffer gaus_offsets_cl = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*gaus_offsets.size(), gaus_offsets.data());

    //attempt 3
    TIME_START_CL("make_alloc_host");
    int size_bytes=src_img.get_size_bytes();
    cl::Buffer cl_buffer(m_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size_bytes);
    cl::Image2D cl_img_aux(m_context, src_img.get_format(), cl_buffer, src_img.get_width(), src_img.get_height());
    TIME_END_CL("make_alloc_host");

    //blurx
    m_kernel_blurx_fast.setArg(0, src_img.get_img());
    m_kernel_blurx_fast.setArg(1, gaus_mask_cl);
    m_kernel_blurx_fast.setArg(2, (int)gaus_mask.size());
    m_kernel_blurx_fast.setArg(3, gaus_offsets_cl);
    m_kernel_blurx_fast.setArg(4, cl_img_aux);
    m_queue.enqueueNDRangeKernel(m_kernel_blurx_fast, cl::NullRange, cl::NDRange(src_img.get_width(), src_img.get_height()), cl::NullRange);

    //blury
    m_kernel_blury_fast.setArg(0, cl_img_aux);
    m_kernel_blury_fast.setArg(1, gaus_mask_cl);
    m_kernel_blury_fast.setArg(2, (int)gaus_mask.size());
    m_kernel_blury_fast.setArg(3, gaus_offsets_cl);
    m_kernel_blury_fast.setArg(4, dest_img.get_img());
    m_queue.enqueueNDRangeKernel(m_kernel_blury_fast, cl::NullRange, cl::NDRange(src_img.get_width(), src_img.get_height()), cl::NullRange);

}

void DepthEstimatorCL::compute_depth(Frame& frame){

    TIME_START_CL("compute_depth");

    TIME_START_CL("postprocess_img");
    cv::Mat img_gray;
    cv::cvtColor(frame.rgb, img_gray, CV_BGR2GRAY);
    cv::resize(img_gray, img_gray, cv::Size(1280,760));
    int width=img_gray.cols;
    int height=img_gray.rows;
    TIME_END_CL("postprocess_img");

    TIME_START_CL("create_cl_img");
    cl::ImageFormat cl_img_format(CL_R,CL_UNORM_INT8);
    cl::Image2DSafe cl_img=cv_mat2cl_img(img_gray, cl_img_format, CL_MEM_READ_WRITE, m_context);
    TIME_END_CL("create_cl_img");


    TIME_START_CL("gaussian_blur");
    gaussian_blur(cl_img,cl_img,1); //sigma 15 at around 60 ms
    TIME_END_CL("gaussian_blur");

    TIME_START_CL("sobel");
    cl::Image2DSafe cl_sob=cl_img_like(cl_img);
    // Set the kernel arguments
    m_kernel_sobel.setArg(0, cl_img.get_img());
    m_kernel_sobel.setArg(1, cl_sob.get_img());
    m_queue.enqueueNDRangeKernel(m_kernel_sobel, cl::NullRange, cl::NDRange(width, height), cl::NullRange);
    TIME_END_CL("sobel");







    //read the results back attempt 2
    TIME_START_CL("map_image");
    gaussian_blur(cl_img,cl_img,1); //si
    auto origin = cl::array<cl::size_type, 3U>{0, 0, 0};
    auto region = cl::array<cl::size_type, 3U>{width, height, 1};
    cl::size_type row_pitch, slice_pitch;
    std::uint8_t* destination = (std::uint8_t*)m_queue.enqueueMapImage(cl_sob.get_img(), CL_TRUE,CL_MAP_READ, origin, region, &row_pitch, &slice_pitch);
    std::cout << "row_pitch is " << row_pitch << '\n';
    TIME_END_CL("map_image");

    //put into a mat so as to view it
    TIME_START_CL("put_int_mat");
    std::cout << "11111" << '\n';
    std::cout << "height and width is " << height << " " << width << '\n';
    cv::Mat wrapped(height, width, CV_8UC1, destination, row_pitch); //row pitch is the step of a opencv mat
    std::cout << "222222" << '\n';
    frame.rgb=wrapped.clone();
    std::cout << "done cloning" << '\n';
    TIME_END_CL("put_int_mat");



    //cleanup
    TIME_START_CL("cleanup");
    std::cout << "33333" << '\n';
    m_queue.enqueueUnmapMemObject(cl_sob.get_img(),destination);
    TIME_END_CL("cleanup");



    // TIME_START("opencv_gaus");
    // cv::GaussianBlur( img_gray, frame.rgb, cv::Size( 45, 45 ), 0, 0);
    //
    // TIME_END("opencv_gaus")


    TIME_END_CL("compute_depth");

}






std::vector<Frame> DepthEstimatorCL::loadDataFromICLNUIM ( const std::string & dataset_path, const int num_images_to_read){
   std::vector< Frame > frames;
   std::string filename_img = dataset_path + "/associations.txt";
   std::string filename_gt = dataset_path + "/livingRoom2.gt.freiburg";

   //K is from here https://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html
   Eigen::Matrix3d K;
   K.setZero();
   K(0,0)=481.2; //fx
   K(1,1)=-480; //fy
   K(0,2)=319.5; // cx
   K(1,2)=239.5; //cy
   K(2,2)=1.0;

   std::ifstream imageFile ( filename_img, std::ifstream::in );
   std::ifstream grtruFile ( filename_gt, std::ifstream::in );

   int imagesRead = 0;
   for ( imagesRead = 0; imageFile.good() && grtruFile.good() && imagesRead <= num_images_to_read ; ++imagesRead ){
      std::string depthFileName, colorFileName;
      int idc, idd, idg;
      double tsc, tx, ty, tz, qx, qy, qz, qw;
      imageFile >> idd >> depthFileName >> idc >> colorFileName;

      if ( idd == 0 )
          continue;
      grtruFile >> idg >> tx >> ty >> tz >> qx >> qy >> qz >> qw;

      if ( idc != idd || idc != idg ){
          std::cerr << "Error during reading... not correct anymore!" << std::endl;
          break;
      }

      if ( ! depthFileName.empty() ){
         Eigen::Affine3d pose_wc = Eigen::Affine3d::Identity();
         pose_wc.translation() << tx,ty,tz;
         pose_wc.linear() = Eigen::Quaterniond(qw,qx,qy,qz).toRotationMatrix();
         Eigen::Affine3d pose_cw = pose_wc.inverse();

         cv::Mat rgb_cv=cv::imread(dataset_path + "/" + colorFileName);
         cv::Mat depth_cv=cv::imread(dataset_path + "/" + depthFileName, CV_LOAD_IMAGE_UNCHANGED);
         depth_cv.convertTo ( depth_cv, CV_32F, 1./5000. ); //ICLNUIM stores theis weird units so we transform to meters


         Frame cur_frame;
         cur_frame.rgb=rgb_cv;
         cv::cvtColor ( cur_frame.rgb, cur_frame.gray, CV_BGR2GRAY );
         cur_frame.depth=depth_cv;
         cur_frame.tf_cam_world=pose_cw;
         cur_frame.gray.convertTo ( cur_frame.gray, CV_32F );
         cur_frame.K=K;
         cur_frame.frame_id=imagesRead;

         frames.push_back(cur_frame);
         VLOG(1) << "read img " << imagesRead << " " << colorFileName;
      }
   }
   std::cout << "read " << imagesRead << " images. (" << frames.size() <<", " << ")" << std::endl;
   return frames;
}

Mesh DepthEstimatorCL::compute_depth2(Frame& frame){
    //read images from ICL_NUIM
    //calculate pyramid and gradients
    //grab the first frame and calculate inmature points for it

    //for each new frame afterwards we grab pass it to open and update the inmature points depth

    //----------------------------------------------------------------------------------------------------
    std::string dataset_path="/media/alex/Data/Master/Thesis/data/ICL_NUIM/living_room_traj2_frei_png";
    int num_images_to_read=60;
    bool use_modified=false;
    std::vector<Frame> frames=loadDataFromICLNUIM(dataset_path, num_images_to_read);
    std::cout << "frames size is " << frames.size() << "\n";


    //TODO undistort images

    TIME_START_CL("compute_depth");
    //create inmature points for the first frame
    std::vector<ImmaturePoint> immature_points;
    immature_points=create_immature_points(frames[0]);



    for (size_t i = 1; i < frames.size(); i++) {
        //compute the matrices between the two frames
        Eigen::Affine3d tf_cur_host = frames[i].tf_cam_world * frames[0].tf_cam_world.inverse();
        Eigen::Matrix3d KRKi_cr = frames[i].K * tf_cur_host.linear() * frames[0].K.inverse();
        Eigen::Vector3d Kt_cr = frames[i].K * tf_cur_host.translation();
        Eigen::Vector2d affine_cr = estimate_affine( immature_points, frames[i], KRKi_cr, Kt_cr);

        update_immature_points(immature_points, frames[i], tf_cur_host, KRKi_cr, Kt_cr, affine_cr );
    }

    TIME_END_CL("compute_depth");


    Mesh mesh=create_mesh(immature_points,frames); //creates a mesh from the position of the points and their depth
    return mesh;

}

Eigen::Vector2d DepthEstimatorCL::estimate_affine(std::vector<ImmaturePoint>& immature_points, const Frame&  cur_frame, const Eigen::Matrix3d& KRKi_cr, const Eigen::Vector3d& Kt_cr)
{
    ceres::Problem problem;
    ceres::LossFunction * loss_function = new ceres::HuberLoss(1);
    double scaleA = 1;
    double offsetB = 0;

    TIME_START("creating ceres problem");
    for ( int i = 0; i < immature_points.size(); ++i )
    {
        ImmaturePoint& point = immature_points[i];
        if ( i % 100 != 0 )
            continue;

//
        //get colors at the current frame
        float color_cur_frame[MAX_RES_PER_POINT];
        float color_host_frame[MAX_RES_PER_POINT];


        if ( 1.0/point.gt_depth > 0 ) {

            const Eigen::Vector3d p = KRKi_cr * Eigen::Vector3d(point.u,point.v,1) + Kt_cr*  (1.0/point.gt_depth);
            point.kp_GT = p.hnormalized();

//            //debug
//            if(point.u==258 && point.v==447){
//                std::cout << "point u,v gt_idepth is " << point.u << " " << point.v << " " << (1.0/point.gt_depth) << "\n";
//                std::cout << "kp_GT is " << point.kp_GT << '\n';
//            }

            if ( point.kp_GT(0) > 4 && point.kp_GT(0) < cur_frame.gray.cols-4 && point.kp_GT(1) > 3 && point.kp_GT(1) < cur_frame.gray.rows-4 ) {

                Pattern pattern_rot=m_pattern.get_rotated_pattern( KRKi_cr.topLeftCorner<2,2>() );

                for(int idx=0;idx<m_pattern.get_nr_points();++idx) {
                    Eigen::Vector2d offset=pattern_rot.get_offset(idx);

                    color_cur_frame[idx]=texture_interpolate(cur_frame.gray, point.kp_GT(0)+offset(0), point.kp_GT(1)+offset(1) , InterpolationType::LINEAR);
                    color_host_frame[idx]=point.color[idx];

//                    //debug
//                    if(point.u==258 && point.v==447){
//                        std::cout << "offset pattern_rot is " << offset(0) << " " << offset(1) << "\n";
//                        std::cout << "color_host_frame is " << point.color[idx] << "\n";
//                        std::cout << "color_cur_frame is  " << color_cur_frame[idx] << "\n";
//                    }
                }
            }
        }


        for ( int i = 0; i < m_pattern.get_nr_points(); ++i) {
            if ( !std::isfinite(color_host_frame[i]) || ! std::isfinite(color_cur_frame[i]) )
                continue;
            if ( color_host_frame[i] <= 0 || color_host_frame[i] >= 255 || color_cur_frame[i] <= 0 || color_cur_frame[i] >= 255  )
                continue;
            ceres::CostFunction * cost_function = AffineAutoDiffCostFunctor::Create( color_cur_frame[i], color_host_frame[i] );
            problem.AddResidualBlock( cost_function, loss_function, &scaleA, & offsetB );
        }
    }
    TIME_END("creating ceres problem");
    ceres::Solver::Options solver_options;
    //solver_options.linear_solver_type = ceres::DENSE_QR;//DENSE_SCHUR;//QR;
    solver_options.minimizer_progress_to_stdout = false;
    solver_options.max_num_iterations = 1000;
    solver_options.function_tolerance = 1e-6;
    solver_options.num_threads = 8;
    ceres::Solver::Summary summary;
    ceres::Solve( solver_options, & problem, & summary );
    //std::cout << summary.FullReport() << std::endl;
    std::cout << "scale= " << scaleA << " offset= "<< offsetB << std::endl;
    return Eigen::Vector2d ( scaleA, offsetB );
}

float DepthEstimatorCL::texture_interpolate ( const cv::Mat& img, const float x, const float y , const InterpolationType type){
    //sample only from a cv mat that is of type float with 1 channel
    if(type2string(img.type())!="32FC1"){
        LOG(FATAL) << "trying to use texture inerpolate on an image that is not float valued with 1 channel. Image is of type " <<
        type2string(img.type());
    }

    //Dumb nearest interpolation
//    int clamped_y=clamp((int)y,0,img.rows);
//    int clamped_x=clamp((int)x,0,img.cols);
//    float val=img.at<float>(clamped_y,clamped_x);

    //from oepncv https://github.com/opencv/opencv/blob/master/modules/cudawarping/test/interpolation.hpp
    if(type==InterpolationType::NEAREST){
        return NearestInterpolator<float>::getValue(img,y,x,0,cv::BORDER_REPLICATE);
    }else if(type==InterpolationType::LINEAR){
        return LinearInterpolator<float>::getValue(img,y,x,0,cv::BORDER_REPLICATE);
    }else if(type==InterpolationType::CUBIC){
        return CubicInterpolator<float>::getValue(img,y,x,0,cv::BORDER_REPLICATE);
    }

}

std::vector<ImmaturePoint> DepthEstimatorCL::create_immature_points (const Frame& frame){


    //create all of the pixels as inmature points
    // std::vector<ImmaturePoint> immature_points;
    // for (size_t i = 0; i < frame.gray.rows; i++) {
    //     for (size_t j = 0; j < frame.gray.cols; j++) {
    //         ImmaturePoint point;
    //         point.u=j;
    //         point.v=i;
    //         point.depth=frame.depth.at<float>(i,j);
    //         immature_points.push_back(point);
    //     }
    // }

    //make the sobel in x and y because we need to to calculate the hessian in order to select the immature point
    TIME_START_CL("sobel_host_frame");
    cv::Mat grad_x, grad_y;
    cv::Scharr( frame.gray, grad_x, CV_32F, 1, 0);
    cv::Scharr( frame.gray, grad_y, CV_32F, 0, 1);
    TIME_END_CL("sobel_host_frame");

    TIME_START_CL("hessian_host_frame");
    std::vector<ImmaturePoint> immature_points;
    immature_points.reserve(200000);
    for (size_t i = 10; i < frame.gray.rows-10; i++) {  //--------Do not look around the borders to avoid pattern accesing outside img
        for (size_t j = 10; j < frame.gray.cols-10; j++) {

            //check if this point has enough determinant in the hessian
            Eigen::Matrix2d gradient_hessian;
            gradient_hessian.setZero();
            for (size_t p = 0; p < m_pattern.get_nr_points(); p++) {
                int dx = m_pattern.get_offset_x(p);
                int dy = m_pattern.get_offset_y(p);

                float gradient_x=grad_x.at<float>(i+dy,j+dx); //TODO should be interpolated
                float gradient_y=grad_y.at<float>(i+dy,j+dx);

                Eigen::Vector2d grad;
                grad << gradient_x, gradient_y;
                // std::cout << "gradients are " << gradient_x << " " << gradient_y << '\n';

                gradient_hessian+= grad*grad.transpose();
            }

            //determinant is high enough, add the point
            float hessian_det=gradient_hessian.determinant();
            if(hessian_det > 100000000){
                ImmaturePoint point;
                point.u=j;
                point.v=i;
                point.gradH=gradient_hessian;

                //Seed::Seed
                point.f = (frame.K.inverse() * Eigen::Vector3d(point.u,point.v,1)).normalized();

                //start at an initial value for depth at around 4 meters (depth_filter->Seed::reinit)
                float mean_starting_depth=4.0;
                float min_starting_depth=0.1;
                point.mu = (1.0/mean_starting_depth);
                point.z_range = (1.0/min_starting_depth);
                point.sigma2 = (point.z_range*point.z_range/36);

                float z_inv_min = point.mu + sqrt(point.sigma2);
                float z_inv_max = std::max<float>(point.mu- sqrt(point.sigma2), 0.00000001f);
                point.idepth_min = z_inv_min;
                point.idepth_max = z_inv_max;

                point.a=10.0;
                point.b=10.0;


                //get data for the color of that point (depth_point->ImmaturePoint::ImmaturePoint)---------------------
                for(int p_idx=0;p_idx<m_pattern.get_nr_points();p_idx++){
                    Eigen::Vector2d offset = m_pattern.get_offset(p_idx);

                    point.color[p_idx]=texture_interpolate(frame.gray, point.u+offset(0), point.v+offset(1));

                    float grad_x_val=texture_interpolate(grad_x, point.u+offset(0), point.v+offset(1));
                    float grad_y_val=texture_interpolate(grad_y, point.u+offset(0), point.v+offset(1));
                    float squared_norm=grad_x_val*grad_x_val + grad_y_val*grad_y_val;
                    point.weights[p_idx] = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + squared_norm));
                }
                point.ncc_sum_templ    = 0.0f;
                float ncc_sum_templ_sq = 0.0f;
                for(int p_idx=0;p_idx<m_pattern.get_nr_points();p_idx++){
                    const float templ = point.color[p_idx];
                    point.ncc_sum_templ += templ;
                    ncc_sum_templ_sq += templ*templ;
                }
                point.ncc_const_templ = m_pattern.get_nr_points() * ncc_sum_templ_sq - (double) point.ncc_sum_templ*point.ncc_sum_templ;

                point.energyTH = m_pattern.get_nr_points()*setting_outlierTH;
                point.energyTH *= setting_overallEnergyTHWeight*setting_overallEnergyTHWeight;

                point.quality=10000;
                //-------------------------------------

                //debug stuff
                point.gradient_hessian_det=hessian_det;
                point.last_visible_frame=0;
                point.gt_depth=frame.depth.at<float>(i,j); //add also the gt depth just for visualization purposes

                immature_points.push_back(point);
            }

        }
    }
    TIME_END_CL("hessian_host_frame");




    return immature_points;
}

void DepthEstimatorCL::update_immature_points(std::vector<ImmaturePoint>& immature_points, const Frame& frame, const Eigen::Affine3d& tf_cur_host, const Eigen::Matrix3d& KRKi_cr, const Eigen::Vector3d& Kt_cr, const Eigen::Vector2d& affine_cr){

    TIME_START_CL("update_immature_points");

    const double focal_length = abs(frame.K(0,0));
    double px_noise = 1.0;
    double px_error_angle = atan(px_noise/(2.0*focal_length))*2.0; // law of chord (sehnensatz)
    const Eigen::Affine3d tf_host_cur = tf_cur_host.inverse();


    for (auto &point : immature_points){
//        Eigen::Vector3d point_host_screen;
//        point_host_screen << point.u, point.v, 1.0;
//
//        Eigen::Vector3d point_host_cam; //point in the coordinate of the host frame
//        point_host_cam=frame.K.inverse()*point_host_screen*point.gt_depth; //TODO should use the K of the host frame
//
//        Eigen::Vector3d point_cur_cam; //point in the coordinate of the cur frame
//        point_cur_cam= tf_cur_host * point_host_cam;
//
//        Eigen::Vector2d point_cur_screen;
//        point_cur_screen= (frame.K * point_cur_cam).hnormalized();
//
//        if(point_cur_cam.z() < 0.0)  {
//            continue; // behind the camera
//        }
//
//        if ( point_cur_screen(0) < 0 || point_cur_screen(0) >= frame.gray.cols || point_cur_screen(1) < 0 || point_cur_screen(1) >= frame.gray.rows ){
//            continue; // point does not project in image
//        }


        // check if point is visible in the current image
        const Eigen::Vector3d xyz_f( tf_cur_host*(1.0/point.mu * point.f) );
        if(xyz_f.z() < 0.0)  {
            continue;
        }
        const Eigen::Vector2d kp_c = (frame.K * xyz_f).hnormalized();
        if ( kp_c(0) < 0 || kp_c(0) >= frame.gray.cols || kp_c(1) < 0 || kp_c(1) >= frame.gray.rows ) {
            continue;
        }


        //point is visible
        point.last_visible_frame=frame.frame_id;

        //update inverse depth coordinates for min and max
        point.idepth_min = point.mu + sqrt(point.sigma2);
        point.idepth_max = std::max<float>(point.mu - sqrt(point.sigma2), 0.00000001f);
//
        //search epiline----------------------
        // std::cout << "search epi line for point " << i << "\n";
//        search_epiline_ncc (point, frame, KRKi_cr, Kt_cr );
        search_epiline_bca (point, frame, KRKi_cr, Kt_cr, affine_cr);

        double idepth = -1;
        double z = 0;
        if( point.lastTraceStatus == ImmaturePointStatus::IPS_GOOD ) {
            idepth = std::max<double>(1e-5,.5*(point.idepth_min+point.idepth_max));
            z = 1./idepth;
        }
        if ( point.lastTraceStatus == ImmaturePointStatus::IPS_OOB  || point.lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED ) {
            continue;
        }
        if ( !std::isfinite(idepth) || point.lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER || point.lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION ) {
            point.b++; // increase outlier probability when no match was found
            continue;
        }


        update_idepth(point,tf_host_cur, z, px_error_angle);



    }

    // //-----------------------------
    // const Eigen::Vector3d xyz_f( T_cur_ref*(1.0/it->mu * it->f) );
    // if(xyz_f.z() < 0.0)  {
    //     //++it; // behind the camera
    //     continue;
    // }
    //
    // const Eigen::Vector2d kp_c = (curImgPtr->K_c[0] * xyz_f).hnormalized();
    // if ( kp_c(0) < 0 || kp_c(0) >= curImgPtr->grayImages[0].cols || kp_c(1) < 0 || kp_c(1) >= curImgPtr->grayImages[0].rows )
    // {        //if(!frame->cam_->isInFrame(frame->f2c(xyz_f).cast<int>())) {
    //     //++it; // point does not project in image
    //     continue;
    // }

    TIME_END_CL("update_immature_points");
}

void DepthEstimatorCL::search_epiline_bca(ImmaturePoint& point, const Frame& frame, const Eigen::Matrix3d& KRKi_cr, const Eigen::Vector3d& Kt_cr, const Eigen::Vector2d& affine_cr){

    if(point.lastTraceStatus == ImmaturePointStatus::IPS_OOB || point.lastTraceStatus == ImmaturePointStatus::IPS_DELETED ) return;


    float idepth_mean = (point.idepth_min + point.idepth_max)*0.5;
    Eigen::Vector3d pr = KRKi_cr * Eigen::Vector3d(point.u,point.v, 1);
    Eigen::Vector3d ptpMean = pr + Kt_cr*idepth_mean;
    Eigen::Vector3d ptpMin = pr + Kt_cr*point.idepth_min;
    Eigen::Vector3d ptpMax = pr + Kt_cr*point.idepth_max;
    Eigen::Vector2d uvMean = ptpMean.hnormalized();
    Eigen::Vector2d uvMin = ptpMin.hnormalized();
    Eigen::Vector2d uvMax = ptpMax.hnormalized();

    Pattern pattern_rot=m_pattern.get_rotated_pattern( KRKi_cr.topLeftCorner<2,2>() );

    Eigen::Vector2d epi_line = uvMax - uvMin;
    float norm_epi = std::max<float>(1e-5f,epi_line.norm());
    Eigen::Vector2d epi_dir = epi_line / norm_epi;
    const float  half_length = 0.5f * norm_epi;

    Eigen::Vector2d bestKp;
    float bestEnergy = 1e10;

    for(float l = -half_length; l <= half_length; l += 0.7f)
    {
        float energy = 0;
        Eigen::Vector2d kp = uvMean + l*epi_dir;

        if( !kp.allFinite() || ( kp(0) >= (frame.gray.cols-10) )  || ( kp(1) >= (frame.gray.rows-10) ) || ( kp(0) < 10 ) || ( kp(1) < 10) )
        {
            continue;
        }

        for(int idx=0;idx<pattern_rot.get_nr_points(); ++idx)
        {
            //float hitColor = getInterpolatedElement31(frame->dI, (float)(kp(0)+rotatetPattern[idx][0]), (float)(kp(1)+rotatetPattern[idx][1]), wG[0]);
            Eigen::Vector2d offset=pattern_rot.get_offset(idx);
            float hit_color=texture_interpolate(frame.gray, kp(0)+offset(0), kp(1)+offset(1) , InterpolationType::CUBIC);
            if(!std::isfinite(hit_color)) {energy-=1e5; continue;}

            const float residual = hit_color - (float)(affine_cr[0] * point.color[idx] + affine_cr[1]);

            float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
            energy += hw *residual*residual*(2-hw);
        }
        if ( energy < bestEnergy )
        {
            bestKp = kp; bestEnergy = energy;
        }
    }

    if ( bestEnergy > point.energyTH * 1.2f ) {
        point.lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
    }
    else
    {
        float a = (Eigen::Vector2d(epi_dir(0),epi_dir(1)).transpose() * point.gradH * Eigen::Vector2d(epi_dir(0),epi_dir(1)));
        float b = (Eigen::Vector2d(epi_dir(1),-epi_dir(0)).transpose() * point.gradH * Eigen::Vector2d(epi_dir(1),-epi_dir(0)));
        float errorInPixel = 0.2f + 0.2f * (a+b) / a; // WO kommt das her? Scheint nicht zu NGF zu passen !

        if( epi_dir(0)*epi_dir(0)>epi_dir(1)*epi_dir(1) )
        {
            point.idepth_min = (pr[2]*(bestKp(0)-errorInPixel*epi_dir(0)) - pr[0]) / (Kt_cr[0] - Kt_cr[2]*(bestKp(0)-errorInPixel*epi_dir(0)));
            point.idepth_max = (pr[2]*(bestKp(0)+errorInPixel*epi_dir(0)) - pr[0]) / (Kt_cr[0] - Kt_cr[2]*(bestKp(0)+errorInPixel*epi_dir(0)));
        }
        else
        {
            point.idepth_min = (pr[2]*(bestKp(1)-errorInPixel*epi_dir(1)) - pr[1]) / (Kt_cr[1] - Kt_cr[2]*(bestKp(1)-errorInPixel*epi_dir(1)));
            point.idepth_max = (pr[2]*(bestKp(1)+errorInPixel*epi_dir(1)) - pr[1]) / (Kt_cr[1] - Kt_cr[2]*(bestKp(1)+errorInPixel*epi_dir(1)));
        }
        if(point.idepth_min > point.idepth_max) std::swap<float>(point.idepth_min, point.idepth_max);

        point.lastTraceStatus = ImmaturePointStatus::IPS_GOOD;
    }
}

void DepthEstimatorCL::search_epiline_ncc(ImmaturePoint& point, const Frame& frame, const Eigen::Matrix3d& KRKi_cr, const Eigen::Vector3d& Kt_cr){

//    Eigen::Vector3d pr = KRKi_cr * Eigen::Vector3d(point.u,point.v, 1);
//    Eigen::Vector3d ptpMean = pr + Kt_cr*point.mu;
//    Eigen::Vector3d ptpMin = pr + Kt_cr*point.idepth_min;
//    Eigen::Vector3d ptpMax = pr + Kt_cr*point.idepth_max;
//    Eigen::Vector2d uvMean = ptpMean.hnormalized();
//    Eigen::Vector2d uvMin = ptpMin.hnormalized();
//    Eigen::Vector2d uvMax = ptpMax.hnormalized();
//
//    Pattern pattern_rot=m_pattern.get_rotated_pattern( KRKi_cr.topLeftCorner<2,2>() );
//
//    Eigen::Vector2d epi_line = uvMax - uvMin;
//    float norm_epi = std::max<float>(1e-5f,epi_line.norm());
//    Eigen::Vector2d epi_dir = epi_line / norm_epi;
//    const float  half_length = 0.5f * norm_epi;
//
//    Eigen::Vector2d bestKp;
//    float bestEnergy = -1.0f;
//
//     // Retrieve template statistics for NCC matching;
//     const float sum_templ = point.ncc_sum_templ ;
//     const float const_templ_denom = point.ncc_const_templ;
//
//     for(float l = -half_length; l <= half_length; l += 0.7f)
//     {
//         float energy = 0;
//         float sum_img = 0.f;
//         float sum_img_sq = 0.f;
//         float sum_img_templ = 0.f;
//
//         Eigen::Vector2d kp = uvMean + l*epi_dir;
//
//         if( !kp.allFinite() || ( kp(0) >= (frame.gray.cols-7) )  || ( kp(1) >= (frame.gray.rows-7) ) || ( kp(0) < 7 ) || ( kp(1) < 7) ) {
//           continue;
//         }
//
//         for(int idx=0;idx<pattern_rot.get_nr_points(); ++idx)
//         {
//             //float hitColor = getInterpolatedElement31(frame->dI, (float)(kp(0)+rotatetPattern[idx][0]), (float)(kp(1)+rotatetPattern[idx][1]), wG[0]);
//
//
//             Eigen::Vector2d offset=pattern_rot.get_offset(idx);
//             float hit_color=texture_interpolate(frame.gray, kp(0)+offset(0), kp(1)+offset(1) );
//
//
//             const float templ = point.color[idx];
//             const float img = hit_color;
//             sum_img    += img;
//             sum_img_sq += img*img;
//             sum_img_templ += img*templ;
//         }
//         const float ncc_numerator = pattern_rot.get_nr_points()*sum_img_templ - sum_img*sum_templ;
//         const float ncc_denominator = (pattern_rot.get_nr_points()*sum_img_sq - sum_img*sum_img)*const_templ_denom;
//         energy += ncc_numerator * sqrt(ncc_denominator + 1e-10);
//
//         if( energy > bestEnergy )
//         {
//             bestKp = kp; bestEnergy = energy;
//         }
//     }
//
//     if( bestEnergy < .5f ) {
//         point.lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
//     } else {
//         //TODO WTF is this??
////         float a = (Eigen::Vector2f(epi_dir(0),epi_dir(1)).transpose() * gradH * Eigen::Vector2f(epi_dir(0),epi_dir(1)));
////         float b = (Eigen::Vector2f(epi_dir(1),-epi_dir(0)).transpose() * gradH * Eigen::Vector2f(epi_dir(1),-epi_dir(0)));
////         float errorInPixel = 0.2f + 0.2f * (a+b) / a; // WO kommt das her? Scheint nicht zu NGF zu passen !
//         float errorInPixel=0.0;
//
//         if( epi_dir(0)*epi_dir(0)>epi_dir(1)*epi_dir(1) )
//         {
//             point.idepth_min = (pr[2]*(bestKp(0)-errorInPixel*epi_dir(0)) - pr[0]) / (Kt_cr[0] - Kt_cr[2]*(bestKp(0)-errorInPixel*epi_dir(0)));
//             point.idepth_max = (pr[2]*(bestKp(0)+errorInPixel*epi_dir(0)) - pr[0]) / (Kt_cr[0] - Kt_cr[2]*(bestKp(0)+errorInPixel*epi_dir(0)));
//         }
//         else
//         {
//             point.idepth_min = (pr[2]*(bestKp(1)-errorInPixel*epi_dir(1)) - pr[1]) / (Kt_cr[1] - Kt_cr[2]*(bestKp(1)-errorInPixel*epi_dir(1)));
//             point.idepth_max = (pr[2]*(bestKp(1)+errorInPixel*epi_dir(1)) - pr[1]) / (Kt_cr[1] - Kt_cr[2]*(bestKp(1)+errorInPixel*epi_dir(1)));
//         }
//         if(point.idepth_min > point.idepth_max) std::swap<float>(point.idepth_min, point.idepth_max);
//
////         lastTraceUV = bestKp;
//         point.lastTraceStatus = ImmaturePointStatus::IPS_GOOD;
//     }
}

void DepthEstimatorCL::update_idepth(ImmaturePoint& point, const Eigen::Affine3d& tf_host_cur, const float z, const double px_error_angle){

    // compute tau
    double tau = compute_tau(tf_host_cur, point.f, z, px_error_angle);
    double tau_inverse = 0.5 * (1.0/std::max<double>(0.0000001, z-tau) - 1.0/(z+tau));

    // update the estimate
    updateSeed(point, 1.0/z, tau_inverse*tau_inverse);

//    const float eta_inlier = .6f;
//    const float eta_outlier = .05f;
//    // if E(inlier_ratio) > eta_inlier && sigma_sq < epsilon
//    if( ((it->a / (it->a + it->b)) > eta_inlier) &&
//        //(it->sigma2 < epsilon)
//        (sqrt(it->sigma2) < it->z_range/options_.seed_convergence_sigma2_thresh)
//            )
//    {
//        it->is_outlier = false;
//    }

}

double DepthEstimatorCL::compute_tau(const Eigen::Affine3d & tf_host_cur, const Eigen::Vector3d& f, const double z, const double px_error_angle){
    Eigen::Vector3d t(tf_host_cur.translation());
    Eigen::Vector3d a = f*z-t;
    double t_norm = t.norm();
    double a_norm = a.norm();
    double alpha = acos(f.dot(t)/t_norm); // dot product
    double beta = acos(a.dot(-t)/(t_norm*a_norm)); // dot product
    double beta_plus = beta + px_error_angle;
    double gamma_plus = M_PI-alpha-beta_plus; // triangle angles sum to PI
    double z_plus = t_norm*sin(beta_plus)/sin(gamma_plus); // law of sines
    return (z_plus - z); // tau

}

void DepthEstimatorCL::updateSeed(ImmaturePoint& point, const float x, const float tau2) {
    float norm_scale = sqrt(point.sigma2 + tau2);
    if(std::isnan(norm_scale))
        return;
    boost::math::normal_distribution<float> nd(point.mu, norm_scale);
    float s2 = 1./(1./point.sigma2 + 1./tau2);
    float m = s2*(point.mu/point.sigma2 + x/tau2);
    float C1 = point.a/(point.a+point.b) * boost::math::pdf(nd, x);
    float C2 = point.b/(point.a+point.b) * 1./point.z_range;
    float normalization_constant = C1 + C2;
    C1 /= normalization_constant;
    C2 /= normalization_constant;
    float f = C1*(point.a+1.)/(point.a+point.b+1.) + C2*point.a/(point.a+point.b+1.);
    float e = C1*(point.a+1.)*(point.a+2.)/((point.a+point.b+1.)*(point.a+point.b+2.))
              + C2*point.a*(point.a+1.0f)/((point.a+point.b+1.0f)*(point.a+point.b+2.0f));

    // update parameters
    float mu_new = C1*m+C2*point.mu;
    point.sigma2 = C1*(s2 + m*m) + C2*(point.sigma2 + point.mu*point.mu) - mu_new*mu_new;
    point.mu = mu_new;
    point.a = (e-f)/(f-e/f);
    point.b = point.a*(1.0f-f)/f;
}

Mesh DepthEstimatorCL::create_mesh(const std::vector<ImmaturePoint>& immature_points, const std::vector<Frame>& frames){
    Mesh mesh;
    mesh.V.resize(immature_points.size(),3);
    mesh.V.setZero();

    for (size_t i = 0; i < immature_points.size(); i++) {
        int u=(int)immature_points[i].u;
        int v=(int)immature_points[i].v;
        float depth=1/immature_points[i].mu;

        if(std::isfinite(immature_points[i].mu) && immature_points[i].mu>=0.1){
            //backproject the immature point
            Eigen::Vector3d point_screen;
            point_screen << u, v, 1.0;
            Eigen::Vector3d point_dir=frames[0].K.inverse()*point_screen; //TODO get the K and also use the cam to world corresponding to the immature point
            // point_dir.normalize(); //this is just the direction (in cam coordinates) of that pixel
            Eigen::Vector3d point_cam = point_dir*depth;
            point_cam(2)=-point_cam(2); //flip the depth because opengl has a camera which looks at the negative z axis (therefore, more depth means a more negative number)


            Eigen::Vector3d point_world=frames[0].tf_cam_world.inverse()*point_cam;

            mesh.V.row(i)=point_world;
        }

//


        //attempt 2 at doing it right
//        Eigen::Vector3d point_screen;
//        point_screen << u, v, 1.0;
//        Eigen::Vector3d point_unprojected=frames[0].K.inverse()*point_screen; //TODO get the K and also use the cam to world corresponding to the immature point
////        Eigen::Vector3d point_unprojected_hom=point_unprojected/point_unprojected(2);
//        Eigen::Vector3d point_unprojected_hom=point_unprojected.normalized();
//        Eigen::Vector3d point_world=point_unprojected_hom*depth;
//        mesh.V.row(i)=point_world;



//        //Attempt 3
//        Eigen::Vector3d f = (frames[0].K.inverse()* Eigen::Vector3d(u,v,1));
//        const Eigen::Vector3d xyz_f(depth * f);
//        mesh.V.row(i)=xyz_f;



    }

    //make also some colors based on depth
    mesh.C.resize(immature_points.size(),3);
    double min_z, max_z;
    min_z = mesh.V.col(2).minCoeff();
    max_z = mesh.V.col(2).maxCoeff();
    std::cout << "min max z is " << min_z << " " << max_z << '\n';
    for (size_t i = 0; i < mesh.C.rows(); i++) {
        float gray_val = lerp(mesh.V(i,2), min_z, max_z, 0.0, 1.0 );
        mesh.C(i,0)=mesh.C(i,1)=mesh.C(i,2)=gray_val;
    }

    //colors based on gradient_hessian det
    // float min=9999999999, max=-9999999999;
    // for (size_t i = 0; i < immature_points.size(); i++) {
    //     if(immature_points[i].gradient_hessian_det<min){
    //         min=immature_points[i].gradient_hessian_det;
    //     }
    //     if(immature_points[i].gradient_hessian_det>max){
    //         max=immature_points[i].gradient_hessian_det;
    //     }
    // }
    // for (size_t i = 0; i < mesh.C.rows(); i++) {
    //      float gray_val = lerp(immature_points[i].gradient_hessian_det, min, max, 0.0, 1.0 );
    //      mesh.C(i,0)=mesh.C(i,1)=mesh.C(i,2)=gray_val;
    //  }

    //colors based on last frame seen
//    float min=9999999999, max=-9999999999;
//    for (size_t i = 0; i < immature_points.size(); i++) {
//        if(immature_points[i].last_visible_frame<min){
//            min=immature_points[i].last_visible_frame;
//        }
//        if(immature_points[i].last_visible_frame>max){
//            max=immature_points[i].last_visible_frame;
//        }
//    }
//    std::cout << "min max z is " << min << " " << max << '\n';
//    for (size_t i = 0; i < mesh.C.rows(); i++) {
//         float gray_val = lerp(immature_points[i].last_visible_frame, min, max, 0.0, 1.0 );
//         mesh.C(i,0)=mesh.C(i,1)=mesh.C(i,2)=gray_val;
//     }






    return mesh;
}


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
