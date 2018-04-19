__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
__constant sampler_t sampler_linear = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

//
enum PointStatus {
    GOOD=0,					// traced well and good
    OOB,					// OOB: end tracking & marginalize!
    OUTLIER,				// energy too high: if happens again: outlier!
    SKIPPED,				// traced well and good (but not actually traced).
    BADCONDITION,			// not traced because of bad condition.
    DELETED,                            // merged with other point or deleted
    UNINITIALIZED};			// not even traced once.


struct  Point{
    int idx_host_frame; //idx in the array of frames of the frame which "hosts" this inmature points
    float u,v; //position in host frame of the point

    // float test_array[16];
    // int test_bool_array[16]; //--break it
    // // cl_bool bool_1; //--also breaks it
    // // cl_bool bool_2;
    //  int test_int_array[16];

    float a;                     //!< a of Beta distribution: When high, probability of inlier is large.
    float b;                     //!< b of Beta distribution: When high, probability of outlier is large.
    float mu;                    //!< Mean of normal distribution.
    float z_range;               //!< Max range of the possible depth.
    float sigma2;                //!< Variance of normal distribution.
    float idepth_min;
    float idepth_max;
    float energyTH;
    float quality;
    float3 f; // heading range = Ki * (u,v,1)
    // enum PointStatus lastTraceStatus;
    // bool converged;
    // bool is_outlier;

    float color[10]; 		// colors in host frame
    float weights[10]; 		// host-weights for respective residuals.
    // Vec2f colorD[MAX_RES_PER_POINT];
    // Vec2f colorGrad[MAX_RES_PER_POINT];
    // Vec2f rotatetPattern[MAX_RES_PER_POINT];
    // bool skipZero [10];

    float ncc_sum_templ;
    float ncc_const_templ;

    // Stuff that may be to be removed
    float2 kp_GT;
    // float kp_GT[2];


    //debug stuff
    float gradient_hessian_det;
    int last_visible_frame;
    float gt_depth;

};

#pragma OPENCL EXTENSION cl_amd_printf : enable
#pragma OPENCL EXTENSION cl_intel_printf : enable
__kernel void struct_test( __global struct Point* points, __read_only image2d_t img)
{
    int id = get_global_id(0);
    points[id].last_visible_frame=id;

    //try also to read from the img
    points[id].last_visible_frame=read_imageui(img, sampler_linear, (int2)(points[id].u,points[id].v)).x;

    //see if u is correct
    // points[id].last_visible_frame=points[id].v;



    // printf("thread %d last visible frame is%d \n",id,input[id].last_visible_frame);


    // const int2 coords = {get_global_id(0), get_global_id(1)};
    //
    // float4 value = read_imagef(input_image, sampler, coords);
    // value[2]=0.0;
    // write_imagef(output_image, coords, value);
}
