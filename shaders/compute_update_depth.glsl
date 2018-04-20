#version 430

layout (local_size_x = 32) in;

const int MAX_RES_PER_POINT=16;

struct Point{
    int idx_host_frame; //idx in the array of frames of the frame which "hosts" this inmature points
    float u,v; //position in host frame of the point

    // cl_float test_array[16];
    // cl_int test_bool_array[16]; //--break it
    // // cl_bool bool_1; //--also breaks it
    // // cl_bool bool_2;
    //  cl_int test_int_array[16];

    float a;                     //!< a of Beta distribution: When high, probability of inlier is large.
    float b;                     //!< b of Beta distribution: When high, probability of outlier is large.
    float mu;                    //!< Mean of normal distribution.
    float z_range;               //!< Max range of the possible depth.
    float sigma2;                //!< Variance of normal distribution.
    float idepth_min;
    float idepth_max;
    float energyTH;
    float quality;
    vec4 f; // heading range = Ki * (u,v,1) //make it float 4 becuse float 3 gets padded to 4 either way
    // // PointStatus lastTraceStatus;
    // // cl_bool converged;
    // // cl_bool is_outlier;
    //
    float color[MAX_RES_PER_POINT]; 		// colors in host frame
    float weights[MAX_RES_PER_POINT]; 		// host-weights for respective residuals.
    // Vec2f colorD[MAX_RES_PER_POINT];
    // Vec2f colorGrad[MAX_RES_PER_POINT];
    // Vec2f rotatetPattern[MAX_RES_PER_POINT];
    // cl_bool skipZero [cl_MAX_RES_PER_POINT];
    //
    float ncc_sum_templ;
    float ncc_const_templ;

    //Stuff that may be to be removed
    vec2 kp_GT;
    // // cl_float kp_GT[2];
    //
    //
    //debug stuff
    float gradient_hessian_det;
    float gt_depth;
    int last_visible_frame;
    float debug;

};

//you change it to  layout (binding = 0, std430) or layout (binding = 0, std140) in case stuff breaks
layout (binding = 0) buffer array_points_block
{
Point p[];
};


uniform vec2 frame_size; //x, y
uniform mat4 tf_cur_host;
uniform mat4 tf_host_cur;
uniform mat3 K;
uniform mat3 KRKi_cr;
uniform vec3 Kt_cr;
uniform vec2 affine_cr;
uniform float px_error_angle;

uniform sampler2D gray_img_sampler;


void main(void) {

    int id = int(gl_GlobalInvocationID.x);


    p[id].debug=int(p[id].u);
    float color=texture(gray_img_sampler, vec2(p[id].u/640, p[id].v/480)).x;
    p[id].debug=float(color);


    // check if point is visible in the current image
    const vec3 p_backproj_xyz= p[id].f.xyz * 1.0/p[id].mu;
    const vec4 p_backproj_xyzw=vec4(p_backproj_xyz,1.0);
    const vec4 xyz_f_xyzw = tf_cur_host*  p_backproj_xyzw ;
    const vec3 xyz_f=xyz_f_xyzw.xyz/xyz_f_xyzw.w;
    if(xyz_f.z < 0.0)  {
        return;
    }
    const vec3 kp_c = K * xyz_f;
    const vec2 kp_c_h=kp_c.xy/kp_c.z;
    if ( kp_c_h.x < 0 || kp_c_h.x >= frame_size.x || kp_c_h.y < 0 || kp_c_h.y >= frame_size.y ) {
        return;
    }


    //point is visible

    //update inverse depth coordinates for min and max
    p[id].idepth_min = p[id].mu + sqrt(p[id].sigma2);
    p[id].idepth_max = max(p[id].mu - sqrt(p[id].sigma2), 0.00000001f);

    //search epiline----------------------
    // search_epiline_bca (point, frame, KRKi_cr, Kt_cr, affine_cr);



    // double idepth = -1;
    // double z = 0;
    // if( point.lastTraceStatus == ImmaturePointStatus::IPS_GOOD ) {
    //     idepth = std::max<double>(1e-5,.5*(point.idepth_min+point.idepth_max));
    //     z = 1./idepth;
    // }
    // if ( point.lastTraceStatus == ImmaturePointStatus::IPS_OOB  || point.lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED ){
    //     continue;
    // }
    // if ( !std::isfinite(idepth) || point.lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER || point.lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION ) {
    //     point.b++; // increase outlier probability when no match was found
    //     continue;
    // }
    //
    //
    // update_idepth(point,tf_host_cur, z, px_error_angle);



}
