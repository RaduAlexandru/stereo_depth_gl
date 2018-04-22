#version 430

layout (local_size_x = 256) in;

#define M_PI 3.1415926535897932384626433832795

const int MAX_RES_PER_POINT=16;
const float setting_huberTH = 9; // Huber Threshold

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
uniform vec2 pattern_rot_offsets[MAX_RES_PER_POINT];
uniform int pattern_rot_nr_points;


uniform sampler2D gray_img_sampler;

//https://www.boost.org/doc/libs/1_47_0/libs/math/doc/sf_and_dist/html/math_toolkit/dist/dist_ref/dists/normal_dist.html
float gaus_pdf(float mean, float sd, float x){
    return exp(- (x-mean)*(x-mean)/(2*sd)*(2*sd)  )  / (sd*sqrt(2*M_PI));
}

void main(void) {

    int id = int(gl_GlobalInvocationID.x);


    //debug if pattern_rot is corectly passed
    // if(id < 8){
    //     vec2 offset=pattern_rot_offsets[id];
    //     p[id].u=offset.x;
    //     p[id].v=offset.y;
    // }


    // p[id].debug=int(p[id].u);
    // float color=texture(gray_img_sampler, vec2(p[id].u/640, p[id].v/480)).x;
    // p[id].debug=float(color);


    // check if point is visible in the current image
    const vec3 p_backproj_xyz= p[id].f.xyz * 1.0/p[id].mu;
    const vec4 p_backproj_xyzw=vec4(p_backproj_xyz,1.0);
    const vec4 xyz_f_xyzw = tf_cur_host*  p_backproj_xyzw ;
    const vec3 xyz_f=xyz_f_xyzw.xyz/xyz_f_xyzw.w;
    if(xyz_f.z < 0.0)  {
        // p[id].debug=1.0;
        return;
    }
    const vec3 kp_c = K * xyz_f;
    const vec2 kp_c_h=kp_c.xy/kp_c.z;
    if ( kp_c_h.x < 0 || kp_c_h.x >= frame_size.x || kp_c_h.y < 0 || kp_c_h.y >= frame_size.y ) {
        // p[id].debug=1.0;
        return;
    }


    //point is visible

    //update inverse depth coordinates for min and max
    p[id].idepth_min = p[id].mu + sqrt(p[id].sigma2);
    p[id].idepth_max = max(p[id].mu - sqrt(p[id].sigma2), 0.00000001f);
    memoryBarrier();
    barrier();

    //search epiline---------------------------------------------------------------
    // search_epiline_bca (point, frame, KRKi_cr, Kt_cr, affine_cr);


    float idepth_mean = (p[id].idepth_min + p[id].idepth_max)*0.5;
    vec3 pr = KRKi_cr * vec3(p[id].u,p[id].v, 1);
    vec3 ptpMean = pr + Kt_cr*idepth_mean;
    vec3 ptpMin = pr + Kt_cr*p[id].idepth_min;
    vec3 ptpMax = pr + Kt_cr*p[id].idepth_max;
    vec2 uvMean = ptpMean.xy/ptpMean.z;
    vec2 uvMin = ptpMin.xy/ptpMin.z;
    vec2 uvMax = ptpMax.xy/ptpMax.z;

    // Pattern pattern_rot=m_pattern.get_rotated_pattern( KRKi_cr.topLeftCorner<2,2>() );

    vec2 epi_line = uvMax - uvMin;
    float norm_epi = max(1e-5f, length(epi_line));
    vec2 epi_dir = epi_line / norm_epi;
    const float  half_length = 0.5f * norm_epi;

    vec2 bestKp;
    float bestEnergy = 1e15;

    //debug stuff
    float residual_debug=0;
    int nr_time_switched_best=0;

    for(float l = -half_length; l <= half_length; l += 0.7f){
        float energy = 0;
        residual_debug=0;
        vec2 kp = uvMean + l*epi_dir;

        if( ( kp.x >= (frame_size.x-10) )  || ( kp.y >= (frame_size.y-10) ) || ( kp.x < 10 ) || ( kp.y < 10) ){
            continue;
        }

        for(int idx=0;idx<pattern_rot_nr_points; ++idx) {

            vec2 offset=pattern_rot_offsets[idx];
            float hit_color=texture(gray_img_sampler, vec2( (kp.x + offset.x)/640, (kp.y + offset.y)/480)).x;
            // nr_times_textured_fetched++;
            // float hit_color=texture_interpolate(frame.gray, kp(0)+offset(0), kp(1)+offset(1) , InterpolationType::LINEAR);
            // if(!std::isfinite(hit_color)) {energy-=1e5; continue;}
            //
            const float residual = hit_color - float(affine_cr.x * p[id].color[idx] + affine_cr.y);
            residual_debug+=residual;

            float hw = abs(residual) < setting_huberTH ? 1 : setting_huberTH / abs(residual);
            energy += hw *residual*residual*(2-hw);
        }

        // p[id].debug=energy;
        if ( energy < bestEnergy ){
            bestKp = kp; bestEnergy = energy;
            nr_time_switched_best++;
        }
    }

    if(bestEnergy<1e10 ){
        p[id].debug=bestEnergy;
    }

    // p[id].debug=nr_time_switched_best;


    if ( bestEnergy > p[id].energyTH * 1.2f ) {
        // point.lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
    }else{
        // float a = (Eigen::Vector2d(epi_dir(0),epi_dir(1)).transpose() * point.gradH * Eigen::Vector2d(epi_dir(0),epi_dir(1)));
        // float b = (Eigen::Vector2d(epi_dir(1),-epi_dir(0)).transpose() * point.gradH * Eigen::Vector2d(epi_dir(1),-epi_dir(0)));
        // float errorInPixel = 0.2f + 0.2f * (a+b) / a; // WO kommt das her? Scheint nicht zu NGF zu passen !
        float errorInPixel=0;

        if( epi_dir.x*epi_dir.x>epi_dir.y*epi_dir.y )
        {
            p[id].idepth_min = (pr.z*(bestKp.x-errorInPixel*epi_dir.x) - pr.x) / (Kt_cr.x - Kt_cr.z*(bestKp.x-errorInPixel*epi_dir.x));
            p[id].idepth_max = (pr.z*(bestKp.x+errorInPixel*epi_dir.x) - pr.x) / (Kt_cr.x - Kt_cr.z*(bestKp.x+errorInPixel*epi_dir.x));
        }else{
            p[id].idepth_min = (pr.z*(bestKp.y-errorInPixel*epi_dir.y) - pr.y) / (Kt_cr.y - Kt_cr.z*(bestKp.y-errorInPixel*epi_dir.y));
            p[id].idepth_max = (pr.z*(bestKp.y+errorInPixel*epi_dir.y) - pr.y) / (Kt_cr.y - Kt_cr.z*(bestKp.y+errorInPixel*epi_dir.y));
        }
        memoryBarrier();
        barrier();
        if(p[id].idepth_min > p[id].idepth_max) {
            // std::swap<float>(point.idepth_min, point.idepth_max);
            float tmp=p[id].idepth_min;
            p[id].idepth_min=p[id].idepth_max;
            p[id].idepth_max=tmp;
        }
        memoryBarrier();
        barrier();

        // point.lastTraceStatus = ImmaturePointStatus::IPS_GOOD;
    }
    //
    //
    //
    //
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



    float idepth = -1;
    float z = 0;
    idepth = max(1e-5,.5*(p[id].idepth_min+p[id].idepth_max));
    z = 1./idepth;




    // compute tau-------------------------------------------------------------------------
    // double tau = compute_tau(tf_host_cur, point.f, z, px_error_angle);
    vec3 t= vec3(tf_host_cur[0][3], tf_host_cur[1][3], tf_host_cur[2][3]);
    vec3 a = p[id].f.xyz*z-t;
    float t_norm = length(t);
    float a_norm = length(a);
    float alpha = acos(dot(p[id].f.xyz,t)/t_norm); // dot product
    float beta = acos(dot(a,-t)/(t_norm*a_norm)); // dot product
    float beta_plus = beta + px_error_angle;
    float gamma_plus = 3.1415-alpha-beta_plus; // triangle angles sum to PI
    float z_plus = t_norm*sin(beta_plus)/sin(gamma_plus); // law of sines
    float tau= (z_plus - z); // tau



    float tau_inverse = 0.5 * (1.0/max(0.0000001, z-tau) - 1.0/(z+tau));

    // update the estimate------------------------------------------------------------------
    // updateSeed(point, 1.0/z, tau_inverse*tau_inverse);

    float x= 1.0/z;
    float tau2=tau_inverse*tau_inverse;
    float norm_scale = sqrt(p[id].sigma2 + tau2);
    // if(std::isnan(norm_scale))
    //     return;
    float s2 = 1./(1./p[id].sigma2 + 1./tau2);
    float m = s2*(p[id].mu/p[id].sigma2 + x/tau2);
    float C1 = p[id].a/(p[id].a+p[id].b) *  gaus_pdf(p[id].mu, norm_scale, x);
    float C2 = p[id].b/(p[id].a+p[id].b) * 1.0/p[id].z_range;
    float normalization_constant = C1 + C2;
    C1 /= normalization_constant;
    C2 /= normalization_constant;
    float f = C1*(p[id].a+1.)/(p[id].a+p[id].b+1.) + C2*p[id].a/(p[id].a+p[id].b+1.);
    float e = C1*(p[id].a+1.)*(p[id].a+2.)/((p[id].a+p[id].b+1.)*(p[id].a+p[id].b+2.))
              + C2*p[id].a*(p[id].a+1.0f)/((p[id].a+p[id].b+1.0f)*(p[id].a+p[id].b+2.0f));

    // update parameters
    float mu_new = C1*m+C2*p[id].mu;
    p[id].sigma2 = C1*(s2 + m*m) + C2*(p[id].sigma2 + p[id].mu*p[id].mu) - mu_new*mu_new;
    p[id].mu = mu_new;
    p[id].a = (e-f)/(f-e/f);
    p[id].b = p[id].a*(1.0f-f)/f;
    memoryBarrier();
    barrier();

    // p[id].debug=p[id].mu;





    //NOT YET implemented
    // const float eta_inlier = .6f;
    // const float eta_outlier = .05f;
    // if( ((point.a / (point.a + point.b)) > eta_inlier) && (sqrt(point.sigma2) < point.z_range/seed_convergence_sigma2_thresh)) {
    //     point.is_outlier = false; // The seed converged
    // }else if((point.a-1) / (point.a + point.b - 2) < eta_outlier){ // The seed failed to converge
    //     point.is_outlier = true;
    //     // it->reinit();
    //     //TODO do a better reinit inside a point class
    //     point.a = 10;
    //     point.b = 10;
    //     point.mu = (1.0/4.0);
    //     point.z_range = (1.0/0.1);
    //     point.sigma2 = (point.z_range*point.z_range/36);
    // }
    // // if the seed has converged, we initialize a new candidate point and remove the seed
    // if(sqrt(point.sigma2) < point.z_range/seed_convergence_sigma2_thresh){
    //     point.converged = true;
    // }

}
