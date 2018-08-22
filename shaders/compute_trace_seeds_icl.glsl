#version 430

#define M_PI 3.1415926535897932384626433832795

const int MAX_RES_PER_POINT=16; //IMPORTANT to change this value also in the DepthEstimatorGL.h

layout (local_size_x = 256) in;

struct MinimalDepthFilter{
    int m_converged;
    int m_is_outlier;
    int m_initialized;
    float m_f_scale; //the scale of at the current level on the pyramid
    vec4 m_f; // heading range = Ki * (u,v,1) MAKE IT VECTOR4 so it's memory aligned for GPU usage
    int m_lvl; //pyramid lvl at which the depth filter was created
    float m_a;                 //!< a of Beta distribution: When high, probability of inlier is large.
    float m_b;                  //!< b of Beta distribution: When high, probability of outlier is large.
    float m_mu;                    //!< Mean of normal distribution.
    float m_z_range;               //!< Max range of the possible depth.
    float m_sigma2;                //!< Variance of normal distribution.
    float m_mu_denoised; //for TVL1 denoising
    float m_mu_head; //for TVL1 denoising
    vec2 m_p; //for TVL1 denoising
    float m_g; //for TVL1 denoising
    float pad;
};
struct Seed{
    int idx_keyframe; //idx in the array of keyframes which "hosts" this inmature points
    int m_time_alive;
    int m_nr_times_visible;
    float m_energyTH;
    float m_intensity[MAX_RES_PER_POINT]; //gray value for each point on the pattern
    vec2 m_normalized_grad[MAX_RES_PER_POINT];
    mat2 m_gradH; //2x2 matrix for the hessian (gx2, gxgy, gxgy, gy2), used for calculating the alpha value
    vec2 m_uv; //position in x,y of the seed in th host_frame
    vec2 m_scaled_uv; //scaled uv position depending on the pyramid level of the image
    vec2 m_idepth_minmax;
    vec2 m_best_kp; //position at which the matching energy was minimal in another frame
    vec2 m_min_uv; //uv cooresponding to the minimum depth at which to trace
    vec2 m_max_uv; //uv cooresponding to the maximum depth at which to trace
    int m_zero_grad [MAX_RES_PER_POINT]; //indicates fro each point on the pattern if it has zero grad and therefore can be skipped STORE it as int because bools are nasty for memory alignemnt on GPU as they are have different sizes in memory

    int m_active_pattern_points; //nr of points of the pattern that don't have zero_grad
    int m_lvl; //TODO why two time here andfilter?
    float m_igt_depth;
    float m_last_error;
    float m_last_idepth;
    float m_last_tau2;
    float pad2[2]; //padded until 16 now

    MinimalDepthFilter depth_filter;

    //for denoising (indexes iinto the array of points of each of the 8 neighbours)
    int  left;
    int  right;
    int  above;
    int  below;
    int  left_upper;
    int  right_upper;
    int  left_lower;
    int right_lower;

    float debug[16];
};

//you change it to  layout (binding = 0, std430) or layout (binding = 0, std140) in case stuff breaks
layout (binding = 0, std430) coherent buffer array_seeds_block{
    Seed p[];
};



//if you assign a binding to this one it doesn't read the values anymore, god knows why..
layout (std140) uniform params_block{
    float maxPerPtError;
    float slackFactor;
    float residualTH;					// higher -> less strict
    float overallEnergyTHWeight;
    float outlierTHSumComponent; 		// higher -> less strong gradient-based reweighting .
    float huberTH; // Huber Threshold
    float convergence_sigma2_thresh;      //!< threshold on depth uncertainty for convergence.
    float eta;

    float gradH_th;
    int search_epi_method; //0=bca, 1=ngf
    //pad to 16 bytes if needed  (blocks of 4 floats)
    // float pad_1;
    // float pad_2;
    //until here it's paded correctly to 16 bytes-----

    int denoise_nr_iterations;
    float denoise_depth_range;
    float denoise_lambda;
    float denoise_L;
    float denoise_tau;
    float denoise_theta;
    // float pad_1;
    // float pad_2;
}params ;

vec4 hqfilter(sampler2D samp, vec2 tc){
    // Get the size of the texture we'll be sampling from
    vec2 texSize = textureSize(samp, 0);
    // Scale our input texture coordinates up, move to center of texel
    vec2 uvScaled = tc * texSize + 0.5;
    // Find integer and fractional parts of texture coordinate
    vec2 uvInt = floor(uvScaled);
    vec2 uvFrac = fract(uvScaled);
    // Replace fractional part of texture coordinate
    uvFrac = smoothstep(0.0, 1.0, uvFrac);
    // Reassemble texture coordinate, remove bias, and
    // scale back to 0.0 to 1.0 range
    vec2 uv = (uvInt + uvFrac - 0.5) / texSize;
    // Regular texture lookup
    return texture(samp, uv);
}

//https://www.boost.org/doc/libs/1_47_0/libs/math/doc/sf_and_dist/html/math_toolkit/dist/dist_ref/dists/normal_dist.html
float gaus_pdf(float mean, float sd, float x){
    return exp(- (x-mean)*(x-mean)/(2*sd)*(2*sd)  )  / (sd*sqrt(2*M_PI));
}

float mean(vec2 vec){
    return (vec.x+vec.y)*0.5;
}


uniform sampler2D gray_with_gradients_img_sampler; //contains gray val and gradx and grady
layout(binding=2, rgba32f) uniform writeonly image2D debug;

uniform vec2 frame_size; //x, y
uniform mat4 tf_cur_host;
uniform mat4 tf_host_cur;
uniform mat3 K;
uniform mat3 KRKi_cr;
uniform vec3 Kt_cr;
uniform float px_error_angle;
uniform vec2 pattern_rot_offsets[MAX_RES_PER_POINT];
uniform int pattern_rot_nr_points;



void main(void) {
    int min_border=20;

    int id = int(gl_GlobalInvocationID.x);

    p[id].m_time_alive++;

    // // check if point is visible in the current image
    const vec3 p_backproj_xyz= p[id].depth_filter.m_f.xyz * 1.0f/ p[id].depth_filter.m_mu;
    const vec4 p_backproj_xyzw=vec4(p_backproj_xyz.x,p_backproj_xyz.y,p_backproj_xyz.z,1.0);
    const vec4 xyz_f_xyzw = tf_cur_host*  p_backproj_xyzw ;
    const vec3 xyz_f=xyz_f_xyzw.xyz/xyz_f_xyzw.w;
    if(xyz_f.z < 0.0)  {
        return; // TODO in gl this is a return
    }


    const vec3 kp_c = K * xyz_f;
    const vec2 kp_c_h=kp_c.xy/kp_c.z;
    if ( kp_c_h.x < min_border || kp_c_h.x >= frame_size.x-min_border || kp_c_h.y < min_border || kp_c_h.y >= frame_size.y-min_border ) {
        return; // TODO in gl this is a return
    }

    p[id].m_nr_times_visible++;




    //point is visible
    // point.last_visible_frame=frames[i].frame_id;

    //update inverse depth coordinates for min and max
    p[id].m_idepth_minmax.x = p[id].depth_filter.m_mu + sqrt(p[id].depth_filter.m_sigma2);
    p[id].m_idepth_minmax.y = max(p[id].depth_filter.m_mu - sqrt(p[id].depth_filter.m_sigma2), 0.00000001f);
    // memoryBarrier();
    // barrier();
    // memoryBarrier();

    //search epiline-----------------------------------------------------------------------
   // search_epiline_ncc (point, frame, KRKi_cr, Kt_cr );
    // search_epiline_bca (point, frames[i], KRKi_cr, Kt_cr, affine_cr);
    float idepth_mean = mean(p[id].m_idepth_minmax);
    vec3 pr = KRKi_cr * vec3(p[id].m_uv, 1);
    vec3 ptpMean = pr + Kt_cr*idepth_mean;
    vec3 ptpMin = pr + Kt_cr*p[id].m_idepth_minmax.x;
    vec3 ptpMax = pr + Kt_cr*p[id].m_idepth_minmax.y;
    vec2 uvMean = ptpMean.xy/ptpMean.z;
    vec2 uvMin = ptpMin.xy/ptpMin.z;
    vec2 uvMax = ptpMax.xy/ptpMax.z;


    vec2 epi_line = uvMax - uvMin;
    float norm_epi = max(1e-5f,length(epi_line));
    vec2 epi_dir = epi_line / norm_epi;
    const float  half_length = 0.5f * norm_epi;

    //the epiline is too long, and it would take too much time to search
    if(norm_epi>200){
        p[id].depth_filter.m_is_outlier=1; //discard the point
        return;
    }

    //if the alive time is bigger than 15 and it was visible less than 5 frames , we ignore this points
    if(p[id].m_time_alive>10 && p[id].m_nr_times_visible<5){
        p[id].depth_filter.m_is_outlier=1; //discard the point
        return;
    }


    vec2 bestKp=vec2(-1.0,-1.0);
    float bestEnergy = 1e10;
    float second_best_energy = 1e10;
    vec2 second_best_kp = vec2(-1.0,-1.0);


    int nr_times_ambigous=0;


    float step_size=0.7f;
    for(float l = -half_length; l <= half_length; l += step_size)
    {
        float energy = 0;
        vec2 kp = uvMean + l*epi_dir;

        if( ( kp.x >= (frame_size.x-min_border) )  || ( kp.y >= (frame_size.y-min_border) ) || ( kp.x < min_border ) || ( kp.y < min_border) ){
            continue ;
        }



        for(int idx=0;idx<pattern_rot_nr_points; ++idx){
            //float hitColor = getInterpolatedElement31(frame->dI, (float)(kp(0)+rotatetPattern[idx][0]), (float)(kp(1)+rotatetPattern[idx][1]), wG[0]);
            vec2 offset=pattern_rot_offsets[idx];
            // float hit_color=texture(gray_img_sampler, vec2( (kp.x + offset.x+0.5)/640.0, (kp.y + offset.y+0.5)/480.0)).x;
            // float hit_color=texelFetch(gray_img_sampler, ivec2( (kp.x + offset.x), (kp.y + offset.y)), 0).x;
            // float hit_color=texture_interpolate(frames[i].gray, kp.x+offset.x, kp.y+offset.y , InterpolationType::LINEAR);
            // if(!std::isfinite(hit_color)) {energy-=1e5; continue;}

            //for the case when the image is padded
            // float hit_color=texture(gray_img_sampler, vec2( (kp.x + offset.x)/1024, ( 1024-480+  kp.y + offset.y)/1024)).x;



            //check if the gradient is perpendicular to the epipolar direction in which case we set the points as outlier because we won't be able to estimate the correct depth
            vec3 hit_color_and_grads=hqfilter(gray_with_gradients_img_sampler, vec2( (kp.x + offset.x+0.5)/frame_size.x, (kp.y + offset.y+0.5)/frame_size.y)).xyz;
            float hit_color=hit_color_and_grads.x;
            vec2 grads=hit_color_and_grads.yz;
            //grads stores the gradient direction in x and in y
            float along_epi=1-abs(dot(epi_dir, normalize(grads))); //increases when the points are ambigous on the epiline
            float along_epi_agresiv=0.8; //the smaller the value the more aggresive we are in discarding points
            float min_grad=50; //to avoid discarding points with really small gradients
            if(along_epi>along_epi_agresiv && length(grads)>min_grad){
                nr_times_ambigous++;
                // p[id].depth_filter.m_is_outlier=1; //discard the point
                // return;
            }

            //(Brightness Constancy Assumption) high qualty filter from openglsuperbible
            // float hit_color=hqfilter(gray_with_gradients_img_sampler, vec2( (kp.x + offset.x+0.5)/frame_size.x, (kp.y + offset.y+0.5)/frame_size.y)).x;
            const float residual = hit_color - (p[id].m_intensity[idx]);
            float hw = abs(residual) < params.huberTH ? 1 : params.huberTH / abs(residual);
            energy += hw *residual*residual*(2-hw);

            // //Uni modal NGF
            // vec3 hit_color_and_grads=hqfilter(gray_with_gradients_img_sampler, vec2( (kp.x + offset.x+0.5)/frame_size.x, (kp.y + offset.y+0.5)/frame_size.y)).xyz;
            // vec2 grads=hit_color_and_grads.yz;
            // grads /= sqrt(dot(grads,grads)+params.eta);
            // const float nn = dot(grads,p[id].m_normalized_grad[idx]);
            // const float residual = max(0.f,min(1.f,nn < 0 ? 1.f : 1-nn ));// uni modal ngf
            // //const float residual = std::max<float>(0.f,std::min<float>(1.f,1.f-nn*nn)); // original ngf residual
            // const float fr = abs(residual);
            // float hw = fr < params.huberTH ? 1 : params.huberTH / fr;
            // energy += hw *residual*residual*(2-hw);


        }


        //store also the second best energy
        if ( energy < bestEnergy ){
            //got a new global maximum
            second_best_energy=bestEnergy;
            second_best_kp=bestKp;
            bestKp = kp;
            bestEnergy = energy;
        }else if(energy < second_best_energy && energy != second_best_energy){
            //got a new second max that is not as low as the bestEnergy but still lowers than the than the previous second best
            second_best_energy=energy;
            second_best_kp=kp;
        }



    }


    float nr_pixel_acceses_along_epi=norm_epi/step_size*pattern_rot_nr_points;
    //nr_times_along_epi epi is big for the vertical lines that we can triangulate well, and small for the horzontal lines which ar ambigous
    if(nr_times_ambigous/nr_pixel_acceses_along_epi>0.9 && nr_times_ambigous!=0){ //the higher the value the more aggresive we are in dropping points
        p[id].depth_filter.m_is_outlier=1; //discard the point
        imageStore(debug, ivec2(p[id].m_uv) , vec4(255,0,0,255) );
        return;
    }

    // if(nr_times_along_epi<10){ //the higher the value the more aggresive we are in dropping points
    //     p[id].depth_filter.m_is_outlier=1; //discard the point
    //     return;
    // }

    imageStore(debug, ivec2(p[id].m_uv) , vec4(0,0,nr_times_ambigous/nr_pixel_acceses_along_epi,255) );


    // //store in debug the disparity
    // float disparity=(p[id].m_uv.x-bestKp.x)/255.0;
    // float pseudo_disp=distance(p[id].m_uv, bestKp)/100.0;
    // imageStore(debug, ivec2(p[id].m_uv) , vec4(0,pseudo_disp,0,255) );

    // //store in debug the best kp
    // imageStore(debug, ivec2(bestKp) , vec4(255,0,0,255) );



    p[id].m_last_error=bestEnergy;

    //check that the best energy is different enough from the second best
    int is_outlier=0;
    if(bestEnergy*1.1>second_best_energy && second_best_energy!=1e10
        && distance(bestKp,second_best_kp)>3 ){
        is_outlier=1;
         p[id].depth_filter.m_is_outlier=1;
    }


    if ( bestEnergy > p[id].m_energyTH * 1.2f ) {
        is_outlier=1;
        // p[id].depth_filter.m_is_outlier=1;
    }
    else
    {
        // vec2 epi_dir_inv=vec2(epi_dir.y,-epi_dir.x);
        // float a = epi_dir * p[id].gradH * epi_dir;
        // float b = epi_dir_inv * point.gradH * epi_dir_inv;
        // float errorInPixel = 0.2f + 0.2f * (a+b) / a; // WO kommt das her? Scheint nicht zu NGF zu passen !
        float errorInPixel=0.0f;

        if( epi_dir.x*epi_dir.x>epi_dir.y*epi_dir.y )
        {
            p[id].m_idepth_minmax.x = (pr.z*(bestKp.x-errorInPixel*epi_dir.x) - pr.x) / (Kt_cr.x - Kt_cr.z*(bestKp.x-errorInPixel*epi_dir.x));
            p[id].m_idepth_minmax.y = (pr.z*(bestKp.x+errorInPixel*epi_dir.x) - pr.x) / (Kt_cr.x - Kt_cr.z*(bestKp.x+errorInPixel*epi_dir.x));
        }
        else
        {
            p[id].m_idepth_minmax.x = (pr.z*(bestKp.y-errorInPixel*epi_dir.y) - pr.y) / (Kt_cr.y - Kt_cr.z*(bestKp.y-errorInPixel*epi_dir.y));
            p[id].m_idepth_minmax.y = (pr.z*(bestKp.y+errorInPixel*epi_dir.y) - pr.y) / (Kt_cr.y - Kt_cr.z*(bestKp.y+errorInPixel*epi_dir.y));
        }
        // memoryBarrier();
        // barrier();
        // memoryBarrier();
        if(p[id].m_idepth_minmax.x > p[id].m_idepth_minmax.y) {
            // std::swap<float>(point.idepth_min, point.idepth_max);
            float tmp=p[id].m_idepth_minmax.x;
            p[id].m_idepth_minmax.x=p[id].m_idepth_minmax.y;
            p[id].m_idepth_minmax.y=tmp;
        }
        // memoryBarrier();
        // barrier();
        // memoryBarrier();
    }
    // memoryBarrier();
    // barrier();
    // memoryBarrier();















    //set this to 1 so that we don't represent the point in the mesh
    // if(is_outlier==1){
    //     p[id].depth_filter.m_is_outlier=1;
    // }







    float idepth = -1;
    float z = 0;
    idepth = max(1e-5f,.5*(p[id].m_idepth_minmax.x+p[id].m_idepth_minmax.y));
    z = 1.0f/idepth;
    if ( idepth<0.00000001 || idepth>99999999 || is_outlier==1 ) {
        p[id].depth_filter.m_b++; // increase outlier probability when no match was found
        return;
    }



    // update_idepth(point,tf_host_cur, z, px_error_angle);

    // compute tau----------------------------------------------------------------------------
    // double tau = compute_tau(tf_host_cur, point.f, z, px_error_angle);
    vec3 t=  vec3(tf_host_cur[0][3], tf_host_cur[1][3], tf_host_cur[2][3]);
    // Eigen::Vector3f t(tf_host_cur.translation());
    vec3 a = p[id].depth_filter.m_f.xyz*z-t;
    float t_norm = length(t);
    float a_norm = length(a);
    float alpha = acos(dot(p[id].depth_filter.m_f.xyz,t)/t_norm); // dot product
    float beta = acos(dot(a,-t)/(t_norm*a_norm)); // dot product
    float beta_plus = beta + px_error_angle;
    float gamma_plus = M_PI-alpha-beta_plus; // triangle angles sum to PI
    float z_plus = t_norm*sin(beta_plus)/sin(gamma_plus); // law of sines
    float tau= (z_plus - z); // tau
    float tau_inverse = 0.5 * (1.0/max(0.0000001, z-tau) - 1.0/(z+tau));

    // update the estimate--------------------------------------------------
    float x=1.0/z;
    float tau2=tau_inverse*tau_inverse;
    // updateSeed(point, 1.0/z, tau_inverse*tau_inverse);
    float norm_scale = sqrt(p[id].depth_filter.m_sigma2 + tau2);
    float s2 = 1./(1./p[id].depth_filter.m_sigma2 + 1./tau2);
    float m = s2*(p[id].depth_filter.m_mu/p[id].depth_filter.m_sigma2 + x/tau2);
    float C1 = p[id].depth_filter.m_a/(p[id].depth_filter.m_a+p[id].depth_filter.m_b) * gaus_pdf(p[id].depth_filter.m_mu, norm_scale, x);
    float C2 = p[id].depth_filter.m_b/(p[id].depth_filter.m_a+p[id].depth_filter.m_b) * 1./p[id].depth_filter.m_z_range;
    float normalization_constant = C1 + C2;
    C1 /= normalization_constant;
    C2 /= normalization_constant;
    float f = C1*(p[id].depth_filter.m_a+1.)/(p[id].depth_filter.m_a+p[id].depth_filter.m_b+1.) + C2*p[id].depth_filter.m_a/(p[id].depth_filter.m_a+p[id].depth_filter.m_b+1.);
    float e = C1*(p[id].depth_filter.m_a+1.)*(p[id].depth_filter.m_a+2.)/((p[id].depth_filter.m_a+p[id].depth_filter.m_b+1.)*(p[id].depth_filter.m_a+p[id].depth_filter.m_b+2.))
              + C2*p[id].depth_filter.m_a*(p[id].depth_filter.m_a+1.0f)/((p[id].depth_filter.m_a+p[id].depth_filter.m_b+1.0f)*(p[id].depth_filter.m_a+p[id].depth_filter.m_b+2.0f));
    // update parameters
    float mu_new = C1*m+C2*p[id].depth_filter.m_mu;
    p[id].depth_filter.m_sigma2 = C1*(s2 + m*m) + C2*(p[id].depth_filter.m_sigma2 + p[id].depth_filter.m_mu*p[id].depth_filter.m_mu) - mu_new*mu_new;
    p[id].depth_filter.m_mu = mu_new;
    p[id].depth_filter.m_a = (e-f)/(f-e/f);
    // memoryBarrier();
    // barrier();
    // memoryBarrier();
    p[id].depth_filter.m_b = p[id].depth_filter.m_a*(1.0f-f)/f;
    // memoryBarrier();
    // barrier(); //TODO add again the barrier
    // memoryBarrier();

    // // // not implemented in opengl
    // const float eta_inlier = .6f;
    // const float eta_outlier = .05f;
    // if( ((p[id].a / (p[id].a + p[id].b)) > eta_inlier) && (sqrt(p[id].sigma2) < p[id].z_range/params.convergence_sigma2_thresh)) {
    //     p[id].is_outlier = 0; // The seed converged
    // }else if((p[id].a-1) / (p[id].a + p[id].b - 2) < eta_outlier){ // The seed failed to converge
    //     p[id].is_outlier = 1;
    //     // it->reinit();
    //     //TODO do a better reinit inside a point class
    //     p[id].a = 10;
    //     p[id].b = 10;
    //     p[id].mu = (1.0/4.0);
    //     p[id].z_range = (1.0/0.1);
    //     p[id].sigma2 = (p[id].z_range*p[id].z_range/36);
    // }
    // // if the seed has converged, we initialize a new candidate point and remove the seed
    // if(sqrt(p[id].sigma2) < p[id].z_range/params.convergence_sigma2_thresh){
    //     p[id].converged = 1;
    // }

}
