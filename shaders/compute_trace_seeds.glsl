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
    float m_alpha;                 //!< a of Beta distribution: When high, probability of inlier is large.
    float m_beta;                  //!< b of Beta distribution: When high, probability of outlier is large.
    float m_mu;                    //!< Mean of normal distribution.
    float m_z_range;               //!< Max range of the possible depth.
    float m_sigma2;                //!< Variance of normal distribution.
    float pad[2]; //padded to 16 until now
};
struct Seed{
    int idx_keyframe; //idx in the array of keyframes which "hosts" this inmature points
    float m_energyTH;
    float pad[2]; //padded to 16 until now
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

    float debug[16];
};
struct EpiData{
    // mat4 tf_cur_host; //the of corresponds to a 4x4 matrix
    // mat4 tf_host_cur;
    mat4 KRKi_cr;
    // vec3 Kt_cr;
    // vec2 pattern_rot_offsets[MAX_RES_PER_POINT];
};

//you change it to  layout (binding = 0, std430) or layout (binding = 0, std140) in case stuff breaks
layout (binding = 0, std430) coherent buffer array_seeds_block{
    Seed p[];
};

layout (binding = 1, std430) coherent buffer array_epidata_block{
    EpiData e[];
};


//if you assign a binding to this one it doesn't read the values anymore, god knows why..
layout (std140) uniform params_block{
    float maxPerPtError;
    float slackFactor;
    float outlierTH;					// higher -> less strict
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
    float pad_1;
    float pad_2;
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

uniform int pattern_rot_nr_points;
uniform float px_error_angle;
uniform mat3 K;

void main(void) {
    int id = int(gl_GlobalInvocationID.x);

    if(id==0){
        p[id].debug[0]=e[0].KRKi_cr[0][0]; //[col][row]
        p[id].debug[1]=e[0].KRKi_cr[1][0];
        p[id].debug[2]=e[0].KRKi_cr[2][0];

        p[id].debug[3]=e[0].KRKi_cr[0][1];
        p[id].debug[4]=e[0].KRKi_cr[1][1];
        p[id].debug[5]=e[0].KRKi_cr[2][1];

        p[id].debug[6]=e[0].KRKi_cr[0][2];
        p[id].debug[7]=e[0].KRKi_cr[1][2];
        p[id].debug[8]=e[0].KRKi_cr[2][2];

        p[id].debug[9]=999999;

        // p[id].debug[6]=uvMean.x; // Huber Threshold
        // p[id].debug[7]=uvMean.y;
        // p[id].debug[8]=999999;      //!< threshold on depth uncertainty for convergence.
        // p[id].debug[5]=p[id].depth_filter.m_mu;
        // p[id].debug[6]=p[id].depth_filter.m_sigma2;
        // p[id].debug[7]=ptpMean.x;
        // p[id].debug[8]=ptpMean.y;
        // p[id].debug[9]=ptpMean.z;
        //
        // p[id].debug[10]=pr.x;
        // p[id].debug[11]=pr.y;
        // p[id].debug[12]=pr.z;
        //
        // // p[id].debug[13]=vec3(p[id].m_uv, 1).x;
        // // p[id].debug[14]=vec3(p[id].m_uv, 1).y;
        // // p[id].debug[15]=vec3(p[id].m_uv, 1).z;
        //
        // p[id].debug[13]=k;
    }

   //  vec2 frame_size;
   //  frame_size=textureSize(gray_with_gradients_img_sampler, 0);
   //
   //  //which keyframe host this seed?
   //  int k=p[id].idx_keyframe;
   //
   //  // // check if point is visible in the current image
   //  const vec3 p_backproj_xyz= p[id].depth_filter.m_f.xyz * 1.0f/ p[id].depth_filter.m_mu;
   //  const vec4 p_backproj_xyzw=vec4(p_backproj_xyz.x,p_backproj_xyz.y,p_backproj_xyz.z,1.0);
   //  const vec4 xyz_f_xyzw = e[k].tf_cur_host*  p_backproj_xyzw ;
   //  const vec3 xyz_f=xyz_f_xyzw.xyz/xyz_f_xyzw.w;
   //  if(xyz_f.z < 0.0)  {
   //      return; //behind
   //  }
   //
   //
   //  const vec3 kp_c = K * xyz_f;
   //  const vec2 kp_c_h=kp_c.xy/kp_c.z;
   //  if ( kp_c_h.x < 0 || kp_c_h.x >= frame_size.x || kp_c_h.y < 0 || kp_c_h.y >= frame_size.y ) {
   //      return; //outside of image
   //  }
   //
   //
   //  //point is visible
   //  // point.last_visible_frame=frames[i].frame_id;
   //
   //  //update inverse depth coordinates for min and max
   //  p[id].m_idepth_minmax.x = p[id].depth_filter.m_mu + sqrt(p[id].depth_filter.m_sigma2);
   //  p[id].m_idepth_minmax.y = max(p[id].depth_filter.m_mu - sqrt(p[id].depth_filter.m_sigma2), 0.00000001f);
   //  // memoryBarrier();
   //  // barrier();
   //  // memoryBarrier();
   //
   //  //search epiline-----------------------------------------------------------------------
   // // search_epiline_ncc (point, frame, KRKi_cr, Kt_cr );
   //  // search_epiline_bca (point, frames[i], KRKi_cr, Kt_cr, affine_cr);
   //  float idepth_mean = mean(p[id].m_idepth_minmax);
   //  vec3 pr =e[k].KRKi_cr * vec3(p[id].m_uv, 1);
   //  vec3 ptpMean = pr + e[k].Kt_cr*idepth_mean;
   //  vec3 ptpMin = pr + e[k].Kt_cr*p[id].m_idepth_minmax.x;
   //  vec3 ptpMax = pr + e[k].Kt_cr*p[id].m_idepth_minmax.y;
   //  vec2 uvMean = ptpMean.xy/ptpMean.z;
   //  vec2 uvMin = ptpMin.xy/ptpMin.z;
   //  vec2 uvMax = ptpMax.xy/ptpMax.z;
   //
   //
   //
   //  //debug the point
   //  // if(id==0){
   //  //     p[id].debug[0]=p[id].m_uv.x;
   //  //     p[id].debug[1]=p[id].m_uv.y;
   //  //     p[id].debug[2]=uvMean.x; 		// higher -> less strong gradient-based reweighting .
   //  //     p[id].debug[3]=uvMean.y; // Huber Threshold
   //  //     p[id].debug[4]=idepth_mean;      //!< threshold on depth uncertainty for convergence.
   //  //     p[id].debug[5]=p[id].depth_filter.m_mu;
   //  //     p[id].debug[6]=p[id].depth_filter.m_sigma2;
   //  //     p[id].debug[7]=ptpMean.x;
   //  //     p[id].debug[8]=ptpMean.y;
   //  //     p[id].debug[9]=ptpMean.z;
   //  //
   //  //     p[id].debug[10]=pr.x;
   //  //     p[id].debug[11]=pr.y;
   //  //     p[id].debug[12]=pr.z;
   //  //
   //  //     // p[id].debug[13]=vec3(p[id].m_uv, 1).x;
   //  //     // p[id].debug[14]=vec3(p[id].m_uv, 1).y;
   //  //     // p[id].debug[15]=vec3(p[id].m_uv, 1).z;
   //  //
   //  //     p[id].debug[13]=k;
   //  // }
   //
   //  //debug the KRKi_cr
   //  if(id==0){
   //      p[id].debug[0]=e[0].KRKi_cr[0][0]; //[col][row]
   //      p[id].debug[1]=e[0].KRKi_cr[1][0];
   //      p[id].debug[2]=e[0].KRKi_cr[2][0];
   //
   //      p[id].debug[3]=e[0].KRKi_cr[0][1];
   //      p[id].debug[4]=e[0].KRKi_cr[1][1];
   //      p[id].debug[5]=e[0].KRKi_cr[2][1];
   //
   //      p[id].debug[6]=e[0].KRKi_cr[0][2];
   //      p[id].debug[7]=e[0].KRKi_cr[1][2];
   //      p[id].debug[8]=e[0].KRKi_cr[2][2];
   //
   //      p[id].debug[9]=999999;
   //
   //      // p[id].debug[6]=uvMean.x; // Huber Threshold
   //      // p[id].debug[7]=uvMean.y;
   //      // p[id].debug[8]=999999;      //!< threshold on depth uncertainty for convergence.
   //      // p[id].debug[5]=p[id].depth_filter.m_mu;
   //      // p[id].debug[6]=p[id].depth_filter.m_sigma2;
   //      // p[id].debug[7]=ptpMean.x;
   //      // p[id].debug[8]=ptpMean.y;
   //      // p[id].debug[9]=ptpMean.z;
   //      //
   //      // p[id].debug[10]=pr.x;
   //      // p[id].debug[11]=pr.y;
   //      // p[id].debug[12]=pr.z;
   //      //
   //      // // p[id].debug[13]=vec3(p[id].m_uv, 1).x;
   //      // // p[id].debug[14]=vec3(p[id].m_uv, 1).y;
   //      // // p[id].debug[15]=vec3(p[id].m_uv, 1).z;
   //      //
   //      // p[id].debug[13]=k;
   //  }
   //
   //
   //

    // vec2 epi_line = uvMax - uvMin;
    // float norm_epi = max(1e-5f,length(epi_line));
    // vec2 epi_dir = epi_line / norm_epi;
    // const float  half_length = 0.5f * norm_epi;
    //
    // vec2 bestKp=vec2(-1.0,-1.0);
    // float bestEnergy = 1e10;
    //
    //
    // for(float l = -half_length; l <= half_length; l += 0.7f)
    // {
    //     float energy = 0;
    //     vec2 kp = uvMean + l*epi_dir;
    //
    //     //pattern will be a bit outside of the image so we ignore those points
    //     if( ( kp.x >= (frame_size.x-20) )  || ( kp.y >= (frame_size.y-20) ) || ( kp.x < 20 ) || ( kp.y < 20) ){
    //         continue;
    //     }
    //
    //     for(int idx=0;idx<pattern_rot_nr_points; ++idx){
    //         //float hitColor = getInterpolatedElement31(frame->dI, (float)(kp(0)+rotatetPattern[idx][0]), (float)(kp(1)+rotatetPattern[idx][1]), wG[0]);
    //         vec2 offset=e[k].pattern_rot_offsets[idx];
    //         // float hit_color=texture(gray_img_sampler, vec2( (kp.x + offset.x+0.5)/640.0, (kp.y + offset.y+0.5)/480.0)).x;
    //         // float hit_color=texelFetch(gray_img_sampler, ivec2( (kp.x + offset.x), (kp.y + offset.y)), 0).x;
    //         // float hit_color=texture_interpolate(frames[i].gray, kp.x+offset.x, kp.y+offset.y , InterpolationType::LINEAR);
    //         // if(!std::isfinite(hit_color)) {energy-=1e5; continue;}
    //
    //         //for the case when the image is padded
    //         // float hit_color=texture(gray_img_sampler, vec2( (kp.x + offset.x)/1024, ( 1024-480+  kp.y + offset.y)/1024)).x;
    //
    //         //high qualty filter from openglsuperbible
    //         float hit_color=hqfilter(gray_with_gradients_img_sampler, vec2( (kp.x + offset.x+0.5)/frame_size.x, (kp.y + offset.y+0.5)/frame_size.y)).x;
    //
    //         // float hit_color=0.0;
    //
    //         const float residual = hit_color - p[id].m_intensity[idx];
    //
    //         float hw = abs(residual) < params.huberTH ? 1 : params.huberTH / abs(residual);
    //         energy += hw *residual*residual*(2-hw);
    //     }
    //     if ( energy < bestEnergy )
    //     {
    //         bestKp = kp; bestEnergy = energy;
    //     }
    // }
    //
    //
    // // if ( bestEnergy > p[id].energyTH * 1.2f ) {
    // //     p[id].lastTraceStatus = STATUS_OUTLIER;
    // // }
    // // else
    // // {
    //     // vec2 epi_dir_inv=vec2(epi_dir.y,-epi_dir.x);
    //     // float a = epi_dir * p[id].gradH * epi_dir;
    //     // float b = epi_dir_inv * point.gradH * epi_dir_inv;
    //     // float errorInPixel = 0.2f + 0.2f * (a+b) / a; // WO kommt das her? Scheint nicht zu NGF zu passen !
    //     float errorInPixel=0.0f;
    //
    //     if( epi_dir.x*epi_dir.x>epi_dir.y*epi_dir.y )
    //     {
    //         p[id].m_idepth_minmax.x = (pr.z*(bestKp.x-errorInPixel*epi_dir.x) - pr.x) / (e[k].Kt_cr.x - e[k].Kt_cr.z*(bestKp.x-errorInPixel*epi_dir.x));
    //         p[id].m_idepth_minmax.y = (pr.z*(bestKp.x+errorInPixel*epi_dir.x) - pr.x) / (e[k].Kt_cr.x - e[k].Kt_cr.z*(bestKp.x+errorInPixel*epi_dir.x));
    //     }
    //     else
    //     {
    //         p[id].m_idepth_minmax.x = (pr.z*(bestKp.y-errorInPixel*epi_dir.y) - pr.y) / (e[k].Kt_cr.y - e[k].Kt_cr.z*(bestKp.y-errorInPixel*epi_dir.y));
    //         p[id].m_idepth_minmax.y = (pr.z*(bestKp.y+errorInPixel*epi_dir.y) - pr.y) / (e[k].Kt_cr.y - e[k].Kt_cr.z*(bestKp.y+errorInPixel*epi_dir.y));
    //     }
    //     // memoryBarrier();
    //     // barrier();
    //     // memoryBarrier();
    //     if(p[id].m_idepth_minmax.x > p[id].m_idepth_minmax.y) {
    //         // std::swap<float>(point.idepth_min, point.idepth_max);
    //         float tmp=p[id].m_idepth_minmax.x;
    //         p[id].m_idepth_minmax.x=p[id].m_idepth_minmax.y;
    //         p[id].m_idepth_minmax.y=tmp;
    //     }
    //     // p[id].lastTraceStatus = STATUS_GOOD;
    //     // memoryBarrier();
    //     // barrier();
    //     // memoryBarrier();
    // // }
    // // memoryBarrier();
    // // barrier();
    // // memoryBarrier();
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    // float idepth = -1;
    // float z = 0;
    // idepth = max(1e-5f, mean(p[id].m_idepth_minmax) );
    // z = 1.0f/idepth;
    //
    // // memoryBarrier();
    // // barrier();
    // // memoryBarrier();
    //
    //
    // // update_idepth(point,tf_host_cur, z, px_error_angle);
    //
    // // compute tau----------------------------------------------------------------------------
    // // double tau = compute_tau(tf_host_cur, point.f, z, px_error_angle);
    // vec3 t=  vec3(e[k].tf_host_cur[0][3], e[k].tf_host_cur[1][3], e[k].tf_host_cur[2][3]);
    // // Eigen::Vector3f t(tf_host_cur.translation());
    // vec3 a = p[id].depth_filter.m_f.xyz*z-t;
    // float t_norm = length(t);
    // float a_norm = length(a);
    // float alpha = acos(dot(p[id].depth_filter.m_f.xyz,t)/t_norm); // dot product
    // float beta = acos(dot(a,-t)/(t_norm*a_norm)); // dot product
    // float beta_plus = beta + px_error_angle;
    // float gamma_plus = M_PI-alpha-beta_plus; // triangle angles sum to PI
    // float z_plus = t_norm*sin(beta_plus)/sin(gamma_plus); // law of sines
    // float tau= (z_plus - z); // tau
    // float tau_inverse = 0.5 * (1.0/max(0.0000001, z-tau) - 1.0/(z+tau));
    //
    // // update the estimate--------------------------------------------------
    // float x=1.0/z;
    // float tau2=tau_inverse*tau_inverse;
    // // updateSeed(point, 1.0/z, tau_inverse*tau_inverse);
    // float norm_scale = sqrt(p[id].depth_filter.m_sigma2 + tau2);
    // float s2 = 1./(1./p[id].depth_filter.m_sigma2 + 1./tau2);
    // float m = s2*(p[id].depth_filter.m_mu/p[id].depth_filter.m_sigma2 + x/tau2);
    // float C1 = p[id].depth_filter.m_alpha/(p[id].depth_filter.m_alpha+p[id].depth_filter.m_beta) * gaus_pdf(p[id].depth_filter.m_mu, norm_scale, x);
    // float C2 = p[id].depth_filter.m_beta/(p[id].depth_filter.m_alpha+p[id].depth_filter.m_beta) * 1./p[id].depth_filter.m_z_range;
    // float normalization_constant = C1 + C2;
    // C1 /= normalization_constant;
    // C2 /= normalization_constant;
    // float f = C1*(p[id].depth_filter.m_alpha+1.)/(p[id].depth_filter.m_alpha+p[id].depth_filter.m_beta+1.) + C2*p[id].depth_filter.m_alpha/(p[id].depth_filter.m_alpha+p[id].depth_filter.m_beta+1.);
    // float e = C1*(p[id].depth_filter.m_alpha+1.)*(p[id].depth_filter.m_alpha+2.)/((p[id].depth_filter.m_alpha+p[id].depth_filter.m_beta+1.)*(p[id].depth_filter.m_alpha+p[id].depth_filter.m_beta+2.))
    //           + C2*p[id].depth_filter.m_alpha*(p[id].depth_filter.m_alpha+1.0f)/((p[id].depth_filter.m_alpha+p[id].depth_filter.m_beta+1.0f)*
    //           (p[id].depth_filter.m_alpha+p[id].depth_filter.m_beta+2.0f));
    // // update parameters
    // float mu_new = C1*m+C2*p[id].depth_filter.m_mu;
    // p[id].depth_filter.m_sigma2 = C1*(s2 + m*m) + C2*(p[id].depth_filter.m_sigma2 + p[id].depth_filter.m_mu*p[id].depth_filter.m_mu) - mu_new*mu_new;
    // p[id].depth_filter.m_mu = mu_new;
    // p[id].depth_filter.m_alpha = (e-f)/(f-e/f);
    // p[id].depth_filter.m_beta = p[id].depth_filter.m_alpha*(1.0f-f)/f;
    //
    // // memoryBarrier();
    // // barrier(); //TODO add again the barrier
    // // memoryBarrier();
    //
    // // // // TODO not implemented in opengl
    // // const float eta_inlier = .6f;
    // // const float eta_outlier = .05f;
    // // if( ((p[id].a / (p[id].a + p[id].b)) > eta_inlier) && (sqrt(p[id].sigma2) < p[id].z_range/params.convergence_sigma2_thresh)) {
    // //     p[id].is_outlier = 0; // The seed converged
    // // }else if((p[id].a-1) / (p[id].a + p[id].b - 2) < eta_outlier){ // The seed failed to converge
    // //     p[id].is_outlier = 1;
    // //     // it->reinit();
    // //     //TODO do a better reinit inside a point class
    // //     p[id].a = 10;
    // //     p[id].b = 10;
    // //     p[id].mu = (1.0/4.0);
    // //     p[id].z_range = (1.0/0.1);
    // //     p[id].sigma2 = (p[id].z_range*p[id].z_range/36);
    // // }
    // // // if the seed has converged, we initialize a new candidate point and remove the seed
    // // if(sqrt(p[id].sigma2) < p[id].z_range/params.convergence_sigma2_thresh){
    // //     p[id].converged = 1;
    // // }

}
