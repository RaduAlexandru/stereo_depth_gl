#version 430

#define M_PI 3.1415926535897932384626433832795

const int MAX_RES_PER_POINT=16; //IMPORTANT to change this value also in the DepthEstimatorGL.h

layout (local_size_x = 16, local_size_y=16) in;

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
    // float eta;
    float pad;

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
uniform float focal_length;
uniform vec2 pattern_rot_offsets[MAX_RES_PER_POINT];
uniform int pattern_rot_nr_points;
uniform float ngf_eta;



void main(void) {
    int min_border=20;

    ivec2 img_coords = ivec2(gl_GlobalInvocationID.xy);

    vec3 hit_color_and_grads=hqfilter(gray_with_gradients_img_sampler, vec2( (img_coords.x +0.5)/frame_size.x, (img_coords.y +0.5)/frame_size.y)).xyz;
    vec2 grads=hit_color_and_grads.yz;
    grads /= sqrt(dot(grads,grads)+10);

    vec2 to_show=grads;
    imageStore(debug, img_coords , vec4(to_show,0,255) );



}
