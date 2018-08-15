#version 430

#define M_PI 3.1415926535897932384626433832795

const int MAX_RES_PER_POINT=16; //IMPORTANT to change this value also in the DepthEstimatorGL.h

layout (local_size_x = 32, local_size_y = 16) in;

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

//you change it to  layout (binding = 0, std430) or layout (binding = 0, std140) in case stuff breaks
layout (binding = 0, std430) coherent buffer array_seeds_block{
    Seed p[];
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

uniform sampler2D hessian_blurred_sampler; //contains gray val and gradx and grady
uniform sampler2D gray_with_gradients_img_sampler; //contains gray val and gradx and grady
layout(binding = 1) uniform atomic_uint nr_seeds_created;
layout(binding=2, rgba32f) uniform writeonly image2D debug;


uniform vec2 pattern_rot_offsets[MAX_RES_PER_POINT];
uniform int pattern_rot_nr_points;
uniform mat3 K;
uniform mat3 K_inv;
uniform float min_starting_depth;
uniform float mean_starting_depth;
uniform int seeds_start_idx;
uniform int idx_keyframe;


Seed create_seed(ivec2 img_coords, vec3 hessian){
    Seed s;


    //at position nr_seeds_created fill out whatever is necesaary for the seed
    s.idx_keyframe=idx_keyframe;
    s.m_uv=img_coords;
    s.m_gradH=mat2(hessian.x,hessian.y,hessian.y,hessian.z); //column major filling

    for(int p_idx=0;p_idx<pattern_rot_nr_points; ++p_idx){
        vec2 offset=pattern_rot_offsets[p_idx];

        vec3 hit_color_and_grads=texelFetch(gray_with_gradients_img_sampler, img_coords + ivec2(offset), 0 ).xyz;

        s.m_intensity[p_idx]=hit_color_and_grads.x;
        float squared_norm=dot(hit_color_and_grads.yz,hit_color_and_grads.yz);
        float grad_normalization= sqrt(squared_norm + params.eta); //TODO would it be faster to add the eta to vector and then use length?
        vec2 grad_normalized=hit_color_and_grads.yz / grad_normalization;
        s.m_normalized_grad[p_idx]=grad_normalized;

        if(length(grad_normalized)<1e-3){
            s.m_zero_grad[p_idx]=1;
        }else{
            s.m_zero_grad[p_idx]=0;
            s.m_active_pattern_points++;
        }

    }
    s.m_energyTH = s.m_active_pattern_points * params.maxPerPtError * params.slackFactor;


    //stuff for the depthfilter
    //stuff in reinit
    s.depth_filter.m_converged = 0;
    s.depth_filter.m_is_outlier = 0;
    s.depth_filter.m_alpha = 10;
    s.depth_filter.m_beta = 10;
    s.depth_filter.m_z_range = (1.0/min_starting_depth);
    s.depth_filter.m_sigma2 = (s.depth_filter.m_z_range*s.depth_filter.m_z_range/36);
    s.depth_filter.m_mu = (1.0/mean_starting_depth);
    //stuff in the constructor
    s.depth_filter.m_f.xyz = K_inv * vec3(img_coords,1.0);
    s.depth_filter.m_f_scale = length(s.depth_filter.m_f.xyz);
    s.depth_filter.m_f.xyz /= s.depth_filter.m_f_scale;


    s.m_idepth_minmax.x = s.depth_filter.m_mu + sqrt(s.depth_filter.m_sigma2);
    s.m_idepth_minmax.y = max(s.depth_filter.m_mu - sqrt(s.depth_filter.m_sigma2), 0.00000001f);


    //debug
    for(int i = 0; i < 16; i++){
        s.debug[i]=i;
    }

    return s;
}

void main(void) {

    ivec2 img_coords = ivec2(gl_GlobalInvocationID.xy);


    vec3 hessian=texelFetch(hessian_blurred_sampler, img_coords, 0).xyz;
    float determinant=abs(hessian.x*hessian.z-hessian.y*hessian.y);
    float trace=hessian.x+hessian.z;
    // if(determinant>0.005){
    if(trace>0.9){
        uint id=atomicCounterIncrement(nr_seeds_created); //increments and returns the previous value

        Seed s=create_seed(img_coords, hessian);

        p[id+seeds_start_idx]=s;

        imageStore(debug, img_coords , vec4(0,255,0,255) );
    }



    // //attempt 2 with the pattern
    // mat2 gradH=mat2(0.0, 0.0 ,0.0 ,0.0);
    // for(int p_idx=0;p_idx<pattern_rot_nr_points; ++p_idx){
    //     vec2 offset=pattern_rot_offsets[p_idx];
    //     vec3 hit_color_and_grads=texelFetch(gray_with_gradients_img_sampler, img_coords + ivec2(offset), 0 ).xyz;
    //     gradH+=outerProduct(hit_color_and_grads.yz,hit_color_and_grads.yz);
    // }
    // float trace=gradH[0][0]+gradH[1][1];
    // float determinant=determinant(gradH);
    // float grad_length=length(vec2(gradH[0][0],gradH[1][1]));
    // if(trace>3.0){
    // // if(determinant>0.2){
    // // if(grad_length>2.0){
    //     uint id=atomicCounterIncrement(nr_seeds_created); //increments and returns the previous value
    //
    //     Seed s=create_seed();
    //
    //     imageStore(debug, img_coords , vec4(0,255,0,255) );
    // }

}
