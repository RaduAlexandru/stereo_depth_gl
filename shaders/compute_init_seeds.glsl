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

uniform sampler2D hessian_blurred_sampler; //contains gray val and gradx and grady
uniform sampler2D gray_with_gradients_img_sampler; //contains gray val and gradx and grady
layout(binding=2, rgba32f) uniform writeonly image2D debug;


uniform vec2 pattern_rot_offsets[MAX_RES_PER_POINT];
uniform int pattern_rot_nr_points;
uniform mat3 K;
uniform mat3 K_inv;
uniform float min_starting_depth;
uniform float mean_starting_depth;
uniform int seeds_start_idx;
uniform int idx_keyframe;
uniform float ngf_eta;


Seed create_seed(ivec2 img_coords, vec3 hessian, int seed_idx){
    Seed s;
    //TODO scale m_f


    // //at position nr_seeds_created fill out whatever is necesaary for the seed
    // s.idx_keyframe=idx_keyframe;
    // s.m_uv=img_coords;
    // s.m_gradH=mat2(hessian.x,hessian.y,hessian.y,hessian.z); //column major filling
    // s.m_time_alive=0;
    // s.m_nr_times_visible=0;
    //
    // for(int p_idx=0;p_idx<pattern_rot_nr_points; ++p_idx){
    //     vec2 offset=pattern_rot_offsets[p_idx];
    //
    //     vec3 hit_color_and_grads=texelFetch(gray_with_gradients_img_sampler, img_coords + ivec2(offset), 0 ).xyz;
    //
    //     s.m_intensity[p_idx]=hit_color_and_grads.x;
    //     float squared_norm=dot(hit_color_and_grads.yz,hit_color_and_grads.yz);
    //     float grad_normalization= sqrt(squared_norm + params.eta); //TODO would it be faster to add the eta to vector and then use length?
    //     vec2 grad_normalized=hit_color_and_grads.yz / grad_normalization;
    //     s.m_normalized_grad[p_idx]=grad_normalized;
    //
    //     if(length(grad_normalized)<1e-3){
    //         s.m_zero_grad[p_idx]=1;
    //     }else{
    //         s.m_zero_grad[p_idx]=0;
    //         s.m_active_pattern_points++;
    //     }
    //
    // }
    // s.m_energyTH = s.m_active_pattern_points * params.residualTH;
    //
    //
    // //stuff for the depthfilter
    // //stuff in reinit
    // s.depth_filter.m_converged = 0;
    // s.depth_filter.m_is_outlier = 0;
    // s.depth_filter.m_alpha = 10;
    // s.depth_filter.m_beta = 10;
    // s.depth_filter.m_z_range = (1.0/min_starting_depth);
    // s.depth_filter.m_sigma2 = (s.depth_filter.m_z_range*s.depth_filter.m_z_range/36);
    // s.depth_filter.m_mu = (1.0/mean_starting_depth);
    // //stuff in the constructor
    // s.depth_filter.m_f.xyz = K_inv * vec3(img_coords,1.0);
    // // s.depth_filter.m_f_scale = length(s.depth_filter.m_f.xyz);
    // // s.depth_filter.m_f.xyz /= s.depth_filter.m_f_scale;
    //
    //
    // s.m_idepth_minmax.x = s.depth_filter.m_mu + sqrt(s.depth_filter.m_sigma2);
    // s.m_idepth_minmax.y = max(s.depth_filter.m_mu - sqrt(s.depth_filter.m_sigma2), 0.00000001f);
    //
    // s.m_last_error = -1;
    // s.depth_filter.m_is_outlier = 0;
    //
    //
    // s.left = seed_idx;
    // s.right = seed_idx;
    // s.above = seed_idx;
    // s.below = seed_idx;
    // s.left_upper = seed_idx;
    // s.right_upper = seed_idx;
    // s.left_lower = seed_idx;
    // s.right_lower = seed_idx;
    //
    //
    // //debug
    // for(int i = 0; i < 16; i++){
    //     s.debug[i]=0;
    // }



    // attempt 2
    s.m_uv =img_coords;
    s.m_gradH=mat2(hessian.x,hessian.y,hessian.y,hessian.z); //column major filling
    s.m_time_alive=0;
    s.m_nr_times_visible=0;

    //Seed::Seed
    s.depth_filter.m_f.xyz = normalize(K_inv * vec3(img_coords,1.0));
    s.depth_filter.m_f.w=1.0;
    s.depth_filter.m_mu = (1.0/mean_starting_depth);
    s.depth_filter.m_z_range = (1.0/min_starting_depth);
    s.depth_filter.m_sigma2 = (s.depth_filter.m_z_range*s.depth_filter.m_z_range/36);

    float z_inv_min = s.depth_filter.m_mu + sqrt(s.depth_filter.m_sigma2);
    float z_inv_max = max(s.depth_filter.m_mu- sqrt(s.depth_filter.m_sigma2), 0.00000001f);
    s.m_idepth_minmax =vec2(z_inv_min, z_inv_max );

    s.depth_filter.m_alpha=10.0;
    s.depth_filter.m_beta=10.0;



    //get data for the color of that point (depth_point->ImmaturePoint::ImmaturePoint)---------------------
    for(int p_idx=0;p_idx<pattern_rot_nr_points;p_idx++){

        vec2 offset=pattern_rot_offsets[p_idx];

        vec3 hit_color_and_grads=texelFetch(gray_with_gradients_img_sampler, img_coords + ivec2(offset), 0 ).xyz;
        float hit_color=hit_color_and_grads.x;
        vec2 grads=hit_color_and_grads.yz;

        s.m_intensity[p_idx]=hit_color;



        //for ngf
        s.m_normalized_grad[p_idx] = grads;
        s.m_normalized_grad[p_idx] /= sqrt( dot(grads,grads) + ngf_eta);
        if( length(s.m_normalized_grad[p_idx])<1e-3){
            s.m_zero_grad[p_idx]=1;
        }else{
            s.m_zero_grad[p_idx]=0;
            s.m_active_pattern_points++;
        }

    }
    s.m_energyTH = s.m_active_pattern_points * params.residualTH;
    // point.m_energyTH *= m_params.overallEnergyTHWeight*m_params.overallEnergyTHWeight;

    s.m_last_error = -1;
    s.depth_filter.m_is_outlier = 0;


    //as the neighbours indices set the current points so if we access it we get this point
    s.left = seed_idx;
    s.right = seed_idx;
    s.above = seed_idx;
    s.below = seed_idx;
    s.left_upper = seed_idx;
    s.right_upper = seed_idx;
    s.left_lower = seed_idx;
    s.right_lower = seed_idx;








    return s;
}

void main(void) {

    int id = int(gl_GlobalInvocationID.x);


    // vec3 hessian=texelFetch(hessian_blurred_sampler, img_coords, 0).xyz;
    // float determinant=abs(hessian.x*hessian.z-hessian.y*hessian.y);
    // float trace=hessian.x+hessian.z;
    // if(determinant>0.005){

    // //TODO get hessian
    ivec2 img_coords=ivec2(p[id].m_uv); //TODO........
    vec3 hessian=texelFetch(hessian_blurred_sampler, img_coords, 0).xyz;
    Seed s=create_seed(img_coords, hessian, id);
    // Seed s;
    // s.m_uv.x+=1;
    p[id]=s;

    // //debug why are the colors different
    // vec3 hit_color_and_grads=texelFetch(gray_with_gradients_img_sampler, img_coords, 0 ).xyz;
    // float hit_color=hit_color_and_grads.y;
    // imageStore(debug, img_coords , vec4(0,hit_color,0,255) );

    // //attempt 2 debug why are the colors different
    // vec3 hit_color_and_grads=texelFetch(gray_with_gradients_img_sampler, img_coords, 0 ).xyz;
    // // float hit_color=hit_color_and_grads.y;
    // imageStore(debug, img_coords , vec4(hit_color_and_grads,255) );





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
    // vec3 hessian=vec3(gradH[0][0], gradH[0][1], gradH[1][1]);
    // if(trace>10){
    // // if(determinant>0.2){
    // // if(grad_length>2.0){
    //     uint id=atomicCounterIncrement(nr_seeds_created); //increments and returns the previous value
    //
    //     Seed s=create_seed(img_coords,hessian);
    //
    //     p[id+seeds_start_idx]=s;
    //
    //     imageStore(debug, img_coords , vec4(0,255,0,255) );
    // }



    // //attempt 3
    // vec3 hit_color_and_grads=texelFetch(gray_with_gradients_img_sampler, img_coords , 0 ).xyz;
    // float hit_color=hit_color_and_grads.x;
    // imageStore(debug, img_coords , vec4(0,hit_color,0,255) );


    // //attempt 4
    // //total time of seed creation is 70ms out of which 50ms is spent on checking if the trace is high and incrementing the counter
    // //the time for just incrementing the counter for all pixels is 300ms... so it seems that the counter is the bottleneck
    // mat2 gradH=mat2(0.0, 0.0 ,0.0 ,0.0);
    // for(int p_idx=0;p_idx<pattern_rot_nr_points; ++p_idx){
    //     vec2 offset=pattern_rot_offsets[p_idx];
    //     vec3 hit_color_and_grads=texelFetch(gray_with_gradients_img_sampler, img_coords + ivec2(offset), 0 ).xyz;
    //     vec2 grads_abs=abs(hit_color_and_grads.yz);
    //     gradH+=outerProduct(grads_abs,grads_abs);
    // }
    // float trace=gradH[0][0]+gradH[1][1];
    // // float determinant=determinant(gradH);
    // // float grad_length=length(vec2(gradH[0][0],gradH[1][1]));
    // vec3 hessian=vec3(gradH[0][0], gradH[0][1], gradH[1][1]);
    // if(trace>17){
    // // if(determinant>0.2){
    // // if(grad_length>2.0){
    //     uint id=atomicCounterIncrement(nr_seeds_created); //increments and returns the previous value
    //
    //     Seed s=create_seed(img_coords,hessian,int(id));
    //
    //     p[id]=s;
    //     // p[id+seeds_start_idx]=s;
    //
    //     // imageStore(debug, img_coords , vec4(0,255,0,255) );
    // }
    // // imageStore(debug, img_coords , vec4(0,trace/8,0,255) );


    // //attempt 5 debug the gradients because it seems that the absolute value doesnt work
    // vec3 hit_color_and_grads=texelFetch(gray_with_gradients_img_sampler, img_coords, 0).xyz;
    // float hit_color=clamp(abs(hit_color_and_grads.y),0,1);
    // // float hit_color=hit_color_and_grads.y;
    // // float sign=1-sign(hit_color_and_grads.y);
    // // float sign=sign(hit_color_and_grads.y);
    // // float val_sign=0;
    // // if(sign==-1.0){
    // //     val_sign=255.0;
    // // }
    // // float hit_color=hit_color_and_grads.y;
    // // float zeros=0;
    // // if(hit_color==0){
    // //     zeros=1.0;
    // // }
    // imageStore(debug, img_coords , vec4(0,hit_color,0,255) );



}
