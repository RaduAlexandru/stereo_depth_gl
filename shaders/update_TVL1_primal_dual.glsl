#version 430

layout (local_size_x = 256) in;

const int MAX_RES_PER_POINT=16;

struct Point{
    int idx_host_frame; //idx in the array of frames of the frame which "hosts" this inmature p[id]s
    float u,v; //position in host frame of the p[id]
    float a;                     //!< a of Beta distribution: When high, probability of inlier is large.
    float b;                     //!< b of Beta distribution: When high, probability of outlier is large.
    float mu;                    //!< Mean of normal distribution.
    float z_range;               //!< Max range of the possible depth.
    float sigma2;                //!< Variance of normal distribution.
    float idepth_min;
    float idepth_max;
    float energyTH;
    float quality;
    //-----------------up until here we have 48 bytes so it's padded correctly to 16 bytes

    vec4 f; // heading range = Ki * (u,v,1) //make it float 4 becuse float 3 gets padded to 4 either way
    int lastTraceStatus;
    int converged;
    int is_outlier;
    int pad_1;
    //
    float color[MAX_RES_PER_POINT]; 		// colors in host frame
    float weights[MAX_RES_PER_POINT]; 		// host-weights for respective residuals.
    vec2 colorD[MAX_RES_PER_POINT];  //gradient in x and y at the pixel of the pattern normalized by the sqrt
    vec2 colorGrad[MAX_RES_PER_POINT]; //just the raw gradient in x and y at the pixel offset of the pattern

    float ncc_sum_templ;
    float ncc_const_templ;
    float pad_2;
    float pad_3;

    //Stuff that may be to be removed
    mat2 gradH;

    //for denoising (indexes iinto the array of p[id]s of each of the 8 neighbours)
    int left;
    int right;
    int above;
    int below;
    int left_upper;
    int right_upper;
    int left_lower;
    int right_lower;

    //some other things for denoising
    float g_val; //Cannot be called g because it will be interpreted as a swizzle
    float val_mu_denoised; //Cannot be called _mu_denoised because for some reason it is interpreted as a swizzle
    float mu_head;
    float pad_6;
    vec2 p_vec; //Cannot be called p because it will be interpreted as a swizzle
    // glm::vec2 p;
    float pad_7;
    float pad_8;


    //debug stuff
    float gradient_hessian_det;
    float gt_depth;
    int last_visible_frame;

    float debug; //serves as both debug and padding to 16 bytes
    // float padding_1; //to gt the struc to be aligned to 16 bytes

    float debug2[16];

};



//you change it to  layout (binding = 0, std430) or layout (binding = 0, std140) in case stuff breaks
layout (binding = 0, std430) coherent buffer array_points_block{
    Point p[];
};

//if you assign a binding to this one it doesn't read the values anymore, god knows why..
layout (std140) uniform params_block{
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



void main(void) {

    int id = int(gl_GlobalInvocationID.x);

    // const float L = params.denoise_L;
    // const float tau = params.denoise_tau;
    // const float theta = params.denoise_theta;
    // const float lambda=params.denoise_lambda;
    // const float sigma = ((1 / (L*L)) / tau);
    //
    //
    // // update dual
    // const float g_val = p[id].g_val;
    // const vec2 p_vec = p[id].p_vec;
    // vec2 grad_uhead;
    // const float current_u = p[id].val_mu_denoised;
    // float right_mu_head = (p[id].right == -1) ? p[id].mu_head : p[p[id].right].mu_head;
    // float below_mu_head = (p[id].below == -1) ? p[id].mu_head : p[p[id].below].mu_head;
    // grad_uhead.x = right_mu_head - current_u; //->atXY(min<int>(c_img_size.width-1, x+1), y)  - current_u;
    // grad_uhead.y = below_mu_head - current_u; //->atXY(x, min<int>(c_img_size.height-1, y+1)) - current_u;
    // const vec2 temp_p = g_val * grad_uhead * sigma + p_vec;
    // const float sqrt_p = length(temp_p); //sqrt(temp_p[0] * temp_p[0] + temp_p[1] * temp_p[1]);
    // p[id].p_vec = temp_p / max(1.0f, sqrt_p);
    //
    // // update primal:
    // const float noisy_depth = p[id].mu;
    // const float old_u = p[id].val_mu_denoised;
    // const float g = p[id].g_val;
    //
    // vec2 current_p = p[id].p_vec;
    // vec2 left_p = (p[id].left == -1) ? p[id].p_vec : p[p[id].left].p_vec;
    // vec2 above_p = (p[id].above == -1) ? p[id].p_vec : p[p[id].above].p_vec;
    // vec2 w_p = left_p;
    // vec2 n_p = above_p;
    //
    // const float x = p[id].u;
    // const float y = p[id].v;
    // if ( x == 0)
    //     w_p.x = 0.f;
    // else if ( x >= 480-1 )
    //     current_p.x = 0.f;
    // if ( y == 0 )
    //     n_p.y = 0.f;
    // else if ( y >= 640-1 )
    //     current_p.y = 0.f;
    //
    // const float divergence = current_p.x - w_p.x + current_p.y - n_p.y;
    // const float tauLambda = tau*lambda;
    // const float temp_u = old_u + tau * g * divergence;
    //
    // if ((temp_u - noisy_depth) > (tauLambda)){
    //     p[id].val_mu_denoised = temp_u - tauLambda;
    // }else if ((temp_u - noisy_depth) < (-tauLambda)){
    //     p[id].val_mu_denoised = temp_u + tauLambda;
    // }else{
    //     p[id].val_mu_denoised = noisy_depth;
    // }
    // p[id].mu_head = p[id].val_mu_denoised + theta * (p[id].val_mu_denoised - old_u);



    //the only data we need from the point is:
        //float val_mu_denoised
        //float g_val
        //vec2 p_vec
        //float mu
        //float mu_head
        // that would require an image with 6 channels, maybe use a 3d texture or a texture 2d array


    //attemot at making at faster
    const float sigma = ((1.0 / (params.denoise_L*params.denoise_L)) / params.denoise_tau);


    // update dual
    float right_mu_head = (p[id].right == -1) ? p[id].mu_head : p[p[id].right].mu_head;
    float below_mu_head = (p[id].below == -1) ? p[id].mu_head : p[p[id].below].mu_head;
    vec2 grad_uhead=vec2(right_mu_head - p[id].val_mu_denoised, below_mu_head - p[id].val_mu_denoised);
    const vec2 temp_p = p[id].g_val * grad_uhead * sigma + p[id].p_vec;
    const float sqrt_p = length(temp_p); //sqrt(temp_p[0] * temp_p[0] + temp_p[1] * temp_p[1]);
    p[id].p_vec = temp_p / max(1.0f, sqrt_p);

    // update primal:
    const float old_u = p[id].val_mu_denoised;
    vec2 current_p = p[id].p_vec;
    vec2 left_p = (p[id].left == -1) ? p[id].p_vec : p[p[id].left].p_vec;
    vec2 above_p = (p[id].above == -1) ? p[id].p_vec : p[p[id].above].p_vec;
    const float divergence = current_p.x - left_p.x + current_p.y - above_p.y;
    const float tauLambda = params.denoise_tau*params.denoise_lambda;
    const float temp_u = old_u + params.denoise_tau * p[id].g_val * divergence;
    float diff =temp_u - p[id].mu;
    if (diff> (tauLambda)){
        p[id].val_mu_denoised = temp_u - tauLambda;
    }else if (diff < (-tauLambda)){
        p[id].val_mu_denoised = temp_u + tauLambda;
    }else{
        p[id].val_mu_denoised = p[id].mu;
    }
    p[id].mu_head = p[id].val_mu_denoised + params.denoise_theta * (p[id].val_mu_denoised - old_u);



}
