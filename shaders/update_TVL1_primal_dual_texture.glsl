#version 430

layout (local_size_x = 32, local_size_y = 32) in;

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



// //you change it to  layout (binding = 0, std430) or layout (binding = 0, std140) in case stuff breaks
// layout (binding = 0, std430) coherent buffer array_points_block{
//     Point p[];
// };

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


layout(binding=0, r32f) uniform coherent image2DArray tex_volume; //GL_R32F


const int G_VAL_DEPTH=0; //--readonly
const int MU_DEPTH=1; //-readonly
const int VAL_MU_DENOISED_DEPTH=2;
const int MU_HEAD_DEPTH=3;
const int P_VEC_DEPTH=4; //we leave it for last since it will occupy actually possition 4 and 5 since it's a vector
//5 is occupied by pvecdepth still

void main(void) {



    ivec2 id = ivec2(gl_GlobalInvocationID.xy);
    ivec2 pos_xy=ivec2( id );
    ivec2 pos_right=ivec2( id.x+1, id.y );
    ivec2 pos_below=ivec2( id.x,  id.y-1 );
    ivec2 pos_left=ivec2(  id.x-1, id.y );
    ivec2 pos_above=ivec2(  id.x, id.y+1 );

    //check if the current pixel is valid
    float check_curr=imageLoad(tex_volume, ivec3(pos_xy, G_VAL_DEPTH) ).x;
    if(check_curr==-1){
        return;
    }

    // //debug just write a value of 1 in the mu denoised
    // imageStore(tex_volume, ivec3(pos_xy, VAL_MU_DENOISED_DEPTH) , vec4( 1 ,0,0,0) );
    // return;

    //attemot at making at faster
    const float sigma = ((1.0 / (params.denoise_L*params.denoise_L)) / params.denoise_tau);
    //read once the read only stuff to speed up the things
    float g_val=imageLoad(tex_volume, ivec3(pos_xy, G_VAL_DEPTH) ).x;
    float mu=imageLoad(tex_volume, ivec3(pos_xy, MU_DEPTH) ).x;
    //Read some other stuff even though they will be updated later
    float val_mu_denoised=imageLoad(tex_volume, ivec3(pos_xy, VAL_MU_DENOISED_DEPTH) ).x;
    vec2 current_p_vec;
    current_p_vec.x=imageLoad(tex_volume, ivec3(pos_xy, P_VEC_DEPTH) ).x;
    current_p_vec.y=imageLoad(tex_volume, ivec3(pos_xy, P_VEC_DEPTH+1) ).x;

    //update dual
    //check if we have neighbours and get the mu_head of right and below
    float check_right=imageLoad(tex_volume, ivec3(pos_right, G_VAL_DEPTH) ).x;
    float check_below=imageLoad(tex_volume, ivec3(pos_below, G_VAL_DEPTH) ).x;
    float right_mu_head;
    float below_mu_head;
    if(check_right==-1){
        //get the current one
        right_mu_head=imageLoad(tex_volume, ivec3(pos_xy, MU_HEAD_DEPTH) ).x;
    }else{
        right_mu_head=imageLoad(tex_volume, ivec3(pos_right, MU_HEAD_DEPTH) ).x;
    }
    if(check_below==-1){
        //get the current one
        right_mu_head=imageLoad(tex_volume, ivec3(pos_xy, MU_HEAD_DEPTH) ).x;
    }else{
        below_mu_head=imageLoad(tex_volume, ivec3(pos_below, MU_HEAD_DEPTH) ).x;
    }
    vec2 grad_uhead=vec2(right_mu_head - val_mu_denoised, below_mu_head - val_mu_denoised);
    const vec2 temp_p = g_val * grad_uhead * sigma + current_p_vec;
    const float sqrt_p = length(temp_p); //sqrt(temp_p[0] * temp_p[0] + temp_p[1] * temp_p[1]);
    current_p_vec = temp_p / max(1.0f, sqrt_p);
    imageStore(tex_volume, ivec3(pos_xy, P_VEC_DEPTH) , vec4(current_p_vec.x,0,0,0) );
    imageStore(tex_volume, ivec3(pos_xy, P_VEC_DEPTH+1) , vec4(current_p_vec.y,0,0,0) );


    // memoryBarrier();
    // barrier();
    //
    // // update primal:
    const float old_u = imageLoad(tex_volume, ivec3(pos_xy, VAL_MU_DENOISED_DEPTH) ).x;
    //check left and above for pvec
    vec2 left_p;
    vec2 above_p;
    float check_left=imageLoad(tex_volume, ivec3(pos_left, G_VAL_DEPTH) ).x;
    float check_above=imageLoad(tex_volume, ivec3(pos_above, G_VAL_DEPTH) ).x;
    if(check_left==-1){
        //get the current one
        left_p=current_p_vec;
    }else{
        left_p.x=imageLoad(tex_volume, ivec3(pos_left, P_VEC_DEPTH) ).x;
        left_p.y=imageLoad(tex_volume, ivec3(pos_left, P_VEC_DEPTH+1) ).x;
    }
    if(check_above==-1){
        //get the current one
        above_p=current_p_vec;
    }else{
        above_p.x=imageLoad(tex_volume, ivec3(pos_above, P_VEC_DEPTH) ).x;
        above_p.y=imageLoad(tex_volume, ivec3(pos_above, P_VEC_DEPTH+1) ).x;
    }
    const float divergence = current_p_vec.x - left_p.x + current_p_vec.y - above_p.y;
    const float tauLambda = params.denoise_tau*params.denoise_lambda;
    const float temp_u = old_u + params.denoise_tau * g_val * divergence;
    float diff =temp_u - mu;
    if (diff> (tauLambda)){
        val_mu_denoised = temp_u - tauLambda;
    }else if (diff < (-tauLambda)){
        val_mu_denoised = temp_u + tauLambda;
    }else{
        val_mu_denoised = mu;
    }
    float mu_head = val_mu_denoised + params.denoise_theta * (val_mu_denoised - old_u);
    imageStore(tex_volume, ivec3(pos_xy, MU_HEAD_DEPTH) , vec4( mu_head ,0,0,0) );
    imageStore(tex_volume, ivec3(pos_xy, VAL_MU_DENOISED_DEPTH) , vec4( val_mu_denoised ,0,0,0) );




}
