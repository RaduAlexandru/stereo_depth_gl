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

const int G_VAL_DEPTH=0; //--readonly
const int MU_DEPTH=1; //-readonly
const int VAL_MU_DENOISED_DEPTH=2;
const int MU_HEAD_DEPTH=3;
const int P_VEC_DEPTH=4; //we leave it for last since it will occupy actually possition 4 and 5 since it's a vector
//5 is occupied by pvecdepth still


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


void main(void) {

    int id = int(gl_GlobalInvocationID.x);

    //the only data we need from the point is:
        //float val_mu_denoised -readwrite
        //float g_val - READ ONLY
        //vec2 p_vec -readwrite
        //float mu  - READ ONLY
        //float mu_head - readwrie
        // that would require an image with 6 channels, maybe use a 3d texture or a texture 2d array

    ivec2 pos_xy=ivec2( int(p[id].u), int(p[id].v) );

    float new_mu=imageLoad(tex_volume, ivec3(pos_xy, VAL_MU_DENOISED_DEPTH) ).x;
    p[id].mu=new_mu;
    // imageStore(tex_volume, ivec3(pos_xy, MU_DEPTH) , vec4(new_mu,0,0,0) );

    // float g_val=imageStore(tex_volume, ivec3(pos_xy, G_VAL_DEPTH) , vec4(p[id].g_val,0,0,0) );
    // float mu_depth=imageStore(tex_volume, ivec3(pos_xy, MU_DEPTH) , vec4(p[id].mu,0,0,0) );
    // float val_mu_denoised=imageStore(tex_volume, ivec3(pos_xy, VAL_MU_DENOISED_DEPTH) , vec4(p[id].val_mu_denoised,0,0,0) );
    // float mu_head=imageStore(tex_volume, ivec3(pos_xy, MU_HEAD_DEPTH) , vec4(p[id].mu_head,0,0,0) );
    // float p_vec_x=imageStore(tex_volume, ivec3(pos_xy, P_VEC_DEPTH) , vec4(p[id].p_vec.x,0,0,0) );
    // float p_vec_y=imageStore(tex_volume, ivec3(pos_xy, P_VEC_DEPTH+1) , vec4(p[id].p_vec.y,0,0,0) );



}
