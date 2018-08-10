#version 430

#define M_PI 3.1415926535897932384626433832795

const int MAX_RES_PER_POINT=16;

layout (local_size_x = 32, local_size_y = 16) in;

uniform sampler2D hessian_blurred_sampler; //contains gray val and gradx and grady
struct Seed{
    int idx_host_frame; //idx in the array of frames of the frame which "hosts" this inmature points
    float u,v; //position in host frame of the point
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


    //for denoising (indexes iinto the array of points of each of the 8 neighbours)
    int left;
    int right;
    int above;
    int below;
    int left_upper;
    int right_upper;
    int left_lower;
    int right_lower;

    //some other things for denoising
    float g;
    float mu_denoised;
    float mu_head;
    float pad_6;
    vec2 p;
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
layout (binding = 0, std430) coherent buffer array_seeds_block{
    Seed p[];
};
layout(binding = 1) uniform atomic_uint nr_seeds_created;
layout(binding=2, rgba32f) uniform writeonly image2D debug;

void main(void) {

    ivec2 img_coords = ivec2(gl_GlobalInvocationID.xy);

    vec3 hessian=texelFetch(hessian_blurred_sampler, img_coords, 0).xyz;
    float determinant=abs(hessian.x*hessian.z-hessian.y*hessian.y);
    float trace=hessian.x+hessian.z;
    // if(determinant>0.005){
    if(trace>0.9){
        atomicCounterIncrement(nr_seeds_created);
        imageStore(debug, img_coords , vec4(0,255,0,255) );

    }

    // imageStore(debug, img_coords , vec4(determinant,0,0,255) );

    // atomicCounterIncrement(nr_seeds_created);


    // //load the gradx and grady
    // vec2 grads=texelFetch(gray_with_gradients_img_sampler, img_coords, 0).yz;
    //
    // //calculate the hessian elements
    // float gx2=grads.x*grads.x;
    // float gxgy=grads.x*grads.y;
    // float gy2=grads.y*grads.y;
    //
    // imageStore(hessian_pointwise_tex, img_coords , vec4(gx2,gxgy,gy2,255) );
    // imageStore(hessian_pointwise_tex, img_coords , vec4(255,255,255,255) );

}
