#version 430

//from grad_x and grad_y we compute a hessian_pointwise which is a 4d texture which contains the elements of
// [gx] [gx, gy]  = gx2 gxgy, gy2
// [gy]
//we don't store the gxgy two times

layout (local_size_x = 32, local_size_y = 16) in;

uniform sampler2D gray_with_gradients_img_sampler; //contains gray val and gradx and grady
layout(binding=0, rgba32f) uniform writeonly image2D hessian_pointwise_tex;

void main(void) {

    ivec2 img_coords = ivec2(gl_GlobalInvocationID.xy);

    //load the gradx and grady
    vec2 grads=texelFetch(gray_with_gradients_img_sampler, img_coords, 0).yz;

    //calculate the hessian elements
    float gx2=grads.x*grads.x;
    float gxgy=grads.x*grads.y;
    float gy2=grads.y*grads.y;

    imageStore(hessian_pointwise_tex, img_coords , vec4(gx2,gxgy,gy2,255) );

    // imageStore(hessian_pointwise_tex, img_coords , vec4(grads.x,0,0,255) );
    // imageStore(hessian_pointwise_tex, img_coords , vec4(255,255,255,255) );

}
