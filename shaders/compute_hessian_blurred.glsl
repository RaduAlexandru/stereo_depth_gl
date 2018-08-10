#version 430

//from grad_x and grad_y we compute a hessian_pointwise which is a 4d texture which contains the elements of
// [gx] [gx, gy]  = gx2 gxgy, gy2
// [gy]
//we don't store the gxgy two times

layout (local_size_x = 32, local_size_y = 16) in;

uniform sampler2D hessian_pointwise_tex_sampler; //contains gray val and gradx and grady
layout(binding=0, rgba32f) uniform writeonly image2D hessian_blurred;

void main(void) {

    ivec2 img_coords = ivec2(gl_GlobalInvocationID.xy);

    int window_size=3;
    int window_half_size=window_size/2;
    vec3 hessian_blurred_val=vec3(0,0,0);
    for(int x = -window_half_size; x <= window_half_size; x ++){
        for(int y = -window_half_size; y <= window_half_size; y ++){
            vec3 hessian_pointwise=texelFetch(hessian_pointwise_tex_sampler, img_coords + ivec2(x,y), 0).xyz;
            hessian_blurred_val+=hessian_pointwise;
        }
    }
    hessian_blurred_val=hessian_blurred_val/(window_size*window_size); //normalize;

    imageStore(hessian_blurred, img_coords , vec4(hessian_blurred_val,255) );

}
