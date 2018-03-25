#version 430

layout (local_size_x = 32, local_size_y = 32) in;

uniform usampler2D  rgb_global_tex_sampler;
uniform usampler2DArray  rgb_modified_volume_sampler;

layout(binding=0, rgba8ui) uniform coherent uimage2D rgb_global_tex;  //TODO set back to write only
layout(binding=1, r8ui) uniform coherent uimage2DArray rgb_modified_volume; //GL_R8UI


void main(void) {

    ivec2 img_coords_rgb = ivec2(gl_GlobalInvocationID.xy);

    //if the texel was not updated in this frame then we don't need to fill anything around it
    uint vis=texelFetch(rgb_modified_volume_sampler, ivec3(img_coords_rgb,1),0 ).x;
    if(vis==0){
        return;
    }

    //this texel was modified
    // uvec4 color=texelFetch(rgb_global_tex_sampler, img_coords_rgb, 0); /TODO i thought it doesn't work at all but now I found it it doesnt work because img_coords_rgb is in range [0-rgb_size] and it should be in [0,1] because it's a texture alo probalby it hould be onyl sampler and not usampler (check how you did it in the splat_frag)
    uvec4 color=imageLoad(rgb_global_tex, img_coords_rgb);


    //blur it into adyacent ones if their value of modified is 0 over ALL frames
    for (int i=0; i<3; i++){
       for (int j=0; j<3; j++) {
           if (i != 1 && j != 1){
               //see if the neighbouring texel has never been modified
               ivec2 neigh_coords=img_coords_rgb + ivec2(i-1,j-1);
               uint modif=texelFetch(rgb_modified_volume_sampler, ivec3(neigh_coords,0),0 ).x;
               if(modif==0){
                   imageStore(rgb_global_tex, neigh_coords , color );
               }
           }
       }
    }

    //set as unmodified
    imageStore(rgb_modified_volume, ivec3(img_coords_rgb,1) , uvec4(0,0,0,0) );

}
