#version 430

layout (local_size_x = 32, local_size_y = 32) in;

uniform sampler2DArray classes_probs_global_volume_sampler;
uniform usampler2DArray pages_commited_volume_sampler;
uniform sampler2D rgb_global_sampler; //the global rgb so that we can overlay on tope of that one
uniform sampler2D semantics_global_sampler; //the global semantics colors  so that we can overlay on tope of that one

layout(binding=0, rgba8ui) uniform writeonly uimage2D semantics_one_class_global_tex;
layout(binding=1, r32f) uniform coherent image2DArray classes_probs_global_volume;
layout(binding=2, r16ui) uniform coherent uimage2D semantics_nr_times_modified_global_tex;
layout(binding=3, rgba8ui) uniform coherent uimage2D rgb_global_tex;
layout(binding=4, rgba8ui) uniform coherent uimage2D semantics_global_tex;

uniform int class_id=0;
uniform int page_size_x = 0;
uniform int page_size_y = 0;
uniform float min_prob;
uniform float max_prob;
uniform int global_texture_type = 0;
uniform int rgb_scale_multiplier = 0;

float map(float value, float inMin, float inMax, float outMin, float outMax) {
  return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

void main(void) {

    ivec2 img_coords_sem = ivec2(gl_GlobalInvocationID.xy);
    ivec2 img_coords_rgb = ivec2(img_coords_sem * rgb_scale_multiplier);
    int x_page_idx=img_coords_sem[0]/page_size_x;
    int y_page_idx=img_coords_sem[1]/page_size_y;

    uvec4 slice_color;


    //background color is the either the rgb, semantics or just a black color
    uvec4 bg_color;
    if(global_texture_type==0){
        // bg_color = texelFetch(rgb_global_sampler,  img_coords_rgb, 0);
        bg_color=imageLoad(rgb_global_tex, img_coords_rgb);
        // uvec4 wtf=imageLoad(rgb_global_tex, img_coords_rgb);
    }else if(global_texture_type==1){
        // bg_color = texelFetch(semantics_global_sampler,  img_coords_sem, 0);
        bg_color=imageLoad(semantics_global_tex, img_coords_sem);
    }else if(global_texture_type==2){
        bg_color=uvec4(0,0,0,255);
    }


    // //if the page is not allocated color is black
    // uint is_page_commited=texelFetch(pages_commited_volume_sampler,  ivec3(x_page_idx, y_page_idx, class_id), 0).x;
    // if(int(is_page_commited)==0){
    //     color=uvec4(0,0,0,255);
    // }else{ // page IS allocated
    //     float cur_prob=texelFetch(classes_probs_global_volume_sampler, ivec3(img_coords_sem, class_id), 0).x;
    //
    //     //cap it between min prob and max prob and interpolate
    //     if(cur_prob<min_prob) cur_prob=min_prob;
    //     if(cur_prob>max_prob) cur_prob=max_prob;
    //     cur_prob=map(cur_prob, min_prob, max_prob, 0.0, 255);
    //
    //     color=uvec4(cur_prob,cur_prob,cur_prob,255);
    // }



    //attempt 2 normalizing by the nr of times modified
    uint cur_times_modified=imageLoad(semantics_nr_times_modified_global_tex, img_coords_sem).x;
    uint is_page_commited=texelFetch(pages_commited_volume_sampler,  ivec3(x_page_idx, y_page_idx, class_id), 0).x;
    float cur_prob=texelFetch(classes_probs_global_volume_sampler, ivec3(img_coords_sem, class_id), 0).x;

    if(int(is_page_commited)==0 || int(cur_times_modified)==0 || cur_prob==0.0){
        slice_color=uvec4(bg_color.xyz,255); //if the page is not allocated color is the background
        // slice_color=uvec4(0,0,0,255);
        // slice_color=uvec4(uint(bg_color.x*255), uint(bg_color.y*255), uint(bg_color.z*255), 255 );
        // slice_color=bg_color;
    }else{ // page IS allocated and the is some data there


        // //cap it between min prob and max prob and interpolate
        // if(cur_prob<min_prob) cur_prob=min_prob;
        // if(cur_prob>max_prob) cur_prob=max_prob;
        // cur_prob=map(cur_prob, min_prob, max_prob, 0.0, 255);

        //normalize by the nr of times modified
        // cur_prob=(cur_prob/cur_times_modified)*255;



        // slice_color=uvec4(0,cur_prob,0,cur_prob);
        // color=uvec4(cur_times_modified*255,0,0,255);

        //merge bg and the slice color
        cur_prob=(cur_prob/cur_times_modified);
        slice_color= uvec4(cur_prob*uvec4(0,255,0,255)) + uvec4((1-cur_prob)*uvec4(bg_color.xyz,255));


    }





    // //check if the reads from page commited are correct
    // //if the page is not allocated color is black
    // uint is_page_commited=texelFetch(pages_commited_volume_sampler,  ivec3(x_page_idx, y_page_idx, class_id), 0).x;
    // if(int(is_page_commited)==0){
    //     color=uvec4(0,0,0,255);
    // }else{ // page IS allocated
    //     color=uvec4(255,0,0,255);
    // }


    //store
    imageStore(semantics_one_class_global_tex, img_coords_sem , slice_color );
    // imageStore(semantics_one_class_global_tex, img_coords_sem , bg_color );



}
