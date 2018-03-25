#version 430

layout (local_size_x = 32, local_size_y = 32) in;

uniform sampler2DArray classes_probs_global_volume_sampler;
uniform usampler2DArray  semantics_modified_volume_sampler;
uniform usampler2DArray pages_commited_volume_sampler;

// layout(binding=0, rgba8ui) uniform writeonly uimage2D semantics_global_tex;
layout(binding=0, r8ui) uniform writeonly uimage2D semantics_idxs_global_tex;
layout(binding=1, r32f) uniform writeonly image2D semantics_probs_global_tex;
layout(binding=2, r8ui) uniform coherent uimage2DArray semantics_modified_volume; //GL_R8UI
layout(binding=3, r32f) uniform coherent image2DArray classes_probs_global_volume;
layout(binding=4, r16ui) uniform coherent uimage2D semantics_nr_times_modified_global_tex;

layout(location=4) uniform int page_size_x = 0;
layout(location=5) uniform int page_size_y = 0;

//https://www.opengl.org/discussion_boards/showthread.php/170193-Constant-vec3-array-no-go
vec3 COLOR_MASKS[66] = vec3[](
    vec3(0.647059, 0.164706, 0.164706),
    vec3(       0, 0.752941,        0),
    vec3(0.768627, 0.768627, 0.768627),
    vec3(0.745098,      0.6,      0.6),
    vec3(0.705882, 0.647059, 0.705882),
    vec3(     0.4,      0.4, 0.611765),
    vec3(     0.4,      0.4, 0.611765),
    vec3(0.501961,  0.25098,        1),
    vec3( 0.54902,  0.54902, 0.784314),
    vec3(0.666667, 0.666667, 0.666667),
    vec3(0.980392, 0.666667, 0.627451),
    vec3(0.376471, 0.376471, 0.376471),
    vec3(0.901961, 0.588235,  0.54902),
    vec3(0.501961,  0.25098, 0.501961),
    vec3(0.431373, 0.431373, 0.431373),
    vec3(0.956863, 0.137255, 0.909804),
    vec3(0.588235, 0.392157, 0.392157),
    vec3( 0.27451,  0.27451,  0.27451),
    vec3(0.588235, 0.470588, 0.352941),
    vec3(0.862745,0.0784314, 0.235294),
    vec3(       1,        0,        0),
    vec3(       1,        0,        0),
    vec3(       1,        0,        0),
    vec3(0.784314, 0.501961, 0.501961),
    vec3(       1,        1,        1),
    vec3( 0.25098, 0.666667,  0.25098),
    vec3(0.501961,  0.25098,  0.25098),
    vec3( 0.27451, 0.509804, 0.705882),
    vec3(       1,        1,        1),
    vec3(0.596078, 0.984314, 0.596078),
    vec3(0.419608, 0.556863, 0.137255),
    vec3(       0, 0.666667, 0.117647),
    vec3(       1,        1, 0.501961),
    vec3(0.980392,        0, 0.117647),
    vec3(       0,        0,        0),
    vec3(0.862745, 0.862745, 0.862745),
    vec3(0.666667, 0.666667, 0.666667),
    vec3(0.870588, 0.156863, 0.156863),
    vec3(0.392157, 0.666667, 0.117647),
    vec3(0.156863, 0.156863, 0.156863),
    vec3(0.129412, 0.129412, 0.129412),
    vec3(0.666667, 0.666667, 0.666667),
    vec3(       0,        0, 0.556863),
    vec3(0.666667, 0.666667, 0.666667),
    vec3(0.823529, 0.666667, 0.392157),
    vec3(     0.6,      0.6,      0.6),
    vec3(0.501961, 0.501961, 0.501961),
    vec3(       0,        0, 0.556863),
    vec3(0.980392, 0.666667, 0.117647),
    vec3(0.752941, 0.752941, 0.752941),
    vec3(0.862745, 0.862745,        0),
    vec3(0.705882, 0.647059, 0.705882),
    vec3(0.466667,0.0431373,  0.12549),
    vec3(       0,        0, 0.556863),
    vec3(       0, 0.235294, 0.392157),
    vec3(       0,        0, 0.556863),
    vec3(       0,        0, 0.352941),
    vec3(       0,        0, 0.901961),
    vec3(       0, 0.313725, 0.392157),
    vec3(0.501961,  0.25098,  0.25098),
    vec3(       0,        0, 0.431373),
    vec3(       0,        0,  0.27451),
    vec3(       0,        0, 0.752941),
    vec3( 0.12549,  0.12549,  0.12549),
    vec3(       0,        0,        0),
    vec3(       0,        0,        0)

                             );

void main(void) {

    ivec2 img_coords_sem = ivec2(gl_GlobalInvocationID.xy);


    //if the texel was not updated in this frame then we don't need to normalize anything for it
    // uint vis=imageLoad(semantics_modified_volume,ivec3(img_coords_sem,1)).x;
    uint vis=texelFetch(semantics_modified_volume_sampler, ivec3(img_coords_sem,1),0 ).x;
    if(vis==0){
        return;
    }
    // imageStore(semantics_modified_volume, ivec3(img_coords_sem,1) , uvec4(0,0,0,0) );




    // //get Z and normalize by it
    // float Z=0;
    // for(int i = 0; i < 66; i++) {
    //     // float val=imageLoad(classes_probs_global_volume, ivec3(img_coords_sem.xy,i) ).x;
    //     float val=texelFetch(classes_probs_global_volume_sampler, ivec3(img_coords_sem.xy,i),0 ).x;
    //     // if(val==0) val=1.0/66;
    //     // if(i==int(class_idx)) val=new_prob;
    //     Z=Z+val;
    // }
    //
    // // //normalize
    // // if(Z!=0){
    // //     for(int i = 0; i < 66; i++) {
    // //         float val=imageLoad(classes_probs_global_volume, ivec3(img_coords_sem.xy,i) ).x;
    // //         // if(val==0) val=1.0/66;
    // //         float new_val=val/Z;
    // //         imageStore(classes_probs_global_volume, ivec3(img_coords_sem.xy,i) , vec4(new_val,0,0,0) );
    // //     }
    // // }
    //
    // //normalize and argmax all in one loop (we do it like this because the argmax doesn't change weather or not the pdf is normalized or not)
    // int  max_idx=0;
    // float max_val=0;
    // for(int i = 0; i < 66; i++) {
    //     // float val=imageLoad(classes_probs_global_volume, ivec3(img_coords_sem.xy,i) ).x;
    //     float val=texelFetch(classes_probs_global_volume_sampler, ivec3(img_coords_sem.xy,i),0 ).x;
    //     if(val>max_val){
    //         max_val=val;
    //         max_idx=i;
    //     }
    //     if(Z!=0){
    //         // if(val==0) val=1.0/66;
    //         float new_val=val/Z;
    //         imageStore(classes_probs_global_volume, ivec3(img_coords_sem.xy,i) , vec4(new_val,0,0,0) );
    //     }
    // }
    //
    //
    // // //argmax
    // // int  max_idx=0;
    // // float max_val=0;
    // // for(int i = 0; i < 66; i++) {
    // //     float val=imageLoad(classes_probs_global_volume, ivec3(img_coords_sem.xy,i)).x;
    // //     // if(val==0) val=1.0/66;
    // //     if(val>max_val){
    // //         max_val=val;
    // //         max_idx=i;
    // //     }
    // // }
    //
    // //color that argmax idx (either rg or with the class color )
    // vec4 color=vec4(COLOR_MASKS[max_idx],1.0);
    //
    // imageStore(semantics_global_tex, img_coords_sem , uvec4(color*255) );





    // //attempt 2 more close to semantic fusion
    // //get Z and normalize by it
    // float Z=0;
    // for(int i = 0; i < 66; i++) {
    //     float val=texelFetch(classes_probs_global_volume_sampler, ivec3(img_coords_sem.xy,i),0 ).x;
    //     // if(val==0) val=1.0/255;
    //     Z=Z+val;
    // }
    // //normalize and argmax all in one loop (we do it like this because the argmax doesn't change weather or not the pdf is normalized or not)
    // int  max_idx=0;
    // float max_val=0;
    // for(int i = 0; i < 66; i++) {
    //     float val=texelFetch(classes_probs_global_volume_sampler, ivec3(img_coords_sem.xy,i),0 ).x;
    //     // if(val==0) val=1.0/66;
    //     if(Z<=1e-5){
    //         // Something has gone unexpectedly wrong - reinitialse
    //         val = 1.0f / 66;
    //         imageStore(classes_probs_global_volume, ivec3(img_coords_sem.xy,i) , vec4(val,0,0,0) );
    //     }else{
    //         float new_val=val/Z;
    //         imageStore(classes_probs_global_volume, ivec3(img_coords_sem.xy,i) , vec4(new_val,0,0,0) );
    //         if(new_val>max_val){
    //             max_val=new_val;
    //             max_idx=i;
    //         }
    //     }
    // }
    // //color that argmax idx (either rg or with the class color )
    // vec4 color=vec4(COLOR_MASKS[max_idx],1.0);
    // imageStore(semantics_global_tex, img_coords_sem , uvec4(color*255) );


    // //attempt 3 intended to work with just the summing of probs
    // memoryBarrier();
    // barrier();
    // int x_page_idx=img_coords_sem[0]/page_size_x;
    // int y_page_idx=img_coords_sem[1]/page_size_y;
    // //argmax
    // int  max_idx=0;
    // float max_val=0;
    // for(int i = 0; i < 66; i++) {
    //     uint is_page_commited=imageLoad(pages_commited_volume,  ivec3(x_page_idx, y_page_idx ,i)).x;
    //     if(int(is_page_commited)==0){
    //         //page is not commited
    //         continue;
    //     }
    //     float val=imageLoad(classes_probs_global_volume, ivec3(img_coords_sem.xy,i)).x;
    //     // if(val==0) val=1.0/66;
    //     if(val>max_val){
    //         max_val=val;
    //         max_idx=i;
    //     }
    // }
    // //color that argmax idx (either rg or with the class color )
    // vec4 color=vec4(COLOR_MASKS[max_idx],1.0);
    // imageStore(semantics_global_tex, img_coords_sem , uvec4(color*255) );



    //attempt 4 intended to work with the negative log likelihood
    memoryBarrier();
    barrier();
    int x_page_idx=img_coords_sem[0]/page_size_x;
    int y_page_idx=img_coords_sem[1]/page_size_y;
    //argmax
    int  best_class_idx=0;
    float best_likelihood=-999999;
    for(int i = 0; i < 66; i++) {
        uint is_page_commited=texelFetch(pages_commited_volume_sampler,  ivec3(x_page_idx, y_page_idx, i), 0).x;
        if(int(is_page_commited)==0){
            continue; //page is not commited
        }
        float val=imageLoad(classes_probs_global_volume, ivec3(img_coords_sem.xy,i)).x;
        if(val>best_likelihood){
            best_likelihood=val;
            best_class_idx=i;
        }
    }
    //store the best class idx and best likelihood
    imageStore(semantics_idxs_global_tex, img_coords_sem , uvec4(best_class_idx,0,0,0) );
    imageStore(semantics_probs_global_tex, img_coords_sem , vec4(best_likelihood,0,0,0) );

    //increase by 1 the nr of times we modify this texel
    uint cur_times_modified=imageLoad(semantics_nr_times_modified_global_tex, img_coords_sem).x;
    imageStore(semantics_nr_times_modified_global_tex, img_coords_sem , uvec4(cur_times_modified+1,0,0,0) );



    // //color that argmax idx (either rg or with the class color )
    // vec4 color=vec4(COLOR_MASKS[max_idx],1.0);
    // imageStore(semantics_global_tex, img_coords_sem , uvec4(color*255) );




    memoryBarrier();
    barrier();


    //fill around the fragment so we don't have texels around the charts which are not updated
    for (int i=0; i<3; i++){
       for (int j=0; j<3; j++) {
           if (i != 1 && j != 1){
               //see if the neighbouring texel has never been modified
               ivec2 neigh_coords=img_coords_sem + ivec2(i-1,j-1);
               uint modif=texelFetch(semantics_modified_volume_sampler, ivec3(neigh_coords,0),0 ).x;
               if(modif==0){
                   // imageStore(semantics_global_tex, neigh_coords , uvec4(color*255) );
                   imageStore(semantics_idxs_global_tex, neigh_coords , uvec4(best_class_idx,0,0,0) );
                   imageStore(semantics_probs_global_tex, neigh_coords , vec4(best_likelihood,0,0,0) );
                   imageStore(semantics_modified_volume, ivec3(neigh_coords,1) , uvec4(1,0,0,0) ); //set the neigh pixel as modified so that it gets affected by the subsequent color mapping
               }
           }
       }
    }






}
