#version 430

layout (local_size_x = 32, local_size_y = 32) in;

uniform usampler2D semantics_idxs_global_tex_sampler;
uniform usampler2DArray  semantics_modified_volume_sampler;

layout(binding=0, rgba8ui) uniform writeonly uimage2D semantics_global_tex;
layout(binding=1, r8ui) uniform writeonly uimage2DArray semantics_modified_volume; //GL_R8UI


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
    uint vis=texelFetch(semantics_modified_volume_sampler, ivec3(img_coords_sem,1),0 ).x;
    if(vis==0){
        return; //texel not modified in this frame, no need to do anything
    }else{
        imageStore(semantics_modified_volume, ivec3(img_coords_sem,1) , uvec4(0,0,0,0) ); //Texel was modified, set to unmodified
    }



    //apply color map
    uint best_class_idx=texelFetch(semantics_idxs_global_tex_sampler, img_coords_sem, 0).x;
    vec4 color=vec4(COLOR_MASKS[best_class_idx],1.0);
    imageStore(semantics_global_tex, img_coords_sem , uvec4(color*255) );



}
