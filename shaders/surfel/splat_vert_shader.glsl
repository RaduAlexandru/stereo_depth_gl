#version 330 core
#extension GL_ARB_separate_shader_objects : require  //in order to specify the location of the output
layout(location = 0) in vec3 v_pos_in;
layout(location = 1) in vec3 v_normal_in;
layout(location = 2) in float v_radius_in;

layout(location=0) out vec3 v_pos_out;
layout(location=1) out vec3 v_normal_out;
layout(location=2) out float v_radius_out;
//layout 3 is texcoord from the geom shader

uniform mat4 MVP;

void main(){
//    gl_Position =  MVP * vec4(v_pos_in,1.0);
//    gl_PointSize = v_radius_in;



    v_pos_out=v_pos_in;
    v_normal_out=v_normal_in;
    v_radius_out=v_radius_in;

}