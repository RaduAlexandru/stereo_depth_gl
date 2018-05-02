#version 330 core
layout(location = 0) in vec3 v_pos_in;
void main(){
    gl_Position =  vec4(v_pos_in,1.0);
}
