#version 330 core
layout(location = 0) in vec3 v_pos_in;
uniform mat4 MVP;
void main(){
    gl_Position =  MVP * vec4(v_pos_in,1.0);
}

// uniform mat4 MVP;
// attribute vec3 vCol;
// attribute vec2 vPos;
// varying vec3 color;
// void main()
// {
//     gl_Position = MVP * vec4(vPos, 0.0, 1.0);
//     color = vCol;
// }
