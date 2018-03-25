#version 330 core

layout(points) in;
//layout(points, max_vertices = 1) out;
layout(triangle_strip, max_vertices = 4) out;

layout(location=0) in vec3 v_pos_in[];
layout(location=1) in vec3 v_normal_in[];
layout(location=2) in float v_radius_in[]; // Output from vertex shader for each vertex

layout(location=3) out vec2 tex_coord;
layout(location=4) out vec3 position_eye_out;
layout(location=5) out vec3 normal_eye_out;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform mat4 MVP;

void main()
{
//    //simple passthgough
//    gl_Position = gl_in[0].gl_Position;
//    gl_PointSize = gl_in[0].gl_PointSize;
//    EmitVertex();
//    EndPrimitive();

//    //emit 4 vertices
//    gl_Position = gl_in[0].gl_Position + vec4(-0.1, -0.1, 0.0, 0.0);
//    EmitVertex();
//
//    gl_Position = gl_in[0].gl_Position + vec4(-0.1, 0.1, 0.0, 0.0);
//    EmitVertex();
//
//    gl_Position = gl_in[0].gl_Position + vec4(0.1, -0.1, 0.0, 0.0);
//    EmitVertex();
//
//    gl_Position = gl_in[0].gl_Position + vec4(0.1, 0.1, 0.0, 0.0);
//    EmitVertex();
//    EndPrimitive();


//    //simple passthgough but getting in the position in world space
//    gl_Position = MVP * vec4(v_pos_in[0],1.0);
//    gl_PointSize = v_radius_in[0];
//    EmitVertex();
//    EndPrimitive();


    //make the quad
    //get 2 vectors in the tangent plane
    //first one is the cross product between the normal and any other vector (just ensure it's not paralel to it)
    vec3 random_vec=vec3(v_normal_in[0].y,v_normal_in[0].x,v_normal_in[0].z); //to ensure it's not paralel we just shuffle the numbers in the vec3
    vec3 u=normalize(cross(v_normal_in[0],random_vec ));
    vec3 v=normalize(cross(v_normal_in[0], u));
    vec3 pos_quad_corner;


    pos_quad_corner=v_pos_in[0] + u*v_radius_in[0];
    tex_coord = vec2(-1.0, -1.0);
    gl_Position = MVP * vec4(pos_quad_corner ,1.0);
    position_eye_out = vec3 (view * model * vec4 (pos_quad_corner , 1.0));
    normal_eye_out = vec3 (view * model * vec4 (v_normal_in[0], 0.0));
    normal_eye_out = normalize(normal_eye_out);
    EmitVertex();


    pos_quad_corner=v_pos_in[0] + v*v_radius_in[0];
    tex_coord = vec2(1.0, -1.0);
    gl_Position = MVP * vec4(pos_quad_corner ,1.0);
    position_eye_out = vec3 (view * model * vec4 (pos_quad_corner , 1.0));
    normal_eye_out = vec3 (view * model * vec4 (v_normal_in[0], 0.0));
    normal_eye_out = normalize(normal_eye_out);
    EmitVertex();


    pos_quad_corner=v_pos_in[0] - v*v_radius_in[0];
    tex_coord = vec2(-1.0, 1.0);
    gl_Position = MVP * vec4(pos_quad_corner ,1.0);
    position_eye_out = vec3 (view * model * vec4 (pos_quad_corner , 1.0));
    normal_eye_out = vec3 (view * model * vec4 (v_normal_in[0], 0.0));
    normal_eye_out = normalize(normal_eye_out);
    EmitVertex();


    pos_quad_corner=v_pos_in[0] - u*v_radius_in[0];
    tex_coord = vec2(1.0, 1.0);
    gl_Position = MVP * vec4(pos_quad_corner ,1.0);
    position_eye_out = vec3 (view * model * vec4 (pos_quad_corner , 1.0));
    normal_eye_out = vec3 (view * model * vec4 (v_normal_in[0], 0.0));
    normal_eye_out = normalize(normal_eye_out);
    EmitVertex();
    EndPrimitive();


}