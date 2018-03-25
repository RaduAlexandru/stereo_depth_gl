#version 330 core
#extension GL_ARB_separate_shader_objects : require  //in order to specify the location of the output
// #extension  GL_NV_conservative_raster: require //to also rasterize around the borders fo the charts
#extension GL_ARB_explicit_uniform_location : require


//in
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texcoord;
layout(location = 2) in vec3 v_normal;

//out
layout(location = 0) out vec3 position_out;
layout(location = 1) out vec2 texcoord_out;
layout(location = 2) out vec3 v_normal_out;
layout(location = 3) out vec4 shadow_coord_out;
layout(location = 4) out vec3 eye_vector; //vector from cam to vertex position (used for rgb quality)


//sampler and images

//misc uniforms (shares uniform with the frag shader so don't use the same location)
layout(location=0) uniform mat4 depth_bias_mvp;
layout(location=1) uniform vec3 eye_pos; //position of the eye (the cam) in world coordinates (used for rgb quality)

void main(){
	//texcoords are in range [0,1] but to put them in screen coordinates and render them we need them in [-1,1]
    vec2 texcoord_scaled=texcoord;
    // texcoord_scaled[1]=1-0-texcoord_scaled[1]; //flip also the y axis just because it's nicer to see it like that
    texcoord_scaled=texcoord_scaled*2;
    texcoord_scaled=texcoord_scaled - 1.0;
    gl_Position = vec4(texcoord_scaled,0,1);

    position_out=position;
    texcoord_out=texcoord;
    v_normal_out=v_normal;

    //https://www.opengl.org/discussion_boards/showthread.php/146682-eye-vector
    eye_vector=normalize(eye_pos-position);

    shadow_coord_out=depth_bias_mvp * vec4(position,1.0);
}
