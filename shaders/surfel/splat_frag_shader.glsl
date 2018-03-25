#version 330 core
#extension GL_ARB_separate_shader_objects : require  //in order to specify the location of the output
layout(location=1) in vec3 v_normal_in;
layout(location=3) in vec2 tex_coord;
layout(location=4) in vec3 position_eye_in;
layout(location=5) in vec3 normal_eye_in;

layout(location = 0) out vec4 out_color;

uniform sampler2D  surfel_tex_sampler;



//stuff neeed for lighting
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
vec3 Ls = vec3 (1, 1, 1);
vec3 Ld = vec3 (1, 1, 1);
vec3 La = vec3 (1, 1, 1);
float specular_exponent=35.0; //shininess
float lighting_factor=1.0;
vec4 Ksi = vec4(0.0, 0.0, 1.0, 1.0);
vec4 Kdi = vec4(0.3, 0.5, 0.2, 1.0);
vec4 Kai = vec4(0.2, 0.2, 1.2, 1.0);
uniform vec3 light_position_world;

void main(){
    vec4 surfel_color=texture(surfel_tex_sampler, tex_coord);
    if(surfel_color.w==0.0){ //it's in the trasnsparent part of the texture
        discard;
    }


    vec3 Ia = La * vec3(Kai);    // ambient intensity

    vec3 light_position_eye = vec3 (view * vec4 (light_position_world, 1.0));
    vec3 vector_to_light_eye = light_position_eye - position_eye_in;
    vec3 direction_to_light_eye = normalize (vector_to_light_eye);
    float dot_prod = dot (direction_to_light_eye, normal_eye_in);
    float clamped_dot_prod = max (dot_prod, 0.0);
    vec3 Id = Ld * vec3(Kdi) * clamped_dot_prod;    // Diffuse intensity
    vec3 reflection_eye = reflect (-direction_to_light_eye, normal_eye_in);
    vec3 surface_to_viewer_eye = normalize (-position_eye_in);
    float dot_prod_specular = dot (reflection_eye, surface_to_viewer_eye);
    dot_prod_specular = float(abs(dot_prod)==dot_prod) * max (dot_prod_specular, 0.0);
    float specular_factor = pow (dot_prod_specular, specular_exponent);
    vec3 Is = Ls * vec3(Ksi) * specular_factor;    // specular intensity
    vec4 color = vec4(lighting_factor * (Is + Id) + Ia + (1.0-lighting_factor) * vec3(Kdi),(Kai.a+Ksi.a+Kdi.a)/3);

//  	out_color = vec4(1.0,1.0,1.0, 1.0);
//    out_color = surfel_color;
    out_color=color;

}