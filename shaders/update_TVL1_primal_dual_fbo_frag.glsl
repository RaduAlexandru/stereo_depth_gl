#version 430 core
#extension GL_ARB_separate_shader_objects : require  //in order to specify the location of the input
#extension  GL_ARB_shader_image_load_store : require
#extension GL_ARB_explicit_uniform_location : require
#extension GL_ARB_shading_language_420pack : require  //for specfy location of sampler

layout(location = 0) out vec4 color;





//if you assign a binding to this one it doesn't read the values anymore, god knows why..
layout (std140) uniform params_block{
    float outlierTH;					// higher -> less strict
    float overallEnergyTHWeight;
    float outlierTHSumComponent; 		// higher -> less strong gradient-based reweighting .
    float huberTH; // Huber Threshold
    float convergence_sigma2_thresh;      //!< threshold on depth uncertainty for convergence.
    float eta;

    float gradH_th;
    int search_epi_method; //0=bca, 1=ngf
    //pad to 16 bytes if needed  (blocks of 4 floats)
    // float pad_1;
    // float pad_2;
    //until here it's paded correctly to 16 bytes-----

    int denoise_nr_iterations;
    float denoise_depth_range;
    float denoise_lambda;
    float denoise_L;
    float denoise_tau;
    float denoise_theta;
    float pad_1;
    float pad_2;

}params ;


layout(binding=0, rgba32f) uniform coherent image2D fbo_rgba_texture; //stores val_mu_denoised mu_head amd p_vec
layout(binding=1, rg32f) uniform coherent image2D tex_texture; //GL_RG32F //stores the things gval and mu


// const int G_VAL_DEPTH=0; //--readonly
// const int MU_DEPTH=1; //-readonly
// const int VAL_MU_DENOISED_DEPTH=2;
// const int MU_HEAD_DEPTH=3;
// const int P_VEC_DEPTH=4; //we leave it for last since it will occupy actually possition 4 and 5 since it's a vector
// //5 is occupied by pvecdepth still

void main(void) {

    ivec2 id =ivec2(gl_FragCoord.xy);
    ivec2 pos_xy=ivec2( id );
    ivec2 pos_right=ivec2( id.x+1, id.y );
    ivec2 pos_below=ivec2( id.x,  id.y-1 );
    ivec2 pos_left=ivec2(  id.x-1, id.y );
    ivec2 pos_above=ivec2(  id.x, id.y+1 );


    //check if we hav a valid current pixel
    float check_curr=imageLoad(tex_texture, ivec2(gl_FragCoord.xy) ).x;
    if(check_curr==-1){
        discard;
    }


    const float sigma = ((1.0 / (params.denoise_L*params.denoise_L)) / params.denoise_tau);
    //read once the read only stuff to speed up the things
    float g_val=imageLoad(tex_texture, pos_xy).x;
    float mu=imageLoad(tex_texture, pos_xy).y;
    //Read some other stuff even though they will be updated later
    float val_mu_denoised=imageLoad(fbo_rgba_texture,pos_xy ).x;
    vec2 current_p_vec;
    current_p_vec.x=imageLoad(fbo_rgba_texture,pos_xy ).z;
    current_p_vec.y=imageLoad(fbo_rgba_texture,pos_xy ).w;



    //check if we have neighbours and get the mu_head of right and below
    float check_right=imageLoad(tex_texture, pos_right ).x;
    float check_below=imageLoad(tex_texture, pos_right ).x;
    float right_mu_head;
    float below_mu_head;
    if(check_right==-1){
        //get the current one
        right_mu_head=imageLoad(fbo_rgba_texture, pos_xy ).y;
    }else{
        right_mu_head=imageLoad(fbo_rgba_texture, pos_right ).y;
    }
    if(check_below==-1){
        //get the current one
        right_mu_head=imageLoad(fbo_rgba_texture, pos_xy ).y;
    }else{
        below_mu_head=imageLoad(fbo_rgba_texture, pos_below ).y;
    }
    vec2 grad_uhead=vec2(right_mu_head - val_mu_denoised, below_mu_head - val_mu_denoised);
    const vec2 temp_p = g_val * grad_uhead * sigma + current_p_vec;
    const float sqrt_p = length(temp_p); //sqrt(temp_p[0] * temp_p[0] + temp_p[1] * temp_p[1]);
    current_p_vec = temp_p / max(1.0f, sqrt_p);
    // imageStore(tex_volume, ivec3(pos_xy, P_VEC_DEPTH) , vec4(current_p_vec.x,0,0,0) );
    // imageStore(tex_volume, ivec3(pos_xy, P_VEC_DEPTH+1) , vec4(current_p_vec.y,0,0,0) );
    //TODO maybe we actually NEED to store the thing here
    float cur_mu_head=imageLoad(fbo_rgba_texture, pos_xy ).y;
    imageStore(fbo_rgba_texture, pos_xy, vec4(val_mu_denoised,cur_mu_head,current_p_vec.x,current_p_vec.y) );
    // imageStore(fbo_rgba_texture, pos_xy, vec4(current_p_vec.y,0,0,0) );


    // update primal:
    const float old_u = imageLoad(fbo_rgba_texture, pos_xy ).x;
    //check left and above for pvec
    vec2 left_p;
    vec2 above_p;
    float check_left=imageLoad(tex_texture, pos_left ).x;
    float check_above=imageLoad(tex_texture, pos_above ).x;
    if(check_left==-1){
        //get the current one
        left_p=current_p_vec;
    }else{
        left_p.x=imageLoad(fbo_rgba_texture, pos_left ).z;
        left_p.y=imageLoad(fbo_rgba_texture, pos_left ).w;
    }
    if(check_above==-1){
        //get the current one
        above_p=current_p_vec;
    }else{
        above_p.x=imageLoad(fbo_rgba_texture, pos_above ).z;
        above_p.y=imageLoad(fbo_rgba_texture, pos_above ).w;
    }
    const float divergence = current_p_vec.x - left_p.x + current_p_vec.y - above_p.y;
    const float tauLambda = params.denoise_tau*params.denoise_lambda;
    const float temp_u = old_u + params.denoise_tau * g_val * divergence;
    float diff =temp_u - mu;
    if (diff> (tauLambda)){
        val_mu_denoised = temp_u - tauLambda;
    }else if (diff < (-tauLambda)){
        val_mu_denoised = temp_u + tauLambda;
    }else{
        val_mu_denoised = mu;
    }
    float mu_head = val_mu_denoised + params.denoise_theta * (val_mu_denoised - old_u);
    // imageStore(tex_volume, ivec3(pos_xy, MU_HEAD_DEPTH) , vec4( mu_head ,0,0,0) );
    // imageStore(tex_volume, ivec3(pos_xy, VAL_MU_DENOISED_DEPTH) , vec4( val_mu_denoised ,0,0,0) );
    // imageStore(tex_volume, ivec3(pos_xy, P_VEC_DEPTH) , vec4(current_p_vec.x,0,0,0) );
    // imageStore(tex_volume, ivec3(pos_xy, P_VEC_DEPTH+1) , vec4(current_p_vec.y,0,0,0) );

    //output the new p[id].val_mu_denoised, p[id].mu_head, p[id].p_vec.x, p[id].p_vec.y
    color=vec4(val_mu_denoised,mu_head,current_p_vec.x,current_p_vec.y);

















    //debug if we reached here by making a really big loop
    // for(int i = 0; i < 10000000; i ++){
    //     float val=pow(3,1000);
    // }


    // ivec2 id = ivec2(gl_FragCoord.xy);
    // ivec2 pos_xy=ivec2( id );
    // ivec2 pos_right=ivec2( id.x+1, id.y );
    // ivec2 pos_below=ivec2( id.x,  id.y-1 );
    // ivec2 pos_left=ivec2(  id.x-1, id.y );
    // ivec2 pos_above=ivec2(  id.x, id.y+1 );
    //
    // //check if the current pixel is valid
    // float check_curr=imageLoad(tex_volume, ivec3(pos_xy, G_VAL_DEPTH) ).x;
    // if(check_curr==-1){
    //     return;
    // }
    //
    // // //debug just write a value of 1 in the mu denoised
    // // imageStore(tex_volume, ivec3(pos_xy, VAL_MU_DENOISED_DEPTH) , vec4( 1 ,0,0,0) );
    // // return;
    //
    // //attemot at making at faster
    // const float sigma = ((1.0 / (params.denoise_L*params.denoise_L)) / params.denoise_tau);
    // //read once the read only stuff to speed up the things
    // float g_val=imageLoad(tex_volume, ivec3(pos_xy, G_VAL_DEPTH) ).x;
    // float mu=imageLoad(tex_volume, ivec3(pos_xy, MU_DEPTH) ).x;
    // //Read some other stuff even though they will be updated later
    // float val_mu_denoised=imageLoad(tex_volume, ivec3(pos_xy, VAL_MU_DENOISED_DEPTH) ).x;
    // vec2 current_p_vec;
    // current_p_vec.x=imageLoad(tex_volume, ivec3(pos_xy, P_VEC_DEPTH) ).x;
    // current_p_vec.y=imageLoad(tex_volume, ivec3(pos_xy, P_VEC_DEPTH+1) ).x;
    //
    // //update dual
    // //check if we have neighbours and get the mu_head of right and below
    // float check_right=imageLoad(tex_volume, ivec3(pos_right, G_VAL_DEPTH) ).x;
    // float check_below=imageLoad(tex_volume, ivec3(pos_below, G_VAL_DEPTH) ).x;
    // float right_mu_head;
    // float below_mu_head;
    // if(check_right==-1){
    //     //get the current one
    //     right_mu_head=imageLoad(tex_volume, ivec3(pos_xy, MU_HEAD_DEPTH) ).x;
    // }else{
    //     right_mu_head=imageLoad(tex_volume, ivec3(pos_right, MU_HEAD_DEPTH) ).x;
    // }
    // if(check_below==-1){
    //     //get the current one
    //     right_mu_head=imageLoad(tex_volume, ivec3(pos_xy, MU_HEAD_DEPTH) ).x;
    // }else{
    //     below_mu_head=imageLoad(tex_volume, ivec3(pos_below, MU_HEAD_DEPTH) ).x;
    // }
    // vec2 grad_uhead=vec2(right_mu_head - val_mu_denoised, below_mu_head - val_mu_denoised);
    // const vec2 temp_p = g_val * grad_uhead * sigma + current_p_vec;
    // const float sqrt_p = length(temp_p); //sqrt(temp_p[0] * temp_p[0] + temp_p[1] * temp_p[1]);
    // current_p_vec = temp_p / max(1.0f, sqrt_p);
    // imageStore(tex_volume, ivec3(pos_xy, P_VEC_DEPTH) , vec4(current_p_vec.x,0,0,0) );
    // imageStore(tex_volume, ivec3(pos_xy, P_VEC_DEPTH+1) , vec4(current_p_vec.y,0,0,0) );
    //
    //
    // // update primal:
    // const float old_u = imageLoad(tex_volume, ivec3(pos_xy, VAL_MU_DENOISED_DEPTH) ).x;
    // //check left and above for pvec
    // vec2 left_p;
    // vec2 above_p;
    // float check_left=imageLoad(tex_volume, ivec3(pos_left, G_VAL_DEPTH) ).x;
    // float check_above=imageLoad(tex_volume, ivec3(pos_above, G_VAL_DEPTH) ).x;
    // if(check_left==-1){
    //     //get the current one
    //     left_p=current_p_vec;
    // }else{
    //     left_p.x=imageLoad(tex_volume, ivec3(pos_left, P_VEC_DEPTH) ).x;
    //     left_p.y=imageLoad(tex_volume, ivec3(pos_left, P_VEC_DEPTH+1) ).x;
    // }
    // if(check_above==-1){
    //     //get the current one
    //     above_p=current_p_vec;
    // }else{
    //     above_p.x=imageLoad(tex_volume, ivec3(pos_above, P_VEC_DEPTH) ).x;
    //     above_p.y=imageLoad(tex_volume, ivec3(pos_above, P_VEC_DEPTH+1) ).x;
    // }
    // const float divergence = current_p_vec.x - left_p.x + current_p_vec.y - above_p.y;
    // const float tauLambda = params.denoise_tau*params.denoise_lambda;
    // const float temp_u = old_u + params.denoise_tau * g_val * divergence;
    // float diff =temp_u - mu;
    // if (diff> (tauLambda)){
    //     val_mu_denoised = temp_u - tauLambda;
    // }else if (diff < (-tauLambda)){
    //     val_mu_denoised = temp_u + tauLambda;
    // }else{
    //     val_mu_denoised = mu;
    // }
    // float mu_head = val_mu_denoised + params.denoise_theta * (val_mu_denoised - old_u);
    // imageStore(tex_volume, ivec3(pos_xy, MU_HEAD_DEPTH) , vec4( mu_head ,0,0,0) );
    // imageStore(tex_volume, ivec3(pos_xy, VAL_MU_DENOISED_DEPTH) , vec4( val_mu_denoised ,0,0,0) );

    //output the new p[id].val_mu_denoised, p[id].mu_head, p[id].p_vec.x, p[id].p_vec.y
    // color=vec4(0.1,1.0,1.0,1.0);
}
