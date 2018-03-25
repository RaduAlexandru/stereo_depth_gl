#version 330 core
#extension GL_ARB_separate_shader_objects : require  //in order to specify the location of the input
#extension  GL_ARB_shader_image_load_store : require
#extension GL_ARB_explicit_uniform_location : require
#extension GL_ARB_shading_language_420pack : require  //for specfy location of sampler

//in
layout(location = 0) in vec3 position_in;
layout(location = 1) in vec2 texcoord_in;
layout(location = 2) in vec3 v_normal_in;
layout(location = 3) in vec4 shadow_coord_in;
layout(location = 4) in vec3 eye_vector;

//out
layout(location = 0) out vec4 color;

//sampler and images
uniform sampler2D shadow_map;  //assigning a binding to them doesn't seem to work for some reason
uniform sampler2D rgb_sampler;
uniform usampler2D mask_sampler;
uniform sampler2D probs_sampler;
uniform usampler2D classes_idxs_sampler;
uniform usampler2DArray pages_commited_volume_sampler;

layout(binding=0, rgba8ui) uniform writeonly uimage2D rgb_global_tex;
layout(binding=1, r8ui) uniform writeonly uimage2DArray rgb_modified_volume; //GL_R8UI
layout(binding=2, r8ui) uniform coherent uimage2DArray semantics_modified_volume; //GL_R8UI
layout(binding=3, r32f) uniform coherent image2DArray classes_probs_global_volume;
layout(binding=4, r8ui) uniform writeonly uimage2DArray pages_to_be_commited_volume; //GL_R8UI
layout(binding=5, r32f) uniform coherent image2D rgb_quality_tex;


//misc uniforms (shares uniform with the vertex shader so don't use the same location)
layout(location=2) uniform int rgb_global_tex_size = 0;
layout(location=3) uniform int semantics_global_tex_size = 0;
layout(location=4) uniform int page_size_x = 0;
layout(location=5) uniform int page_size_y = 0;


void main(){

  vec4 scPostW = shadow_coord_in / shadow_coord_in.w;

  if (shadow_coord_in.w <= 0.0f || (scPostW.x < 0 || scPostW.y < 0) || (scPostW.x >= 1 || scPostW.y >= 1)) {
      // Behind or outside frustrum: no shadow
      discard; // return doesn't really stop the computation, you need to discard
  }

  float dist_from_cam = textureProj(shadow_map, shadow_coord_in).x;
  float epsilon = 0.00001;
  if (dist_from_cam + epsilon < scPostW.z){
      //in shadow
      discard;
  }

  //check if the fragment is in the masked out area
  uint mask_val=textureProj(mask_sampler,shadow_coord_in).x;
  if(int(mask_val)==0){
      discard;
  }

  //check if we are seeing the fragment from behind (the dot between the normal and the eye vector is negative)
  float view_angle=dot(eye_vector,v_normal_in);
  if(view_angle<0.05){ //not actually completely behind but close to it
      discard;
  }



  ivec2 img_coords_rgb;
  img_coords_rgb[0]=int(floor(texcoord_in[0]*rgb_global_tex_size));
  img_coords_rgb[1]=int(floor(texcoord_in[1]*rgb_global_tex_size));



  //RGB----------------
  float cur_rgb_quality=imageLoad(rgb_quality_tex, img_coords_rgb).x;
  //compute quality for this fragment
  float weight_dist=1.0;
  float weight_angle=0.1;
  float new_rgb_quality=weight_dist*(1.0/dist_from_cam) + weight_angle*view_angle;  //the shorther the distance the highter the quality

  //if it's better store it and update our rgb color
  float quality_improvement=new_rgb_quality-cur_rgb_quality;
  if(quality_improvement>0.0001){
      imageStore(rgb_quality_tex, img_coords_rgb , vec4(new_rgb_quality,0,0,1) );
      imageStore(rgb_modified_volume, ivec3(img_coords_rgb.xy,0) , uvec4(1,0,0,0) );
      imageStore(rgb_modified_volume, ivec3(img_coords_rgb.xy,1) , uvec4(1,0,0,0) );

      color=textureProj(rgb_sampler,shadow_coord_in);
      imageStore(rgb_global_tex, img_coords_rgb , uvec4(color[0]*255,color[1]*255,color[2]*255,255) );
  }


  //SEMANTICS----------
  ivec2 img_coords_sem;
  img_coords_sem[0]=int(floor(texcoord_in[0]*semantics_global_tex_size));
  img_coords_sem[1]=int(floor(texcoord_in[1]*semantics_global_tex_size));

  //see if we didn't already update the semantic texel in this frame
  uint vis_sem=imageLoad(semantics_modified_volume, ivec3(img_coords_sem.xy,1)).x;
  if(int(vis_sem)==0){
      //store the visibility of the class so that we can blur later the fragments around the charts that were not rasterized
      imageStore(semantics_modified_volume, ivec3(img_coords_sem.xy,0) , uvec4(1,0,0,0) );
      imageStore(semantics_modified_volume, ivec3(img_coords_sem.xy,1) , uvec4(1,0,0,0) );

      //get the class that is being visible and voted for that texel
      uint class_idx=textureProj(classes_idxs_sampler,shadow_coord_in).x;

      //store in a 3D page array the fact that the page correspondin to this fragment and class should be commited
      //get position of the page that contains point y,x
      int x_page_idx=img_coords_sem[0]/page_size_x;
      int y_page_idx=img_coords_sem[1]/page_size_y;
      imageStore(pages_to_be_commited_volume,  ivec3(x_page_idx, y_page_idx ,int(class_idx)) , uvec4(1,0,0,0) );

      uint is_page_commited=texelFetch(pages_commited_volume_sampler,  ivec3(x_page_idx, y_page_idx, int(class_idx)), 0).x;
      if(int(is_page_commited)==0){
          discard; //page is not commited so no need to bake anything
      }


      //bake the info we have in the volume
      float prob=textureProj(probs_sampler,shadow_coord_in).x;
      // prob=log(prob*66);

      //attempt 1, just summing probabilities
      float cur_prob=imageLoad(classes_probs_global_volume, ivec3(img_coords_sem.xy,int(class_idx)) ).x;
      imageStore(classes_probs_global_volume, ivec3(img_coords_sem.xy,int(class_idx)) , vec4(cur_prob+prob,0,0,0) );
      // imageStore(classes_probs_global_volume, ivec3(img_coords_sem.xy,int(class_idx)) , vec4(prob,0,0,0) );


      // //attempt 2 update volume bayesian update
      // float cur_prob=imageLoad(classes_probs_global_volume, ivec3(img_coords_sem.xy,int(class_idx)) ).x;
      // float new_prob=(cur_prob)*prob*66;  //bayesian update as in herman http://strands.acin.tuwien.ac.at/publications/2014/hermans-icra14.pdf
      // //they divide by the prior which is 1/nr of classes, os the same as multiplying with the nr of classes
      // imageStore(classes_probs_global_volume,  ivec3(img_coords_sem.xy,int(class_idx)) , vec4(new_prob,0,0,0) );

      // //attempt 3 more close to semantic fusion
      // float cur_prob=imageLoad(classes_probs_global_volume, ivec3(img_coords_sem.xy,int(class_idx)) ).x;
      // float new_prob=cur_prob*prob*66;
      // imageStore(classes_probs_global_volume,  ivec3(img_coords_sem.xy,int(class_idx)) , vec4(new_prob,0,0,0) );


  }






      // //get Z and normalize by it
      // float Z=0;
      // for(int i = 0; i < 66; i++) {
      //     float val=imageLoad(classes_probs_global_volume, ivec3(img_coords.xy,i) ).x;
      //     // if(val==0) val=1.0/66;
      //     // if(i==int(class_idx)) val=new_prob;
      //     Z=Z+val;
      // }
      // if(Z!=0){
      //     //normalize
      //     for(int i = 0; i < 66; i++) {
      //         float val=imageLoad(classes_probs_global_volume, ivec3(img_coords.xy,i) ).x;
      //         // if(val==0) val=1.0/66;
      //         float new_val=val/Z;
      //         imageStore(classes_probs_global_volume, ivec3(img_coords.xy,i) , vec4(new_val,0,0,0) );
      //     }
      // }
      //
      //
      // //argmax
      // int  max_idx=0;
      // float max_val=0;
      // for(int i = 0; i < 66; i++) {
      //     float val=imageLoad(classes_probs_global_volume, ivec3(img_coords.xy,i)).x;
      //     // if(val==0) val=1.0/66;
      //     if(val>max_val){
      //         max_val=val;
      //         max_idx=i;
      //     }
      // }
      //
      // //color that argmax idx (either rg or with the class color )
      //
      // color=vec4(COLOR_MASKS[max_idx],1.0);



   // imageStore(rgb_global_tex, img_coords , uvec4(color[0]*255,color[1]*255,color[2]*255,color[3]*255) );

   // //just store the same color in the 8 neighbours also HACK
   // uint class_idx_0=imageLoad(visibility_tex,img_coords+ivec2(1,1)).x; imageStore(rgb_global_tex, img_coords+ivec2(1,1) , uvec4(color[0]*255*class_idx_0,color[1]*255*class_idx_0,color[2]*255*class_idx_0,255) );
   // uint class_idx_1=imageLoad(visibility_tex,img_coords+ivec2(1,0)).x; imageStore(rgb_global_tex, img_coords+ivec2(1,0) , uvec4(color[0]*255*class_idx_1,color[1]*255*class_idx_1,color[2]*255*class_idx_1,255) );
   // uint class_idx_2=imageLoad(visibility_tex,img_coords+ivec2(1,-1)).x; imageStore(rgb_global_tex, img_coords+ivec2(1,-1) , uvec4(color[0]*255*class_idx_2,color[1]*255*class_idx_2,color[2]*255*class_idx_2,255) );
   //
   // uint class_idx_3=imageLoad(visibility_tex,img_coords+ivec2(0,1)).x; imageStore(rgb_global_tex, img_coords+ivec2(0,1) , uvec4(color[0]*255*class_idx_3,color[1]*255*class_idx_3,color[2]*255*class_idx_3,255) );
   // uint class_idx_4=imageLoad(visibility_tex,img_coords+ivec2(0,0)).x; imageStore(rgb_global_tex, img_coords+ivec2(0,0) , uvec4(color[0]*255*class_idx_4,color[1]*255*class_idx_4,color[2]*255*class_idx_4,255) );
   // uint class_idx_5=imageLoad(visibility_tex,img_coords+ivec2(0,-1)).x; imageStore(rgb_global_tex, img_coords+ivec2(0,-1) , uvec4(color[0]*255*class_idx_5,color[1]*255*class_idx_5,color[2]*255*class_idx_5,255) );
   //
   // uint class_idx_6=imageLoad(visibility_tex,img_coords+ivec2(-1,1)).x; imageStore(rgb_global_tex, img_coords+ivec2(-1,1) , uvec4(color[0]*255*class_idx_6,color[1]*255*class_idx_6,color[2]*255*class_idx_6,255) );
   // uint class_idx_7=imageLoad(visibility_tex,img_coords+ivec2(-1,0)).x; imageStore(rgb_global_tex, img_coords+ivec2(-1,0) , uvec4(color[0]*255*class_idx_7,color[1]*255*class_idx_7,color[2]*255*class_idx_7,255) );
   // uint class_idx_8=imageLoad(visibility_tex,img_coords+ivec2(-1,-1)).x; imageStore(rgb_global_tex, img_coords+ivec2(-1,-1) , uvec4(color[0]*255*class_idx_8,color[1]*255*class_idx_8,color[2]*255*class_idx_8,255) );

   // discard;



   // //store it
   // imageStore(visibility_tex, img_coords , uvec4(class_idx,class_idx,class_idx,255) );
   // // imageStore(visibility_tex, img_coords , uvec4(255,0,0,255) );
   //
   // // vec4 pixel = imageLoad(visibility_tex, img_coords);
   // // pixel.r = 1;
   // // imageStore(visibility_tex, img_coords, uvec4(class_idx,0,0,255));
   //
   //
   // float class_idx_f=float(class_idx)/255.0;
   // color=vec4(class_idx_f,class_idx_f,class_idx_f,1.0);



}



//debug stuff that can be shown as color
    // //show the normals (normals have components with values in range [-1,1], we want them in [0,1])
    // vec3 normal_scaled=(v_normal_in+1)*0.5;
    // color=vec4(normal_scaled.x, normal_scaled.y, normal_scaled.z, 1);

    //show the eye vector (it's not correct in some parts so therefore the dot is also wrong)
    // color=vec4(eye_vector,1);

    //show the dot between eye vector and the normal NOT normalized to [0,1]
    // float dot_val=dot(eye_vector,v_normal_in);
    // color=vec4(dot_val, dot_val, dot_val, 1);
