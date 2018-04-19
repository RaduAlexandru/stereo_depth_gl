__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
__constant sampler_t sampler_linear = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

__kernel void simple_copy(  __read_only image2d_t input_image
                         , __write_only image2d_t output_image
                         )
{
    const int2 coords = {get_global_id(0), get_global_id(1)};

    float4 value = read_imagef(input_image, sampler, coords);
    value[2]=0.0;
    write_imagef(output_image, coords, value);
}


__kernel void gaussian_blur( __read_only image2d_t input_image,
                             __constant float * mask,
                             __private int mask_size,
                             __write_only image2d_t output_image
                         )
{
    const int2 coords = {get_global_id(0), get_global_id(1)};

    // Collect neighbor values and multiply with Gaussian
   float sum = 0.0f;
   for(int a = -mask_size; a < mask_size+1; a++) {
       for(int b = -mask_size; b < mask_size+1; b++) {
           sum += mask[a+mask_size+(b+mask_size)*(mask_size*2+1)]
              *read_imagef(input_image, sampler, coords + (int2)(a,b)).x;
       }
   }

   float4 color_out=(float4)(sum,sum,sum,sum);
   write_imagef(output_image, coords, color_out);

    // float4 value = read_imagef(input_image, sampler, coords);
    // value[2]=0.0;
    // write_imagef(output_image, coords, value);
}



constant float mask[3][3] =
{
  {-1, -2, -1},
  {0, 0, 0},
  {1, 2, 1}
};
__kernel void sobel( __read_only image2d_t input,
                      __write_only image2d_t output
                         )
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    // float g_x = 0.f;
    // float g_y = 0.f;
    // for (int j = -1; j <= 1; j++){
    //     for (int i = -1; i <= 1; i++){
    //       float4 p = read_imagef(input, sampler, (int2)(x+i, y+j));
    //       g_x += (p.x*0.299f + p.y*0.587f + p.z*0.114f) * mask[i+1][j+1];
    //       g_y += (p.x*0.299f + p.y*0.587f + p.z*0.114f) * mask[j+1][i+1];
    //   }
    // }
    // float g_mag = sqrt(g_x*g_x + g_y*g_y);
    // write_imagef(output, (int2)(x, y), (float4)(g_mag,g_mag,g_mag,1));


    //attempt 2 for grayscale images
    float g_x = 0.f;
    float g_y = 0.f;
    for (int j = -1; j <= 1; j++){
        for (int i = -1; i <= 1; i++){
          float4 p = read_imagef(input, sampler, (int2)(x+i, y+j));
          g_x += (p.x*0.299f + p.y*0.587f + p.z*0.114f) * mask[i+1][j+1];
          g_y += (p.x*0.299f + p.y*0.587f + p.z*0.114f) * mask[j+1][i+1];
      }
    }
    float g_mag = sqrt(g_x*g_x + g_y*g_y);
    write_imagef(output, (int2)(x, y), (float4)(g_mag,0,0,1));


    // //simple passthrough
    // write_imagef(output, (int2)(x, y),read_imagef(input, sampler, (int2)(x, y)));
}

kernel void blurx( read_only image2d_t input, constant float * mask, private int mask_size, write_only image2d_t output){
    int x = get_global_id(0);
    int y = get_global_id(1);

    // __local float local_mem[2048];
    // //read into local mem
    // int x_local=get_local_id(0);
    // int y_local=get_local_id(1);
    // int idx_local = (y_local * 32) + x_local; // Indexes
    // float4 p_local = read_imagef(input, sampler, (int2)(x, y));
    // local_mem[idx_local]=p_local.x;
    //
    // barrier(CLK_LOCAL_MEM_FENCE);



    float acum=0.0;
    int half_size=mask_size/2;
    for(int i = -half_size; i < half_size +1; i++){
        float4 p = read_imagef(input, sampler, (int2)(x+i, y));
        acum+= p.x* mask[i+half_size];
    }


    // //attemp 2
    // float acum=0.0;
    // int half_size=mask_size/2;
    // for(int i = -half_size; i < half_size +1; i++){
    //     int l = (y_local * 32) + x_local+1; // Indexes
    //     float p = local_mem[l];
    //     acum+= p* mask[i+half_size];
    // }



    //attempt 3 using the approach of
    //http://rastergrid.com/blog/2010/09/efficient-gaussian-blur-with-linear-sampling/
    //http://roxlu.com/2014/045/fast-opengl-blur-shader

     // __local float offsets[23];
     // __local float weights[23];
     // for(int i = 0; i < 23; i ++){
     //     offsets[i]=i;
     //     weights[i]=mask[i+22];
     // }
     //
     // __local float new_offsets[21]; //2 less than the other one
     // __local float new_weights[21];
     // new_weights[0]=weights[0];
     // for(int i = 1; i < 23-1; i++){
     //     new_weights[i]=weights[i]+weights[i+1];
     // }
     // new_offsets[0] = 0.0f;
     // for(int i = 1; i < 23-1; i++) {
     //     new_offsets[i]=( (weights[i] * offsets[i]) + (weights[i+1] * offsets[i+1]) ) / new_weights[i];
     // }
     //
     //
     // // //original formulation didnt need to normalize but otherwise it's too bright
     // // float sum=0.0;
     // // for(int i = 0; i < 21; i++){
     // //     sum+=new_weights[i];
     // // }
     // // for(int i = 0; i < 21; i++){
     // //     new_weights[i]/=sum;
     // // }
     //
     //
     // float acum=0.0;
     // //step through the offset, get samples and accumulate them
     // acum=(read_imagef(input, sampler, (int2)(x, y))).x * new_weights[0];
     // for(int i = 1; i < 23-1; i ++){
     //     acum+=(read_imagef(input, sampler_linear, (int2)(x, y) + (int2)(new_offsets[i], 0)  )).x * new_weights[i];
     //     acum+=(read_imagef(input, sampler_linear, (int2)(x, y) - (int2)(new_offsets[i], 0)  )).x * new_weights[i];
     // }


     // // //attempt 3---------------
     //  __local float new_offsets[3];
     //  new_offsets[0]=0.0;
     //  new_offsets[1]=1.3846153846;
     //  new_offsets[2]=3.2307692308;
     //  __local float new_weights[3];
     //  new_weights[0]=0.2270270270;
     //  new_weights[1]=0.3162162162;
     //  new_weights[2]=0.0702702703;
     //
     //  float acum=0.0;
     //  //step through the offset, get samples and accumulate them
     //  acum=(read_imagef(input, sampler_linear, (int2)(x, y))).x * new_weights[0];
     //  for(int i = 1; i < 3; i ++){
     //      acum+=(read_imagef(input, sampler_linear, (int2)(x, y) + (int2)(new_offsets[i], 0)  )).x * new_weights[i];
     //      acum+=(read_imagef(input, sampler_linear, (int2)(x, y) - (int2)(new_offsets[i], 0)  )).x * new_weights[i];
     //  }





    write_imagef(output, (int2)(x, y), (float4)(acum,0,0,1));

}

kernel void blury( read_only image2d_t input, constant float * mask, private int mask_size, write_only image2d_t output){
    int x = get_global_id(0);
    int y = get_global_id(1);

    float acum=0.0;
    int half_size=mask_size/2;
    for(int i = -half_size; i < half_size +1; i++){
        float4 p = read_imagef(input, sampler, (int2)(x, y+i));
        acum+= p.x* mask[i+half_size];
    }
    write_imagef(output, (int2)(x, y), (float4)(acum,0,0,1));

}


kernel void blurx_fast( read_only image2d_t input, constant float * mask, private int mask_size, constant float * offsets, write_only image2d_t output){
    int x = get_global_id(0);
    int y = get_global_id(1);

    float acum=0.0;
    acum=(read_imagef(input, sampler, (int2)(x, y))).x * mask[0];
    for (int i=1; i<mask_size; i++) {
        acum += (read_imagef(input, sampler_linear, (int2)(x, y) + (int2)(offsets[i], 0) )).x * mask[i];
        acum += (read_imagef(input, sampler_linear, (int2)(x, y) - (int2)(offsets[i], 0) )).x * mask[i];
    }

    write_imagef(output, (int2)(x, y), (float4)(acum,0,0,1));

    // //attempt 2 with local mem
    // __local float local_mem[2048];
    // //read into local mem
    // int x_local=get_local_id(0);
    // int y_local=get_local_id(1);
    // int idx_local = (y_local * 32) + x_local; // Indexes
    // float4 p_local = read_imagef(input, sampler, (int2)(x, y));
    // local_mem[idx_local]=p_local.x;
    // barrier(CLK_LOCAL_MEM_FENCE);

    // // //attempt 2 with local mem
    // __local float local_mem[2048];
    // //read into local mem
    // int x_local=get_local_id(0);
    // int y_local=get_local_id(1);
    // int idx_local = (y_local * 32) + x_local; // Indexes
    // float4 p_local = read_imagef(input, sampler, (int2)(x, y));
    // local_mem[idx_local]=p_local.x;
    // barrier(CLK_LOCAL_MEM_FENCE);
    //
    // float acum=0.0;
    // acum=(read_imagef(input, sampler, (int2)(x, y))).x * mask[0];
    // for (int i=1; i<mask_size; i++) {
    //     int l = (y_local * 32) + x_local + offsets[i]; // Indexes
    //     float p = local_mem[l];
    //
    //     acum += p * mask[i];
    //     acum += p * mask[i];
    // }
    //
    // write_imagef(output, (int2)(x, y), (float4)(acum,0,0,1));

}


kernel void blury_fast( read_only image2d_t input, constant float * mask, private int mask_size, constant float * offsets, write_only image2d_t output){
    int x = get_global_id(0);
    int y = get_global_id(1);

    float acum=0.0;
    acum=(read_imagef(input, sampler, (int2)(x, y))).x * mask[0];
    for (int i=1; i<mask_size; i++) {
        acum += (read_imagef(input, sampler_linear, (int2)(x, y) + (int2)(0, offsets[i]) )).x * mask[i];
        acum += (read_imagef(input, sampler_linear, (int2)(x, y) - (int2)(0, offsets[i]) )).x * mask[i];
    }

    write_imagef(output, (int2)(x, y), (float4)(acum,0,0,1));

}
