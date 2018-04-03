__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;;

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

kernel void gaussian_blur_fast( read_only image2d_t input_image, constant float * mask, private int mask_size, write_only image2d_t output_image){

   //  enqueue_kernel(get_default_queue(), ndrange, ^{my_func_A(a, b, c);});
   //
   //  const int2 coords = {get_global_id(0), get_global_id(1)};
   //
   //  // Collect neighbor values and multiply with Gaussian
   // float sum = 0.0f;
   // for(int a = -mask_size; a < mask_size+1; a++) {
   //     for(int b = -mask_size; b < mask_size+1; b++) {
   //         sum += mask[a+mask_size+(b+mask_size)*(mask_size*2+1)]
   //            *read_imagef(input_image, sampler, coords + (int2)(a,b)).x;
   //     }
   // }
   //
   // float4 color_out=(float4)(sum,sum,sum,sum);
   // write_imagef(output_image, coords, color_out);
   //
   //  // float4 value = read_imagef(input_image, sampler, coords);
   //  // value[2]=0.0;
   //  // write_imagef(output_image, coords, value);
}
