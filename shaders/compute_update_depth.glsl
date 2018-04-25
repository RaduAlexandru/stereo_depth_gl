#version 430

layout (local_size_x = 256) in;

#define M_PI 3.1415926535897932384626433832795

const int MAX_RES_PER_POINT=16;
// const float setting_huberTH = 9; // Huber Threshold
// const float seed_convergence_sigma2_thresh=200;      //!< threshold on depth uncertainty for convergence.
// const float settings_Eta = 5;

//https://stackoverflow.com/a/34259806
const int STATUS_GOOD    = 0x00000001;
const int STATUS_OOB       = 0x00000002;
const int STATUS_OUTLIER  = 0x00000004;
const int STATUS_SKIPPED  = 0x00000006;
const int STATUS_BADCONDITION  = 0x00000008;
const int STATUS_DELETED  = 0x00000010;
const int STATUS_UNINITIALIZED  = 0x00000012;

struct Point{
    int idx_host_frame; //idx in the array of frames of the frame which "hosts" this inmature points
    float u,v; //position in host frame of the point
    float a;                     //!< a of Beta distribution: When high, probability of inlier is large.
    float b;                     //!< b of Beta distribution: When high, probability of outlier is large.
    float mu;                    //!< Mean of normal distribution.
    float z_range;               //!< Max range of the possible depth.
    float sigma2;                //!< Variance of normal distribution.
    float idepth_min;
    float idepth_max;
    float energyTH;
    float quality;
    //-----------------up until here we have 48 bytes so it's padded correctly to 16 bytes

    vec4 f; // heading range = Ki * (u,v,1) //make it float 4 becuse float 3 gets padded to 4 either way
    int lastTraceStatus;
    int converged;
    int is_outlier;
    int pad_1;
    //
    float color[MAX_RES_PER_POINT]; 		// colors in host frame
    float weights[MAX_RES_PER_POINT]; 		// host-weights for respective residuals.
    vec2 colorD[MAX_RES_PER_POINT];  //gradient in x and y at the pixel of the pattern normalized by the sqrt
    vec2 colorGrad[MAX_RES_PER_POINT]; //just the raw gradient in x and y at the pixel offset of the pattern

    float ncc_sum_templ;
    float ncc_const_templ;
    float pad_2;
    float pad_3;

    //Stuff that may be to be removed
    mat2 gradH;
    vec2 kp_GT;
    float pad_4;
    float pad_5;

    //for denoising (indexes iinto the array of points of each of the 8 neighbours)
    int left;
    int right;
    int above;
    int below;
    int left_upper;
    int right_upper;
    int left_lower;
    int right_lower;

    //some other things for denoising
    float g;
    float mu_denoised;
    float mu_head;
    float pad_6;
    vec2 p;
    // glm::vec2 p;
    float pad_7;
    float pad_8;


    //debug stuff
    float gradient_hessian_det;
    float gt_depth;
    int last_visible_frame;

    float debug; //serves as both debug and padding to 16 bytes
    // float padding_1; //to gt the struc to be aligned to 16 bytes

    float debug2[16];

};



//you change it to  layout (binding = 0, std430) or layout (binding = 0, std140) in case stuff breaks
layout (binding = 0, std430) coherent buffer array_points_block{
    Point p[];
};

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


uniform vec2 frame_size; //x, y
uniform mat4 tf_cur_host;
uniform mat4 tf_host_cur;
uniform mat3 K;
uniform mat3 KRKi_cr;
uniform vec3 Kt_cr;
uniform vec2 affine_cr;
uniform float px_error_angle;
uniform vec2 pattern_rot_offsets[MAX_RES_PER_POINT];
uniform int pattern_rot_nr_points;


uniform sampler2D gray_img_sampler;

//https://www.boost.org/doc/libs/1_47_0/libs/math/doc/sf_and_dist/html/math_toolkit/dist/dist_ref/dists/normal_dist.html
float gaus_pdf(float mean, float sd, float x){
    return exp(- (x-mean)*(x-mean)/(2*sd)*(2*sd)  )  / (sd*sqrt(2*M_PI));
}

vec4 hqfilter(sampler2D samp, vec2 tc){
    // Get the size of the texture we'll be sampling from
    vec2 texSize = textureSize(samp, 0);
    // Scale our input texture coordinates up, move to center of texel
    vec2 uvScaled = tc * texSize + 0.5;
    // Find integer and fractional parts of texture coordinate
    vec2 uvInt = floor(uvScaled);
    vec2 uvFrac = fract(uvScaled);
    // Replace fractional part of texture coordinate
    uvFrac = smoothstep(0.0, 1.0, uvFrac);
    // Reassemble texture coordinate, remove bias, and
    // scale back to 0.0 to 1.0 range
    vec2 uv = (uvInt + uvFrac - 0.5) / texSize;
    // Regular texture lookup
    return texture(samp, uv);
}

void main(void) {

    int id = int(gl_GlobalInvocationID.x);


    // p[id].debug=float(id);
    // p[id].mu=float(id);
    // return;


    //debug the parameter block
    p[id].debug2[0]=params.outlierTH;
    p[id].debug2[1]=params.overallEnergyTHWeight;
    p[id].debug2[2]=params.outlierTHSumComponent; 		// higher -> less strong gradient-based reweighting .
    p[id].debug2[3]=params.huberTH; // Huber Threshold
    p[id].debug2[4]=params.convergence_sigma2_thresh;      //!< threshold on depth uncertainty for convergence.
    p[id].debug2[5]=params.eta;


    //debug the parameter block
    // p[id].debug2[0]=outlierTH;
    // p[id].debug2[1]=overallEnergyTHWeight;
    // p[id].debug2[2]=outlierTHSumComponent; 		// higher -> less strong gradient-based reweighting .
    // p[id].debug2[3]=huberTH; // Huber Threshold
    // p[id].debug2[4]=convergence_sigma2_thresh;      //!< threshold on depth uncertainty for convergence.
    // p[id].debug2[5]=eta;



    // // check if point is visible in the current image
    const vec3 p_backproj_xyz= p[id].f.xyz * 1.0f/ p[id].mu;
    const vec4 p_backproj_xyzw=vec4(p_backproj_xyz.x,p_backproj_xyz.y,p_backproj_xyz.z,1.0);
    const vec4 xyz_f_xyzw = tf_cur_host*  p_backproj_xyzw ;
    const vec3 xyz_f=xyz_f_xyzw.xyz/xyz_f_xyzw.w;
    if(xyz_f.z < 0.0)  {
        return; // TODO in gl this is a return
    }


    const vec3 kp_c = K * xyz_f;
    const vec2 kp_c_h=kp_c.xy/kp_c.z;
    if ( kp_c_h.x < 0 || kp_c_h.x >= frame_size.x || kp_c_h.y < 0 || kp_c_h.y >= frame_size.y ) {
        return; // TODO in gl this is a return
    }


    //point is visible
    // point.last_visible_frame=frames[i].frame_id;

    //update inverse depth coordinates for min and max
    p[id].idepth_min = p[id].mu + sqrt(p[id].sigma2);
    p[id].idepth_max = max(p[id].mu - sqrt(p[id].sigma2), 0.00000001f);
    // memoryBarrier();
    // barrier();
    // memoryBarrier();

    if(params.search_epi_method==0){
        //search epiline-----------------------------------------------------------------------
       // search_epiline_ncc (point, frame, KRKi_cr, Kt_cr );
        // search_epiline_bca (point, frames[i], KRKi_cr, Kt_cr, affine_cr);
        float idepth_mean = (p[id].idepth_min + p[id].idepth_max)*0.5;
        vec3 pr = KRKi_cr * vec3(p[id].u,p[id].v, 1);
        vec3 ptpMean = pr + Kt_cr*idepth_mean;
        vec3 ptpMin = pr + Kt_cr*p[id].idepth_min;
        vec3 ptpMax = pr + Kt_cr*p[id].idepth_max;
        vec2 uvMean = ptpMean.xy/ptpMean.z;
        vec2 uvMin = ptpMin.xy/ptpMin.z;
        vec2 uvMax = ptpMax.xy/ptpMax.z;



        vec2 epi_line = uvMax - uvMin;
        float norm_epi = max(1e-5f,length(epi_line));
        vec2 epi_dir = epi_line / norm_epi;
        const float  half_length = 0.5f * norm_epi;

        vec2 bestKp=vec2(-1.0,-1.0);
        float bestEnergy = 1e10;


        for(float l = -half_length; l <= half_length; l += 0.7f)
        {
            float energy = 0;
            vec2 kp = uvMean + l*epi_dir;

            if( ( kp.x >= (frame_size.x-10) )  || ( kp.y >= (frame_size.y-10) ) || ( kp.x < 10 ) || ( kp.y < 10) ){
                continue;
            }

            for(int idx=0;idx<pattern_rot_nr_points; ++idx){
                //float hitColor = getInterpolatedElement31(frame->dI, (float)(kp(0)+rotatetPattern[idx][0]), (float)(kp(1)+rotatetPattern[idx][1]), wG[0]);
                vec2 offset=pattern_rot_offsets[idx];
                // float hit_color=texture(gray_img_sampler, vec2( (kp.x + offset.x+0.5)/640.0, (kp.y + offset.y+0.5)/480.0)).x;
                // float hit_color=texelFetch(gray_img_sampler, ivec2( (kp.x + offset.x), (kp.y + offset.y)), 0).x;
                // float hit_color=texture_interpolate(frames[i].gray, kp.x+offset.x, kp.y+offset.y , InterpolationType::LINEAR);
                // if(!std::isfinite(hit_color)) {energy-=1e5; continue;}

                //for the case when the image is padded
                // float hit_color=texture(gray_img_sampler, vec2( (kp.x + offset.x)/1024, ( 1024-480+  kp.y + offset.y)/1024)).x;

                //high qualty filter from openglsuperbible
                float hit_color=hqfilter(gray_img_sampler, vec2( (kp.x + offset.x+0.5)/640.0, (kp.y + offset.y+0.5)/480.0)).x;

                const float residual = hit_color - (affine_cr.x * p[id].color[idx] + affine_cr.y);

                float hw = abs(residual) < params.huberTH ? 1 : params.huberTH / abs(residual);
                energy += hw *residual*residual*(2-hw);
            }
            if ( energy < bestEnergy )
            {
                bestKp = kp; bestEnergy = energy;
            }
        }


        if ( bestEnergy > p[id].energyTH * 1.2f ) {
            p[id].lastTraceStatus = STATUS_OUTLIER;
        }
        else
        {
            // vec2 epi_dir_inv=vec2(epi_dir.y,-epi_dir.x);
            // float a = epi_dir * p[id].gradH * epi_dir;
            // float b = epi_dir_inv * point.gradH * epi_dir_inv;
            // float errorInPixel = 0.2f + 0.2f * (a+b) / a; // WO kommt das her? Scheint nicht zu NGF zu passen !
            float errorInPixel=0.0f;

            if( epi_dir.x*epi_dir.x>epi_dir.y*epi_dir.y )
            {
                p[id].idepth_min = (pr.z*(bestKp.x-errorInPixel*epi_dir.x) - pr.x) / (Kt_cr.x - Kt_cr.z*(bestKp.x-errorInPixel*epi_dir.x));
                p[id].idepth_max = (pr.z*(bestKp.x+errorInPixel*epi_dir.x) - pr.x) / (Kt_cr.x - Kt_cr.z*(bestKp.x+errorInPixel*epi_dir.x));
            }
            else
            {
                p[id].idepth_min = (pr.z*(bestKp.y-errorInPixel*epi_dir.y) - pr.y) / (Kt_cr.y - Kt_cr.z*(bestKp.y-errorInPixel*epi_dir.y));
                p[id].idepth_max = (pr.z*(bestKp.y+errorInPixel*epi_dir.y) - pr.y) / (Kt_cr.y - Kt_cr.z*(bestKp.y+errorInPixel*epi_dir.y));
            }
            // memoryBarrier();
            // barrier();
            // memoryBarrier();
            if(p[id].idepth_min > p[id].idepth_max) {
                // std::swap<float>(point.idepth_min, point.idepth_max);
                float tmp=p[id].idepth_min;
                p[id].idepth_min=p[id].idepth_max;
                p[id].idepth_max=tmp;
            }
            p[id].lastTraceStatus = STATUS_GOOD;
            // memoryBarrier();
            // barrier();
            // memoryBarrier();
        }
        // memoryBarrier();
        // barrier();
        // memoryBarrier();
    }else if(params.search_epi_method==1){
         //search epiline ngf-----------------------------------------------------------------------
        // search_epiline_ncc (point, frame, KRKi_cr, Kt_cr );
         // search_epiline_bca (point, frames[i], KRKi_cr, Kt_cr, affine_cr);
         float idepth_mean = (p[id].idepth_min + p[id].idepth_max)*0.5;
         vec3 pr = KRKi_cr * vec3(p[id].u,p[id].v, 1);
         vec3 ptpMean = pr + Kt_cr*idepth_mean;
         vec3 ptpMin = pr + Kt_cr*p[id].idepth_min;
         vec3 ptpMax = pr + Kt_cr*p[id].idepth_max;
         vec2 uvMean = ptpMean.xy/ptpMean.z;
         vec2 uvMin = ptpMin.xy/ptpMin.z;
         vec2 uvMax = ptpMax.xy/ptpMax.z;



         vec2 epi_line = uvMax - uvMin;
         float norm_epi = max(1e-5f,length(epi_line));
         vec2 epi_dir = epi_line / norm_epi;
         const float  half_length = 0.5f * norm_epi;

         vec2 bestKp=vec2(-1.0,-1.0);
         float bestEnergy = 1e10;


         for(float l = -half_length; l <= half_length; l += 0.7f)
         {
             float energy = 0;
             vec2 kp = uvMean + l*epi_dir;

             if( ( kp.x >= (frame_size.x-10) )  || ( kp.y >= (frame_size.y-10) ) || ( kp.x < 10 ) || ( kp.y < 10) ){
                 continue;
             }

             for(int idx=0;idx<pattern_rot_nr_points; ++idx){

                 vec2 offset=pattern_rot_offsets[idx];
                 // float hit_color=texture(gray_img_sampler, vec2( (kp.x + offset.x+0.5)/640.0, (kp.y + offset.y+0.5)/480.0)).x;
                 //high qualty filter from openglsuperbible
                 // float hit_color=hqfilter(gray_img_sampler, vec2( (kp.x + offset.x+0.5)/640.0, (kp.y + offset.y+0.5)/480.0)).x;

                 // vec3 hit_color=texture(gray_img_sampler, vec2( (kp.x + offset.x+0.5)/640.0, (kp.y + offset.y+0.5)/480.0)).xyz;
                 vec3 hit_color=hqfilter(gray_img_sampler, vec2( (kp.x + offset.x+0.5)/640.0, (kp.y + offset.y+0.5)/480.0)).xyz;
                 vec2 hitD=hit_color.yz;//gradient in xy
                 hitD = hitD/sqrt(pow(length(hitD),2)+params.eta);

                 const float nn =dot(hitD, p[id].colorD[idx]);
                 const float residual = max(0.0f,min(1.0f,nn < 0 ? 1.f : 1-nn ));// uni modal ngf

                 // float hw = abs(residual) < params.huberTH ? 1 : params.huberTH / abs(residual);
                 // energy += hw *residual*residual*(2-hw);


                 energy += residual;
             }
             if ( energy < bestEnergy )
             {
                 bestKp = kp; bestEnergy = energy;
             }
         }


         if ( bestEnergy > p[id].energyTH * 1.2f ) {
             p[id].lastTraceStatus = STATUS_OUTLIER;
         }
         else
         {
             // vec2 epi_dir_inv=vec2(epi_dir.y,-epi_dir.x);
             // float a = epi_dir * p[id].gradH * epi_dir;
             // float b = epi_dir_inv * point.gradH * epi_dir_inv;
             // float errorInPixel = 0.2f + 0.2f * (a+b) / a; // WO kommt das her? Scheint nicht zu NGF zu passen !
             float errorInPixel=0.0f;

             if( epi_dir.x*epi_dir.x>epi_dir.y*epi_dir.y )
             {
                 p[id].idepth_min = (pr.z*(bestKp.x-errorInPixel*epi_dir.x) - pr.x) / (Kt_cr.x - Kt_cr.z*(bestKp.x-errorInPixel*epi_dir.x));
                 p[id].idepth_max = (pr.z*(bestKp.x+errorInPixel*epi_dir.x) - pr.x) / (Kt_cr.x - Kt_cr.z*(bestKp.x+errorInPixel*epi_dir.x));
             }
             else
             {
                 p[id].idepth_min = (pr.z*(bestKp.y-errorInPixel*epi_dir.y) - pr.y) / (Kt_cr.y - Kt_cr.z*(bestKp.y-errorInPixel*epi_dir.y));
                 p[id].idepth_max = (pr.z*(bestKp.y+errorInPixel*epi_dir.y) - pr.y) / (Kt_cr.y - Kt_cr.z*(bestKp.y+errorInPixel*epi_dir.y));
             }
             // memoryBarrier();
             // barrier();
             // memoryBarrier();
             if(p[id].idepth_min > p[id].idepth_max) {
                 // std::swap<float>(point.idepth_min, point.idepth_max);
                 float tmp=p[id].idepth_min;
                 p[id].idepth_min=p[id].idepth_max;
                 p[id].idepth_max=tmp;
             }
             p[id].lastTraceStatus = STATUS_GOOD;
             // memoryBarrier();
             // barrier();
             // memoryBarrier();
         }
         // memoryBarrier();
         // barrier();
         // memoryBarrier();
    }























    float idepth = -1;
    float z = 0;
    if( p[id].lastTraceStatus == STATUS_GOOD ) {
        idepth = max(1e-5f,.5*(p[id].idepth_min+p[id].idepth_max));
        z = 1.0f/idepth;
    }
    if ( p[id].lastTraceStatus == STATUS_OOB  || p[id].lastTraceStatus == STATUS_SKIPPED ){
        return;
    }
    if ( idepth<0.00000001 || idepth>99999999 || p[id].lastTraceStatus == STATUS_OUTLIER || p[id].lastTraceStatus == STATUS_BADCONDITION ) {
        p[id].b++; // increase outlier probability when no match was found
        return;
    }
    // memoryBarrier();
    // barrier();
    // memoryBarrier();


    // update_idepth(point,tf_host_cur, z, px_error_angle);

    // compute tau----------------------------------------------------------------------------
    // double tau = compute_tau(tf_host_cur, point.f, z, px_error_angle);
    vec3 t=  vec3(tf_host_cur[0][3], tf_host_cur[1][3], tf_host_cur[2][3]);
    // Eigen::Vector3f t(tf_host_cur.translation());
    vec3 a = p[id].f.xyz*z-t;
    float t_norm = length(t);
    float a_norm = length(a);
    float alpha = acos(dot(p[id].f.xyz,t)/t_norm); // dot product
    float beta = acos(dot(a,-t)/(t_norm*a_norm)); // dot product
    float beta_plus = beta + px_error_angle;
    float gamma_plus = M_PI-alpha-beta_plus; // triangle angles sum to PI
    float z_plus = t_norm*sin(beta_plus)/sin(gamma_plus); // law of sines
    float tau= (z_plus - z); // tau
    float tau_inverse = 0.5 * (1.0/max(0.0000001, z-tau) - 1.0/(z+tau));

    // update the estimate--------------------------------------------------
    float x=1.0/z;
    float tau2=tau_inverse*tau_inverse;
    // updateSeed(point, 1.0/z, tau_inverse*tau_inverse);
    float norm_scale = sqrt(p[id].sigma2 + tau2);
    float s2 = 1./(1./p[id].sigma2 + 1./tau2);
    float m = s2*(p[id].mu/p[id].sigma2 + x/tau2);
    float C1 = p[id].a/(p[id].a+p[id].b) * gaus_pdf(p[id].mu, norm_scale, x);
    float C2 = p[id].b/(p[id].a+p[id].b) * 1./p[id].z_range;
    float normalization_constant = C1 + C2;
    C1 /= normalization_constant;
    C2 /= normalization_constant;
    float f = C1*(p[id].a+1.)/(p[id].a+p[id].b+1.) + C2*p[id].a/(p[id].a+p[id].b+1.);
    float e = C1*(p[id].a+1.)*(p[id].a+2.)/((p[id].a+p[id].b+1.)*(p[id].a+p[id].b+2.))
              + C2*p[id].a*(p[id].a+1.0f)/((p[id].a+p[id].b+1.0f)*(p[id].a+p[id].b+2.0f));
    // update parameters
    float mu_new = C1*m+C2*p[id].mu;
    p[id].sigma2 = C1*(s2 + m*m) + C2*(p[id].sigma2 + p[id].mu*p[id].mu) - mu_new*mu_new;
    p[id].mu = mu_new;
    p[id].a = (e-f)/(f-e/f);
    // memoryBarrier();
    // barrier();
    // memoryBarrier();
    p[id].b = p[id].a*(1.0f-f)/f;
    // memoryBarrier();
    // barrier(); //TODO add again the barrier
    // memoryBarrier();

    // // not implemented in opengl
    const float eta_inlier = .6f;
    const float eta_outlier = .05f;
    if( ((p[id].a / (p[id].a + p[id].b)) > eta_inlier) && (sqrt(p[id].sigma2) < p[id].z_range/params.convergence_sigma2_thresh)) {
        p[id].is_outlier = 0; // The seed converged
    }else if((p[id].a-1) / (p[id].a + p[id].b - 2) < eta_outlier){ // The seed failed to converge
        p[id].is_outlier = 1;
        // it->reinit();
        //TODO do a better reinit inside a point class
        p[id].a = 10;
        p[id].b = 10;
        p[id].mu = (1.0/4.0);
        p[id].z_range = (1.0/0.1);
        p[id].sigma2 = (p[id].z_range*p[id].z_range/36);
    }
    // if the seed has converged, we initialize a new candidate point and remove the seed
    if(sqrt(p[id].sigma2) < p[id].z_range/params.convergence_sigma2_thresh){
        p[id].converged = 1;
    }







}
