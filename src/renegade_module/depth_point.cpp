/**
* This file is part of DSO.
*
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#include "depth_point.h"

//#define NGF 1

// dso for bullshit calibration
int wG[1] = {640};
int hG[1] = {480};

int wM3G = wG[0]-3;
int hM3G = hG[0]-3;

int staticPattern[10][40][2] = {
    {{0,0}, 	  {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},	// .
     {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
     {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
     {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

    {{0,-1},	  {-1,0},	   {0,0},	    {1,0},	     {0,1}, 	  {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},	// +
     {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
     {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
     {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

    {{-1,-1},	  {1,1},	   {0,0},	    {-1,1},	     {1,-1}, 	  {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},	// x
     {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
     {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
     {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

    {{-1,-1},	  {-1,0},	   {-1,1},		{-1,0},		 {0,0},		  {0,1},	   {1,-1},		{1,0},		 {1,1},       {-100,-100},	// full-tight
     {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
     {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
     {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

    {{0,-2},	  {-1,-1},	   {1,-1},		{-2,0},		 {0,0},		  {2,0},	   {-1,1},		{1,1},		 {0,2},       {-100,-100},	// full-spread-9
     {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
     {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
     {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

    {{0,-2},	  {-1,-1},	   {1,-1},		{-2,0},		 {0,0},		  {2,0},	   {-1,1},		{1,1},		 {0,2},       {-2,-2},   // full-spread-13
     {-2,2},      {2,-2},      {2,2},       {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
     {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
     {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

    {{-2,-2},     {-2,-1}, {-2,-0}, {-2,1}, {-2,2}, {-1,-2}, {-1,-1}, {-1,-0}, {-1,1}, {-1,2}, 										// full-25
     {-0,-2},     {-0,-1}, {-0,-0}, {-0,1}, {-0,2}, {+1,-2}, {+1,-1}, {+1,-0}, {+1,1}, {+1,2},
     {+2,-2}, 	  {+2,-1}, {+2,-0}, {+2,1}, {+2,2}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
     {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

    {{0,-2},	  {-1,-1},	   {1,-1},		{-2,0},		 {0,0},		  {2,0},	   {-1,1},		{1,1},		 {0,2},       {-2,-2},   // full-spread-21
     {-2,2},      {2,-2},      {2,2},       {-3,-1},     {-3,1},      {3,-1}, 	   {3,1},       {1,-3},      {-1,-3},     {1,3},
     {-1,3},      {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
     {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

    {{0,-2},	  {-1,-1},	   {1,-1},		{-2,0},		 {0,0},		  {2,0},	   {-1,1},		{0,2},		 {-100,-100}, {-100,-100},	// 8 for SSE efficiency
     {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
     {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
     {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

    {{-4,-4},     {-4,-2}, {-4,-0}, {-4,2}, {-4,4}, {-2,-4}, {-2,-2}, {-2,-0}, {-2,2}, {-2,4}, 										// full-45-SPREAD
     {-0,-4},     {-0,-2}, {-0,-0}, {-0,2}, {-0,4}, {+2,-4}, {+2,-2}, {+2,-0}, {+2,2}, {+2,4},
     {+4,-4}, 	  {+4,-2}, {+4,-0}, {+4,2}, {+4,4}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200},
     {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}},
};

#define patternNum 8
#define patternP staticPattern[8]
#define patternPngf staticPattern[8]

float getInterpolatedElement31(const Eigen::Vector3f* const mat, const float x, const float y, const int width)
{
    int ix = (int)x;
    int iy = (int)y;
    float dx = x - ix;
    float dy = y - iy;
    float dxdy = dx*dy;
    const Eigen::Vector3f* bp = mat+ix+iy*width;


    return dxdy * (*(const Eigen::Vector3f*)(bp+1+width))[0]
            + (dy-dxdy) * (*(const Eigen::Vector3f*)(bp+width))[0]
            + (dx-dxdy) * (*(const Eigen::Vector3f*)(bp+1))[0]
            + (1-dx-dy+dxdy) * (*(const Eigen::Vector3f*)(bp))[0];
}

Eigen::Vector3f getInterpolatedElement33(const Eigen::Vector3f* const mat, const float x, const float y, const int width)
{
    int ix = (int)x;
    int iy = (int)y;
    float dx = x - ix;
    float dy = y - iy;
    float dxdy = dx*dy;
    const Eigen::Vector3f* bp = mat +ix+iy*width;

    if ( ( ix+iy*width)+1+width >= 640*480 || (ix+iy*width) < 0 )
        std::cerr << "pos: " << (( ix+iy*width)+1+width)<< " max: "<< (640*480)  << " di=" << (ix+iy*width) << " x="<<x << " y="<<y << std::endl;
    if ( ix >= 639 || ix < 0 )
        std::cerr << "ix="<<ix << " "<< x << " " << ( ix+iy*width) << std::endl;
    if ( iy >= 479 || iy < 0 )
        std::cerr << "ix="<<ix << " " << y << " " << ( ix+iy*width) << std::endl;

    return dxdy * *(const Eigen::Vector3f*)(bp+1+width)
            + (dy-dxdy) * *(const Eigen::Vector3f*)(bp+width)
            + (dx-dxdy) * *(const Eigen::Vector3f*)(bp+1)
            + (1-dx-dy+dxdy) * *(const Eigen::Vector3f*)(bp);
}
Eigen::Vector3f getInterpolatedElement33BiLin(const Eigen::Vector3f* const mat, const float x, const float y, const int width)
{
    int ix = (int)x;
    int iy = (int)y;
    const Eigen::Vector3f* bp = mat +ix+iy*width;

    float tl = (*(bp))[0];
    float tr = (*(bp+1))[0];
    float bl = (*(bp+width))[0];
    float br = (*(bp+width+1))[0];

    float dx = x - ix;
    float dy = y - iy;
    float topInt = dx * tr + (1-dx) * tl;
    float botInt = dx * br + (1-dx) * bl;
    float leftInt = dy * bl + (1-dy) * tl;
    float rightInt = dy * br + (1-dy) * tr;

    return Eigen::Vector3f( dx * rightInt + (1-dx) * leftInt, rightInt-leftInt, botInt-topInt);
    //return Eigen::Vector3f( dx * rightInt + (1-dx) * leftInt, 0.5f*(rightInt-leftInt), 0.5f*(botInt-topInt)); // but this should be correct !
}
Eigen::Vector3f getInterpolatedElement33BiLin2(const Eigen::Vector3f* const mat, const float x, const float y, const int width)
{
    int ix = (int)x;
    int iy = (int)y;
    const Eigen::Vector3f* bp = mat +ix+iy*width;

    float tl = (*(bp))[0];
    float tr = (*(bp+1))[0];
    float bl = (*(bp+width))[0];
    float br = (*(bp+width+1))[0];

    float dx = x - ix;
    float dy = y - iy;
    float topInt = dx * tr + (1-dx) * tl;
    float botInt = dx * br + (1-dx) * bl;
    float leftInt = dy * bl + (1-dy) * tl;
    float rightInt = dy * br + (1-dy) * tr;

    //return Eigen::Vector3f( dx * rightInt + (1-dx) * leftInt, rightInt-leftInt, botInt-topInt);
    return Eigen::Vector3f( dx * rightInt + (1-dx) * leftInt, 0.5f*(rightInt-leftInt), 0.5f*(botInt-topInt)); // but this should be correct !
}

float getInterpolatedElement11Cub(const float* const p, const float x)	// for x=0, this returns p[1].
{
        return p[1] + 0.5f * x*(p[2] - p[0] + x*(2.0f*p[0] - 5.0f*p[1] + 4.0f*p[2] - p[3] + x*(3.0f*(p[1] - p[2]) + p[3] - p[0])));
}

Eigen::Vector2f getInterpolatedElement12Cub(const float* const p, const float x)	// for x=0, this returns p[1].
{
        float c1 = 0.5f * (p[2] - p[0]);
        float c2 = p[0]-2.5f*p[1]+2*p[2]-0.5f*p[3];
        float c3 = 0.5f*(3.0f*(p[1]-p[2])+p[3]-p[0]);
        float xx = x*x;
        float xxx = xx*x;
        return Eigen::Vector2f(p[1] + x*c1 + xx*c2 + xxx*c3, c1 + x*2.0f*c2 + xx*3.0f*c3);
}

Eigen::Vector3f getInterpolatedElement12Cub2(const float* const p, const float x )	// for x=0, this returns p[1].
{
        float c1 = 0.5f * (p[2] - p[0]);
        float c2 = p[0]-2.5f*p[1]+2*p[2]-0.5f*p[3];
        float c3 = 0.5f*(3.0f*(p[1]-p[2])+p[3]-p[0]);
        float xx = x*x;
        float xxx = xx*x;
        return Eigen::Vector3f(p[1] + x*c1 + xx*c2 + xxx*c3, c1 + x*2.0f*c2 + xx*3.0f*c3, 2.0f*c2 + x*6.0f*c3);
}

Eigen::Vector2f getInterpolatedElement32Cub(const Eigen::Vector3f* const p, const float x)	// for x=0, this returns p[1].
{
        float c1 = 0.5f * (p[2][0] - p[0][0]);
        float c2 = p[0][0]-2.5f*p[1][0]+2*p[2][0]-0.5f*p[3][0];
        float c3 = 0.5f*(3.0f*(p[1][0]-p[2][0])+p[3][0]-p[0][0]);
        float xx = x*x;
        float xxx = xx*x;
        return Eigen::Vector2f(p[1][0] + x*c1 + xx*c2 + xxx*c3, c1 + x*2.0f*c2 + xx*3.0f*c3);
}

Eigen::Vector3f getInterpolatedElement32Cub2(const Eigen::Vector3f* const p, const float x)	// for x=0, this returns p[1].
{
        float c1 = 0.5f * (p[2][0] - p[0][0]);
        float c2 = p[0][0]-2.5f*p[1][0]+2*p[2][0]-0.5f*p[3][0];
        float c3 = 0.5f*(3.0f*(p[1][0]-p[2][0])+p[3][0]-p[0][0]);
        float xx = x*x;
        float xxx = xx*x;
        return Eigen::Vector3f(p[1][0] + x*c1 + xx*c2 + xxx*c3, c1 + x*2.0f*c2 + xx*3.0f*c3, 2.0f*c2 + x*6.0f*c3);
}

Eigen::Vector3f getInterpolatedElement33BiCub(const Eigen::Vector3f* const mat, const float x, const float y, const int width)
{
        int ix = (int)x;
        int iy = (int)y;
        float dx = x - ix;
        const Eigen::Vector3f* bp = mat +ix+iy*width; // mat[iy][ix]

        float val[4];
        float grad[4];
        Eigen::Vector2f v = getInterpolatedElement32Cub(bp-width-1, dx); // mat[iy-1][ix-1] bis mat[iy-1][ix+2]
        val[0] = v[0]; grad[0] = v[1];

        v = getInterpolatedElement32Cub(bp-1, dx); // mat[iy][ix-1] bis mat[iy][ix+2]
        val[1] = v[0]; grad[1] = v[1];

        v = getInterpolatedElement32Cub(bp+width-1, dx);// mat[iy+1][ix-1] bis mat[iy+1][ix+2]
        val[2] = v[0]; grad[2] = v[1];

        v = getInterpolatedElement32Cub(bp+2*width-1, dx);// mat[iy+2][ix-1] bis mat[iy+2][ix+2]
        val[3] = v[0]; grad[3] = v[1];

        float dy = y - iy;
        v = getInterpolatedElement12Cub(val, dy);

        return Eigen::Vector3f(v[0], getInterpolatedElement11Cub(grad, dy), v[1]); // v[x,y], v[x,y]/dx, v[x,y]/dy
}

Eigen::Vector3f getInterpolatedElement33BiCub2(const Eigen::Vector3f* const mat, const float x, const float y, const int width, Eigen::Vector2f & d2f)
{
        int ix = (int)x;
        int iy = (int)y;
        float dx = x - ix;
        const Eigen::Vector3f* bp = mat +ix+iy*width; // mat[iy][ix]

        float val[4];
        float grad[4];
        Eigen::Vector2f v = getInterpolatedElement32Cub(bp-width-1, dx); // mat[iy-1][ix-1] bis mat[iy-1][ix+2]
        val[0] = v[0]; grad[0] = v[1];

        v = getInterpolatedElement32Cub(bp-1, dx); // mat[iy][ix-1] bis mat[iy][ix+2]
        val[1] = v[0]; grad[1] = v[1];

        v = getInterpolatedElement32Cub(bp+width-1, dx);// mat[iy+1][ix-1] bis mat[iy+1][ix+2]
        val[2] = v[0]; grad[2] = v[1];

        v = getInterpolatedElement32Cub(bp+2*width-1, dx);// mat[iy+2][ix-1] bis mat[iy+2][ix+2]
        val[3] = v[0]; grad[3] = v[1];

        float dy = y - iy;
        Eigen::Vector3f v2 = getInterpolatedElement12Cub2( val, dy );
        d2f[1] = v2[2];
        v = getInterpolatedElement12Cub(grad, dy);
        d2f[0] = v[1];
        // dxy =
        return Eigen::Vector3f(v2[0], v[0], v2[1]); // v[x,y], v[x,y]/dx, v[x,y]/dy
}

Eigen::Vector3f getInterpolatedElement33BiCub3(const Eigen::Vector3f* const mat, const float x, const float y, const int width, Eigen::Matrix2f & d2f)
{
        int ix = (int)x;
        int iy = (int)y;
        float dx = x - ix;
        const Eigen::Vector3f* bp = mat +ix+iy*width; // mat[iy][ix]

        float val[4];
        float grad[4];
        float hess[4];
        Eigen::Vector3f v = getInterpolatedElement32Cub2(bp-width-1, dx); // mat[iy-1][ix-1] bis mat[iy-1][ix+2]
        val[0] = v[0]; grad[0] = v[1]; hess[0] = v[2];

        v = getInterpolatedElement32Cub2(bp-1, dx); // mat[iy][ix-1] bis mat[iy][ix+2]
        val[1] = v[0]; grad[1] = v[1]; hess[1] = v[2];

        v = getInterpolatedElement32Cub2(bp+width-1, dx);// mat[iy+1][ix-1] bis mat[iy+1][ix+2]
        val[2] = v[0]; grad[2] = v[1]; hess[2] = v[2];

        v = getInterpolatedElement32Cub2(bp+2*width-1, dx);// mat[iy+2][ix-1] bis mat[iy+2][ix+2]
        val[3] = v[0]; grad[3] = v[1]; hess[3] = v[2];

        float dy = y - iy;
        v = getInterpolatedElement12Cub2( val, dy );
        d2f(1,1) = v[2];

        Eigen::Vector2f v2 = getInterpolatedElement12Cub(grad, dy);
        d2f(0,0) = v2[1];
        // dxy -> TODO: correct computation
        v2 = getInterpolatedElement12Cub(hess, dy);
        d2f(1,0) = v2[1];
        d2f(0,1) = d2f(1,0);
        return Eigen::Vector3f(v[0], v2[0], v[1]); // v[x,y], v[x,y]/dx, v[x,y]/dy
}

void copyImage ( ImageDataPtr frame )
{
    if ( frame->dI != nullptr )
    {
        return;
        delete frame->dI;
        frame->dI = nullptr;
    }
    frame->dI = new Eigen::Vector3f[wG[0]*hG[0]];

    // make d0
    int w=wG[0];
    int h=hG[0];
    uchar * gray = frame->grayImages[0].ptr<uchar>();
    for(int i=0;i<w*h;i++)
    {
        frame->dI[i](0) = gray[i];
    }
    int wl = w;
    int hl = h;
    Eigen::Vector3f* dI_l = frame->dI;
    for(int idx=wl;idx < wl*(hl-1);++idx)
    {
        float dx = 0.5f*(dI_l[idx+1][0] - dI_l[idx-1][0]);
        float dy = 0.5f*(dI_l[idx+wl][0] - dI_l[idx-wl][0]);


        if(!std::isfinite(dx)) dx=0;
        if(!std::isfinite(dy)) dy=0;

        dI_l[idx][1] = dx;
        dI_l[idx][2] = dy;
    }
}

float setting_maxPixSearch = 0.027; // max length of the ep. line segment searched during immature point tracking. relative to image resolution.
float setting_minTraceQuality = 3;
int setting_minTraceTestRadius = 2;
int setting_GNItsOnPointActivation = 3;
float setting_trace_stepsize = 1.0;				// stepsize for initial discrete search.
int setting_trace_GNIterations = 3;				// max # GN iterations
float setting_trace_GNThreshold = 0.1;				// GN stop after this stepsize.
float setting_trace_extraSlackOnTH = 1.2;			// for energy-based outlier check, be slightly more relaxed by this factor.
float setting_trace_slackInterval = 1.5;			// if pixel-interval is smaller than this, leave it be.
float setting_trace_minImprovementFactor = 2;		// if pixel-interval is smaller than this, leave it be.

float setting_frameEnergyTHConstWeight = 0.5;
float setting_frameEnergyTHN = 0.7f;
float setting_frameEnergyTHFacMedian = 1.5;
float setting_overallEnergyTHWeight = 1;
float setting_coarseCutoffTH = 20;

float setting_outlierTHSumComponent = 50*50; 		// higher -> less strong gradient-based reweighting .

float setting_minIdepthH_act = 100;

float settings_Eta = 25;//1e-3;//100;//25;//1e-3; // for NGF
float setting_huber_ngf_TH = 3;
float setting_outlierTH_ngf = 0.9;
#ifdef NGF


float setting_outlierTH = 6; // not yet used.
#else
float setting_huberTH = 9; // Huber Threshold
float setting_outlierTH = 12*12;					// higher -> less strict
#endif

ImmaturePoint::ImmaturePoint(int u_, int v_, ImageDataPtr host_, float idepth_gt )
    : u(u_), v(v_), host(host_), idepth_min(0), idepth_max(std::numeric_limits<float>::signaling_NaN()), lastTraceStatus(IPS_UNINITIALIZED), idepth_GT ( idepth_gt )
{
    if ( host->dI == nullptr ){
        std::cout << "inmature point constructor copying the whole frame!" << '\n';
        copyImage ( host );
    }

    gradH.setZero();

    for(int idx=0;idx<patternNum;idx++)
    {
        int dx = patternP[idx][0];
        int dy = patternP[idx][1];
        //Vec3f ptc = getInterpolatedElement33BiLin(host->dI, u+dx, v+dy, wG[0]);
        Vec3f ptc = getInterpolatedElement33BiCub(host->dI, u+dx, v+dy, wG[0]);
        colorD[idx] = ptc.tail<2>();
        colorD[idx] /= sqrt(colorD[idx].squaredNorm()+settings_Eta);
        colorGrad[idx] = ptc.tail<2>();
    }
    int not_skipped = 0;
    for(int idx=0;idx<patternNum;idx++)
    {
        skipZero[idx] = colorD[idx].isApprox(Eigen::Vector2f::Zero(),1e-3);
        if ( ! skipZero[idx] ) ++not_skipped;
    }

    for(int idx=0;idx<patternNum;idx++)
    {
        int dx = patternP[idx][0];
        int dy = patternP[idx][1];
        //Vec3f ptc = getInterpolatedElement33BiLin(host->dI, u+dx, v+dy, wG[0]);
        Vec3f ptc = getInterpolatedElement33BiCub(host->dI, u+dx, v+dy, wG[0]);

        color[idx] = ptc[0];

        if(!std::isfinite(color[idx])) {energyTH=std::numeric_limits<float>::signaling_NaN(); return;}

        gradH += ptc.tail<2>()  * ptc.tail<2>().transpose();

        weights[idx] = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + ptc.tail<2>().squaredNorm()));
    }

    ncc_sum_templ    = 0.0f;
    float ncc_sum_templ_sq = 0.0f;
    for(int idx=0;idx<patternNum;++idx)
    {
        const float templ = color[idx];
        ncc_sum_templ += templ;
        ncc_sum_templ_sq += templ*templ;
    }
    ncc_const_templ = patternNum * ncc_sum_templ_sq - (double) ncc_sum_templ*ncc_sum_templ;

    energyTHngf = not_skipped * setting_outlierTH_ngf;
    energyTH = patternNum*setting_outlierTH;
    energyTH *= setting_overallEnergyTHWeight*setting_overallEnergyTHWeight;

    //idepth_GT=0;
    quality=10000;
}

ImmaturePoint::~ImmaturePoint() {}

bool ImmaturePoint::isContained ( const ImmaturePointPtr & otherPt ) const
{
    bool contained = false;
    // secondBest is within bests range
    if ( (idepth_min > otherPt->idepth_min && idepth_min < otherPt->idepth_max) || (idepth_max > otherPt->idepth_min && idepth_max < otherPt->idepth_max) )
        contained = true;
    // best is within secondBests range
    if ( (otherPt->idepth_min > idepth_min && otherPt->idepth_min < idepth_max) || (otherPt->idepth_max > idepth_min && otherPt->idepth_max < idepth_max) )
        contained = true;
    return contained;
}

void ImmaturePoint::rotatePattern( const Mat33f & hostToFrame_KRKi )
{
    const Mat22f Rplane = hostToFrame_KRKi.topLeftCorner<2,2>();
    for(int idx=0;idx<patternNum;++idx)
        rotatetPattern[idx] = Rplane * Vec2f(patternP[idx][0], patternP[idx][1]);
}

/*
 * returns
 * * OOB -> point is optimized and marginalized
 * * UPDATED -> point has been updated.
 * * SKIP -> point has not been updated.
 */
ImmaturePointPtr ImmaturePoint::traceOn(ImageDataPtr frame,const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt, const Vec2f& hostToFrame_affine, bool debugPrint)
{
    ImmaturePointPtr newPt ( nullptr );

    if(lastTraceStatus == ImmaturePointStatus::IPS_OOB || lastTraceStatus == ImmaturePointStatus::IPS_DELETED ) return newPt;//lastTraceStatus;

    ++numTraced;
//    if ( child )
//    {
//        if ( child->lastTraceStatus == ImmaturePointStatus::IPS_DELETED )
//            child.reset();
//        else
//        {
//            if ( numTraced > 3 )
//            {
//                bool anyoneBetter = false;
//                bool isChildBetter = false;
//                // compare against other hypothesis
//                bool contained = isContained ( child );
//                isChildBetter = quality < child->quality;
//                anyoneBetter = contained || isChildBetter || !isChildBetter;
//                if ( anyoneBetter )
//                {
//                    if ( isChildBetter )
//                        lastTraceStatus = IPS_DELETED;
//                    else
//                    {
//                        child->lastTraceStatus = ImmaturePointStatus::IPS_DELETED;
//                        child.reset();
//                    }
//                }
//            }
//        }
//    }

    if ( !frame )
    {
        lastTraceStatus = ImmaturePointStatus::IPS_SKIPPED;
        return newPt;
    }
    if ( frame->dI == nullptr )
        copyImage ( frame );

    debugPrint = false;//rand()%100==0;
    float maxPixSearch = (wG[0]+hG[0])*setting_maxPixSearch;

    //    if(debugPrint)
    //        printf("trace pt (%.1f %.1f) from frame %d to %d. Range %f -> %f. t %f %f %f!\n",
    //               u,v,
    //               host->shell->id, frame->shell->id,
    //               idepth_min, idepth_max,
    //               hostToFrame_Kt[0],hostToFrame_Kt[1],hostToFrame_Kt[2]);

    //	const float stepsize = 1.0;				// stepsize for initial discrete search.
    //	const int GNIterations = 3;				// max # GN iterations
    //	const float GNThreshold = 0.1;				// GN stop after this stepsize.
    //	const float extraSlackOnTH = 1.2;			// for energy-based outlier check, be slightly more relaxed by this factor.
    //	const float slackInterval = 0.8;			// if pixel-interval is smaller than this, leave it be.
    //	const float minImprovementFactor = 2;		// if pixel-interval is smaller than this, leave it be.
    // ============== project min and max. return if one of them is OOB ===================
    Vec3f pr = hostToFrame_KRKi * Vec3f(u,v, 1);
    Vec3f ptpMin = pr + hostToFrame_Kt*idepth_min;
    float uMin = ptpMin[0] / ptpMin[2];
    float vMin = ptpMin[1] / ptpMin[2];

    if ( idepth_GT > 0 )
    {
        Vec3f p = pr + hostToFrame_Kt*idepth_GT;
        kp_GT = p.hnormalized();
    }

    if(!(uMin > 4 && vMin > 4 && uMin < wG[0]-5 && vMin < hG[0]-5))
    {
        if(debugPrint) printf("OOB uMin %f %f - %f %f %f (id %f-%f)!\n",
                              u,v,uMin, vMin,  ptpMin[2], idepth_min, idepth_max);

        Vec3f ptpMax = pr + hostToFrame_Kt* ( std::isfinite(idepth_max) ? idepth_max : 0.01 );
        float uMax = ptpMax[0] / ptpMax[2];
        float vMax = ptpMax[1] / ptpMax[2];

        lastValidUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
        lastTraceUV = Vec2f(-1,-1);
        lastTracePixelInterval=0;
        lastTraceStatus = ImmaturePointStatus::IPS_OOB;
        return newPt;
    }

    float dist;
    float uMax;
    float vMax;
    Vec3f ptpMax;
    if(std::isfinite(idepth_max))
    {
        ptpMax = pr + hostToFrame_Kt*idepth_max;
        uMax = ptpMax[0] / ptpMax[2];
        vMax = ptpMax[1] / ptpMax[2];


        if(!(uMax > 4 && vMax > 4 && uMax < wG[0]-5 && vMax < hG[0]-5))
        {
            if(debugPrint) printf("OOB uMax  %f %f - %f %f!\n",u,v, uMax, vMax);
            lastValidUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
            lastTraceUV = Vec2f(-1,-1);
            lastTracePixelInterval=0;
            lastTraceStatus = ImmaturePointStatus::IPS_OOB;
            return newPt;
        }

        // ============== check their distance. everything below 2px is OK (-> skip). ===================
        dist = (uMin-uMax)*(uMin-uMax) + (vMin-vMax)*(vMin-vMax);
        dist = sqrtf(dist);
        if(dist < setting_trace_slackInterval)
        {
//            rotatePattern( hostToFrame_KRKi );
//            float dx = setting_trace_stepsize*(uMax-uMin);
//            float dy = setting_trace_stepsize*(vMax-vMin);
//            dx /= dist;
//            dy /= dist;
//            float bestU = (uMin + uMax)*.5;
//            float bestV = (vMin + vMax)*.5;
//            float bestEnergy = doRefinement ( frame, hostToFrame_affine, dx, dy, bestU, bestV );
//            checkResult ( hostToFrame_Kt, pr, 1, dx, dy, uMin, uMax, vMin, vMax, bestU, bestV, bestEnergy );
            if(debugPrint)
                printf("TOO CERTAIN ALREADY (dist %f)!\n", dist);
            lastValidUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
            lastTraceUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
            lastTracePixelInterval=dist;
            lastTraceStatus = ImmaturePointStatus::IPS_SKIPPED;
            return newPt;
        }
        assert(dist>0);
    }
    else
    {
        dist = maxPixSearch;

        // project to arbitrary depth to get direction.
        ptpMax = pr + hostToFrame_Kt*0.01;
        uMax = ptpMax[0] / ptpMax[2];
        vMax = ptpMax[1] / ptpMax[2];

        // direction.
        float dx = uMax-uMin;
        float dy = vMax-vMin;
        float d = 1.0f / sqrtf(dx*dx+dy*dy);

        // set to [setting_maxPixSearch].
        uMax = uMin + dist*dx*d;
        vMax = vMin + dist*dy*d;

        // may still be out!
        if(!(uMax > 4 && vMax > 4 && uMax < wG[0]-5 && vMax < hG[0]-5))
        {
            if(debugPrint) printf("OOB uMax-coarse %f %f %f!\n", uMax, vMax,  ptpMax[2]);
            lastTraceUV = Vec2f(-1,-1);
            lastValidUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
            lastTracePixelInterval=0;
            lastTraceStatus = ImmaturePointStatus::IPS_OOB;
            return newPt;
        }
        assert(dist>0);
    }


    // set OOB if scale change too big.
    if(!(idepth_min<0 || (ptpMin[2]>0.75 && ptpMin[2]<1.5)))
    {
        if(debugPrint) printf("OOB SCALE %f %f %f!\n", uMax, vMax,  ptpMin[2]);
        lastTraceUV = Vec2f(-1,-1);
        lastValidUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
        lastTracePixelInterval=0;
        lastTraceStatus = ImmaturePointStatus::IPS_OOB;
        return newPt;
    }

    // ============== compute error-bounds on result in pixel. if the new interval is not at least 1/2 of the old, SKIP ===================
    float dx = setting_trace_stepsize*(uMax-uMin);
    float dy = setting_trace_stepsize*(vMax-vMin);

    float a = (Vec2f(dx,dy).transpose() * gradH * Vec2f(dx,dy));
    float b = (Vec2f(dy,-dx).transpose() * gradH * Vec2f(dy,-dx));
    float errorInPixel = 0.2f + 0.2f * (a+b) / a; // WO kommt das her? Scheint nicht zu NGF zu passen !

    if(errorInPixel*setting_trace_minImprovementFactor > dist && std::isfinite(idepth_max))
    {
        if(debugPrint)
            printf("NO SIGNIFICANT IMPROVMENT (%f)!\n", errorInPixel);
        lastTraceUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
        lastValidUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
        lastTracePixelInterval=dist;
        lastTraceStatus = ImmaturePointStatus::IPS_BADCONDITION;
        return newPt;
    }

    if(errorInPixel >10) errorInPixel=10;

    // ============== do the discrete search ===================
    dx /= dist;
    dy /= dist;

    //    if(debugPrint)
    //        printf("trace pt (%.1f %.1f) from frame %d to %d. Range %f (%.1f %.1f) -> %f (%.1f %.1f)! ErrorInPixel %.1f!\n",
    //               u,v,
    //               host->shell->id, frame->shell->id,
    //               idepth_min, uMin, vMin,
    //               idepth_max, uMax, vMax,
    //               errorInPixel
    //               );


    if(dist>maxPixSearch)
    {
        uMax = uMin + maxPixSearch*dx;
        vMax = vMin + maxPixSearch*dy;
        dist = maxPixSearch;
    }

    rotatePattern( hostToFrame_KRKi );

    int numSteps = 1.9999f + dist / setting_trace_stepsize;
    float randShift = uMin*1000-floorf(uMin*1000);
    float ptx = uMin-randShift*dx;
    float pty = vMin-randShift*dy;


    if(!std::isfinite(dx) || !std::isfinite(dy))
    {
        //printf("COUGHT INF / NAN dxdy (%f %f)!\n", dx, dx);
        lastValidUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
        lastTracePixelInterval=0;
        lastTraceUV = Vec2f(-1,-1);
        lastTraceStatus = ImmaturePointStatus::IPS_OOB;
        return newPt;
    }

    float gtFloatIdx = -1;
    if ( idepth_GT > 0 )
    {
        gtFloatIdx = (((kp_GT[0] - ptx) / dx) + ((kp_GT[1] - pty) / dy)) * 0.5f;
        gtIdx = int(std::round(gtFloatIdx));
    }

    #define MAX_ELEMS 100
    float errors[MAX_ELEMS];
    float bestU=0, bestV=0, bestEnergy=1e10;
    int bestIdx=-1;
    if(numSteps >= MAX_ELEMS) numSteps = MAX_ELEMS-1;

    for(int i=0;i<numSteps;i++)
    {
        float energy=0;
        for(int idx=0;idx<patternNum;idx++)
        {
            //Vec3f hit = getInterpolatedElement33BiCub(frame->dI, (float)(ptx+rotatetPattern[idx][0]), (float)(pty+rotatetPattern[idx][1]), wG[0]);
            //float hitColor = hit[0];
            float hitColor = getInterpolatedElement31(frame->dI, (float)(ptx+rotatetPattern[idx][0]), (float)(pty+rotatetPattern[idx][1]), wG[0]);
            if(!std::isfinite(hitColor)) {energy+=1e5; continue;}
            float residual = hitColor - (float)(hostToFrame_affine[0] * color[idx] + hostToFrame_affine[1]);
            //float residual = hitColor - float(color[idx]);
            float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
            energy += hw *residual*residual*(2-hw);
        }

        if(debugPrint)
            printf("step %.1f %.1f (id %f): energy = %f!\n",
                   ptx, pty, 0.0f, energy);


        errors[i] = energy;
        if(energy < bestEnergy)
        {
            bestU = ptx; bestV = pty; bestEnergy = energy; bestIdx = i;
        }

        ptx+=dx;
        pty+=dy;
    }

    // find best score outside a +-2px radius.
    float secondBest=1e10;
    for(int i=0;i<numSteps;i++)
    {
        if((i < bestIdx-setting_minTraceTestRadius || i > bestIdx+setting_minTraceTestRadius) && errors[i] < secondBest)
            secondBest = errors[i];
    }
    float newQuality = secondBest / bestEnergy;

    if( newQuality < quality || numSteps > 10 )
    {
        quality = newQuality;
        bestIdxDSO = bestIdx;
    }

    if(setting_trace_GNIterations>0)
        bestEnergy = doRefinement( frame, hostToFrame_affine, dx, dy, bestU, bestV );

    checkResult ( hostToFrame_Kt, pr, errorInPixel, dx, dy, uMin, uMax, vMin, vMax, bestU, bestV, bestEnergy );
    return newPt;
}

void ImmaturePoint::checkResult ( const Vec3f & hostToFrame_Kt, const Vec3f & pr, const float & errorInPixel, const float & dx, const float & dy, const float & uMin, const float & uMax, const float & vMin, const float & vMax, const float & bestU, const float & bestV, const float & bestEnergy, const bool debugPrint )
{
    // ============== detect energy-based outlier. ===================
    //	float absGrad0 = getInterpolatedElement(frame->absSquaredGrad[0],bestU, bestV, wG[0]);
    //	float absGrad1 = getInterpolatedElement(frame->absSquaredGrad[1],bestU*0.5-0.25, bestV*0.5-0.25, wG[1]);
    //	float absGrad2 = getInterpolatedElement(frame->absSquaredGrad[2],bestU*0.25-0.375, bestV*0.25-0.375, wG[2]);
    if( !(bestEnergy < energyTH*setting_trace_extraSlackOnTH))
        //			|| (absGrad0*areaGradientSlackFactor < host->frameGradTH
        //		     && absGrad1*areaGradientSlackFactor < host->frameGradTH*0.75f
        //			 && absGrad2*areaGradientSlackFactor < host->frameGradTH*0.50f))
    {
        if(debugPrint)
            printf("OUTLIER!\n");
        lastValidUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
        lastTracePixelInterval=0;
        lastTraceUV = Vec2f(-1,-1);
        if(lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER)
        {
            lastTraceStatus = ImmaturePointStatus::IPS_OOB;
            return;
        }
        else
        {
            lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
            return;
        }
    }


    // ============== set new interval ===================
    if(dx*dx>dy*dy)
    {
        idepth_min = (pr[2]*(bestU-errorInPixel*dx) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2]*(bestU-errorInPixel*dx));
        idepth_max = (pr[2]*(bestU+errorInPixel*dx) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2]*(bestU+errorInPixel*dx));
    }
    else
    {
        idepth_min = (pr[2]*(bestV-errorInPixel*dy) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2]*(bestV-errorInPixel*dy));
        idepth_max = (pr[2]*(bestV+errorInPixel*dy) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2]*(bestV+errorInPixel*dy));
    }
    if(idepth_min > idepth_max) std::swap<float>(idepth_min, idepth_max);

    idepthBCA_est = ((idepth_min+idepth_max)/2);

    if(!std::isfinite(idepth_min) || !std::isfinite(idepth_max) || (idepth_max<0))
    {
        //printf("COUGHT INF / NAN minmax depth (%f %f)!\n", idepth_min, idepth_max);
        lastValidUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
        lastTracePixelInterval=0;
        lastTraceUV = Vec2f(-1,-1);
        lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
        return;
    }

    lastValidUV = Vec2f(bestU, bestV);
    lastTracePixelInterval=2*errorInPixel;
    lastTraceUV = Vec2f(bestU, bestV);
    lastTraceStatus = ImmaturePointStatus::IPS_GOOD;

//    if ( idepth_GT > 0 )
//    {
//        float gtEnergy=0;
//        float gtNgfEnergy=0;
//        Eigen::Matrix2Xf NgfGtVals ( 2, patternNum);
//        if ( kp_GT(0) > 3 && kp_GT(0) < wM3G && kp_GT(1) > 3 && kp_GT(1) < hM3G )
//        {
//            for(int idx=0;idx<patternNum;idx++)
//            {
//                float hitColor = getInterpolatedElement31(frame->dI, (float)(kp_GT(0)+rotatetPattern[idx][0]), (float)(kp_GT(1)+rotatetPattern[idx][1]), wG[0]);

//                if(!std::isfinite(hitColor)) {gtEnergy+=1e5; continue;}
//                //float residual = hitColor - (float)(hostToFrame_affine[0] * color[idx] + hostToFrame_affine[1]);
//                float residual = hitColor - float(color[idx]);
//                float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
//                gtEnergy += hw *residual*residual*(2-hw);
//            }
//            for(int idx=0;idx<patternNum;idx++)
//            {
//                if ( skipZero[idx] ) continue;
//                //const Vec3f hitColor = getInterpolatedElement33BiLin(frame->dI, (float)(kp_GT(0)+rotatetPatternNgf[idx][0]), (float)(kp_GT(1)+rotatetPatternNgf[idx][1]), wG[0]);
//                const Vec3f hitColor = getInterpolatedElement33BiCub(frame->dI, (float)(kp_GT(0)+rotatetPatternNgf[idx][0]), (float)(kp_GT(1)+rotatetPatternNgf[idx][1]), wG[0]);
//                if( !hitColor.allFinite() ) {gtNgfEnergy+=1e5; continue;}
//                Vec2f hitD = hitColor.tail<2>();
//                hitD /= sqrt(hitD.squaredNorm()+settings_Eta);
//                const float nn = hitD.dot(colorD[idx]);
//                NgfGtVals.col(idx) = hitD;
//                const float residual = std::max<float>(0.f,std::min<float>(1.f,nn < 0 ? 1.f : 1-nn ));
//                //const float residual = std::max<float>(0.f,std::min<float>(1.f,1.f-nn*nn)); // original ngf residual
//                //float hw = fabs(residual) < setting_huber_ngf_TH ? 1 : setting_huber_ngf_TH / fabs(residual);
//                //gtNgfEnergy += hw *residual*residual*(2-hw);
//                gtNgfEnergy += residual;
//            }
//        }
//        bool gtInRangeBca = idepth_min < idepth_GT && idepth_GT < idepth_max;
//        bool gtInRangeNgf = idepthNgf_min < idepth_GT && idepth_GT < idepthNgf_max;
//        bool closeGtBca = (Vec2f(bestU,bestV) - kp_GT).norm() < 2;
//        bool closeGtNgf = (Vec2f(bestNgfU,bestNgfV) - kp_GT).norm() < 2;
//        std::cout << "bestIdx="<<bestIdx<< " ("<<bestEnergy<< ") bestNgfIdx=" << bestNgfIdx << " (" << bestNgfEnergy <<") gtIdx= " << gtIdx << " ("<< gtEnergy << " / " << gtNgfEnergy<< ") o: "<< errors[gtIdx] << "/" << ngfErrors[gtIdx] << " gtFloatIdx=" <<gtFloatIdx << " inRange:" << gtInRangeBca << " " << gtInRangeNgf << " close: " << closeGtBca << " " << closeGtNgf << std::endl;
//        if ( !gtInRangeNgf && gtInRangeBca )
//        {
//            std::stringstream gfX, gfY;
//            for(int i=0;i<patternNum;++i)
//            {
//                gfX << std::fixed << std::setw( 3 ) << std::setprecision ( 3 ) << " " << colorD[i](0);
//                gfY << std::fixed << std::setw( 3 ) << std::setprecision ( 3 ) << " " << colorD[i](1);
//            }
//            std::cout << "TngfX=["<< ngfValsX.col(bestNgfIdx).transpose() << "] RngfY=[" << ngfValsY.col(bestNgfIdx).transpose() << "];"<< std::endl;
//            std::cout << "RngfX=["<< gfX.str() <<"] RngfY=[" << gfY.str() << "];"<< std::endl;
//            std::cout << "NgfGtVals=["<< NgfGtVals <<"];"<< std::endl;
//        }
//    }

}

void ImmaturePoint::checkResultNgf ( const Vec3f & hostToFrame_Kt, const Vec3f & pr, const float & errorInPixel, const float & dx, const float & dy, const float & uMin, const float & uMax, const float & vMin, const float & vMax, const float & bestU, const float & bestV, const float & bestEnergy, const bool debugPrint )
{
    // ============== detect energy-based outlier. ===================
    //	float absGrad0 = getInterpolatedElement(frame->absSquaredGrad[0],bestU, bestV, wG[0]);
    //	float absGrad1 = getInterpolatedElement(frame->absSquaredGrad[1],bestU*0.5-0.25, bestV*0.5-0.25, wG[1]);
    //	float absGrad2 = getInterpolatedElement(frame->absSquaredGrad[2],bestU*0.25-0.375, bestV*0.25-0.375, wG[2]);
    if( !( bestEnergy < energyTHngf*setting_trace_extraSlackOnTH ))
        //			|| (absGrad0*areaGradientSlackFactor < host->frameGradTH
        //		     && absGrad1*areaGradientSlackFactor < host->frameGradTH*0.75f
        //			 && absGrad2*areaGradientSlackFactor < host->frameGradTH*0.50f))
    {
        if(debugPrint)
            printf("OUTLIER!\n");
        lastValidUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
        lastTracePixelInterval=0;
        lastTraceUV = Vec2f(-1,-1);
        if(lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER)
        {
            lastTraceStatus = ImmaturePointStatus::IPS_OOB;
            return;
        }
        else
        {
            lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
            return;
        }
    }


    // ============== set new interval ===================
    if(dx*dx>dy*dy)
    {
        idepth_min = (pr[2]*(bestU-errorInPixel*dx) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2]*(bestU-errorInPixel*dx));
        idepth_max = (pr[2]*(bestU+errorInPixel*dx) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2]*(bestU+errorInPixel*dx));
    }
    else
    {
        idepth_min = (pr[2]*(bestV-errorInPixel*dy) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2]*(bestV-errorInPixel*dy));
        idepth_max = (pr[2]*(bestV+errorInPixel*dy) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2]*(bestV+errorInPixel*dy));
    }
    if(idepth_min > idepth_max) std::swap<float>(idepth_min, idepth_max);

    idepthNgf_est = ((idepth_min+idepth_max)/2);


    if(!std::isfinite(idepth_min) || !std::isfinite(idepth_max) || (idepth_max<0))
    {
        //printf("COUGHT INF / NAN minmax depth (%f %f)!\n", idepth_min, idepth_max);
        lastValidUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
        lastTracePixelInterval=0;
        lastTraceUV = Vec2f(-1,-1);
        lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
        return;
    }

    lastValidUV = Vec2f(bestU, bestV);
    lastTracePixelInterval=2*errorInPixel;
    lastTraceUV = Vec2f(bestU, bestV);
    lastTraceStatus = ImmaturePointStatus::IPS_GOOD;
    lastTraceEnergy = bestEnergy;
}

void ImmaturePoint::checkResultNcc ( const Vec3f & hostToFrame_Kt, const Vec3f & pr, const float & errorInPixel, const float & dx, const float & dy, const float & uMin, const float & uMax, const float & vMin, const float & vMax, const float & bestU, const float & bestV, const float & bestEnergy, const bool debugPrint )
{
    // ============== detect energy-based outlier. ===================
    //	float absGrad0 = getInterpolatedElement(frame->absSquaredGrad[0],bestU, bestV, wG[0]);
    //	float absGrad1 = getInterpolatedElement(frame->absSquaredGrad[1],bestU*0.5-0.25, bestV*0.5-0.25, wG[1]);
    //	float absGrad2 = getInterpolatedElement(frame->absSquaredGrad[2],bestU*0.25-0.375, bestV*0.25-0.375, wG[2]);
    if( bestEnergy < 0.5f )
        //			|| (absGrad0*areaGradientSlackFactor < host->frameGradTH
        //		     && absGrad1*areaGradientSlackFactor < host->frameGradTH*0.75f
        //			 && absGrad2*areaGradientSlackFactor < host->frameGradTH*0.50f))
    {
        if(debugPrint)
            printf("OUTLIER!\n");
        lastValidUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
        lastTracePixelInterval=0;
        lastTraceUV = Vec2f(-1,-1);
        if(lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER)
        {
            lastTraceStatus = ImmaturePointStatus::IPS_OOB;
            return;
        }
        else
        {
            lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
            return;
        }
    }

    // ============== set new interval ===================
    if(dx*dx>dy*dy)
    {
        idepth_min = (pr[2]*(bestU-errorInPixel*dx) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2]*(bestU-errorInPixel*dx));
        idepth_max = (pr[2]*(bestU+errorInPixel*dx) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2]*(bestU+errorInPixel*dx));
    }
    else
    {
        idepth_min = (pr[2]*(bestV-errorInPixel*dy) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2]*(bestV-errorInPixel*dy));
        idepth_max = (pr[2]*(bestV+errorInPixel*dy) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2]*(bestV+errorInPixel*dy));
    }
    if(idepth_min > idepth_max) std::swap<float>(idepth_min, idepth_max);

    idepthNgf_est = ((idepth_min+idepth_max)/2);


    if(!std::isfinite(idepth_min) || !std::isfinite(idepth_max) || (idepth_max<0))
    {
        //printf("COUGHT INF / NAN minmax depth (%f %f)!\n", idepth_min, idepth_max);
        lastValidUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
        lastTracePixelInterval=0;
        lastTraceUV = Vec2f(-1,-1);
        lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
        return;
    }

    lastValidUV = Vec2f(bestU, bestV);
    lastTracePixelInterval=2*errorInPixel;
    lastTraceUV = Vec2f(bestU, bestV);
    lastTraceStatus = ImmaturePointStatus::IPS_GOOD;
    lastTraceEnergy = bestEnergy;
}


/*
 * returns
 * * OOB -> point is optimized and marginalized
 * * UPDATED -> point has been updated.
 * * SKIP -> point has not been updated.
 */
ImmaturePointPtr ImmaturePoint::traceOnNgf(ImageDataPtr frame,const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt, bool debugPrint)
{
    ImmaturePointPtr newPt ( nullptr );

    if(lastTraceStatus == ImmaturePointStatus::IPS_OOB || lastTraceStatus == ImmaturePointStatus::IPS_DELETED ) return newPt;//lastTraceStatus;

    ++numTraced;

    if ( !frame )
    {
        lastTraceStatus = ImmaturePointStatus::IPS_SKIPPED;
        return newPt;
    }
    if ( frame->dI == nullptr )
        copyImage ( frame );

    debugPrint = false;//rand()%100==0;
    float maxPixSearch = (wG[0]+hG[0])*setting_maxPixSearch;

    //    if(debugPrint)
    //        printf("trace pt (%.1f %.1f) from frame %d to %d. Range %f -> %f. t %f %f %f!\n",
    //               u,v,
    //               host->shell->id, frame->shell->id,
    //               idepth_min, idepth_max,
    //               hostToFrame_Kt[0],hostToFrame_Kt[1],hostToFrame_Kt[2]);

    //	const float stepsize = 1.0;				// stepsize for initial discrete search.
    //	const int GNIterations = 3;				// max # GN iterations
    //	const float GNThreshold = 0.1;				// GN stop after this stepsize.
    //	const float extraSlackOnTH = 1.2;			// for energy-based outlier check, be slightly more relaxed by this factor.
    //	const float slackInterval = 0.8;			// if pixel-interval is smaller than this, leave it be.
    //	const float minImprovementFactor = 2;		// if pixel-interval is smaller than this, leave it be.
    // ============== project min and max. return if one of them is OOB ===================
    Vec3f pr = hostToFrame_KRKi * Vec3f(u,v, 1);
    Vec3f ptpMin = pr + hostToFrame_Kt*idepth_min;
    float uMin = ptpMin[0] / ptpMin[2];
    float vMin = ptpMin[1] / ptpMin[2];

    if ( idepth_GT > 0 )
    {
        Vec3f p = pr + hostToFrame_Kt*idepth_GT;
        kp_GT = p.hnormalized();
    }

    if(!(uMin > 4 && vMin > 4 && uMin < wG[0]-5 && vMin < hG[0]-5))
    {
        if(debugPrint) printf("OOB uMin %f %f - %f %f %f (id %f-%f)!\n",
                              u,v,uMin, vMin,  ptpMin[2], idepth_min, idepth_max);

        Vec3f ptpMax = pr + hostToFrame_Kt* ( std::isfinite(idepth_max) ? idepth_max : 0.01 );
        float uMax = ptpMax[0] / ptpMax[2];
        float vMax = ptpMax[1] / ptpMax[2];

        lastValidUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
        lastTraceUV = Vec2f(-1,-1);
        lastTracePixelInterval=0;
        lastTraceStatus = ImmaturePointStatus::IPS_OOB;
        return newPt;
    }

    float dist;
    float uMax;
    float vMax;
    Vec3f ptpMax;
    if(std::isfinite(idepth_max))
    {
        ptpMax = pr + hostToFrame_Kt*idepth_max;
        uMax = ptpMax[0] / ptpMax[2];
        vMax = ptpMax[1] / ptpMax[2];


        if(!(uMax > 4 && vMax > 4 && uMax < wG[0]-5 && vMax < hG[0]-5))
        {
            if(debugPrint) printf("OOB uMax  %f %f - %f %f!\n",u,v, uMax, vMax);
            lastValidUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
            lastTraceUV = Vec2f(-1,-1);
            lastTracePixelInterval=0;
            lastTraceStatus = ImmaturePointStatus::IPS_OOB;
            return newPt;
        }

        // ============== check their distance. everything below 2px is OK (-> skip). ===================
        dist = (uMin-uMax)*(uMin-uMax) + (vMin-vMax)*(vMin-vMax);
        dist = sqrtf(dist);
        if(dist < setting_trace_slackInterval)
        {
            //            rotatePattern( hostToFrame_KRKi );
            //            float dx = setting_trace_stepsize*(uMax-uMin);
            //            float dy = setting_trace_stepsize*(vMax-vMin);
            //            dx /= dist;
            //            dy /= dist;
            //            float bestU = (uMin + uMax)*.5;
            //            float bestV = (vMin + vMax)*.5;
            //            float bestEnergy = doRefinementNgf ( frame, dx, dy, bestU, bestV );
            //            checkResultNgf ( hostToFrame_Kt, pr, 1, dx, dy, uMin, uMax, vMin, vMax, bestU, bestV, bestEnergy );

            if(debugPrint)
                printf("TOO CERTAIN ALREADY (dist %f)!\n", dist);
            lastValidUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
            lastTraceUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
            lastTracePixelInterval=dist;
            lastTraceStatus = ImmaturePointStatus::IPS_SKIPPED;
            return newPt;
        }
        assert(dist>0);
    }
    else
    {
        dist = maxPixSearch;

        // project to arbitrary depth to get direction.
        ptpMax = pr + hostToFrame_Kt*0.01;
        uMax = ptpMax[0] / ptpMax[2];
        vMax = ptpMax[1] / ptpMax[2];

        // direction.
        float dx = uMax-uMin;
        float dy = vMax-vMin;
        float d = 1.0f / sqrtf(dx*dx+dy*dy);

        // set to [setting_maxPixSearch].
        uMax = uMin + dist*dx*d;
        vMax = vMin + dist*dy*d;

        // may still be out!
        if(!(uMax > 4 && vMax > 4 && uMax < wG[0]-5 && vMax < hG[0]-5))
        {
            if(debugPrint) printf("OOB uMax-coarse %f %f %f!\n", uMax, vMax,  ptpMax[2]);
            lastTraceUV = Vec2f(-1,-1);
            lastValidUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
            lastTracePixelInterval=0;
            lastTraceStatus = ImmaturePointStatus::IPS_OOB;
            return newPt;
        }
        assert(dist>0);
    }


    // set OOB if scale change too big.
    if(!(idepth_min<0 || (ptpMin[2]>0.75 && ptpMin[2]<1.5)))
    {
        if(debugPrint) printf("OOB SCALE %f %f %f!\n", uMax, vMax,  ptpMin[2]);
        lastTraceUV = Vec2f(-1,-1);
        lastValidUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
        lastTracePixelInterval=0;
        lastTraceStatus = ImmaturePointStatus::IPS_OOB;
        return newPt;
    }


    // ============== compute error-bounds on result in pixel. if the new interval is not at least 1/2 of the old, SKIP ===================
    float dx = setting_trace_stepsize*(uMax-uMin);
    float dy = setting_trace_stepsize*(vMax-vMin);

    float a = (Vec2f(dx,dy).transpose() * gradH * Vec2f(dx,dy));
    float b = (Vec2f(dy,-dx).transpose() * gradH * Vec2f(dy,-dx));
    float errorInPixel = 0.2f + 0.2f * (a+b) / a; // WO kommt das her? Scheint nicht zu NGF zu passen !

    if(errorInPixel*setting_trace_minImprovementFactor > dist && std::isfinite(idepth_max))
    {
        if(debugPrint)
            printf("NO SIGNIFICANT IMPROVMENT (%f)!\n", errorInPixel);
        lastTraceUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
        lastValidUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
        lastTracePixelInterval=dist;
        lastTraceStatus = ImmaturePointStatus::IPS_BADCONDITION;
        return newPt;
    }

    if(errorInPixel >10) errorInPixel=10;



    // ============== do the discrete search ===================
    dx /= dist;
    dy /= dist;

    //    if(debugPrint)
    //        printf("trace pt (%.1f %.1f) from frame %d to %d. Range %f (%.1f %.1f) -> %f (%.1f %.1f)! ErrorInPixel %.1f!\n",
    //               u,v,
    //               host->shell->id, frame->shell->id,
    //               idepth_min, uMin, vMin,
    //               idepth_max, uMax, vMax,
    //               errorInPixel
    //               );


    if(dist>maxPixSearch)
    {
        uMax = uMin + maxPixSearch*dx;
        vMax = vMin + maxPixSearch*dy;
        dist = maxPixSearch;
    }

    int numSteps = 1.9999f + dist / setting_trace_stepsize;
    float randShift = uMin*1000-floorf(uMin*1000);
    float ptx = uMin-randShift*dx;
    float pty = vMin-randShift*dy;

    rotatePattern( hostToFrame_KRKi );

    if(!std::isfinite(dx) || !std::isfinite(dy))
    {
        //printf("COUGHT INF / NAN dxdy (%f %f)!\n", dx, dx);
        lastValidUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
        lastTracePixelInterval=0;
        lastTraceUV = Vec2f(-1,-1);
        lastTraceStatus = ImmaturePointStatus::IPS_OOB;
        return newPt;
    }

    float gtFloatIdx = -1;
    if ( idepth_GT > 0 )
    {
        gtFloatIdx = (((kp_GT[0] - ptx) / dx) + ((kp_GT[1] - pty) / dy)) * 0.5f;
        gtIdx = int(std::round(gtFloatIdx));

    //std::cout << "gtIdx=[" << gtIdx << ","<<gtIdxV << "];" << std::endl;
    }

    #define MAX_ELEMS 100

    float errors[MAX_ELEMS];
    float bestU=0, bestV=0, bestEnergy=1e10;
    int bestIdx=-1;
    if(numSteps >= MAX_ELEMS) numSteps = MAX_ELEMS-1;

    for(int i=0;i<numSteps;i++)
    {
        float energy=0;
        for(int idx=0;idx<patternNum; ++idx)
        {
            if ( skipZero[idx] ) continue;
            const Vec3f hitColor = getInterpolatedElement33BiCub(frame->dI, (float)(ptx+rotatetPattern[idx][0]), (float)(pty+rotatetPattern[idx][1]), wG[0]);
            if( !hitColor.allFinite() ) {energy+=1e5; continue;}
            Vec2f hitD = hitColor.tail<2>();
            hitD /= sqrt(hitD.squaredNorm()+settings_Eta);
            const float nn = hitD.dot(colorD[idx]);
            const float residual = std::max<float>(0.f,std::min<float>(1.f,nn < 0 ? 1.f : 1-nn ));// uni modal ngf
            //const float residual = std::max<float>(0.f,std::min<float>(2.f, 1-nn ));// uni modal ngf
            //const float residual = std::max<float>(0.f,std::min<float>(1.f,1.f-nn*nn)); // original ngf residual
            float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
            energy += hw *residual*residual*(2-hw);
            //energy += residual;
        }

        if(debugPrint)
            printf("step %.1f %.1f (id %f): energy = %f!\n",
                   ptx, pty, 0.0f, energy);

        errors[i] = energy;
        if(energy < bestEnergy)
        {
            bestU = ptx; bestV = pty; bestEnergy = energy; bestIdx = i;
        }

        ptx+=dx;
        pty+=dy;
    }

    // find best score outside a +-2px radius.
    float secondBest=1e10;
    for(int i=0;i<numSteps;i++)
    {
        if((i < bestIdx-setting_minTraceTestRadius || i > bestIdx+setting_minTraceTestRadius) && errors[i] < secondBest)
            secondBest = errors[i];
    }
    float newQuality = secondBest / bestEnergy;

    if( newQuality < quality || numSteps > 10 )
    {
        quality = newQuality;
        bestIdxNgf = bestIdx;
    }

    if(setting_trace_GNIterations>0)
        bestEnergy = doRefinementNgf( frame, dx, dy, bestU, bestV );

    checkResultNgf ( hostToFrame_Kt, pr, errorInPixel, dx, dy, uMin, uMax, vMin, vMax, bestU, bestV, bestEnergy );

    return newPt;
}

/*
 * returns
 * * OOB -> point is optimized and marginalized
 * * UPDATED -> point has been updated.
 * * SKIP -> point has not been updated.
 */
ImmaturePointPtr ImmaturePoint::traceOnNcc(ImageDataPtr frame,const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt, bool debugPrint)
{
    ImmaturePointPtr newPt ( nullptr );

    if(lastTraceStatus == ImmaturePointStatus::IPS_OOB || lastTraceStatus == ImmaturePointStatus::IPS_DELETED ) return newPt;//lastTraceStatus;

    ++numTraced;

    if ( !frame )
    {
        lastTraceStatus = ImmaturePointStatus::IPS_SKIPPED;
        return newPt;
    }
    if ( frame->dI == nullptr )
        copyImage ( frame );

    debugPrint = false;//rand()%100==0;
    float maxPixSearch = (wG[0]+hG[0])*setting_maxPixSearch;

    //    if(debugPrint)
    //        printf("trace pt (%.1f %.1f) from frame %d to %d. Range %f -> %f. t %f %f %f!\n",
    //               u,v,
    //               host->shell->id, frame->shell->id,
    //               idepth_min, idepth_max,
    //               hostToFrame_Kt[0],hostToFrame_Kt[1],hostToFrame_Kt[2]);

    //	const float stepsize = 1.0;				// stepsize for initial discrete search.
    //	const int GNIterations = 3;				// max # GN iterations
    //	const float GNThreshold = 0.1;				// GN stop after this stepsize.
    //	const float extraSlackOnTH = 1.2;			// for energy-based outlier check, be slightly more relaxed by this factor.
    //	const float slackInterval = 0.8;			// if pixel-interval is smaller than this, leave it be.
    //	const float minImprovementFactor = 2;		// if pixel-interval is smaller than this, leave it be.
    // ============== project min and max. return if one of them is OOB ===================
    Vec3f pr = hostToFrame_KRKi * Vec3f(u,v, 1);
    Vec3f ptpMin = pr + hostToFrame_Kt*idepth_min;
    float uMin = ptpMin[0] / ptpMin[2];
    float vMin = ptpMin[1] / ptpMin[2];

    if ( idepth_GT > 0 )
    {
        Vec3f p = pr + hostToFrame_Kt*idepth_GT;
        kp_GT = p.hnormalized();
    }

    if(!(uMin > 4 && vMin > 4 && uMin < wG[0]-5 && vMin < hG[0]-5))
    {
        if(debugPrint) printf("OOB uMin %f %f - %f %f %f (id %f-%f)!\n",
                              u,v,uMin, vMin,  ptpMin[2], idepth_min, idepth_max);

        Vec3f ptpMax = pr + hostToFrame_Kt* ( std::isfinite(idepth_max) ? idepth_max : 0.01 );
        float uMax = ptpMax[0] / ptpMax[2];
        float vMax = ptpMax[1] / ptpMax[2];

        lastValidUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
        lastTraceUV = Vec2f(-1,-1);
        lastTracePixelInterval=0;
        lastTraceStatus = ImmaturePointStatus::IPS_OOB;
        return newPt;
    }

    float dist;
    float uMax;
    float vMax;
    Vec3f ptpMax;
    if(std::isfinite(idepth_max))
    {
        ptpMax = pr + hostToFrame_Kt*idepth_max;
        uMax = ptpMax[0] / ptpMax[2];
        vMax = ptpMax[1] / ptpMax[2];


        if(!(uMax > 4 && vMax > 4 && uMax < wG[0]-5 && vMax < hG[0]-5))
        {
            if(debugPrint) printf("OOB uMax  %f %f - %f %f!\n",u,v, uMax, vMax);
            lastValidUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
            lastTraceUV = Vec2f(-1,-1);
            lastTracePixelInterval=0;
            lastTraceStatus = ImmaturePointStatus::IPS_OOB;
            return newPt;
        }

        // ============== check their distance. everything below 2px is OK (-> skip). ===================
        dist = (uMin-uMax)*(uMin-uMax) + (vMin-vMax)*(vMin-vMax);
        dist = sqrtf(dist);
        if(dist < setting_trace_slackInterval)
        {
            //            rotatePattern( hostToFrame_KRKi );
            //            float dx = setting_trace_stepsize*(uMax-uMin);
            //            float dy = setting_trace_stepsize*(vMax-vMin);
            //            dx /= dist;
            //            dy /= dist;
            //            float bestU = (uMin + uMax)*.5;
            //            float bestV = (vMin + vMax)*.5;
            //            float bestEnergy = doRefinementNgf ( frame, dx, dy, bestU, bestV );
            //            checkResultNgf ( hostToFrame_Kt, pr, 1, dx, dy, uMin, uMax, vMin, vMax, bestU, bestV, bestEnergy );

            if(debugPrint)
                printf("TOO CERTAIN ALREADY (dist %f)!\n", dist);
            lastValidUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
            lastTraceUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
            lastTracePixelInterval=dist;
            lastTraceStatus = ImmaturePointStatus::IPS_SKIPPED;
            return newPt;
        }
        assert(dist>0);
    }
    else
    {
        dist = maxPixSearch;

        // project to arbitrary depth to get direction.
        ptpMax = pr + hostToFrame_Kt*0.01;
        uMax = ptpMax[0] / ptpMax[2];
        vMax = ptpMax[1] / ptpMax[2];

        // direction.
        float dx = uMax-uMin;
        float dy = vMax-vMin;
        float d = 1.0f / sqrtf(dx*dx+dy*dy);

        // set to [setting_maxPixSearch].
        uMax = uMin + dist*dx*d;
        vMax = vMin + dist*dy*d;

        // may still be out!
        if(!(uMax > 4 && vMax > 4 && uMax < wG[0]-5 && vMax < hG[0]-5))
        {
            if(debugPrint) printf("OOB uMax-coarse %f %f %f!\n", uMax, vMax,  ptpMax[2]);
            lastTraceUV = Vec2f(-1,-1);
            lastValidUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
            lastTracePixelInterval=0;
            lastTraceStatus = ImmaturePointStatus::IPS_OOB;
            return newPt;
        }
        assert(dist>0);
    }


    // set OOB if scale change too big.
    if(!(idepth_min<0 || (ptpMin[2]>0.75 && ptpMin[2]<1.5)))
    {
        if(debugPrint) printf("OOB SCALE %f %f %f!\n", uMax, vMax,  ptpMin[2]);
        lastTraceUV = Vec2f(-1,-1);
        lastValidUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
        lastTracePixelInterval=0;
        lastTraceStatus = ImmaturePointStatus::IPS_OOB;
        return newPt;
    }


    // ============== compute error-bounds on result in pixel. if the new interval is not at least 1/2 of the old, SKIP ===================
    float dx = setting_trace_stepsize*(uMax-uMin);
    float dy = setting_trace_stepsize*(vMax-vMin);

    float a = (Vec2f(dx,dy).transpose() * gradH * Vec2f(dx,dy));
    float b = (Vec2f(dy,-dx).transpose() * gradH * Vec2f(dy,-dx));
    float errorInPixel = 0.2f + 0.2f * (a+b) / a; // WO kommt das her? Scheint nicht zu NGF zu passen !

    if(errorInPixel*setting_trace_minImprovementFactor > dist && std::isfinite(idepth_max))
    {
        if(debugPrint)
            printf("NO SIGNIFICANT IMPROVMENT (%f)!\n", errorInPixel);
        lastTraceUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
        lastValidUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
        lastTracePixelInterval=dist;
        lastTraceStatus = ImmaturePointStatus::IPS_BADCONDITION;
        return newPt;
    }

    if(errorInPixel >10) errorInPixel=10;



    // ============== do the discrete search ===================
    dx /= dist;
    dy /= dist;

    //    if(debugPrint)
    //        printf("trace pt (%.1f %.1f) from frame %d to %d. Range %f (%.1f %.1f) -> %f (%.1f %.1f)! ErrorInPixel %.1f!\n",
    //               u,v,
    //               host->shell->id, frame->shell->id,
    //               idepth_min, uMin, vMin,
    //               idepth_max, uMax, vMax,
    //               errorInPixel
    //               );


    if(dist>maxPixSearch)
    {
        uMax = uMin + maxPixSearch*dx;
        vMax = vMin + maxPixSearch*dy;
        dist = maxPixSearch;
    }

    int numSteps = 1.9999f + dist / setting_trace_stepsize;
    float randShift = uMin*1000-floorf(uMin*1000);
    float ptx = uMin-randShift*dx;
    float pty = vMin-randShift*dy;

    rotatePattern( hostToFrame_KRKi );

    if(!std::isfinite(dx) || !std::isfinite(dy))
    {
        //printf("COUGHT INF / NAN dxdy (%f %f)!\n", dx, dx);
        lastValidUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
        lastTracePixelInterval=0;
        lastTraceUV = Vec2f(-1,-1);
        lastTraceStatus = ImmaturePointStatus::IPS_OOB;
        return newPt;
    }

    float gtFloatIdx = -1;
    if ( idepth_GT > 0 )
    {
        gtFloatIdx = (((kp_GT[0] - ptx) / dx) + ((kp_GT[1] - pty) / dy)) * 0.5f;
        gtIdx = int(std::round(gtFloatIdx));

    //std::cout << "gtIdx=[" << gtIdx << ","<<gtIdxV << "];" << std::endl;
    }

    #define MAX_ELEMS 100

    float errors[MAX_ELEMS];
    float bestU=0, bestV=0, bestEnergy=-1.0f;
    int bestIdx=-1;
    if(numSteps >= MAX_ELEMS) numSteps = MAX_ELEMS-1;

    // Retrieve template statistics for NCC matching;
    const float sum_templ = ncc_sum_templ ;
    const float const_templ_denom = ncc_const_templ;

    for(int i=0;i<numSteps;i++)
    {
        float energy = 0;
        float sum_img = 0.f;
        float sum_img_sq = 0.f;
        float sum_img_templ = 0.f;
        for(int idx=0;idx<patternNum; ++idx)
        {
            float hitColor = getInterpolatedElement31(frame->dI, (float)(ptx+rotatetPattern[idx][0]), (float)(pty+rotatetPattern[idx][1]), wG[0]);
            if(!std::isfinite(hitColor)) {energy-=1e5; continue;}

            const float templ = color[idx];
            const float img = hitColor;
            sum_img    += img;
            sum_img_sq += img*img;
            sum_img_templ += img*templ;
        }
        const float ncc_numerator = patternNum*sum_img_templ - sum_img*sum_templ;
        const float ncc_denominator = (patternNum*sum_img_sq - sum_img*sum_img)*const_templ_denom;
        energy += ncc_numerator * sqrt(ncc_denominator + 1e-10);

        if(debugPrint)
            printf("step %.1f %.1f (id %f): energy = %f!\n",
                   ptx, pty, 0.0f, energy);

        errors[i] = energy;
        if(energy > bestEnergy)
        {
            bestU = ptx; bestV = pty; bestEnergy = energy; bestIdx = i;
        }

        ptx+=dx;
        pty+=dy;
    }

    // find best score outside a +-2px radius.
    float secondBest=1e10;
    for(int i=0;i<numSteps;i++)
    {
        if((i < bestIdx-setting_minTraceTestRadius || i > bestIdx+setting_minTraceTestRadius) && errors[i] > secondBest)
            secondBest = errors[i];
    }
    float newQuality = secondBest / bestEnergy;

    if( newQuality < quality || numSteps > 10 )
    {
        quality = newQuality;
        bestIdxNgf = bestIdx;
    }

    //if(setting_trace_GNIterations>0)
    //    bestEnergy = doRefinementNgf( frame, dx, dy, bestU, bestV );

    checkResultNcc (hostToFrame_Kt, pr, errorInPixel, dx, dy, uMin, uMax, vMin, vMax, bestU, bestV, bestEnergy );

    return newPt;
}

//const Eigen::Affine3d & T_cur_ref, const Mat33d & K,
void ImmaturePoint::searchEpiLineNcc(ImageDataPtr frame, const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt )
{
    if(lastTraceStatus == ImmaturePointStatus::IPS_OOB || lastTraceStatus == ImmaturePointStatus::IPS_DELETED ) return;

    ++numTraced;

    if ( !frame )
    {
        lastTraceStatus = ImmaturePointStatus::IPS_SKIPPED;
        return;
    }
    if ( frame->dI == nullptr )
        copyImage ( frame );

    float idepth_mean = (idepth_min + idepth_max)*0.5;
    Vec3f pr = hostToFrame_KRKi * Vec3f(u,v, 1);
    Vec3f ptpMean = pr + hostToFrame_Kt*idepth_mean;
    Vec3f ptpMin = pr + hostToFrame_Kt*idepth_min;
    Vec3f ptpMax = pr + hostToFrame_Kt*idepth_max;
    Vec2f uvMean = ptpMean.hnormalized();
    Vec2f uvMin = ptpMin.hnormalized();
    Vec2f uvMax = ptpMax.hnormalized();

    rotatePattern( hostToFrame_KRKi );

    Vec2f epi_line = uvMax - uvMin;
    float norm_epi = std::max<float>(1e-5f,epi_line.norm());
    Vec2f epi_dir = epi_line / norm_epi;
    const float  half_length = 0.5f * norm_epi;

    Vec2f bestKp;
    float bestEnergy = -1.0f;

    // Retrieve template statistics for NCC matching;
    const float sum_templ = ncc_sum_templ ;
    const float const_templ_denom = ncc_const_templ;

    for(float l = -half_length; l <= half_length; l += 0.7f)
    {
        float energy = 0;
        float sum_img = 0.f;
        float sum_img_sq = 0.f;
        float sum_img_templ = 0.f;

        Vec2f kp = uvMean + l*epi_dir;

        if( !kp.allFinite() || ( kp(0) >= (wG[0]-5) )  || ( kp(1) >= (hG[0]-5) ) || ( kp(0) < 5 ) || ( kp(1) < 5) )
        {
          continue;
        }

        for(int idx=0;idx<patternNum; ++idx)
        {
            //float hitColor = getInterpolatedElement31(frame->dI, (float)(kp(0)+rotatetPattern[idx][0]), (float)(kp(1)+rotatetPattern[idx][1]), wG[0]);
            float hitColor = getInterpolatedElement33BiCub(frame->dI, (float)(kp(0)+rotatetPattern[idx][0]), (float)(kp(1)+rotatetPattern[idx][1]), wG[0])[0];
            if(!std::isfinite(hitColor)) {energy-=1e5; continue;}

            const float templ = color[idx];
            const float img = hitColor;
            sum_img    += img;
            sum_img_sq += img*img;
            sum_img_templ += img*templ;
        }
        const float ncc_numerator = patternNum*sum_img_templ - sum_img*sum_templ;
        const float ncc_denominator = (patternNum*sum_img_sq - sum_img*sum_img)*const_templ_denom;
        energy += ncc_numerator * sqrt(ncc_denominator + 1e-10);

        if( energy > bestEnergy )
        {
            bestKp = kp; bestEnergy = energy;
        }
    }

    if( bestEnergy < .5f )
    {
        lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
    }
    else
    {
        float a = (Vec2f(epi_dir(0),epi_dir(1)).transpose() * gradH * Vec2f(epi_dir(0),epi_dir(1)));
        float b = (Vec2f(epi_dir(1),-epi_dir(0)).transpose() * gradH * Vec2f(epi_dir(1),-epi_dir(0)));
        float errorInPixel = 0.2f + 0.2f * (a+b) / a; // WO kommt das her? Scheint nicht zu NGF zu passen !

        if( epi_dir(0)*epi_dir(0)>epi_dir(1)*epi_dir(1) )
        {
            idepth_min = (pr[2]*(bestKp(0)-errorInPixel*epi_dir(0)) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2]*(bestKp(0)-errorInPixel*epi_dir(0)));
            idepth_max = (pr[2]*(bestKp(0)+errorInPixel*epi_dir(0)) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2]*(bestKp(0)+errorInPixel*epi_dir(0)));
        }
        else
        {
            idepth_min = (pr[2]*(bestKp(1)-errorInPixel*epi_dir(1)) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2]*(bestKp(1)-errorInPixel*epi_dir(1)));
            idepth_max = (pr[2]*(bestKp(1)+errorInPixel*epi_dir(1)) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2]*(bestKp(1)+errorInPixel*epi_dir(1)));
        }
        if(idepth_min > idepth_max) std::swap<float>(idepth_min, idepth_max);

        lastTraceUV = bestKp;
        lastTraceStatus = ImmaturePointStatus::IPS_GOOD;
    }
}

void ImmaturePoint::searchEpiLineNgf (ImageDataPtr frame, const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt )
{
    if(lastTraceStatus == ImmaturePointStatus::IPS_OOB || lastTraceStatus == ImmaturePointStatus::IPS_DELETED ) return;

    ++numTraced;

    if ( !frame )
    {
        lastTraceStatus = ImmaturePointStatus::IPS_SKIPPED;
        return;
    }
    if ( frame->dI == nullptr )
        copyImage ( frame );

    float idepth_mean = (idepth_min + idepth_max)*0.5;
    Vec3f pr = hostToFrame_KRKi * Vec3f(u,v, 1);
    Vec3f ptpMean = pr + hostToFrame_Kt*idepth_mean;
    Vec3f ptpMin = pr + hostToFrame_Kt*idepth_min;
    Vec3f ptpMax = pr + hostToFrame_Kt*idepth_max;
    Vec2f uvMean = ptpMean.hnormalized();
    Vec2f uvMin = ptpMin.hnormalized();
    Vec2f uvMax = ptpMax.hnormalized();

    rotatePattern( hostToFrame_KRKi );

    Vec2f epi_line = uvMax - uvMin;
    float norm_epi = std::max<float>(1e-5f,epi_line.norm());
    Vec2f epi_dir = epi_line / norm_epi;
    const float  half_length = 0.5f * norm_epi;

    Vec2f bestKp;
    float bestEnergy = 1e10;

    for(float l = -half_length; l <= half_length; l += 0.7f)
    {
        float energy = 0;
        Vec2f kp = uvMean + l*epi_dir;

        if( !kp.allFinite() || ( kp(0) >= (wG[0]-10) )  || ( kp(1) >= (hG[0]-10) ) || ( kp(0) < 10 ) || ( kp(1) < 10) )
        {
          continue;
        }

        for(int idx=0;idx<patternNum; ++idx)
        {
            if ( skipZero[idx] ) continue;
            const Vec3f hitColor = getInterpolatedElement33BiCub(frame->dI, (float)(kp(0)+rotatetPattern[idx][0]), (float)(kp(1)+rotatetPattern[idx][1]), wG[0]);
            if( !hitColor.allFinite() ) {energy+=1e5; continue;}
            Vec2f hitD = hitColor.tail<2>();
            hitD /= sqrt(hitD.squaredNorm()+settings_Eta);
            const float nn = hitD.dot(colorD[idx]);
            const float residual = std::max<float>(0.f,std::min<float>(1.f,nn < 0 ? 1.f : 1-nn ));// uni modal ngf
            //const float residual = std::max<float>(0.f,std::min<float>(2.f, 1-nn ));// uni modal ngf
            //const float residual = std::max<float>(0.f,std::min<float>(1.f,1.f-nn*nn)); // original ngf residual
            //float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
            //energy += hw *residual*residual*(2-hw);
            energy += residual;
        }
        if ( energy < bestEnergy )
        {
            bestKp = kp; bestEnergy = energy;
        }
    }

    if( bestEnergy > energyTHngf * 1.2f )
//    if ( bestEnergy > 100 )
    {
        lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
    }
    else
    {
        float a = (Vec2f(epi_dir(0),epi_dir(1)).transpose() * gradH * Vec2f(epi_dir(0),epi_dir(1)));
        float b = (Vec2f(epi_dir(1),-epi_dir(0)).transpose() * gradH * Vec2f(epi_dir(1),-epi_dir(0)));
        float errorInPixel = 0.2f + 0.2f * (a+b) / a; // WO kommt das her? Scheint nicht zu NGF zu passen !

        if( epi_dir(0)*epi_dir(0)>epi_dir(1)*epi_dir(1) )
        {
            idepth_min = (pr[2]*(bestKp(0)-errorInPixel*epi_dir(0)) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2]*(bestKp(0)-errorInPixel*epi_dir(0)));
            idepth_max = (pr[2]*(bestKp(0)+errorInPixel*epi_dir(0)) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2]*(bestKp(0)+errorInPixel*epi_dir(0)));
        }
        else
        {
            idepth_min = (pr[2]*(bestKp(1)-errorInPixel*epi_dir(1)) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2]*(bestKp(1)-errorInPixel*epi_dir(1)));
            idepth_max = (pr[2]*(bestKp(1)+errorInPixel*epi_dir(1)) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2]*(bestKp(1)+errorInPixel*epi_dir(1)));
        }
        if(idepth_min > idepth_max) std::swap<float>(idepth_min, idepth_max);

        lastTraceUV = bestKp;
        lastTraceStatus = ImmaturePointStatus::IPS_GOOD;
    }
}

void ImmaturePoint::searchEpiLineBca (ImageDataPtr frame, const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt, const Vec2f& hostToFrame_affine  )
{
    if(lastTraceStatus == ImmaturePointStatus::IPS_OOB || lastTraceStatus == ImmaturePointStatus::IPS_DELETED ) return;

    ++numTraced;

    if ( !frame )
    {
        lastTraceStatus = ImmaturePointStatus::IPS_SKIPPED;
        return;
    }
    if ( frame->dI == nullptr )
        copyImage ( frame );

    float idepth_mean = (idepth_min + idepth_max)*0.5;
    Vec3f pr = hostToFrame_KRKi * Vec3f(u,v, 1);
    Vec3f ptpMean = pr + hostToFrame_Kt*idepth_mean;
    Vec3f ptpMin = pr + hostToFrame_Kt*idepth_min;
    Vec3f ptpMax = pr + hostToFrame_Kt*idepth_max;
    Vec2f uvMean = ptpMean.hnormalized();
    Vec2f uvMin = ptpMin.hnormalized();
    Vec2f uvMax = ptpMax.hnormalized();

    rotatePattern( hostToFrame_KRKi );

    Vec2f epi_line = uvMax - uvMin;
    float norm_epi = std::max<float>(1e-5f,epi_line.norm());
    Vec2f epi_dir = epi_line / norm_epi;
    const float  half_length = 0.5f * norm_epi;

    Vec2f bestKp;
    float bestEnergy = 1e10;

    for(float l = -half_length; l <= half_length; l += 0.7f)
    {
        float energy = 0;
        Vec2f kp = uvMean + l*epi_dir;

        if( !kp.allFinite() || ( kp(0) >= (wG[0]-10) )  || ( kp(1) >= (hG[0]-10) ) || ( kp(0) < 10 ) || ( kp(1) < 10) )
        {
          continue;
        }

        for(int idx=0;idx<patternNum; ++idx)
        {
            //float hitColor = getInterpolatedElement31(frame->dI, (float)(kp(0)+rotatetPattern[idx][0]), (float)(kp(1)+rotatetPattern[idx][1]), wG[0]);
            float hitColor = getInterpolatedElement33BiCub(frame->dI, (float)(kp(0)+rotatetPattern[idx][0]), (float)(kp(1)+rotatetPattern[idx][1]), wG[0])[0];
            if(!std::isfinite(hitColor)) {energy-=1e5; continue;}

            const float residual = hitColor - (float)(hostToFrame_affine[0] * color[idx] + hostToFrame_affine[1]);

            float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
            energy += hw *residual*residual*(2-hw);
        }
        if ( energy < bestEnergy )
        {
            bestKp = kp; bestEnergy = energy;
        }
    }

    if ( bestEnergy > energyTH * 1.2f )
    {
        lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
    }
    else
    {
        float a = (Vec2f(epi_dir(0),epi_dir(1)).transpose() * gradH * Vec2f(epi_dir(0),epi_dir(1)));
        float b = (Vec2f(epi_dir(1),-epi_dir(0)).transpose() * gradH * Vec2f(epi_dir(1),-epi_dir(0)));
        float errorInPixel = 0.2f + 0.2f * (a+b) / a; // WO kommt das her? Scheint nicht zu NGF zu passen !

        if( epi_dir(0)*epi_dir(0)>epi_dir(1)*epi_dir(1) )
        {
            idepth_min = (pr[2]*(bestKp(0)-errorInPixel*epi_dir(0)) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2]*(bestKp(0)-errorInPixel*epi_dir(0)));
            idepth_max = (pr[2]*(bestKp(0)+errorInPixel*epi_dir(0)) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2]*(bestKp(0)+errorInPixel*epi_dir(0)));
        }
        else
        {
            idepth_min = (pr[2]*(bestKp(1)-errorInPixel*epi_dir(1)) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2]*(bestKp(1)-errorInPixel*epi_dir(1)));
            idepth_max = (pr[2]*(bestKp(1)+errorInPixel*epi_dir(1)) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2]*(bestKp(1)+errorInPixel*epi_dir(1)));
        }
        if(idepth_min > idepth_max) std::swap<float>(idepth_min, idepth_max);

        lastTraceUV = bestKp;
        lastTraceStatus = ImmaturePointStatus::IPS_GOOD;
    }
}

ImmaturePointStatus ImmaturePoint::evaluate ( ImageDataPtr frame, const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt, const Vec2f& hostToFrame_affine )
{
    float energy = 0;
    const float idepth = (idepth_min + idepth_max) / 2;
    const Vec3f p = hostToFrame_KRKi * Vec3f(u,v,1) + hostToFrame_Kt * idepth;
    const Vec2f kp = p.hnormalized();
    if ( kp(0) > 3 && kp(0) < wM3G && kp(1) > 3 && kp(1) < hM3G )
    {
        const Mat22f Rplane = hostToFrame_KRKi.topLeftCorner<2,2>();
        Vec2f rotatetPattern[MAX_RES_PER_POINT];
        for(int idx=0;idx<patternNum;++idx)
            rotatetPattern[idx] = Rplane * Vec2f(patternP[idx][0], patternP[idx][1]);

        energy=0;
        for(int idx=0;idx<patternNum; ++idx)
        {
            //Vec3f hit = getInterpolatedElement33BiCub(frame->dI, (float)(ptx+rotatetPattern[idx][0]), (float)(pty+rotatetPattern[idx][1]), wG[0]);
            //float hitColor = hit[0];
            float hitColor = getInterpolatedElement31(frame->dI, (float)(kp(0)+rotatetPattern[idx][0]), (float)(kp(1)+rotatetPattern[idx][1]), wG[0]);
            if(!std::isfinite(hitColor)) {energy+=1e5; continue;}
            float residual = hitColor - (float)(hostToFrame_affine[0] * color[idx] + hostToFrame_affine[1]);
            //float residual = hitColor - float(color[idx]);
            float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
            energy += hw *residual*residual*(2-hw);
        }
    }
    else
    {
        energy = 1e5;
    }

    if( energy < 1.2 * lastTraceEnergy )
    {
        return ImmaturePointStatus::IPS_GOOD;
    }
    else
    {
        return ImmaturePointStatus::IPS_OUTLIER;
    }
}


ImmaturePointStatus ImmaturePoint::evaluateNgf ( ImageDataPtr frame, const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt )
{
    float energy = 0;
    const float idepth = (idepth_min + idepth_max) / 2;
    const Vec3f p = hostToFrame_KRKi * Vec3f(u,v,1) + hostToFrame_Kt * idepth;
    const Vec2f kp = p.hnormalized();
    if ( kp(0) > 3 && kp(0) < wM3G && kp(1) > 3 && kp(1) < hM3G )
    {
        const Mat22f Rplane = hostToFrame_KRKi.topLeftCorner<2,2>();
        Vec2f rotatetPattern[MAX_RES_PER_POINT];
        for(int idx=0;idx<patternNum;++idx)
            rotatetPattern[idx] = Rplane * Vec2f(patternP[idx][0], patternP[idx][1]);

        bool skipZero [patternNum];
        for(int idx=0;idx<patternNum;idx++)
            skipZero[idx] = colorD[idx].isApprox(Eigen::Vector2f::Zero(),settings_Eta);

        energy=0;
        for(int idx=0;idx<patternNum; ++idx)
        {
            if ( skipZero[idx] ) continue;
            const Vec3f hitColor = getInterpolatedElement33BiCub(frame->dI, (float)(kp(0)+rotatetPattern[idx][0]), (float)(kp(1)+rotatetPattern[idx][1]), wG[0]);
            if( !hitColor.allFinite() ) {energy+=1e5; continue;}
            Vec2f hitD = hitColor.tail<2>();
            hitD /= sqrt(hitD.squaredNorm()+settings_Eta);
            const float nn = hitD.dot(colorD[idx]);
            const float residual = std::max<float>(0.f,std::min<float>(1.f,nn < 0 ? 1.f : 1-nn ));// uni modal ngf
            //const float residual = std::max<float>(0.f,std::min<float>(1.f,1.f-nn*nn)); // original ngf residual
            float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
            energy += hw *residual*residual*(2-hw);
            //energy += residual;
        }
    }
    else
    {
        energy = 1e5;
    }

    if( energy < 1.2 * lastTraceEnergy )
    {
        return ImmaturePointStatus::IPS_GOOD;
    }
    else
    {
        return ImmaturePointStatus::IPS_OUTLIER;
    }
}


Eigen::VectorXd ImmaturePoint::getGTColor ( ImageDataPtr frame , const Mat33f & hostToFrame_KRKi, const Vec3f & hostToFrame_Kt, Eigen::VectorXd & refColor )
{
    if ( frame->dI == nullptr )
        copyImage ( frame );
    Eigen::VectorXd gtColor = Eigen::VectorXd::Zero(patternNum,1);
    refColor = Eigen::VectorXd::Zero(patternNum, 1);

    if ( idepth_GT > 0 )
    {
        const Vec3f p = hostToFrame_KRKi * Vec3f(u,v,1) + hostToFrame_Kt*idepth_GT;
        kp_GT = p.hnormalized();
        if ( kp_GT(0) > 3 && kp_GT(0) < wM3G && kp_GT(1) > 3 && kp_GT(1) < hM3G )
        {
            const Mat22f Rplane = hostToFrame_KRKi.topLeftCorner<2,2>();
            Vec2f rotatetPattern[MAX_RES_PER_POINT];
            for(int idx=0;idx<patternNum;++idx)
                rotatetPattern[idx] = Rplane * Vec2f(patternP[idx][0], patternP[idx][1]);

            for(int idx=0;idx<patternNum;++idx)
            {
                gtColor(idx) = getInterpolatedElement31(frame->dI, (float)(kp_GT(0)+rotatetPattern[idx][0]), (float)(kp_GT(1)+rotatetPattern[idx][1]), wG[0]);
                refColor(idx) = color[idx];
            }
        }
    }
    return gtColor;
}


float ImmaturePoint::doRefinement ( ImageDataPtr frame, const Vec2f & hostToFrame_affine, const float & dx, const float & dy, float & bestU, float & bestV, const bool debugPrint )
{
    float bestEnergy = 1e10;
    // ============== do GN optimization ===================
    float uBak=bestU, vBak=bestV, gnstepsize=1, stepBack=0;
    if(setting_trace_GNIterations>0) bestEnergy = 1e5;
    int gnStepsGood=0, gnStepsBad=0;
    for(int it=0;it<setting_trace_GNIterations; ++it)
    {
        float H = 1, b=0, energy=0;
        for(int idx=0;idx<patternNum;idx++)
        {
            float x = (float)(bestU+rotatetPattern[idx][0]);
            float y = (float)(bestV+rotatetPattern[idx][1]);

            if ( x < 3 || x > wM3G || y < 3 || y > hM3G ) {energy+=1e5; continue;}
            Vec3f hitColor = getInterpolatedElement33BiLin(frame->dI, x, y, wG[0]);

            if(!std::isfinite((float)hitColor[0])) {energy+=1e5; continue;}
            float residual = hitColor[0] - (hostToFrame_affine[0] * color[idx] + hostToFrame_affine[1]);
            float dResdDist = dx*hitColor[1] + dy*hitColor[2];
            float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

            H += hw*dResdDist*dResdDist;
            b += hw*residual*dResdDist;
            energy += weights[idx]*weights[idx]*hw *residual*residual*(2-hw);
        }


        if(energy > bestEnergy)
        {
            ++gnStepsBad;

            // do a smaller step from old point.
            stepBack*=0.5;
            bestU = uBak + stepBack*dx;
            bestV = vBak + stepBack*dy;
            if(debugPrint)
                printf("GN BACK %d: E %f, H %f, b %f. id-step %f. UV %f %f -> %f %f.\n",
                       it, energy, H, b, stepBack,
                       uBak, vBak, bestU, bestV);
        }
        else
        {
            ++gnStepsGood;

            float step = -gnstepsize*b/H;
            if(step < -0.5) step = -0.5;
            else if(step > 0.5) step=0.5;

            if(!std::isfinite(step)) step=0;

            uBak=bestU;
            vBak=bestV;
            stepBack=step;

            bestU += step*dx;
            bestV += step*dy;
            bestEnergy = energy;

            if(debugPrint)
                printf("GN step %d: E %f, H %f, b %f. id-step %f. UV %f %f -> %f %f.\n",
                       it, energy, H, b, step,
                       uBak, vBak, bestU, bestV);
        }

        if(fabsf(stepBack) < setting_trace_GNThreshold) break;
    }
    return bestEnergy;
}

float ImmaturePoint::doRefinementNgf ( ImageDataPtr frame, const float & dx, const float & dy, float & bestU, float & bestV, const bool debugPrint )
{
    float bestEnergy = 1e10;
    // ============== do GN optimization ===================
    Vec2f d2f = Vec2f::Ones(); //Mat22f dH2f;
    float uBak=bestU, vBak=bestV, gnstepsize=1, stepBack=0;
    if(setting_trace_GNIterations>0) bestEnergy = 1e5;
    int gnStepsGood=0, gnStepsBad=0;
    for(int it=0;it<setting_trace_GNIterations;it++)
    {
        float H = 1, b=0, energy=0;
        for(int idx=0;idx<patternNum;idx++)
        {
            if ( skipZero[idx] ) continue;
            float x = bestU + rotatetPattern[idx][0];
            float y = bestV + rotatetPattern[idx][1];
            if ( x <= 4 || y <= 4 || x >= (wG[0]-4) || y >= (hG[0]-4) ) { energy+=1e5; break; }

            //const Vec3f hitColor = getInterpolatedElement33BiCub3(frame->dI, x, y, wG[0],dH2f);
            const Vec3f hitColor = getInterpolatedElement33BiCub2(frame->dI, x, y, wG[0], d2f );
            if( !hitColor.allFinite() ) {energy+=1e5; continue;}
            Vec2f hitD = hitColor.tail<2>();
            float iNormG = 1./sqrt(hitD.squaredNorm()+settings_Eta);
            hitD *= iNormG;
            const float nn = hitD.dot(colorD[idx]);
            const float residual = std::max<float>(0.f,std::min<float>(1.f,nn < 0 ? 1.f : 1-nn )); // uni modal
            //const float residual = std::max<float>(0.f,std::min<float>(2.f, 1-nn )); // uni modal
            //const float residual = std::max<float>(0.f,std::min<float>(1.f, 1-nn*nn ));

    //            float xp1 = (x+1), xm1 = (x-1), yp1 = (y+1), ym1 = (y-1);
    //            //if ( xm1 > 4 && xp1 < (wG[0]-4) && ym1 > 4 && yp1 < (hG[0]-4) )
    //            {
    //                const Vec3f Ixp1 = getInterpolatedElement33BiCub(frame->dI, xp1, y, wG[0]);
    //                const Vec3f Ixm1 = getInterpolatedElement33BiCub(frame->dI, xm1, y, wG[0]);
    //                const Vec3f Iyp1 = getInterpolatedElement33BiCub(frame->dI, x, yp1, wG[0]);
    //                const Vec3f Iym1 = getInterpolatedElement33BiCub(frame->dI, x, ym1, wG[0]);
    //                const Vec3f Ixp1yp1 = getInterpolatedElement33BiCub(frame->dI, xp1, yp1, wG[0]);
    //                const Vec3f Ixm1ym1 = getInterpolatedElement33BiCub(frame->dI, xm1, ym1, wG[0]);
    ////                d2f[0] = (Ixp1[0] - 2 * hitColor[0] + Ixm1[0]);
    ////                d2f[1] = (Iyp1[0] - 2 * hitColor[0] + Iym1[0]);
    //                dH2f(0,0) = (Ixp1[0] - 2 * hitColor[0] + Ixm1[0]);
    //                dH2f(1,1) = (Iyp1[0] - 2 * hitColor[0] + Iym1[0]);
    //                dH2f(0,1) = .5f * (Ixp1yp1[0]-Ixp1[0]-Iyp1[0] + 2*hitColor[0] -Ixm1[0]-Iym1[0] + Ixm1ym1[0]);
    //                dH2f(1,0) = dH2f(0,1);
    //            }

            //Vec2f dn = - (( colorD[idx].transpose() - nn*hitD.transpose() ) * dH2f * iNormG).transpose();
            Vec2f dn = - (( colorD[idx].transpose() - nn*hitD.transpose() ) * d2f.asDiagonal() * iNormG).transpose(); // uni modal
            //Vec2f dn = -2*nn*(( colorD[idx].transpose() - nn*hitD.transpose() ) * d2f.asDiagonal() * iNormG).transpose(); // original ngf

            float dResdDist = dx*dn[0] + dy*dn[1];
            float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

            H += hw*dResdDist*dResdDist;
            b += hw*residual*dResdDist;
            energy += weights[idx]*weights[idx]*hw *residual*residual*(2-hw);
        }

        if(energy > bestEnergy)
        {
            ++gnStepsBad;

            // do a smaller step from old point.
            stepBack*=0.5;
            bestU = uBak + stepBack*dx;
            bestV = vBak + stepBack*dy;
            if(debugPrint)
                printf("GN BACK %d: E %f, H %f, b %f. id-step %f. UV %f %f -> %f %f.\n",
                       it, energy, H, b, stepBack,
                       uBak, vBak, bestU, bestV);
        }
        else
        {
            ++gnStepsGood;

            float step = -gnstepsize*b/H;
            if(step < -0.5) step = -0.5;
            else if(step > 0.5) step=0.5;

            if(!std::isfinite(step)) step=0;

            uBak=bestU;
            vBak=bestV;
            stepBack=step;

            bestU += step*dx;
            bestV += step*dy;
            bestEnergy = energy;

            if(debugPrint)
                printf("GN step %d: E %f, H %f, b %f. id-step %f. UV %f %f -> %f %f.\n",
                       it, energy, H, b, step,
                       uBak, vBak, bestU, bestV);
        }

        if(fabsf(stepBack) < setting_trace_GNThreshold) break;
    }
    return bestEnergy;
}


bool projectPoint(
                const float &u_pt,const float &v_pt,
                const float &idepth,
                const int &dx, const int &dy,
                const Mat33f & K,
                const Mat33f &R, const Vec3f &t,
                float &drescale, float &u, float &v,
                float &Ku, float &Kv, Vec3f &KliP, float &new_idepth)
{
        KliP = K.inverse() * Vec3f( (u_pt+dx), (v_pt+dy), 1);

        Vec3f ptp = R * KliP + t*idepth;
        drescale = 1.0f/ptp[2];
        new_idepth = idepth*drescale;

        if(!(drescale>0)) return false;

        u = ptp[0] * drescale;
        v = ptp[1] * drescale;
        Ku = u*K(0,0) + K(0,2);
        Kv = v*K(1,1) + K(1,2);

        return Ku>1.1f && Kv>1.1f && Ku<wM3G && Kv<hM3G;
}
inline float derive_idepth(
                const Vec3f &t, const float &u, const float &v,
                //const int &dx, const int &dy,
                const float &dxInterp, const float &dyInterp, const float &drescale)
{
        return (dxInterp*drescale * (t[0]-t[2]*u) + dyInterp*drescale * (t[1]-t[2]*v))*SCALE_IDEPTH;
}


double ImmaturePoint::linearizeResidual( const float outlierTHSlack,
        ImmaturePointTemporaryResidual * tmpRes,
        float &Hdd, float &bd,
        float idepth)
{
    if(tmpRes->state_state == ResState::OOB)
    { tmpRes->state_NewState = ResState::OOB; return tmpRes->state_energy; }

    // check OOB due to scale angle change.

    float energyLeft=0;
    const Eigen::Vector3f* dIl = tmpRes->target->dI;
    const Mat33f PRE_RTll = tmpRes->target->m_pose_cr.linear().cast<float>();
    const Vec3f PRE_tTll = tmpRes->target->m_pose_cr.translation().cast<float>();

    const Mat33f K = tmpRes->target->K_c[0].cast<float>();
    const float & fx = K(0,0);
    const float & fy = K(1,1);
    for(int idx=0;idx<patternNum;idx++)
    {
        int dx = patternP[idx][0];
        int dy = patternP[idx][1];

        float drescale, u, v, new_idepth;
        float Ku, Kv;
        Vec3f KliP;

        if(!projectPoint(this->u,this->v, idepth, dx, dy, K, PRE_RTll,PRE_tTll, drescale, u, v, Ku, Kv, KliP, new_idepth))
        {tmpRes->state_NewState = ResState::OOB; return tmpRes->state_energy;}


        Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));

        if(!std::isfinite((float)hitColor[0])) {tmpRes->state_NewState = ResState::OOB; return tmpRes->state_energy;}
        float residual = hitColor[0] - color[idx];//(affLL[0] * color[idx] + affLL[1]);

        float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
        energyLeft += weights[idx]*weights[idx]*hw *residual*residual*(2-hw);

        // depth derivatives.
        float dxInterp = hitColor[1]*fx;
        float dyInterp = hitColor[2]*fy;
        float d_idepth = derive_idepth(PRE_tTll, u, v,
                                       //dx, dy,
                                       dxInterp, dyInterp, drescale);

        hw *= weights[idx]*weights[idx];

        Hdd += (hw*d_idepth)*d_idepth;
        bd += (hw*residual)*d_idepth;
    }

    if(energyLeft > energyTH*outlierTHSlack)
    {
        energyLeft = energyTH*outlierTHSlack;
        tmpRes->state_NewState = ResState::OUTLIER;
    }
    else
    {
        tmpRes->state_NewState = ResState::IN;
    }

    tmpRes->state_NewEnergy = energyLeft;
    return energyLeft;
}

PointPtr optimizeIdepth( ImmaturePoint * point, ImmaturePointTemporaryResidual* residuals, std::vector<ImageDataPtr> & frameHessians )
{
    int minObs = 2;
    int nres = 0;
    for(ImageDataPtr fh : frameHessians)
    {
        if(fh != point->host)
        {
            residuals[nres].state_NewEnergy = residuals[nres].state_energy = 0;
            residuals[nres].state_NewState = ResState::OUTLIER;
            residuals[nres].state_state = ResState::IN;
            residuals[nres].target = fh;
            nres++;
        }
    }

    bool print = false;//rand()%50==0;

    float lastEnergy = 0;
    float lastHdd=0;
    float lastbd=0;
    float currentIdepth=(point->idepth_max+point->idepth_min)*0.5f;

    for(int i=0;i<nres;i++)
    {
        lastEnergy += point->linearizeResidual(1000, residuals+i,lastHdd, lastbd, currentIdepth);
        residuals[i].state_state = residuals[i].state_NewState;
        residuals[i].state_energy = residuals[i].state_NewEnergy;
    }

    if(!std::isfinite(lastEnergy) || lastHdd < setting_minIdepthH_act)
    {
        if(print)
            printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
                   nres, lastHdd, lastEnergy);
        return 0;
    }

    if(print) printf("Activate point. %d residuals. H=%f. Initial Energy: %f. Initial Id=%f\n" ,
                     nres, lastHdd,lastEnergy,currentIdepth);

    float lambda = 0.1;
    for(int iteration=0;iteration<setting_GNItsOnPointActivation;iteration++)
    {
        float H = lastHdd;
        H *= 1+lambda;
        float step = (1.0/H) * lastbd;
        float newIdepth = currentIdepth - step;

        float newHdd=0; float newbd=0; float newEnergy=0;
        for(int i=0;i<nres;i++)
            newEnergy += point->linearizeResidual(1, residuals+i,newHdd, newbd, newIdepth);

        if(!std::isfinite(lastEnergy) || newHdd < setting_minIdepthH_act)
        {
            if(print) printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
                             nres,
                             newHdd,
                             lastEnergy);
            return 0;
        }

        if(print) printf("%s %d (L %.2f) %s: %f -> %f (idepth %f)!\n",
                         (true || newEnergy < lastEnergy) ? "ACCEPT" : "REJECT",
                         iteration,
                         log10(lambda),
                         "",
                         lastEnergy, newEnergy, newIdepth);

        if(newEnergy < lastEnergy)
        {
            currentIdepth = newIdepth;
            lastHdd = newHdd;
            lastbd = newbd;
            lastEnergy = newEnergy;
            for(int i=0;i<nres;i++)
            {
                residuals[i].state_state = residuals[i].state_NewState;
                residuals[i].state_energy = residuals[i].state_NewEnergy;
            }

            lambda *= 0.5;
        }
        else
        {
            lambda *= 5;
        }

        if(fabsf(step) < 0.0001*currentIdepth)
            break;
    }

    if(!std::isfinite(currentIdepth))
    {
        printf("MAJOR ERROR! point idepth is nan after initialization (%f).\n", currentIdepth);
        return nullptr;
    }


    int numGoodRes=0;
    for(int i=0;i<nres;i++)
        if(residuals[i].state_state == ResState::IN) numGoodRes++;

    if(numGoodRes < minObs)
    {
        if(print) printf("OptPoint: OUTLIER!\n");
        return nullptr;
    }


    PointPtr p ( new Point( point ) );
    if(!std::isfinite(p->energyTH))
        return nullptr;

    //        p->lastResiduals[0].first = 0;
    //        p->lastResiduals[0].second = ResState::OOB;
    //        p->lastResiduals[1].first = 0;
    //        p->lastResiduals[1].second = ResState::OOB;
    p->setIdepthZero(currentIdepth);
    p->setIdepth(currentIdepth);
    p->setPointStatus(Point::ACTIVE);

    //        for(int i=0;i<nres;i++)
    //                if(residuals[i].state_state == ResState::IN)
    //                {
    //                        PointFrameResidual* r = new PointFrameResidual(p, p->host, residuals[i].target);
    //                        r->state_NewEnergy = r->state_energy = 0;
    //                        r->state_NewState = ResState::OUTLIER;
    //                        r->setState(ResState::IN);
    //                        p->residuals.push_back(r);

    //                        if(r->target == frameHessians.back())
    //                        {
    //                                p->lastResiduals[0].first = r;
    //                                p->lastResiduals[0].second = ResState::IN;
    //                        }
    //                        else if(r->target == (frameHessians.size()<2 ? 0 : frameHessians[frameHessians.size()-2]))
    //                        {
    //                                p->lastResiduals[1].first = r;
    //                                p->lastResiduals[1].second = ResState::IN;
    //                        }
    //                }

    //        if(print) printf("point activated!\n");

    //        statistics_numActivatedPoints++;
    return p;
}


PointPtr ImmaturePoint::optimize( std::vector<ImageDataPtr> & frameHessians )
{
    //std::cout << "FrameHessians: " << frameHessians.size() << std::endl;

    ImmaturePointTemporaryResidual* residuals = new ImmaturePointTemporaryResidual[frameHessians.size()];
    //std::cout << "got residuals."<< std::endl;

    PointPtr p = optimizeIdepth( this, residuals, frameHessians );
    //std::cout << "delete residuals."<< std::endl;

    //PointPtr p = optimize( this, residuals, frameHessians );
    delete[] residuals;
    return p;
}


Point::Point(ImmaturePoint * rawPoint)
{
    host = rawPoint->host;
    hasDepthPrior=false;

    idepth_hessian=0;
    maxRelBaseline=0;
    numGoodResiduals=0;

    // set static values & initialization.
    u = rawPoint->u;
    v = rawPoint->v;
    assert(std::isfinite(rawPoint->idepth_max));
    //idepth_init = rawPoint->idepth_GT;

    setIdepthScaled((rawPoint->idepth_max + rawPoint->idepth_min)*0.5);
    setPointStatus(Point::INACTIVE);

    int n = patternNum;
    memcpy(color, rawPoint->color, sizeof(float)*n);
    memcpy(weights, rawPoint->weights, sizeof(float)*n);
    energyTH = rawPoint->energyTH;
}


void Point::release()
{
    //for(unsigned int i=0;i<residuals.size();++i) delete residuals[i];
    //residuals.clear();
}
