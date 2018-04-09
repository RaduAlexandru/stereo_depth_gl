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

#ifndef DEPTH_POINT_H_
#define DEPTH_POINT_H_

#include "types.h"

//#include "util/NumType.h" 
//#include "FullSystem/HessianBlocks.h"

const int MAX_RES_PER_POINT = 10;
typedef Eigen::Matrix2f Mat22f;
typedef Eigen::Matrix2d Mat22d;
typedef Eigen::Matrix3f Mat33f;
typedef Eigen::Vector2f Vec2f;
typedef Eigen::Vector3f Vec3f;
typedef Eigen::Vector2d Vec2d;
typedef Eigen::Vector3d Vec3d;

#define SCALE_IDEPTH (1.0f)
#define SCALE_IDEPTH_INVERSE (1.0f/SCALE_IDEPTH)

enum ResState {IN=0, OOB, OUTLIER};

 struct ImmaturePointTemporaryResidual
 {
 public:
        ResState state_state;
        double state_energy;
        ResState state_NewState;
        double state_NewEnergy;
        ImageDataPtr target;
 };

enum ImmaturePointStatus {
	IPS_GOOD=0,					// traced well and good
	IPS_OOB,					// OOB: end tracking & marginalize!
	IPS_OUTLIER,				// energy too high: if happens again: outlier!
	IPS_SKIPPED,				// traced well and good (but not actually traced).
	IPS_BADCONDITION,			// not traced because of bad condition.
        IPS_DELETED,                            // merged with other point or deleted
	IPS_UNINITIALIZED};			// not even traced once.

class Point;
typedef std::shared_ptr<Point> PointPtr;
class ImmaturePoint;
typedef std::shared_ptr<ImmaturePoint> ImmaturePointPtr;

class ImmaturePoint
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	// static values
	float color[MAX_RES_PER_POINT];
	float weights[MAX_RES_PER_POINT];
   
        Vec2f colorD[MAX_RES_PER_POINT];
        Vec2f colorGrad[MAX_RES_PER_POINT];

        Vec2f rotatetPattern[MAX_RES_PER_POINT];
        bool skipZero [MAX_RES_PER_POINT];

        float ncc_sum_templ;
        float ncc_const_templ;


	Mat22f gradH;
	Vec2f gradH_ev;
	Mat22f gradH_eig;
	float energyTH;
        float energyTHngf;
	float u,v;
        ImageDataPtr host;
	int idxInImmaturePoints;

	float quality;

        float idepth_min;
	float idepth_max;
        ImmaturePoint(int u_, int v_, ImageDataPtr host_, float depth_gt = 0 );
	~ImmaturePoint();


        ImmaturePointStatus evaluate ( ImageDataPtr frame, const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt, const Vec2f& hostToFrame_affine );
        ImmaturePointStatus evaluateNgf ( ImageDataPtr frame, const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt );


        float doRefinement ( ImageDataPtr frame, const Vec2f & hostToFrame_affine, const float & dx, const float & dy, float & bestU, float & bestV, const bool debugPrint = false );
        float doRefinementNgf ( ImageDataPtr frame, const float & dx, const float & dy, float & bestU, float & bestV, const bool debugPrint = false );
        void checkResult ( const Vec3f & hostToFrame_Kt, const Vec3f & pr, const float & errorInPixel, const float & dx, const float & dy, const float & uMin, const float & uMax, const float & vMin, const float & vMax, const float & bestU, const float & bestV, const float & bestEnergy, const bool debugPrint = false );
        void checkResultNgf ( const Vec3f & hostToFrame_Kt, const Vec3f & pr, const float & errorInPixel, const float & dx, const float & dy, const float & uMin, const float & uMax, const float & vMin, const float & vMax, const float & bestU, const float & bestV, const float & bestEnergy, const bool debugPrint = false );
        void checkResultNcc ( const Vec3f & hostToFrame_Kt, const Vec3f & pr, const float & errorInPixel, const float & dx, const float & dy, const float & uMin, const float & uMax, const float & vMin, const float & vMax, const float & bestU, const float & bestV, const float & bestEnergy, const bool debugPrint = false );
        void rotatePattern( const Mat33f & hostToFrame_KRKi );

        void searchEpiLineBca (ImageDataPtr frame,const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt , const Vec2f& hostToFrame_affine );
        void searchEpiLineNcc (ImageDataPtr frame,const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt );
        void searchEpiLineNgf (ImageDataPtr frame, const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt );

        ImmaturePointPtr traceOn( ImageDataPtr frame, const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt, const Vec2f &hostToFrame_affine, bool debugPrint=false);
        ImmaturePointPtr traceOnNgf(ImageDataPtr frame,const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt, bool debugPrint = false);
        ImmaturePointPtr traceOnNcc( ImageDataPtr frame, const Mat33f & hostToFrame_KRKi, const Vec3f & hostToFrame_Kt, bool debugPrint = false );
        Eigen::VectorXd getGTColor ( ImageDataPtr frame, const Mat33f & hostToFrame_KRKi, const Vec3f & hostToFrame_Kt, Eigen::VectorXd & refColor );

        PointPtr optimize( std::vector<ImageDataPtr> & frameHessians );

	ImmaturePointStatus lastTraceStatus;
	Vec2f lastTraceUV;
        Vec2f lastValidUV;
	float lastTracePixelInterval;
        float lastTraceEnergy;

	float idepth_GT;
        float idepthBCA_est = -1;
        float idepthNgf_est = -1;
        int bestIdxDSO = -1;
        int bestIdxNgf = -1;
        Vec2f kp_GT;
        int gtIdx = -1;

        ImmaturePointPtr child;
        int numTraced = 0;

        bool isContained ( const ImmaturePointPtr & otherPt ) const;

        double linearizeResidual( const float outlierTHSlack,
                        ImmaturePointTemporaryResidual* tmpRes,
                        float &Hdd, float &bd,
                        float idepth);
// 	float getdPixdd(
// 			CalibHessian *  HCalib,
// 			ImmaturePointTemporaryResidual* tmpRes,
// 			float idepth);
// 
// 	float calcResidual(
// 			CalibHessian *  HCalib, const float outlierTHSlack,
// 			ImmaturePointTemporaryResidual* tmpRes,
// 			float idepth);
};




// hessian component associated with one point.
struct Point
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // static values
    float color[MAX_RES_PER_POINT];			// colors in host frame
    float weights[MAX_RES_PER_POINT];		// host-weights for respective residuals.

    float u,v;
    int idx;
    float energyTH;
    ImageDataPtr host;
    bool hasDepthPrior;

    float idepth_scaled;
    float idepth_zero_scaled;
    float idepth_zero;
    float idepth;
    float step;
    float step_backup;
    float idepth_backup;

    float nullspaces_scale;
    float idepth_hessian;
    float maxRelBaseline;
    int numGoodResiduals;

    enum PtStatus {ACTIVE=0, INACTIVE, OUTLIER, OOB, MARGINALIZED};
    PtStatus status;

    inline void setPointStatus(PtStatus s) {status=s;}


    inline void setIdepth(float idepth) {
        this->idepth = idepth;
        this->idepth_scaled = SCALE_IDEPTH *    idepth;
    }
    inline void setIdepthScaled(float idepth_scaled) {
        this->idepth = SCALE_IDEPTH_INVERSE * idepth_scaled;
        this->idepth_scaled = idepth_scaled;
    }
    inline void setIdepthZero(float idepth) {
        idepth_zero = idepth;
        idepth_zero_scaled = SCALE_IDEPTH * idepth;
        nullspaces_scale = -(idepth*1.001 - idepth/1.001)*500;
    }

    //std::vector<PointFrameResidual*> residuals;					// only contains good residuals (not OOB and not OUTLIER). Arbitrary order.
    //std::pair<PointFrameResidual*, ResState> lastResiduals[2]; 	// contains information about residuals to the last two (!) frames. ([0] = latest, [1] = the one before).

    void release();
    Point( ImmaturePoint * rawPoint );
    inline ~Point() { release(); }


//    inline bool isOOB(const std::vector<ImageDataPtr>& toKeep, const std::vector<ImageDataPtr>& toMarg) const
//    {

//        int visInToMarg = 0;
//        for(PointFrameResidual* r : residuals)
//        {
//            if(r->state_state != ResState::IN) continue;
//            for(FrameHessian* k : toMarg)
//                if(r->target == k) visInToMarg++;
//        }
//        if((int)residuals.size() >= setting_minGoodActiveResForMarg &&
//                numGoodResiduals > setting_minGoodResForMarg+10 &&
//                (int)residuals.size()-visInToMarg < setting_minGoodActiveResForMarg)
//            return true;





//        if(lastResiduals[0].second == ResState::OOB) return true;
//        if(residuals.size() < 2) return false;
//        if(lastResiduals[0].second == ResState::OUTLIER && lastResiduals[1].second == ResState::OUTLIER) return true;
//        return false;
//    }


//    inline bool isInlierNew()
//    {
//        return (int)residuals.size() >= setting_minGoodActiveResForMarg
//                && numGoodResiduals >= setting_minGoodResForMarg;
//    }
};

#endif // DEPTH_POINT_H_
