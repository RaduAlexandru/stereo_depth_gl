#include "depth_filter.h"
#include <boost/math/distributions/normal.hpp>

Seed::Seed(ImmaturePointPtr ftr_, float depth_mean, float depth_min) :
    ftr(ftr_)
{
    reinit( depth_mean, depth_min );
    f = (ftr->host->K_c[0].inverse() * Eigen::Vector3d(ftr->u,ftr->v,1)).normalized();
}

void Seed::reinit( const float depth_mean, const float depth_min )
{
    a = (10);
    b = (10);
    mu = (1.0/depth_mean);
    z_range = (1.0/depth_min);
    sigma2 = (z_range*z_range/36);
}

DepthFilter::DepthFilter() :
    new_keyframe_min_depth_(0.0),
    new_keyframe_mean_depth_(0.0)
{}

bool firstRun = true;

void DepthFilter::initializeSeeds ( std::vector<ImmaturePointPtr> & invDepthPts )
{
    int numRows = -1, numCols = -1;
    seeds_.clear();
    seeds_.reserve(invDepthPts.size());
    for ( ImmaturePointPtr & invDepthPt : invDepthPts )
    {
        if ( !invDepthPt )
            continue;
        seeds_.push_back( Seed ( invDepthPt, 4, 0.1 ) );
        if ( numRows < 0 )
        {
            numRows = invDepthPt->host->grayImages[0].rows;
            numCols = invDepthPt->host->grayImages[0].cols;
        }
    }
    std::cout << "created seeds. " << int(seeds_.size()) << std::endl;
    const int numels = numRows * numCols;
    std::vector<Seed *> texSeed ( numels, nullptr );
    for ( size_t i = 0; i < seeds_.size(); ++i )
    {
        const int idx = seeds_[i].ftr->v * numCols + seeds_[i].ftr->u;
        texSeed[idx] = &(seeds_[i]);
    }
    std::cout << "created seed texture. "<< std::endl;
    for ( size_t i = 0; i < seeds_.size(); ++i )
    {
        const int u = seeds_[i].ftr->u;
        const int v = seeds_[i].ftr->v;
        const int up1 = (u+1);
        const int um1 = (u-1);
        const int vp1 = (v+1);
        const int vm1 = (v-1);

        const int idxLeft = v * numCols + um1;
        seeds_[i].left = ( idxLeft >= 0 && um1 >= 0 ) ? texSeed[idxLeft] : nullptr;

        const int idxRight = v * numCols + up1;
        seeds_[i].right = ( idxRight < int(texSeed.size()) && up1 < numCols ) ? texSeed[idxRight] : nullptr;

        const int idxAbove = vm1 * numCols + u;
        seeds_[i].above = ( idxAbove >= 0 && vm1 >= 0 ) ? texSeed[idxAbove] : nullptr;

        const int idxBelow = vp1 * numCols + u;
        seeds_[i].below = ( idxBelow < int(texSeed.size()) && vp1 < numRows ) ? texSeed[idxBelow] : nullptr;

        const int idxLeftUpper = vm1 * numCols + um1;
        seeds_[i].leftUpper = ( idxLeftUpper >= 0 && um1 >= 0 ) ? texSeed[idxLeftUpper] : nullptr;

        const int idxRightUpper = vm1 * numCols + up1;
        seeds_[i].rightUpper = ( idxRightUpper >= 0 && up1 < numCols ) ? texSeed[idxRightUpper] : nullptr;

        const int idxLeftLower= vp1 * numCols + um1;
        seeds_[i].leftLower = ( idxLeftLower < int(texSeed.size()) && vm1 >= 0 ) ? texSeed[idxLeftLower] : nullptr;

        const int idxRightLower = vp1 * numCols + up1;
        seeds_[i].rightLower = ( idxRightLower < int(texSeed.size()) && vp1 < numRows ) ? texSeed[idxRightLower] : nullptr;
    }
    std::cout << "initialized seeds. " << int(seeds_.size()) << std::endl;

}

float DepthFilter::getConvergencePercentage ( ) const
{
    int n_converged = 0;
    for ( auto it = seeds_.begin(); it!=seeds_.end(); ++it)
    {
        if ( it->converged )
            ++n_converged;
    }
    float perc = float(n_converged)/float(seeds_.size());
    std::cout << "converged: " << n_converged << " numSeeds: " << int ( seeds_.size() ) << " perc: " << perc << std::endl;
    return perc;
}

bool depthFromTriangulation(
    const Eigen::Affine3d & T_search_ref,
    const Eigen::Vector3d& f_ref,
    const Eigen::Vector3d& f_cur,
    double& depth)
{
  Eigen::Matrix<double,3,2> A; A << T_search_ref.linear() * f_ref, f_cur;
  const Eigen::Matrix2d AtA = A.transpose()*A;
  if(AtA.determinant() < 0.000001)
    return false;
  const Eigen::Vector2d depth2 = - AtA.inverse()*A.transpose()*T_search_ref.translation();
  depth = fabs(depth2[0]);
  return true;
}

void DepthFilter::updateSeeds( ImageDataPtr refImgPtr, ImageDataPtr curImgPtr, const Eigen::Affine3d & T_cur_ref, const Vec2f & affine_cr )
{
    // update only a limited number of seeds, because we don't have time to do it
    // for all the seeds in every frame!
    size_t n_updates=0, n_failed_matches=0, n_converged = 0, n_reinit_seeds = 0, wereGood = 0, n_seeds = seeds_.size();

    const Mat33f KRKi_cr = (curImgPtr->K_c[0] * T_cur_ref.linear() * refImgPtr->K_c[0].inverse()).cast<float>();
    const Vec3f Kt_cr = (curImgPtr->K_c[0] * T_cur_ref.translation()).cast<float>();
    //const Vec2f affine_cr  ( 1, 0 );


    const double focal_length = abs(curImgPtr->K_c[0](0,0));
    double px_noise = 1.0;
    double px_error_angle = atan(px_noise/(2.0*focal_length))*2.0; // law of chord (sehnensatz)
    const Eigen::Affine3d T_ref_cur = T_cur_ref.inverse();

    std::cout << " now iterating over seeds: " << n_seeds << std::endl;
    for ( auto it = seeds_.begin(); it!=seeds_.end(); ++it)
    {
        if ( it->converged )
            ++n_converged;

        // check if point is visible in the current image
        //SE3 T_ref_cur = it->ftr->frame->T_f_w_ * frame->T_f_w_.inverse();
        const Eigen::Vector3d xyz_f( T_cur_ref*(1.0/it->mu * it->f) );
        if(xyz_f.z() < 0.0)  {
            //++it; // behind the camera
            continue;
        }

        const Eigen::Vector2d kp_c = (curImgPtr->K_c[0] * xyz_f).hnormalized();
        if ( kp_c(0) < 0 || kp_c(0) >= curImgPtr->grayImages[0].cols || kp_c(1) < 0 || kp_c(1) >= curImgPtr->grayImages[0].rows )
        {        //if(!frame->cam_->isInFrame(frame->f2c(xyz_f).cast<int>())) {
            //++it; // point does not project in image
            continue;
        }

        // we are using inverse depth coordinates
        float z_inv_min = it->mu + sqrt(it->sigma2);
        float z_inv_max = std::max<float>(it->mu - sqrt(it->sigma2), 0.00000001f);

        if ( ! (it->ftr) )
            std::cerr << "why is the seeds ftr empty?" << std::endl;

        it->ftr->idepth_min = z_inv_min;
        it->ftr->idepth_max = z_inv_max;
        double idepth = -1;
        double z = 0;
        bool useTrace = false;
        if ( useTrace )
        {
            //it->ftr->traceOn( curImgPtr, KRKi_cr, Kt_cr, affine_cr );
            //it->ftr->traceOnNgf( curImgPtr, KRKi_cr, Kt_cr );
            it->ftr->traceOnNcc ( curImgPtr, KRKi_cr, Kt_cr );
            idepth = std::max<double>(1e-5,.5*(it->ftr->idepth_min+it->ftr->idepth_max));
            z = 1./idepth;
        }
        else
        {
            //it->ftr->searchEpiLineBca ( curImgPtr, KRKi_cr, Kt_cr, affine_cr ); // T_cur_ref, curImgPtr->K_c[0],
            //it->ftr->searchEpiLineNgf ( curImgPtr, KRKi_cr, Kt_cr ); // T_cur_ref, curImgPtr->K_c[0],
            it->ftr->searchEpiLineNcc ( curImgPtr, KRKi_cr, Kt_cr ); // T_cur_ref, curImgPtr->K_c[0],
            if( it->ftr->lastTraceStatus == ImmaturePointStatus::IPS_GOOD )
            {
                ++wereGood;
                idepth = std::max<double>(1e-5,.5*(it->ftr->idepth_min+it->ftr->idepth_max));
                z = 1./idepth;
//                Vec3d bestKpH = (it->ftr->host->K_c[0].inverse() * it->ftr->lastTraceUV.cast<double>().homogeneous()).normalized();
//                //Vec3d pt_xyz_ref = triangulatenNonLin ( it->f, bestKpH, T_ref_cur );
//                if ( ! depthFromTriangulation( T_cur_ref, it->f, bestKpH, z ) )
//                    continue;
//                if ( z < 0 )
//                    continue;
//                idepth = 1./z;
            }
        }
        //    if(!matcher_.findEpipolarMatchDirect(
        //        *it->ftr->frame, *frame, *it->ftr, 1.0/it->mu, 1.0/z_inv_min, 1.0/z_inv_max, z))

        if ( it->ftr->lastTraceStatus == ImmaturePointStatus::IPS_OOB  || it->ftr->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED )
        {
            // ++it;
            continue;
        }
        if ( !std::isfinite(idepth) || it->ftr->lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER || it->ftr->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION )
        {
            it->b++; // increase outlier probability when no match was found
            //++it;
            ++n_failed_matches;
            continue;
        }

        // compute tau
        double tau = computeTau(T_ref_cur, it->f, z, px_error_angle);
        double tau_inverse = 0.5 * (1.0/std::max<double>(0.0000001, z-tau) - 1.0/(z+tau));

        // update the estimate
        updateSeed(1./z, tau_inverse*tau_inverse, &*it);
        ++n_updates;

        const float eta_inlier = .6f;
        const float eta_outlier = .05f;
        // if E(inlier_ratio) > eta_inlier && sigma_sq < epsilon
        if( ((it->a / (it->a + it->b)) > eta_inlier) &&
                //(it->sigma2 < epsilon)
                (sqrt(it->sigma2) < it->z_range/options_.seed_convergence_sigma2_thresh)
                )
        {
            it->is_outlier = false;
            // The seed converged
            //    dev_ptr->convergence->atXY(x, y) = ConvergenceStates::CONVERGED;
        }
        // The seed failed to converge
        else if((it->a-1) / (it->a + it->b - 2) < eta_outlier)
        {
            it->is_outlier = true;
            it->reinit();
            ++n_reinit_seeds;
        }


        // if the seed has converged, we initialize a new candidate point and remove the seed
        if(sqrt(it->sigma2) < it->z_range/options_.seed_convergence_sigma2_thresh)
        {
            it->converged = true;
            //assert(it->ftr->point == NULL); // TODO this should not happen anymore
            //      Vector3d xyz_world(it->ftr->frame->T_f_w_.inverse() * (it->ftr->f * (1.0/it->mu)));
            //      Point* point = new Point(xyz_world, it->ftr);
            //      it->ftr->point = point;
            //      {
            //        seed_converged_cb_(point, it->sigma2); // put in candidate list
            //      }
            //      it = seeds_.erase(it);
        }
        //    else if(isnan(z_inv_min))
        //    {
        //      //SVO_WARN_STREAM("z_min is NaN");
        //      it = seeds_.erase(it);
        //    }
        //else
        //++it;
    }

    int n_updates_reg = 0, goodOnes = 0, skipped = 0, out = 0, bad = 0, oob = 0;
    int null = 0, range = 0, behind = 0, ooob = 0;
    static int n_iters = 0;
    ++n_iters;
    // new regularization, based on patch match idea:
    bool useNeighborRegularization = false;
    //if ( n_iters > 3 && useNeighborRegularization )
    if ( useNeighborRegularization )
    {
        const int nns = 8; // 4;
        Eigen::Matrix<double,nns,Eigen::Dynamic> newInvDepthMeasurements ( nns, seeds_.size() );

        for ( size_t k = 0; k < seeds_.size(); ++k )
        {
            newInvDepthMeasurements.col(k).setConstant( -1 );
            Seed * it = & ( seeds_[k] );
            Seed * neighbors[nns];
            neighbors[0] = it->left;
            neighbors[1] = it->right;
            neighbors[2] = it->above;
            neighbors[3] = it->below;
            if ( nns == 8 )
            {
                neighbors[4] = it->leftUpper;
                neighbors[5] = it->rightUpper;
                neighbors[6] = it->leftLower;
                neighbors[7] = it->rightLower;
            }
            const float seediZMin = it->mu + sqrt(it->sigma2);
            const float seediZMax = std::max<float>(it->mu - sqrt(it->sigma2), 0.00000001f);
            for ( int i = 0; i < nns; ++i )
            {
                if ( neighbors[i] == nullptr )
                {
                    ++null;
                    continue;
                }

                if ( !neighbors[i]->converged )
                    continue;

                if ( ! (seediZMin > neighbors[i]->mu && neighbors[i]->mu > seediZMax ) )
                {
                    ++range;
                    continue;
                }

                // check if point is visible in the current image
                //SE3 T_ref_cur = it->ftr->frame->T_f_w_ * frame->T_f_w_.inverse();
                const Eigen::Vector3d xyz_f( T_cur_ref*(1.0/neighbors[i]->mu * it->f) );
                if(xyz_f.z() < 0.0)  {
                    //++it; // behind the camera
                    ++behind;
                    continue;
                }

                const Eigen::Vector2d kp_c = (curImgPtr->K_c[0] * xyz_f).hnormalized();
                if ( kp_c(0) < 0 || kp_c(0) >= curImgPtr->grayImages[0].cols || kp_c(1) < 0 || kp_c(1) >= curImgPtr->grayImages[0].rows )
                {        //if(!frame->cam_->isInFrame(frame->f2c(xyz_f).cast<int>())) {
                    //++it; // point does not project in image
                    ++ooob;
                    continue;
                }

                float z_inv_min = neighbors[i]->mu + sqrt(neighbors[i]->sigma2);
                float z_inv_max = std::max<float>(neighbors[i]->mu - sqrt(neighbors[i]->sigma2), 0.00000001f);

                if ( ! (it->ftr) )
                    std::cerr << "why is the seeds ftr empty?" << std::endl;

                it->ftr->idepth_min = z_inv_min;
                it->ftr->idepth_max = z_inv_max;

                auto st = it->ftr->lastTraceStatus;
                bool useTrace = false;
                if ( useTrace )
                {
                    //const auto st = it->ftr->evaluate( curImgPtr, KRKi_cr, Kt_cr, affine_cr );
                    //const auto st = it->ftr->evaluateNgf( curImgPtr, KRKi_cr, Kt_cr, affine_cr );
                    const auto lst = it->ftr->lastTraceStatus;
                    it->ftr->lastTraceStatus = ImmaturePointStatus::IPS_GOOD;
                    it->ftr->traceOn( curImgPtr, KRKi_cr, Kt_cr, affine_cr );
                    //it->ftr->traceOnNgf( curImgPtr, KRKi_cr, Kt_cr );
                    st = it->ftr->lastTraceStatus;
                    it->ftr->lastTraceStatus = lst;
                    if ( st == ImmaturePointStatus::IPS_SKIPPED )
                        ++skipped;
                    if ( st == ImmaturePointStatus::IPS_OOB )
                        ++oob;

                    if ( st == ImmaturePointStatus::IPS_OUTLIER )
                        ++out;

                    if ( st == ImmaturePointStatus::IPS_BADCONDITION )
                        ++bad;
                }
                else
                {
                    //it->ftr->searchEpiLineBca ( curImgPtr, KRKi_cr, Kt_cr, affine_cr ); // T_cur_ref, curImgPtr->K_c[0],
                    //it->ftr->searchEpiLineNgf ( curImgPtr, KRKi_cr, Kt_cr ); // T_cur_ref, curImgPtr->K_c[0],
                    it->ftr->searchEpiLineNcc ( curImgPtr, KRKi_cr, Kt_cr ); // T_cur_ref, curImgPtr->K_c[0],
                    st = it->ftr->lastTraceStatus;
                }

                if ( st != ImmaturePointStatus::IPS_GOOD )
                {
                    it->b++;
                    continue;
                }

                //newInvDepthMeasurements(i,k) = neighbors[i]->mu;
                newInvDepthMeasurements(i,k) = std::max<double>(1e-5,.5*(it->ftr->idepth_min+it->ftr->idepth_max));
                ++goodOnes;
            }
        }
        for ( size_t k = 0; k < seeds_.size(); ++k)
        {
            Seed * it = & ( seeds_[k] );
            for ( int i = 0; i < nns; ++i)
            {
                if ( newInvDepthMeasurements(i,k) < 0 || !std::isfinite(newInvDepthMeasurements(i,k)) )
                    continue;

                const float z = 1/newInvDepthMeasurements(i,k);

                // compute tau
                double tau = computeTau(T_ref_cur, it->f, z, px_error_angle);
                double tau_inverse = 0.5 * (1.0/std::max<double>(0.0000001, z-tau) - 1.0/(z+tau));

                // update the estimate
                updateSeed(1./z, tau_inverse*tau_inverse, it);
                ++n_updates_reg;
            }
        }
    }
    std::cout << "DepthFilter: round: "<< n_iters << " updated: "<< int(n_updates) << " failed: " << int(n_failed_matches) << " wereGood: " << int(wereGood) << " converged: " << int(n_converged) << " reinited: " << n_reinit_seeds << " regs: " << n_updates_reg << " good: " << goodOnes << " oob: " << oob << " bad: " << bad << " skip: " << skipped << " outlier: " << out << " null: " << null << " range: " << range << " behind: " << behind << " ooob: " << ooob << std::endl;
    //firstRun = false;
}

void DepthFilter::updateSeed(const float x, const float tau2, Seed* seed)
{
    float norm_scale = sqrt(seed->sigma2 + tau2);
    if(std::isnan(norm_scale))
        return;
    boost::math::normal_distribution<float> nd(seed->mu, norm_scale);
    float s2 = 1./(1./seed->sigma2 + 1./tau2);
    float m = s2*(seed->mu/seed->sigma2 + x/tau2);
    float C1 = seed->a/(seed->a+seed->b) * boost::math::pdf(nd, x);
    float C2 = seed->b/(seed->a+seed->b) * 1./seed->z_range;
    float normalization_constant = C1 + C2;
    C1 /= normalization_constant;
    C2 /= normalization_constant;
    float f = C1*(seed->a+1.)/(seed->a+seed->b+1.) + C2*seed->a/(seed->a+seed->b+1.);
    float e = C1*(seed->a+1.)*(seed->a+2.)/((seed->a+seed->b+1.)*(seed->a+seed->b+2.))
            + C2*seed->a*(seed->a+1.0f)/((seed->a+seed->b+1.0f)*(seed->a+seed->b+2.0f));

    // update parameters
    float mu_new = C1*m+C2*seed->mu;
    seed->sigma2 = C1*(s2 + m*m) + C2*(seed->sigma2 + seed->mu*seed->mu) - mu_new*mu_new;
    seed->mu = mu_new;
    seed->a = (e-f)/(f-e/f);
    seed->b = seed->a*(1.0f-f)/f;
}

double DepthFilter::computeTau(
        const Eigen::Affine3d & T_ref_cur,
        const Eigen::Vector3d& f,
        const double z,
        const double px_error_angle)
{
    Eigen::Vector3d t(T_ref_cur.translation());
    Eigen::Vector3d a = f*z-t;
    double t_norm = t.norm();
    double a_norm = a.norm();
    double alpha = acos(f.dot(t)/t_norm); // dot product
    double beta = acos(a.dot(-t)/(t_norm*a_norm)); // dot product
    double beta_plus = beta + px_error_angle;
    double gamma_plus = M_PI-alpha-beta_plus; // triangle angles sum to PI
    double z_plus = t_norm*sin(beta_plus)/sin(gamma_plus); // law of sines
    return (z_plus - z); // tau
}

// lambda = 0.5f, iterations = 200
void DepthFilter::denoise( ImageDataPtr refImgPtr, const float depth_range, const float & lambda, const int iterations, const bool applyToSeeds )
{
    if ( ! refImgPtr ) return;

    std::cout << "starting to denoise." << std::endl;
    const float large_sigma2 = depth_range * depth_range / 72.f;

    // computeWeightsAndMu( )
    for ( size_t i = 0; i < seeds_.size(); ++i )
    {
        Seed & seed = seeds_[i];
        const float E_pi = seed.a / ( seed.a + seed.b);

        seed.g = std::max<float> ( (E_pi * seed.sigma2 + (1.0f-E_pi) * large_sigma2) / large_sigma2, 1.0f );
        seed.u = seed.mu;
        seed.u_head = seed.u;
        seed.p.setZero();
    }

    const int numCols = refImgPtr->grayImages[0].cols;
    const int numRows = refImgPtr->grayImages[0].rows;

    for(int i = 0; i < iterations; ++i)
    {
        std::cout << "updating primal dual. " << i << std::endl;
        updateTVL1PrimalDual( lambda, numCols, numRows );
    }

    //denoised = u_;
    if ( applyToSeeds )
    {
        for ( size_t i = 0; i < seeds_.size(); ++i )
        {
            Seed & seed = seeds_[i];
            seed.mu = seed.u;
        }
    }
}

void DepthFilter::updateTVL1PrimalDual( const float & lambda, const int & numCols, const int & numRows  )
{
    const float L = sqrt(8.0f);
    const float tau = (0.02f);
    const float sigma = ((1 / (L*L)) / tau);
    const float theta = 0.5f;
    //    const float lambda = 0.2f;

    // update dual
    for ( size_t i = 0; i < seeds_.size(); ++i )
    {
        Seed & seed = seeds_[i];
        const float g = seed.g;
        const Eigen::Vector2f p = seed.p;
        Eigen::Vector2f grad_uhead = Eigen::Vector2f::Zero();
        const float current_u = seed.u;

        Seed * right = (seed.right == nullptr) ? &seed : seed.right;
        Seed * below = (seed.below == nullptr) ? &seed : seed.below;
        grad_uhead[0] = right->u_head - current_u; //->atXY(min<int>(c_img_size.width-1, x+1), y)  - current_u;
        grad_uhead[1] = below->u_head - current_u; //->atXY(x, min<int>(c_img_size.height-1, y+1)) - current_u;
        const Eigen::Vector2f temp_p = g * grad_uhead * sigma + p;
        const float sqrt_p = temp_p.norm(); //sqrt(temp_p[0] * temp_p[0] + temp_p[1] * temp_p[1]);
        seed.p = temp_p / std::max<float>(1.0f, sqrt_p);
    }

    // update primal:
    for ( size_t i = 0; i < seeds_.size(); ++i )
    {
        Seed & seed = seeds_[i];
        const float noisy_depth = seed.mu;
        const float old_u = seed.u;
        const float g = seed.g;

        Eigen::Vector2f current_p = seed.p;
        Seed * left = (seed.left == nullptr) ? &seed : seed.left;
        Seed * above = (seed.above == nullptr) ? &seed : seed.above;
        Eigen::Vector2f w_p = left->p;
        Eigen::Vector2f n_p = above->p;

        const int x = seed.ftr->u;
        const int y = seed.ftr->v;
        if ( x == 0)
            w_p[0] = 0.f;
        else if ( x >= numCols-1 )
            current_p[0] = 0.f;
        if ( y == 0 )
            n_p[1] = 0.f;
        else if ( y >= numRows-1 )
            current_p[1] = 0.f;

        const float divergence = current_p[0] - w_p[0] + current_p[1] - n_p[1];

        const float tauLambda = tau*lambda;
        const float temp_u = old_u + tau * g * divergence;
        if ((temp_u - noisy_depth) > (tauLambda))
        {
            seed.u = temp_u - tauLambda;
        }
        else if ((temp_u - noisy_depth) < (-tauLambda))
        {
            seed.u = temp_u + tauLambda;
        }
        else
        {
            seed.u = noisy_depth;
        }
        seed.u_head = seed.u + theta * (seed.u - old_u);
    }
}
