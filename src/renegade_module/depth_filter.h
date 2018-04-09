// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

// strongly modified for my own work.

#ifndef SVO_DEPTH_FILTER_H_
#define SVO_DEPTH_FILTER_H_

//#include <list>
#include "types.h"
#include "depth_point.h"

/// A seed is a probabilistic 1depth estimate for a single pixel.
struct Seed
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  bool converged = false;
  bool is_outlier = true;
  int id;                      //!< Seed ID, only used for visualization.
  Eigen::Vector3d f; // heading range = Ki * (u,v,1)
  ImmaturePointPtr ftr;
  //Feature* ftr;                //!< Feature in the keyframe for which the depth should be computed.
  float a;                     //!< a of Beta distribution: When high, probability of inlier is large.
  float b;                     //!< b of Beta distribution: When high, probability of outlier is large.
  float mu;                    //!< Mean of normal distribution.
  float z_range;               //!< Max range of the possible depth.
  float sigma2;                //!< Variance of normal distribution.
  Eigen::Matrix2d patch_cov;          //!< Patch covariance in reference image.
  Seed(ImmaturePointPtr ftr_, float depth_mean, float depth_min);

  void reinit( const float depth_mean = 1.f, const float depth_min = 0.1f );
  // for denoising
  Seed * left = nullptr;
  Seed * right = nullptr;
  Seed * above = nullptr;
  Seed * below = nullptr;
  Seed * leftUpper = nullptr;
  Seed * rightUpper = nullptr;
  Seed * leftLower = nullptr;
  Seed * rightLower = nullptr;

  float g;
  float u;
  float u_head;
  Eigen::Vector2f p;
};

/// Depth filter implements the Bayesian Update proposed in:
/// "Video-based, Real-Time Multi View Stereo" by G. Vogiatzis and C. Hernández.
/// In Image and Vision Computing, 29(7):434-441, 2011.
///
/// The class uses a callback mechanism such that it can be used also by other
/// algorithms than nslam and for simplified testing.
class DepthFilter
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// Depth-filter config parameters
  struct Options
  {
    bool check_ftr_angle;                       //!< gradient features are only updated if the epipolar line is orthogonal to the gradient.
    bool epi_search_1d;                         //!< restrict Gauss Newton in the epipolar search to the epipolar line.
    bool verbose;                               //!< display output.
    bool use_photometric_disparity_error;       //!< use photometric disparity error instead of 1px error in tau computation.
    int max_n_kfs;                              //!< maximum number of keyframes for which we maintain seeds.
    double sigma_i_sq;                          //!< image noise.
    double seed_convergence_sigma2_thresh;      //!< threshold on depth uncertainty for convergence.
    Options()
    : check_ftr_angle(false),
      epi_search_1d(false),
      verbose(false),
      use_photometric_disparity_error(false),
      max_n_kfs(3),
      sigma_i_sq(5e-4),
      seed_convergence_sigma2_thresh(200.0)
    {}
  } options_;

  DepthFilter();

  ~DepthFilter(){}

  /// Initialize new seeds from a frame.
  void initializeSeeds ( std::vector<ImmaturePointPtr> & invDepthPts );

  const std::vector<Seed,Eigen::aligned_allocator<Seed>> & getSeeds(){ return seeds_; }

  /// Update all seeds with a new measurement frame.
  void updateSeeds(ImageDataPtr refImgPtr, ImageDataPtr curImgPtr, const Eigen::Affine3d & T_cur_ref, const Vec2f & affine_cr );

  float getConvergencePercentage ( ) const;

  /// Bayes update of the seed, x is the measurement, tau2 the measurement uncertainty
  static void updateSeed(
      const float x,
      const float tau2,
      Seed* seed);

  /// Compute the uncertainty of the measurement.
  static double computeTau(
      const Eigen::Affine3d & T_ref_cur,
      const Eigen::Vector3d& f,
      const double z,
      const double px_error_angle);

  void denoise( ImageDataPtr refImgPtr, const float depth_range, const float & lambda = 0.2f, const int iterations = 200, const bool applyToSeeds = false );
  void updateTVL1PrimalDual( const float & lambda, const int & numCols, const int & numRows  );

protected:
  std::vector<Seed, Eigen::aligned_allocator<Seed> > seeds_;
  double new_keyframe_min_depth_;       //!< Minimum depth in the new keyframe. Used for range in new seeds.
  double new_keyframe_mean_depth_;      //!< Maximum depth in the new keyframe. Used for range in new seeds.

};

typedef std::shared_ptr<DepthFilter> DepthFilterPtr;
#endif // SVO_DEPTH_FILTER_H_ 
