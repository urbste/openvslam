#ifndef OPENVSLAM_SPARSE_IMAGE_ALIGNER_H
#define OPENVSLAM_SPARSE_IMAGE_ALIGNER_H

#include "openvslam/match/base.h"
#include "openvslam/type.h"
#include "openvslam/data/frame.h"
#include "openvslam/data/keyframe.h"
#include "openvslam/data/landmark.h"

namespace openvslam {

namespace data {
class frame;
} // namespace data

namespace match {

const int IMAGE_ALIGN_PATCH_HALF_SIZE = 2;
const int IMAGE_ALIGN_PATCH_SIZE = 2 * IMAGE_ALIGN_PATCH_HALF_SIZE;
const int IMAGE_ALIGN_PATCH_AREA = IMAGE_ALIGN_PATCH_SIZE * IMAGE_ALIGN_PATCH_SIZE;

inline void get_interp_weights(
        const float x, const float y,
        Eigen::Vector4f& interp_weights) {
    const float subpix_x = x - std::floor(x);
    const float subpix_y = y - std::floor(y);
    interp_weights << (1.0 - subpix_x) * (1.0 - subpix_y),
            subpix_x * (1.0 - subpix_y),
            (1.0 - subpix_x) * subpix_y,
            subpix_x * subpix_y;
}


template <typename T>
T get_interpolation_weighted_grayvalue(
    const cv::Mat& img,
    const int& x,
    const int& y,
    const Eigen::Vector4f& weights) {
    // central pixel value
    Eigen::Vector4f image_values;
    if (img.type() == CV_32FC1) {
        image_values << static_cast<float>(img.ptr<float>(y)[x]),
                        static_cast<float>(img.ptr<float>(y)[x+1]),
                        static_cast<float>(img.ptr<float>(y+1)[x]),
                        static_cast<float>(img.ptr<float>(y+1)[x+1]);
    } else if (img.type() == CV_8UC1) {
        image_values << static_cast<float>(img.ptr<uint8_t>(y)[x]),
                        static_cast<float>(img.ptr<uint8_t>(y)[x+1]),
                        static_cast<float>(img.ptr<uint8_t>(y+1)[x]),
                        static_cast<float>(img.ptr<uint8_t>(y+1)[x+1]);
    }
    return (T)image_values.dot(weights);
}

class sparse_image_aligner final : public base {
public:
EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    sparse_image_aligner() : base(0.0, false) {}

    sparse_image_aligner(const float lowe_ratio, const bool check_orientation)
        : base(lowe_ratio, check_orientation) {}

    ~sparse_image_aligner() final = default;


    cv::Mat resimg_;

    void set_parameters(int max_level,
            int min_level,
            int n_iter = 10,
            bool display = false,
            bool verbose = false);
    /**
     * 计算 ref 和 current 之间的运动
     * @brief compute the relative motion between ref frame and current frame
     * @param[in] ref_frame the reference
     * @param[in] cur_frame the current frame
     * @param[out] TCR motion from ref to current
     */
    size_t run(const data::frame *ref_frame,
               const data::frame *cur_frame,
               Mat44_t &TCR);

    /// Return fisher information matrix, i.e. the Hessian of the log-likelihood
    /// at the converged state.
    Mat66_t getFisherInformation();

protected:
    const data::frame *ref_frame_;              //!< reference frame, has depth for gradient pixels.
    const data::frame *cur_frame_;              //!< only the image is known!

    // cache:
    Eigen::Matrix<double, 6, Eigen::Dynamic, Eigen::ColMajor> jacobian_cache_;

    bool have_ref_patch_cache_;
    cv::Mat ref_patch_cache_;
    std::vector<bool> visible_fts_;

    Mat66_t H_;       //!< Hessian approximation
    Vec6_t Jres_;    //!< Jacobian x Residual
    Vec6_t x_;       //!< update step

    bool have_prior_;
    ::g2o::SE3Quat prior_;
    Mat66_t I_prior_; //!< Prior information matrix (inverse covariance)

    void optimize(::g2o::SE3Quat& model);

    void optimizeGaussNewton(::g2o::SE3Quat& model);

    void precomputeReferencePatches();

    double computeResiduals(const ::g2o::SE3Quat &model,
                           bool linearize_system,
                           bool compute_weight_scale = false);

    int solve();

    void reset();

    void update(const ::g2o::SE3Quat &old_model,
                ::g2o::SE3Quat &new_model);

    void startIteration();

    void finishIteration();

    inline Mat26_t JacobXYZ2Cam(const Vec3_t &xyz) {
        Mat26_t J;
        const double x = xyz[0];
        const double y = xyz[1];
        const double z_inv = 1. / xyz[2];
        const double z_inv_2 = z_inv * z_inv;

        J(0, 0) = -z_inv;           // -1/z
        J(0, 1) = 0.0;              // 0
        J(0, 2) = x * z_inv_2;        // x/z^2
        J(0, 3) = y * J(0, 2);      // x*y/z^2
        J(0, 4) = -(1.0 + x * J(0, 2)); // -(1.0 + x^2/z^2)
        J(0, 5) = y * z_inv;          // y/z

        J(1, 0) = 0.0;              // 0
        J(1, 1) = -z_inv;           // -1/z
        J(1, 2) = y * z_inv_2;        // y/z^2
        J(1, 3) = 1.0 + y * J(1, 2); // 1.0 + y^2/z^2
        J(1, 4) = -J(0, 3);       // -x*y/z^2
        J(1, 5) = -x * z_inv;         // x/z
        return J;
    }


private:
    double scale_estimator(const std::vector<double>& errors);

    int level_;                     //!< current pyramid level on which the optimization runs.
    bool display_;                  //!< display residual image.
    //!< coarsest pyramid level for the alignment.
    int max_level_;

    //!< finest pyramid level for the alignment.
    int min_level_;

    int n_iter_;

    int n_iter_init_;

    bool verbose_;

    double eps_;

    bool error_increased_ = false;

    bool stop_ = false;

    double chi2_ = std::numeric_limits<double>::max();

    bool use_weights_ = false;

    int iter_;

    double rho_;

    int n_meas_;

    double mu_;
    double nu_;

    double mu_init_ = 0.1;

    double nu_init_ = 2.0;

    double res_scale_ = 1.0;
};

} // namespace match
} // namespace openvslam

#endif // OPENVSLAM_SPARSE_IMAGE_ALIGNER_H
