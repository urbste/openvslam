
#include "openvslam/match/sparse_image_aligner.h"
#include "openvslam/match/sparse_feature_aligner.h"
#include "openvslam/camera/perspective.h"
#include "openvslam/camera/fisheye.h"
#include "openvslam/camera/equirectangular.h"
#include "openvslam/camera/radial_division.h"

#include <g2o/types/slam3d/se3quat.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Core>

#include <spdlog/spdlog.h>
#include <iostream>

namespace openvslam {

namespace match {

double norm_max(const Eigen::VectorXd &v) {
    double max = -1.0;
    for (int i = 0; i < v.size(); i++) {
        double abs = fabs(v[i]);
        if (abs > max) {
            max = abs;
        }
    }
    return max;
}

void sparse_image_aligner::set_parameters(
        int max_level, int min_level, int n_iter,
        bool display, bool verbose) {


    min_level_ = min_level;
    max_level_ = max_level;
    n_iter_ = n_iter;
    n_iter_init_ = n_iter_;
    verbose_ = verbose;
    eps_ = 0.0000001;
    display_ = display;
    use_weights_ = true;
}
size_t sparse_image_aligner::run(
        const data::frame *ref_frame,
        const data::frame *cur_frame,
        Mat44_t &TCR) {

    reset();

    if (ref_frame->keypts_.empty()) {
        spdlog::warn("SparseImgAlign: no features to track! Returning.");
        return 0;
    }

    ref_frame_ = ref_frame;
    cur_frame_ = cur_frame;

    ref_patch_cache_ = cv::Mat(ref_frame->num_keypts_, IMAGE_ALIGN_PATCH_AREA, CV_32F);
    jacobian_cache_.resize(Eigen::NoChange, ref_patch_cache_.rows * IMAGE_ALIGN_PATCH_AREA);
    visible_fts_ = std::vector<bool>(ref_patch_cache_.rows, false);
    Mat44_t tmp = cur_frame->cam_pose_cw_ * ref_frame_->cam_pose_cw_.inverse();
    ::g2o::SE3Quat T_cur_from_ref(tmp.block<3,3>(0,0), tmp.block<3,1>(0,3));
    int iterations[] = {10, 10, 10, 10, 10, 10};
    for (level_ = max_level_; level_ >= min_level_; level_ -= 1) {
        mu_ = 0.1;
        jacobian_cache_.setZero();
        have_ref_patch_cache_ = false;
        n_iter_ = iterations[level_];
        optimize(T_cur_from_ref);

        //if (norm_max(x_) <= eps_ && level_ <= 1 ) {
        //    break;
        //}
    }
//    if (error_increased_) {
//        // if this happens we tell tracking that we were not successfull
//        std::cout<<"Error increased in sparse align, returning to feature method.\n";
//        return 0;
//    }
    TCR = T_cur_from_ref.to_homogeneous_matrix();
    return n_meas_ / IMAGE_ALIGN_PATCH_AREA;
}

Mat66_t sparse_image_aligner::getFisherInformation() {
    double sigma_i_sq = 5e-4 * 255.0 * 255.0; // image noise
    Mat66_t I = H_ / sigma_i_sq;
    return I;
}

void sparse_image_aligner::precomputeReferencePatches() {
    const int border = IMAGE_ALIGN_PATCH_HALF_SIZE + 1;
    const cv::Mat& ref_img = ref_frame_->image_pyramid_[level_];

    const int stride = ref_img.cols;
    const double scale = ref_frame_->inv_scale_factors_[level_];
    size_t feature_counter = 0;

    for (unsigned int i = 0; i < ref_frame_->num_keypts_; i++, ++feature_counter) {
        data::landmark *mp = ref_frame_->landmarks_[i];
        if (mp == nullptr || mp->will_be_erased() || ref_frame_->outlier_flags_[i] == true)
            continue;

        // check if reference with patch size is within image
        const cv::KeyPoint &kp = ref_frame_->keypts_[i];
        const float u_ref = kp.pt.x * scale;
        const float v_ref = kp.pt.y * scale;
        const int u_ref_i = floor(u_ref);
        const int v_ref_i = floor(v_ref);
        if (u_ref_i - border < 0 || v_ref_i - border < 0 || u_ref_i + border >= ref_img.cols ||
            v_ref_i + border >= ref_img.rows)
            continue;

        visible_fts_[i] = true;

        // cannot just take the 3d points coordinate because of the reprojection errors in the reference image!!!
        // const double depth ( ( ( *it )->_mappoint->_pos_world - ref_pos ).norm() );
        // LOG(INFO)<<"depth = "<<depth<<", features depth = "<<(*it)->_depth<<endl;
        const Vec3_t xyz_ref =
                ref_frame_->cam_pose_cw_.block<3,3>(0,0) * mp->get_pos_in_world() +
                ref_frame_->cam_pose_cw_.block<3,1>(0,3);

        // evaluate projection jacobian
        Mat26_t frame_jac;
        switch (ref_frame_->camera_->model_type_) {
        case camera::model_type_t::Perspective: {
            auto c = static_cast<camera::perspective*>(ref_frame_->camera_);
            c->jacobian_xyz_to_cam(xyz_ref, frame_jac, scale);
        } break;
        case camera::model_type_t::Fisheye: {
            auto c = static_cast<camera::fisheye*>(ref_frame_->camera_);
            c->jacobian_xyz_to_cam(xyz_ref, frame_jac, scale);
        }break;
        case camera::model_type_t::Equirectangular: {
            auto c = static_cast<camera::equirectangular*>(ref_frame_->camera_);
            c->jacobian_xyz_to_cam(xyz_ref, frame_jac, scale);
        }break;
        case camera::model_type_t::RadialDivision: {
            auto c = static_cast<camera::radial_division*>(ref_frame_->camera_);
            c->jacobian_xyz_to_cam(xyz_ref, frame_jac, scale);
        }break;
        }
        // compute bilateral interpolation weights for reference image
        const float subpix_u_ref = u_ref - u_ref_i;
        const float subpix_v_ref = v_ref - v_ref_i;
        const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
        const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
        const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
        const float w_ref_br = subpix_u_ref * subpix_v_ref;
        size_t pixel_counter = 0;
        float *cache_ptr = reinterpret_cast<float *> ( ref_patch_cache_.data ) +
                IMAGE_ALIGN_PATCH_AREA * feature_counter;
        for (int y = 0; y < IMAGE_ALIGN_PATCH_SIZE; ++y) {
            uint8_t *ref_img_ptr = (uint8_t *) ref_img.data +
                    (v_ref_i + y - IMAGE_ALIGN_PATCH_HALF_SIZE) * stride +
                    (u_ref_i - IMAGE_ALIGN_PATCH_HALF_SIZE);

            for (int x = 0; x < IMAGE_ALIGN_PATCH_SIZE;
                 ++x, ++ref_img_ptr, ++cache_ptr, ++pixel_counter) {
                // precompute interpolated reference patch color
                *cache_ptr =
                        w_ref_tl * ref_img_ptr[0] +
                        w_ref_tr * ref_img_ptr[1] +
                        w_ref_bl * ref_img_ptr[stride] +
                        w_ref_br * ref_img_ptr[stride + 1];

                // we use the inverse compositional: thereby we can take the gradient always at the same position
                // get gradient of warped image (~gradient at warped position)
                const double dx = 0.5 * ((w_ref_tl * ref_img_ptr[1] + w_ref_tr * ref_img_ptr[2] +
                                    w_ref_bl * ref_img_ptr[stride + 1] + w_ref_br * ref_img_ptr[stride + 2])
                                   - (w_ref_tl * ref_img_ptr[-1] + w_ref_tr * ref_img_ptr[0] +
                                      w_ref_bl * ref_img_ptr[stride - 1] + w_ref_br * ref_img_ptr[stride]));
                const double dy = 0.5 * ((w_ref_tl * ref_img_ptr[stride] + w_ref_tr * ref_img_ptr[1 + stride] +
                                    w_ref_bl * ref_img_ptr[stride * 2] + w_ref_br * ref_img_ptr[stride * 2 + 1])
                                   - (w_ref_tl * ref_img_ptr[-stride] + w_ref_tr * ref_img_ptr[1 - stride] +
                                      w_ref_bl * ref_img_ptr[0] + w_ref_br * ref_img_ptr[1]));

                // cache the jacobian
                jacobian_cache_.col(feature_counter * IMAGE_ALIGN_PATCH_AREA + pixel_counter) =
                        (dx * frame_jac.row(0) + dy * frame_jac.row(1));
            }
        }
    }
    //std::cout<<"feature_counter: "<<feature_counter<<"\n";

    have_ref_patch_cache_ = true;
}

double huber_loss(const double err, const double huber_const = 1.345){
    const double err_abs = std::abs(err);

    if (err_abs < huber_const) {
        return 1.0;
    } else {
        return huber_const / err_abs;
    }
}

double sparse_image_aligner::computeResiduals(
        const ::g2o::SE3Quat &T_cur_from_ref,
        bool linearize_system,
        bool compute_weight_scale) {
    // Warp the (cur)rent image such that it aligns with the (ref)erence image
    const cv::Mat &cur_img = cur_frame_->image_pyramid_[level_];

    if (linearize_system && display_)
        resimg_ = cv::Mat(cur_img.size(), CV_32F, cv::Scalar(0));

    if (have_ref_patch_cache_ == false)
        precomputeReferencePatches();

    // compute the weights on the first iteration
    std::vector<double> errors;
    if (compute_weight_scale)
        errors.reserve(visible_fts_.size());
    const int stride = cur_img.cols;
    const int border = IMAGE_ALIGN_PATCH_HALF_SIZE + 1;
    const double scale = ref_frame_->inv_scale_factors_[level_];
    double chi2 = 0.0;
    size_t feature_counter = 0; // is used to compute the index of the cached jacobian

    size_t visible = 0;
    for (unsigned int i = 0; i < ref_frame_->num_keypts_; i++, feature_counter++) {
        // check if feature is within image
        if (visible_fts_[i] == false)
            continue;
        data::landmark *mp = ref_frame_->landmarks_[i];
        if (mp == nullptr || mp->will_be_erased() || ref_frame_->outlier_flags_[i] == true)
            continue;

        // compute pixel location in cur img
        const Vec3_t xyz_ref =
                ref_frame_->cam_pose_cw_.block<3,3>(0,0) * mp->get_pos_in_world() +
                ref_frame_->cam_pose_cw_.block<3,1>(0,3);

        Vec2_t uv_cur;
        if (!project_to_image_distorted(cur_frame_, T_cur_from_ref.to_homogeneous_matrix(), xyz_ref, uv_cur))
            continue;

        const Vec2_t uv_cur_pyr(uv_cur * scale);
        const double u_cur = uv_cur_pyr[0];
        const double v_cur = uv_cur_pyr[1];
        const int u_cur_i = std::floor(u_cur);
        const int v_cur_i = std::floor(v_cur);

        // check if projection is within the image
        if (u_cur_i < 0 || v_cur_i < 0 || u_cur_i - border < 0 || v_cur_i - border < 0 ||
            u_cur_i + border >= cur_img.cols || v_cur_i + border >= cur_img.rows)
            continue;

        visible++;

        // compute bilateral interpolation weights for the current image
        const float subpix_u_cur = u_cur - u_cur_i;
        const float subpix_v_cur = v_cur - v_cur_i;
        const float w_cur_tl = (1.0f - subpix_u_cur) * (1.0f - subpix_v_cur);
        const float w_cur_tr = subpix_u_cur * (1.0f - subpix_v_cur);
        const float w_cur_bl = (1.0f - subpix_u_cur) * subpix_v_cur;
        const float w_cur_br = subpix_u_cur * subpix_v_cur;
        float *ref_patch_cache_ptr =
                reinterpret_cast<float *> ( ref_patch_cache_.data ) + IMAGE_ALIGN_PATCH_AREA * feature_counter;
        size_t pixel_counter = 0; // is used to compute the index of the cached jacobian
        for (int y = 0; y < IMAGE_ALIGN_PATCH_SIZE; ++y) {
            uint8_t *cur_img_ptr = (uint8_t *) cur_img.data +
                    (v_cur_i + y - IMAGE_ALIGN_PATCH_HALF_SIZE) * stride +
                    (u_cur_i - IMAGE_ALIGN_PATCH_HALF_SIZE);

            for (int x = 0; x < IMAGE_ALIGN_PATCH_SIZE;
                 ++x, ++pixel_counter, ++cur_img_ptr, ++ref_patch_cache_ptr) {
                // compute residual
                const float intensity_cur =
                        w_cur_tl * cur_img_ptr[0] +
                        w_cur_tr * cur_img_ptr[1] +
                        w_cur_bl * cur_img_ptr[stride] +
                        w_cur_br * cur_img_ptr[stride + 1];
                const double res = static_cast<double>(intensity_cur - (*ref_patch_cache_ptr));
                //std::cout<<"res "<<res<<"\n";
                // used to compute scale for robust cost
                if (compute_weight_scale)
                    errors.push_back(std::abs(res));

                // robustification
                double weight = 1.0;
                if (use_weights_) {
                    weight = huber_loss(res, 5);
                }

                chi2 += res * res * weight;
                n_meas_++;

                if (linearize_system) {
                    // compute Jacobian, weighted Hessian and weighted "steepest descend images" (times error)
                    const Vec6_t J(jacobian_cache_.col(feature_counter * IMAGE_ALIGN_PATCH_AREA + pixel_counter));
                    H_.noalias() += J * J.transpose() * weight;
                    Jres_.noalias() -= J * res * weight;
                    if (display_)
                        resimg_.at<float>((int) v_cur + y - IMAGE_ALIGN_PATCH_HALF_SIZE,
                                          (int) u_cur + x - IMAGE_ALIGN_PATCH_HALF_SIZE) =
                                           res / 255.0;
                }
            }
        }
    }


    // compute the weights on the first iteration
    //if (compute_weight_scale && iter_ == 0)
    //    scale_ = scale_estimator_->compute(errors);
    // std::cout<<"feature_counter: "<<feature_counter<<"\n";
    // std::cout<<"visible: "<<visible<<std::endl;
    // std::cout<<"n_meas_: "<<n_meas_<<std::endl;

    return chi2 / n_meas_;
}

int sparse_image_aligner::solve() {
    x_ = H_.ldlt().solve(Jres_);
    if ((bool) std::isnan((float) x_[0]))
        return 0;
    return 1;
}

void sparse_image_aligner::update(
        const ::g2o::SE3Quat &T_curold_from_ref,
        ::g2o::SE3Quat &T_curnew_from_ref) {
    Vec6_t new_x;
    T_curnew_from_ref = T_curold_from_ref * ::g2o::SE3Quat(-x_);
}

void sparse_image_aligner::startIteration() {}

void sparse_image_aligner::finishIteration() {
    if (display_) {
        cv::namedWindow("residuals", cv::WINDOW_AUTOSIZE);
        cv::imshow("residuals", resimg_ * 10);
        cv::waitKey(100);
    }
}

void sparse_image_aligner::optimize(::g2o::SE3Quat& model) {
    // switch LM or GN?
    optimizeGaussNewton(model);
}

void sparse_image_aligner::optimizeGaussNewton(g2o::SE3Quat& model) {
    // Compute weight scale
    error_increased_ = false;
    //if (use_weights_) {
    //    computeResiduals(model, false, true);
    //}

    // Save the old model to rollback in case of unsuccessful update
    g2o::SE3Quat old_model(model);

    // perform iterative estimation
    for (iter_ = 0; iter_ < n_iter_; ++iter_) {
        rho_ = 0;
        startIteration();

        H_.setZero();
        Jres_.setZero();

        // compute initial error
        n_meas_ = 0;
        double new_chi2 = computeResiduals(model, true, false);

        // add prior
        //if (have_prior_) {
            //applyPrior(model);
        //}

        // solve the linear system
        if (!solve()) {
            // matrix was singular and could not be computed
            std::cout << "Matrix is close to singular! Stop Optimizing." << std::endl;
            std::cout << "H = " << H_ << std::endl;
            std::cout << "Jres = " << Jres_ << std::endl;
            stop_ = true;
        }

        // check if error increased since last optimization
        if ((iter_ > 0 && new_chi2 > 1.2 * chi2_) || stop_) {
            error_increased_ = true;
            if (verbose_) {
                std::cout <<  "It. "<< iter_
                           <<"\t Failure"
                          << "\t new_chi2 = "<<new_chi2
                         << "\t Error increased. Stop optimizing.\n";
            }
            model = old_model; // rollback
            break;
        }

        // update the model
        g2o::SE3Quat new_model;
        update(model, new_model);
        old_model = model;
        model = new_model;

        chi2_ = new_chi2;

        if (verbose_) {
            std::cout << "It. " << iter_
                      << "\t Success"
                      << "\t new_chi2 = " << new_chi2
                      << "\t n_meas = " << n_meas_
                      << "\t x_norm = " << norm_max(x_)
                      << std::endl;
        }

        finishIteration();

        // stop when converged, i.e. update step too small
        if (norm_max(x_) <= eps_) {
            break;
        }
    }
}

void sparse_image_aligner::reset() {
    have_prior_ = false;
    chi2_ = std::numeric_limits<double>::max();
    mu_ = mu_init_;
    nu_ = nu_init_;
    n_meas_ = 0;
    n_iter_ = n_iter_init_;
    iter_ = 0;
    stop_ = false;
}

} // namespace match
} // namespace openvslam

