
#include "openvslam/match/sparse_feature_aligner.h"
#include "openvslam/camera/perspective.h"
#include "openvslam/camera/equirectangular.h"
#include "openvslam/camera/fisheye.h"

#include <opencv2/core.hpp>

namespace openvslam {

namespace match {

bool project_to_image_distorted(
        const data::frame* curr_frame,
        const Mat44_t& world_to_cam,
        const Vec3_t& pt3,
        Vec2_t& projection) {
    float x_right;
    bool is_visible = false;
    switch (curr_frame->camera_->model_type_) {
    case camera::model_type_t::Perspective: {
        auto c = static_cast<camera::perspective*>(curr_frame->camera_);
        is_visible = c->reproject_to_image_distorted(world_to_cam.block<3,3>(0,0), world_to_cam.block<3,1>(0,3),
                              pt3, projection, x_right);
    } break;
    case camera::model_type_t::Fisheye: {
        auto c = static_cast<camera::fisheye*>(curr_frame->camera_);
        is_visible = c->reproject_to_image_distorted(world_to_cam.block<3,3>(0,0), world_to_cam.block<3,1>(0,3),
                              pt3, projection, x_right);
    }break;
    case camera::model_type_t::Equirectangular: {
        auto c = static_cast<camera::equirectangular*>(curr_frame->camera_);
        is_visible = c->reproject_to_image_distorted(world_to_cam.block<3,3>(0,0), world_to_cam.block<3,1>(0,3),
                              pt3, projection, x_right);
    }break;
    }
    return is_visible;
}

// need undistortion first as we get the 3D point from this
// pt -> pt_undist -> bearing -> bearing*depth
void convert_keypt_to_bearing(
        const data::keyframe* curr_frame,
        const cv::KeyPoint& kp,
        Vec3_t& bearing) {
    switch (curr_frame->camera_->model_type_) {
    case camera::model_type_t::Perspective: {
        auto c = static_cast<camera::perspective*>(curr_frame->camera_);
        const cv::KeyPoint undist_kp = c->undistort_keypoint(kp);
        bearing = c->convert_keypoint_to_bearing(undist_kp);
    } break;
    case camera::model_type_t::Fisheye: {
        auto c = static_cast<camera::fisheye*>(curr_frame->camera_);
        const cv::KeyPoint undist_kp = c->undistort_keypoint(kp);
        bearing = c->convert_keypoint_to_bearing(undist_kp);
    }break;
    case camera::model_type_t::Equirectangular: {
        auto c = static_cast<camera::equirectangular*>(curr_frame->camera_);
        const cv::KeyPoint undist_kp = c->undistort_keypoint(kp);
        bearing = c->convert_keypoint_to_bearing(undist_kp);
    }break;
    }
}

bool sparse_feature_aligner::find_projection_direct(
        data::keyframe *ref, data::frame *curr,
        data::landmark *mp, Vec2_t &px_curr,
        int &search_level) {
    Mat22_t ACR;
    const int index = mp->get_index_in_keyframe(ref);
    const cv::KeyPoint kp = ref->keypts_[index];
    const Vec2_t px_ref(kp.pt.x, kp.pt.y);
    const Mat44_t pose_ref = ref->get_cam_pose();
    const Mat44_t TCR = curr->cam_pose_cw_ * pose_ref.inverse();

    // 计算带边界的affine wrap，边界是为了便于计算梯度
    if (!get_warp_affine_matrix(ref, curr, px_ref, mp, kp.octave, TCR, ACR)) {
        return false;
    }
    search_level = get_best_search_level(ACR, ref->image_pyramid_.size() - 1, ref);
    warp_affine(ACR, ref->image_pyramid_[kp.octave],
            px_ref, kp.octave, ref, search_level, FEAT_ALIGN_WARP_HALF_PATCH_SIZE + 1,
            patch_with_border_);

    // remove the boarder
    uint8_t *ref_patch_ptr = patch_;
    for (int y = 1; y < FEAT_ALIGN_WARP_PATCH_SIZE + 1; ++y, ref_patch_ptr += FEAT_ALIGN_WARP_PATCH_SIZE) {
        uint8_t *ref_patch_border_ptr = patch_with_border_ + y * (FEAT_ALIGN_WARP_PATCH_SIZE + 2) + 1;
        for (int x = 0; x < FEAT_ALIGN_WARP_PATCH_SIZE; ++x)
            ref_patch_ptr[x] = ref_patch_border_ptr[x];
    }

    Vec2_t px_scaled = px_curr * curr->inv_scale_factors_[search_level];
    const bool success = align_pt_2d(curr->image_pyramid_[search_level], patch_with_border_, patch_, 10, px_scaled);
    px_curr = px_scaled * curr->scale_factors_[search_level];
    return success;
}

bool sparse_feature_aligner::get_warp_affine_matrix(data::keyframe *ref,
        data::frame *curr,
        const Vec2_t &px_ref,
        data::landmark *mp,
        int level,
        const Mat44_t &TCR,
        Mat22_t &ACR) {
    const Vec3_t pt_world = mp->get_pos_in_world();
    const Mat44_t ref_pose = ref->get_cam_pose();
    const Vec3_t pt_ref = ref_pose.block<3,3>(0,0) * pt_world + ref_pose.block<3,1>(0,3);
    double depth = pt_ref[2];

    // 偏移之后的3d点，深度取成和pt_ref一致
    cv::KeyPoint pt1;
    pt1.pt.x = px_ref(0)+FEAT_ALIGN_WARP_HALF_PATCH_SIZE*ref->scale_factors_[level];
    pt1.pt.y = px_ref(1);
    cv::KeyPoint pt2;
    pt2.pt.x = px_ref(0);
    pt2.pt.y = px_ref(1)+FEAT_ALIGN_WARP_HALF_PATCH_SIZE*ref->scale_factors_[level];

    Vec3_t bearing_1;
    Vec3_t bearing_2;
    convert_keypt_to_bearing(ref, pt1, bearing_1);
    convert_keypt_to_bearing(ref, pt2, bearing_2);
    bearing_1 /= bearing_1[2];
    bearing_2 /= bearing_2[2];
    const Vec3_t pt_du_ref = bearing_1 * depth;
    const Vec3_t pt_dv_ref = bearing_2 * depth;

    Vec2_t px_cur, px_du, px_dv;
    project_to_image_distorted(curr, TCR, pt_ref, px_cur);
    project_to_image_distorted(curr, TCR, pt_du_ref, px_du);
    project_to_image_distorted(curr, TCR, pt_dv_ref, px_dv);

    ACR.col(0) = (px_du - px_cur) / FEAT_ALIGN_WARP_HALF_PATCH_SIZE;
    ACR.col(1) = (px_dv - px_cur) / FEAT_ALIGN_WARP_HALF_PATCH_SIZE;
    return true;
}

void sparse_feature_aligner::warp_affine(
    const Mat22_t &ACR,
    const cv::Mat &img_ref,
    const Vec2_t &px_ref,
    const int &level_ref,
    const data::keyframe *ref,
    const int &search_level,
    const int &half_patch_size,
    uint8_t *patch) {
    const int patch_size = half_patch_size * 2;
    const Mat22_t ARC = ACR.inverse();

    // Affine warp
    uint8_t *patch_ptr = patch;
    const Vec2_t px_ref_pyr = px_ref / ref->scale_factors_[level_ref];
    for (int y = 0; y < patch_size; y++) {
        for (int x = 0; x < patch_size; x++, ++patch_ptr) {
            Vec2_t px_patch(x - half_patch_size, y - half_patch_size);
            px_patch *= ref->scale_factors_[search_level];
            const Vec2_t px(ARC * px_patch + px_ref_pyr);
            if (px[0] < 0 || px[1] < 0 || px[0] >= img_ref.cols - 1 || px[1] >= img_ref.rows - 1) {
                *patch_ptr = 0;
            } else {
                *patch_ptr = get_bilateral_interp_uchar(px[0], px[1], img_ref);
            }
        }
    }
}

uchar sparse_feature_aligner::get_bilateral_interp_uchar(
    const double &x, const double &y, const cv::Mat &gray) {
    const double xx = x - floor(x);
    const double yy = y - floor(y);
    uchar *data = &gray.data[int(y) * gray.step + int(x)];
    return uchar(
            (1 - xx) * (1 - yy) * data[0] +
            xx * (1 - yy) * data[1] +
            (1 - xx) * yy * data[gray.step] +
            xx * yy * data[gray.step + 1]);
}


int sparse_feature_aligner::get_best_search_level(const Mat22_t &ACR,
        const int &max_level,
        const data::keyframe *ref) {
    int search_level = 0;
    float D = ACR.determinant();
    while (D > 3.0 && search_level < max_level) {
        search_level += 1;
        D *= ref->inv_level_sigma_sq_[1];
    }
    return search_level;
}

bool sparse_feature_aligner::align_pt_2d(
            const cv::Mat &cur_img,
            uint8_t *ref_patch_with_border,
            uint8_t *ref_patch,
            const int n_iter,
            Vec2_t &cur_px_estimate) {
    bool converged = false;

    // compute derivative of template and prepare inverse compositional
    //float __attribute__ (( __aligned__ ( 16 ))) ref_patch_dx[FEAT_ALIGN_WARP_PATCH_AREA];
    //float __attribute__ (( __aligned__ ( 16 ))) ref_patch_dy[FEAT_ALIGN_WARP_PATCH_AREA];
    float ref_patch_dx[FEAT_ALIGN_WARP_PATCH_AREA];
    float ref_patch_dy[FEAT_ALIGN_WARP_PATCH_AREA];
    Mat33_t H;
    H.setZero();

    // compute gradient and hessian
    const int ref_step = FEAT_ALIGN_WARP_PATCH_SIZE + 2;
    float *it_dx = ref_patch_dx;
    float *it_dy = ref_patch_dy;
    for (int y = 0; y < FEAT_ALIGN_WARP_PATCH_SIZE; ++y) {
        uint8_t *it = ref_patch_with_border + (y + 1) * ref_step + 1;
        for (int x = 0; x < FEAT_ALIGN_WARP_PATCH_SIZE; ++x, ++it, ++it_dx, ++it_dy) {
            Vec3_t J;
            J[0] = 0.5 * (it[1] - it[-1]);
            J[1] = 0.5 * (it[ref_step] - it[-ref_step]);
            J[2] = 1;
            *it_dx = J[0];
            *it_dy = J[1];
            H += J * J.transpose();
        }
    }
    Mat33_t Hinv = H.inverse();
    float mean_diff = 0;

    // Compute pixel location in new image:
    float u = cur_px_estimate.x();
    float v = cur_px_estimate.y();

    // termination condition
    const float min_update_squared = 0.03 * 0.03;
    const int cur_step = cur_img.step.p[0];
    Vec3_t update;
    update.setZero();
    float chi2 = 0;
    for (int iter = 0; iter < n_iter; ++iter) {
        chi2 = 0;
        int u_r = floor(u);
        int v_r = floor(v);
        if (u_r < FEAT_ALIGN_WARP_HALF_PATCH_SIZE ||
            v_r < FEAT_ALIGN_WARP_HALF_PATCH_SIZE ||
            u_r >= cur_img.cols - FEAT_ALIGN_WARP_HALF_PATCH_SIZE ||
            v_r >= cur_img.rows - FEAT_ALIGN_WARP_HALF_PATCH_SIZE)
            break;

        if (std::isnan(u) ||
            std::isnan(v)) // TODO very rarely this can happen, maybe H is singular? should not be at corner.. check
            return false;

        // compute interpolation weights
        float subpix_x = u - u_r;
        float subpix_y = v - v_r;
        float wTL = (1.0 - subpix_x) * (1.0 - subpix_y);
        float wTR = subpix_x * (1.0 - subpix_y);
        float wBL = (1.0 - subpix_x) * subpix_y;
        float wBR = subpix_x * subpix_y;

        // loop through search_patch, interpolate
        uint8_t *it_ref = ref_patch;
        float *it_ref_dx = ref_patch_dx;
        float *it_ref_dy = ref_patch_dy;
        Vec3_t Jres;
        Jres.setZero();
        for (int y = 0; y < FEAT_ALIGN_WARP_PATCH_SIZE; ++y) {
            uint8_t *it = (uint8_t *) cur_img.data +
                    (v_r + y - FEAT_ALIGN_WARP_HALF_PATCH_SIZE) * cur_step +
                    u_r - FEAT_ALIGN_WARP_HALF_PATCH_SIZE;
            for (int x = 0; x < FEAT_ALIGN_WARP_PATCH_SIZE; ++x, ++it, ++it_ref, ++it_ref_dx, ++it_ref_dy) {
                float search_pixel = wTL * it[0] + wTR * it[1] + wBL * it[cur_step] + wBR * it[cur_step + 1];
                float res = search_pixel - *it_ref + mean_diff;
                Jres[0] -= res * (*it_ref_dx);
                Jres[1] -= res * (*it_ref_dy);
                Jres[2] -= res;
                chi2 += res * res;
            }
        }
        update = Hinv * Jres;
        u += update[0];
        v += update[1];
        mean_diff += update[2];
        if (update[0] * update[0] + update[1] * update[1] < min_update_squared) {
            converged = true;
            break;
        }
    }

    cur_px_estimate << u, v;
    return converged;
}

} // namespace match
} // namespace openvslam

