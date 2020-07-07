#ifndef OPENVSLAM_SPARSE_FEATURE_ALIGNER_H
#define OPENVSLAM_SPARSE_FEATURE_ALIGNER_H

#include "openvslam/match/base.h"
#include "openvslam/type.h"
#include "openvslam/data/frame.h"
#include "openvslam/data/keyframe.h"
#include "openvslam/data/landmark.h"
#include "openvslam/match/sparse_image_aligner.h"

namespace openvslam {

namespace data {
class frame;
} // namespace data

namespace match {
const int FEAT_ALIGN_WARP_PATCH_SIZE = 8;
const int FEAT_ALIGN_WARP_HALF_PATCH_SIZE = FEAT_ALIGN_WARP_PATCH_SIZE / 2;
const int FEAT_ALIGN_WARP_PATCH_AREA = FEAT_ALIGN_WARP_PATCH_SIZE * FEAT_ALIGN_WARP_PATCH_SIZE;

bool project_to_image_distorted(
        const data::frame* curr_frame,
        const Mat44_t& world_to_cam,
        const Vec3_t& pt3,
        Vec2_t& projection);

void convert_keypt_to_bearing(
        const data::keyframe* curr_frame,
        const cv::KeyPoint& kp,
        Vec3_t& bearing);

template <class T>
class sparse_feature_aligner final : public base {
public:
    sparse_feature_aligner() : base(0.0, false) {}

    sparse_feature_aligner(const float lowe_ratio, const bool check_orientation)
        : base(lowe_ratio, check_orientation) {}

    ~sparse_feature_aligner() final = default;

    bool find_projection_direct(data::keyframe *ref, data::frame *curr,
                                data::landmark *mp, Vec2_t &px_curr,
                                int &search_level);
private:

    bool align_pt_2d(const cv::Mat &cur_img,
            T *ref_patch_with_border,
            T *ref_patch,
            const int n_iter,
            Vec2_t &cur_px_estimate);

    bool get_warp_affine_matrix(data::keyframe *ref,
            data::frame *curr,
            const Vec2_t &px_ref,
            data::landmark *mp,
            int level,
            const Mat44_t &TCR,
            Mat22_t &ACR);

    void warp_affine(const Mat22_t &ACR,
            const cv::Mat &img_ref,
            const Vec2_t &px_ref,
            const int &level_ref,
            const data::keyframe *ref,
            const int &search_level,
            const int &half_patch_size,
            T *patch);

    int get_best_search_level(const Mat22_t &ACR,
                              const int &max_level,
                              const data::keyframe *ref);


    T patch_[FEAT_ALIGN_WARP_PATCH_SIZE * FEAT_ALIGN_WARP_PATCH_SIZE];
    T patch_with_border_[(FEAT_ALIGN_WARP_PATCH_SIZE + 2) * (FEAT_ALIGN_WARP_PATCH_SIZE + 2)];
};



template <typename T>
T get_bilateral_interp(
    const double &x, const double &y, const cv::Mat &gray) {
    const double xx = x - floor(x);
    const double yy = y - floor(y);
    const int px = int(x);
    const int py = int(y);
    if (gray.type() == CV_8UC1) {
        return T(
                (1 - xx) * (1 - yy) * gray.ptr<uint8_t>(py)[px] +
                xx * (1 - yy) * gray.ptr<uint8_t>(py)[px+1] +
                (1 - xx) * yy * gray.ptr<uint8_t>(py+1)[px] +
                xx * yy * gray.ptr<uint8_t>(py+1)[px+1]);
    } else if (gray.type() == CV_32FC1) {
        return T(
            (1 - xx) * (1 - yy) * gray.ptr<float>(py)[px] +
            xx * (1 - yy) * gray.ptr<float>(py)[px+1] +
            (1 - xx) * yy * gray.ptr<float>(py+1)[px] +
            xx * yy * gray.ptr<float>(py+1)[px+1]);
    }
}

template <class T>
bool sparse_feature_aligner<T>::find_projection_direct(data::keyframe *ref, data::frame *curr,
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
    T *ref_patch_ptr = patch_;
    for (int y = 1; y < FEAT_ALIGN_WARP_PATCH_SIZE + 1; ++y, ref_patch_ptr += FEAT_ALIGN_WARP_PATCH_SIZE) {
        T *ref_patch_border_ptr = patch_with_border_ + y * (FEAT_ALIGN_WARP_PATCH_SIZE + 2) + 1;
        for (int x = 0; x < FEAT_ALIGN_WARP_PATCH_SIZE; ++x)
            ref_patch_ptr[x] = ref_patch_border_ptr[x];
    }

    Vec2_t px_scaled = px_curr * curr->inv_scale_factors_[search_level];
    const bool success = align_pt_2d(curr->image_pyramid_[search_level], patch_with_border_, patch_, 10, px_scaled);
    px_curr = px_scaled * curr->scale_factors_[search_level];
    return success;
}

template <class T>
bool sparse_feature_aligner<T>::get_warp_affine_matrix(data::keyframe *ref,
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

template <class T>
void sparse_feature_aligner<T>::warp_affine(
    const Mat22_t &ACR,
    const cv::Mat &img_ref,
    const Vec2_t &px_ref,
    const int &level_ref,
    const data::keyframe *ref,
    const int &search_level,
    const int &half_patch_size,
    T *patch) {
    const int patch_size = half_patch_size * 2;
    const Mat22_t ARC = ACR.inverse();

    // Affine warp
    T *patch_ptr = patch;
    const Vec2_t px_ref_pyr = px_ref / ref->scale_factors_[level_ref];
    for (int y = 0; y < patch_size; y++) {
        for (int x = 0; x < patch_size; x++, ++patch_ptr) {
            Vec2_t px_patch(x - half_patch_size, y - half_patch_size);
            px_patch *= ref->scale_factors_[search_level];
            const Vec2_t px(ARC * px_patch + px_ref_pyr);
            if (px[0] < 0 || px[1] < 0 || px[0] >= img_ref.cols - 1 || px[1] >= img_ref.rows - 1) {
                *patch_ptr = 0;
            } else {
                Eigen::Vector4f interp_weights;
                get_interp_weights(px[0], px[1], interp_weights);
                *patch_ptr = get_interpolation_weighted_grayvalue<T>(
                            img_ref, std::floor(px[0]), std::floor(px[1]), interp_weights);
            }
        }
    }
}


template <class T>
int sparse_feature_aligner<T>::get_best_search_level(const Mat22_t &ACR,
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

template <class T>
bool sparse_feature_aligner<T>::align_pt_2d(const cv::Mat &cur_img,
            T *ref_patch_with_border,
            T *ref_patch,
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
        T *it = ref_patch_with_border + (y + 1) * ref_step + 1;
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
        Eigen::Vector4f interp_vals;
        get_interp_weights(u, v, interp_vals);

        // loop through search_patch, interpolate
        T *it_ref = ref_patch;
        float *it_ref_dx = ref_patch_dx;
        float *it_ref_dy = ref_patch_dy;
        Vec3_t Jres;
        Jres.setZero();
        const int x_start = u_r - FEAT_ALIGN_WARP_HALF_PATCH_SIZE;
        const int y_start = v_r - FEAT_ALIGN_WARP_HALF_PATCH_SIZE;

        for (int y = y_start; y < FEAT_ALIGN_WARP_PATCH_SIZE + y_start; ++y) {
            for (int x = x_start; x < FEAT_ALIGN_WARP_PATCH_SIZE + x_start; ++x, ++it_ref, ++it_ref_dx, ++it_ref_dy) {
                //float search_pixel = wTL * it[0] + wTR * it[1] + wBL * it[cur_step] + wBR * it[cur_step + 1];
                float search_pixel = get_interpolation_weighted_grayvalue<T>(cur_img, x, y, interp_vals);
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

#endif // OPENVSLAM_SPARSE_FEATURE_ALIGNER_H
