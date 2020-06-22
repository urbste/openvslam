#ifndef OPENVSLAM_SPARSE_FEATURE_ALIGNER_H
#define OPENVSLAM_SPARSE_FEATURE_ALIGNER_H

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
const int FEAT_ALIGN_WARP_PATCH_SIZE = 8;
const int FEAT_ALIGN_WARP_HALF_PATCH_SIZE = FEAT_ALIGN_WARP_PATCH_SIZE / 2;
const int FEAT_ALIGN_WARP_PATCH_AREA = FEAT_ALIGN_WARP_PATCH_SIZE * FEAT_ALIGN_WARP_PATCH_SIZE;

bool project_to_image_distorted(
        const data::frame* curr_frame,
        const Mat44_t& world_to_cam,
        const Vec3_t& pt3,
        Vec2_t& projection);


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
            uint8_t *ref_patch_with_border,
            uint8_t *ref_patch,
            const int n_iter,
            Vec2_t &cur_px_estimate);

    // 计算affine wrap矩阵
    bool get_warp_affine_matrix(data::keyframe *ref,
            data::frame *curr,
            const Vec2_t &px_ref,
            data::landmark *mp,
            int level,
            const Mat44_t &TCR,
            Mat22_t &ACR);

    // perform affine warp
    void warp_affine(const Mat22_t &ACR,
            const cv::Mat &img_ref,
            const Vec2_t &px_ref,
            const int &level_ref,
            const data::keyframe *ref,
            const int &search_level,
            const int &half_patch_size,
            uint8_t *patch);

    int get_best_search_level(const Mat22_t &ACR,
                              const int &max_level,
                              const data::keyframe *ref);

    uchar get_bilateral_interp_uchar(const double &x, const double &y, const cv::Mat &gray);


    uchar patch_[FEAT_ALIGN_WARP_PATCH_SIZE * FEAT_ALIGN_WARP_PATCH_SIZE];
    // 带边界的，左右各1个像素
    uchar patch_with_border_[(FEAT_ALIGN_WARP_PATCH_SIZE + 2) * (FEAT_ALIGN_WARP_PATCH_SIZE + 2)];
};

} // namespace match
} // namespace openvslam

#endif // OPENVSLAM_SPARSE_FEATURE_ALIGNER_H
